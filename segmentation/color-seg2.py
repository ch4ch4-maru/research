import cv2
import numpy as np

# ================= 設定 =================
IMG_PATH = "Input/right1.png"
OUT_PATH = "Output/edge_color_segment.png"

K = 5                 # 色クラスタ数（4〜6推奨）
ALPHA = 0.01         # 空間重み（重要）
LOW_THRESH = 80
HIGH_THRESH = 200

KERNEL_SIZE = 5

# ================= 画像読み込み =================
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ================= 色クラスタリング（Lab + xy） =================
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

features = []
for y in range(h):
    for x in range(w):
        L, a, b = lab[y, x]
        features.append([L, a, b, ALPHA * x, ALPHA * y])

Z = np.array(features, dtype=np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
_, labels, _ = cv2.kmeans(
    Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
)

labels_img = labels.reshape((h, w))

# ================= エッジ検出 =================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, LOW_THRESH, HIGH_THRESH)

# エッジを少し太く（境界を安定させる）
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)

# ================= クラスタごとに連結成分 =================
label_map = np.zeros((h, w), dtype=np.int32)
current_label = 1

for k in range(K):
    # クラスタマスク
    mask = (labels_img == k).astype(np.uint8) * 255

    # エッジで分断
    mask[edges_dilated > 0] = 0

    # モルフォロジーで穴埋め
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 連結成分抽出（＝物体）
    num_cc, cc = cv2.connectedComponents(mask)

    for i in range(1, num_cc):
        label_map[cc == i] = current_label
        current_label += 1

num_objects = current_label - 1
print("検出された物体数:", num_objects)

# ================= ランダム色で可視化 =================
rng = np.random.default_rng(0)
seg_vis = np.zeros((h, w, 3), dtype=np.uint8)

for label in range(1, current_label):
    color = rng.integers(50, 255, size=3, dtype=np.uint8)
    seg_vis[label_map == label] = color

cv2.imwrite(OUT_PATH, seg_vis)
print("保存しました:", OUT_PATH)
