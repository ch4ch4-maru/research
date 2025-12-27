import cv2
import numpy as np

# ========= 設定 =========
IMG_PATH = "Input/right1.png"
OUT_PATH = "Output/edge_color_seg.png"

K = 5                   # 色クラスタ数
ALPHA = 0.01            # 空間重み
LOW_THRESH = 50
HIGH_THRESH = 150
KERNEL_SIZE = 5

# ========= 画像読み込み =========
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ========= 色クラスタリング（Lab + xy） =========
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

features = []
for y in range(h):
    for x in range(w):
        L, a, b = lab[y, x]
        features.append([L, a, b, ALPHA * x, ALPHA * y])

Z = np.array(features, dtype=np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, _ = cv2.kmeans(
    Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
)

labels_img = labels.reshape((h, w))

# ========= エッジ検出 =========
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, LOW_THRESH, HIGH_THRESH)

# ========= エッジでクラスタを分断 =========
binary = np.zeros((h, w), dtype=np.uint8)

for k in range(K):
    mask = (labels_img == k).astype(np.uint8) * 255
    mask[edges > 0] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    binary = cv2.bitwise_or(binary, mask)

# ========= 連結成分ラベリング（＝物体） =========
num_labels, labels_cc = cv2.connectedComponents(binary)

# ========= 物体ごとにランダム色 =========
rng = np.random.default_rng(0)
seg_vis = np.zeros((h, w, 3), dtype=np.uint8)

for label in range(1, num_labels):  # 0は背景
    color = rng.integers(0, 255, size=3, dtype=np.uint8)
    seg_vis[labels_cc == label] = color

cv2.imwrite(OUT_PATH, seg_vis)
print("保存しました:", OUT_PATH)
