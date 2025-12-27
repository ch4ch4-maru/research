import cv2
import numpy as np

# ================= 設定 =================
IMG_PATH = "Input/right1.png"
OUT_PATH = "Output/edge_color_segment.png"

K = 5                  # 初期色クラスタ数（4〜6推奨）
ALPHA = 0.007          # 空間重み
LOW_THRESH = 80
HIGH_THRESH = 200
KERNEL_SIZE = 5

COLOR_MERGE_THRESH = 20.0   # Lab色距離の統合閾値

# ================= 画像読み込み =================
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ================= Lab 色空間 =================
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# ================= k-means（色 + 空間） =================
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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges = cv2.dilate(edges, kernel, iterations=1)

# ================= クラスタごとに連結成分 =================
label_map = np.zeros((h, w), dtype=np.int32)
current_label = 1

for k in range(K):
    mask = (labels_img == k).astype(np.uint8) * 255
    mask[edges > 0] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_cc, cc = cv2.connectedComponents(mask)

    for i in range(1, num_cc):
        label_map[cc == i] = current_label
        current_label += 1

# ================= 領域ごとの平均色（Lab） =================
region_colors = {}

for label in range(1, current_label):
    region_mask = (label_map == label)
    if np.count_nonzero(region_mask) == 0:
        continue
    region_colors[label] = np.mean(lab[region_mask], axis=0)

# ================= 色が近い領域を統合（Union-Find） =================
parent = {l: l for l in region_colors.keys()}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[ry] = rx

labels_list = list(region_colors.keys())

for i in range(len(labels_list)):
    for j in range(i + 1, len(labels_list)):
        li, lj = labels_list[i], labels_list[j]
        d = np.linalg.norm(region_colors[li] - region_colors[lj])
        if d < COLOR_MERGE_THRESH:
            union(li, lj)

# ================= 統合ラベル反映 =================
merged_label_map = np.zeros_like(label_map)
new_id = {}
next_id = 1

for y in range(h):
    for x in range(w):
        lbl = label_map[y, x]
        if lbl == 0:
            continue
        root = find(lbl)
        if root not in new_id:
            new_id[root] = next_id
            next_id += 1
        merged_label_map[y, x] = new_id[root]

print("最終領域数:", next_id - 1)

# ================= ランダム色で可視化 =================
rng = np.random.default_rng(0)
vis = np.zeros((h, w, 3), dtype=np.uint8)

for label in range(1, next_id):
    color = rng.integers(50, 255, size=3, dtype=np.uint8)
    vis[merged_label_map == label] = color

cv2.imwrite(OUT_PATH, vis)
print("保存しました:", OUT_PATH)
