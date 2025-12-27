import cv2
import numpy as np

# ========= 設定 =========
IMG_PATH = "Input/straight1.png"
K = 5
OUT_PATH = "Output/kmeans_lab_xy.png"

ALPHA = 0.01    # 空間重み（0.005〜0.02で調整）

# ========= 画像読み込み =========
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ========= Lab変換 =========
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# ========= 特徴ベクトル =========
features = []
for y in range(h):
    for x in range(w):
        L, a, b = lab[y, x]
        features.append([L, a, b, ALPHA * x, ALPHA * y])

Z = np.array(features, dtype=np.float32)

# ========= k-means =========
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(
    Z, K, None, criteria,
    attempts=5,
    flags=cv2.KMEANS_RANDOM_CENTERS
)

labels_img = labels.reshape((h, w))

# ========= 可視化 =========
seg = np.zeros_like(img)
for k in range(K):
    mask = (labels_img == k)
    seg[mask] = np.mean(img[mask], axis=0)

cv2.imwrite(OUT_PATH, seg)
print("保存しました:", OUT_PATH)
