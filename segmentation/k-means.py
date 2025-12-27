import cv2
import numpy as np

# ========= 設定 =========
IMG_PATH = "Input/straight1.png"
K = 5                    # クラスタ数
OUT_PATH = "Output/kmeans_lab.png"

# ========= 画像読み込み =========
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ========= Lab変換 =========
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# ========= 特徴ベクトル作成 =========
Z = lab.reshape((-1, 3)).astype(np.float32)

# ========= k-means =========
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
ret, labels, centers = cv2.kmeans(
    Z, K, None, criteria,
    attempts=5,
    flags=cv2.KMEANS_RANDOM_CENTERS
)

labels_img = labels.reshape((h, w))

# ========= 可視化 =========
# クラスタ中心色で再構成
centers = centers.astype(np.uint8)
seg = centers[labels.flatten()]
seg = seg.reshape((h, w, 3))
seg_bgr = cv2.cvtColor(seg, cv2.COLOR_LAB2BGR)

cv2.imwrite(OUT_PATH, seg_bgr)
print("保存しました:", OUT_PATH)
