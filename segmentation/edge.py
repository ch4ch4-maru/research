import cv2
import numpy as np

# ========= 設定 =========
IMG_PATH = "Input/straight1.png"
OUT_PATH = "Output/edges_canny.png"

LOW_THRESH = 50     # 低閾値
HIGH_THRESH = 150   # 高閾値
GAUSS_SIZE = 5      # ガウシアンカーネルサイズ（奇数）

# ========= 画像読み込み =========
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

# ========= 前処理 =========
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (GAUSS_SIZE, GAUSS_SIZE), 0)

# ========= Canny =========
edges = cv2.Canny(blur, LOW_THRESH, HIGH_THRESH)

# ========= 保存 =========
cv2.imwrite(OUT_PATH, edges)
print("保存しました:", OUT_PATH)
