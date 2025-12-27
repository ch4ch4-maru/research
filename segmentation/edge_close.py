import cv2
import numpy as np

# ========= 設定 =========
IMG_PATH = "Input/right1.png"

EDGE_PATH = "Output/close/edges.png"
MASK_PATH = "Output/close/closed_regions.png"
LABEL_PATH = "Output/close/labeled_regions.png"

LOW_THRESH = 50
HIGH_THRESH = 150

KERNEL_SIZE = 1      # モルフォロジー構造要素サイズ
ITER_DILATE = 10000
ITER_CLOSE = 10

# ========= 画像読み込み =========
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ========= Canny =========
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, LOW_THRESH, HIGH_THRESH)
cv2.imwrite(EDGE_PATH, edges)

# ========= モルフォロジー =========
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
)

# エッジを太くして連結
dilated = cv2.dilate(edges, kernel, iterations=ITER_DILATE)

# 隙間を閉じる（閉演算）
closed = cv2.morphologyEx(
    dilated, cv2.MORPH_CLOSE, kernel, iterations=ITER_CLOSE
)

# ========= 塗りつぶし（内部領域） =========
# 背景をFloodFill
floodfill = closed.copy()
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(floodfill, mask, (0, 0), 255)

# 反転して前景抽出
floodfill_inv = cv2.bitwise_not(floodfill)
filled = closed | floodfill_inv

cv2.imwrite(MASK_PATH, filled)

# ========= 領域ラベリング =========
num_labels, labels = cv2.connectedComponents(filled)

# 可視化
label_img = np.zeros((h, w, 3), dtype=np.uint8)
rng = np.random.default_rng(0)

for label in range(1, num_labels):
    color = rng.integers(0, 255, size=3)
    label_img[labels == label] = color

cv2.imwrite(LABEL_PATH, label_img)

print("完了:")
print(" - エッジ:", EDGE_PATH)
print(" - 閉領域マスク:", MASK_PATH)
print(" - ラベル画像:", LABEL_PATH)
