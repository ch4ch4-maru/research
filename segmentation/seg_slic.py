import cv2
import numpy as np

# ========= 設定 =========
IMG_PATH = "Input/right1.png"      # 入力画像
OUT_PATH = "Output/slic_result.png"

N_SUPERPIXELS = 400         # 分割数（粗分割なら300〜600）
RULER = 10.0                # 色と距離のバランス
NUM_ITER = 10               # 反復回数

# ========= 画像読み込み =========
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("画像を読み込めません")

h, w = img.shape[:2]

# ========= SLIC =========
region_size = int(np.sqrt(h * w / N_SUPERPIXELS))

slic = cv2.ximgproc.createSuperpixelSLIC(
    img,
    algorithm=cv2.ximgproc.SLICO,  # SLICO = 自動正規化（おすすめ）
    region_size=region_size,
    ruler=RULER
)

slic.iterate(NUM_ITER)

# ========= 結果取得 =========
labels = slic.getLabels()                    # (H, W)
num_labels = slic.getNumberOfSuperpixels()   # 実際の分割数
mask = slic.getLabelContourMask()            # 境界マスク

print("Superpixels:", num_labels)

# ========= 可視化 =========
vis = img.copy()
vis[mask == 255] = (0, 0, 255)  # 境界を赤で描画

cv2.imwrite(OUT_PATH, vis)
print("保存しました:", OUT_PATH)
