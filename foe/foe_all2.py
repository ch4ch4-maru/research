import cv2
import numpy as np

# ===============================
# 入力画像（グレースケール）
# ===============================
img1 = cv2.imread("Input/right2.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Input/right1.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise RuntimeError("画像が読み込めません")

h, w = img1.shape

# ===============================
# オプティカルフロー
# ===============================
flow = cv2.calcOpticalFlowFarneback(
    img1, img2, None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

u = flow[..., 0]
v = flow[..., 1]

# フロー強度
mag = np.sqrt(u*u + v*v)

# ===============================
# 有効画素選択（重要）
# ===============================
mask = mag > 1.0   # ★小さすぎるフローは除外
ys, xs = np.where(mask)

u = u[mask]
v = v[mask]

if len(u) < 100:
    raise RuntimeError("有効なフローが少なすぎます")

# ===============================
# FOE 推定（最小二乗）
# ===============================
# v*x0 - u*y0 = v*x - u*y

A = np.column_stack((v, -u))
b = v * xs - u * ys

# 最小二乗解
foe, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
x0, y0 = foe

print(f"Estimated FOE: ({x0:.2f}, {y0:.2f})")

# ===============================
# 可視化
# ===============================
vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.circle(vis, (int(x0), int(y0)), 8, (0, 0, 255), -1)

cv2.imshow("FOE", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
