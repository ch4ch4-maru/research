import cv2
import numpy as np

# ===============================
# 入力画像
# ===============================
img1 = cv2.imread("Input/straight1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Input/straight2.png", cv2.IMREAD_GRAYSCALE)

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

mag = np.sqrt(u*u + v*v)

# ===============================
# 有効画素（ノイズ除去）
# ===============================
mask = mag > 1.0
ys, xs = np.where(mask)
u = u[mask]
v = v[mask]

# ===============================
# 外積誤差関数
# ===============================
def cross_error(x0, y0):
    rx = xs - x0
    ry = ys - y0
    return np.mean((rx * v - ry * u) ** 2)

# ===============================
# 粗探索（グリッド）
# ===============================
step = 20
best_err = np.inf
best_foe = (0, 0)

for y0 in range(0, h, step):
    for x0 in range(0, w, step):
        err = cross_error(x0, y0)
        if err < best_err:
            best_err = err
            best_foe = (x0, y0)

# ===============================
# 局所探索（微調整）
# ===============================
x0, y0 = best_foe

for step in [10, 5, 2, 1]:
    candidates = []
    for dy in [-step, 0, step]:
        for dx in [-step, 0, step]:
            xx = x0 + dx
            yy = y0 + dy
            if 0 <= xx < w and 0 <= yy < h:
                candidates.append((cross_error(xx, yy), xx, yy))
    _, x0, y0 = min(candidates)

print(f"Estimated FOE: ({x0}, {y0})")

# ===============================
# 可視化
# ===============================
vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.circle(vis, (int(x0), int(y0)), 8, (0, 0, 255), -1)

cv2.imwrite("foe_cross.png", vis)
print("Saved: foe_cross.png")
