import cv2
import numpy as np
import os

# ========= 設定 =========
IMG1_PATH = "Input/right1.png"
IMG2_PATH = "Input/right2.png"
OUTPUT_PATH = "Output/region_foe_result.png"

N_SUPERPIXELS = 400
MIN_POINTS = 25
STEP = 10


# ========= FOE 推定 (RANSAC) =========
def estimate_foe_ransac(points, flows, iterations=300, threshold=4.0):
    if len(points) < MIN_POINTS:
        return None

    lines = []
    for p, v in zip(points, flows):
        if np.linalg.norm(v) < 0.5:
            continue
        lines.append((p.astype(np.float64), v.astype(np.float64)))

    if len(lines) < MIN_POINTS:
        return None

    best_inliers = []
    best_point = None
    L = len(lines)

    for _ in range(iterations):
        i1, i2 = np.random.choice(L, 2, replace=False)
        p1, v1 = lines[i1]
        p2, v2 = lines[i2]

        A = np.array([[v1[0], -v2[0]],
                      [v1[1], -v2[1]]], dtype=np.float64)
        b = p2 - p1

        if np.linalg.matrix_rank(A) < 2:
            continue

        try:
            t = np.linalg.solve(A, b)
        except:
            continue

        foe = p1 + t[0] * v1

        inliers = []
        for p, v in lines:
            diff = foe - p
            cross = v[0] * diff[1] - v[1] * diff[0]
            dist = abs(cross) / (np.linalg.norm(v) + 1e-6)
            if dist < threshold:
                inliers.append((p, v))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_point = foe

    return best_point


# ========= メイン =========
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)

if img1 is None or img2 is None:
    raise RuntimeError("画像を読み込めません")

h, w = img1.shape[:2]

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ===== Optical Flow =====
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    0.5, 3, 15, 3, 5, 1.2, 0
)

# ===== SLIC Superpixel =====
region_size = int(np.sqrt(h * w / N_SUPERPIXELS))
slic = cv2.ximgproc.createSuperpixelSLIC(
    img2, algorithm=cv2.ximgproc.SLICO,
    region_size=region_size, ruler=10.0
)
slic.iterate(10)

labels = slic.getLabels()
num_labels = slic.getNumberOfSuperpixels()

vis = img2.copy()

# ===== SuperpixelごとのFOE =====
for label in range(num_labels):
    mask = (labels == label)
    ys, xs = np.where(mask)

    pts = []
    vecs = []

    for y, x in zip(ys[::STEP], xs[::STEP]):
        fx, fy = flow[y, x]
        if abs(fx) + abs(fy) < 0.5:
            continue
        pts.append([x, y])
        vecs.append([fx, fy])

    if len(pts) < MIN_POINTS:
        continue

    pts = np.array(pts)
    vecs = np.array(vecs)

    foe = estimate_foe_ransac(pts, vecs)
    if foe is None:
        continue

    fx, fy = int(foe[0]), int(foe[1])

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # 描画
    cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)
    cv2.line(vis, (cx, cy), (fx, fy), (0, 0, 255), 1)
    cv2.circle(vis, (fx, fy), 3, (0, 0, 255), -1)

cv2.imwrite(OUTPUT_PATH, vis)
print("完了:", OUTPUT_PATH)
