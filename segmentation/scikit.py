from skimage.segmentation import felzenszwalb
import cv2
import numpy as np

img = cv2.imread("Input/straight1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

segments = felzenszwalb(
    img,
    scale=200,
    sigma=1.0,
    min_size=200
)

# 可視化
vis = np.zeros_like(img)
num_segments = segments.max() + 1

rng = np.random.default_rng(0)
for seg_id in range(num_segments):
    color = rng.integers(0, 255, size=3)
    vis[segments == seg_id] = color

cv2.imwrite("felzenszwalb_segments.png",
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
