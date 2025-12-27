import cv2
import numpy as np

# -----------------------------
# パラメータ
# -----------------------------
COS_POS_TH = 0.7     # cosθ がこれ以上なら放射方向とみなす
COS_NEG_TH = -0.5    # cosθ がこれ以下なら逆方向とみなす

# -----------------------------
# Dense Optical Flow
# -----------------------------
def compute_dense_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow


# -----------------------------
# 自己運動判定
# -----------------------------
def estimate_motion(flow):
    h, w = flow.shape[:2]
    cx, cy = w // 2, h // 2

    cos_map = np.zeros((3, 3))

    for iy in range(3):
        for ix in range(3):
            x0 = ix * w // 3
            x1 = (ix + 1) * w // 3
            y0 = iy * h // 3
            y1 = (iy + 1) * h // 3

            region_flow = flow[y0:y1, x0:x1]
            mean_flow = region_flow.mean(axis=(0, 1))

            # 区画中心ベクトル
            rx = (x0 + x1) / 2 - cx
            ry = (y0 + y1) / 2 - cy
            r = np.array([rx, ry])

            v = mean_flow
            nr = np.linalg.norm(r)
            nv = np.linalg.norm(v)

            if nr < 1e-6 or nv < 1e-6:
                cos_theta = 0.0
            else:
                cos_theta = np.dot(r, v) / (nr * nv)

            cos_map[iy, ix] = cos_theta

    # 中央を除く
    surround = cos_map.copy()
    surround[1, 1] = np.nan

    # -----------------------------
    # 判定
    # -----------------------------
    if np.nanmin(surround) > COS_POS_TH:
        return "straight"

    # 左右旋回判定
    left_score = np.nanmean(cos_map[:, 0])
    right_score = np.nanmean(cos_map[:, 2])

    if left_score < COS_NEG_TH:
        return "left"
    if right_score < COS_NEG_TH:
        return "right"

    # 上下振動判定
    top_score = np.nanmean(cos_map[0, :])
    bottom_score = np.nanmean(cos_map[2, :])

    if top_score < COS_NEG_TH or bottom_score < COS_NEG_TH:
        return "vibration"

    return "straight"


# -----------------------------
# main
# -----------------------------
def main():
    img1 = cv2.imread("Input/right2.png")
    img2 = cv2.imread("Input/right1.png")

    if img1 is None or img2 is None:
        raise FileNotFoundError("画像が読み込めません")

    flow = compute_dense_flow(img1, img2)
    motion = estimate_motion(flow)

    print("Estimated motion:", motion)


if __name__ == "__main__":
    main()
