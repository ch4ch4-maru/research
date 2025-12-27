import cv2
import os
import numpy as np

# ========= 設定 =========
INPUT_DIR = "Input"
OUTPUT_DIR = "Output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ========= 入力動画の選択 =========
def select_video():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    print("Input フォルダ内の動画:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    idx = int(input("使用する動画番号を選んでください: "))
    return files[idx]


# ========= FOE 推定 (RANSAC) =========
def estimate_foe_ransac(points, flows, iterations=400, threshold=4.0):
    """
    points: (N,2) float
    flows:  (N,2) float
    """

    if len(points) < 20:
        return None

    # (p, v) のセットを作る
    lines = []
    for (x, y), (fx, fy) in zip(points, flows):
        vx, vy = float(fx), float(fy)
        if abs(vx) + abs(vy) < 1e-3:
            continue
        p = np.array([float(x), float(y)], dtype=np.float64)
        v = np.array([vx, vy], dtype=np.float64)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
            continue
        lines.append((p, v))

    if len(lines) < 10:
        return None

    best_inliers = []
    best_point = None
    L = len(lines)

    for _ in range(iterations):
        i1, i2 = np.random.choice(L, 2, replace=False)
        p1, v1 = lines[i1]
        p2, v2 = lines[i2]

        # 直線 p1 + t * v1 と p2 + s * v2 の交点を計算
        A = np.array([
            [v1[0], -v2[0]],
            [v1[1], -v2[1]],
        ], dtype=np.float64)
        b = (p2 - p1).astype(np.float64)

        if np.linalg.matrix_rank(A) < 2:
            continue

        try:
            t = np.linalg.solve(A, b)
        except Exception:
            continue

        intersect = p1 + t[0] * v1
        if not np.all(np.isfinite(intersect)):
            continue

        # インライヤ判定
        inliers = []
        for p, v in lines:
            diff = intersect - p
            vx, vy = v
            wx, wy = diff

            # 2D cross (scalar)
            cross_val = vx * wy - vy * wx
            denom = np.hypot(vx, vy)
            if denom < 1e-6:
                continue

            dist = abs(cross_val) / denom

            if dist < threshold:
                inliers.append((p, v))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_point = intersect

    return best_point


# ========= メイン =========
video_name = select_video()
video_path = os.path.join(INPUT_DIR, video_name)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("動画を読み込めませんでした。")

# 出力準備
fps_input = cap.get(cv2.CAP_PROP_FPS)
fps_input = 30 if fps_input <= 0 else fps_input

target_fps = 10
skip = max(1, int(round(fps_input / target_fps)))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0] + "_flow.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

prev_gray = None
frame_count = 0

# FOE 平滑化用
foe_ema = None
ema_alpha = 0.2

print("処理中...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % skip != 0:
        frame_count += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        frame_count += 1
        continue

    # ===== Dense Optical Flow =====
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    vis = frame.copy()

    # ===== 描画 + FOE 用データ収集 =====
    step = 12
    pts = []
    vecs = []

    for y in range(0, height, step):
        for x in range(0, width, step):
            fx, fy = flow[y, x]

            # ノイズ除去
            if abs(fx) + abs(fy) < 0.5:
                continue

            cv2.arrowedLine(vis, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.3)

            pts.append([x, y])
            vecs.append([fx, fy])

    pts = np.array(pts, dtype=np.float64)
    vecs = np.array(vecs, dtype=np.float64)

    # ===== FOE 推定 =====
    foe = estimate_foe_ransac(pts, vecs)

    if foe is not None:
        if foe_ema is None:
            foe_ema = foe
        else:
            foe_ema = ema_alpha * foe + (1 - ema_alpha) * foe_ema

        fx, fy = int(foe_ema[0]), int(foe_ema[1])

        # 画面内
        if 0 <= fx < width and 0 <= fy < height:
            cv2.circle(vis, (fx, fy), 8, (0, 0, 255), -1)
            cv2.putText(vis, "FOE", (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 画面外
        else:
            cx, cy = width // 2, height // 2
            dx, dy = fx - cx, fy - cy
            norm = np.hypot(dx, dy) + 1e-6
            ux, uy = dx / norm, dy / norm
            tip = (int(cx + ux * 150), int(cy + uy * 150))
            cv2.arrowedLine(vis, (cx, cy), tip, (0, 0, 255), 2)
            cv2.putText(vis, "FOE(outside)", (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(vis)
    prev_gray = gray
    frame_count += 1

cap.release()
out.release()
print("完了しました:", output_path)
