import torch
import cv2
import numpy as np
from model.fast_scnn import FastSCNN   # 実装に依存

# ================= 設定 =================
MODEL_PATH = "fast_scnn_citys.pth"
IMG_PATH = "Input/right1.png"

DEVICE = torch.device("cpu")

# ================= モデル =================
model = FastSCNN(num_classes=19)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ================= 入力 =================
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig_h, orig_w = img.shape[:2]

img_resized = cv2.resize(img, (512, 256))
img_tensor = torch.from_numpy(img_resized).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
img_tensor = img_tensor.to(DEVICE)

# ================= 推論 =================
with torch.no_grad():
    out = model(img_tensor)
    label_map = out.argmax(dim=1)[0].cpu().numpy()

# ================= 元サイズへ =================
label_map = cv2.resize(
    label_map.astype(np.uint8),
    (orig_w, orig_h),
    interpolation=cv2.INTER_NEAREST
)
