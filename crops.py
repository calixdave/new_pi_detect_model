import os
import cv2

# ============================================
# Simple tile cropper for Pi retraining
# ============================================

RAW_DIR = os.path.expanduser("~/Downloads/pi_color_retrain/raw/session_01")
OUT_DIR = os.path.expanduser("~/Downloads/pi_color_retrain/crops")

os.makedirs(OUT_DIR, exist_ok=True)

# Crop settings (tuned for your grid view)
CROP_SIZE = 160   # size of tile crop (adjust if needed)

# Optional: number of crops per image (center + offsets)
OFFSETS = [
    (0, 0),       # center
    (-80, 0),     # left
    (80, 0),      # right
    (0, -80),     # up
    (0, 80)       # down
]

img_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg")]

count = 0

for fname in img_files:
    path = os.path.join(RAW_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    for dx, dy in OFFSETS:
        x = cx + dx
        y = cy + dy

        x1 = int(x - CROP_SIZE // 2)
        y1 = int(y - CROP_SIZE // 2)
        x2 = int(x + CROP_SIZE // 2)
        y2 = int(y + CROP_SIZE // 2)

        # Bounds check
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        crop = img[y1:y2, x1:x2]

        out_name = f"crop_{count:05d}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)

        cv2.imwrite(out_path, crop)
        count += 1

print(f"Done. Total crops saved: {count}")
