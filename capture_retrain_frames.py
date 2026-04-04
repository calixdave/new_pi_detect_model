import os
import cv2
import time

# =========================================================
# Simple Pi color retraining frame collector
# Saves full frames into:
#   ~/Downloads/pi_color_retrain/raw/<session_name>/
# Press:
#   c = capture one frame
#   a = auto-capture burst
#   q = quit
# =========================================================

# ---------- Settings ----------
SESSION_NAME = "session_01"
BASE_DIR = os.path.expanduser("~/Downloads/pi_color_retrain/raw")
SAVE_DIR = os.path.join(BASE_DIR, SESSION_NAME)

CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480

# Auto-capture settings
AUTO_COUNT = 20          # how many frames to save in burst
AUTO_DELAY_SEC = 0.6     # delay between saves

# Optional preview resize (only for display)
PREVIEW_SCALE = 1.0

# ---------- Setup ----------
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit(1)

print("==========================================")
print("Pi retraining frame collector")
print(f"Saving to: {SAVE_DIR}")
print("Keys:")
print("  c = capture one frame")
print("  a = auto-capture burst")
print("  q = quit")
print("==========================================")

img_count = 0

# Start count from existing files if folder already has images
existing = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if existing:
    img_count = len(existing)

def save_frame(frame, prefix="img"):
    global img_count
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{img_count:04d}_{ts}.jpg"
    path = os.path.join(SAVE_DIR, name)
    ok = cv2.imwrite(path, frame)
    if ok:
        print(f"Saved: {path}")
        img_count += 1
    else:
        print("ERROR: Failed to save image.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("WARNING: Failed to grab frame.")
        continue

    preview = frame.copy()

    # On-screen text
    cv2.putText(preview, f"Session: {SESSION_NAME}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(preview, f"Saved: {img_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(preview, "c=save  a=auto  q=quit", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if PREVIEW_SCALE != 1.0:
        preview = cv2.resize(
            preview,
            (int(preview.shape[1] * PREVIEW_SCALE), int(preview.shape[0] * PREVIEW_SCALE)),
            interpolation=cv2.INTER_AREA
        )

    cv2.imshow("Pi Retrain Capture", preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        save_frame(frame, prefix="manual")

    elif key == ord("a"):
        print(f"Starting auto-capture: {AUTO_COUNT} frames")
        for i in range(AUTO_COUNT):
            ret2, frame2 = cap.read()
            if not ret2 or frame2 is None:
                print("WARNING: Failed to grab frame during auto-capture.")
                continue
            save_frame(frame2, prefix="auto")
            time.sleep(AUTO_DELAY_SEC)
        print("Auto-capture finished.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
