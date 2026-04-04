import os
import cv2
import shutil

# =========================================================
# Pi crop labeling tool
# Shows one crop at a time and moves it to the right folder
#
# Keys:
#   b = blue
#   g = green
#   m = pink
#   p = purple
#   r = red
#   y = yellow
#   s = skip
#   q = quit
# =========================================================

CROPS_DIR = os.path.expanduser("~/Downloads/pi_color_retrain/crops")
LABELED_DIR = os.path.expanduser("~/Downloads/pi_color_retrain/labeled")

CLASS_MAP = {
    ord('b'): 'blue',
    ord('g'): 'green',
    ord('m'): 'pink',
    ord('p'): 'purple',
    ord('r'): 'red',
    ord('y'): 'yellow',
}

VALID_EXTS = (".jpg", ".jpeg", ".png")

for cls in CLASS_MAP.values():
    os.makedirs(os.path.join(LABELED_DIR, cls), exist_ok=True)

files = sorted(
    [f for f in os.listdir(CROPS_DIR) if f.lower().endswith(VALID_EXTS)]
)

if not files:
    print("No crop images found.")
    raise SystemExit(0)

total = len(files)
index = 0
moved = 0
skipped = 0

print("==========================================")
print("Pi crop labeling tool")
print(f"Crops folder:   {CROPS_DIR}")
print(f"Labeled folder: {LABELED_DIR}")
print("Keys:")
print("  b = blue")
print("  g = green")
print("  m = pink")
print("  p = purple")
print("  r = red")
print("  y = yellow")
print("  s = skip")
print("  q = quit")
print("==========================================")

while index < total:
    fname = files[index]
    src_path = os.path.join(CROPS_DIR, fname)

    if not os.path.exists(src_path):
        index += 1
        continue

    img = cv2.imread(src_path)
    if img is None:
        print(f"Could not read: {src_path}")
        index += 1
        continue

    view = img.copy()
    h, w = view.shape[:2]

    info1 = f"[{index+1}/{total}] {fname}"
    info2 = f"moved={moved}  skipped={skipped}"
    info3 = "b blue | g green | m pink | p purple | r red | y yellow | s skip | q quit"

    cv2.putText(view, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(view, info2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(view, info3, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # enlarge small crop for easier viewing
    scale = 3
    view_big = cv2.resize(view, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Label Crops", view_big)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        skipped += 1
        index += 1

    elif key in CLASS_MAP:
        cls = CLASS_MAP[key]
        dst_path = os.path.join(LABELED_DIR, cls, fname)
        shutil.move(src_path, dst_path)
        print(f"Moved {fname} -> {cls}/")
        moved += 1
        index += 1

    else:
        print("Unknown key. Use b/g/m/p/r/y/s/q")

cv2.destroyAllWindows()
print("Done.")
print(f"Moved:   {moved}")
print(f"Skipped: {skipped}")
