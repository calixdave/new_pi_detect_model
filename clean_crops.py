import os
import cv2

FOLDER = "crops/near"   # <-- change if needed

images = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(('.jpg','.png','.jpeg'))])

i = 0

while i < len(images):
    path = os.path.join(FOLDER, images[i])
    img = cv2.imread(path)

    if img is None:
        print("Error loading:", path)
        i += 1
        continue

    cv2.imshow("Crop Cleaner", img)
    key = cv2.waitKey(0)

    # d = delete
    if key == ord('d'):
        os.rename(path, "bad/" + images[i])
        print("Deleted:", images[i])
    
    # k = keep (next)
    elif key == ord('k'):
        print("Kept:", images[i])
    
    # b = go back
    elif key == ord('b') and i > 0:
        i -= 2  # go back one
    
    # q = quit
    elif key == ord('q'):
        break

    i += 1

cv2.destroyAllWindows()
