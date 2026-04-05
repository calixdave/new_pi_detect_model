import os
import glob
import joblib
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# Train Pi-specific tile color model
# =========================================================

DATA_DIR = os.path.expanduser("~/Downloads/pi_color_retrain/labeled")
MODEL_OUT = os.path.expanduser("~/Downloads/pi_color_retrain/models/tile_color_model_pi.joblib")

CLASSES = ["blue", "green", "pink", "purple", "red", "yellow"]
VALID_EXTS = ("*.jpg", "*.jpeg", "*.png")

def extract_features(img):
    h, w = img.shape[:2]

    # center crop
    y0, y1 = int(0.25 * h), int(0.75 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    roi = img[y0:y1, x0:x1]

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    feats = []

    for arr in (lab, hsv):
        flat = arr.reshape(-1, 3).astype(np.float32)
        means = flat.mean(axis=0)
        stds  = flat.std(axis=0)
        feats.extend(means.tolist())
        feats.extend(stds.tolist())

    return np.array(feats, dtype=np.float32)

def load_dataset():
    X = []
    y = []

    print("Loading dataset from:", DATA_DIR)
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = []
        for ext in VALID_EXTS:
            files.extend(glob.glob(os.path.join(cls_dir, ext)))

        print(f"{cls}: {len(files)} images")

        for path in files:
            img = cv2.imread(path)
            if img is None:
                print("Warning: could not read", path)
                continue

            feat = extract_features(img)
            if feat is None:
                print("Warning: empty feature from", path)
                continue

            X.append(feat)
            y.append(cls)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y

def main():
    X, y = load_dataset()

    print("\nTotal samples:", len(y))
    if len(y) < 20:
        print("Not enough data to train.")
        return

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining SVC...")
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    payload = {
        "model": model,
        "classes": CLASSES
    }
    joblib.dump(payload, MODEL_OUT)

    print("\nSaved model to:")
    print(MODEL_OUT)

if __name__ == "__main__":
    main()
