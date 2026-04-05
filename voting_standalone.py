import os
import glob
from collections import Counter

import cv2
import numpy as np
import joblib


# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
MODEL_PATH = "tile_color_model.joblib"

HEADINGS = ["front", "right", "back", "left"]

# minimum confidence to accept model label
COLOR_CONF_THRESH = 0.40

# how much of the lower image to use
ROI_TOP_FRAC = 0.62
ROI_BOT_FRAC = 0.93

# slot padding for color
SLOT_PAD_X_FRAC = 0.02
SLOT_PAD_Y_FRAC = 0.06

# slot padding for object
OBJ_PAD_X_FRAC = 0.01
OBJ_PAD_Y_FRAC = 0.03

# output folders
SAVE_DIR = "voted_results"
SAVE_DEBUG = True
DEBUG_DIR = "debug_vote"


# =========================================================
# BIG PICTURE MAPPING OF LOCAL 3x3 POSITIONS
# =========================================================
# final local 3x3 uses:
# [(-1,+1), (0,+1), (+1,+1)]
# [(-1, 0), (0, 0), (+1, 0)]
# [(-1,-1), (0,-1), (+1,-1)]
#
# center is agent A
# =========================================================

HEADING_TO_NEIGHBORS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


# =========================================================
# MODEL LOADING
# =========================================================

def load_color_model(model_path):
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        model = obj.get("model", None)
        classes = obj.get("classes", None)

        if model is None:
            raise ValueError("Joblib dict does not contain 'model'")

        if classes is not None:
            classes = list(classes)
        elif hasattr(model, "classes_"):
            classes = list(model.classes_)
        else:
            raise ValueError("Could not determine class names from joblib")
    else:
        model = obj
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        else:
            raise ValueError("Model has no classes_ and joblib is not a dict")

    return model, classes


# =========================================================
# COLOR FEATURES
# =========================================================

def extract_features(img_bgr):
    h, w = img_bgr.shape[:2]
    y0, y1 = int(0.25 * h), int(0.75 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    roi = img_bgr[y0:y1, x0:x1]

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    feats = []
    for arr in (lab, hsv):
        flat = arr.reshape(-1, 3).astype(np.float32)
        feats.extend(flat.mean(axis=0).tolist())
        feats.extend(flat.std(axis=0).tolist())

    return np.array(feats, dtype=np.float32)


def label_to_char(label):
    mapping = {
        "blue": "B",
        "green": "G",
        "pink": "M",
        "purple": "P",
        "red": "R",
        "yellow": "Y",
    }
    return mapping.get(label.lower(), "?")


# =========================================================
# IMAGE / SLOT HELPERS
# =========================================================

def get_bottom_roi(img):
    h, w = img.shape[:2]
    y0 = int(h * ROI_TOP_FRAC)
    y1 = int(h * ROI_BOT_FRAC)
    return img[y0:y1, :], y0, y1


def split_into_3_slots(roi, pad_x_frac, pad_y_frac):
    h, w = roi.shape[:2]
    slot_w = w // 3

    slots = []
    boxes = []

    pad_x = int(slot_w * pad_x_frac)
    pad_y = int(h * pad_y_frac)

    for i in range(3):
        x0 = i * slot_w + pad_x
        x1 = (i + 1) * slot_w - pad_x
        y0 = pad_y
        y1 = h - pad_y

        x0 = max(0, x0)
        x1 = min(w, x1)
        y0 = max(0, y0)
        y1 = min(h, y1)

        crop = roi[y0:y1, x0:x1]
        slots.append(crop)
        boxes.append((x0, y0, x1, y1))

    return slots, boxes


# =========================================================
# COLOR DETECTION PER SLOT
# =========================================================

def detect_color_slots(img, model, classes, debug_prefix=None):
    roi, _, _ = get_bottom_roi(img)
    slots, _ = split_into_3_slots(roi, SLOT_PAD_X_FRAC, SLOT_PAD_Y_FRAC)

    out = []

    for idx, slot in enumerate(slots):
        if slot.size == 0:
            out.append(("?", 0.0))
            continue

        feats = extract_features(slot)
        if feats is None:
            out.append(("?", 0.0))
            continue

        feats = feats.reshape(1, -1)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feats)[0]
            best_i = int(np.argmax(probs))
            label = classes[best_i]
            conf = float(probs[best_i])
        else:
            label = model.predict(feats)[0]
            conf = 1.0

        if conf < COLOR_CONF_THRESH:
            char = "?"
        else:
            char = label_to_char(label)

        out.append((char, conf))

        if SAVE_DEBUG and debug_prefix is not None:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"{debug_prefix}_color_slot{idx}.jpg"),
                slot
            )

    return out


# =========================================================
# OBJECT DETECTION PER SLOT
# =========================================================
# O = obstacle  -> white box with thick red X
# T = target    -> white box with thick black X
# E = empty
# ? = unknown
# =========================================================

def detect_object_slots(img, debug_prefix=None):
    roi, _, _ = get_bottom_roi(img)
    slots, _ = split_into_3_slots(roi, OBJ_PAD_X_FRAC, OBJ_PAD_Y_FRAC)

    out = []

    for idx, slot in enumerate(slots):
        if slot.size == 0:
            out.append("?")
            continue

        state = detect_one_object_slot(slot)
        out.append(state)

        if SAVE_DEBUG and debug_prefix is not None:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"{debug_prefix}_obj_slot{idx}.jpg"),
                slot
            )

    return out


def detect_one_object_slot(slot):
    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)

    # ---------- red X detection ----------
    lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)

    red_ratio = np.count_nonzero(red_mask) / red_mask.size

    # ---------- white area detection ----------
    white_mask = cv2.inRange(gray, 180, 255)
    white_ratio = np.count_nonzero(white_mask) / white_mask.size

    # ---------- black X detection ----------
    black_mask = cv2.inRange(gray, 0, 60)
    black_ratio = np.count_nonzero(black_mask) / black_mask.size

    # ---------- line detection for X ----------
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=20,
        maxLineGap=8
    )

    diag_pos = 0
    diag_neg = 0

    if lines is not None:
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = ln
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            ang = np.degrees(np.arctan2(dy, dx))

            if 25 <= ang <= 65:
                diag_pos += 1
            elif -65 <= ang <= -25:
                diag_neg += 1

    has_x_shape = (diag_pos >= 1 and diag_neg >= 1)

    # obstacle = white object area + red X
    if white_ratio > 0.18 and red_ratio > 0.03 and has_x_shape:
        return "O"

    # target = white box with black X
    if white_ratio > 0.18 and black_ratio > 0.05 and has_x_shape:
        return "T"

    # empty if not enough sign of object
    if white_ratio < 0.12 and red_ratio < 0.01 and black_ratio < 0.03:
        return "E"

    return "?"


# =========================================================
# VOTING
# =========================================================

def vote_chars(chars, unknown_char="?"):
    filtered = [c for c in chars if c != unknown_char]

    if not filtered:
        return unknown_char

    count = Counter(filtered)
    best, _ = count.most_common(1)[0]
    return best


def vote_color_results(results):
    # results = [[('B',0.9),('G',0.8),('R',0.7)], [...], ...]
    voted = []
    for slot_idx in range(3):
        slot_chars = [one_result[slot_idx][0] for one_result in results]
        voted.append(vote_chars(slot_chars, "?"))
    return voted


def vote_object_results(results):
    # results = [['E','O','E'], ['E','O','?'], ...]
    voted = []
    for slot_idx in range(3):
        slot_states = [one_result[slot_idx] for one_result in results]
        voted.append(vote_chars(slot_states, "?"))
    return voted


# =========================================================
# LOAD HEADING IMAGES
# =========================================================

def find_heading_images(scan_dir, heading):
    patterns = [
        os.path.join(scan_dir, f"{heading}_*.jpg"),
        os.path.join(scan_dir, f"{heading}_*.png"),
        os.path.join(scan_dir, f"{heading}.jpg"),
        os.path.join(scan_dir, f"{heading}.png"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    files = sorted(set(files))
    return files


# =========================================================
# BUILD FINAL LOCAL 3x3
# =========================================================

def empty_3x3(fill="?"):
    return [
        [fill, fill, fill],
        [fill, "A",  fill],
        [fill, fill, fill],
    ]


def coord_to_index(dx, dy):
    # x: -1, 0, +1 left->right
    # y: +1, 0, -1 top->bottom
    row = 1 - dy
    col = dx + 1
    return row, col


def place_heading_into_matrix(mat, heading, voted_slots):
    coords = HEADING_TO_NEIGHBORS[heading]
    for slot_i, (dx, dy) in enumerate(coords):
        r, c = coord_to_index(dx, dy)
        mat[r][c] = voted_slots[slot_i]


def pretty_matrix(mat):
    return "\n".join([" ".join(row) for row in mat])


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if SAVE_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    model, classes = load_color_model(MODEL_PATH)

    final_color = empty_3x3("?")
    final_object = empty_3x3("?")

    print("\n===== PER-HEADING VOTING =====\n")

    for heading in HEADINGS:
        files = find_heading_images(SCAN_DIR, heading)

        if not files:
            print(f"{heading.upper()}: no images found")
            continue

        color_results = []
        object_results = []

        print(f"{heading.upper()} files:")
        for f in files:
            print("  ", f)

        for i, path in enumerate(files):
            img = cv2.imread(path)

            if img is None:
                print(f"Could not read: {path}")
                continue

            debug_prefix = f"{heading}_{i}"

            colors = detect_color_slots(img, model, classes, debug_prefix=debug_prefix)
            objects = detect_object_slots(img, debug_prefix=debug_prefix)

            color_results.append(colors)
            object_results.append(objects)

            color_chars = [x[0] for x in colors]

            print(f"\n{heading} image {i+1}:")
            print("  color :", color_chars)
            print("  object:", objects)

        if not color_results:
            print(f"\nNo valid images processed for heading: {heading}")
            continue

        voted_colors = vote_color_results(color_results)
        voted_objects = vote_object_results(object_results)

        print(f"\nVOTED {heading}:")
        print("  color :", voted_colors)
        print("  object:", voted_objects)
        print()

        place_heading_into_matrix(final_color, heading, voted_colors)
        place_heading_into_matrix(final_object, heading, voted_objects)

    print("\n===== FINAL VOTED LOCAL 3x3 =====\n")
    print("COLOR MATRIX:")
    print(pretty_matrix(final_color))

    print("\nOBJECT MATRIX:")
    print(pretty_matrix(final_object))

    color_save_path = os.path.join(SAVE_DIR, "voted_color_3x3.txt")
    object_save_path = os.path.join(SAVE_DIR, "voted_object_3x3.txt")

    with open(color_save_path, "w") as f:
        f.write(pretty_matrix(final_color) + "\n")

    with open(object_save_path, "w") as f:
        f.write(pretty_matrix(final_object) + "\n")

    print("\nSaved:")
    print(" ", color_save_path)
    print(" ", object_save_path)


if __name__ == "__main__":
    main()
