import os
import time
import cv2
import joblib
import numpy as np

# =========================================================
# GLOBAL CONFIG
# =========================================================

# ---------- capture ----------
SAVE_DIR = "scan_images"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
HEADINGS = ["front", "right", "back", "left"]

# ---------- color detection ----------
MODEL_PATH = "tile_color_model_pi.joblib"
COLOR_ROI_TOP_FRAC = 0.55
COLOR_ROI_BOT_FRAC = 0.95
COLOR_SLOT_PAD_X_FRAC = 0.03
COLOR_SLOT_PAD_Y_FRAC = 0.06
CONF_THRESH = 0.00

LABEL_TO_CHAR = {
    "blue": "B",
    "green": "G",
    "red": "R",
    "yellow": "Y",
    "pink": "M",
    "purple": "P"
}

# ---------- object detection ----------
OBJ_ROI_TOP_FRAC = 0.34
OBJ_ROI_BOT_FRAC = 0.94
OBJ_SLOT_PAD_X_FRAC = 0.03
OBJ_SLOT_PAD_Y_FRAC = 0.06

WHITE_MIN = 126
BLACK_MAX = 95
RED_S_MIN = 95
RED_V_MIN = 120

WHITE_RATIO_TH = 0.35
RED_RATIO_TH = 0.18
BLACK_RATIO_TH = 0.14

EMPTY_WHITE_TH = 0.80
EMPTY_RED_TH = 0.80
EMPTY_BLACK_TH = 0.90

CANNY1 = 40
CANNY2 = 120
HOUGH_TH = 30
MIN_LINE = 8
MAX_GAP = 40
BLUR_ODD = 1

# ---------- mapping ----------
RESULTS_DIR = "results"
COMPACT_RESULT_FILE = os.path.join(RESULTS_DIR, "compact_map_result.txt")

BIG_GRID = [
    ['G', 'R', 'P', 'Y', 'P', 'P'],
    ['P', 'Y', 'B', 'R', 'M', 'G'],
    ['P', 'P', 'Y', 'R', 'R', 'B'],
    ['M', 'G', 'G', 'M', 'Y', 'Y'],
    ['B', 'M', 'Y', 'M', 'M', 'B'],
    ['R', 'G', 'G', 'B', 'R', 'B'],
]

MIN_KNOWN_NEIGHBORS = 5
MAX_MISMATCHES = 4

SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}

# =========================================================
# CLEANUP
# =========================================================

def cleanup_previous_run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files_to_delete = [
        os.path.join(SAVE_DIR, "front.jpg"),
        os.path.join(SAVE_DIR, "right.jpg"),
        os.path.join(SAVE_DIR, "back.jpg"),
        os.path.join(SAVE_DIR, "left.jpg"),
        COMPACT_RESULT_FILE,
    ]

    print("\nCleaning previous run files... - grid_program_pi_runtime_still_capture_markerfix.py:104")
    for path in files_to_delete:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Deleted: {path} - grid_program_pi_runtime_still_capture_markerfix.py:109")
            except Exception as e:
                print(f"WARNING: Could not delete {path}: {e} - grid_program_pi_runtime_still_capture_markerfix.py:111")

# =========================================================
# SHARED HELPERS
# =========================================================

# =========================================================
# DISPLAY HELPER
# CUT THIS FUNCTION LATER IF YOU NO LONGER WANT ANY DISPLAY
# =========================================================
def put_text(img, text, y, scale=0.7, thickness=2):
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


def matrix_rows_from_grid(final_grid):
    rows = []
    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(final_grid.get((col, row), "?"))
        rows.append(vals)
    return rows


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


def pretty_print_grid(final_grid, title):
    print(f"\n{title} - grid_program_pi_runtime_still_capture_markerfix.py:149")
    rows = matrix_rows_from_grid(final_grid)
    for row in rows:
        print(" ".join(row))

# =========================================================
# STEP 1 - CAPTURE SCAN
# =========================================================

def capture_scan():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. - grid_program_pi_runtime_still_capture_markerfix.py:166")
        return False

    print("Camera opened. - grid_program_pi_runtime_still_capture_markerfix.py:169")
    print("Instructions: - grid_program_pi_runtime_still_capture_markerfix.py:170")
    print("Press 'c' to capture current heading - grid_program_pi_runtime_still_capture_markerfix.py:171")
    print("Capture order: front > right > back > left - grid_program_pi_runtime_still_capture_markerfix.py:172")
    print("After the 4th capture, the program will continue automatically. - grid_program_pi_runtime_still_capture_markerfix.py:173")

    idx = 0
    last_capture_msg = ""

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR: Failed to read frame from camera. - grid_program_pi_runtime_still_capture_markerfix.py:181")
            cap.release()
            cv2.destroyAllWindows()
            return False

        # =====================================================
        # DISPLAY BLOCK START
        # CUT THIS WHOLE BLOCK LATER IF YOU NO LONGER WANT DISPLAY
        # Also cut the cv2.imshow(...) line below
        # =====================================================
        display = frame.copy()

        if idx < len(HEADINGS):
            current_heading = HEADINGS[idx]
            put_text(display, f"Current heading: {current_heading}", 30, 0.9, 2)
            put_text(display, "Press 'c' to capture this view", 65)
            put_text(display, "Rotate camera/robot manually before each capture", 95)
        else:
            put_text(display, "All 4 captures completed. Continuing...", 30, 0.8, 2)

        if last_capture_msg:
            put_text(display, last_capture_msg, FRAME_HEIGHT - 20, 0.65, 2)

        cv2.imshow("capture_scan", display)
        # =====================================================
        # DISPLAY BLOCK END
        # =====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if idx >= len(HEADINGS):
                continue

            heading = HEADINGS[idx]
            filename = os.path.join(SAVE_DIR, f"{heading}.jpg")

            ok = cv2.imwrite(filename, frame)
            if ok:
                print(f"Saved: {filename} - grid_program_pi_runtime_still_capture_markerfix.py:220")
                last_capture_msg = f"Saved {heading}.jpg"
                idx += 1
                time.sleep(0.4)

                if idx == len(HEADINGS):
                    print("All 4 captures completed. Proceeding automatically... - grid_program_pi_runtime_still_capture_markerfix.py:226")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
            else:
                print(f"ERROR: Failed to save {filename} - grid_program_pi_runtime_still_capture_markerfix.py:231")
                last_capture_msg = f"Failed to save {heading}.jpg"

# =========================================================
# STEP 2 - DETECT COLORS
# =========================================================

def load_model_bundle(model_path):
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        print("Joblib file contains a dict. Keys found: - grid_program_pi_runtime_still_capture_markerfix.py:242", list(obj.keys()))

        if "model" not in obj:
            raise ValueError("Joblib dict does not contain key 'model'.")

        model = obj["model"]
        class_names = obj.get("classes", None)

        if class_names is not None:
            class_names = [str(x).lower() for x in class_names]
            print("Loaded class names from joblib: - grid_program_pi_runtime_still_capture_markerfix.py:252", class_names)
        else:
            print("No explicit class names found in joblib. - grid_program_pi_runtime_still_capture_markerfix.py:254")

        return model, class_names

    if hasattr(obj, "predict") or hasattr(obj, "predict_proba"):
        print("Loaded model directly from joblib file. - grid_program_pi_runtime_still_capture_markerfix.py:259")
        return obj, None

    raise ValueError("No usable classifier found in joblib file.")


def extract_features(img):
    h, w = img.shape[:2]

    y0 = int(0.25 * h)
    y1 = int(0.75 * h)
    x0 = int(0.25 * w)
    x1 = int(0.75 * w)

    roi = img[y0:y1, x0:x1]
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


def normalize_predicted_label(raw_label, class_names):
    s = str(raw_label).lower()

    if s in LABEL_TO_CHAR:
        return s

    if class_names is not None:
        try:
            idx = int(raw_label)
            if 0 <= idx < len(class_names):
                return str(class_names[idx]).lower()
        except Exception:
            pass

    return s


def classify_tile(model, class_names, tile_bgr):
    feats = extract_features(tile_bgr)
    if feats is None:
        return "unknown", 0.0, "?", {}

    x = feats.reshape(1, -1)
    prob_map = {}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]

        if hasattr(model, "classes_"):
            raw_classes = list(model.classes_)
        else:
            raw_classes = list(range(len(probs)))

        best_idx = int(np.argmax(probs))
        raw_label = raw_classes[best_idx]
        label = normalize_predicted_label(raw_label, class_names)
        conf = float(probs[best_idx])

        for c, p in zip(raw_classes, probs):
            cname = normalize_predicted_label(c, class_names)
            prob_map[str(cname)] = float(p)
    else:
        raw_label = model.predict(x)[0]
        label = normalize_predicted_label(raw_label, class_names)
        conf = 1.0

    ch = LABEL_TO_CHAR[label] if label in LABEL_TO_CHAR else "?"

    if conf < CONF_THRESH:
        return "unknown", conf, "?", prob_map

    return label, conf, ch, prob_map


def get_three_slot_rois(img, roi_top_frac, roi_bot_frac, pad_x_frac, pad_y_frac):
    h, w = img.shape[:2]

    y0 = int(roi_top_frac * h)
    y1 = int(roi_bot_frac * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]
    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(pad_x_frac * (sx1 - sx0))
        pad_y = int(pad_y_frac * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def detect_colors():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH} - grid_program_pi_runtime_still_capture_markerfix.py:378")
        return None

    try:
        model, class_names = load_model_bundle(MODEL_PATH)
    except Exception as e:
        print("ERROR loading model: - grid_program_pi_runtime_still_capture_markerfix.py:384", e)
        return None

    final_grid = {
        (-1, +1): "?",
        ( 0, +1): "?",
        (+1, +1): "?",
        (-1,  0): "?",
        ( 0,  0): "A",
        (+1,  0): "?",
        (-1, -1): "?",
        ( 0, -1): "?",
        (+1, -1): "?",
    }

    for heading in HEADINGS:
        path = os.path.join(SAVE_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            print(f"ERROR: Missing image: {path} - grid_program_pi_runtime_still_capture_markerfix.py:403")
            return None

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path} - grid_program_pi_runtime_still_capture_markerfix.py:408")
            return None

        slots = get_three_slot_rois(
            img,
            COLOR_ROI_TOP_FRAC,
            COLOR_ROI_BOT_FRAC,
            COLOR_SLOT_PAD_X_FRAC,
            COLOR_SLOT_PAD_Y_FRAC
        )

        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading} - grid_program_pi_runtime_still_capture_markerfix.py:420")
            return None

        print(f"\nHeading: {heading} - grid_program_pi_runtime_still_capture_markerfix.py:423")
        for i, tile in enumerate(slots):
            label, conf, ch, prob_map = classify_tile(model, class_names, tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            print(f"slot {i}: label={label}, conf={conf:.4f}, char={ch} - grid_program_pi_runtime_still_capture_markerfix.py:429")
            if prob_map:
                rounded_probs = {k: round(v, 4) for k, v in prob_map.items()}
                print("probs: - grid_program_pi_runtime_still_capture_markerfix.py:432", rounded_probs)

    pretty_print_grid(final_grid, "Final 3x3 color matrix:")
    return matrix_rows_from_grid(final_grid)

# =========================================================
# STEP 3 - DETECT OBJECTS
# =========================================================

def detect_one_object_slot(slot_bgr):
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {}

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

    if BLUR_ODD > 1:
        k = BLUR_ODD if BLUR_ODD % 2 == 1 else BLUR_ODD + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size

    white_mask = cv2.inRange(gray, WHITE_MIN, 255)
    white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size

    black_mask = cv2.inRange(gray, 0, BLACK_MAX)
    black_ratio = float(np.count_nonzero(black_mask)) / black_mask.size

    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(1, HOUGH_TH),
        minLineLength=max(1, MIN_LINE),
        maxLineGap=max(0, MAX_GAP)
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

    metrics = {
        "red_ratio": round(red_ratio, 4),
        "white_ratio": round(white_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "diag_pos": int(diag_pos),
        "diag_neg": int(diag_neg),
        "has_x_shape": bool(has_x_shape),
    }

    if white_ratio > WHITE_RATIO_TH and red_ratio > RED_RATIO_TH and has_x_shape:
        return "O", metrics

    if white_ratio > WHITE_RATIO_TH and black_ratio > BLACK_RATIO_TH and has_x_shape:
        return "T", metrics

    if white_ratio < EMPTY_WHITE_TH and red_ratio < EMPTY_RED_TH and black_ratio < EMPTY_BLACK_TH:
        return "E", metrics

    return "?", metrics


def detect_objects():
    final_grid = {
        (-1, +1): "?",
        ( 0, +1): "?",
        (+1, +1): "?",
        (-1,  0): "?",
        ( 0,  0): "A",
        (+1,  0): "?",
        (-1, -1): "?",
        ( 0, -1): "?",
        (+1, -1): "?",
    }

    for heading in HEADINGS:
        path = os.path.join(SAVE_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            print(f"ERROR: Missing image: {path} - grid_program_pi_runtime_still_capture_markerfix.py:537")
            return None

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path} - grid_program_pi_runtime_still_capture_markerfix.py:542")
            return None

        slots = get_three_slot_rois(
            img,
            OBJ_ROI_TOP_FRAC,
            OBJ_ROI_BOT_FRAC,
            OBJ_SLOT_PAD_X_FRAC,
            OBJ_SLOT_PAD_Y_FRAC
        )

        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading} - grid_program_pi_runtime_still_capture_markerfix.py:554")
            return None

        print(f"\nHeading: {heading} - grid_program_pi_runtime_still_capture_markerfix.py:557")
        for i, tile in enumerate(slots):
            obj_char, metrics = detect_one_object_slot(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = obj_char

            print(f"slot {i}: object={obj_char} - grid_program_pi_runtime_still_capture_markerfix.py:563")
            print(f"metrics: {metrics} - grid_program_pi_runtime_still_capture_markerfix.py:564")

    pretty_print_grid(final_grid, "Final 3x3 object matrix:")
    return matrix_rows_from_grid(final_grid)

# =========================================================
# STEP 4 - MAP LOCATION
# =========================================================

def rotate_3x3_ccw(mat):
    return [
        [mat[0][2], mat[1][2], mat[2][2]],
        [mat[0][1], mat[1][1], mat[2][1]],
        [mat[0][0], mat[1][0], mat[2][0]],
    ]


def rotate_n_ccw(mat, n):
    out = [row[:] for row in mat]
    for _ in range(n % 4):
        out = rotate_3x3_ccw(out)
    return out


def get_window_3x3(grid, center_r, center_c):
    rows = len(grid)
    cols = len(grid[0])

    if center_r - 1 < 0 or center_r + 1 >= rows:
        return None
    if center_c - 1 < 0 or center_c + 1 >= cols:
        return None

    return [
        [grid[center_r - 1][center_c - 1], grid[center_r - 1][center_c], grid[center_r - 1][center_c + 1]],
        [grid[center_r][center_c - 1],     'A',                          grid[center_r][center_c + 1]],
        [grid[center_r + 1][center_c - 1], grid[center_r + 1][center_c], grid[center_r + 1][center_c + 1]],
    ]


def score_match(local_3x3, window_3x3):
    known = 0
    matches = 0
    mismatches = 0

    for r in range(3):
        for c in range(3):
            lv = local_3x3[r][c]
            wv = window_3x3[r][c]

            if lv == 'A':
                continue
            if lv == '?':
                continue

            known += 1
            if lv == wv:
                matches += 1
            else:
                mismatches += 1

    return {
        "known": known,
        "matches": matches,
        "mismatches": mismatches,
        "score": matches
    }


def rotation_to_facing(rotation_ccw_deg):
    mapping = {
        0: "UP",
        90: "RIGHT",
        180: "DOWN",
        270: "LEFT",
    }
    return mapping[rotation_ccw_deg]


def rotate_direction(direction, steps_ccw):
    dirs = ["UP", "LEFT", "DOWN", "RIGHT"]
    idx = dirs.index(direction)
    return dirs[(idx + steps_ccw) % 4]


def get_scan_order(scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    scan_start_local = scan_start_local.upper()
    scan_sweep = scan_sweep.lower()

    if scan_sweep == "cw":
        base_order = ["FRONT", "RIGHT", "BACK", "LEFT"]
    elif scan_sweep == "ccw":
        base_order = ["FRONT", "LEFT", "BACK", "RIGHT"]
    else:
        raise ValueError(f"scan_sweep must be 'cw' or 'ccw', got: {scan_sweep}")

    if scan_start_local not in base_order:
        raise ValueError(f"scan_start_local must be one of {base_order}, got: {scan_start_local}")

    start_idx = base_order.index(scan_start_local)
    ordered = base_order[start_idx:] + base_order[:start_idx]
    return ordered[:num_views]


def local_heading_to_map_direction(start_map_direction, local_heading):
    local_heading = local_heading.upper()

    local_steps_ccw = {
        "FRONT": 0,
        "LEFT": 1,
        "BACK": 2,
        "RIGHT": 3,
    }

    return rotate_direction(start_map_direction, local_steps_ccw[local_heading])


def get_final_camera_direction_after_scan(start_map_direction, scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    order = get_scan_order(scan_start_local, scan_sweep, num_views)
    final_local_heading = order[-1]
    final_map_direction = local_heading_to_map_direction(start_map_direction, final_local_heading)
    return final_local_heading, final_map_direction


def direction_to_char(direction):
    mapping = {
        "UP": "U",
        "RIGHT": "R",
        "DOWN": "D",
        "LEFT": "L",
    }
    return mapping[direction]


def build_compact_17char(color_3x3, object_3x3, final_direction):
    out = []

    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                continue

            color_char = str(color_3x3[r][c]).strip().upper()[:1]
            obj_char = str(object_3x3[r][c]).strip().upper()[:1]

            out.append(color_char + obj_char)

    out.append(direction_to_char(final_direction))
    return "".join(out)


def find_best_match(local_3x3, big_grid):
    rows = len(big_grid)
    cols = len(big_grid[0])

    candidates = []

    for rot_steps in range(4):
        rotated = rotate_n_ccw(local_3x3, rot_steps)
        rotation_ccw_deg = rot_steps * 90

        for center_r in range(1, rows - 1):
            for center_c in range(1, cols - 1):
                window = get_window_3x3(big_grid, center_r, center_c)
                if window is None:
                    continue

                s = score_match(rotated, window)

                if s["known"] < MIN_KNOWN_NEIGHBORS:
                    continue
                if s["mismatches"] > MAX_MISMATCHES:
                    continue

                candidates.append({
                    "center_row": center_r,
                    "center_col": center_c,
                    "rotation_ccw_deg": rotation_ccw_deg,
                    "facing": rotation_to_facing(rotation_ccw_deg),
                    "known": s["known"],
                    "matches": s["matches"],
                    "mismatches": s["mismatches"],
                    "score": s["score"],
                    "rotated_local": rotated,
                    "window": window,
                })

    if not candidates:
        return None, []

    candidates.sort(
        key=lambda x: (x["score"], -x["mismatches"], x["known"]),
        reverse=True
    )

    return candidates[0], candidates


def map_location(local_color_3x3, local_object_3x3):
    best, candidates = find_best_match(local_color_3x3, BIG_GRID)

    if best is None:
        print("\nNo valid match found. - grid_program_pi_runtime_still_capture_markerfix.py:766")
        print("Try: - grid_program_pi_runtime_still_capture_markerfix.py:767")
        print("lowering MIN_KNOWN_NEIGHBORS - grid_program_pi_runtime_still_capture_markerfix.py:768")
        print("increasing MAX_MISMATCHES - grid_program_pi_runtime_still_capture_markerfix.py:769")
        print("improving color detection - grid_program_pi_runtime_still_capture_markerfix.py:770")
        return None

    camera_direction_before_scan = best["facing"]

    final_local_heading_after_scan, camera_direction_after_scan = get_final_camera_direction_after_scan(
        start_map_direction=camera_direction_before_scan,
        scan_start_local=SCAN_START_LOCAL,
        scan_sweep=SCAN_SWEEP,
        num_views=NUM_VIEWS
    )

    compact_17char = build_compact_17char(
        best["window"],
        local_object_3x3,
        camera_direction_after_scan
    )

    print("\nBEST MATCH FOUND - grid_program_pi_runtime_still_capture_markerfix.py:788")
    print("")
    print(f"center_row                     = {best['center_row']} - grid_program_pi_runtime_still_capture_markerfix.py:790")
    print(f"center_col                     = {best['center_col']} - grid_program_pi_runtime_still_capture_markerfix.py:791")
    print(f"rotation_ccw_deg               = {best['rotation_ccw_deg']} - grid_program_pi_runtime_still_capture_markerfix.py:792")
    print(f"facing_in_big_grid             = {best['facing']} - grid_program_pi_runtime_still_capture_markerfix.py:793")
    print(f"camera_direction_before_scan   = {camera_direction_before_scan} - grid_program_pi_runtime_still_capture_markerfix.py:794")
    print(f"scan_start_local               = {SCAN_START_LOCAL} - grid_program_pi_runtime_still_capture_markerfix.py:795")
    print(f"scan_sweep                     = {SCAN_SWEEP} - grid_program_pi_runtime_still_capture_markerfix.py:796")
    print(f"final_local_heading_after_scan = {final_local_heading_after_scan} - grid_program_pi_runtime_still_capture_markerfix.py:797")
    print(f"camera_direction_after_scan    = {camera_direction_after_scan} - grid_program_pi_runtime_still_capture_markerfix.py:798")
    print(f"compact_17char                 = {compact_17char} - grid_program_pi_runtime_still_capture_markerfix.py:799")
    print("\nMATCHED WINDOW USED FOR COMPACT COLOR: - grid_program_pi_runtime_still_capture_markerfix.py:800")
    print(pretty_matrix(best["window - grid_program_pi_runtime_still_capture_markerfix.py:801"]))
    print(f"known_neighbors                = {best['known']} - grid_program_pi_runtime_still_capture_markerfix.py:802")
    print(f"matches                        = {best['matches']} - grid_program_pi_runtime_still_capture_markerfix.py:803")
    print(f"mismatches                     = {best['mismatches']} - grid_program_pi_runtime_still_capture_markerfix.py:804")
    print(f"score                          = {best['score']}/{best['known']} - grid_program_pi_runtime_still_capture_markerfix.py:805")

    return compact_17char

# =========================================================
# STEP 5 - SAVE FINAL RESULT
# =========================================================

def save_compact_result(compact_17char):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(COMPACT_RESULT_FILE, "w") as f:
        f.write(compact_17char + "\n")
    print(f"\nSaved final compact result: {COMPACT_RESULT_FILE} - grid_program_pi_runtime_still_capture_markerfix.py:817")

# =========================================================
# MAIN
# =========================================================

def main():
    cleanup_previous_run()

    print("\n - grid_program_pi_runtime_still_capture_markerfix.py:826" + "=" * 70)
    print("STEP 1  SCANNING - grid_program_pi_runtime_still_capture_markerfix.py:827")
    print("= - grid_program_pi_runtime_still_capture_markerfix.py:828" * 70)
    ok = capture_scan()
    if not ok:
        print("\nPipeline stopped at scan stage. - grid_program_pi_runtime_still_capture_markerfix.py:831")
        return

    print("\n - grid_program_pi_runtime_still_capture_markerfix.py:834" + "=" * 70)
    print("STEP 2  COLOR DETECTION - grid_program_pi_runtime_still_capture_markerfix.py:835")
    print("= - grid_program_pi_runtime_still_capture_markerfix.py:836" * 70)
    local_color_3x3 = detect_colors()
    if local_color_3x3 is None:
        print("\nPipeline stopped at color detection stage. - grid_program_pi_runtime_still_capture_markerfix.py:839")
        return

    print("\n - grid_program_pi_runtime_still_capture_markerfix.py:842" + "=" * 70)
    print("STEP 3  OBJECT DETECTION - grid_program_pi_runtime_still_capture_markerfix.py:843")
    print("= - grid_program_pi_runtime_still_capture_markerfix.py:844" * 70)
    local_object_3x3 = detect_objects()
    if local_object_3x3 is None:
        print("\nPipeline stopped at object detection stage. - grid_program_pi_runtime_still_capture_markerfix.py:847")
        return

    print("\n - grid_program_pi_runtime_still_capture_markerfix.py:850" + "=" * 70)
    print("STEP 4  LOCATION MAPPING - grid_program_pi_runtime_still_capture_markerfix.py:851")
    print("= - grid_program_pi_runtime_still_capture_markerfix.py:852" * 70)
    compact_17char = map_location(local_color_3x3, local_object_3x3)
    if compact_17char is None:
        print("\nPipeline stopped at mapping stage. - grid_program_pi_runtime_still_capture_markerfix.py:855")
        return

    print("\n - grid_program_pi_runtime_still_capture_markerfix.py:858" + "=" * 70)
    print("STEP 5  SAVE FINAL RESULT - grid_program_pi_runtime_still_capture_markerfix.py:859")
    print("= - grid_program_pi_runtime_still_capture_markerfix.py:860" * 70)
    save_compact_result(compact_17char)

    print("\n - grid_program_pi_runtime_still_capture_markerfix.py:863" + "=" * 70)
    print("PIPELINE FINISHED - grid_program_pi_runtime_still_capture_markerfix.py:864")
    print("= - grid_program_pi_runtime_still_capture_markerfix.py:865" * 70)
    print(f"Final compact result: {compact_17char} - grid_program_pi_runtime_still_capture_markerfix.py:866")


if __name__ == "__main__":
    main()