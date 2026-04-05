import os
import cv2
import numpy as np

# =========================================================
# INPUT SCAN IMAGES
# =========================================================
SCAN_DIR = "scan_views"

IMAGE_PATHS = {
    "front": os.path.join(SCAN_DIR, "front.jpg"),
    "right": os.path.join(SCAN_DIR, "right.jpg"),
    "back":  os.path.join(SCAN_DIR, "back.jpg"),
    "left":  os.path.join(SCAN_DIR, "left.jpg"),
}

# =========================================================
# OUTPUT
# =========================================================
DEBUG_DIR = "debug_object_scan"
os.makedirs(DEBUG_DIR, exist_ok=True)

# =========================================================
# ROI / SLOT SETTINGS
# same style as your program
# =========================================================
ROI_TOP_FRAC = 0.62
ROI_BOT_FRAC = 0.93
SLOT_PAD_X_FRAC = 0.02
SLOT_PAD_Y_FRAC = 0.06

# =========================================================
# WHITE BOX DETECTION
# =========================================================
WHITE_THRESH = 170
MIN_WHITE_AREA = 900
MIN_WHITE_FILL = 0.45

# =========================================================
# TARGET = white box + thick black X
# =========================================================
BLACK_THRESH = 85
MIN_BLACK_DIAG_SCORE = 0.12
MIN_BLACK_COMBINED = 0.28

# =========================================================
# OBSTACLE = white box + thick red X
# =========================================================
RED_MIN_R = 120
RED_MARGIN = 40
MIN_RED_DIAG_SCORE = 0.10
MIN_RED_COMBINED = 0.24

SHOW_DEBUG = True

EMPTY = "EMPTY"
TARGET = "TARGET"
OBSTACLE = "OBSTACLE"
AGENT = "AGENT"

HEADINGS = ["front", "right", "back", "left"]

# local coordinates around agent
HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


def state_to_char(state):
    if state == TARGET:
        return "T"
    if state == OBSTACLE:
        return "X"
    if state == AGENT:
        return "A"
    return "E"


def blank_local_grid():
    grid = {}
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            grid[(dx, dy)] = EMPTY
    grid[(0, 0)] = AGENT
    return grid


def pretty_matrix(grid):
    rows = []
    for dy in [1, 0, -1]:
        row = []
        for dx in [-1, 0, 1]:
            row.append(state_to_char(grid[(dx, dy)]))
        rows.append(row)
    return rows


def print_matrix(grid):
    mat = pretty_matrix(grid)
    print("\nFinal 3x3 object matrix:")
    for row in mat:
        print(" ".join(row))


def make_slot_boxes(w, h):
    y0 = int(h * ROI_TOP_FRAC)
    y1 = int(h * ROI_BOT_FRAC)

    roi_h = y1 - y0
    slot_w = w // 3

    boxes = []
    for i in range(3):
        x0 = i * slot_w
        x1 = (i + 1) * slot_w

        pad_x = int(slot_w * SLOT_PAD_X_FRAC)
        pad_y = int(roi_h * SLOT_PAD_Y_FRAC)

        sx0 = max(0, x0 + pad_x)
        sx1 = min(w, x1 - pad_x)
        sy0 = max(0, y0 + pad_y)
        sy1 = min(h, y1 - pad_y)

        boxes.append((sx0, sy0, sx1, sy1))

    return boxes


def diagonal_band_masks(h, w, band_frac=0.08):
    yy, xx = np.indices((h, w))
    band = max(2, int(min(h, w) * band_frac))

    d1 = np.abs(yy - ((h - 1) / max(1, (w - 1))) * xx) <= band
    d2 = np.abs(yy - ((h - 1) - ((h - 1) / max(1, (w - 1))) * xx)) <= band

    return d1, d2


def get_white_candidates(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    white_mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_WHITE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 25 or h < 25:
            continue

        aspect = w / float(h)
        if aspect < 0.60 or aspect > 1.40:
            continue

        roi_white = white_mask[y:y+h, x:x+w]
        if roi_white.size == 0:
            continue

        white_fill = np.mean(roi_white > 0)
        if white_fill < MIN_WHITE_FILL:
            continue

        candidates.append((x, y, w, h, white_fill))

    return candidates


def inner_crop(arr, frac=0.12):
    h, w = arr.shape[:2]
    mx = int(w * frac)
    my = int(h * frac)

    x0 = mx
    x1 = w - mx
    y0 = my
    y1 = h - my

    if x1 <= x0 or y1 <= y0:
        return arr

    return arr[y0:y1, x0:x1]


def black_x_scores(inner_bgr):
    gray = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)
    black = (gray < BLACK_THRESH)

    h, w = gray.shape
    if h < 20 or w < 20:
        return 0.0, 0.0

    d1, d2 = diagonal_band_masks(h, w)
    s1 = float(black[d1].mean()) if np.any(d1) else 0.0
    s2 = float(black[d2].mean()) if np.any(d2) else 0.0
    return s1, s2


def red_x_scores(inner_bgr):
    b = inner_bgr[:, :, 0].astype(np.int16)
    g = inner_bgr[:, :, 1].astype(np.int16)
    r = inner_bgr[:, :, 2].astype(np.int16)

    red = (r >= RED_MIN_R) & ((r - g) >= RED_MARGIN) & ((r - b) >= RED_MARGIN)

    h, w = red.shape
    if h < 20 or w < 20:
        return 0.0, 0.0

    d1, d2 = diagonal_band_masks(h, w)
    s1 = float(red[d1].mean()) if np.any(d1) else 0.0
    s2 = float(red[d2].mean()) if np.any(d2) else 0.0
    return s1, s2


def classify_marker_in_slot(slot_bgr):
    candidates = get_white_candidates(slot_bgr)

    best_score = -1.0
    best_state = EMPTY
    best_rect = None
    best_info = {}

    for (x, y, w, h, white_fill) in candidates:
        roi = slot_bgr[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        inner = inner_crop(roi, frac=0.12)
        if inner.size == 0:
            continue

        b1, b2 = black_x_scores(inner)
        r1, r2 = red_x_scores(inner)

        black_combined = b1 + b2
        red_combined = r1 + r2

        is_target = (
            b1 >= MIN_BLACK_DIAG_SCORE and
            b2 >= MIN_BLACK_DIAG_SCORE and
            black_combined >= MIN_BLACK_COMBINED
        )

        is_obstacle = (
            r1 >= MIN_RED_DIAG_SCORE and
            r2 >= MIN_RED_DIAG_SCORE and
            red_combined >= MIN_RED_COMBINED
        )

        state = EMPTY
        score = -1.0

        if is_target or is_obstacle:
            target_score = black_combined + 0.25 * white_fill
            obstacle_score = red_combined + 0.25 * white_fill

            if is_target and (not is_obstacle or target_score >= obstacle_score):
                state = TARGET
                score = target_score
            elif is_obstacle:
                state = OBSTACLE
                score = obstacle_score

        if score > best_score:
            best_score = score
            best_state = state
            best_rect = (x, y, w, h)
            best_info = {
                "white_fill": white_fill,
                "black_combined": black_combined,
                "red_combined": red_combined,
                "score": score
            }

    return best_state, best_rect, best_info


def apply_heading_capture(local_grid, heading, states):
    positions = HEADING_TO_POSITIONS[heading]
    for pos, state in zip(positions, states):
        if pos != (0, 0):
            local_grid[pos] = state


def draw_text(img, text, x, y, color=(255, 255, 255), scale=0.55, thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def color_for_state(state):
    if state == TARGET:
        return (0, 255, 0)
    if state == OBSTACLE:
        return (0, 0, 255)
    if state == AGENT:
        return (255, 255, 255)
    return (180, 180, 180)


def process_heading_image(heading, img):
    view = img.copy()
    h, w = img.shape[:2]
    slots = make_slot_boxes(w, h)

    states = []
    rects = []
    infos = []

    for i, (x0, y0, x1, y1) in enumerate(slots):
        slot = img[y0:y1, x0:x1]
        state, rect, info = classify_marker_in_slot(slot)

        states.append(state)
        rects.append(rect)
        infos.append(info)

        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 255, 0), 2)
        draw_text(view, f"slot {i}: {state}", x0 + 5, y0 + 20, color_for_state(state))

        if rect is not None and state != EMPTY:
            rx, ry, rw, rh = rect
            cv2.rectangle(
                view,
                (x0 + rx, y0 + ry),
                (x0 + rx + rw, y0 + ry + rh),
                color_for_state(state),
                2
            )

            if SHOW_DEBUG:
                b = info.get("black_combined", 0.0)
                r = info.get("red_combined", 0.0)
                draw_text(view, f"B:{b:.2f}", x0 + 5, y1 - 28, color_for_state(state), scale=0.45, thick=1)
                draw_text(view, f"R:{r:.2f}", x0 + 5, y1 - 10, color_for_state(state), scale=0.45, thick=1)

    draw_text(view, f"Heading: {heading}", 10, 25, (255, 255, 255))
    draw_text(view, f"Row: {','.join(state_to_char(s) for s in states)}", 180, 25, (255, 255, 255))

    out_path = os.path.join(DEBUG_DIR, f"{heading}_debug.jpg")
    cv2.imwrite(out_path, view)

    return states, out_path


def main():
    local_grid = blank_local_grid()

    print("=== Object detection from saved scan pictures ===")

    for heading in HEADINGS:
        path = IMAGE_PATHS[heading]

        if not os.path.exists(path):
            print(f"ERROR: Missing image for {heading}: {path}")
            return

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        states, debug_path = process_heading_image(heading, img)
        apply_heading_capture(local_grid, heading, states)

        print(f"\nHeading: {heading}")
        print("image      =", path)
        print("states     =", states)
        print("state_row  =", [state_to_char(s) for s in states])
        print("debug_view =", debug_path)

    print_matrix(local_grid)

    mat = pretty_matrix(local_grid)
    print("\nMatrix rows for packet-style use:")
    for row in mat:
        print(",".join(row))

    # Optional combined text file
    txt_path = os.path.join(DEBUG_DIR, "final_object_matrix.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Final 3x3 object matrix\n")
        for row in mat:
            f.write(",".join(row) + "\n")

    print(f"\nSaved final matrix text to: {txt_path}")
    print(f"Saved debug images to: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
