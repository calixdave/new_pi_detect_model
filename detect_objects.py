import cv2
import numpy as np
import time

# =========================================
# CAMERA SETTINGS
# =========================================
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480

# =========================================
# ROI / SLOT SETTINGS
# same spirit as your grid program
# =========================================
ROI_TOP_FRAC = 0.62
ROI_BOT_FRAC = 0.93

SLOT_PAD_X_FRAC = 0.02
SLOT_PAD_Y_FRAC = 0.06

# =========================================
# WHITE BOX DETECTION
# =========================================
WHITE_THRESH = 170
MIN_WHITE_AREA = 900
MIN_WHITE_FILL = 0.45

# =========================================
# BLACK-X TARGET DETECTION
# =========================================
BLACK_THRESH = 85
MIN_BLACK_DIAG_SCORE = 0.12
MIN_BLACK_COMBINED = 0.28

# =========================================
# RED-X OBSTACLE DETECTION
# tuned to be lightweight
# =========================================
MIN_RED_DIAG_SCORE = 0.10
MIN_RED_COMBINED = 0.24

# red mask rule:
# pixel is "red enough" if R is high and clearly above G/B
RED_MIN_R = 120
RED_MARGIN = 40

SHOW_DEBUG = True


# state names
EMPTY = "EMPTY"
TARGET = "TARGET"      # white box + black X
OBSTACLE = "OBSTACLE"  # white box + red X


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

    # main diagonal
    d1 = np.abs(yy - ((h - 1) / max(1, (w - 1))) * xx) <= band

    # anti diagonal
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

    return gray, white_mask, candidates


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
    """
    Return:
        state: EMPTY / TARGET / OBSTACLE
        rect:  (x,y,w,h) inside slot, or None
        info:  debug dictionary
    """
    gray, white_mask, candidates = get_white_candidates(slot_bgr)

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

        # avoid confusion if both somehow respond
        # choose stronger type
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
                "black_d1": b1,
                "black_d2": b2,
                "black_combined": black_combined,
                "red_d1": r1,
                "red_d2": r2,
                "red_combined": red_combined,
                "score": score
            }

    return best_state, best_rect, best_info


def state_to_char(state):
    if state == TARGET:
        return "T"
    if state == OBSTACLE:
        return "X"
    return "E"


def draw_text(img, text, x, y, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Press q to quit.")

    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame.")
            break

        h, w = frame.shape[:2]
        view = frame.copy()

        slots = make_slot_boxes(w, h)
        states = []

        for i, (x0, y0, x1, y1) in enumerate(slots):
            slot = frame[y0:y1, x0:x1]

            state, rect, info = classify_marker_in_slot(slot)
            states.append(state)

            # slot outline
            cv2.rectangle(view, (x0, y0), (x1, y1), (255, 255, 0), 2)

            if state == TARGET:
                color = (0, 255, 0)
            elif state == OBSTACLE:
                color = (0, 0, 255)
            else:
                color = (180, 180, 180)

            draw_text(view, f"slot {i}: {state}", x0 + 5, y0 + 20, color)

            if rect is not None and state != EMPTY:
                rx, ry, rw, rh = rect
                cv2.rectangle(view, (x0 + rx, y0 + ry), (x0 + rx + rw, y0 + ry + rh), color, 2)

                if SHOW_DEBUG:
                    dbg1 = f"B:{info.get('black_combined', 0):.2f}"
                    dbg2 = f"R:{info.get('red_combined', 0):.2f}"
                    draw_text(view, dbg1, x0 + 5, y1 - 28, color)
                    draw_text(view, dbg2, x0 + 5, y1 - 8, color)

        state_row = [state_to_char(s) for s in states]

        now = time.time()
        if now - last_print > 0.8:
            print("state_row =", state_row)
            print("states    =", states)
            last_print = now

        draw_text(view, f"row: {','.join(state_row)}", 10, 25, (255, 255, 255))

        cv2.imshow("Lightweight Object Detector", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
