import cv2
import numpy as np
import time

# =========================
# CAMERA SETTINGS
# =========================
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480

# =========================
# ROI / SLOT SETTINGS
# bottom band where nearest row of tiles is expected
# =========================
ROI_TOP_FRAC = 0.62
ROI_BOT_FRAC = 0.93

SLOT_PAD_X_FRAC = 0.02
SLOT_PAD_Y_FRAC = 0.06

# =========================
# DETECTION THRESHOLDS
# =========================
WHITE_THRESH = 170          # pixel must be brighter than this to count as white
BLACK_THRESH = 90           # pixel must be darker than this to count as black
MIN_WHITE_AREA = 900        # minimum white contour area
MIN_WHITE_FILL = 0.45       # how much of candidate box should be white
MIN_DIAG_SCORE = 0.12       # black-on-diagonal strength
MIN_COMBINED_SCORE = 0.28   # diag1 + diag2 must be at least this

SHOW_DEBUG = True


def make_slot_boxes(w, h):
    """Return 3 slot boxes in the bottom ROI."""
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


def diagonal_scores(gray_roi):
    """
    Cheap test for black X:
    Count dark pixels near both diagonals.
    """
    h, w = gray_roi.shape
    if h < 20 or w < 20:
        return 0.0, 0.0

    black = (gray_roi < BLACK_THRESH).astype(np.uint8)

    yy, xx = np.indices((h, w))
    band = max(2, int(min(h, w) * 0.08))

    # main diagonal: y = (h-1)/(w-1) * x
    d1 = np.abs(yy - ((h - 1) / max(1, (w - 1))) * xx) <= band

    # other diagonal: y = (h-1) - ((h-1)/(w-1)) * x
    d2 = np.abs(yy - ((h - 1) - ((h - 1) / max(1, (w - 1))) * xx)) <= band

    score1 = black[d1].mean() if np.any(d1) else 0.0
    score2 = black[d2].mean() if np.any(d2) else 0.0

    return float(score1), float(score2)


def detect_white_box_with_black_x(bgr):
    """
    Detect one white box with a black X in this image region.
    Returns:
        found (bool),
        best_rect (x, y, w, h) or None,
        info (dict)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold bright white regions
    white_mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)[1]

    # clean mask a little
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_rect = None
    best_info = {}

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

        roi_gray = gray[y:y+h, x:x+w]
        roi_white = white_mask[y:y+h, x:x+w]

        if roi_gray.size == 0:
            continue

        white_fill = np.mean(roi_white > 0)
        if white_fill < MIN_WHITE_FILL:
            continue

        # focus more on inner part of the white box
        mx = int(w * 0.12)
        my = int(h * 0.12)
        ix0 = mx
        ix1 = w - mx
        iy0 = my
        iy1 = h - my
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        inner_gray = roi_gray[iy0:iy1, ix0:ix1]

        d1, d2 = diagonal_scores(inner_gray)
        combined = d1 + d2

        # score favors good diagonal X and decent white fill
        score = combined + 0.3 * white_fill

        if d1 >= MIN_DIAG_SCORE and d2 >= MIN_DIAG_SCORE and combined >= MIN_COMBINED_SCORE:
            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h)
                best_info = {
                    "white_fill": white_fill,
                    "diag1": d1,
                    "diag2": d2,
                    "combined": combined,
                    "score": score
                }

    found = best_rect is not None
    return found, best_rect, best_info


def draw_text(img, text, x, y, ok=True):
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Press q to quit.")

    last_print = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame.")
            break

        h, w = frame.shape[:2]
        view = frame.copy()

        # make 3 slot boxes in bottom ROI
        slots = make_slot_boxes(w, h)

        present_flags = []

        for i, (x0, y0, x1, y1) in enumerate(slots):
            slot = frame[y0:y1, x0:x1]
            found, rect, info = detect_white_box_with_black_x(slot)
            present_flags.append(found)

            # draw slot box
            cv2.rectangle(view, (x0, y0), (x1, y1), (255, 255, 0), 2)

            label = f"slot {i}: PRESENT" if found else f"slot {i}: empty"
            draw_text(view, label, x0 + 5, y0 + 20, ok=found)

            if found and rect is not None:
                rx, ry, rw, rh = rect
                cv2.rectangle(view, (x0 + rx, y0 + ry), (x0 + rx + rw, y0 + ry + rh), (0, 255, 0), 2)

                if SHOW_DEBUG:
                    dbg = f"d1={info['diag1']:.2f} d2={info['diag2']:.2f}"
                    draw_text(view, dbg, x0 + 5, y1 - 10, ok=True)

        # print sometimes, not every frame
        now = time.time()
        if now - last_print > 0.8:
            print("object_presence:", present_flags)
            last_print = now

        cv2.imshow("Lightweight Object Presence Detector", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
