import cv2
import numpy as np

# =========================================================
# DEFAULTS
# =========================================================

CAM_INDEX = 0
FRAME_W = 960
FRAME_H = 540

WINDOW_MAIN = "Live Object Tuner"
WINDOW_CTRL = "Controls"

# =========================================================
# UI
# =========================================================

def nothing(x):
    pass


def make_trackbars():
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CTRL, 520, 520)

    # ROI / slot tuning
    cv2.createTrackbar("ROI_TOP_%", WINDOW_CTRL, 55, 100, nothing)
    cv2.createTrackbar("ROI_BOT_%", WINDOW_CTRL, 95, 100, nothing)
    cv2.createTrackbar("PAD_X_%", WINDOW_CTRL, 3, 30, nothing)
    cv2.createTrackbar("PAD_Y_%", WINDOW_CTRL, 6, 30, nothing)

    # white / black thresholds
    cv2.createTrackbar("WHITE_MIN", WINDOW_CTRL, 180, 255, nothing)
    cv2.createTrackbar("BLACK_MAX", WINDOW_CTRL, 60, 255, nothing)

    # HSV red thresholds
    cv2.createTrackbar("RED_S_MIN", WINDOW_CTRL, 100, 255, nothing)
    cv2.createTrackbar("RED_V_MIN", WINDOW_CTRL, 80, 255, nothing)

    # decision thresholds as integer percentages
    cv2.createTrackbar("WHITE_RATIO_x1000", WINDOW_CTRL, 180, 1000, nothing)   # 0.180
    cv2.createTrackbar("RED_RATIO_x1000", WINDOW_CTRL, 30, 1000, nothing)      # 0.030
    cv2.createTrackbar("BLACK_RATIO_x1000", WINDOW_CTRL, 50, 1000, nothing)    # 0.050
    cv2.createTrackbar("EMPTY_WHITE_x1000", WINDOW_CTRL, 120, 1000, nothing)   # 0.120
    cv2.createTrackbar("EMPTY_RED_x1000", WINDOW_CTRL, 10, 1000, nothing)      # 0.010
    cv2.createTrackbar("EMPTY_BLACK_x1000", WINDOW_CTRL, 30, 1000, nothing)    # 0.030

    # Hough / edges
    cv2.createTrackbar("CANNY1", WINDOW_CTRL, 60, 255, nothing)
    cv2.createTrackbar("CANNY2", WINDOW_CTRL, 160, 255, nothing)
    cv2.createTrackbar("HOUGH_TH", WINDOW_CTRL, 25, 100, nothing)
    cv2.createTrackbar("MIN_LINE", WINDOW_CTRL, 20, 300, nothing)
    cv2.createTrackbar("MAX_GAP", WINDOW_CTRL, 8, 100, nothing)

    # optional blur
    cv2.createTrackbar("BLUR_ODD", WINDOW_CTRL, 1, 15, nothing)


def get_params():
    roi_top = cv2.getTrackbarPos("ROI_TOP_%", WINDOW_CTRL) / 100.0
    roi_bot = cv2.getTrackbarPos("ROI_BOT_%", WINDOW_CTRL) / 100.0
    pad_x = cv2.getTrackbarPos("PAD_X_%", WINDOW_CTRL) / 100.0
    pad_y = cv2.getTrackbarPos("PAD_Y_%", WINDOW_CTRL) / 100.0

    white_min = cv2.getTrackbarPos("WHITE_MIN", WINDOW_CTRL)
    black_max = cv2.getTrackbarPos("BLACK_MAX", WINDOW_CTRL)
    red_s_min = cv2.getTrackbarPos("RED_S_MIN", WINDOW_CTRL)
    red_v_min = cv2.getTrackbarPos("RED_V_MIN", WINDOW_CTRL)

    p = {
        "roi_top": roi_top,
        "roi_bot": roi_bot,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "white_min": white_min,
        "black_max": black_max,
        "red_s_min": red_s_min,
        "red_v_min": red_v_min,
        "white_ratio_th": cv2.getTrackbarPos("WHITE_RATIO_x1000", WINDOW_CTRL) / 1000.0,
        "red_ratio_th": cv2.getTrackbarPos("RED_RATIO_x1000", WINDOW_CTRL) / 1000.0,
        "black_ratio_th": cv2.getTrackbarPos("BLACK_RATIO_x1000", WINDOW_CTRL) / 1000.0,
        "empty_white_th": cv2.getTrackbarPos("EMPTY_WHITE_x1000", WINDOW_CTRL) / 1000.0,
        "empty_red_th": cv2.getTrackbarPos("EMPTY_RED_x1000", WINDOW_CTRL) / 1000.0,
        "empty_black_th": cv2.getTrackbarPos("EMPTY_BLACK_x1000", WINDOW_CTRL) / 1000.0,
        "canny1": cv2.getTrackbarPos("CANNY1", WINDOW_CTRL),
        "canny2": cv2.getTrackbarPos("CANNY2", WINDOW_CTRL),
        "hough_th": cv2.getTrackbarPos("HOUGH_TH", WINDOW_CTRL),
        "min_line": cv2.getTrackbarPos("MIN_LINE", WINDOW_CTRL),
        "max_gap": cv2.getTrackbarPos("MAX_GAP", WINDOW_CTRL),
        "blur_odd": cv2.getTrackbarPos("BLUR_ODD", WINDOW_CTRL),
    }

    # safety
    if p["roi_bot"] <= p["roi_top"]:
        p["roi_bot"] = min(1.0, p["roi_top"] + 0.05)

    if p["blur_odd"] < 1:
        p["blur_odd"] = 1
    if p["blur_odd"] % 2 == 0:
        p["blur_odd"] += 1

    return p


# =========================================================
# ROI HELPERS
# =========================================================

def get_three_slot_rois(img, params):
    h, w = img.shape[:2]

    y0 = int(params["roi_top"] * h)
    y1 = int(params["roi_bot"] * h)

    if y1 <= y0:
        return [], None, []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]

    slots = []
    boxes = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(params["pad_x"] * (sx1 - sx0))
        pad_y = int(params["pad_y"] * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

        # back to full-frame coordinates
        fx0 = cx0
        fx1 = cx1
        fy0 = y0 + cy0
        fy1 = y0 + cy1
        boxes.append((fx0, fy0, fx1, fy1))

    return slots, (0, y0, w, y1), boxes


# =========================================================
# OBJECT DETECTION
# =========================================================

def detect_one_object_slot(slot_bgr, params):
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {}, None, None, None

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

    if params["blur_odd"] > 1:
        gray = cv2.GaussianBlur(gray, (params["blur_odd"], params["blur_odd"]), 0)

    # red X
    lower_red1 = np.array([0, params["red_s_min"], params["red_v_min"]], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, params["red_s_min"], params["red_v_min"]], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size

    # white object area
    white_mask = cv2.inRange(gray, params["white_min"], 255)
    white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size

    # black X
    black_mask = cv2.inRange(gray, 0, params["black_max"])
    black_ratio = float(np.count_nonzero(black_mask)) / black_mask.size

    # diagonal lines
    edges = cv2.Canny(gray, params["canny1"], params["canny2"])
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(1, params["hough_th"]),
        minLineLength=max(1, params["min_line"]),
        maxLineGap=max(0, params["max_gap"])
    )

    diag_pos = 0
    diag_neg = 0
    line_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
                cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif -65 <= ang <= -25:
                diag_neg += 1
                cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    has_x_shape = (diag_pos >= 1 and diag_neg >= 1)

    metrics = {
        "red_ratio": round(red_ratio, 4),
        "white_ratio": round(white_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "diag_pos": int(diag_pos),
        "diag_neg": int(diag_neg),
        "has_x_shape": bool(has_x_shape),
    }

    # obstacle = white box + red X
    if white_ratio > params["white_ratio_th"] and red_ratio > params["red_ratio_th"] and has_x_shape:
        return "O", metrics, red_mask, white_mask, line_overlay

    # target = white box + black X
    if white_ratio > params["white_ratio_th"] and black_ratio > params["black_ratio_th"] and has_x_shape:
        return "T", metrics, black_mask, white_mask, line_overlay

    # empty
    if white_ratio < params["empty_white_th"] and red_ratio < params["empty_red_th"] and black_ratio < params["empty_black_th"]:
        return "E", metrics, red_mask, white_mask, line_overlay

    return "?", metrics, red_mask, white_mask, line_overlay


# =========================================================
# DRAW
# =========================================================

def draw_main(frame, band_box, slot_boxes, labels):
    out = frame.copy()

    if band_box is not None:
        x0, y0, x1, y1 = band_box
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)

    for i, (box, lab) in enumerate(zip(slot_boxes, labels)):
        x0, y0, x1, y1 = box
        color = (0, 255, 0) if lab in ("O", "T") else (0, 200, 255) if lab == "?" else (180, 180, 180)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"S{i}:{lab}", (x0 + 5, max(20, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    cv2.putText(out, "q=quit  p=print params", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_slot_views(slots, masks_a, masks_b, overlays, labels, metrics_list):
    panels = []

    for i in range(3):
        if i >= len(slots) or slots[i] is None or slots[i].size == 0:
            blank = np.zeros((180, 240, 3), dtype=np.uint8)
            panels.append(blank)
            continue

        slot = cv2.resize(slots[i], (240, 180))
        ma = cv2.resize(masks_a[i], (240, 180)) if masks_a[i] is not None else np.zeros((180, 240), dtype=np.uint8)
        mb = cv2.resize(masks_b[i], (240, 180)) if masks_b[i] is not None else np.zeros((180, 240), dtype=np.uint8)
        ov = cv2.resize(overlays[i], (240, 180)) if overlays[i] is not None else np.zeros((180, 240, 3), dtype=np.uint8)

        ma_bgr = cv2.cvtColor(ma, cv2.COLOR_GRAY2BGR)
        mb_bgr = cv2.cvtColor(mb, cv2.COLOR_GRAY2BGR)

        top = np.hstack([slot, ma_bgr])
        bot = np.hstack([mb_bgr, ov])
        panel = np.vstack([top, bot])

        txt = f"S{i}:{labels[i]} rr={metrics_list[i]['red_ratio']:.3f} wr={metrics_list[i]['white_ratio']:.3f} br={metrics_list[i]['black_ratio']:.3f}"
        cv2.putText(panel, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        txt2 = f"diag+= {metrics_list[i]['diag_pos']} diag-= {metrics_list[i]['diag_neg']} X={metrics_list[i]['has_x_shape']}"
        cv2.putText(panel, txt2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        panels.append(panel)

    return np.hstack(panels)


# =========================================================
# MAIN
# =========================================================

def main():
    make_trackbars()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, 1400, 700)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("ERROR: Could not grab frame.")
            break

        params = get_params()
        slots, band_box, slot_boxes = get_three_slot_rois(frame, params)

        labels = []
        masks_a = []
        masks_b = []
        overlays = []
        metrics_list = []

        for slot in slots:
            lab, metrics, mask_a, mask_b, overlay = detect_one_object_slot(slot, params)
            labels.append(lab)
            masks_a.append(mask_a)
            masks_b.append(mask_b)
            overlays.append(overlay)
            metrics_list.append(metrics)

        while len(labels) < 3:
            labels.append("?")
            masks_a.append(None)
            masks_b.append(None)
            overlays.append(None)
            metrics_list.append({
                "red_ratio": 0.0,
                "white_ratio": 0.0,
                "black_ratio": 0.0,
                "diag_pos": 0,
                "diag_neg": 0,
                "has_x_shape": False
            })

        view_main = draw_main(frame, band_box, slot_boxes, labels)
        view_slots = stack_slot_views(slots, masks_a, masks_b, overlays, labels, metrics_list)

        combined = np.vstack([
            cv2.resize(view_main, (view_slots.shape[1], 360)),
            view_slots
        ])

        cv2.imshow(WINDOW_MAIN, combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            print("\nCurrent params:")
            for k, v in params.items():
                print(f"{k} = {v}")
            print("slot labels =", labels)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
