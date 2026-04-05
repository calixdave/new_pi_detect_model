import os
import json
import cv2
import numpy as np

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
HEADINGS = ["front", "right", "back", "left"]

# Same slot logic style as your color script
ROI_TOP_FRAC = 0.55
ROI_BOT_FRAC = 0.95

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}

# -------------------------------------------------
# ORANGE obstacle thresholds (HSV)
# Tune later if needed
# -------------------------------------------------
ORANGE_H_MIN = 3
ORANGE_H_MAX = 30
ORANGE_S_MIN = 90
ORANGE_V_MIN = 90

ORANGE_MIN_AREA_FRAC = 0.02
ORANGE_MAX_AREA_FRAC = 0.75
ORANGE_MIN_FILL = 0.35
ORANGE_MIN_SCORE = 0.10

# -------------------------------------------------
# WHITE target with black border thresholds
# -------------------------------------------------
WHITE_MIN_V = 170
WHITE_MAX_S = 85
WHITE_MIN_AREA_FRAC = 0.03
WHITE_MAX_AREA_FRAC = 0.75
WHITE_MIN_FILL = 0.40
TARGET_MIN_SCORE = 0.20

# -------------------------------------------------
# Final decision
# -------------------------------------------------
EMPTY_SCORE_MAX = 0.10


def get_three_slot_rois(img):
    h, w = img.shape[:2]

    y0 = int(ROI_TOP_FRAC * h)
    y1 = int(ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]

    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def pretty_print_matrix(mat):
    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(mat.get((col, row), "?"))
        print(" ".join(vals))


def contour_fill_ratio(cnt):
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = float(w * h)
    if rect_area <= 0:
        return 0.0
    return float(area) / rect_area


def squareish_score(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    if h <= 0:
        return 0.0
    aspect = float(w) / float(h)
    return max(0.0, 1.0 - abs(aspect - 1.0))


def detect_orange_obstacle(tile):
    h, w = tile.shape[:2]
    tile_area = float(h * w)

    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask = (
        (H >= ORANGE_H_MIN) &
        (H <= ORANGE_H_MAX) &
        (S >= ORANGE_S_MIN) &
        (V >= ORANGE_V_MIN)
    ).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_rect = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / tile_area
        if area_frac < ORANGE_MIN_AREA_FRAC or area_frac > ORANGE_MAX_AREA_FRAC:
            continue

        fill = contour_fill_ratio(cnt)
        if fill < ORANGE_MIN_FILL:
            continue

        sq = squareish_score(cnt)
        x, y, ww, hh = cv2.boundingRect(cnt)

        roi_hsv = hsv[y:y+hh, x:x+ww]
        if roi_hsv.size == 0:
            continue

        mean_sat = float(np.mean(roi_hsv[:, :, 1])) / 255.0
        mean_val = float(np.mean(roi_hsv[:, :, 2])) / 255.0

        score = (
            0.35 * area_frac +
            0.25 * fill +
            0.20 * sq +
            0.10 * mean_sat +
            0.10 * mean_val
        )

        if score > best_score:
            best_score = score
            best_rect = (x, y, ww, hh)

    return best_score, best_rect, mask


def detect_white_target(tile):
    h, w = tile.shape[:2]
    tile_area = float(h * w)

    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_mask = ((V >= WHITE_MIN_V) & (S <= WHITE_MAX_S)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    best_score = 0.0
    best_rect = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / tile_area
        if area_frac < WHITE_MIN_AREA_FRAC or area_frac > WHITE_MAX_AREA_FRAC:
            continue

        fill = contour_fill_ratio(cnt)
        if fill < WHITE_MIN_FILL:
            continue

        sq = squareish_score(cnt)
        x, y, ww, hh = cv2.boundingRect(cnt)

        if ww < 12 or hh < 12:
            continue

        border = max(2, int(min(ww, hh) * 0.12))
        x0, y0, x1, y1 = x, y, x + ww, y + hh

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)

        roi_gray = gray[y0:y1, x0:x1]
        roi_hsv = hsv[y0:y1, x0:x1]

        if roi_gray.size == 0:
            continue

        rh, rw = roi_gray.shape[:2]
        if rh <= 2 * border or rw <= 2 * border:
            continue

        inner_hsv = roi_hsv[border:rh-border, border:rw-border]
        inner_white_ratio = float(
            np.mean((inner_hsv[:, :, 2] >= WHITE_MIN_V) & (inner_hsv[:, :, 1] <= WHITE_MAX_S))
        )

        ring_mask = np.ones((rh, rw), dtype=np.uint8)
        ring_mask[border:rh-border, border:rw-border] = 0
        border_pixels = roi_gray[ring_mask == 1]
        if border_pixels.size == 0:
            continue

        border_dark_ratio = float(np.mean(border_pixels < 110))

        score = (
            0.30 * area_frac +
            0.20 * fill +
            0.15 * sq +
            0.20 * inner_white_ratio +
            0.15 * border_dark_ratio
        )

        if score > best_score:
            best_score = score
            best_rect = (x, y, ww, hh)

    return best_score, best_rect, white_mask


def classify_slot_object(tile):
    orange_score, orange_rect, orange_mask = detect_orange_obstacle(tile)
    target_score, target_rect, white_mask = detect_white_target(tile)

    if orange_score >= ORANGE_MIN_SCORE:
        return "obstacle", orange_score, "X", target_rect, orange_rect, orange_mask, white_mask

    if orange_score <= EMPTY_SCORE_MAX:
        return "empty", orange_score, "E", target_rect, orange_rect, orange_mask, white_mask

    return "unknown", orange_score, "?", target_rect, orange_rect, orange_mask, white_mask

def draw_debug(tile, label, score, target_rect, orange_rect):
    dbg = tile.copy()

    if orange_rect is not None:
        x, y, w, h = orange_rect
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(dbg, "obs", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    if target_rect is not None:
        x, y, w, h = target_rect
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(dbg, "target", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(dbg, f"{label} {score:.3f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return dbg


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

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

    detailed = {}

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            print(f"ERROR: Missing image: {path}")
            return

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        slots = get_three_slot_rois(img)
        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading}")
            return

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            label, score, ch, target_rect, orange_rect, orange_mask, white_mask = classify_slot_object(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            dbg = draw_debug(tile, label, score, target_rect, orange_rect)

            dbg_name = os.path.join(DEBUG_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, dbg)

            cv2.imwrite(os.path.join(DEBUG_DIR, f"{heading}_slot{i}_orangemask.jpg"), orange_mask)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{heading}_slot{i}_whitemask.jpg"), white_mask)

            print(f"  slot {i}: label={label}, score={score:.4f}, char={ch}, saved={dbg_name}")

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "label": label,
                "score": round(float(score), 4),
                "char": ch,
                "debug_crop": dbg_name
            })

        detailed[heading] = heading_info

    print("\nFinal 3x3 object matrix:")
    pretty_print_matrix(final_grid)

    out = {
        "center": [0, 0],
        "agent": "A",
        "grid_objects": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    with open("object_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved: object_results.json")
    print(f"Saved debug images in: {DEBUG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
