import os
import json
import cv2
import numpy as np

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
HEADINGS = ["front", "right", "back", "left"]

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

# Very loose orange detection to make it work first
ORANGE_H_MIN = 0
ORANGE_H_MAX = 35
ORANGE_S_MIN = 60
ORANGE_V_MIN = 60

ORANGE_MIN_AREA_FRAC = 0.015
ORANGE_MAX_AREA_FRAC = 0.80

OBSTACLE_MIN_SCORE = 0.06
EMPTY_SCORE_MAX = 0.03


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

        sq = squareish_score(cnt)
        x, y, ww, hh = cv2.boundingRect(cnt)

        roi_mask = mask[y:y+hh, x:x+ww]
        if roi_mask.size == 0:
            continue

        fill = float(np.mean(roi_mask > 0))

        score = (
            0.50 * area_frac +
            0.25 * fill +
            0.25 * sq
        )

        if score > best_score:
            best_score = score
            best_rect = (x, y, ww, hh)

    return best_score, best_rect, mask


def classify_slot_object(tile):
    orange_score, orange_rect, orange_mask = detect_orange_obstacle(tile)

    if orange_score >= OBSTACLE_MIN_SCORE:
        return "obstacle", orange_score, "X", orange_rect, orange_mask

    if orange_score <= EMPTY_SCORE_MAX:
        return "empty", orange_score, "E", orange_rect, orange_mask

    return "unknown", orange_score, "?", orange_rect, orange_mask


def draw_debug(tile, label, score, orange_rect):
    dbg = tile.copy()

    if orange_rect is not None:
        x, y, w, h = orange_rect
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(dbg, "obs", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

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
            label, score, ch, orange_rect, orange_mask = classify_slot_object(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            dbg = draw_debug(tile, label, score, orange_rect)

            dbg_name = os.path.join(DEBUG_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, dbg)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{heading}_slot{i}_orangemask.jpg"), orange_mask)

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
