# vote_results.py

import os
from collections import Counter


RESULTS_DIR = "results"
SAVE_DIR = "voted_results"

HEADINGS = ["front", "right", "back", "left"]


# --------------------------------------------------
# helpers
# --------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_lines(path):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                print(f"[WARN] Bad row in {path}: {line}")
                continue

            rows.append(parts)

    return rows


def vote_one_slot(values, unknown="?"):
    valid = [v for v in values if v != unknown]
    if not valid:
        return unknown

    counts = Counter(valid)
    best, _ = counts.most_common(1)[0]
    return best


def vote_rows(rows, unknown="?"):
    """
    rows example:
        [
          ['B','G','R'],
          ['B','G','?'],
          ['B','P','R']
        ]
    returns:
        ['B','G','R']
    """
    if not rows:
        return [unknown, unknown, unknown]

    voted = []
    for col in range(3):
        col_vals = [row[col] for row in rows if len(row) == 3]
        voted.append(vote_one_slot(col_vals, unknown=unknown))

    return voted


def save_voted_line(path, voted):
    with open(path, "w") as f:
        f.write(" ".join(voted) + "\n")


def empty_3x3(fill="?"):
    return [
        [fill, fill, fill],
        [fill, "A",  fill],
        [fill, fill, fill],
    ]


def coord_to_index(dx, dy):
    # x: -1,0,+1 ; y:+1,0,-1
    row = 1 - dy
    col = dx + 1
    return row, col


HEADING_TO_NEIGHBORS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


def place_heading_into_matrix(mat, heading, voted_slots):
    coords = HEADING_TO_NEIGHBORS[heading]
    for i, (dx, dy) in enumerate(coords):
        r, c = coord_to_index(dx, dy)
        mat[r][c] = voted_slots[i]


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


# --------------------------------------------------
# voting
# --------------------------------------------------

def process_one_type(prefix, unknown="?"):
    """
    prefix = 'color' or 'object'
    reads:
        results/color_front.txt, ...
    saves:
        voted_results/voted_color_front.txt, ...
    """
    voted_by_heading = {}

    for heading in HEADINGS:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{heading}.txt")
        rows = read_lines(path)

        print(f"\n[{prefix.upper()}] {heading}")
        if not rows:
            print("  no rows found")
            voted = [unknown, unknown, unknown]
        else:
            print("  raw rows:")
            for row in rows:
                print("   ", row)

            voted = vote_rows(rows, unknown=unknown)

        print("  voted:", voted)
        voted_by_heading[heading] = voted

        save_path = os.path.join(SAVE_DIR, f"voted_{prefix}_{heading}.txt")
        save_voted_line(save_path, voted)

    return voted_by_heading


def build_matrix_from_headings(voted_by_heading):
    mat = empty_3x3("?")
    for heading in HEADINGS:
        place_heading_into_matrix(mat, heading, voted_by_heading[heading])
    return mat


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
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

    os.makedirs(SAVE_DIR, exist_ok=True)

    color_save_path = os.path.join(SAVE_DIR, "voted_color_3x3.txt")
    object_save_path = os.path.join(SAVE_DIR, "voted_object_3x3.txt")

    with open(color_save_path, "w") as f:
        f.write(pretty_matrix(final_color) + "\n")

    with open(object_save_path, "w") as f:
        f.write(pretty_matrix(final_object) + "\n")

    print("\nSaved:")
    print(" ", color_save_path)
    print(" ", object_save_path)
    main()
