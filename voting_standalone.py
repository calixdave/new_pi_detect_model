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
    ensure_dir(SAVE_DIR)

    color_voted = process_one_type("color", unknown="?")
    object_voted = process_one_type("object", unknown="?")

    color_mat = build_matrix_from_headings(color_voted)
    object_mat = build_matrix_from_headings(object_voted)

    color_mat_path = os.path.join(SAVE_DIR, "voted_color_3x3.txt")
    object_mat_path = os.path.join(SAVE_DIR, "voted_object_3x3.txt")

    with open(color_mat_path, "w") as f:
        f.write(pretty_matrix(color_mat) + "\n")

    with open(object_mat_path, "w") as f:
        f.write(pretty_matrix(object_mat) + "\n")

    print("\n==============================")
    print("FINAL VOTED COLOR 3x3")
    print(pretty_matrix(color_mat))

    print("\nFINAL VOTED OBJECT 3x3")
    print(pretty_matrix(object_mat))
    print("==============================")

    print("\nSaved files:")
    print(" ", color_mat_path)
    print(" ", object_mat_path)


if __name__ == "__main__":
    main()
