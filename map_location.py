import os

# =========================================================
# CONFIG
# =========================================================

VOTED_COLOR_FILE = "voted_results/voted_color_3x3.txt"

# Your 10x10 color grid
BIG_GRID = [
    ['B','Y','R','G','B','Y','Y','M','P','Y'],
    ['B','R','R','Y','P','B','R','R','M','M'],
    ['Y','G','B','B','G','G','Y','M','M','M'],
    ['R','B','P','G','M','Y','P','B','Y','P'],
    ['R','M','M','G','P','Y','G','Y','R','B'],
    ['P','Y','G','R','B','M','P','G','G','Y'],
    ['G','P','Y','B','R','R','M','Y','B','G'],
    ['Y','B','G','P','Y','G','B','R','M','R'],
    ['M','R','P','Y','G','B','Y','P','G','B'],
    ['B','G','Y','M','R','P','G','B','Y','M'],
]

# minimum known neighbors required to accept a match
MIN_KNOWN_NEIGHBORS = 5

# allow up to this many mismatches among known cells
MAX_MISMATCHES = 1


# =========================================================
# HELPERS
# =========================================================

def read_local_3x3(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Each row must have 3 entries. Bad row: {line}")
            rows.append(parts)

    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in local 3x3 file, found {len(rows)}")

    return rows


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


def rotate_3x3_ccw(mat):
    # 90 deg counterclockwise
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
        [grid[center_r][center_c - 1],     'A',                     grid[center_r][center_c + 1]],
        [grid[center_r + 1][center_c - 1], grid[center_r + 1][center_c], grid[center_r + 1][center_c + 1]],
    ]


def score_match(local_3x3, window_3x3):
    """
    Compare local against candidate window.
    Ignore local '?' and local 'A'.
    """
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
    """
    rotation needed to align local scan with BIG_GRID.

    If local had to be rotated CCW by:
      0   -> facing UP
      90  -> facing RIGHT
      180 -> facing DOWN
      270 -> facing LEFT
    """
    mapping = {
        0: "UP",
        90: "RIGHT",
        180: "DOWN",
        270: "LEFT",
    }
    return mapping[rotation_ccw_deg]


# =========================================================
# MAIN MATCHING
# =========================================================

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

    best = candidates[0]
    return best, candidates


# =========================================================
# MAIN
# =========================================================

def main():
    local_3x3 = read_local_3x3(VOTED_COLOR_FILE)

    print("\nLOCAL VOTED COLOR 3x3:")
    print(pretty_matrix(local_3x3))

    best, candidates = find_best_match(local_3x3, BIG_GRID)

    if best is None:
        print("\nNo valid match found.")
        print("Try:")
        print("- lowering MIN_KNOWN_NEIGHBORS")
        print("- increasing MAX_MISMATCHES")
        print("- improving voting/color detection")
        return

    print("\nBEST MATCH FOUND")
    print("-------------------------")
    print(f"center_row         = {best['center_row']}")
    print(f"center_col         = {best['center_col']}")
    print(f"rotation_ccw_deg   = {best['rotation_ccw_deg']}")
    print(f"facing_in_big_grid = {best['facing']}")
    print(f"known_neighbors    = {best['known']}")
    print(f"matches            = {best['matches']}")
    print(f"mismatches         = {best['mismatches']}")
    print(f"score              = {best['score']}/{best['known']}")

    print("\nROTATED LOCAL 3x3 USED FOR MATCH:")
    print(pretty_matrix(best["rotated_local"]))

    print("\nMATCHED WINDOW IN BIG_GRID:")
    print(pretty_matrix(best["window"]))

    print(f"\nTotal valid candidates: {len(candidates)}")
    if len(candidates) > 1:
        print("\nTop candidates:")
        for i, c in enumerate(candidates[:5], start=1):
            print(
                f"{i}. center=({c['center_row']},{c['center_col']}), "
                f"rot={c['rotation_ccw_deg']} deg, "
                f"facing={c['facing']}, "
                f"score={c['score']}/{c['known']}, "
                f"mismatches={c['mismatches']}"
            )


if __name__ == "__main__":
    main()
