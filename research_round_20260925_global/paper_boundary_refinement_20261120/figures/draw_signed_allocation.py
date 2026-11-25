"""Draw the equal-total allocation example. All occupancies are illustrative."""

import argparse
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


STYLE = runpy.run_path(str(Path(__file__).with_name("draw_core_method_redesign.py")))
frame, heading, rule, label, grid = (STYLE[key] for key in ("frame", "heading", "rule", "label", "grid"))
NAVY, INK, MUTED, TEAL, TEAL_PALE, GRID = (STYLE[key] for key in
                                           ("NAVY", "INK", "MUTED", "TEAL", "TEAL_PALE", "GRID"))
ADD = ((2, 1), (5, 1), (6, 2), (0, 4))
DROP = ((1, 2), (2, 2), (5, 3), (4, 4))
Q_ADD = np.array([[.96, .89, .73, .14], [.93, .85, .69, .12]])
Q_DROP = np.array([[.06, .55, .74, .85], [.10, .57, .76, .81]])
FLIPS = 4
MASK = np.pad(np.ones((3, 5), dtype=bool), ((2, 1), (1, 1)))


def estimate(add, drop):
    unchanged = int(MASK.sum()) - len(DROP)
    truth = unchanged + Q_ADD.sum(axis=1) + Q_DROP.sum(axis=1)
    overlap = unchanged + Q_DROP.sum(axis=1) + Q_ADD[:, add].sum(axis=1) - Q_DROP[:, drop].sum(axis=1)
    area = int(MASK.sum()) + len(add) - len(drop)
    return overlap / (area + truth - overlap)


def allocations():
    incumbent = ([1, 3], [1, 2])
    available = []
    for a in range(FLIPS + 1):
        d = FLIPS - a
        for ratio in np.linspace(0, 1, 11):
            weight = np.array([ratio, 1 - ratio])
            add = np.argsort(-(weight @ Q_ADD), kind="stable")[:a].tolist()
            drop = np.argsort(weight @ Q_DROP, kind="stable")[:d].tolist()
            available.append((float(np.min(estimate(add, drop) - estimate(*incumbent))), add, drop))
    fixed = max((row for row in available if len(row[1]) == FLIPS // 2), key=lambda row: row[0])
    free = max(available, key=lambda row: row[0])
    assert tuple((len(a), len(d)) for _, a, d in (fixed, free)) == ((2, 2), (3, 1))
    assert all(not MASK[y, x] for x, y in ADD) and all(MASK[y, x] for x, y in DROP)
    return fixed, free


def show_mask(ax, x, y, name, action):
    _, plus, minus = action
    add = [ADD[index] for index in plus]
    drop = [DROP[index] for index in minus]
    new = MASK.copy()
    for col, row in add:
        new[row, col] = True
    for col, row in drop:
        new[row, col] = False
    assert np.count_nonzero(MASK != new) == FLIPS and new.sum() == MASK.sum() + len(add) - len(drop)
    label(ax, x + 26, y - 8, name, align="center", size=7.5, weight="bold")
    grid(ax, x, y, 7.45, new, add, drop, edits=True)
    label(ax, x + 26, y + 55, f"+{len(add)} and −{len(drop)}", align="center", size=7.4)
    label(ax, x + 26, y + 66, f"Area {new.sum()} cells", align="center", size=7.0, color=MUTED)


def draw(output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "SARship-signed-quota-example-v1"})
    fixed, free = allocations()
    fig, ax = frame(143)
    for x, letter, title in ((10, "a", "Freeze the candidate"), (151, "b", "Compare feasible splits"),
                              (338, "c", "Apply four flips")):
        heading(ax, x, 13, letter, title)
    rule(ax, 10, 24, 505, 24, color=NAVY)
    rule(ax, 138, 31, 138, 128)
    rule(ax, 328, 31, 328, 128)

    grid(ax, 40, 39, 8.7, MASK, ADD, DROP)
    label(ax, 70, 103, "Native mask · 15 cells", align="center", size=7.5)
    label(ax, 70, 116, "Four endpoints per side", align="center", size=7.2, color=MUTED)

    label(ax, 153, 37, r"Keep the total $a+d=2k_s=4$", size=8)
    for a in range(FLIPS + 1):
        x = 154 + 32 * a
        is_free = a == len(free[1])
        color = TEAL_PALE if is_free else "white"
        edge = TEAL if is_free else NAVY if a == len(fixed[1]) else GRID
        ax.add_patch(Rectangle((x, 51), 26, 16, facecolor=color, edgecolor=edge, linewidth=.65))
        label(ax, x + 13, 59, f"{a} / {FLIPS-a}", size=7.3, align="center",
              color=TEAL if is_free else INK, weight="bold" if is_free else "normal")
    label(ax, 153, 78, "Additions / removals", size=7.1, color=MUTED)
    label(ax, 153, 96, r"Maximize $min_r$ estimated IoU gain", size=7.8)
    label(ax, 153, 110, "Same ranking rule within each split", size=7.2, color=MUTED)

    show_mask(ax, 353, 47, "Equal split", fixed)
    show_mask(ax, 435, 47, "Variable split", free)
    rule(ax, 10, 128, 505, 128)
    label(ax, 10, 136, "Illustrative occupancies. Detection, score and writable endpoints are shared.",
          size=7.0, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in ax.texts:
        if not all(fig.bbox.contains(x, y) for x, y in text.get_window_extent(renderer).get_points()):
            raise ValueError("Text extends outside the artwork")
    for suffix in (".svg", ".pdf"):
        metadata = {"Date": None} if suffix == ".svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(output / ("method_signed_allocation" + suffix), metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    draw(parser.parse_args().output)
