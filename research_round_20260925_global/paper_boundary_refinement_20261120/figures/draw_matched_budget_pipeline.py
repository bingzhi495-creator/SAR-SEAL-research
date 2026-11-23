"""Draw the native-grid refinement comparison with editable vector labels."""

from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


NAVY = "#20384D"
TEAL = "#006E70"
RUST = "#9A4025"
INK = "#172A39"
SLATE = "#485A65"
RULE = "#A7B6BE"
PALE_TEAL = "#DCF0EE"
PALE_RUST = "#FAE8DF"
PALE_BLUE = "#EDF2F5"
WHITE = "#FFFFFF"
WIDTH = 7.15 * 72
HEIGHT = 3.35 * 72

# Coordinates are (column, row); the array and drawing both count rows downward.
BASE_MASK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
], dtype=bool)
ADD = ((2, 1), (3, 1), (4, 2), (5, 2))
DROP = ((2, 2), (3, 2), (4, 3), (5, 3))
ADD_SCORE = np.array([.92, .82, .50, .30])
DROP_SCORE = np.array([.70, .33, .10, .20])


def example():
    """Return verified decisions and masks for the worked two-exchange example."""
    costs = np.sum((np.array(ADD)[:, None, :] - np.array(DROP)[None, :, :]) ** 2, axis=2)
    assignments = [(sum(costs[i, j] for i, j in enumerate(p)), p)
                   for p in permutations(range(4))]
    best_cost = min(cost for cost, _ in assignments)
    assert [p for cost, p in assignments if cost == best_cost] == [(0, 1, 2, 3)]
    gains = ADD_SCORE - DROP_SCORE
    limit = 2
    fixed = np.argsort(-gains, kind="stable")[:limit]
    selections = (
        (fixed, fixed),
        (np.argsort(-ADD_SCORE, kind="stable")[:limit],
         np.argsort(DROP_SCORE, kind="stable")[:limit]),
    )
    outputs = []
    for add_ids, drop_ids in selections:
        after = BASE_MASK.copy()
        for i in add_ids:
            x, y = ADD[i]
            assert not BASE_MASK[y, x]
            after[y, x] = True
        for i in drop_ids:
            x, y = DROP[i]
            assert BASE_MASK[y, x]
            after[y, x] = False
        assert (after & ~BASE_MASK).sum() == (~after & BASE_MASK).sum() == limit
        assert after.sum() == BASE_MASK.sum()
        gain = ADD_SCORE[add_ids].sum() - DROP_SCORE[drop_ids].sum()
        outputs.append((set(add_ids), set(drop_ids), after, gain))
    assert outputs[0][:2] == ({1, 2}, {1, 2})
    assert outputs[1][:2] == ({0, 1}, {2, 3})
    np.testing.assert_allclose([row[3] for row in outputs], [.89, 1.44])
    return gains, outputs


def draw():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none",
        "svg.hashsalt": "matched-budget-pipeline",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "axes.unicode_minus": True,
    })
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    def label(x, y, value, *, size=8.2, color=INK, bold=False, align="left", gid=None):
        return ax.text(x, y, value, ha=align, va="center", fontsize=size, color=color,
                       fontweight="bold" if bold else "normal", gid=gid)

    def rect(x, y, w, h, fill=WHITE, edge=RULE, lw=.45, gid=None):
        patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                          linewidth=lw, joinstyle="miter", gid=gid)
        ax.add_patch(patch)
        return patch

    def line(x1, y1, x2, y2, color=RULE, lw=.55, style="-"):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style,
                solid_capstyle="butt", dash_capstyle="butt")

    def arrow(x1, y1, x2, y2, color=NAVY, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     shrinkA=0, shrinkB=0, mutation_scale=7.5,
                                     linewidth=.85, color=color,
                                     linestyle=(0, (3, 2)) if dashed else "-"))

    def header(x, y, letter, title):
        label(x, y, letter, color=TEAL, size=10.6, bold=True)
        label(x + 19, y, title, color=NAVY, size=10.6, bold=True)

    def grid(x0, y0, cell, mask, *, slot_ids=False, additions=(), removals=(), gid):
        add_lookup = {point: i for i, point in enumerate(ADD)}
        drop_lookup = {point: i for i, point in enumerate(DROP)}
        for row, col in np.ndindex(mask.shape):
            position = (col, row)
            add_id = add_lookup.get(position)
            drop_id = drop_lookup.get(position)
            fill = NAVY if mask[row, col] else WHITE
            text, ink = "", WHITE
            if slot_ids and add_id is not None:
                fill, ink, text = PALE_TEAL, TEAL, str(add_id + 1)
            elif slot_ids and drop_id is not None:
                fill, ink, text = PALE_RUST, RUST, str(drop_id + 1)
            elif add_id in additions:
                fill, text = TEAL, "+"
            elif drop_id in removals:
                fill, ink, text = PALE_RUST, RUST, "−"
            left, top = x0 + col * cell, y0 + row * cell
            rect(left, top, cell, cell, fill, "#9AABB4", .35, f"{gid}-cell-{col}-{row}")
            if text:
                label(left + cell / 2, top + cell / 2 + .15, text, color=ink,
                      size=8.0, bold=True, align="center")
        rect(x0, y0, mask.shape[1] * cell, mask.shape[0] * cell,
             "none", SLATE, .6, f"{gid}-frame")

    header(7, 12, "a", "Frozen detection and editable cells")
    header(228, 12, "b", "Predict native-cell occupancy")
    line(220, 3, 220, 101)
    grid(8, 27, 9.6, BASE_MASK, slot_ids=True, gid="native")
    box = rect(8 + .67 * 9.6, 27 + .65 * 9.6, 6.66 * 9.6, 5.16 * 9.6,
               "none", NAVY, .8, "frozen-detection-box")
    box.set_linestyle((0, (3, 2)))
    label(8, 94, "Dashed box boundary", size=8.0, color=SLATE)
    label(94, 31, "Mask, box, logits and score", bold=True)
    label(94, 46, "Sample the two-pixel band")
    label(94, 57, "on each side, inside the box.")
    rect(94, 69, 8, 8, PALE_TEAL, TEAL)
    label(107, 73, "Outside mask: add slots", size=8.0)
    rect(94, 83, 8, 8, PALE_RUST, RUST)
    label(107, 87, "Inside mask: drop slots", size=8.0)

    # Training-only target: a half-covered unit cell and its polygon boundary.
    rect(228, 25, 22, 22)
    ax.add_patch(Polygon([(228, 47), (250, 25), (250, 47)], closed=True,
                         facecolor=PALE_TEAL, edgecolor="none", gid="fractional-target"))
    line(228, 47, 250, 25, TEAL, 1.1)
    rect(228, 25, 22, 22, "none", SLATE, .6)
    label(257, 29, "Source supervision", bold=True, color=TEAL)
    label(257, 42, "q = area(cell ∩ polygon)")
    line(351, 36, 370, 36, TEAL, .85, (0, (3, 2)))
    arrow(370, 36, 370, 56, TEAL, dashed=True)
    label(380, 36, "Training only", color=TEAL)
    rect(228, 57, 66, 27, PALE_BLUE)
    label(261, 65, "Point inputs", bold=True, align="center")
    label(261, 77, "z joined to z", align="center")
    arrow(296, 70.5, 308, 70.5)
    rect(311, 57, 113, 27, PALE_BLUE)
    label(367.5, 65, "28 → 24 → 24 → 1", bold=True, align="center")
    label(367.5, 77, "ReLU hidden layers", align="center")
    arrow(426, 70.5, 437, 70.5)
    label(441, 64.5, "Occupancy", bold=True)
    label(441, 77.5, "q̂ = σ(ℓ + r)")
    label(228, 96, "z: 14 standardized features     ℓ: decoder logit     r: residual", size=8.0)

    line(7, 105, WIDTH - 7, 105, NAVY, .9)
    header(7, 117.5, "c", "Hold the edit count fixed and compare the selected cells")
    line(169, 132, 169, 225)
    line(339, 132, 339, 225)
    label(8, 138, "Minimum-distance pairing", bold=True, size=9.1, color=NAVY)
    label(178, 138, "Fixed pairs", bold=True, size=9.1, color=NAVY)
    label(348, 138, "Independent rankings", bold=True, size=9.1, color=NAVY)

    gains, outputs = example()
    cols = [73, 98, 123, 148]
    rows = [
        (155, "Pair slot", ["1", "2", "3", "4"], INK),
        (171, "Add q̂", [f"{v:.2f}" for v in ADD_SCORE], TEAL),
        (187, "Drop q̂", [f"{v:.2f}" for v in DROP_SCORE], RUST),
        (205, "Pair gain", [f"{v:.2f}" for v in gains], INK),
    ]
    for y, side, values, color in rows:
        label(8, y, side, color=color, bold=True)
        for x, value in zip(cols, values):
            label(x, y, value, color=color, align="center")
    line(8, 196, 160, 196)
    label(8, 220, "Hungarian distance assignment", size=8.0, color=SLATE)

    for left, (added, removed, mask, gain), rule, key in zip(
            (178, 348), outputs,
            ("Take the two largest pair gains.", "Add highest p, drop lowest p."),
            ("fixed", "independent")):
        grid(left, 151, 8.5, mask, additions=added, removals=removed, gid=key)
        def indices(ids):
            return ", ".join(str(i + 1) for i in sorted(ids))
        label(left + 76, 157, "Add   " + indices(added), color=TEAL, bold=True)
        label(left + 76, 171, "Drop  " + indices(removed), color=RUST, bold=True)
        label(left + 76, 187, f"Est. gain  {gain:.2f}", size=8.0)
        label(left, 216, rule, size=8.0)

    line(7, 227, WIDTH - 7, 227)
    label(8, 235, f"At k = 2, each mask adds two cells (+) and removes two (−). Both retain {BASE_MASK.sum()} foreground cells and the detector score.",
          size=8.0, color=SLATE)
    target = Path(__file__).resolve().parent / "matched_budget_pipeline"
    fig.savefig(target.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(target.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None,
                                                      "Title": "Matched-budget boundary refinement"})
    plt.close(fig)


if __name__ == "__main__":
    draw()
