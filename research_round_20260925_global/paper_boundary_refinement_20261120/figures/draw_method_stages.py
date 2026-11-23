"""Draw native-grid occupancy, signed selection, and Phoenix control diagrams.

All geometry and scores in these illustrations are schematic. Render the PDF
and SVG versions from the same vector artists using Matplotlib.
"""

from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np


WIDTH = 514.8
INK = "#172A39"
NAVY = "#20384D"
MUTED = "#455966"
TEAL = "#006E70"
RUST = "#9A4025"
RULE = "#A8B9C3"
LINE = "#C6D2D9"
SOFT_BLUE = "#EDF3F6"
SOFT_TEAL = "#DCF0EE"
SOFT_RUST = "#FAE8DF"
WHITE = "#FFFFFF"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "sar-native-method-stages",
})


class Diagram:
    def __init__(self, height):
        self.figure = plt.figure(figsize=(WIDTH / 72, height / 72))
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, WIDTH), ylim=(height, 0))
        self.ax.set_axis_off()

    def text(self, x, y, text, *, size=8.6, color=INK, weight="normal",
             align="left"):
        self.ax.text(x, y, text, fontsize=size, fontweight=weight, color=color,
                     ha=align, va="center", linespacing=1.14)

    def rect(self, x, y, w, h, fill=WHITE, edge=RULE, lw=.52):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                                    edgecolor=edge, linewidth=lw))

    def line(self, x1, y1, x2, y2, *, color=LINE, lw=.55, dash=False):
        self.ax.plot((x1, x2), (y1, y2), color=color, linewidth=lw,
                     linestyle=(0, (2, 2)) if dash else "-",
                     solid_capstyle="butt")

    def arrow(self, x1, y1, x2, y2, *, color=NAVY, dash=False):
        self.ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", color=color, linewidth=.9,
            mutation_scale=7.2, shrinkA=0, shrinkB=0, zorder=5,
            linestyle=(0, (2, 2)) if dash else "-"))

    def circle(self, x, y, radius, *, fill=WHITE, edge=RULE, lw=.55):
        self.ax.add_patch(Circle((x, y), radius, facecolor=fill,
                                 edgecolor=edge, linewidth=lw))

    def polygon(self, vertices, fill, edge, lw=.7):
        self.ax.add_patch(Polygon(vertices, facecolor=fill, edgecolor=edge,
                                  linewidth=lw, closed=True))

    def heading(self, x, y, letter, title):
        self.text(x, y, letter, size=10.2, color=TEAL, weight="bold")
        self.text(x + 15, y, title, size=10.2, color=NAVY, weight="bold")

    def chip(self, x, y, w, h, value, *, fill=SOFT_BLUE,
             color=INK, size=8.6):
        self.rect(x, y, w, h, fill=fill)
        self.text(x + w / 2, y + h / 2, value, size=size, color=color,
                  align="center")

    def lattice(self, x, y, s, mask, added=(), dropped=(), markers=False):
        added, dropped = set(added), set(dropped)
        for row, col in np.ndindex(mask.shape):
            a, b = x + s * col, y + s * row
            fill = NAVY if mask[row, col] else WHITE
            point = (col, row)
            if point in added:
                self.rect(a, b, s, s, TEAL if not markers else SOFT_TEAL,
                          edge=LINE, lw=.32)
                if not markers:
                    self.text(a + s / 2, b + s / 2, "+", size=8, color=WHITE,
                              weight="bold", align="center")
                else:
                    self.circle(a + s / 2, b + s / 2, s * .17, edge=TEAL,
                                fill=TEAL, lw=0)
            elif point in dropped:
                self.rect(a, b, s, s, SOFT_RUST, edge=LINE, lw=.32)
                if not markers:
                    self.text(a + s / 2, b + s / 2, "−", size=8, color=RUST,
                              weight="bold", align="center")
                else:
                    self.rect(a + .34 * s, b + .34 * s, .32 * s, .32 * s,
                              RUST, edge=RUST, lw=0)
            else:
                self.rect(a, b, s, s, fill, edge=LINE, lw=.32)
        self.rect(x, y, mask.shape[1] * s, mask.shape[0] * s, "none",
                  edge=RULE, lw=.55)

    def save(self, name, out):
        stamp = datetime(2026, 11, 23, tzinfo=timezone.utc)
        self.figure.savefig(out / (name + ".pdf"),
                            metadata={"CreationDate": stamp, "ModDate": stamp})
        self.figure.savefig(out / (name + ".svg"),
                            metadata={"Date": stamp.isoformat()})
        plt.close(self.figure)


def draw_native_occupancy(out):
    d = Diagram(161)
    d.heading(8, 11, "a", "Decode on native grid")
    d.heading(163, 11, "b", "Sample signed endpoints")
    d.heading(325, 11, "c", "Fit cell occupancy")
    d.line(153, 2, 153, 154)
    d.line(315, 2, 315, 154)

    d.text(8, 31, "D-FINE", size=8.9, weight="bold")
    d.chip(8, 40, 135, 24, r"Probability plane $\rightarrow$ bilinear")
    d.text(8, 77, "RF-DETR", size=8.9, weight="bold")
    d.chip(8, 86, 135, 24, r"Logit plane $\rightarrow$ bilinear $\rightarrow \sigma$")
    d.text(8, 127, r"Sample at native cell centers $\to p_i$", size=8.3)
    d.text(8, 143, r"Original mask $M$ fixes edit sign", size=8.3)

    mask = np.zeros((7, 8), dtype=bool)
    mask[2:7, 1:8] = True
    mask[1, 5:7] = True
    d.lattice(163, 31, 10.5, mask, added=((2, 1), (3, 1), (5, 0)),
              dropped=((2, 2), (3, 2), (5, 1)), markers=True)
    d.circle(261, 48, 2, fill=TEAL, edge=TEAL, lw=0)
    d.text(268, 48, r"$S_+$ outside", size=8.4, color=TEAL)
    d.rect(259, 62, 4, 4, RUST, edge=RUST, lw=0)
    d.text(268, 64, r"$S_-$ inside", size=8.4, color=RUST)
    d.text(163, 111, r"Two-cell bands intersect box $B$", size=8.4)
    d.text(163, 127, r"Rank by $|p_i-0.5|$ on each side", size=8.4)
    d.text(163, 143, r"Keep $n=\min(32,n_+,n_-)$ per side", size=8.4)

    d.chip(325, 30, 179, 20, r"14 input features $\rightarrow$ standardized $z_i$")
    d.arrow(387, 51, 387, 58)
    d.chip(325, 60, 124, 24, r"$[z_i,z_i]\ \to\ 24\ \to\ 24\ \to\ 1$",
           fill=SOFT_TEAL, size=8.8)
    d.text(455, 72, "1,321 params\nReLU hidden", size=7.6, color=MUTED)
    d.arrow(387, 85, 387, 92)
    d.chip(325, 94, 179, 22,
           r"$\widehat q_i=\sigma[\operatorname{logit}(\widetilde p_i)+f_\theta([z_i,z_i])]$",
           fill=SOFT_TEAL, size=8.8)

    # The green fill is the measured polygon–cell intersection, outlined below.
    d.rect(326, 128, 22, 21, WHITE, edge=RULE)
    d.polygon([(326, 143), (333, 139), (339, 128), (348, 128),
               (348, 149), (326, 149)], SOFT_TEAL, TEAL)
    d.arrow(337, 127, 337, 117, dash=True, color=TEAL)
    d.text(354, 134, r"Source polygon $G$ supplies $q_i=|C_i\cap G|/|C_i|$",
           size=7.6)
    d.text(354, 146, "Fractional BCE on sampled cells", size=7.8, color=MUTED)
    d.save("method_native_occupancy", out)


def check_selection(add, drop, add_score, drop_score):
    costs = np.sum((add[:, None, :] - drop[None, :, :]) ** 2, axis=2)
    trials = [(sum(costs[i, j] for i, j in enumerate(order)), order)
              for order in permutations(range(len(add)))]
    best = min(cost for cost, _ in trials)
    assert [order for cost, order in trials if cost == best] == [(0, 1, 2, 3)]
    gains = add_score - drop_score
    coupled = np.argsort(-gains, kind="stable")[:2]
    independent = (np.argsort(-add_score, kind="stable")[:2],
                   np.argsort(drop_score, kind="stable")[:2])
    assert set(coupled) == {1, 2}
    assert [set(ids) for ids in independent] == [{0, 1}, {2, 3}]
    return coupled, independent


def draw_signed_selection(out):
    d = Diagram(155)
    d.heading(8, 11, "a", "Construct distance pairs")
    d.heading(178, 11, "b", "Choose pairs jointly")
    d.heading(348, 11, "c", "Choose each side separately")
    d.line(169, 2, 169, 150)
    d.line(339, 2, 339, 150)

    mask = np.zeros((7, 11), dtype=bool)
    mask[2:7, 1:10] = True
    add = np.array([(2, 1), (4, 1), (6, 1), (8, 1)])
    drop = np.array([(2, 2), (4, 2), (6, 2), (8, 2)])
    coupled, (plus, minus) = check_selection(
        add, drop, np.array([.92, .82, .50, .30]), np.array([.70, .33, .10, .20]))
    edited = ((coupled, coupled), (plus, minus))
    add_cells = tuple(map(tuple, add))
    drop_cells = tuple(map(tuple, drop))
    starts = (9, 179, 349)
    top, cell = 29, 11.4
    d.lattice(starts[0], top, cell, mask, add_cells, drop_cells, markers=True)
    for i, ((xa, ya), (xd, yd)) in enumerate(zip(add, drop, strict=True), 1):
        px = starts[0] + cell * (xa + .5)
        py = top + cell * (ya + .5)
        qx = starts[0] + cell * (xd + .5)
        qy = top + cell * (yd + .5)
        d.line(px, py + 2.1, qx, qy - 2.1, color=MUTED, lw=.8, dash=True)
        d.text(px, top + .5 * cell, str(i), size=7.9,
               color=MUTED, align="center")
    for x, (add_ids, drop_ids) in zip(starts[1:], edited, strict=True):
        cells_plus = tuple(add_cells[i] for i in add_ids)
        cells_minus = tuple(drop_cells[i] for i in drop_ids)
        after = mask.copy()
        for col, row in cells_plus:
            assert not after[row, col]
            after[row, col] = True
        for col, row in cells_minus:
            assert after[row, col]
            after[row, col] = False
        assert int(after.sum()) == int(mask.sum())
        d.lattice(x, top, cell, mask, cells_plus, cells_minus)

    d.text(9, 123, r"Minimize $\sum_j \|a_j-d_j\|_2^2$", size=8.6)
    d.text(9, 140, "One-to-one assignment", size=8.3, color=MUTED)
    d.text(179, 123, r"Rank positive $\widehat q_{a_j}-\widehat q_{d_j}$", size=8.6)
    d.text(179, 140, r"Realized $k=2$ in this example", size=8.3, color=MUTED)
    d.text(349, 123, r"Highest $k$ additions, lowest $k$ removals", size=8.4)
    d.text(349, 140, "Same counts, different locations", size=8.3, color=MUTED)
    d.save("method_signed_selection", out)


def draw_phoenix_controls(out):
    d = Diagram(150)
    d.heading(8, 11, "a", "Frozen refiner")
    d.heading(118, 11, "b", "Cross support and write rule")
    d.heading(376, 11, "c", "Fix the sign split")
    d.line(110, 2, 110, 145)
    d.line(368, 2, 368, 145)

    d.chip(8, 29, 93, 20, r"Image and mask $M$")
    d.arrow(54.5, 50, 54.5, 58)
    d.chip(8, 59, 93, 22, "Phoenix XL1", fill=SOFT_TEAL)
    d.text(54.5, 89, "Five iterations", size=8.0, align="center", color=MUTED)
    d.arrow(54.5, 94, 54.5, 102)
    d.chip(8, 103, 93, 21, r"Native probability $r_i$")
    d.text(54.5, 137, "Same field in every arm", size=7.8, align="center", color=MUTED)

    d.text(235, 36, "Author threshold", size=8.6, weight="bold", align="center")
    d.text(319, 36, "Exact total count", size=8.6, weight="bold", align="center")
    d.text(118, 58, "Endpoint\nsupport", size=8.6, weight="bold")
    d.text(118, 87, "Whole\nnative grid", size=8.6, weight="bold")
    d.chip(192, 46, 83, 26, "Narrow threshold", size=8.1)
    d.chip(278, 46, 82, 26, "Narrow count", size=8.1, fill=SOFT_TEAL)
    d.chip(192, 74, 83, 26, "Whole threshold", size=8.1)
    d.chip(278, 74, 82, 26, "Whole count", size=8.1, fill=SOFT_TEAL)
    d.text(118, 117, "Threshold writes all author-binary disagreements in support.",
           size=7.7, color=MUTED)
    d.text(118, 134, r"Count writes exactly $2k_{\rm att}$ flips with a free sign split.",
           size=7.7, color=MUTED)

    d.text(376, 33, r"Hold $S_+\cup S_-$ and $2k_{\rm att}$ fixed", size=8.5)
    d.chip(376, 46, 129, 24, "Free addition/removal split")
    d.text(440.5, 79, "versus", size=7.7, color=MUTED, align="center")
    d.chip(376, 89, 129, 24, r"$k_{\rm att}$ adds and $k_{\rm att}$ removes",
           fill=SOFT_TEAL, size=8.5)
    d.text(376, 129, r"Rank by $r_i-0.5$ (add)", size=8.2)
    d.text(376, 141, r"and $0.5-r_i$ (remove).", size=8.2)
    d.save("method_phoenix_controls", out)


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    draw_native_occupancy(output)
    draw_signed_selection(output)
    draw_phoenix_controls(output)
