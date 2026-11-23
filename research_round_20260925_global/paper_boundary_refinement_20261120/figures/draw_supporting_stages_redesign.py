"""Draw the paper's native point geometry and frozen Phoenix controls at column width."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np


WIDTH = 3.45 * 72
INK = "#111D2B"
NAVY = "#25384B"
TEAL = "#087E8B"
RUST = "#A94E37"
MUTED = "#526370"
GRID = "#C9D3DA"
PAPER = "#F5F8FA"
TEAL_PALE = "#E3F1F1"
WHITE = "#FFFFFF"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "sar-native-supporting-figures-20261123",
})


def frame(height):
    figure = plt.figure(figsize=(WIDTH / 72, height / 72), facecolor=WHITE)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(height, 0))
    ax.set_axis_off()
    return figure, ax


def text(ax, x, y, value, *, size=7.5, color=INK, weight="normal", align="left"):
    ax.text(x, y, value, fontsize=size, color=color, weight=weight, ha=align, va="center", linespacing=1.16)


def rule(ax, x1, y1, x2, y2, color=GRID, width=.55, dash=False):
    ax.plot([x1, x2], [y1, y2], color=color, lw=width, solid_capstyle="butt",
            linestyle=(0, (2, 2)) if dash else "-", zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7.4,
                                 shrinkA=0, shrinkB=0, linewidth=.75, color=color, zorder=5))


def heading(ax, letter, title, y):
    text(ax, 11, y, letter, size=8.3, color=TEAL, weight="bold")
    text(ax, 22, y, title, size=8.3, weight="bold")


def mark(ax, x, y, side, size):
    if side == "+":
        ax.add_patch(Rectangle((x-size/2, y-size/2), size, size, fc=TEAL, ec=WHITE, lw=.34, zorder=4))
    else:
        ax.add_patch(Rectangle((x-size/2, y-size/2), size, size, fc=WHITE, ec=RUST, lw=.9,
                               linestyle=(0, (1.4, .75)), zorder=4))


def native_grid(ax, x, y, step, native, additions=(), removals=()):
    for row, col in np.ndindex(native.shape):
        ax.add_patch(Rectangle((x+col*step, y+row*step), step, step, fc=NAVY if native[row, col] else PAPER,
                               ec=GRID, lw=.30, zorder=1))
    for sign, cells in (("+", additions), ("−", removals)):
        for col, row in cells:
            mark(ax, x+(col+.5)*step, y+(row+.5)*step, sign, step*.68)


def native_occupancy():
    figure, ax = frame(299)
    heading(ax, "a", "Native endpoints", 12)
    native = np.zeros((8, 10), dtype=bool)
    native[2:7, 2:9] = True
    native[1, 5:7] = True
    x, y, cell = 12, 27.5, 8.3
    add = ((3, 1), (4, 0), (6, 0), (8, 1), (9, 3))
    drop = ((3, 2), (5, 1), (7, 2), (8, 3), (8, 5))
    assert all(not native[row, col] for col, row in add)
    assert all(native[row, col] for col, row in drop)
    native_grid(ax, x, y, cell, native, add, drop)
    ax.add_patch(Rectangle((x+cell-.9, y-1.6), 9*cell+2.5, 7*cell+2.8, fill=False, ec=MUTED, lw=.65,
                           linestyle=(0, (3, 2)), zorder=3))
    text(ax, 99, 28, r"$B$", size=7.5, color=MUTED)
    arrow(ax, 13, 101, 29, 101)
    text(ax, 32, 101, r"$x$", size=7.3, color=MUTED)
    arrow(ax, 99, 76, 99, 93)
    text(ax, 105, 94, r"$y$", size=7.3, color=MUTED)
    mark(ax, 124, 38, "+", 6.3)
    text(ax, 133, 38, r"$S_+$ outside $M$")
    mark(ax, 124, 53, "−", 6.3)
    text(ax, 133, 53, r"$S_-$ inside $M$")
    text(ax, 120, 70, r"Two rings within box $B$", size=7.4)
    text(ax, 120, 82, r"Rank by $|p_i-0.5|$, then index", size=7.3)
    text(ax, 120, 94, r"Keep $n=\min(32,n_+,n_-)$ per side", size=7.2)

    rule(ax, 11, 110, WIDTH-11, 110)
    heading(ax, "b", "Point readout and training target", 122)
    # These four vertices are decoder lattice samples. The point inside them is the native center.
    dx, dy, span = 34, 142, 35
    ax.add_patch(Rectangle((dx, dy), span, span, fc=PAPER, ec=GRID, lw=.60))
    center = (dx+span*.61, dy+span*.55)
    for sx, sy in ((dx, dy), (dx+span, dy), (dx, dy+span), (dx+span, dy+span)):
        rule(ax, sx, sy, *center, color=GRID, width=.52, dash=True)
        ax.add_patch(Circle((sx, sy), 1.8, fc=MUTED, ec=WHITE, lw=.35, zorder=4))
    ax.add_patch(Circle(center, 2.5, fc=INK, ec=WHITE, lw=.40, zorder=5))
    text(ax, 76, 158, r"$c_i$", size=7.3, color=MUTED)
    text(ax, 11, 186, r"Bilinear at $c_i=(x+\frac{1}{2},y+\frac{1}{2})$", size=7.2)
    text(ax, 11, 198, "D-FINE probabilities", size=7.2, color=MUTED)
    text(ax, 11, 209, r"RF-DETR logits, then $\sigma$", size=7.2, color=MUTED)

    px, py, edge = 157, 140, 40
    tile = Rectangle((px, py), edge, edge, fc=WHITE, ec=GRID, lw=.65)
    ax.add_patch(tile)
    vertices = [(px-8, py+32), (px+8, py+21), (px+22, py-5),
                (px+47, py+7), (px+46, py+47), (px-2, py+45)]
    fill = Polygon(vertices, closed=True, fc=TEAL_PALE, ec="none", zorder=2)
    ax.add_patch(fill)
    fill.set_clip_path(tile)
    ax.add_patch(Polygon(vertices, closed=True, fill=False, ec=NAVY, lw=.75, zorder=3))
    ax.add_patch(Rectangle((px, py), edge, edge, fill=False, ec=MUTED, lw=.65, zorder=4))
    text(ax, px-10, py+4, r"$C_i$", size=7.5, color=MUTED)
    text(ax, px+52, py+12, r"$G$", size=7.5, color=NAVY)
    text(ax, 127, 191, r"$q_i=|C_i\cap G|/|C_i|$", size=8.2)
    text(ax, 127, 205, "Source polygon only", size=7.2, color=MUTED)

    rule(ax, 11, 219, WIDTH-11, 219)
    heading(ax, "c", "Residual point head", 231)
    for xpos, value in ((29, r"$v_i\ (14)$"), (97, r"$[z_i,z_i]\ (28)$"), (150, "24"), (185, "24"), (220, "1")):
        text(ax, xpos, 248, value, size=7.9, align="center", weight="bold")
    for first, second in ((47, 64), (124, 139), (160, 174), (195, 211)):
        arrow(ax, first, 248, second, 248)
    text(ax, 64, 260, "Standardize and duplicate", size=7.2, align="center", color=MUTED)
    text(ax, 167, 260, "ReLU hidden layers", size=7.2, align="center", color=MUTED)
    text(ax, 12, 276, r"$\widehat q_i=\sigma[\mathrm{logit}(p_i^\epsilon)+f_\theta([z_i,z_i])]$", size=8.1)
    text(ax, 12, 289, "Fractional BCE on source cells · 1,321 parameters", size=7.2, color=MUTED)
    return figure


def phoenix_example():
    native = np.zeros((7, 8), dtype=bool)
    native[1:6, 1:7] = True
    native[0, 3:5] = True
    probabilities = np.where(native, .73, .18)
    narrow_add = ((0, 2), (0, 4), (7, 3), (2, 6))
    narrow_drop = ((1, 2), (1, 4), (6, 3), (2, 5))
    values = dict(zip(narrow_add+narrow_drop, (.88, .76, .72, .42, .20, .36, .47, .67), strict=True))
    values.update({(7, 1): .96, (7, 6): .95, (3, 3): .21})
    for (x, y), p in values.items():
        probabilities[y, x] = p
    all_cells = tuple((x, y) for y, x in np.ndindex(native.shape))
    k = 2

    def ranked(support, write):
        scores = [(float(.5-probabilities[y,x] if native[y,x] else probabilities[y,x]-.5), x, y)
                  for x, y in support]
        scores.sort(key=lambda point: (-point[0], point[2], point[1]))
        if write == "threshold":
            selected = [row for row in scores if row[0] > 0]
        elif write == "count":
            selected = scores[:2*k]
        else:
            selected = [row for row in scores if not native[row[2], row[1]]][:k]
            selected += [row for row in scores if native[row[2], row[1]]][:k]
        return tuple((x,y) for _,x,y in selected if not native[y,x]), tuple((x,y) for _,x,y in selected if native[y,x])

    examples = {("narrow", write): ranked(narrow_add+narrow_drop, write) for write in ("threshold", "count", "balanced")}
    examples.update({("whole", write): ranked(all_cells, write) for write in ("threshold", "count")})
    assert tuple(map(len, examples["narrow", "count"])) == (3, 1)
    assert tuple(map(len, examples["narrow", "balanced"])) == (2, 2)
    return native, probabilities, examples


def phoenix_controls():
    figure, ax = frame(260)
    native, probabilities, examples = phoenix_example()
    text(ax, 11, 12, "Frozen Phoenix XL1", size=8.3, weight="bold")
    shade = LinearSegmentedColormap.from_list("posteriors", (PAPER, "#8295A1", NAVY))
    for row, col in np.ndindex(probabilities.shape):
        ax.add_patch(Rectangle((12+5.5*col, 25+5.5*row), 5.5, 5.5,
                               fc=shade(float(probabilities[row,col])), ec=WHITE, lw=.26))
    text(ax, 66, 32, "Image and native mask", size=7.25)
    arrow(ax, 145, 32, 159, 32)
    text(ax, 163, 32, "Five iterations", size=7.25)
    text(ax, 66, 46, r"One response $r_i$ on the native grid", size=7.65, weight="bold")
    text(ax, 66, 59, "Fixed candidates and detector scores", size=7.25, color=MUTED)

    rule(ax, 11, 73, WIDTH-11, 73, color=NAVY, width=.65)
    text(ax, 11, 87, "Support", size=7.4, weight="bold")
    text(ax, 104, 84, "Threshold", size=7.6, align="center", weight="bold")
    text(ax, 184, 84, "Exact total", size=7.6, align="center", weight="bold")
    text(ax, 104, 95, "Author binary decisions", size=7.2, align="center", color=MUTED)
    text(ax, 184, 95, r"$2k_{\rm att}$, free sign split", size=7.2, align="center", color=MUTED)
    rule(ax, 11, 102, WIDTH-11, 102)
    rule(ax, 11, 151, WIDTH-11, 151)
    for scope, y in (("narrow", 106), ("whole", 156)):
        text(ax, 11, y+10, "Narrow" if scope == "narrow" else "Whole", size=7.5, weight="bold")
        text(ax, 11, y+24, r"$S_+\cup S_-$" if scope == "narrow" else r"$\Omega$", size=7.6, color=MUTED)
        for writer, x in (("threshold", 83), ("count", 163)):
            plus, minus = examples[scope, writer]
            native_grid(ax, x, y, 5.25, native, plus, minus)

    rule(ax, 11, 199, WIDTH-11, 199, color=NAVY, width=.65)
    text(ax, 11, 211, "Narrow support, same total", size=8.1, weight="bold")
    text(ax, 11, 225, "Free allocation", size=7.4)
    text(ax, 11, 237, r"$a$ adds, $2k_{\rm att}-a$ removes", size=7.2, color=MUTED)
    arrow(ax, 112, 225, 128, 225)
    text(ax, 133, 225, "Balanced allocation", size=7.4, weight="bold")
    text(ax, 133, 237, r"$k_{\rm att}$ adds, $k_{\rm att}$ removes", size=7.2, color=MUTED)
    mark(ax, 16, 251, "+", 5.8)
    text(ax, 23, 251, "Added cell", size=7.2, color=MUTED)
    mark(ax, 83, 251, "−", 5.8)
    text(ax, 91, 251, "Removed cell", size=7.2, color=MUTED)
    text(ax, 236, 251, "Schematic", size=7.2, align="right", color=MUTED)
    return figure


def save(figure, output, name):
    stamp = datetime(2026, 11, 23, tzinfo=timezone.utc)
    figure.savefig(output / (name + ".pdf"), metadata={"CreationDate": stamp, "ModDate": stamp})
    figure.savefig(output / (name + ".svg"), metadata={"Date": stamp.isoformat()})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True, help="Directory for the two PDF and SVG pairs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    save(native_occupancy(), args.out, "method_native_occupancy")
    save(phoenix_controls(), args.out, "method_phoenix_controls")


if __name__ == "__main__":
    main()
