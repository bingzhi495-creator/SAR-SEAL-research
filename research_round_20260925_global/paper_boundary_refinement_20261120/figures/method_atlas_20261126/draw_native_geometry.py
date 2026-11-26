"""Draw two exact geometric explanations used in the NORA manuscript."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

INK = "#382B26"
ORANGE = "#AB5B38"
GREEN = "#356B5D"
WINE = "#8F4958"
SAND = "#F8F3EB"
RULE = "#D7CBBE"
SHARED = "#C9DBCC"
UNION = "#E9DFD1"
BACKGROUND = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans", "axes.autolimit_mode": "data",
})


def canvas(width, height):
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor=BACKGROUND)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    return fig, ax


def rect(ax, x, y, w, h, fill, stroke="none", lw=.55, **kwargs):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=stroke, linewidth=lw, **kwargs))


def label(ax, x, y, text, size=8, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, text, va="center", ha=ha, color=color, fontsize=size, weight=weight)


def line(ax, x0, y0, x1, y1, color=RULE, lw=.65, **kwargs):
    ax.plot((x0, x1), (y0, y1), color=color, lw=lw, **kwargs)


def span(ax, x0, x1, y, color=ORANGE):
    line(ax, x0, y, x1, y, color, .75)
    for x in (x0, x1):
        line(ax, x, y - 2.8, x, y + 2.8, color, .75)


def arrow(ax, a, b, color=ORANGE):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                 color=color, linewidth=.8, shrinkA=0, shrinkB=0))


def legend_line(ax, x, y, value, color, dashed=False):
    line(ax, x, y, x + 16, y, color, 1.0, ls=(0, (3, 2)) if dashed else "-")
    label(ax, x + 20, y, value, 7.5)


def save(fig, out, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1:
            raise ValueError(f"{stem}: label outside page: {t.get_text()}")
    plt.rcParams["svg.hashsalt"] = stem
    fig.savefig(out / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{stem}.png", dpi=170)
    plt.close(fig)


def one_pixel(out):
    fig, ax = canvas(248, 136)
    label(ax, 10, 12, "One-cell displacement", 10.1, weight="semibold")
    x, y, step = 14, 43, 12
    cols, width, offset = 11, 4, 1
    overlap = cols * (width - offset)
    union = cols * (width + offset)
    if overlap / union != .60:
        raise ValueError("Incorrect geometric overlap")
    for row in range(width + offset):
        fill = SAND if row in (0, 4) else SHARED
        for col in range(cols):
            rect(ax, x + col*step, y + row*step, step, step, fill, RULE, .46)
    rect(ax, x, y, cols*step, width*step, "none", ORANGE, 1.1, linestyle=(0, (3, 2)), zorder=3)
    rect(ax, x, y + offset*step, cols*step, width*step, "none", GREEN, 1.08, zorder=3)
    arrow(ax, (x + 26, y), (x + 26, y + step))
    label(ax, x + 35, 31, "1 px", 8.3, ORANGE, "semibold")
    line(ax, x+cols*step+3, y, x+cols*step+3, y+5*step, ORANGE, .65)
    for row in range(6):
        line(ax, x+cols*step+1, y+row*step, x+cols*step+5, y+row*step, ORANGE, .66)

    label(ax, 164, 48, "Across the hull", 7.7, weight="semibold")
    for col in range(5):
        rect(ax, 165+col*13, 57, 13, 13, SHARED if 0<col<4 else SAND, RULE, .55)
    span(ax, 178, 217, 76, GREEN)
    label(ax, 197.5, 85, "3 shared", 7.7, GREEN, ha="center")
    span(ax, 165, 230, 93)
    label(ax, 197.5, 102, "5 in union", 7.6, ORANGE, ha="center")
    legend_line(ax, 14, 121, "Reference", ORANGE, True)
    legend_line(ax, 82, 121, "Shifted mask", GREEN)
    label(ax, 161, 122, r"$\mathrm{IoU}=3/5=0.60$", 9.2, weight="semibold")
    save(fig, out, "nora_one_pixel_motivation")


def overlap_accounting(out):
    fig, ax = canvas(510, 167)
    reference = {(x,y) for x in range(1,5) for y in range(1,4)}
    original = {(x,y) for x in range(2,6) for y in range(1,4)}
    two_edits = (
        ("Native", set(), set()),
        ("Balanced", {(1,2)}, {(5,2)}),
        ("Variable", {(1,2),(5,4)}, set()),
    )
    expected = [(9, 15), (10, 14), (10, 16)]
    positions = (13, 178, 343)
    label(ax, 13, 12, "Intersection and union on the native grid", 10.8, weight="semibold")
    for x, (name, added, removed), goal in zip(positions, two_edits, expected, strict=True):
        if not added.isdisjoint(original) or not removed.issubset(original):
            raise ValueError("Invalid signed edit")
        edited = (original - removed) | added
        shared, total = len(edited & reference), len(edited | reference)
        if (shared,total) != goal or len(added)+len(removed) not in (0,2):
            raise ValueError("Inconsistent union schematic")
        label(ax, x, 30, name, 9.2, weight="semibold")
        xg, yg, step = x+4, 43, 13
        for row in range(5):
            for col in range(7):
                coord = (col,row)
                fill = (SHARED if coord in edited and coord in reference else
                        UNION if coord in edited else BACKGROUND)
                rect(ax, xg+col*step, yg+row*step, step, step, fill, RULE, .47)
                if coord in added or coord in removed:
                    color = GREEN if coord in added else WINE
                    if coord in removed:
                        rect(ax, xg+col*step+1.2, yg+row*step+1.2, step-2.4, step-2.4,
                             "none", color, .93, zorder=4)
                    label(ax, xg+(col+.5)*step, yg+(row+.5)*step,
                          "+" if coord in added else "−", 10, color, "bold", "center").set_zorder(5)
        rect(ax, xg+step, yg+step, step*4, step*3, "none", ORANGE, 1.05, linestyle=(0, (3, 2)), zorder=3)
        d, gain = len(added)-len(removed), shared-len(original&reference)
        if name != "Native":
            label(ax, x+101, 65, rf"$g={gain:+d}$", 8.3, GREEN)
            label(ax, x+101, 84, rf"$d={d:+d}$" if d else r"$d=0$", 8.3, WINE if d else INK)
        bar_x, bar_y, bw = x+4, 119, 5.95
        for i in range(total):
            rect(ax, bar_x+i*bw, bar_y, bw-.65, 7.8, GREEN if i < shared else UNION)
        label(ax, x+106, 123.3, rf"$\frac{{{shared}}}{{{total}}}$", 11, weight="semibold")
    line(ax, 13, 137, 495, 137, RULE, .6)
    legend_line(ax, 13, 151, "Reference", ORANGE, True)
    rect(ax, 104, 147, 9, 7.3, GREEN)
    label(ax, 118, 151, "Intersection", 7.6)
    rect(ax, 194, 147, 9, 7.3, UNION)
    label(ax, 208, 151, "Union remainder", 7.6)
    label(ax, 383, 151, r"$\frac{I+g}{U+d-g}$", 10.5, color=INK)
    save(fig, out, "nora_overlap_accounting")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    dest = parser.parse_args().output
    dest.mkdir(parents=True, exist_ok=True)
    one_pixel(dest)
    overlap_accounting(dest)
