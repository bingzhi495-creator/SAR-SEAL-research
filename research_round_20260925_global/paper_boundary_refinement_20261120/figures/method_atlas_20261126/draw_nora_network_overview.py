"""Rebuild the editable network overview and optional verified author-crop variant.

python draw_nora_network_overview.py --output D:/.../overview
python draw_nora_network_overview.py --output D:/.../overview --author-scene D:/.../nora_scene_to_mask.pdf
"""
import argparse
import hashlib
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.image as mimage
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from io import BytesIO


INK = "#382B26"
ORANGE = "#AB5B38"
GREEN = "#356B5D"
WINE = "#8F4958"
PAPER = "#F8F3EB"
RULE = "#D7CBBE"
GRAY = "#74665D"
LIGHT_GREEN = "#DFEBE3"
LIGHT_WINE = "#F0DFE0"
MASK = "#756963"
WHITE = "#FFFFFF"
WIDTH = 1180
HEIGHT = 388

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
    "svg.fonttype": "none", "svg.hashsalt": "nora-network-overview",
    "mathtext.fontset": "dejavusans", "image.composite_image": False,
})


def text(ax, x, y, value, *, size=10.5, weight="normal", color=INK, align="left"):
    return ax.text(x, y, value, fontsize=size, fontweight=weight, fontfamily="DejaVu Sans",
                   color=color, ha=align, va="center", zorder=20)


def line(ax, points, *, color=GRAY, width=0.9, arrow=False, dashed=False, z=4):
    x, y = zip(*points)
    style = (0, (3.4, 2.2)) if dashed else "-"
    if len(points) > 2 or not arrow:
        plot_end = -1 if arrow else len(points)
        ax.plot(x[:plot_end], y[:plot_end], color=color, lw=width, linestyle=style,
                solid_capstyle="round", solid_joinstyle="round", zorder=z)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], shrinkA=0, shrinkB=0,
                                    arrowstyle="-|>", mutation_scale=7, lw=width, color=color,
                                    zorder=z, linestyle=style))


def rect(ax, x, y, width, height, face=WHITE, edge=RULE, linewidth=0.65, *, z=2):
    return ax.add_patch(Rectangle((x, y), width, height, facecolor=face, edgecolor=edge,
                                  linewidth=linewidth, zorder=z))


def spread(ax, x, y, size, depth, color, face):
    """An oblique stack of feature cards. Card sizes convey no detector dimensions."""
    for t in range(depth-1, -1, -1):
        i = t * 4.3
        ax.add_patch(Polygon(((x+i, y-i), (x+size+i, y-i-2),
                              (x+size+i, y+size-i-2), (x+i, y+size-i)),
                             closed=True, facecolor=face, edgecolor=color,
                             linewidth=0.8, zorder=6+depth-t))


def native_grid(ax, x, y, cell, *, mode):
    rows = (".........", "....##...", "..#####..", ".#######.",
            "...####..", ".....#...", ".........")
    positive = {(7, 4), (8, 3)}
    negative = {(1, 3), (2, 3)}
    for row, values in enumerate(rows):
        for col, val in enumerate(values):
            xy = (col, row)
            fg = val == "#"
            color = MASK if fg else PAPER
            if mode == "sample":
                if xy in positive:
                    color = LIGHT_GREEN
                if xy in negative:
                    color = LIGHT_WINE
            elif mode == "output":
                if xy in positive:
                    color = MASK
                if xy in negative:
                    color = PAPER
            rect(ax, x+col*cell, y+row*cell, cell, cell, color, RULE, 0.34, z=8)
            if mode == "sample" and xy in positive | negative:
                sign, ink = ("+", GREEN) if xy in positive else ("−", WINE)
                text(ax, x+(col+.5)*cell, y+(row+.51)*cell, sign, color=ink,
                     size=8.0, weight="bold", align="center")
    rect(ax, x, y, 9*cell, 7*cell, "none", GRAY, 0.63, z=9)


def sample_chip(ax, author_image):
    rect(ax, 25, 149, 113, 113, PAPER, RULE, 0.8)
    if author_image is not None:
        ax.imshow(author_image, extent=(25, 138, 262, 149), interpolation="nearest",
                  zorder=5, aspect="auto")
    else:
        # A neutral vector placeholder, labeled as a chip schematic, without simulated texture.
        for x in range(36, 137, 10):
            line(ax, [(x, 150), (x, 261)], color=RULE, width=0.37, z=3)
        for y in range(160, 261, 10):
            line(ax, [(26, y), (137, y)], color=RULE, width=0.37, z=3)
        ax.add_patch(Polygon(((46, 235), (101, 167), (116, 177), (124, 190),
                              (73, 244), (50, 249)), facecolor="#BFAD98",
                             edgecolor=GRAY, linewidth=.8, zorder=6))
        ax.add_patch(Polygon(((62, 230), (104, 184), (109, 185), (99, 202),
                              (68, 232)), facecolor=PAPER, edgecolor="none", zorder=7))
    text(ax, 81.5, 273, "Instance-RSDD crop" if author_image is not None else "Chip schematic",
         size=8.8, color=GRAY, align="center")


def source_glyph(ax, x, y):
    for i in range(4):
        for j in range(3):
            rect(ax, x+7*i, y+7*j, 7, 7, WHITE, RULE, 0.36, z=4)
    ax.add_patch(Polygon(((x+3, y+19), (x+21, y+2), (x+26, y+4),
                          (x+23, y+11), (x+8, y+20)), facecolor=LIGHT_GREEN,
                         edgecolor=GREEN, linewidth=.75, zorder=5))


def figure(author_image=None):
    fig = Figure(figsize=(10.55, 10.55*HEIGHT/WIDTH), facecolor=WHITE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()

    # Headings and routes occupy fixed lanes; labels sit beside, rather than on, the arrows.
    for x, heading in ((25, "SAR IMAGE"), (170, "FROZEN FRONT END"),
                       (482, "NATIVE BOUNDARY OCCUPANCY"), (877, "CONSTRAINED MASK WRITE")):
        text(ax, x, 34, heading, size=11.0, weight="bold")
    line(ax, [(25, 52), (1155, 52)], color=RULE, width=0.85, z=1)
    line(ax, [(404, 188), (428, 188), (428, 89), (1116, 89), (1116, 157)],
         color=GRAY, arrow=True)
    text(ax, 669, 74, r"Candidate ID, class, box $B$, confidence $s$", size=10, color=GRAY, align="center")
    text(ax, 1025, 74, "unchanged", size=9.1, color=GRAY, align="center")

    # Frozen detector mechanics are abstract; the two choices share the same diagram.
    sample_chip(ax, author_image)
    line(ax, [(138, 210), (170, 210)], color=ORANGE, arrow=True, width=1.05)
    rect(ax, 170, 132, 234, 137, PAPER, RULE, 0.82)
    rect(ax, 184, 146, 90, 24, WHITE, RULE, 0.64)
    rect(ax, 300, 146, 90, 24, WHITE, RULE, 0.64)
    text(ax, 229, 158, "RF-DETR", size=9.1, weight="bold", align="center")
    text(ax, 345, 158, "D-FINE-seg", size=9.1, weight="bold", align="center")
    text(ax, 287, 157.5, "or", size=8.4, color=GRAY, align="center")
    spread(ax, 207, 194, 29, 4, ORANGE, "#F0E2D4")
    line(ax, [(255, 210), (305, 210)], color=ORANGE, arrow=True)
    for row in range(3):
        for col in range(4):
            ax.scatter([314 + col*11], [199+row*10], s=6.0 if (row+col) % 2 else 9.5,
                       color=ORANGE if (row+col) % 3 else GRAY, linewidth=0, zorder=10)
    text(ax, 224, 248, "Backbone", size=8.9, align="center")
    text(ax, 333, 248, "Query head", size=8.9, align="center")

    # Branch by semantic signal: masks and boxes define the support, and the decoded
    # response both prioritizes uncertain sites and contributes to endpoint features.
    line(ax, [(404, 211), (482, 211)], color=INK, arrow=True, width=0.95)
    text(ax, 433, 201, r"$M,B$", size=10.0, align="center")
    line(ax, [(453, 211), (453, 114), (1045, 114), (1045, 190), (1064, 190)],
         color=GRAY, arrow=True, width=.76)
    text(ax, 949, 104, r"Decoded binary mask $M$", size=9.0, color=GRAY, align="center")
    line(ax, [(404, 244), (439, 244), (439, 289), (642, 289), (642, 248)],
         color=ORANGE, arrow=True, width=.85)
    line(ax, [(509, 289), (509, 251)], color=ORANGE, arrow=True, width=.85)
    text(ax, 485, 279, r"$p_i$", size=9.4, color=ORANGE, align="center")
    line(ax, [(138, 252), (154, 252), (154, 307), (665, 307), (665, 248)],
         color=GRAY, arrow=True, width=.76)
    text(ax, 341, 298, r"Native intensity $L_i$", size=9.0, color=GRAY, align="center")

    # Endpoint signs remain explicit from native sampling to writing.
    native_grid(ax, 482, 168, 11.7, mode="sample")
    text(ax, 534, 153, "Inner / outer rings", size=9.2, weight="bold", align="center")
    text(ax, 550, 265, r"$|p_i - 0.5|$", size=8.4, color=GRAY, align="center")
    line(ax, [(588, 206), (625, 206)], color=GREEN, arrow=True, width=.92)
    line(ax, [(588, 221), (625, 221)], color=WINE, arrow=True, width=.92)
    text(ax, 607, 194, r"$S_{+}$", size=9.1, color=GREEN, align="center")
    text(ax, 607, 234, r"$S_{-}$", size=9.1, color=WINE, align="center")
    text(ax, 652, 154, r"$v_i\in\mathbb{R}^{14}$", size=10.7, weight="bold", align="center")
    text(ax, 652, 172, r"geometry, $p_i,s,L_i$", size=7.9, color=GRAY, align="center")
    feature_fills = [ORANGE] + [GRAY]*8 + [GREEN]*5
    for row in range(2):
        for col in range(7):
            rect(ax, 628+7.0*col, 205+9.5*row, 6.3, 8.5,
                 feature_fills[row*7+col], WHITE, 0.45, z=7)
    line(ax, [(680, 214), (715, 214)], color=ORANGE, arrow=True, width=.9)

    # ROH uses per-endpoint feature cards, not spatial convolution.
    text(ax, 775, 153, "ROH", size=10.0, weight="bold", align="center")
    spread(ax, 720, 201, 27, 4, ORANGE, "#F0E2D4")
    spread(ax, 765, 202, 25, 3, ORANGE, "#F0E2D4")
    spread(ax, 803, 204, 21, 3, ORANGE, "#F0E2D4")
    text(ax, 773, 248, "28 → 24 → 24 → 1 or 2", size=8.7, color=GRAY, align="center")
    line(ax, [(835, 214), (855, 214), (855, 186), (877, 186)],
         color=GREEN, arrow=True, width=.90)
    line(ax, [(855, 214), (855, 251), (877, 251)],
         color=WINE, arrow=True, width=.90)
    text(ax, 850, 174, r"$\hat{q}_i$", size=9.1, color=GREEN, align="center")
    text(ax, 845, 265, r"$\hat{q}_{i1},\hat{q}_{i2}$", size=9.1, color=WINE, align="center")

    # The two panels are alternative writers with a common output interface.
    rect(ax, 877, 164, 149, 46, LIGHT_GREEN, RULE, .72, z=3)
    rect(ax, 877, 229, 149, 46, LIGHT_WINE, RULE, .72, z=3)
    text(ax, 890, 176, "Equal counts", size=8.8, weight="bold")
    text(ax, 890, 196, r"$+k,\,-k$", size=11.0, color=INK)
    text(ax, 890, 241, "Fixed total count", size=8.8, weight="bold")
    text(ax, 890, 261, r"$+a,\,-(2k_s-a)$", size=11.0, color=INK)
    line(ax, [(1026, 187), (1034, 187), (1034, 221), (1064, 221)],
         color=INK, arrow=True, width=.9)
    line(ax, [(1026, 252), (1034, 252), (1034, 221)], color=INK, width=.9)
    native_grid(ax, 1065, 178, 10.0, mode="output")
    text(ax, 1110, 265, r"Edited mask $M'$", size=9.4, weight="bold", align="center")

    # Only the ROH is fitted in the marked training lane.
    line(ax, [(25, 329), (1155, 329)], color=RULE, width=.78, z=1)
    text(ax, 25, 352, "SOURCE FITTING", size=9.5, weight="bold", color=GRAY)
    source_glyph(ax, 181, 341)
    text(ax, 224, 352, "Source polygons", size=9.5)
    line(ax, [(360, 352), (388, 352)], color=GRAY, dashed=True, arrow=True)
    text(ax, 401, 352, r"Fractional coverage", size=9.6)
    line(ax, [(583, 352), (611, 352)], color=GRAY, dashed=True, arrow=True)
    text(ax, 625, 352, "Separate ROH fits", size=9.6)
    line(ax, [(772, 352), (805, 352), (805, 273)], color=GRAY, dashed=True, arrow=True)
    text(ax, 822, 352, r"Parameters $\theta$", size=9.0, color=GRAY)
    return fig


def read_author_crop(path):
    """Read the original 78x78 scene image embedded as the first panel of approved PDF."""
    with fitz.open(path) as document:
        if len(document) != 1:
            raise ValueError("Expected the approved one-page scene-to-mask PDF")
        page = document[0]
        candidates = page.get_image_info(xrefs=True)
        first = candidates[0]
        if tuple((first["width"], first["height"])) != (78, 78):
            raise ValueError("The first scene-to-mask image must be the approved 78x78 SAR crop")
        if not fitz.Rect(7, 30, 145, 168).contains(fitz.Rect(first["bbox"])):
            raise ValueError("The image location differs from the approved first SAR panel")
        original = fitz.Pixmap(document, first["xref"])
        if original.alpha:
            original = fitz.Pixmap(original, 0)
        png = original.tobytes("png")
    print("Author crop extracted from approved scene figure SHA256", hashlib.sha256(png).hexdigest())
    return mimage.imread(BytesIO(png), format="png")


def write(fig, destination):
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / "nora_network_overview"
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None}, facecolor=WHITE, dpi=160)
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None}, facecolor=WHITE, dpi=160)
    with fitz.open(base.with_suffix(".pdf")) as document:
        page = document[0]
        page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(base.with_suffix(".png"))
        print(destination.name, page.rect, "embedded images", len(page.get_image_info()),
              "editable paths", len(page.get_drawings()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--author-scene", type=Path)
    args = parser.parse_args()
    write(figure(), args.output / "public")
    if args.author_scene:
        write(figure(read_author_crop(args.author_scene)), args.output / "local")


if __name__ == "__main__":
    main()
