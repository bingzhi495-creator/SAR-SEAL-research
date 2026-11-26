"""Redraw five method schematics from the original verified worked examples."""
from argparse import ArgumentParser
from pathlib import Path
import runpy

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np


INK, TEAL, ORANGE, BLUE, GRAY, RULE = ("#243247", "#007F83", "#C4552D", "#2D5E91", "#5C6675", "#DCE5EE")
PAPER, BLUE_TINT, TEAL_TINT, ORANGE_TINT, WHITE = ("#F4F7FA", "#EAF1F9", "#E7F5F3", "#FBEDE7", "#FFFFFF")
SINGLE, DOUBLE = 247.9, 514.8
mpl.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "sar-native-edit-method"})


def frame(w, h):
    fig = plt.figure(figsize=(w / 72, h / 72), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, value, size=8, *, color=INK, bold=False, ha="left"):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight="bold" if bold else "normal",
                   ha=ha, va="center", zorder=6)


def center(ax, x, y, value, size=8, **kwargs):
    return txt(ax, x, y, value, size, ha="center", **kwargs)


def rule(ax, x0, y0, x1, y1, color=RULE, width=.75, dashed=False):
    ax.plot((x0, x1), (y0, y1), lw=width, color=color, ls=(0, (3, 2)) if dashed else "-", zorder=4)


def arrow(ax, x0, y0, x1, y1, color=GRAY, dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=.85, color=color, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (3, 2)) if dashed else "-", zorder=5))


def rect(ax, x, y, w, h, fill=WHITE, edge=RULE, width=.75):
    shape = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=width, zorder=2)
    ax.add_patch(shape)
    return shape


def heading(ax, x, y, letter, title):
    txt(ax, x, y, "(" + letter + ")", 8.5, color=BLUE, bold=True)
    txt(ax, x + 17, y, title, 9, bold=True)


def mark(ax, x, y, sign, *, strong=False, size=7.8):
    color = WHITE if strong and sign == "+" else TEAL if sign == "+" else ORANGE
    center(ax, x, y, sign, size, color=color, bold=True)


def grid(ax, x, y, s, mask, adds=(), drops=(), *, edited=False, labels=None, links=()):
    adds, drops = set(adds), set(drops)
    for row, col in np.ndindex(mask.shape):
        pt = (col, row)
        c = INK if mask[row, col] else WHITE
        if pt in adds:
            c = TEAL if edited else TEAL_TINT
        if pt in drops:
            c = ORANGE_TINT
        rect(ax, x+s*col, y+s*row, s, s, c, RULE, .35)
    for a, d in links:
        x0, y0 = x+s*(a[0]+.5), y+s*(a[1]+.5)
        x1, y1 = x+s*(d[0]+.5), y+s*(d[1]+.5)
        ax.plot((x0, x1), (y0, y1), color=BLUE, lw=1.35, alpha=.85, zorder=4)
    for side, points in (("+", adds), ("−", drops)):
        for col, row in sorted(points):
            mark(ax, x+s*(col+.5), y+s*(row+.49), side, strong=edited, size=max(5.8, s*.84))
    if labels:
        for (col, row), value in labels.items():
            center(ax, x+s*(col+.5), y+s*(row+.49), value, 6.3, color=GRAY, bold=True)


def legend(ax, x, y):
    for offset, symbol, caption, tint in ((0, "+", "Add", TEAL_TINT), (45, "−", "Remove", ORANGE_TINT)):
        rect(ax, x+offset, y-4.5, 9, 9, tint)
        mark(ax, x+offset+4.5, y, symbol)
        txt(ax, x+offset+13, y, caption, 7.4, color=GRAY)


def corrected_mask(native, adds, drops):
    after = native.copy()
    for x, y in adds:
        assert not native[y, x]
        after[y, x] = True
    for x, y in drops:
        assert native[y, x]
        after[y, x] = False
    assert int(np.count_nonzero(after != native)) == len(adds) + len(drops)
    return after


def overview(ex):
    fig, ax = frame(DOUBLE, 171)
    base, add, drop = (ex[key] for key in ("BASE_MASK", "ADD", "DROP"))
    _, actions = ex["example"]()
    selected_add, selected_drop, edited, _ = actions[1]
    starts = (9, 136, 263, 390)
    subtitle = ("Frozen candidate", "Sample boundary cells", "Score occupancy", "Write selected cells")
    for index, (x, title) in enumerate(zip(starts, subtitle, strict=True)):
        rect(ax, x, 9, 116, 95, BLUE_TINT if index == 3 else WHITE)
        heading(ax, x+7, 23, "abcd"[index], title)
        if index:
            arrow(ax, x-11, 57, x-2, 57, BLUE if index == 3 else GRAY)
    grid(ax, 36, 37, 7.7, base)
    grid(ax, 163, 37, 7.7, base, add, drop)
    grid(ax, 417, 37, 7.7, edited, [add[i] for i in selected_add], [drop[i] for i in selected_drop], edited=True)
    for x, label, sublabel in ((67, r"Native mask $M$", r"Box $B$ and response $p_i$"),
                               (194, r"Outside $S_+$ · inside $S_-$", "Two rings clipped to the box"),
                               (448, r"Add $k$ · remove $k$", "Candidate and score fixed")):
        center(ax, x, 89, label, 8, bold=True)
        center(ax, x, 98, sublabel, 7.25, color=GRAY)
    locations = (276, 298, 320, 342, 364)
    for x, value in zip(locations, ("14","28","24","24","1"), strict=True):
        rect(ax, x-8, 43, 16, 20, TEAL_TINT, RULE)
        center(ax, x, 53, value, 7.3, bold=True)
    for x0, x1 in zip(locations, locations[1:]):
        arrow(ax, x0+8, 53, x1-8, 53)
    center(ax, 321, 76, "ROH", 7.7, bold=True)
    center(ax, 321, 91, r"Predicted occupancy $\widehat q_i$", 8.15, color=TEAL, bold=True)
    rect(ax, 9, 117, 497, 19, TEAL_TINT, RULE)
    txt(ax, 18, 126, "Training only", 8, color=TEAL, bold=True)
    txt(ax, 92, 126, "Author polygons → fractional cell coverage → source BCE", 8)
    arrow(ax, 317, 117, 317, 105, TEAL, True)
    rect(ax, 9, 143, 497, 19, PAPER, RULE)
    txt(ax, 18, 152.5, "Frozen-refiner control", 8, bold=True)
    txt(ax, 131, 152.5, "Same mask prompt → fixed Phoenix response → vary support or write rule", 7.8, color=GRAY)
    txt(ax, 497, 152.5, "Schematic", 7.2, color=GRAY, ha="right")
    return fig


def occupancy():
    fig, ax = frame(SINGLE, 308)
    native = np.zeros((8, 10), dtype=bool)
    native[2:7, 2:9] = True
    native[1, 5:7] = True
    add = ((3, 1), (4, 0), (6, 0), (8, 1), (9, 3))
    drop = ((3, 2), (5, 1), (7, 2), (8, 3), (8, 5))
    assert all(not native[y, x] for x, y in add) and all(native[y, x] for x, y in drop)
    heading(ax, 10, 14, "a", "Native boundary support")
    grid(ax, 13, 29, 7.7, native, add, drop)
    outline = Rectangle((20, 27.7), 71.2, 55.5, fill=False, ec=GRAY, linewidth=.9, ls=(0, (3, 2)), zorder=4)
    ax.add_patch(outline)
    txt(ax, 95, 36, r"$B$", 8.2, color=GRAY)
    txt(ax, 111, 42, r"$S_+$ outside $M$", 8, color=TEAL, bold=True)
    txt(ax, 111, 57, r"$S_-$ inside $M$", 8, color=ORANGE, bold=True)
    txt(ax, 111, 72, "Two rings inside the box", 7.8)
    txt(ax, 111, 85, r"Rank by $|p_i-0.5|$, then index", 7.5)
    txt(ax, 111, 98, r"Keep $n=\min(32,n_+,n_-)$ per side", 7.1)
    rule(ax, 10, 110, 237, 110)
    heading(ax, 10, 124, "b", "Point readout and supervision")
    rect(ax, 10, 139, 108, 87, PAPER)
    rect(ax, 128, 139, 109, 87, TEAL_TINT)
    center(ax, 64, 150, "Decoder readout", 8, bold=True)
    center(ax, 182, 150, "Training only", 8, color=TEAL, bold=True)
    # Original four-vertex interpolation and polygon-cell geometry.
    x, y, step = 46, 162, 28
    rect(ax, x, y, step, step, WHITE)
    x0, y0 = x+step*.61, y+step*.55
    for dx, dy in ((0, 0), (step, 0), (0, step), (step, step)):
        rule(ax, x+dx, y+dy, x0, y0, RULE, .65, True)
        ax.add_patch(Circle((x+dx, y+dy), 1.8, color=GRAY, zorder=5))
    ax.add_patch(Circle((x0, y0), 2.2, color=BLUE, zorder=5))
    txt(ax, 79, 176, r"$c_i$", 7.7, color=BLUE)
    center(ax, 64, 200, "Bilinear at native center", 7.2)
    center(ax, 64, 211, "D-FINE probabilities", 7.1, color=GRAY)
    center(ax, 64, 220, r"RF-DETR logits, then $\sigma$", 7.1, color=GRAY)
    x, y = 168, 161
    cell = rect(ax, x, y, 29, 29, WHITE, GRAY)
    coords = np.array([(-8, 32), (8, 21), (22, -5), (47, 7), (46, 47), (-2, 45)])*(29/40)+(x,y)
    fill = Polygon(coords, closed=True, fc="#BCDFDB", ec="none", zorder=3)
    ax.add_patch(fill)
    fill.set_clip_path(cell)
    ax.add_patch(Polygon(coords, closed=True, fill=False, ec=TEAL, lw=.8, zorder=4))
    txt(ax, 202, 168, r"$G$", 7.8, color=TEAL)
    txt(ax, 155, 172, r"$C_i$", 7.8, color=GRAY)
    center(ax, 182, 202, r"$q_i=|C_i\cap G|/|C_i|$", 8, bold=True)
    center(ax, 182, 217, "Author polygon coverage", 7.4, color=GRAY)
    rule(ax, 10, 235, 237, 235)
    heading(ax, 10, 247, "c", "ROH")
    for x, desc, width in ((13, r"$v_i$ (14)", 46), (69, r"$[z_i,z_i]$ (28)", 67),
                           (145, "24", 24), (180, "24", 24), (215, "1", 20)):
        rect(ax, x, 257, width, 15, BLUE_TINT if x==215 else PAPER)
        center(ax, x+width/2, 264.5, desc, 8, bold=True)
    for a, b in ((59, 67), (136, 143), (169, 178), (204, 213)):
        arrow(ax, a, 264.5, b, 264.5)
    txt(ax, 12, 283, r"$\widehat q_i=\sigma[\mathrm{logit}(p_i^\epsilon)+f_\theta([z_i,z_i])]$", 9, color=TEAL)
    txt(ax, 12, 297, "Fractional BCE · 1,321 parameters", 7.5, color=GRAY)
    txt(ax, 236, 297, "Schematic", 7.2, color=GRAY, ha="right")
    return fig


def bounded(ex, previous):
    fig, ax = frame(DOUBLE, 176)
    base, drop = ex["BASE_MASK"], ex["DROP"]
    add = previous["ADD"]
    examples = previous["actions"]()
    distance = ((np.array(add)[:, None, :] - np.array(drop)[None, :, :]) ** 2).sum(axis=2)
    threshold = int(max(distance[add.index(a), drop.index(d)]
                        for a, d in zip(examples[0][1], examples[0][2], strict=True)))
    assert threshold == 2
    rect(ax, 9, 9, 497, 23, PAPER)
    txt(ax, 18, 20.5, "Shared candidate, endpoint scores and realized budget", 8.5, bold=True)
    txt(ax, 496, 20.5, "2 adds + 2 removes", 8.5, color=GRAY, bold=True, ha="right")
    names = ("Distance-first pairing", "Bounded re-pairing", "Independent ranking")
    descs = ("Rank the original distance pairs", "Reassign within the inherited cap", "Rank the two sides separately")
    for i, (x, name, desc, result) in enumerate(zip((9, 177, 345), names, descs, examples, strict=True)):
        mask, plus, minus, gain = result
        rect(ax, x, 42, 160, 105, BLUE_TINT if i == 2 else WHITE, BLUE if i == 2 else RULE)
        heading(ax, x+8, 54, "abc"[i], name)
        txt(ax, x+8, 68, desc, 7.45, color=GRAY)
        # The original source returns chosen endpoints in their matched order.
        links = tuple(zip(plus, minus, strict=True)) if i < 2 else ()
        grid(ax, x+10, 79, 9.4, mask, plus, minus, edited=True, links=links)
        txt(ax, x+94, 92, rf"$\widehat g={gain:.2f}$", 10, color=BLUE if i==2 else TEAL, bold=True)
        txt(ax, x+94, 114, "Inherited cap" if i==1 else "Selected cap" if i==0 else "No cap", 7.4, color=GRAY)
        txt(ax, x+94, 130, r"$\rho=\sqrt{2}$ px" if i<2 else "Uncoupled", 7.4, color=GRAY)
        assert np.count_nonzero(mask != base) == 4 and mask.sum() == base.sum()
        if i < 2:
            assert max(sum((u-v)**2 for u,v in zip(a,d)) for a,d in links) <= threshold
    legend(ax, 12, 162)
    txt(ax, 142, 162, r"$\widehat g$ · estimated intersection gain", 7.5, color=GRAY)
    txt(ax, 505, 162, "Worked example", 7.4, color=GRAY, ha="right")
    return fig


def allocations(previous):
    fig, ax = frame(DOUBLE, 157)
    native, add, drop = (previous[key] for key in ("MASK", "ADD", "DROP"))
    fixed, free = previous["allocations"]()
    for x, letter, title in ((14, "a", "Shared native support"), (157, "b", "Search the feasible splits"),
                              (347, "c", "Write four cells")):
        heading(ax, x, 16, letter, title)
    rule(ax, 10, 29, 505, 29, BLUE, .9)
    rule(ax, 139, 39, 139, 133)
    rule(ax, 330, 39, 330, 133)
    grid(ax, 42, 45, 8.6, native, add, drop)
    center(ax, 72, 106, "15 foreground cells", 8.1, bold=True)
    center(ax, 72, 119, "Four endpoints per side", 7.5, color=GRAY)
    txt(ax, 157, 46, r"Total flips $a+d=2k_s=4$", 8.5, bold=True)
    for a in range(5):
        style = (BLUE_TINT, BLUE) if a==3 else (PAPER, INK) if a==2 else (WHITE, RULE)
        x = 157+32*a
        rect(ax, x, 58, 30, 20, *style)
        center(ax, x+15, 68, f"{a} / {4-a}", 8, color=BLUE if a==3 else INK, bold=a in (2,3))
    txt(ax, 157, 90, "Additions / removals", 7.6, color=GRAY)
    txt(ax, 157, 107, r"Maximize $\min_r$ estimated IoU gain", 8.8, bold=True)
    txt(ax, 157, 120, "One candidate menu under both references", 7.45, color=GRAY)
    for x, title, action, tint in ((347, "Equal split", fixed, PAPER), (429, "Variable split", free, BLUE_TINT)):
        _, a_idxs, d_idxs = action
        plus, minus = [add[i] for i in a_idxs], [drop[i] for i in d_idxs]
        mask = corrected_mask(native, plus, minus)
        assert len(plus)+len(minus) == 4 and mask.sum() == native.sum()+len(plus)-len(minus)
        rect(ax, x-3, 36, 77, 99, tint)
        center(ax, x+35, 47, title, 8.2, color=BLUE if action is free else INK, bold=True)
        grid(ax, x+9, 58, 7.35, mask, plus, minus, edited=True)
        center(ax, x+35, 111, f"+{len(plus)} / −{len(minus)}", 8.1, bold=True)
        center(ax, x+35, 125, f"Final area {mask.sum()}", 7.35, color=GRAY)
    legend(ax, 13, 145)
    txt(ax, 505, 145, "Illustrative occupancies", 7.4, color=GRAY, ha="right")
    return fig


def phoenix(previous):
    fig, ax = frame(SINGLE, 262)
    native, response, examples = previous["phoenix_example"]()
    heading(ax, 10, 13, "a", "One frozen response")
    palette = mpl.colors.LinearSegmentedColormap.from_list("refiner_response", (WHITE, "#92A8BC", INK))
    for row, col in np.ndindex(response.shape):
        rect(ax, 14+5*col, 26+5*row, 5, 5, palette(response[row, col]), RULE, .27)
    txt(ax, 65, 32, "Image and native mask", 8)
    txt(ax, 65, 45, "Phoenix XL1 · five iterations", 8, bold=True)
    txt(ax, 65, 58, r"Same native response $r_i$ for every writer", 7.5, color=GRAY)
    rule(ax, 10, 70, 238, 70)
    heading(ax, 10, 83, "b", "Support × write rule")
    for x, label, note in ((111, "Threshold", "Author decisions"), (192, "Exact total", r"$2k_\mathrm{att}$ flips")):
        center(ax, x, 100, label, 8.1, bold=True)
        center(ax, x, 111, note, 7.4, color=GRAY)
    for row, scope in enumerate(("narrow", "whole")):
        y = 121+47*row
        rect(ax, 10, y-2, 228, 45, PAPER if row else WHITE)
        txt(ax, 15, y+12, "Narrow" if row==0 else "Whole", 8.4, bold=True)
        txt(ax, 15, y+26, r"$S_+\cup S_-$" if row==0 else r"$\Omega$", 8.3, color=GRAY)
        for x, method in ((92, "threshold"), (173, "count")):
            plus, minus = examples[scope, method]
            mask = corrected_mask(native, plus, minus)
            grid(ax, x, y+1, 4.9, mask, plus, minus, edited=True)
            if method=="count":
                assert len(plus)+len(minus)==4
    heading(ax, 10, 221, "c", "Narrow support, same total")
    p, m = examples["narrow", "count"]
    q, r = examples["narrow", "balanced"]
    txt(ax, 10, 234, f"Free split  +{len(p)} / −{len(m)}", 7.85, color=GRAY)
    arrow(ax, 115, 234, 129, 234, BLUE)
    txt(ax, 138, 234, f"Balanced  +{len(q)} / −{len(r)}", 7.85, color=BLUE, bold=True)
    legend(ax, 11, 252)
    txt(ax, 236, 252, "Schematic", 7.2, color=GRAY, ha="right")
    return fig


def write_figure(fig, output, name, preview):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    for t in fig.axes[0].texts:
        box = t.get_window_extent(renderer)
        if not (0.5 <= box.x0 < box.x1 <= w-.5 and 0.5 <= box.y0 < box.y1 <= h-.5):
            raise RuntimeError(f"Text outside {name}: {t.get_text()}")
    for ext, meta in (("pdf", {"CreationDate": None, "ModDate": None}), ("svg", {"Date": None})):
        fig.savefig(output / f"{name}.{ext}", metadata=meta, facecolor=WHITE)
    if preview:
        fig.savefig(output / f"{name}.png", dpi=165, facecolor=WHITE)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="directory containing the original verified figure generators")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    examples = {name: runpy.run_path(str(args.fixtures_dir / f"{name}.py")) for name in
                ("draw_matched_budget_pipeline", "draw_selection_controls",
                 "draw_signed_allocation", "draw_supporting_stages_redesign")}
    base = examples["draw_matched_budget_pipeline"]
    figures = (
        ("matched_budget_pipeline", overview(base)),
        ("method_native_occupancy", occupancy()),
        ("method_bounded_selection", bounded(base, examples["draw_selection_controls"])),
        ("method_signed_allocation", allocations(examples["draw_signed_allocation"])),
        ("method_phoenix_controls", phoenix(examples["draw_supporting_stages_redesign"])),
    )
    for name, fig in figures:
        write_figure(fig, args.output, name, args.preview)
        w, h = fig.get_size_inches()*72
        print(f"{name}: {w:.1f} x {h:.1f} pt (SVG and PDF)")


if __name__ == "__main__":
    main()

