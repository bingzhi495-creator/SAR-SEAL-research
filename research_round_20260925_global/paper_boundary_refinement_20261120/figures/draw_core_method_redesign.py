"""Render the paper's overview and matched-count decision from one worked example."""

import argparse
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


INK = "#111D2B"
NAVY = "#25384B"
TEAL = "#087E8B"
RUST = "#A94E37"
MUTED = "#526370"
GRID = "#C9D3DA"
PAPER = "#F5F8FA"
TEAL_PALE = "#E3F1F1"
RUST_PALE = "#F7E9E2"
WIDTH = 514.8


def frame(height):
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(height, 0))
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, value, *, size=8, color=INK, weight="normal", align="left"):
    ax.text(x, y, value, va="center", ha=align, fontsize=size, color=color,
            fontweight=weight, linespacing=1.13)


def rule(ax, x0, y0, x1, y1, color=GRID, width=0.55):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, solid_capstyle="butt")


def arrow(ax, x0, y0, x1, y1, *, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7.5,
                                 color=color, linewidth=0.86, shrinkA=0, shrinkB=0,
                                 linestyle="--" if dashed else "-", zorder=6))


def heading(ax, x, y, letter, text):
    label(ax, x, y, letter, size=9.1, weight="bold", color=TEAL)
    label(ax, x + 13, y, text, size=9.1, weight="bold")


def grid(ax, x, y, step, mask, plus=(), minus=(), edits=False):
    plus = set(plus)
    minus = set(minus)
    for row, col in np.ndindex(mask.shape):
        cell = (col, row)
        left, top = x + col * step, y + row * step
        fill = NAVY if mask[row, col] else "white"
        if cell in plus:
            fill = TEAL if edits else TEAL_PALE
        if cell in minus:
            fill = RUST_PALE
        ax.add_patch(Rectangle((left, top), step, step, facecolor=fill,
                               edgecolor=GRID, linewidth=0.39))
        if cell in plus:
            label(ax, left + step / 2, top + step / 2, "+", size=7.4, align="center",
                  weight="bold", color="white" if edits else TEAL)
        if cell in minus:
            label(ax, left + step / 2, top + step / 2, "−", size=7.4, align="center",
                  weight="bold", color=RUST)


def overview(ex):
    mask, added, drops = ex["BASE_MASK"], ex["ADD"], ex["DROP"]
    _, records = ex["example"]()
    add_ids, drop_ids, updated, _ = records[1]
    fig, ax = frame(155)
    for x, letter, text in (
        (10, "a", "Frozen instance"), (128, "b", "Sample the boundary"),
        (262, "c", "Score each cell"), (390, "d", "Write the mask"),
    ):
        heading(ax, x, 13, letter, text)
    rule(ax, 10, 24, 505, 24, color=NAVY)
    grid(ax, 34, 35, 7.3, mask)
    grid(ax, 165, 35, 7.3, mask, added, drops)
    grid(ax, 417, 35, 7.3, updated,
         (added[i] for i in add_ids), (drops[i] for i in drop_ids), edits=True)
    for a, b in ((103, 153), (233, 263), (370, 406)):
        arrow(ax, a, 56, b, 56)
    label(ax, 63.2, 93, r"Binary mask $M$", align="center")
    label(ax, 63.2, 105, r"Box $B$, response $p_i$", align="center", color=MUTED, size=7.4)
    label(ax, 194.2, 93, r"Outer $S_+$ · inner $S_-$", align="center")
    label(ax, 194.2, 105, "Two rings inside the box", align="center", color=MUTED, size=7.4)
    rule(ax, 279, 49, 350, 49, color=TEAL, width=1.2)
    label(ax, 314.5, 59, r"Occupancy $\widehat q_i$", align="center", size=8.7, weight="bold")
    label(ax, 314.5, 73, "Source polygons train", align="center", size=7.5, color=MUTED)
    label(ax, 314.5, 84, "the residual head", align="center", size=7.5, color=MUTED)
    label(ax, 446.2, 93, r"Add $k$ and remove $k$", align="center")
    label(ax, 446.2, 105, "Independent ranks", align="center", color=MUTED, size=7.4)

    ax.add_patch(Rectangle((10, 120), 495, 25, ec="none", fc=PAPER))
    label(ax, 18, 132.5, "Separate refiner control", size=7.6, weight="bold")
    label(ax, 142, 132.5, r"Same $M$ prompts Phoenix", size=7.6)
    arrow(ax, 248, 132.5, 272, 132.5, dashed=True)
    label(ax, 282, 132.5, "Fixed response", size=7.6)
    arrow(ax, 346, 132.5, 366, 132.5, dashed=True)
    label(ax, 375, 132.5, "Change support or write rule", size=7.6)
    return fig


def decision(ex):
    mask, added, drops = ex["BASE_MASK"], ex["ADD"], ex["DROP"]
    addq, dropq = ex["ADD_SCORE"], ex["DROP_SCORE"]
    gains, records = ex["example"]()
    fig, ax = frame(155)
    for x, letter, text in (
        (10, "a", "Fixed distance assignment"), (178, "b", "Independent ranking"),
        (339, "c", "Equal-count outcomes"),
    ):
        heading(ax, x, 13, letter, text)
    rule(ax, 10, 24, 505, 24, color=NAVY)
    for x in (166, 327):
        rule(ax, x, 34, x, 124)
    label(ax, 19, 36, "Pair", size=7.2, color=MUTED)
    label(ax, 54, 36, r"Add $\hat q$", size=7.2, color=TEAL, align="center")
    label(ax, 91, 36, r"Drop $\hat q$", size=7.2, color=RUST, align="center")
    label(ax, 142, 36, "Gain", size=7.2, color=MUTED, align="right")
    for i, y in enumerate((53, 69, 85, 101)):
        active = i in records[0][0]
        if active:
            ax.add_patch(Rectangle((12, y - 7), 143, 14, ec="none", fc=PAPER))
        label(ax, 19, y, str(i + 1), weight="bold" if active else "normal")
        label(ax, 54, y, f"{addq[i]:.2f}", color=TEAL, align="center")
        rule(ax, 68, y, 75, y)
        label(ax, 91, y, f"{dropq[i]:.2f}", color=RUST, align="center")
        label(ax, 142, y, f"{gains[i]:.2f}", align="right",
              weight="bold" if active else "normal")
    label(ax, 12, 120, "Take the two highest pair gains.", size=7.35, color=MUTED)

    label(ax, 240, 36, r"Add $\hat q$  $\downarrow$", size=7.5, color=TEAL, align="right")
    label(ax, 311, 36, r"Drop $\hat q$  $\uparrow$", size=7.5, color=RUST, align="right")
    plus_rank = np.argsort(-addq, kind="stable")
    drop_rank = np.argsort(dropq, kind="stable")
    for rank, y in enumerate((53, 69, 85, 101)):
        for side, values, xi, color, pale in (
            (plus_rank, addq, 224, TEAL, TEAL_PALE),
            (drop_rank, dropq, 294, RUST, RUST_PALE),
        ):
            i = side[rank]
            if rank < 2:
                ax.add_patch(Rectangle((xi - 38, y - 6.4), 53, 12.8, ec="none", fc=pale))
            label(ax, xi - 31, y, str(i + 1), size=7.45, color=MUTED)
            label(ax, xi + 8, y, f"{values[i]:.2f}", color=color, align="right",
                  weight="bold" if rank < 2 else "normal")
    label(ax, 178, 120, "Take two outside and two inside.", size=7.35, color=MUTED)

    for x, name, (ap, dp, updated, gain) in zip((344, 426), ("Fixed pairs", "Independent"), records):
        label(ax, x + 30, 36, name, color=NAVY, weight="bold", size=7.55, align="center")
        grid(ax, x, 47, 7.5, updated, (added[i] for i in ap), (drops[i] for i in dp), edits=True)
        label(ax, x + 30, 102, rf"$\widehat g={gain:.2f}$", align="center")
    label(ax, 418, 120, r"Each keeps the original foreground area.", size=7.25,
          align="center", color=MUTED)
    rule(ax, 10, 132, 505, 132)
    label(ax, 10, 144, r"Worked example   $k=2$, identical endpoint scores and candidate mask.",
          size=7.5, color=MUTED)
    return fig


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--example-source", type=Path, default=Path(__file__).with_name("draw_matched_budget_pipeline.py"))
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ex = runpy.run_path(str(args.example_source))
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "svg.fonttype": "none",
        "svg.hashsalt": "sar-seal-method-layout",
        "pdf.fonttype": 42,
    })
    for name, render in (("matched_budget_pipeline", overview), ("method_signed_selection", decision)):
        fig = render(ex)
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(args.output / f"{name}.{suffix}", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
