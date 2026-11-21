"""Render the closed FIT109 source-development contrasts from aggregate values.

Requires Python >=3.9 and matplotlib. Run this file to write SVG, PDF, and PNG
alongside it, or pass --output-dir. The SVG preserves individual text objects.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text


ROWS = (
    # control, AP, source-equal continuous IoU, paired source-bootstrap interval
    ("Z (zero)",     -0.021393723684204335,  0.03902225225241953,
     (-0.013975286905107829, 0.09504364451564935)),
    ("F (spatial)",   0.35196628052527323,   0.07280357674035465,
     (-0.019289465831842295, 0.1716044087733218)),
    ("QP (permuted)", -0.1128769682301578, -0.003269124178478346,
     (-0.06603637236052465, 0.057889679022893076)),
)

INK = "#24313A"
NAVY = "#254F70"
RUST = "#8A4933"
ZERO = "#90989D"
GATE = "#9D9C99"


def signed(value, digits):
    return f"{value:+.{digits}f}".replace("-", "−")


def render(output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": True, "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "SAR-SEAL-FIT109-OOF-20261121",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.05), sharey=True)
    fig.subplots_adjust(left=.169, right=.985, bottom=.215, top=.805, wspace=.175)
    panels = (
        ("(a)  Mask AP", (-.16, .43), [-.1, 0, .1, .2, .3, .4], .30),
        ("(b)  Source-equal continuous IoU", (-.082, .235), [-.05, 0, .05, .10, .15, .20], .20),
    )
    positions = [2, 1, 0]
    for axis, (title, xlim, ticks, gate) in zip(axes, panels):
        axis.set_title(title, loc="left", fontsize=9.1, fontweight="bold", pad=10)
        axis.set_xlim(*xlim)
        axis.set_ylim(-.45, 2.72)
        axis.set_yticks(positions, [r[0] for r in ROWS])
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else signed(v, 2)))
        axis.tick_params(axis="x", labelsize=7.25, width=.6, length=3, pad=3)
        axis.tick_params(axis="y", length=0, labelsize=8.3, pad=7)
        for name in ("top", "right", "left"):
            axis.spines[name].set_visible(False)
        axis.spines["bottom"].set_color(ZERO)
        axis.spines["bottom"].set_linewidth(.6)
        axis.axvline(0, lw=.8, color=ZERO, zorder=1)
        axis.axvline(gate, lw=.8, color=GATE, ls=(0, (3, 3)), zorder=1)
        axis.set_xlabel("Q minus control (points)", fontsize=8.1, labelpad=5)

    for y, (_, ap, iou, (low, high)) in zip(positions, ROWS):
        axes[0].plot(ap, y, marker="D", markersize=4.9, color=RUST, zorder=3)
        axes[1].errorbar(iou, y, xerr=[[iou-low], [high-iou]], fmt="o", ms=4.9,
                         color=NAVY, capsize=3, capthick=.9, elinewidth=1.1, zorder=3)
        for axis, value, color in ((axes[0], ap, RUST), (axes[1], iou, NAVY)):
            axis.annotate(signed(value, 3), (value, y), xytext=(0, 5.5),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=7.6, color=color,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .25})

    # Check the actual drawing surface at its nominal print size, including labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        bounds = text.get_window_extent(renderer)
        if not all(fig.bbox.contains(x, y) for x, y in bounds.get_points()):
            raise ValueError(f"Text exceeds figure bounds: {text.get_text()}")

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "fit109_query_contrasts"
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    render(parser.parse_args().output_dir)
