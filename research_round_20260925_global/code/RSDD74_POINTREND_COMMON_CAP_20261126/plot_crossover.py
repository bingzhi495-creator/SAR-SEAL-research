"""Plot the exact common-cap writer crossover from published aggregate scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BLUE = "#243D56"
TEAL = "#087E8B"
INK = "#152839"
KEYS = (
    ("I_minus_L", "Occupancy I − L"),
    ("pr_I_minus_L", "PointRend I − L"),
    ("writer_head_interaction", "Interaction"),
)
VIEWS = (
    ("full", "74-chip reserve", .11, TEAL, "o"),
    ("date_separated", "Nested 42 chips", -.11, BLUE, "s"),
)


def draw(document):
    fig, (ap, overlap) = plt.subplots(1, 2, figsize=(7.15, 2.55))
    fig.subplots_adjust(left=.255, right=.975, bottom=.34, top=.86, wspace=.23)
    for ax, title in ((ap, "Mask AP difference"), (overlap, "Continuous IoU difference")):
        ax.set_title(title, loc="left", fontsize=9.3, color=INK, pad=8)
        ax.axvline(0, color="#ACB6BF", linestyle=(0, (2.2, 2.2)), linewidth=.85)
        ax.axhline(.5, color="#DCE4EA", linewidth=.6)
        ax.set_ylim(-.6, 2.54)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B7C4CD")
        ax.tick_params(axis="y", length=0, pad=7, labelcolor=INK)
        ax.tick_params(axis="x", color="#A4B2BE", labelcolor="#4C6172", length=3)
    ap.set_yticks((2, 1, 0), labels=[label for _, label in KEYS])
    overlap.set_yticks([])
    for group, _, offset, color, marker in VIEWS:
        view = document["views"][group]
        if not view["historical_learned_metrics_reproduced"]:
            raise ValueError("The learned score verification must pass")
        for i, (key, _) in enumerate(KEYS):
            record = view["contrasts"][key]
            y = 2 - i + offset
            overlap_record = record["geometry"]["continuous_iou_points"]
            lo, hi = overlap_record["interval_95"]
            overlap.hlines(y, lo, hi, color=color, linewidth=1.3)
            for ax, value in ((ap, record["ap_points"]), (overlap, overlap_record["delta"])):
                ax.scatter(value, y, marker=marker, edgecolor=color,
                           facecolor=color if marker == "o" else "white",
                           linewidth=.9, s=19, zorder=4)
    fig.text(.62, .198, "Difference in percentage points", ha="center", color=INK)
    handles = [Line2D([0],[0],color=c,marker=m,markerfacecolor=c if m == "o" else "white",
                      linestyle="None",markersize=4.4,label=label)
               for _, label, _, c, m in VIEWS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.62, .022),
               ncol=2, frameon=False, columnspacing=2.0, handletextpad=.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.score.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8.4, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rsdd74-pointrend-common-cap"})
    fig = draw(document)
    for kind in ("pdf", "svg"):
        fig.savefig(args.out / f"writer_score_crossover.{kind}")


if __name__ == "__main__":
    main()
