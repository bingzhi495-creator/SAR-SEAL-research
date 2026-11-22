"""Draw point-head comparisons from the closed HRSID official evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONTRASTS = (
    ("SU=F minus SF", "Independent vs fixed pairing"),
    ("SU minus N", "Natural count vs native"),
    ("SU minus LW=S", "Natural count vs area-free writer"),
)
SEEDS = ("5217", "5218", "5219")
COLORS = ("#42658c", "#7792b1", "#a4b6cc")


def plot(result, prefix):
    if result["status"] != "HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_SCORE_COMPLETE":
        raise ValueError("Use the closed official HRSID score")
    if list(map(str, result["seeds"])) != list(SEEDS):
        raise ValueError("Unexpected point-head seed roster")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.1), sharey=True, layout="constrained")
    for axis, metric, title in zip(axes, ("AP", "AP75"), ("Mask AP", "Mask AP75"), strict=True):
        axis.axvline(0, color="#697782", linewidth=0.9)
        axis.grid(axis="x", color="#dce2e8", linewidth=0.65)
        axis.set_axisbelow(True)
        for i, (key, _) in enumerate(CONTRASTS):
            official = result["paired_contrasts"][key]["official"]
            values = [official["per_head_seed_points"][seed][metric] for seed in SEEDS]
            mean = official["arithmetic_mean_points"][metric]
            axis.hlines(i, min(values), max(values), color="#c2cbd4", linewidth=1)
            for value, color, offset in zip(values, COLORS, (-0.17, 0, 0.17), strict=True):
                axis.scatter(value, i+offset, s=22, color=color if value >= 0 else "#b25040",
                             edgecolors="white", linewidths=0.4, zorder=3)
            axis.scatter(mean, i, s=30, marker="D", facecolor="#183854",
                         edgecolors="white", linewidths=0.4, zorder=4)
        axis.set(xlim=(-0.52, 2.25), xticks=[-0.5, 0, 0.5, 1, 1.5, 2], title=title,
                 xlabel="Difference (percentage points)")
        axis.spines[["top", "left", "right"]].set_visible(False)
        axis.spines["bottom"].set_color("#b0bbc4")
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(CONTRASTS)), [label for _, label in CONTRASTS])
    axes[0].set_ylim(2.48, -0.48)
    legend = [
        Line2D([], [], marker="o", linestyle="none", color="#7792b1", label="Point-head seed", markersize=4.5),
        Line2D([], [], marker="D", linestyle="none", color="#183854", label="Three-seed mean", markersize=4.5),
        Line2D([], [], marker="o", linestyle="none", color="#b25040", label="Decrease", markersize=4.5),
    ]
    figure.legend(handles=legend, loc="outside lower center", frameon=False, ncols=3, borderaxespad=0.25)
    for suffix in (".svg", ".pdf"):
        figure.savefig(prefix.with_suffix(suffix))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.score.read_bytes()), args.output_prefix)


if __name__ == "__main__":
    main()
