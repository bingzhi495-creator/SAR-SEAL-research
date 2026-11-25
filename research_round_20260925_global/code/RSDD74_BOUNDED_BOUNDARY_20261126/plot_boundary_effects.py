"""Plot the closed, fixed-anchor boundary contrasts on the native grid."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INK = "#182A3A"
TEAL = "#087E8B"
BLUE = "#496682"
GRAY = "#738392"
KEYS = ("I_minus_L", "L_minus_F", "I_minus_F")
CONTRASTS = ("Independent − bounded", "Bounded − distance-first", "Independent − distance-first")
VIEWS = (("full", "74-chip reserve"), ("date_separated", "Nested 42-chip view"))


def plot(source):
    if source["status"] != "RSDD74_BOUNDED_PAIRING_NATIVE_BOUNDARY_COMPLETE":
        raise ValueError("Expected the completed aggregate boundary score")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.1, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "rsdd74-native-boundary-contrasts",
    })
    fig, grid = plt.subplots(2, 2, figsize=(7.15, 3.6), sharex=True)
    fig.subplots_adjust(left=.30, right=.98, bottom=.23, top=.92, hspace=.72, wspace=.26)
    intervals = []
    for row, (view_name, label) in enumerate(VIEWS):
        view = source["views"][view_name]
        if not view["original_boundary_i_minus_f_parity"]:
            raise ValueError("The archived boundary scores must reproduce")
        for col, width in enumerate(("1", "2")):
            ax = grid[row][col]
            ax.set_title(f"{label} · {width} px", loc="left", color=INK, fontsize=9.4, pad=7)
            ax.axvline(0, color="#A9B4BE", linestyle=(0, (2, 2)), linewidth=.9, zorder=0)
            ax.set_yticks((2, 1, 0), labels=CONTRASTS if col == 0 else ("", "", ""))
            ax.set_ylim(-.6, 2.65)
            ax.tick_params(axis="y", length=0, pad=7, labelcolor=INK)
            ax.tick_params(axis="x", labelcolor=GRAY, color="#A9B4BE", length=3)
            for side in ("left", "top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#BCC6CF")
            for i, key in enumerate(KEYS):
                record = view["boundary"]["widths"][width]["contrasts"][key]
                center = record["mean_points"]
                color = TEAL if (row, col, i) == (0, 0, 0) else BLUE
                for measure, offset, marker in (("paired_95_interval_points", .105, "o"),
                                                ("block_95_interval_points", -.105, "s")):
                    lo, hi = record[measure]
                    intervals.extend((lo, hi, center))
                    ax.hlines(2-i+offset, lo, hi, linewidth=1.3, color=color)
                    ax.scatter(center, 2-i+offset, s=19, marker=marker, edgecolor=color,
                               facecolor=color if marker == "o" else "white", linewidth=.85, zorder=4)
    pad = max(.2, .10*(max(intervals)-min(intervals)))
    grid[0][0].set_xlim(min(min(intervals), 0)-pad, max(max(intervals), 0)+pad)
    fig.text(.64, .133, "Boundary IoU difference (percentage points)", ha="center", color=INK)
    handles = [Line2D([0], [0], linestyle="-", color=GRAY, marker="o", markersize=4, label="Parent bootstrap"),
               Line2D([0], [0], linestyle="-", color=GRAY, marker="s", markersize=4,
                      markerfacecolor="white", label="Sensor–date block bootstrap")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.64,.012), ncol=2,
               frameon=False, handlelength=1.85, columnspacing=1.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig = plot(json.loads(args.score.read_text(encoding="utf-8")))
    for extension in ("pdf", "svg"):
        fig.savefig(args.out / f"boundary_effects.{extension}")


if __name__ == "__main__":
    main()
