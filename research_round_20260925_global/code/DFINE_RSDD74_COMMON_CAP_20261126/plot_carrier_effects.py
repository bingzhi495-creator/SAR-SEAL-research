"""Render full-reserve matched writer effects from published aggregates."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INK = "#182B3A"
TEAL = "#087E8B"
BLUE = "#415C76"


def effect(view, arm, geometry):
    record = view["contrasts"][arm]
    continuous = record[geometry]["continuous_iou_points"]
    return record["ap_points"], continuous["delta"], continuous["interval_95"]


def plot(dfine, point):
    dfine_view = dfine["views"]["full"]
    point_view = point["views"]["full"]
    if not (dfine_view["images"] == point_view["images"] == 74
            and dfine_view["author_hulls"] == point_view["author_hulls"] == 81
            and point_view["historical_learned_metrics_reproduced"]):
        raise ValueError("Expected the two verified evaluations of the full 74-chip reserve")
    rows = (
        (r"RF-DETR · occupancy", f'{point_view["eligible_original_matches"]} fixed matches',
         effect(point_view, "I_minus_L", "geometry"), TEAL, "o"),
        (r"D-FINE · occupancy", f'{dfine_view["eligible_matches"]} fixed matches',
         effect(dfine_view, "I_minus_L", "continuous_and_boundary"), TEAL, "o"),
        (r"RF-DETR · PointRend", f'{point_view["eligible_original_matches"]} fixed matches',
         effect(point_view, "pr_I_minus_L", "geometry"), BLUE, "s"),
    )
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8.6,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rsdd74-dfine-strong-pair-control"})
    fig, (ap_ax, iou_ax) = plt.subplots(1, 2, figsize=(7.15, 2.26))
    fig.subplots_adjust(left=.31, right=.975, bottom=.33, top=.84, wspace=.26)
    for ax, title in ((ap_ax, "Mask AP"), (iou_ax, "Continuous IoU")):
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=7)
        ax.axvline(0, color="#A4B1BA", linestyle=(0, (2.3, 2.4)), linewidth=.9)
        ax.set_ylim(-.55, 2.62)
        ax.set_yticks([])
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B9C6CE")
        ax.tick_params(axis="x", length=3, color="#A4B1BA", labelcolor="#506473")
    for i, (name, label, (ap, cont, interval), color, marker) in enumerate(rows):
        y = 2-i
        ap_ax.text(-.055,y+.062,name,transform=ap_ax.get_yaxis_transform(),color=INK,
                   va="center",ha="right",fontweight="bold",fontsize=8.1)
        ap_ax.text(-.055,y-.25,label,transform=ap_ax.get_yaxis_transform(),color="#637686",
                   va="center",ha="right",fontsize=7.2)
        iou_ax.hlines(y,*interval,color=color,linewidth=1.35)
        for axis, value in ((ap_ax,ap),(iou_ax,cont)):
            axis.scatter(value,y,marker=marker,s=26,edgecolor=color,
                         facecolor=color if marker=="o" else "white",linewidth=1,zorder=4)
    ap_ax.set_xlim(-.04,.55)
    iou_ax.set_xlim(-.21,.29)
    fig.text(.64,.112,"Independent minus bounded re-pairing (percentage points)",
             ha="center",va="center",color=INK)
    fig.text(.64,.045,"Continuous IoU intervals use paired sensor–date resampling.",
             ha="center",va="center",color="#637686",fontsize=7.1)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine", type=Path, required=True)
    parser.add_argument("--crossover", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in (args.dfine,args.crossover)]
    fig = plot(*results)
    for suffix in ("pdf", "svg"):
        fig.savefig(args.out / f"full_reserve_carrier_effects.{suffix}")


if __name__ == "__main__":
    main()
