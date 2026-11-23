"""Plot AP and fixed-match continuous-IoU contrasts from closed SAFE results."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


SERIES = (("Bounded minus fixed", "#087474", "o", -0.15),
          ("Independent minus bounded", "#9B482D", "s", 0.15))


def points(row, keys, interval_key):
    ap = row["COCO_points"]["AP"]
    iou = keys["continuous_iou_points"]
    lo, hi = iou[interval_key]
    return ap, iou["delta"], lo, hi


def get_data(rsdd, pseg):
    r = json.loads(rsdd.read_text(encoding="utf-8"))
    p = json.loads(pseg.read_text(encoding="utf-8"))
    if (r["status"], p["status"]) != ("RSDD74_PHOENIX_BOUNDED_MATCH_SCORE_COMPLETE",
                                        "PSEG232_PHOENIX_BOUNDED_MATCH_SCORE_COMPLETE"):
        raise ValueError("Final scoring states required")
    rows = []
    for name, label in (("full", "Instance-RSDD · 74"), ("date_separated", "42 · date-separated")):
        view = r["views"][name]
        rows.append((label, [points(view["contrasts"][k], view["geometry"]["paired"][k], "interval_95")
                             for k in ("bounded_minus_paired", "balance_minus_bounded")]))
    for name, label in (("official", "PSeg-SSDD · 232"), ("exclude_exact_fit_overlap", "222 · exact overlaps removed")):
        view = p["views"][name]
        rows.append((label, [points(view["contrasts"][k], view["contrasts"][k]["paired_geometry"], "percentile95")
                             for k in ("phoenix_bounded_match_minus_phoenix_fixed_pair",
                                       "phoenix_balanced_minus_phoenix_bounded_match")]))
    return rows


def draw(rows, output):
    fig, (ap_axis, iou_axis) = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.21, top=0.73, wspace=0.30)
    for ax in (ap_axis, iou_axis):
        for y in (0, 2):
            ax.axhspan(y - 0.49, y + 0.49, color="#EDF3F5", lw=0, zorder=0)
        ax.axvline(0, color="#61727C", lw=0.9, zorder=1)
        ax.set_ylim(3.55, -0.55)
        ax.grid(axis="x", color="#D6DFE4", lw=0.6, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis="both", which="both", length=0, labelsize=8, pad=6, colors="#253744")
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#80909A")
    ap_axis.axvline(0.30, color="#657883", ls="--", lw=0.9, zorder=1)
    ap_axis.set_title("Mask AP", color="#202E38", fontsize=10, loc="left", pad=7)
    iou_axis.set_title("Fixed-match continuous IoU", color="#202E38", fontsize=10, loc="left", pad=7)
    ap_axis.set_yticks(range(len(rows)), [label for label, _ in rows])
    for y, (_, vals) in enumerate(rows):
        for (_, color, marker, offset), (ap, iou, lo, hi) in zip(SERIES, vals):
            ap_axis.plot(ap, y + offset, marker, color=color, ms=5, zorder=3)
            tick = y + offset
            iou_axis.hlines(tick, lo, hi, color=color, lw=1, zorder=2)
            iou_axis.vlines([lo, hi], tick - 0.055, tick + 0.055, color=color, lw=1, zorder=2)
            iou_axis.plot(iou, tick, marker, color=color, ms=5, zorder=3)
    ap_values = [scores[0] for _, vals in rows for scores in vals]
    lows = [scores[2] for _, vals in rows for scores in vals]
    highs = [scores[3] for _, vals in rows for scores in vals]
    ap_axis.set_xlim(min(-0.05, min(ap_values) - 0.04), max(0.35, max(ap_values) + 0.04))
    iou_axis.set_xlim(min(-0.05, min(lows) - 0.04), max(0.10, max(highs) + 0.04))
    ap_axis.set_xlabel("AP points", fontsize=8)
    iou_axis.set_xlabel("IoU points", fontsize=8)
    keys = [Line2D([0], [0], marker=marker, color=color, lw=0, markersize=5, label=name)
            for name, color, marker, _ in SERIES]
    fig.legend(handles=keys, frameon=False, fontsize=8, ncol=2, loc="upper left",
               bbox_to_anchor=(0.265, 0.97), columnspacing=1.15, handletextpad=0.25)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("rsdd", type=Path)
    cli.add_argument("pseg", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    draw(get_data(args.rsdd, args.pseg), args.output)
