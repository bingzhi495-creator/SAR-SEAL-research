"""Plot the exact fixed-total-flip contrast from released results."""
from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(source, destinations):
    result = json.loads(source.read_text(encoding="utf-8"))
    assert result["exact_all_archive_score_parity"]
    navy, orange, ink = "#174567", "#A6532C", "#142634"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.titleweight": "bold", "axes.labelsize": 8, "text.color": ink,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
    })
    fig, (effect, parts) = plt.subplots(1, 2, figsize=(7.05, 2.42), gridspec_kw={"wspace": .19})
    fig.subplots_adjust(left=.13, right=.98, bottom=.235, top=.85)
    for row, ref in enumerate(("spsd", "pseg")):
        stat = result["references"][ref]["variable_iou_minus"]["fixed_iou"]
        terms = result["geometry_decomposition"][ref]["fixed_iou"]["group_means"]
        value, (lo, hi) = stat["iou_points"], stat["percentile95_points"]
        overlap, area = terms["overlap_iou_points"], terms["area_iou_points"]
        assert abs(overlap + area - value) < 1e-8
        effect.errorbar(value, row, xerr=[[value-lo], [hi-value]], fmt="o", color=navy, capsize=2,
                        markersize=4.3, elinewidth=1.25, zorder=3)
        effect.text(value, row+.14, f"{value:+.3f}", ha="center", va="bottom", color=navy)
        effect.text(1.4, row, f"AP {stat['ap_points']:+.3f}", va="center", fontsize=7.6)
        parts.barh(row+.12, overlap, height=.17, color=navy,
                   label="Added intersection" if row == 0 else None, zorder=3)
        parts.barh(row-.12, area, height=.17, color=orange,
                   label="Net area change" if row == 0 else None, zorder=3)
        parts.text(overlap+.07, row+.12, f"{overlap:+.2f}", va="center", color=navy, fontsize=7.6)
        parts.text(area-.07, row-.12, f"{area:+.2f}", ha="right", va="center", color=orange, fontsize=7.6)
    for ax in (effect, parts):
        ax.set(yticks=(0,1), ylim=(-.41,1.38))
        ax.tick_params(length=0)
        ax.grid(axis="x", color="#E2E9ED", linewidth=.64)
        ax.axvline(0, color="#7D919D", linewidth=.82)
        for spine in ax.spines.values():
            spine.set_visible(False)
    effect.set_yticklabels(("SPSD", "PSeg-SSDD"))
    effect.set(xlim=(-.11, 1.91), xticks=(0,.5,1,1.5), xlabel="IoU change (points)")
    parts.set(xlim=(-2.05,2.93), xticks=(-2,-1,0,1,2), xlabel="Contribution to IoU change (points)")
    parts.tick_params(labelleft=False)
    effect.set_title("(a) Variable versus equal add/drop counts", loc="left", pad=8)
    parts.set_title("(b) Intersection and area contributions", loc="left", pad=8)
    parts.legend(loc="lower left", bbox_to_anchor=(0,-.36), ncol=2,
                 frameon=False, handlelength=1.25, columnspacing=1, fontsize=7.3)
    for path in destinations:
        for suffix in ("pdf","svg"):
            fig.savefig(path / f"dual_reference_budget_effect.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--figures", type=Path, nargs="+", required=True)
    args = p.parse_args()
    draw(args.score, args.figures)
