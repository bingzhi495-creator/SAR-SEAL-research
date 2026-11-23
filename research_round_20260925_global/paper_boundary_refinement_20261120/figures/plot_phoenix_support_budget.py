"""Plot the four Phoenix write policies from the completed factorial score JSON.

Example: python plot_phoenix_support_budget.py --results RESULT_SAFE.json --output phoenix_support_budget
The output stem identifies both the vector PDF and the editable-text SVG.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


STATUS = "RSDD74_PHOENIX_SUPPORT_BUDGET_SCORE_COMPLETE"
# In each row, threshold comes first, then the candidate's exact 2k flips.
SUPPORTS = (
    (("nt", "nc"), "Sampled boundary", "#16496D", "o", "-"),
    (("wt", "wc"), "Whole grid", "#A74421", "s", "--"),
)


def number(value, upper, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(f"Missing or out-of-range {name}: {value!r}")
    return value


def read_score(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("status"), result.get("phase"), result.get("primary_view")) != (STATUS, "score", "full"):
        raise ValueError("Expected the completed Phoenix support/budget score with primary view 'full'.")
    full = result["views"]["full"]
    geometry = full["geometry"]
    images, hulls = full["images"], geometry["counts"]["eligible_matches"]
    if (images, hulls) != (74, 47):
        raise ValueError("Expected the primary 74-image view and its 47 eligible native matches.")
    ap, iou = {}, {}
    for arms, *_ in SUPPORTS:
        for arm in arms:
            ap[arm] = 100 * number(full["metrics_fraction"][arm]["AP"], 1, f"{arm} AP fraction")
            iou[arm] = number(geometry["group_equal_arms"][arm]["continuous_iou_points"], 100, f"{arm} IoU points")
    return (("(a)  Mask AP", f"{images} images", ap),
            ("(b)  Fixed continuous IoU", f"{hulls} matched hulls", iou))


def draw_panel(ax, title, population, values):
    ink, muted = "#182735", "#4D5E68"
    ax.set_title(title, loc="left", pad=6, fontsize=9, fontweight="bold", color=ink)
    ax.set_title(population, loc="right", pad=6, fontsize=8, color=muted)
    low, high = min(values.values()), max(values.values())
    center = (low + high) / 2
    half_range = max(0.5, 0.67 * (high - low))
    ax.set_ylim(center - half_range, center + half_range)
    ax.set_xlim(-0.43, 1.43)
    ticks = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(*ax.get_ylim())
    ticks = [t for t in ticks if 0 <= t <= 100 and center - half_range <= t <= center + half_range]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_ylabel("Points", fontsize=8, color=muted, labelpad=3)
    ax.set_xticks([0, 1], ["Threshold", "Exact 2k flips"])
    ax.tick_params(length=0, colors=muted, pad=4, labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D7DDE1", linewidth=0.5)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#A0ADB5")
    ax.spines["bottom"].set_linewidth(0.55)

    for arms, label, color, marker, style in SUPPORTS:
        ax.plot([0, 1], [values[arm] for arm in arms], label=label, color=color,
                linestyle=style, marker=marker, linewidth=1.35, markersize=(4.2 if marker == "o" else 5.2),
                markerfacecolor=(color if marker == "o" else "none"), markeredgewidth=1.1)

    # Put labels outside the lines. Separate labels vertically when values coincide.
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    for x in (0, 1):
        rows = [(values[arms[x]], color) for arms, _, color, _, _ in SUPPORTS]
        gap_pt = abs(rows[0][0] - rows[1][0]) * height_pt / (2 * half_range)
        separation = max(0, (9.8 - gap_pt) / 2)
        for index, (value, color) in enumerate(rows):
            upper = index == (0 if rows[0][0] >= rows[1][0] else 1)
            offset_y = separation if upper else -separation
            ax.annotate(f"{value:.3f}", (x, value),
                        xytext=((-5.5 if x == 0 else 5.5), offset_y), textcoords="offset points",
                        ha=("right" if x == 0 else "left"), va="center", color=color, fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.45))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Completed aggregate score JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output filename stem (or .svg/.pdf).")
    args = parser.parse_args()
    panels = read_score(args.results)
    stem = args.output
    if stem.suffix.lower() in (".svg", ".pdf"):
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "phoenix-support-budget", "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.35), sharex=True)
    fig.subplots_adjust(left=0.165, right=0.97, top=0.835, bottom=0.108, hspace=0.56)
    for ax, panel in zip(axes, panels):
        draw_panel(ax, *panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.53, 0.99),
               frameon=False, fontsize=8, handlelength=2, handletextpad=0.55, columnspacing=1.6)
    for ext, metadata in (("svg", {"Date": None}), ("pdf", {"CreationDate": None, "ModDate": None})):
        fig.savefig(stem.parent / f"{stem.name}.{ext}", metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()
