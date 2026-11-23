"""Plot the four Phoenix support/budget policies from the two completed aggregate score files.

python plot_cross_dataset_phoenix_budget.py --rsdd RSDD.json --pseg PSEG.json --output phoenix_budget_by_dataset
The given output stem identifies both the vector PDF and the editable SVG.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SUPPORTS = (
    (("nt", "nc"), "Sampled boundary", "#16496D", "o", "-"),
    (("wt", "wc"), "Whole grid", "#A74421", "s", "--"),
)


def number(value, upper, description):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(f"Invalid {description}: {value!r}")
    return value


def load_cohort(path, dataset):
    result = json.loads(path.read_text(encoding="utf-8"))
    rsdd = dataset == "Instance-RSDD"
    tag, primary, census = ("RSDD74", "full", (74, 47)) if rsdd else ("PSEG232", "official", (232, 293))
    if (result.get("status"), result.get("phase"), result.get("primary_view")) != (
            tag + "_PHOENIX_SUPPORT_BUDGET_SCORE_COMPLETE", "score", primary):
        raise ValueError(f"Expected the completed {dataset} score with primary view {primary!r}.")
    view = result["views"][primary]
    geometry = view["geometry"]
    if rsdd:
        iou_rows, hulls = geometry["group_equal_arms"], geometry["counts"]["eligible_matches"]
    else:
        iou_rows, hulls = geometry["arms"], geometry["eligible_pairs"]
    if (view["images"], hulls) != census:
        raise ValueError(f"Unexpected image or eligible fixed-match count for {dataset}.")
    ap, iou = {}, {}
    for arms, *_ in SUPPORTS:
        for arm in arms:
            ap[arm] = 100 * number(view["metrics_fraction"][arm]["AP"], 1, f"{dataset} {arm} AP fraction")
            iou[arm] = number(iou_rows[arm]["continuous_iou_points"], 100, f"{dataset} {arm} continuous IoU points")
    return ap, iou, view["images"], hulls


def draw_panel(ax, values, title, detail):
    ink, muted = "#182735", "#4D5E68"
    ax.set_title(title, loc="left", pad=6, fontsize=9, fontweight="bold", color=ink)
    ax.set_title(detail, loc="right", pad=6, fontsize=8, color=muted)
    lowest, highest = min(values.values()), max(values.values())
    center = (lowest + highest) / 2
    half_range = max(0.5, 0.68 * (highest - lowest))
    lower, upper = center - half_range, center + half_range
    ax.set_xlim(-0.43, 1.43)
    ax.set_ylim(lower, upper)
    ticks = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
    ticks = [t for t in ticks if max(0, lower) <= t <= min(100, upper)]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xticks([0, 1], ["Author threshold", "Exact 2k flips"])
    ax.tick_params(length=0, colors=muted, pad=4, labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D7DDE1", linewidth=0.5)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#A0ADB5")
    ax.spines["bottom"].set_linewidth(0.55)

    for arms, label, color, marker, style in SUPPORTS:
        ax.plot([0, 1], [values[arm] for arm in arms], label=label, color=color, linestyle=style,
                marker=marker, linewidth=1.35, markersize=(4.2 if marker == "o" else 5.2),
                markerfacecolor=(color if marker == "o" else "none"), markeredgewidth=1.1)

    # Place number labels on the outward side of each endpoint. Separate exact or close ties.
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    for x in (0, 1):
        labels = [(values[arms[x]], color) for arms, _, color, _, _ in SUPPORTS]
        gap = abs(labels[0][0] - labels[1][0]) * height_pt / (2 * half_range)
        offset = max(0, (9.8 - gap) / 2)
        higher = 0 if labels[0][0] >= labels[1][0] else 1
        for i, (value, color) in enumerate(labels):
            ax.annotate(f"{value:.3f}", (x, value), xytext=((-5.5 if x == 0 else 5.5), (offset if i == higher else -offset)),
                        textcoords="offset points", ha=("right" if x == 0 else "left"), va="center", fontsize=8,
                        color=color, bbox=dict(facecolor="white", edgecolor="none", pad=0.45))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--rsdd", type=Path, required=True, help="Completed Instance-RSDD 74-image aggregate score JSON.")
    p.add_argument("--pseg", type=Path, required=True, help="Completed PSeg-SSDD 232-image aggregate score JSON.")
    p.add_argument("--output", type=Path, required=True, help="Output filename stem, or the corresponding .pdf/.svg filename.")
    args = p.parse_args()
    datasets = ("Instance-RSDD", "PSeg-SSDD")
    tables = [load_cohort(path, dataset) for path, dataset in zip((args.rsdd, args.pseg), datasets)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "phoenix-budget-by-dataset", "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.35), sharex="col")
    fig.subplots_adjust(left=0.082, right=0.986, top=0.82, bottom=0.105, wspace=0.22, hspace=0.55)
    weights = ("Equal parent weight", "Equal image-group weight")
    for j, (ap, iou, images, hulls) in enumerate(tables):
        draw_panel(axes[0,j], ap, f"({chr(97+j)})  {datasets[j]}", f"{images} images")
        draw_panel(axes[1,j], iou, f"({chr(99+j)})  {hulls} fixed hulls", weights[j])
    axes[0,0].set_ylabel("Mask AP (points)", fontsize=8, color="#4D5E68")
    axes[1,0].set_ylabel("Continuous IoU (points)", fontsize=8, color="#4D5E68")
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 0.988),
               frameon=False, fontsize=8.5, handlelength=2.0, handletextpad=0.55, columnspacing=1.6)
    stem = args.output.with_suffix("") if args.output.suffix.lower() in (".svg", ".pdf") else args.output
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, metadata in (("svg", {"Date": None}), ("pdf", {"CreationDate": None, "ModDate": None})):
        fig.savefig(stem.parent / (stem.name + "." + ext), metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()
