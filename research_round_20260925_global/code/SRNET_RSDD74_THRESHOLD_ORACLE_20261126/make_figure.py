"""Build an editable figure from aggregate SRNet diagnostic evidence."""
from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VIEWS = (
    ("full74", "74 images", "#165b74"),
    ("nested42_sensor_date", "Nested 42", "#48a6ac"),
)
MEASURED = "free_matched_count minus fixed"
HEADROOM = "oracle minus free_matched_count"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True, help="Output stem with an existing parent")
    args = parser.parse_args()
    ap_file = args.input_dir / "AP_AGGREGATE_SAFE.json"
    geometry_file = args.input_dir / "GEOMETRY_ORACLE_SAFE.json"
    ap = json.loads(ap_file.read_text(encoding="utf-8"))["views"]
    geo = json.loads(geometry_file.read_text(encoding="utf-8"))["views"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "text.color": "#1f292d",
        "axes.labelcolor": "#1f292d",
        "xtick.color": "#394950",
        "ytick.color": "#394950",
        "svg.fonttype": "none",
        "svg.hashsalt": "srnet74-threshold-oracle",
        "pdf.fonttype": 42,
    })
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.35, 2.48), width_ratios=[1.06, 1.0], layout="constrained"
    )
    thresholds = np.arange(50, 100, 5) / 100
    columns = np.arange(len(thresholds))
    for idx, (view, label, color) in enumerate(VIEWS):
        panel = ap[view]["by_threshold"]
        delta = [panel[f"{value:.2f}"]["contrasts"][MEASURED]["ap_delta_points"]
                 for value in thresholds]
        left.bar(columns + (idx - .5) * .35, delta, .32, label=label, color=color, zorder=3)
    left.axhline(0, color="#536970", linewidth=.75)
    left.set_xticks(columns, [f"{threshold:.2f}" for threshold in thresholds], fontsize=7)
    left.set_yticks([0, 1, 2])
    left.set_ylim(-.32, 2.47)
    left.set_xlabel("COCO IoU threshold")
    left.set_ylabel("Mask AP difference (pp)")
    left.set_title("(a) Independent minus paired", loc="left", fontweight="bold", pad=8)
    left.legend(loc="upper left", frameon=False, borderaxespad=.25, handlelength=1.2)
    left.grid(axis="y", color="#d9e2e5", linewidth=.5, zorder=0)

    right.set_title("(b) Fixed-match continuous IoU", loc="left", fontweight="bold", pad=8)
    right.axvline(0, color="#536970", linewidth=.75)
    right.set_yticks([2.69, .69], ["Independent\nminus paired", "Oracle minus\nindependent"], fontsize=7.6)
    right.set_ylim(-.05, 3.55)
    right.set_xlim(-.05, .91)
    right.set_xticks([0, .2, .4, .6, .8])
    right.set_xlabel("Continuous IoU difference (pp)")
    right.grid(axis="x", color="#d9e2e5", linewidth=.5, zorder=0)
    right.tick_params(axis="y", length=0)
    for contrast_index, (contrast, marker) in enumerate(((MEASURED, "o"), (HEADROOM, "s"))):
        for view_index, (view, _, color) in enumerate(VIEWS):
            values = geo[view]["contrasts_points"][contrast]["continuous_iou_points"]
            mean = values["delta"]
            low, high = values["sensor_date_interval_95"]
            y = 2.96 - 2 * contrast_index - .54 * view_index
            right.errorbar(mean, y, xerr=[[mean-low], [high-mean]], color=color,
                           marker=marker, markersize=4.1, linewidth=1.4, capsize=2, zorder=5)

    for file_type in ("svg", "pdf", "png"):
        fig.savefig(args.output.with_suffix("." + file_type), dpi=260, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

