"""Plot aggregate overlap changes for two closed RF-DETR DEV705 cases."""

import argparse
import json
from math import isclose
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator


INK = "#24313A"
FIXED = "#697780"
INDEPENDENT = "#254F70"
GRID = "#DCE2E5"


def load_cases(source):
    cases = json.loads(source.read_text(encoding="utf-8"))["cases"]
    if [case["sign"] for case in cases] != ["positive", "negative"]:
        raise ValueError("Expected one positive and one negative case")
    for case in cases:
        states = case["arms"]
        raw, fixed, independent = (states[name] for name in ("N", "SF", "SU=F"))
        count = fixed["adds"]
        if count <= 0:
            raise ValueError("The two methods must make nonzero edits")
        for arm in (fixed, independent):
            if arm["adds"] != count or arm["drops"] != count:
                raise ValueError("Unequal addition and removal counts")
            if (arm["P"], arm["T"]) != (raw["P"], raw["T"]):
                raise ValueError("Area differs from the original match")
            net = arm["add_overlap"] - arm["drop_overlap"]
            if not isclose(arm["I"] - raw["I"], net, abs_tol=1e-8):
                raise ValueError("Overlap contributions do not close")
    return cases


def render(cases, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "SAR-SEAL-DEV705-CASES-20261121",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), sharey=True)
    fig.subplots_adjust(left=.093, right=.985, bottom=.18, top=.69, wspace=.20)
    fig.legend(handles=[Patch(facecolor=FIXED, label="Fixed pairing (SF)"),
                        Patch(facecolor=INDEPENDENT, label="Independent (SU=F)")],
               loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False)
    axes[0].set_ylabel("Intersection change\n(pixel area)", labelpad=5)

    bounds = [0.0]
    for axis, case, label in zip(axes, cases, ("(a)  Improved", "(b)  Degraded")):
        fixed = case["arms"]["SF"]
        independent = case["arms"]["SU=F"]
        axis.set_title(f"{label}  |  {fixed['adds']} swaps each", loc="left", pad=21,
                       fontsize=9, fontweight="bold")
        delta = 100 * (independent["iou"] - fixed["iou"])
        comparison = f"IoU  {100 * fixed['iou']:.2f} → {100 * independent['iou']:.2f}  ({delta:+.2f} points)"
        axis.text(0, 1.08, comparison.replace("-", "−"), transform=axis.transAxes)
        for offset, arm, color in ((-.17, fixed, FIXED), (.17, independent, INDEPENDENT)):
            added, removed = arm["add_overlap"], arm["drop_overlap"]
            values = [added, -removed, added - removed]
            bounds.extend(values)
            axis.bar([x + offset for x in range(3)], values, width=.29, color=color, zorder=3)
        axis.set_xticks(range(3), ["Added", "Removed", "Net"])
        axis.set_xlim(-.46, 2.46)
        axis.set_axisbelow(True)
        axis.grid(axis="y", lw=.6, color=GRID)
        axis.axhline(0, lw=.75, color=FIXED, zorder=2)
        axis.tick_params(axis="both", length=0, labelsize=8)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        for spine in axis.spines.values():
            spine.set_visible(False)

    margin = .11 * (max(bounds) - min(bounds))
    axes[0].set_ylim(min(bounds) - margin, max(bounds) + margin)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text():
            corners = label.get_window_extent(renderer).get_points()
            if not all(fig.bbox.contains(x, y) for x, y in corners):
                raise ValueError(f"Label outside the canvas: {label.get_text()}")

    stem = output / "dev705_overlap_cases"
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=directory / "dev705_overlap_cases.json")
    parser.add_argument("--output-dir", type=Path, default=directory)
    args = parser.parse_args()
    render(load_cases(args.input), args.output_dir)
