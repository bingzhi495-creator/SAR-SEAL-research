"""Render an editable publication figure from pooled SAR-SSP statistics only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


INK = "#172535"
MUTED = "#526478"
LINE = "#dde6ed"
POS = "#147c75"
NEG = "#aa5524"
TIE = "#9ba9b9"
BLUE = "#376bb0"


def signed(value, places):
    return f"{value:+.{places}f}".replace("-", "−")


def panel_title(figure, x, y, letter, title):
    figure.text(x, y, f"({letter})  {title}", fontsize=8.8, weight="bold", va="top")


def distribution_panel(figure, data):
    panel_title(figure, .047, .854, "a", "Changed native matches")
    axis = figure.add_axes((.101, .524, .370, .264))
    values = data["distribution"]["counts"]
    assert sum(values) == data["changed"] == 52
    positions = np.arange(len(values))
    bars = axis.bar(positions, values, width=.73, color=([NEG]*4 + [TIE] + [POS]*4), zorder=3)
    for bar, count in zip(bars, values, strict=True):
        if count:
            axis.text(bar.get_x() + bar.get_width()/2, count + .55, str(count),
                      ha="center", va="bottom", color=INK, fontsize=6.5, weight="bold")
    axis.set_xlim(-.58, 8.58)
    axis.set_ylim(0, max(values) + 4)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    axis.set_xticks(positions, ("<−4", "−4…−2", "−2…−1", "−1…0", "0", "0…1", "1…2", "2…4", ">4"))
    axis.tick_params(axis="x", labelsize=7.6, length=0, pad=3, labelrotation=24)
    axis.tick_params(axis="y", labelsize=7.6, length=0, pad=3)
    axis.set_xlabel("Independent − distance-paired discrete IoU (pp)", fontsize=7.8, labelpad=5)
    axis.set_ylabel("Hull count", fontsize=7.8, labelpad=4)
    axis.grid(axis="y", color=LINE, linewidth=.6, zorder=0)
    for spine in ("left", "top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(LINE)


def spatial_panel(figure, data):
    panel_title(figure, .543, .854, "b", "Pooled spatial disagreement")
    axis = figure.add_axes((.605, .521, .205, .274))
    color_axis = figure.add_axes((.829, .537, .016, .242))
    grid = np.asarray([[np.nan if v is None else v for v in row] for row in data["spatial"]["grid"]], dtype=float)
    assert grid.shape == (4, 4) and data["spatial"]["supporting_hulls_minimum"] >= 5
    colors = LinearSegmentedColormap.from_list("sign", ["#a84d24", "#fbedde", "#f7fafb", "#d8f1ec", "#137e76"])
    colors.set_bad("#e3e8ed")
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    mesh = axis.pcolormesh(np.arange(5)/4, np.arange(5)/4, grid, cmap=colors, norm=norm,
                          edgecolors="white", linewidth=.9)
    axis.set_aspect("equal")
    axis.invert_yaxis()
    for iy, ix in np.ndindex(grid.shape):
        value = grid[iy, ix]
        axis.text((ix + .5)/4, (iy + .5)/4, signed(value, 0) if np.isfinite(value) and value != 0
                  else ("0" if value == 0 else "–"), ha="center", va="center",
                  color="white" if abs(value) >= 29 else INK, fontsize=7.1, weight="bold")
    axis.set_xticks((0, .5, 1), ("0", "0.5", "1"))
    axis.set_yticks((0, .5, 1), ("0", "0.5", "1"))
    axis.tick_params(length=0, labelsize=7.6, pad=2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    cbar = figure.colorbar(mesh, cax=color_axis, ticks=(-50, 0, 50))
    cbar.ax.tick_params(labelsize=7.6, length=0, pad=2)
    cbar.outline.set_visible(False)
    cbar.solids.set_rasterized(False)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel("Net correct labels (%)", fontsize=6.8, labelpad=3)
    figure.text(.71, .477, "Relative image-axis position in hull box", ha="center", color=MUTED, fontsize=6.8)
    figure.text(.71, .445, "52 changed hulls pooled · ≥5 hulls per cell", ha="center",
                color=MUTED, fontsize=6.6)


def effect_panel(figure, data):
    panel_title(figure, .047, .378, "c", "Fixed-pair geometry and full-validation AP")
    axis = figure.add_axes((.334, .117, .398, .190))
    axis.set_xlim(-.68, .68)
    axis.set_ylim(-.55, 2.48)
    axis.set_yticks([])
    axis.set_xticks((-.6, -.3, 0, .3, .6), ("−0.6", "−0.3", "0", "+0.3", "+0.6"))
    axis.tick_params(length=0, pad=3, labelsize=6.8)
    axis.set_xlabel("Source-scene-balanced ΔIoU (pp), paired 95% CI", fontsize=6.9, labelpad=4)
    axis.axvline(0, color=TIE, linewidth=.85, linestyle=(0, (3, 3)), zorder=1)
    for spine in ("left", "top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(LINE)
    names = ("Independent − distance-paired", "Learned − dynamic attraction", "Learned − static attraction")
    figure.text(.861, .313, "Δ AP (pp)", ha="center", va="bottom", fontsize=7.15, weight="bold")
    for y, name, row in zip((2, 1, 0), names, data["comparisons"], strict=True):
        value = row["iou_delta_pp"]
        lo, hi = row["iou_ci95_pp"]
        color = BLUE if y == 2 else MUTED
        axis.errorbar(value, y, xerr=[[value-lo], [hi-value]], marker="o", markersize=4.2,
                      capsize=2.8, elinewidth=1.23, color=color, zorder=3)
        fig_y = figure.transFigure.inverted().transform(axis.transData.transform((0, y)))[1]
        figure.text(.051, fig_y, name, va="center", fontsize=7.3, weight="bold" if y == 2 else "normal")
        figure.text(.861, fig_y, signed(row["ap_delta_pp"], 3), va="center", ha="center",
                    color=color, fontsize=7.2, weight="bold" if y == 2 else "normal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.aggregate.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 200,
                         "savefig.dpi": 300, "figure.facecolor": "white"})
    figure = plt.figure(figsize=(7.12, 4.40))
    figure.text(.047, .964, "Boundary edits on SAR-SSP author validation", fontsize=12.4,
                weight="bold", va="top")
    figure.text(.047, .909,
                f"{data['images']} chips · {data['hulls']} author hulls · {data['matched']} fixed native matches"
                f" · {data['changed']} changed matches",
                fontsize=7.8, color=MUTED, va="top")
    distribution_panel(figure, data)
    spatial_panel(figure, data)
    effect_panel(figure, data)
    for extension in ("pdf", "svg"):
        figure.savefig(args.output / f"sarssp_public_summary.{extension}", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
