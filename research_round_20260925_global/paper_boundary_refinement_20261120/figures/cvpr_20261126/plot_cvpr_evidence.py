"""Draw the fixed-count effect ledger and the SAR-SSP outcome panels.

All measurements are read from immutable, released aggregate score files.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


INK = "#243247"
BLUE = "#2D5E91"
TEAL = "#007F83"
ORANGE = "#C4552D"
SLATE = "#5C6675"
RULE = "#DCE5EE"
WASH = "#F3F7FA"
WHITE = "#FFFFFF"

SCORES = {
    "dev": ("LEARNED_BOUNDED_PAIRING_20261126/hrsid/SCORE_SAFE.json",
            "964260bd342ae132d14a95a6352e5cf9b904c3f120e288f264f2d2235cb1827f"),
    "gf3": ("LEARNED_BOUNDED_PAIRING_20261126/rsdd/gf3_score_safe.json",
            "e04e07f1d25bb9ad4f40d98062196700cfb3109b18c657784dfb4a5df933f967"),
    "reserve": ("LEARNED_BOUNDED_PAIRING_20261126/rsdd/reserve74_score_safe.json",
                "648f9509bf889d11da57a5680e14dee95438f0318a32a93eae0085cab6a1c97c"),
    "dfine": ("DFINE_RSDD74_COMMON_CAP_20261126/SCORE_SAFE.json",
              "9176c4ba506549025ea981396cbd71633333d62e3e063d468c17817ef6b8d354"),
    "pseg": ("PSEG232_DFINE_COMMON_CAP_20261126/SCORE_SAFE.json",
             "d4eec6b0fd9898304c6a345b21ee231810dce89e5a4b6be1dd99983c74241544"),
}
SARSSP_SHA = "ba266ae84e368f7b2fd109e2f104460ce54d5b80b9231c6155d98511d6e72949"
CONTRASTS = ("L_minus_F", "I_minus_L")


@dataclass(frozen=True)
class Effect:
    ap: float
    iou: float
    ci: tuple

    def __post_init__(self):
        lo, hi = self.ci
        if not np.isfinite([self.ap, self.iou, lo, hi]).all() or not lo <= self.iou <= hi:
            raise ValueError("Unexpected effect estimate or interval")


def load_score(path, expected):
    content = path.read_bytes()
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"Unexpected released score digest: {path.name}")
    return json.loads(content)


def effects(root):
    scores = {key: load_score(root / path, digest) for key, (path, digest) in SCORES.items()}
    dev = scores["dev"]

    def make(row, ap, geo, interval):
        return Effect(row[ap], geo["delta"], tuple(geo[interval]))

    def rf_rows(view):
        return tuple(make(row, "ap_points", row["geometry"]["continuous_iou_points"], "interval_95")
                     for row in (view["pooled_contrasts"][name] for name in CONTRASTS))

    def df_rows(view):
        return tuple(make(row, "ap_points", row["continuous_and_boundary"]["continuous_iou_points"], "interval_95")
                     for row in (view["contrasts"][name] for name in CONTRASTS))

    def ps_rows(view):
        return tuple(make(row, "ap_points", row["geometry"]["continuous_iou_points"], "percentile95")
                     for row in (view["contrasts"][key]
                                 for key in ("bounded minus fixed", "free_matched_count minus bounded")))

    dev_measures = tuple(make(row, "AP_points", row["metrics"]["iou_points"], "interval95")
                         for row in (dev["contrasts"][name] for name in CONTRASTS))
    rows = [
        ("HRSID DEV705", "RF-DETR 5202 · 3 heads", dev["fixed_matching_counts"]["valid"], dev_measures),
        ("Instance-RSDD Gaofen-3", "RF-DETR 5202 · 2 heads",
         scores["gf3"]["views"]["gf3"]["eligible_original_matches"], rf_rows(scores["gf3"]["views"]["gf3"])),
    ]
    for view_key, title in (("full", "Instance-RSDD reserve · 74"), ("date_separated", "  Nested reserve · 42")):
        rf = scores["reserve"]["views"][view_key]
        df = scores["dfine"]["views"][view_key]
        rows.extend([(title, "RF-DETR 5202", rf["eligible_original_matches"], rf_rows(rf)),
                     ("", "D-FINE E24", df["eligible_matches"], df_rows(df))])
    for key, title in (("official", "PSeg-SSDD official · 232"), ("exclude_exact_fit_overlap", "  Deduplicated · 222")):
        ps = scores["pseg"]["views"][key]
        rows.append((title, "D-FINE E24", ps["fixed_geometry"]["eligible_pairs"], ps_rows(ps)))
    if [row[2] for row in rows] != [1461, 425, 47, 44, 32, 32, 293, 279]:
        raise ValueError("Unexpected fixed matched populations")
    return rows


def clean_axis(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(RULE)
    axis.tick_params(axis="both", length=0, pad=4, labelsize=7)


def panel_title(fig, x, y, letter, text, detail=""):
    fig.text(x, y, f"({letter})  {text}", weight="bold", fontsize=8.5, va="top")
    if detail:
        fig.text(x, y - .044, detail, fontsize=6.8, color=SLATE, va="top")


def save(fig, out, stem):
    fig.savefig(out / (stem + ".pdf"), metadata={"Creator": "SAR-SEAL",
                "CreationDate": None, "ModDate": None}, facecolor=WHITE)
    fig.savefig(out / (stem + ".svg"), metadata={"Creator": "SAR-SEAL", "Date": None}, facecolor=WHITE)
    fig.savefig(out / (stem + ".png"), dpi=145, facecolor=WHITE)
    plt.close(fig)


def draw_overview(rows, output):
    fig = plt.figure(figsize=(7.16, 3.77), facecolor=WHITE)
    left, bottom, height = .032, .139, .736
    ax_text = fig.add_axes((left, bottom, .315, height))
    ax_ap = fig.add_axes((.365, bottom, .250, height))
    ax_iou = fig.add_axes((.662, bottom, .309, height))
    for axis in (ax_text, ax_ap, ax_iou):
        axis.set_ylim(-.5, 7.5)
        axis.invert_yaxis()
        axis.set_yticks([])
        axis.axhspan(1.5, 5.5, color=WASH, ec="none", zorder=0)
        for boundary in (.5, 1.5, 5.5):
            axis.axhline(boundary, color=RULE, linewidth=.68, zorder=1)
    ax_text.axis("off")
    ax_ap.set(xlim=(-.135, .975), xticks=[0, .3, .6, .9])
    ax_iou.set(xlim=(-.10, .49), xticks=[0, .1, .2, .3, .4])
    ax_ap.set_xticklabels(["0", "0.30", "0.60", "0.90"])
    ax_iou.set_xticklabels(["0", "0.10", "0.20", "0.30", "0.40"])
    for axis in (ax_ap, ax_iou):
        clean_axis(axis)
        axis.grid(axis="x", color=RULE, linewidth=.55, zorder=0)
        axis.set_axisbelow(True)
        axis.axvline(0, color=SLATE, linewidth=.85, alpha=.8, zorder=2)
    ax_ap.axvline(.30, color=SLATE, linestyle=(0, (2, 2)), linewidth=.95, zorder=2)
    fig.text(left, .958, "Cohort / frozen pipeline", weight="bold", fontsize=8.5)
    panel_title(fig, .365, .982, "a", "Mask AP", "Difference (pp)")
    panel_title(fig, .662, .982, "b", "Continuous IoU", "Difference (pp) with paired 95% interval")
    for j, (title, pipeline, matches, measures) in enumerate(rows):
        if title:
            ax_text.text(.0, j - .13, title, color=INK, fontsize=7.35, weight="semibold", va="center")
        ax_text.text(.015 if title else .075, j + .19, f"{pipeline} · {matches:,} matches", color=SLATE,
                     fontsize=6.95, va="center")
        for m, yy, color, marker in zip(measures, (j-.148, j+.148), (SLATE, BLUE), ("s", "o"), strict=True):
            ax_ap.scatter(m.ap, yy, s=21, c=color, marker=marker, linewidth=.6, edgecolors=WHITE, zorder=5)
            lo, hi = m.ci
            ax_iou.errorbar(m.iou, yy, xerr=[[m.iou-lo], [hi-m.iou]], fmt=marker, markersize=3.9,
                           color=color, linewidth=1.24, capsize=2.25, zorder=4)
    handles = [Line2D([], [], marker="s", color=SLATE, lw=0, markersize=4.4,
                      label="Bounded − distance-first (L−F)"),
               Line2D([], [], marker="o", color=BLUE, lw=0, markersize=4.4,
                      label="Independent − bounded (I−L)"),
               Line2D([], [], color=SLATE, linestyle=(0, (2, 2)), lw=.95,
                      label="AP criterion on positive side")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.024, .018),
               ncol=3, fontsize=6.9, frameon=False, handlelength=1.45, columnspacing=1.3)
    save(fig, output, "transfer_effects")


def signed(x, digits):
    return f"{x:+.{digits}f}".replace("-", "−")


def draw_sarssp(data, output):
    fig = plt.figure(figsize=(7.12, 4.02), facecolor=WHITE)
    hist = data["distribution"]
    panel_title(fig, .030, .976, "a", "IoU distribution", f'{data["changed"]} changed of {data["matched"]} native matches')
    panel_title(fig, .530, .976, "b", "Which labels became correct?", "Disagreeing pixels pooled in reference-box coordinates")
    ax = fig.add_axes([.088, .507, .377, .299])
    positions = np.arange(len(hist["counts"]))
    if sum(hist["counts"]) != data["changed"]:
        raise ValueError("Distribution denominator mismatch")
    bars = ax.bar(positions, hist["counts"], width=.68, color=[ORANGE]*4+[SLATE]+[TEAL]*4, zorder=3)
    for rectangle, value in zip(bars, hist["counts"], strict=True):
        if value:
            ax.text(rectangle.get_x() + rectangle.get_width()/2, value+.55, str(value),
                    ha="center", va="bottom", color=INK, fontsize=6.45)
    clean_axis(ax)
    ax.set(xlim=(-.55, 8.55), ylim=(0, 19.3), ylabel="Hull count",
           xticks=positions)
    ax.set_xticklabels(("<−4", "−4…−2", "−2…−1", "−1…0", "0", "0…1", "1…2", "2…4", ">4"),
                       fontsize=6.4, rotation=27)
    ax.set_xlabel("Independent − distance-paired IoU (pp)", fontsize=7.3, labelpad=4)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="y", color=RULE, linewidth=.62, zorder=0)
    ax.set_axisbelow(True)
    img = fig.add_axes([.576, .503, .239, .302])
    cax = fig.add_axes([.837, .517, .021, .276])
    values = np.array([[np.nan if v is None else v for v in r] for r in data["spatial"]["grid"]], dtype=float)
    if values.shape != (4, 4) or data["spatial"]["supporting_hulls_minimum"] < 5:
        raise ValueError("Unexpected pooled spatial display shape or support")
    scale = LinearSegmentedColormap.from_list("edit_difference", [ORANGE, "#F6E6DE", WHITE, "#DCEFED", TEAL])
    scale.set_bad(WASH)
    pic = img.pcolormesh(np.arange(5)/4, np.arange(5)/4, values, cmap=scale,
                        norm=TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50),
                        edgecolors=RULE, linewidth=.55)
    img.set_aspect("equal")
    img.invert_yaxis()
    img.set_xticks((0, .5, 1), ("0", "0.5", "1"))
    img.set_yticks((0, .5, 1), ("0", "0.5", "1"))
    img.tick_params(length=0, labelsize=6.6, pad=2)
    for i, j in np.ndindex(values.shape):
        v = values[i, j]
        img.text((j+.5)/4, (i+.5)/4, ("—" if not np.isfinite(v) else "0" if v == 0 else signed(v, 0)),
                 color=WHITE if abs(v) >= 34 else INK, fontsize=7.2, weight="semibold", ha="center", va="center")
    img.spines[:].set_visible(False)
    cbar = fig.colorbar(pic, cax=cax, ticks=(-50, 0, 50))
    cbar.ax.set_yticklabels(["−50", "0", "+50"])
    cbar.ax.set_ylabel("Net correct labels (%)", fontsize=6.9, labelpad=5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=6.5)
    cbar.solids.set_rasterized(False)
    fig.text(.700, .415, "≥5 hulls per displayed cell", color=SLATE, fontsize=6.9, ha="center")
    fig.add_artist(Line2D([.03,.966], [.382,.382], color=RULE, lw=.8, transform=fig.transFigure))
    panel_title(fig, .030, .350, "c", "Effects over the full validation split",
                f'Geometry: {data["hulls"]} hulls · AP: {data["images"]} chips')
    names = ("Independent − distance-paired", "Learned − dynamic attraction", "Learned − static attraction")
    length = fig.add_axes([.376,.113,.427,.131])
    clean_axis(length)
    length.set(xlim=(-.65,.55), ylim=(-.52, 2.52), yticks=[], xticks=(-.6, -.3, 0, .3))
    length.set_xticklabels(("−0.6", "−0.3", "0", "+0.3"))
    length.set_xlabel("Scene-balanced IoU difference (pp), paired 95% CI", fontsize=7.1, labelpad=4)
    length.axvline(0, color=SLATE, lw=.8, linestyle=(0, (2, 2)), zorder=1)
    fig.text(.903,.250, "Mask AP (pp)", fontsize=7.0, weight="bold", ha="center")
    for y, name, row in zip((2, 1, 0), names, data["comparisons"], strict=True):
        v = row["iou_delta_pp"]
        lo, hi = row["iou_ci95_pp"]
        color = BLUE if y == 2 else SLATE
        length.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt="o" if y==2 else "s",
                        color=color, markersize=3.8, lw=1.2, capsize=2.0)
        figure_y = fig.transFigure.inverted().transform(length.transData.transform((0, y)))[1]
        fig.text(.043, figure_y, name, fontsize=7.0, color=INK, va="center", weight="semibold" if y==2 else "regular")
        fig.text(.903, figure_y, signed(row["ap_delta_pp"], 3), fontsize=7.0, color=color,
                 ha="center", va="center")
    save(fig, output, "sarssp_public_summary")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True, help="Project release code directory")
    parser.add_argument("--sarssp", type=Path, required=True, help="Released SAR-SSP aggregate file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "seal-evidence-20261126",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": SLATE})
    draw_overview(effects(args.code), args.output)
    draw_sarssp(load_score(args.sarssp, SARSSP_SHA), args.output)


if __name__ == "__main__":
    main()
