"""Redraw the seven NORA evidence figures from the release's verified summaries."""

import argparse
import math
from pathlib import Path
from statistics import fmean
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


INK = "#473830"
ACCENT = "#A95B36"
TEAL = "#477163"
RUST = "#A14A44"
WINE = "#804859"
GRAY = "#756C65"
RULE = "#DED3C8"
WASH = "#F5EEE4"
PALE_TEAL = "#E4ECE4"
PALE_RUST = "#F3E3DA"
WHITE = "#FFFFFF"
WIDTH = 7.1


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.unicode_minus": True, "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": WHITE,
    })


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "−")


def heading(ax, text, title):
    ax.set_title(f"({text})  {title}", loc="left", fontsize=9.3, weight="semibold", pad=9)


def axis_style(ax, *, grid=None, zero=False):
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=7.7, pad=3)
    if grid:
        ax.grid(axis=grid, color=RULE, linewidth=.62)
        ax.set_axisbelow(True)
    if zero:
        ax.axvline(0, color=GRAY, lw=.81, zorder=1)


def dot(ax, x, y, color, marker="o", interval=None, fill=True):
    error = None if interval is None else [[x - interval[0]], [interval[1] - x]]
    ax.errorbar(x, y, xerr=error, fmt=marker, ms=4.6, color=color, mfc=color if fill else WHITE,
                mew=.95, elinewidth=1.15, capsize=2, zorder=3)


def key(color, marker, label, fill=True):
    return Line2D([], [], marker=marker, color=color, lw=0, ms=4.8,
                  mfc=color if fill else WHITE, label=label)


def save(fig, output, name):
    plt.rcParams["svg.hashsalt"] = f"nora-journal-{name}"
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            bounds = artist.get_window_extent(renderer)
            if (bounds.x0 < -1 or bounds.y0 < -1
                    or bounds.x1 > fig.bbox.x1 + 1 or bounds.y1 > fig.bbox.y1 + 1):
                raise ValueError(f"{name}: text beyond the page: {artist.get_text()}")
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    fig.savefig(output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def transfer(rows, output):
    fig = plt.figure(figsize=(WIDTH, 3.65))
    names = fig.add_axes([.02, .17, .32, .68])
    ap = fig.add_axes([.36, .17, .21, .68])
    iou = fig.add_axes([.61, .17, .365, .68])
    for ax in (names, ap, iou):
        ax.set(ylim=(7.5, -.5), yticks=[])
        for j in (1, 3, 5, 7):
            ax.axhspan(j-.5, j+.5, color=WASH, ec="none", zorder=0)
        for j in (.5, 1.5, 3.5, 5.5):
            ax.axhline(j, color=RULE, lw=.72, zorder=1)
    names.set(xlim=(0, 1), xticks=[])
    axis_style(names)
    ap.set(xlim=(0, 2), xticks=[])
    axis_style(ap)
    heading(ap, "a", "Mask AP (points)")
    ap.text(.5, 1.015, "L − F", ha="center", transform=ap.get_xaxis_transform(), fontsize=8, color=GRAY)
    ap.text(1.5, 1.015, "I − L", ha="center", transform=ap.get_xaxis_transform(), fontsize=8, color=ACCENT)
    ap.axvline(1, color=RULE, lw=.7)
    iou.set(xlim=(-.105, .49), xticks=[-.1, 0, .1, .2, .3, .4],
            xlabel="Difference (points) with paired 95% CI")
    axis_style(iou, grid="x", zero=True)
    heading(iou, "b", "Continuous IoU")
    names.set_title("Cohort / pipeline", loc="left", fontsize=8.6, pad=9, weight="semibold")
    for j, (title, pipeline, matches, measures) in enumerate(rows):
        if title:
            names.text(.02, j-.15, title, fontsize=7.8, weight="semibold", va="center")
        names.text(.02 if title else .10, j+.16 if title else j,
                   f"{pipeline.split(' · ')[0]}  /  {matches:,}", fontsize=7.2, va="center")
        for col, (m, color, marker) in enumerate(zip(measures, (GRAY, ACCENT), ("s", "o"), strict=True)):
            if m.ap >= .30:
                ap.add_patch(plt.Rectangle((col+.10, j-.35), .80, .70, fc=PALE_TEAL, ec="none"))
            ap.text(col+.5, j, signed(m.ap), ha="center", va="center", color=color, fontsize=8)
            dot(iou, m.iou, j + (-.16 if col == 0 else .16), color, marker, m.ci, col == 1)
    fig.legend(handles=[key(GRAY, "s", "L − F   Bounded re-pairing", False),
                        key(ACCENT, "o", "I − L   Independent ranking"),
                        Patch(facecolor=PALE_TEAL, label="AP ≥ 0.30 points")],
               frameon=False, loc="lower center", bbox_to_anchor=(.5, .015), ncol=3,
               fontsize=7.7, handlelength=1.4, columnspacing=1.5)
    save(fig, output, "transfer_effects")


def reference(data, verified, output):
    d, s = data["decomposition"], data["sensitivity"]
    verified.require(d["exact_both_reference_parity"]
                     and d["bootstrap_replicates"] == s["bootstrap_replicates"] == 4096,
                     "Expected common-reference archived parity")
    fig = plt.figure(figsize=(WIDTH, 2.50))
    left = fig.add_axes([.235, .25, .31, .57])
    right = fig.add_axes([.635, .25, .34, .57])
    for ax in (left, right):
        ax.set(xlim=(-.79, .50), xticks=[-.75, -.5, -.25, 0, .25, .5], ylim=(2.5, -.5),
               xlabel="Raster IoU difference (points)")
        ax.axhspan(.5, 1.5, color=WASH, ec="none")
        axis_style(ax, grid="x", zero=True)
    heading(left, "a", "By reference")
    heading(right, "b", "SPSD minus PSeg")
    labels = ("Natural count − native", "Independent − distance\n(equal signed counts)",
              "ROH − static attraction")
    left.set_yticks(range(3), labels)
    right.set_yticks([])
    for j, (name, _) in enumerate(verified.REFERENCE_COMPARISONS):
        a = s["references"]["PSeg_SSDD"]["pairwise"][name]["geometry"]
        b = s["references"]["SPSD"]["pairwise"][name]["geometry"]
        row = d["comparisons"][name]
        total = row["full_shift"]
        conflict = row["components"]["same_candidate_label_disagreement"]
        verified.require(math.isclose(b["points"] - a["points"], total["points"], abs_tol=1e-11)
                         and math.isclose(sum(c["points"] for c in row["components"].values()),
                                          total["points"], abs_tol=1e-11), "Paired shift and decomposition")
        for ax, entries in ((left, ((a, ACCENT, "o"), (b, WINE, "s"))),
                            (right, ((total, ACCENT, "o"), (conflict, TEAL, "D")))):
            for i, (v, color, marker) in enumerate(entries):
                dot(ax, v["points"], j + (-.14 if i == 0 else .14), color, marker, v["percentile95_points"], i == 0)
    for x, entries in ((.215, [key(ACCENT, "o", "PSeg-SSDD"), key(WINE, "s", "SPSD", False)]),
                       (.605, [key(ACCENT, "o", "Total shift"), key(TEAL, "D", "Label disagreement", False)])):
        fig.legend(handles=entries, loc="lower left", bbox_to_anchor=(x, .009),
                   ncol=2, frameon=False, fontsize=7.6, handletextpad=.4, columnspacing=1)
    save(fig, output, "reference_disagreement_effects")


def allocation(data, verified, output):
    v = data["allocation"]
    verified.require(v["exact_all_archive_score_parity"], "Expected archived allocation parity")
    fig = plt.figure(figsize=(WIDTH, 2.45))
    ax = fig.add_axes([.16, .28, .30, .51])
    ap = fig.add_axes([.468, .28, .11, .51])
    terms = fig.add_axes([.627, .28, .35, .51])
    for a in (ax, ap, terms):
        a.set_ylim(1.5, -.5)
        a.axhspan(.5, 1.5, color=WASH, ec="none")
    ax.set(xlim=(-.08, 1.35), xticks=[0, .4, .8, 1.2], xlabel="Raster IoU (points)")
    ax.set_yticks([0, 1], ["PSeg-SSDD", "SPSD"])
    heading(ax, "a", "Variable − equal split")
    axis_style(ax, grid="x", zero=True)
    ap.set(xlim=(0, 1), xticks=[], yticks=[])
    axis_style(ap)
    ap.set_title("AP (pts)", fontsize=7.7, pad=3)
    heading(terms, "b", "Change in IoU")
    terms.set(xlim=(-2.2, 2.9), xticks=[-2, -1, 0, 1, 2], yticks=[], xlabel="Contribution (points)")
    axis_style(terms, grid="x", zero=True)
    for y, ref in enumerate(("pseg", "spsd")):
        r = v["references"][ref]["variable_iou_minus"]["fixed_iou"]
        p = v["geometry_decomposition"][ref]["fixed_iou"]["group_means"]
        inter, area = p["overlap_iou_points"], p["area_iou_points"]
        verified.require(math.isclose(inter+area, r["iou_points"], abs_tol=1e-9), "IoU decomposition")
        dot(ax, r["iou_points"], y, ACCENT, "o", r["percentile95_points"])
        ap.text(.5, y, signed(r["ap_points"], 2), color=ACCENT, ha="center", va="center")
        terms.barh(y-.155, inter, height=.22, color=TEAL, zorder=2)
        terms.barh(y+.155, area, height=.22, color=RUST, zorder=2)
        for pos, value, color in ((y-.155, inter, TEAL), (y+.155, area, RUST)):
            terms.annotate(signed(value, 2), (value, pos), xytext=(4 if value >= 0 else -4, 0),
                           textcoords="offset points", ha="left" if value >= 0 else "right",
                           va="center", fontsize=7.2, color=color)
    fig.legend(handles=[Patch(facecolor=TEAL, label="Intersection term"),
                        Patch(facecolor=RUST, label="Area term")],
               loc="lower right", bbox_to_anchor=(.972, .012), ncol=2, frameon=False,
               fontsize=7.6, handlelength=1.25, columnspacing=1)
    save(fig, output, "dual_reference_budget_effect")


def sides(data, verified, output):
    result = data["sides"]
    verified.require(result["status"] == "ACTUAL_CLOSED_AND_VERIFIED", "Verified signed-side replay")
    views = (("740", "gf3", 583, (5217, 5218), "Gaofen-3"),
             ("74", "full", 74, (5217,), "Reserve"),
             ("74", "date_separated", 42, (5217,), "Nested reserve"))
    fig, ax = plt.subplots(figsize=(3.49, 2.51))
    fig.subplots_adjust(left=.30, bottom=.22, right=.97, top=.80)
    ax.set(ylim=(2.5, -.5), xlim=(0, 1.2), xticks=[0, .25, .5, .75, 1],
           xlabel="Mask AP vs. both sides fixed (points)")
    ax.set_yticks(range(3), [f"{name}\n{n} chips" for _, _, n, _, name in views])
    colors = (TEAL, RUST, ACCENT)
    markers = ("^", "v", "o")
    axis_style(ax, grid="x", zero=True)
    ax.axhspan(.5, 1.5, color=WASH, ec="none")
    for y, (cohort, view, count, fits, _) in enumerate(views):
        source = result["cohorts"][cohort]["score"]
        score = source["views"][view]
        verified.require(score["images"] == count and tuple(source["seeds"]) == fits, "Original signed-side view")
        for i, (arm, color, marker) in enumerate(zip(("IF", "FI", "II"), colors, markers, strict=True)):
            values = score["metrics_fraction"]
            delta = fmean(100*(values[f"{fit}/{arm}"]["AP"] - values[f"{fit}/FF"]["AP"]) for fit in fits)
            verified.require(math.isclose(delta, score["contrast_summaries_points"][f"{arm}_minus_FF"]["AP"]["mean"],
                                          abs_tol=1e-12), "Original signed-side AP")
            pos = y + (i-1)*.235
            dot(ax, delta, pos, color, marker, fill=i == 2)
            ax.annotate(signed(delta, 2), (delta, pos), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.1, color=color)
    fig.legend(handles=[key(c, m, label, i == 2) for i, (c, m, label) in enumerate(
        zip(colors, markers, ("Additions", "Removals", "Both"), strict=True))],
        loc="upper center", bbox_to_anchor=(.5, .985), frameon=False, ncol=3,
        fontsize=7.7, handletextpad=.25, columnspacing=.9)
    save(fig, output, "independent_sides_ablation")


def phoenix(data, verified, output):
    fig = plt.figure(figsize=(WIDTH, 2.53))
    names = fig.add_axes([.02, .25, .24, .51])
    axes = [fig.add_axes([.27, .25, .308, .51]), fig.add_axes([.667, .25, .308, .51])]
    for a in (names, *axes):
        a.set(ylim=(3.5, -.5), yticks=[])
        for y in (1, 3):
            a.axhspan(y-.5, y+.5, color=WASH, ec="none", zorder=0)
        a.axhline(1.5, color=RULE, lw=.73)
    names.set(xlim=(0, 1), xticks=[])
    axis_style(names)
    for i, (a, title) in enumerate(zip(axes, ("Mask AP", "Continuous IoU"), strict=True)):
        a.set(xlim=(-.40, 10.9), xticks=[0, 2, 4, 6, 8, 10], xlabel="Exact count − threshold (points)")
        axis_style(a, grid="x", zero=True)
        heading(a, "ab"[i], title)
    for j, (dataset, images, hulls, _, values) in enumerate(verified.phoenix_scores(data)):
        names.text(.02, 2*j-.15, f"{dataset} · {images} images", weight="semibold", va="center", fontsize=8)
        names.text(.02, 2*j+.2, f"{hulls} eligible fixed matches", color=GRAY, va="center", fontsize=7.3)
        for i, ((threshold, fixed, support, _, _), color, marker) in enumerate(
                zip(verified.SUPPORTS, (TEAL, ACCENT), ("o", "s"), strict=True)):
            y = 2*j + i
            for metric, a in enumerate(axes):
                old, new = values[threshold][metric], values[fixed][metric]
                delta = new-old
                a.hlines(y+.10, 0, delta, lw=.9, color=color)
                dot(a, delta, y+.10, color, marker, fill=i == 0)
                a.text(10.8, y-.17, f"{old:.3f} → {new:.3f}", ha="right", va="center", fontsize=7.1, color=color)
    fig.legend(handles=[key(TEAL, "o", "Sampled boundary"), key(ACCENT, "s", "Whole grid", False)],
               loc="lower center", bbox_to_anchor=(.55, .015), frameon=False, ncol=2, fontsize=7.7)
    save(fig, output, "phoenix_budget_by_dataset")


def dev705(cases, output):
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.48), sharex=True)
    fig.subplots_adjust(left=.14, right=.98, top=.72, bottom=.22, wspace=.30)
    all_values = [abs(q) for c in cases for arm in ("SF", "SU=F") for q in
                  (c["arms"][arm]["add_overlap"], c["arms"][arm]["drop_overlap"])]
    limit = max(all_values) * 1.30
    for j, (ax, case) in enumerate(zip(axs, cases, strict=True)):
        f, i = (case["arms"][arm] for arm in ("SF", "SU=F"))
        ax.set(xlim=(-limit, limit), ylim=(1.5, -.5), xlabel="Intersection change (cell area)")
        ax.set_yticks([0, 1], ["Distance pairing", "Independent"] if j == 0 else [])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        axis_style(ax, grid="x", zero=True)
        ax.axhspan(.5, 1.5, color=WASH, ec="none")
        outcome = "Gain case" if j == 0 else "Loss case"
        heading(ax, "ab"[j], f"{outcome} · {f['adds']} swaps")
        effect = signed(100*(i["iou"]-f["iou"]), 2)
        ax.text(.98, 1.016, f"ΔIoU {effect} points", transform=ax.transAxes,
                fontsize=7.4, color=ACCENT, ha="right", va="bottom")
        for y, arm in enumerate((f, i)):
            add, drop = arm["add_overlap"], arm["drop_overlap"]
            ax.barh(y, add, height=.31, color=TEAL, zorder=2)
            ax.barh(y, -drop, height=.31, color=RUST, zorder=2)
            dot(ax, add-drop, y+.22, INK, "D", fill=False)
    fig.legend(handles=[Patch(facecolor=TEAL, label="Recovered"), Patch(facecolor=RUST, label="Lost"),
                        key(INK, "D", "Net", False)], loc="upper center",
               bbox_to_anchor=(.50, .987), ncol=3, frameon=False, fontsize=7.6)
    save(fig, output, "dev705_overlap_cases")


def sarssp(v, output):
    fig = plt.figure(figsize=(WIDTH, 4.23))
    dist = fig.add_axes([.087, .61, .39, .27])
    grid = fig.add_axes([.636, .59, .25, .30])
    effects = fig.add_axes([.43, .133, .42, .28])
    heading(dist, "a", "Changed matches")
    fig.text(.57, .912, "(b)  Correct-label difference", fontsize=9.0, weight="semibold")
    heading(effects, "c", "Scene-balanced raster IoU")
    hist = v["distribution"]["counts"]
    if sum(hist) != v["changed"] or len(hist) != 9:
        raise ValueError("Unexpected SAR-SSP distribution")
    bars = dist.bar(range(9), hist, color=[RUST]*4+[GRAY]+[TEAL]*4, width=.69)
    dist.bar_label(bars, labels=[str(n) if n else "" for n in hist], padding=2, fontsize=7.0)
    dist.set(xlim=(-.58, 8.58), ylim=(0, max(hist)+4), ylabel="Hulls",
             xlabel="Independent − distance pairing (IoU points)")
    dist.set_xticks(range(9), ["<−4", "−4…−2", "−2…−1", "−1…0", "0", "0…1", "1…2", "2…4", ">4"],
                  rotation=24)
    axis_style(dist, grid="y")
    values = v["spatial"]["grid"]
    if len(values) != 4 or any(len(row) != 4 for row in values) or v["spatial"]["supporting_hulls_minimum"] < 5:
        raise ValueError("Unexpected native hull-grid summary")
    cmap = LinearSegmentedColormap.from_list("sign", [RUST, PALE_RUST, WHITE, PALE_TEAL, TEAL])
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    for y, row in enumerate(values):
        for x, val in enumerate(row):
            fc = WASH if val is None else cmap(norm(max(-50, min(50, val))))
            grid.add_patch(plt.Rectangle((x/4, y/4), .25, .25, fc=fc, ec=WHITE, lw=.7))
            grid.text((x+.5)/4, (y+.5)/4, "—" if val is None else "0" if val == 0 else signed(val, 0),
                      color=WHITE if val is not None and abs(val) >= 36 else INK,
                      ha="center", va="center", fontsize=7.8, weight="medium")
    grid.set(aspect="equal", xlim=(0, 1), ylim=(1, 0), xticks=[0, .5, 1], yticks=[0, .5, 1])
    axis_style(grid)
    ax_cb = fig.add_axes([.881, .609, .017, .241])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax_cb, ticks=(-50, 0, 50))
    cb.ax.set_ylabel("Correct-label difference (%)", fontsize=7.0)
    cb.ax.tick_params(length=0, labelsize=7.0)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    names = ("Independent − distance pairing", "ROH − dynamic attraction", "ROH − static attraction")
    effects.set(xlim=(-.66, .53), ylim=(2.5, -.5), yticks=range(3), xticks=[-.6, -.3, 0, .3],
                xlabel="IoU difference (points) with paired 95% CI")
    effects.set_yticklabels(names)
    effects.axhspan(.5, 1.5, color=WASH, ec="none")
    axis_style(effects, grid="x", zero=True)
    effects.text(1.17, 1.019, "AP (pts)", transform=effects.transAxes, fontsize=7.6, weight="semibold", ha="center")
    for y, record in enumerate(v["comparisons"]):
        marker, color = ("o", ACCENT) if y == 0 else ("s", GRAY)
        dot(effects, record["iou_delta_pp"], y, color, marker, tuple(record["iou_ci95_pp"]), y == 0)
        effects.text(1.17, y, signed(record["ap_delta_pp"]), transform=effects.get_yaxis_transform(),
                     va="center", ha="center", fontsize=7.8, color=color)
    save(fig, output, "sarssp_public_summary")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=Path, required=True, help="Paper directory in the GitHub release")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.paper / "figures"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "cvpr_20261126"))
    import plot_cvpr_evidence as archived
    import plot_quantitative_redesign as verified
    from plot_dev705_cases import load_cases

    style()
    results = args.paper.parent / "results"
    data = verified.load_aggregates(results)
    transfer(archived.effects(args.paper.parent / "code"), args.output)
    reference(data, verified, args.output)
    allocation(data, verified, args.output)
    sides(data, verified, args.output)
    phoenix(data, verified, args.output)
    dev705(load_cases(source / "dev705_overlap_cases.json"), args.output)
    sarssp(archived.load_score(source / "sarssp_public_aggregates.json", archived.SARSSP_SHA), args.output)


if __name__ == "__main__":
    main()
