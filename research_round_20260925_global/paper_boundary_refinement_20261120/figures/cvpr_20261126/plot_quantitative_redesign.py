"""Build four editable quantitative panels from the frozen public aggregate summaries."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INK = "#243247"
TEAL = "#007F83"
ORANGE = "#C4552D"
BLUE = "#2D5E91"
SLATE = "#5C6675"
RULE = "#DCE5EE"

INPUTS = {
    "decomposition": (
        "SPSD_PSEG232_SHARED_HULL_DECOMPOSITION_20261125_SAFE.json",
        "2522ec539f610fb1f01a474d35a2a0c35521c7dd8e3357522f884a63fedee1b1",
    ),
    "sensitivity": (
        "SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125_SAFE.json",
        "dd152bd587d64453b874a9a34e6531adacbd7bf163de4f9b22ffb25c4dea52fc",
    ),
    "allocation": (
        "SPSD_PSEG_SCORE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125_SAFE.json",
        "f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a",
    ),
    "sides": (
        "INDEPENDENT_SIDE_FACTORIAL_ACTUAL_SAFE_20261123.json",
        "098191b4448f02dd7bf2230998493417a9dfb79c3cd4a137b80fc0962d33dd75",
    ),
    "rsdd": (
        "RSDD74_PHOENIX_SUPPORT_BUDGET_ACTUAL_SAFE_20261123.json",
        "d21b50d43f8aaf514d12a4b69f2d2a22393e0a580b3ae24436fb82821bde1394",
    ),
    "pseg": (
        "PSEG232_PHOENIX_SUPPORT_BUDGET_ACTUAL_SAFE_20261123.json",
        "e96c7fe8dc857fa85e0fd6051cf267068ae4a027ea8299a19678f5facfcfbcff",
    ),
}

REFERENCE_COMPARISONS = (
    ("free_natural_minus_raw", "Natural count\nvs. native"),
    ("free_matched_count_minus_fixed", "Independent ranking\nvs. distance pairing"),
    ("learned_at_attraction_count_minus_attraction_static_at_dynamic_count",
     "Occupancy ranking\nvs. static attraction"),
)

SIDE_SERIES = (("IF", "Add only", TEAL, "o"),
               ("FI", "Remove only", ORANGE, "s"),
               ("II", "Both", BLUE, "D"))

SUPPORTS = (("nt", "nc", "Sampled boundary", TEAL, "o"),
            ("wt", "wc", "Whole grid", SLATE, "s"))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_aggregates(results):
    data = {}
    for key, (name, expected) in INPUTS.items():
        raw = (results / name).read_bytes()
        require(sha256(raw).hexdigest() == expected, f"Frozen aggregate mismatch for {name}")
        data[key] = json.loads(raw)
    return data


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 8.2, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": INK,
        "axes.labelsize": 8.2, "axes.titlesize": 8.8, "axes.titleweight": "semibold",
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": True,
    })


def save(fig, destination, stem):
    matplotlib.rcParams["svg.hashsalt"] = stem
    fig.savefig(destination / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination / f"{stem}.svg", metadata={"Date": None})
    plt.close(fig)


def panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", color=INK, pad=7)


def x_axis(ax, limits, ticks, label, *, y_limits):
    ax.set(xlim=limits, xticks=ticks, xlabel=label, ylim=y_limits)
    ax.xaxis.labelpad = 5
    ax.grid(axis="x", color=RULE, lw=.62, zorder=0)
    ax.axvline(0, color=SLATE, lw=.68, alpha=.80, zorder=1)
    ax.tick_params(axis="both", length=0, labelsize=7.9, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "\N{MINUS SIGN}")


def point_interval(ax, value, interval, row, color, marker, fill=True):
    lo, hi = interval
    require(math.isfinite(lo) and math.isfinite(hi) and lo <= value <= hi,
            "Expected a finite estimate within its paired interval")
    ax.errorbar(value, row, xerr=[[value-lo], [hi-value]], linestyle="none",
                marker=marker, color=color, mfc=color if fill else "white", mec=color,
                markersize=4.5, mew=1.1, elinewidth=1.05, capsize=2.1, capthick=.9, zorder=4)


def legend_point(color, marker, label, filled=True):
    return Line2D([], [], color=color, lw=0, marker=marker, markersize=4.8,
                  markerfacecolor=color if filled else "white", markeredgewidth=1.0, label=label)


def reference_disagreement(data, destination):
    decomposition, sensitivity = data["decomposition"], data["sensitivity"]
    require(decomposition["exact_both_reference_parity"] and
            decomposition["bootstrap_replicates"] == sensitivity["bootstrap_replicates"] == 4096,
            "Exact archived reference parity and original paired bootstrap are required")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.06, 2.51))
    fig.subplots_adjust(left=.224, right=.975, bottom=.245, top=.865, wspace=.33)
    positions = (2, 1, 0)
    legends = (
        (legend_point(TEAL, "o", "PSeg-SSDD"), legend_point(ORANGE, "s", "SPSD", False)),
        (legend_point(BLUE, "o", "Total shift"), legend_point(TEAL, "D", "Conflicting labels", False)),
    )
    for y, (key, _) in zip(positions, REFERENCE_COMPARISONS, strict=True):
        p = sensitivity["references"]["PSeg_SSDD"]["pairwise"][key]["geometry"]
        s = sensitivity["references"]["SPSD"]["pairwise"][key]["geometry"]
        row = decomposition["comparisons"][key]
        total = row["full_shift"]
        conflict = row["components"]["same_candidate_label_disagreement"]
        require(math.isclose(s["points"]-p["points"], total["points"], abs_tol=1e-11)
                and math.isclose(sum(c["points"] for c in row["components"].values()),
                                 total["points"], abs_tol=1e-11), "Original paired difference and decomposition")
        for axis, records, hues, symbols in (
            (left, (p, s), (TEAL, ORANGE), ("o", "s")),
            (right, (total, conflict), (BLUE, TEAL), ("o", "D")),
        ):
            for j, (record, hue, symbol) in enumerate(zip(records, hues, symbols, strict=True)):
                z = y + (.153 if j == 0 else -.153)
                point_interval(axis, record["points"], record["percentile95_points"], z, hue, symbol, not j)
                axis.text(.985, z, signed(record["points"]), transform=axis.get_yaxis_transform(),
                          horizontalalignment="right", verticalalignment="center", color=hue,
                          fontsize=7.65)
    for axis, label, handles in zip(
        (left, right), ("Raster IoU effect (points)", "Difference in effects (points)"), legends, strict=True
    ):
        x_axis(axis, (-.80, .50), (-.75, -.50, -.25, 0, .25), label, y_limits=(-.52, 2.49))
        axis.set_yticks(positions)
        axis.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.01, -.36), ncol=2,
                    frameon=False, borderaxespad=0, handletextpad=.35, columnspacing=.75, fontsize=7.6)
    panel_title(left, "a", "Reference-specific effects")
    panel_title(right, "b", "SPSD minus PSeg-SSDD")
    left.set_yticklabels([label for _, label in REFERENCE_COMPARISONS], ha="right")
    right.tick_params(labelleft=False)
    save(fig, destination, "reference_disagreement_effects")


def dual_allocation(data, destination):
    result = data["allocation"]
    require(result["exact_all_archive_score_parity"], "Exact archived allocation-score parity is required")
    fig = plt.figure(figsize=(7.06, 2.43))
    effect = fig.add_axes((.142, .29, .275, .55))
    apcol = fig.add_axes((.434, .29, .086, .55))
    parts = fig.add_axes((.574, .29, .396, .55))
    for y, ref in enumerate(("spsd", "pseg")):
        stat = result["references"][ref]["variable_iou_minus"]["fixed_iou"]
        terms = result["geometry_decomposition"][ref]["fixed_iou"]["group_means"]
        value = stat["iou_points"]
        overlap, area = terms["overlap_iou_points"], terms["area_iou_points"]
        require(math.isclose(overlap + area, value, abs_tol=1e-9),
                "Exact original intersection and area terms reconstruct the total")
        point_interval(effect, value, stat["percentile95_points"], y, BLUE, "o")
        effect.annotate(signed(value), (value, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", color=BLUE, fontsize=7.9)
        apcol.text(.5, y, signed(stat["ap_points"]), color=BLUE, ha="center", va="center", fontsize=8.25)
        for offset, amount, hue in ((.116, overlap, TEAL), (-.116, area, ORANGE)):
            parts.barh(y+offset, amount, height=.16, color=hue, zorder=3)
            parts.annotate(signed(amount, 2), (amount, y+offset), xytext=(4 if amount >= 0 else -4, 0),
                           textcoords="offset points", va="center", ha="left" if amount >= 0 else "right",
                           fontsize=7.8, color=hue)
    shared_rows = (-.46, 1.45)
    x_axis(effect, (-.075, 1.32), (0, .25, .5, .75, 1, 1.25), "Raster IoU (points)", y_limits=shared_rows)
    x_axis(parts, (-2.13, 2.98), (-2, -1, 0, 1, 2), "IoU contribution (points)", y_limits=shared_rows)
    effect.set_yticks((0, 1), ("SPSD", "PSeg-SSDD"))
    effect.tick_params(axis="y", pad=7)
    parts.set_yticks([])
    apcol.set(ylim=shared_rows, xlim=(0, 1))
    apcol.axis("off")
    apcol.text(.5, 1.34, "Mask AP (pts)", transform=apcol.get_yaxis_transform(), ha="center",
               va="top", fontsize=7.5, color=SLATE)
    panel_title(effect, "a", "Variable minus equal split")
    panel_title(parts, "b", "IoU decomposition")
    parts.legend(handles=[Patch(facecolor=TEAL, label="Intersection term"),
                           Patch(facecolor=ORANGE, label="Area term")],
                 loc="lower left", bbox_to_anchor=(0, -.42), ncol=2, frameon=False,
                 borderaxespad=0, handlelength=1.2, columnspacing=1.0, fontsize=7.7)
    save(fig, destination, "dual_reference_budget_effect")


def signed_side_plot(data, destination):
    result = data["sides"]
    require(result["status"] == "ACTUAL_CLOSED_AND_VERIFIED", "Completed same-count signed-side replay")
    views = (("740", "gf3", 583, (5217, 5218), "Gaofen-3\n583 images, 2 seeds"),
             ("74", "full", 74, (5217,), "Reserve\n74 images"),
             ("74", "date_separated", 42, (5217,), "  Date-separated\n  42 of 74"))
    fig = plt.figure(figsize=(3.49, 2.31))
    ax = fig.add_axes((.362, .218, .615, .644))
    x_axis(ax, (0, 1.16), (0, .25, .5, .75, 1.0), "Mask AP gain vs. both fixed (points)",
           y_limits=(2.52, -.50))
    ax.set_yticks(range(len(views)), [r[-1] for r in views])
    ax.tick_params(axis="y", pad=7, labelsize=7.6)
    ax.axhline(1.5, color=RULE, lw=.68, zorder=1)
    for index, (cohort, key, population, seeds, _) in enumerate(views):
        source = result["cohorts"][cohort]["score"]
        view = source["views"][key]
        require(view["images"] == population and tuple(source["seeds"]) == seeds,
                "Exact original signed-side cohort and point-head seed set")
        for offset, (arm, _, color, marker) in zip((-.247, 0, .247), SIDE_SERIES, strict=True):
            ap = view["metrics_fraction"]
            delta = fmean(100*(ap[f"{seed}/{arm}"]["AP"]-ap[f"{seed}/FF"]["AP"]) for seed in seeds)
            expect = view["contrast_summaries_points"][f"{arm}_minus_FF"]["AP"]["mean"]
            require(math.isclose(delta, expect, rel_tol=0, abs_tol=1e-12), "Recomputed original AP contrast")
            row = index + offset
            ax.hlines(row, 0, delta, color=color, lw=.85, alpha=.68, zorder=2)
            ax.plot(delta, row, marker=marker, ms=4.0, color=color, lw=0, zorder=4)
            ax.annotate(signed(delta), (delta, row), xytext=(4.7, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.5, color=color)
    fig.legend(handles=[legend_point(hue, marker, label) for _, label, hue, marker in SIDE_SERIES],
               loc="upper center", bbox_to_anchor=(.54, .976), ncol=3, frameon=False,
               handlelength=1.0, handletextpad=.3, columnspacing=1.0, fontsize=7.65)
    save(fig, destination, "independent_sides_ablation")


def phoenix_scores(data):
    expected = (("rsdd", "RSDD74", "full", "Instance-RSDD", 74, 47, "Parent-weighted IoU"),
                ("pseg", "PSEG232", "official", "PSeg-SSDD", 232, 293, "Image-group-weighted IoU"))
    for key, tag, view_key, dataset, images, hulls, weighting in expected:
        record = data[key]
        require(record["status"] == tag + "_PHOENIX_SUPPORT_BUDGET_SCORE_COMPLETE"
                and record["phase"] == "score" and record["primary_view"] == view_key,
                "Exact original aggregate Phoenix comparison")
        view = record["views"][view_key]
        g = view["geometry"]
        native_matches = g["counts"]["eligible_matches"] if key == "rsdd" else g["eligible_pairs"]
        require(view["images"] == images and native_matches == hulls, "Historical fixed cohort and matched hull count")
        geometry = g["group_equal_arms"] if key == "rsdd" else g["arms"]
        values = {}
        for threshold, fixed, *_ in SUPPORTS:
            for arm in (threshold, fixed):
                pair = (100*view["metrics_fraction"][arm]["AP"], geometry[arm]["continuous_iou_points"])
                require(all(math.isfinite(v) and 0 <= v <= 100 for v in pair), "Numeric Phoenix absolute scores")
                values[arm] = pair
        yield dataset, images, hulls, weighting, values


def phoenix_budget(data, destination):
    fig = plt.figure(figsize=(7.06, 2.75))
    left = fig.add_axes((.223, .212, .203, .60))
    right = fig.add_axes((.603, .212, .203, .60))
    y_limits = (-.59, 4.70)
    for ax, title, letter, limits, ticks in (
        (left, "Mask AP", "a", (-.70, 10.55), (0, 2, 4, 6, 8, 10)),
        (right, "Continuous IoU", "b", (-.70, 10.55), (0, 2, 4, 6, 8, 10)),
    ):
        x_axis(ax, limits, ticks, "Change (points)", y_limits=y_limits)
        panel_title(ax, letter, title)
        ax.set_yticks([])
        ax.axhline(2.15, color=RULE, lw=.68, zorder=0)
        ax.text(1.40, 4.43, "Threshold → exact", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7.35, color=SLATE)
    for (dataset, images, hulls, weighting, values), rows in zip(
        phoenix_scores(data), ((3.58, 2.84), (1.36, .62)), strict=True
    ):
        top = rows[0] + .54
        left.text(-1.0, top, f"{dataset} · {images} images", transform=left.get_yaxis_transform(),
                  fontsize=8.3, fontweight="semibold", ha="left", va="center")
        right.text(-.02, top, f"{hulls} hulls · {weighting[0].lower() + weighting[1:]}", transform=right.get_yaxis_transform(),
                   fontsize=7.65, ha="left", va="center", color=SLATE)
        for y, (threshold, fixed, support, hue, marker) in zip(rows, SUPPORTS, strict=True):
            left.text(-.065, y, support, transform=left.get_yaxis_transform(),
                      fontsize=7.75, ha="right", va="center", color=hue)
            for metric, axis in enumerate((left,right)):
                old, new = values[threshold][metric], values[fixed][metric]
                delta = new-old
                axis.hlines(y, min(0, delta), max(0, delta), color=hue, lw=1.08, zorder=2)
                axis.plot(delta, y, marker=marker, ms=4.15, color=hue, lw=0, zorder=4)
                side = -4 if delta > 7.8 else 4
                axis.annotate(signed(delta), (delta, y), xytext=(side, 5.0), textcoords="offset points",
                              ha="right" if side < 0 else "left", va="center", color=hue, fontsize=7.55)
                axis.text(1.08, y, f"{old:.3f} → {new:.3f}", transform=axis.get_yaxis_transform(),
                          fontsize=7.3, ha="left", va="center", color=INK)
    fig.text(.53, .035, "Change = exact count minus threshold", ha="center", fontsize=7.7, color=SLATE)
    save(fig, destination, "phoenix_budget_by_dataset")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Published aggregate results directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = load_aggregates(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    reference_disagreement(data, args.output_dir)
    dual_allocation(data, args.output_dir)
    signed_side_plot(data, args.output_dir)
    phoenix_budget(data, args.output_dir)


if __name__ == "__main__":
    main()

