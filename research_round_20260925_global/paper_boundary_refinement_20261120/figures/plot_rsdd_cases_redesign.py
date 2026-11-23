"""Redraw the two accepted Instance-RSDD qualitative examples at journal print size."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import Normal, Stroke
import numpy as np


CASE_SOURCE = "6c6726f1e892cf753678a83c5bdc75f017d05176b417b5966ad02b6cb9d34e7f"
CASE_SUMMARY = "016a9dd3ae518c92e944e45ec4c7c6226f495851ddc274b06bbc2138a2882c0b"
NAVY, TEAL, RUST, INK = "#25384B", "#087E8B", "#A94E37", "#111D2B"
OFFWHITE = "#FAFBFA"
TITLES = ("Native", "Local occupancy", "SegRefiner-LR", "Intersection change")


def require(test, message):
    if not test:
        raise ValueError(message)


def pinned(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Released input differs for " + path.name)
    return raw


def module(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def settings():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "rsdd74-segrefiner-print", "axes.unicode_minus": True,
                         "figure.dpi": 140, "savefig.dpi": 300})


def local_panel(ax, case, masks, column, delta, edges, signed):
    side = case["side"]
    extent = (0, side, side, 0)
    if column < 3:
        lo, hi = case["gray_range"]
        ax.imshow(case["gray"], cmap="gray", vmin=lo, vmax=max(hi, lo+1),
                  extent=extent, interpolation="nearest")
        segments = edges(masks[column])
        ax.add_collection(LineCollection(segments, colors="white", linewidths=1.8, alpha=.85))
        ax.add_collection(LineCollection(segments, colors=TEAL, linewidths=.95))
        true_color = RUST
        halo = [Stroke(linewidth=1.8, foreground="white", alpha=.90), Normal()]
    else:
        ax.imshow(delta, cmap=signed, norm=TwoSlopeNorm(0, -1, 1),
                  extent=extent, interpolation="nearest")
        true_color, halo = INK, []
    ax.plot(*case["contour"].T, color=true_color, lw=.83, ls=(0, (3.0, 1.6)), path_effects=halo)
    ax.set(xlim=(0, side), ylim=(side, 0))


def public_panel(ax, case, masks, column, pooled_delta, limit, pooled, shades, signed):
    size = pooled(masks[0]).shape[0]
    breaks = np.arange(size+1)
    centers = breaks[:-1]+.5
    if column < 3:
        foreground = np.rint(4*pooled(masks[column]))
        ax.pcolormesh(breaks, breaks, foreground, cmap=shades, vmin=0, vmax=4,
                      edgecolors="#E6ECEE", linewidth=.19)
    else:
        encoded = np.sign(pooled_delta)*np.maximum(1, np.rint(4*np.abs(pooled_delta)/limit))
        ax.pcolormesh(breaks, breaks, encoded, cmap=signed, norm=TwoSlopeNorm(0, -4, 4),
                      edgecolors="#E6ECEE", linewidth=.19)
    truth = pooled(case["truth"])
    if truth.min() < .25 < truth.max():
        ax.contour(centers, centers, truth, levels=[.25], colors=INK,
                   linestyles=[(0, (3., 1.8))], linewidths=.93)
    ax.set(xlim=(0, size), ylim=(size, 0))


def render(cases, pooled, edges, public):
    settings()
    width, height = 7.15, 3.67
    left, stride, panel = .77, 1.58, 1.43
    row_bottoms = (1.98, .39)
    fig = plt.figure(figsize=(width, height), facecolor="white")
    shades = ListedColormap([OFFWHITE, "#E1E9EF", "#AFC3D2", "#738DA3", NAVY])
    signed = LinearSegmentedColormap.from_list("intersection_change", [RUST, OFFWHITE, TEAL])
    deltas = {name: (c["masks"][2].astype(float)-c["masks"][0])*c["truth"] for name, c in cases.items()}
    coarse = {name: pooled(value) for name, value in deltas.items()} if public else {}
    limit = max((float(np.abs(v).max()) for v in coarse.values()), default=0.) or 1.
    for column, title in enumerate(TITLES):
        center = left + column*stride + panel/2
        fig.text(center/width, .947, title, fontsize=9.5, fontweight="semibold", color=INK, ha="center")
    for index, (sign, case) in enumerate(cases.items()):
        bottom = row_bottoms[index]
        accent = TEAL if sign == "gain" else RUST
        fig.text(.026, (bottom+panel*.53)/height, f"({'ab'[index]})", fontsize=9.4,
                 weight="semibold", color=INK, va="center")
        fig.text(.026, (bottom+panel*.40)/height, sign.capitalize(), fontsize=8.4,
                 color=accent, va="center")
        masks = case["masks"]
        for column in range(4):
            x = left + column*stride
            ax = fig.add_axes([x/width, bottom/height, panel/width, panel/height], facecolor=OFFWHITE)
            if public:
                public_panel(ax, case, masks, column, coarse[sign], limit, pooled, shades, signed)
            else:
                local_panel(ax, case, masks, column, deltas[sign], edges, signed)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#A4B1BA")
                spine.set_linewidth(.50)
            metric = (f"IoU {case['iou_points'][column]:.2f}" if column < 3
                      else f"ΔIoU {case['delta_points']:+.2f} pt")
            fig.text((x+panel/2)/width, (bottom-.09)/height, metric, ha="center", va="center",
                     fontsize=8.45, weight="semibold" if column == 3 else "normal",
                     color=accent if column == 3 else NAVY)
    handles = ([Patch(facecolor=NAVY, label="Mask fraction (8×8)"),
                Line2D([], [], color=INK, ls=(0, (3, 1.8)), label="25% truth contour")] if public else
               [Line2D([], [], color=TEAL, lw=1.35, label="Predicted boundary"),
                Line2D([], [], color=RUST, ls=(0, (3, 1.8)), label="Author hull")])
    handles += [Patch(facecolor=TEAL, label="Gained intersection"), Patch(facecolor=RUST, label="Lost intersection")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .007), ncols=4,
               frameon=False, fontsize=7.3, handlelength=1.45, columnspacing=1.22, borderaxespad=0.)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for arg in ("code", "run", "stock-root", "point-root", "author-labels", "protocol", "public-output", "private-output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    for arg in ("score-terminal", "inference-terminal"):
        parser.add_argument("--" + arg, type=Path)
    args = parser.parse_args()
    for name, path in vars(args).items():
        if isinstance(path, Path):
            setattr(args, name, path.resolve(strict=name not in {"author_labels", "public_output", "private_output"}))
    args.score_terminal = args.score_terminal or args.run / "observed_score" / "TERMINAL.json"
    args.inference_terminal = args.inference_terminal or args.run / "observed_inference" / "TERMINAL.json"
    public, private = args.public_output, args.private_output
    require(public != private and public not in private.parents and private not in public.parents
            and not public.exists() and not private.exists(), "Distinct fresh public and private output directories required")
    stem = args.code / "SEGREFINER_STOCK_20261123"
    original = module(stem / "plot_rsdd_cases.py", CASE_SOURCE)
    scorer = original.module(stem / "score_rsdd.py", original.SCORE_WORKER)
    require(args.protocol.name == scorer.LOCK_NAME, "Released stock comparison protocol required")
    pinned(args.protocol, scorer.LOCK)
    published = json.loads(pinned(args.code.parent / "results" / "RSDD74_SEGREFINER_STOCK_CASES_SAFE_20261123.json",
                                  CASE_SUMMARY))
    boundary = original.module(args.code.parent / "paper_boundary_refinement_20261120"
                               / "figures" / "plot_gf3_matched_cases.py", original.PLOT_SOURCE)
    rows, core = original.qualified_cases(args, scorer)
    selection, sign_counts, decisions = original.choose_cases(rows)
    require(sign_counts == published["sign_counts"] and decisions == published["selection_details"],
            "The accepted fixed median-sign case selection differs")
    cases = {name: original.cropped_case(row, args.stock_root / "conversion", core)
             for name, row in selection.items()}
    for name, case in cases.items():
        expected = published["cases"][name]
        require(np.allclose(case["iou_points"], expected["iou_points"], rtol=0, atol=original.EPS)
                and abs(case["delta_points"]-expected["delta_points"]) <= original.EPS,
                "The accepted continuous case scores differ")

    public.mkdir(parents=True)
    private.mkdir(parents=True)
    fig = render(cases, original.pooled, boundary.edges, public=False)
    fig.savefig(private / (original.PRIVATE + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig = render(cases, original.pooled, boundary.edges, public=True)
    fig.savefig(public / (original.PUBLIC + ".svg"), metadata={"Date": None})
    fig.savefig(public / (original.PUBLIC + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(json.dumps(dict(status="RSDD74_SEGREFINER_CASE_REDRAW_COMPLETE", sign_counts=sign_counts,
                          cases=published["cases"], accepted_case_summary_sha256=CASE_SUMMARY), sort_keys=True))


if __name__ == "__main__":
    main()
