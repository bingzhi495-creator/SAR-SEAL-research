"""Plot frozen matched-side AP deltas from the published verified aggregate."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter


SOURCE_SHA = "098191b4448f02dd7bf2230998493417a9dfb79c3cd4a137b80fc0962d33dd75"
FILE = "INDEPENDENT_SIDE_FACTORIAL_ACTUAL_SAFE_20261123.json"
VIEWS = (
    ("740", "gf3", 583, (5217, 5218), "Gaofen-3\n583 images · 2 seeds"),
    ("74", "full", 74, (5217,), "Reserve\n74 images"),
    ("74", "date_separated", 42, (5217,), "Date-separated\n42 of 74 · sensitivity"),
)
SERIES = (("IF", "Add only", "#006E70", "o"),
          ("FI", "Remove only", "#9A4025", "s"),
          ("II", "Both", "#20384D", "D"))


def read_points(path):
    data = path.read_bytes()
    if sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("Expected the exact released matched-side factorial result")
    result = json.loads(data)
    if result["status"] != "ACTUAL_CLOSED_AND_VERIFIED":
        raise ValueError("Expected the completed matched-side replay")
    points = []
    for cohort, key, count, seeds, label in VIEWS:
        score = result["cohorts"][cohort]["score"]
        view = score["views"][key]
        if (view["images"], tuple(score["seeds"])) != (count, seeds):
            raise ValueError("The view or seed census differs")
        ap = view["metrics_fraction"]
        contrasts = view["contrast_summaries_points"]
        values = []
        for arm, *_ in SERIES:
            paired = []
            for seed in seeds:
                before, after = (ap[f"{seed}/{name}"]["AP"] for name in ("FF", arm))
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (before, after)):
                    raise ValueError("Unexpected COCO mask AP fraction")
                paired.append(100 * (after - before))
            gain = fmean(paired)
            if not math.isclose(gain, contrasts[f"{arm}_minus_FF"]["AP"]["mean"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Recomputed two-sided-constraint delta differs from the aggregate")
            values.append(gain)
        points.append((label, values))
    return points


def draw(points, output):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "sar-matched-side-ablation",
    })
    fig = plt.figure(figsize=(3.49, 2.24))
    ax = fig.add_axes((.36, .235, .61, .60))
    ink, muted = "#172A39", "#455966"
    ax.axhspan(1.53, 2.47, color="#F0F4F6", zorder=0)
    ax.xaxis.set_major_locator(FixedLocator([0, .25, .50, .75, 1.00]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="x", color="#CFDAE0", linewidth=.52)
    ax.axvline(0, color="#81949E", linewidth=.65, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(2.47, -.47)
    ax.set_yticks(range(len(points)), [label for label, _ in points])
    ax.tick_params(axis="y", which="both", length=0, pad=6, labelsize=7.5, colors=ink)
    ax.tick_params(axis="x", length=2.5, pad=2, labelsize=7.3, colors=muted)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#A8B9C3")
    ax.spines["bottom"].set_linewidth(.6)
    ax.set_xlabel("Mask AP gain vs. both fixed (points)", fontsize=8.0, color=ink, labelpad=3)
    for index, (_, values) in enumerate(points):
        for offset, value, (_, _, color, marker) in zip((-.24, 0, .24), values, SERIES, strict=True):
            row = index + offset
            ax.hlines(row, 0, value, color=color, linewidth=.83)
            ax.plot(value, row, marker=marker, color=color, markersize=3.8)
            ax.text(value + .034, row, f"{value:+.3f}", va="center",
                    color=color, fontsize=7.1)
    handles = [Line2D([], [], color=color, marker=marker, lw=.8, markersize=3.7, label=label)
               for _, label, color, marker in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .982), ncol=3,
               frameon=False, handlelength=1.2, handletextpad=.35, columnspacing=.95,
               fontsize=7.7, labelcolor=ink)
    fig.text(.50, .037, "AP point estimates without intervals", ha="center", fontsize=7.2, color=muted)
    stamp = datetime(2026, 11, 23, tzinfo=timezone.utc)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": stamp, "ModDate": stamp})
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": stamp.isoformat()})
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=here.parent.parent / "results" / FILE)
    parser.add_argument("--output", type=Path, default=here / "independent_sides_ablation")
    args = parser.parse_args()
    draw(read_points(args.score), args.output)


if __name__ == "__main__":
    main()
