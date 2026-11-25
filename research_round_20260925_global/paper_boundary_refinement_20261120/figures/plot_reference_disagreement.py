"""Plot paired annotation effects using only released aggregate scores."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPARISONS = (
    ("free_natural_minus_raw", "Natural count\nvs. native"),
    ("free_matched_count_minus_fixed", "Independent\nvs. distance-paired"),
    ("learned_at_attraction_count_minus_attraction_static_at_dynamic_count",
     "Occupancy ranking\nvs. static attraction"),
)
NAVY = "#174567"
ORANGE = "#A85328"
INK = "#152430"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def estimate(axis, result, y, color, marker, filled, label):
    value = result["points"]
    lower, upper = result["percentile95_points"]
    assert lower <= value <= upper
    axis.errorbar(
        value, y, xerr=[[value - lower], [upper - value]], marker=marker,
        ls="none", mfc=color if filled else "white", mec=color, ms=4.2,
        mew=1.04, color=color, elinewidth=1.16, capsize=2, label=label, zorder=3,
    )


def render(results, destinations):
    decomp = read_json(results / "SPSD_PSEG232_SHARED_HULL_DECOMPOSITION_20261125_SAFE.json")
    scores = read_json(results / "SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125_SAFE.json")
    assert decomp["exact_both_reference_parity"]
    assert decomp["bootstrap_replicates"] == scores["bootstrap_replicates"] == 4096

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.2,
        "axes.titlesize": 9, "axes.titleweight": "bold", "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
        "savefig.facecolor": "white",
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.05, 2.48), gridspec_kw={"wspace": .19})
    fig.subplots_adjust(left=.193, right=.987, bottom=.258, top=.877)
    position = (2, 1, 0)
    for y, (key, _) in zip(position, COMPARISONS):
        row = decomp["comparisons"][key]
        pseg = scores["references"]["PSeg_SSDD"]["pairwise"][key]["geometry"]
        spsd = scores["references"]["SPSD"]["pairwise"][key]["geometry"]
        full = row["full_shift"]
        label_only = row["components"]["same_candidate_label_disagreement"]
        assert math.isclose(spsd["points"] - pseg["points"], full["points"], abs_tol=1e-11)
        assert math.isclose(sum(c["points"] for c in row["components"].values()),
                            full["points"], abs_tol=1e-11)
        labels = (("PSeg-SSDD", "SPSD"), ("Total shift", "Conflicting labels")) if y == 2 else (
            (None, None), (None, None))
        estimate(left, pseg, y+.145, NAVY, "o", True, labels[0][0])
        estimate(left, spsd, y-.145, ORANGE, "s", False, labels[0][1])
        estimate(right, full, y+.145, NAVY, "o", True, labels[1][0])
        estimate(right, label_only, y-.145, ORANGE, "s", False, labels[1][1])

    for axis in (left, right):
        axis.set(ylim=(-.64, 2.43), xlim=(-.80, .26), xticks=(-.75, -.50, -.25, 0, .25))
        axis.set_yticks(position)
        axis.tick_params(axis="both", length=0)
        axis.tick_params(axis="y", pad=7)
        axis.grid(axis="x", color="#E3E9ED", lw=.6)
        axis.axvline(0, color="#84959D", lw=.8, zorder=1)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, -.365),
                    frameon=False, handletextpad=.4, columnspacing=.85,
                    borderpad=0, borderaxespad=0, fontsize=7.1)
    left.set_yticklabels([label for _, label in COMPARISONS])
    right.tick_params(labelleft=False)
    left.set_title("(a) Same masks, different references", loc="left", pad=7)
    right.set_title("(b) Change from PSeg-SSDD to SPSD", loc="left", pad=7)
    left.set_xlabel("Raster IoU effect (points)", labelpad=4)
    right.set_xlabel("Difference in raster IoU effects (points)", labelpad=4)
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg"):
            fig.savefig(directory / f"reference_disagreement_effects.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, nargs="+", required=True)
    cli = parser.parse_args()
    render(cli.results, cli.figures)
