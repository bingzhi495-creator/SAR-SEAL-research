"""Render the three fixed-count actions from one verified grid example."""

import argparse
from itertools import combinations, permutations
from pathlib import Path
import runpy
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np

HERE = Path(__file__).resolve().parent
style = SimpleNamespace(**runpy.run_path(str(HERE / "draw_core_method_redesign.py")))
example = SimpleNamespace(**runpy.run_path(str(HERE / "draw_matched_budget_pipeline.py")))


ADD = ((2, 1), (4, 1), (4, 2), (5, 2))
QUOTA = 2


def actions():
    distance = ((np.array(ADD)[:, None, :] - np.array(example.DROP)[None, :, :]) ** 2).sum(axis=2)
    size = len(ADD)
    indices = range(size)
    full = sorted((int(sum(distance[i, j] for i, j in enumerate(p))), p)
                  for p in permutations(indices))
    assert full[0][0] < full[1][0]
    assigned = full[0][1]
    def gain(a, d):
        return example.ADD_SCORE[a] - example.DROP_SCORE[d]

    baseline = sorted(enumerate(assigned), key=lambda pair: -gain(*pair))[:QUOTA]
    assert all(gain(a, d) > 0 for a, d in baseline)
    cap = max(distance[a, d] for a, d in baseline)
    feasible = [
        tuple(zip(a, d, strict=True))
        for a in combinations(indices, QUOTA)
        for d in permutations(indices, QUOTA)
        if all(distance[i, j] <= cap for i, j in zip(a, d, strict=True))
    ]
    bounded = max(feasible, key=lambda pairs: sum(gain(a, d) for a, d in pairs))
    free = tuple(zip(np.argsort(-example.ADD_SCORE)[:QUOTA], np.argsort(example.DROP_SCORE)[:QUOTA], strict=True))
    pairings = (baseline, bounded, free)

    results = []
    for pairing in pairings:
        out = example.BASE_MASK.copy()
        plus, minus = [], []
        for a, d in pairing:
            x, y = ADD[a]
            assert not out[y, x]
            out[y, x] = True
            plus.append((x, y))
            x, y = example.DROP[d]
            assert out[y, x]
            out[y, x] = False
            minus.append((x, y))
        assert (out & ~example.BASE_MASK).sum() == (~out & example.BASE_MASK).sum() == QUOTA
        results.append((out, plus, minus, sum(gain(a, d) for a, d in pairing)))
    assert cap == 2
    np.testing.assert_allclose([out[3] for out in results], (0.89, 0.99, 1.44))
    return results


def draw():
    fig, ax = style.frame(132)
    lefts = (10, 178, 344)
    names = ("Distance-first", "Bounded re-pairing", "Independent ranking")
    descriptions = ("Rank the assigned pairs", "Optimize under the cap", "Rank each side separately")
    limits = (r"$\rho=\sqrt{2}$ pixels", r"$\rho=\sqrt{2}$ pixels", "No displacement cap")
    for index, (x, name, description, limit, result) in enumerate(
        zip(lefts, names, descriptions, limits, actions(), strict=True)
    ):
        mask, added, removed, gain = result
        style.heading(ax, x, 13, "abc"[index], name)
        style.label(ax, x + 13, 32, description, size=7.5, color=style.MUTED)
        style.grid(ax, x + 13, 42, 8.8, mask, added, removed, edits=True)
        style.label(ax, x + 91, 60, rf"$\widehat g={gain:.2f}$", size=9.2, weight="bold", color=style.TEAL)
        style.label(ax, x + 91, 76, limit, size=7.1, color=style.MUTED)
    style.rule(ax, 10, 23, 505, 23, color=style.NAVY)
    for x in (168, 334):
        style.rule(ax, x, 31, x, 97)
    ax.axhspan(106, 124, xmin=10 / 514.8, xmax=505 / 514.8, color=style.PAPER)
    style.label(ax, 18, 115, r"Worked example   Same mask, endpoint scores and budget of $k=2$ additions and removals.",
          size=7.45, color=style.MUTED)
    return fig


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    mpl.rcParams.update({"font.family": "sans-serif",
                        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                        "mathtext.fontset": "stixsans", "svg.fonttype": "none",
                        "svg.hashsalt": "sar-signed-selection-controls", "pdf.fonttype": 42})
    fig = draw()
    for suffix in ("svg", "pdf"):
        fig.savefig(args.output / f"method_bounded_selection.{suffix}")


if __name__ == "__main__":
    main()
