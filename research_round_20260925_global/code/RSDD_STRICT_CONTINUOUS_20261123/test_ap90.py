"""Check weighted strict AP against explicit replicated detection sequences."""
import runpy
from pathlib import Path

import numpy as np


def expanded_ap(events, groups, weights):
    counts = {1: int(weights[0]), 2: int(weights[1])}
    positives, detections = 0, []
    for image in sorted(events):
        copies = counts[groups[image]]
        positives += events[image]["positive_gt"] * copies
        for _, score, ignored, identity in events[image]["detections"]:
            if not ignored:
                detections += [(score, int(identity > 0))] * copies
    ordered = sorted(detections, key=lambda x: -x[0])
    if not ordered:
        return 0.
    tp = np.cumsum([x[1] for x in ordered])
    precision = tp / (np.arange(1, len(tp)+1) + np.spacing(1))
    recall = tp / positives
    return np.mean([max((p for p, r in zip(precision, recall) if r >= q), default=0.)
                    for q in np.linspace(0, 1, 101)])


def main():
    ap90 = runpy.run_path(str(Path(__file__).with_name("ap90.py")))
    events = {
        2: dict(positive_gt=2, detections=[(1, .97, False, 3), (2, .50, False, 0)]),
        1: dict(positive_gt=1, detections=[(1, .97, False, 0), (2, .80, True, 0), (3, .33, False, 1)]),
        3: dict(positive_gt=1, detections=[(1, .44, False, 0)]),
    }
    groups = {1: 1, 2: 2, 3: 1}
    weights = np.array([[1, 1], [3, 0], [0, 3], [2, 1], [1, 2]])
    curve = ap90["score_weighted"](events, groups, (1, 2), weights)
    for w, score in zip(weights, curve, strict=True):
        assert abs(score-expanded_ap(events, groups, w)) < 1e-14
    empty = {1: dict(positive_gt=1, detections=[(1,.9,True,0)])}
    assert ap90["score_weighted"](empty, {1:1}, (1,), np.array([[2]]))[0] == 0.
    print("Expanded and compact strict AP agree, including ties, ignores and absent groups.")


if __name__ == "__main__":
    main()
