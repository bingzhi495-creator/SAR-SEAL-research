"""Paired group-weighted AP90 from exact per-image COCO greedy matches."""

from collections import Counter

import numpy as np


RECALLS = np.linspace(0., 1., 101)


def weights_for(groups, draws=4096):
    keys = tuple(sorted(set(groups.values())))
    random = np.random.Generator(np.random.PCG64(5217))
    samples = random.integers(len(keys), size=(draws, len(keys)))
    counts = np.stack([np.count_nonzero(samples == i, axis=1) for i in range(len(keys))], axis=1)
    return keys, np.vstack((np.ones(len(keys), dtype=np.int64), counts))


def score_weighted(events, groups, keys, weights):
    """Replicate each score-ordered detection proportionally to its group weight."""
    by_group = {g: i for i, g in enumerate(keys)}
    gt = np.zeros(len(keys), dtype=np.int64)
    detections = []
    for image in sorted(events):
        record = events[image]
        group = by_group[groups[image]]
        gt[group] += record["positive_gt"]
        detections.extend((score, group, match > 0) for _, score, ignored, match in record["detections"]
                          if not ignored)
    positives = weights @ gt
    out = np.zeros(len(weights), dtype=np.float64)
    out[positives == 0] = np.nan
    # Between true positives recall is flat and precision cannot improve.
    dt, tp = np.zeros_like(gt), np.zeros_like(gt)
    dt_at_tp, tp_at_tp, tp_groups = [], [], []
    for _, group, matched in sorted(detections, key=lambda x: -x[0]):
        dt[group] += 1
        if matched:
            tp[group] += 1
            dt_at_tp.append(dt.copy())
            tp_at_tp.append(tp.copy())
            tp_groups.append(group)
    if not tp_groups:
        return out
    cum_dt = weights @ np.asarray(dt_at_tp).T
    cum_tp = weights @ np.asarray(tp_at_tp).T
    for i, (p, d, t) in enumerate(zip(positives, cum_dt, cum_tp, strict=True)):
        if p:
            active = weights[i, tp_groups] > 0
            recall = t[active] / p
            precision = t[active] / (d[active] + np.spacing(1))
            best = np.maximum.accumulate(precision[::-1])[::-1]
            locations = np.searchsorted(recall, RECALLS, side="left")
            # A recall level within repeated adjacent TPs attains maximum precision
            # at the end of that block, also its cumulative recall boundary.
            out[i] = np.sum(best[locations[locations < len(best)]]) / len(RECALLS)
    return out


def transitions(raster, continuous):
    """Count exact, same-detection outcome transitions on matched evaluation rows."""
    totals = Counter({f"{a}_to_{b}": 0 for a in ("tp", "fp", "ignore") for b in ("tp", "fp", "ignore")})
    totals.update(tp_identity_changed=0, raster_only_truth_matches=0, continuous_only_truth_matches=0)
    if set(raster) != set(continuous):
        raise ValueError("Representation changed the evaluation image identities")
    for image, r in raster.items():
        c = continuous[image]
        if r["positive_gt"] != c["positive_gt"]:
            raise ValueError("Representation changed the positive GT denominator")
        truth_r, truth_c = set(), set()
        for left, right in zip(r["detections"], c["detections"], strict=True):
            if left[:2] != right[:2]:
                raise ValueError("Representation changed the candidate identity or score")
            _, _, ignored_r, mr = left
            _, _, ignored_c, mc = right
            a = "ignore" if ignored_r else "tp" if mr else "fp"
            b = "ignore" if ignored_c else "tp" if mc else "fp"
            totals[f"{a}_to_{b}"] += 1
            if a == "tp":
                truth_r.add(mr)
            if b == "tp":
                truth_c.add(mc)
            totals["tp_identity_changed"] += int(a == b == "tp" and mr != mc)
        totals["raster_only_truth_matches"] += len(truth_r-truth_c)
        totals["continuous_only_truth_matches"] += len(truth_c-truth_r)
    return dict(totals)


def interval(curve):
    if not np.isfinite(curve).all():
        raise ValueError("The locked bootstrap needs positive GT support in every draw")
    return dict(delta_points=float(100*curve[0]),
                interval_95_points=(100*np.percentile(curve[1:], [2.5,97.5])).tolist())
