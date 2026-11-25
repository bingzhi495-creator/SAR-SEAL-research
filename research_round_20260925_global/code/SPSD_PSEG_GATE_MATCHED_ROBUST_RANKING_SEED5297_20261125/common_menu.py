"""Common label-free action menu and two-reference preference readouts."""
from base64 import b64decode, b64encode
from collections import defaultdict
from itertools import product

import numpy as np


HEADS = ("regression", "ranking")
TOL = 1e-12


def pack(value, dtype):
    return b64encode(np.asarray(value, dtype=dtype).tobytes()).decode("ascii")


def unpack(encoded, dtype, shape):
    return np.frombuffer(b64decode(encoded, validate=True), dtype=dtype).reshape(shape)


def differences(values, actions):
    """Exact change in signed intersection under each reference."""
    add = values[actions[:, 0], 0, :].sum(axis=1)
    drop = values[actions[:, 1], 1, :].sum(axis=1)
    gain = add - drop
    return gain - gain[0]


def freeze_menu(coordinates, posteriors, incumbent, choices, k, selector):
    """Save a union of independently generated finite menus, with no new write action."""
    menu = set()
    for key in HEADS:
        p = np.asarray(posteriors[key], dtype=np.float64)
        a = selector._candidate_sets(p[:, 0, :], incumbent[0], k, largest=True)
        d = selector._candidate_sets(p[:, 1, :], incumbent[1], k, largest=False)
        menu.update(product(a, d))
    menu = [incumbent, *sorted(menu - {incumbent})]
    index = {action: i for i, action in enumerate(menu)}
    chosen = [index[choices[key + "_robust"]] for key in HEADS]
    actions = np.asarray(menu, dtype=np.uint8)
    pred = np.stack([differences(np.asarray(posteriors[key], dtype=np.float64), actions).min(axis=1) / (2*k)
                     for key in HEADS], axis=-1)
    return dict(k=k, n=len(coordinates), actions_count=len(actions), chosen=chosen,
                coordinates=pack(coordinates, "<i4"), actions=pack(actions, "u1"), predicted_min=pack(pred, "<f8"))


def read_menu(doc):
    """Independently validate candidate feasibility and recover stored coordinates and scores."""
    k, n, m = (doc[name] for name in ("k", "n", "actions_count"))
    if not 0 < k <= n <= 32 or not 0 < m <= 287:
        raise ValueError("Original positive quota, endpoint count and legal union size")
    coords = unpack(doc["coordinates"], "<i4", (n,2))
    actions = unpack(doc["actions"], "u1", (m,2,k))
    pred = unpack(doc["predicted_min"], "<f8", (m,2))
    chosen = doc["chosen"]
    keys = [(tuple(a), tuple(d)) for a,d in actions]
    if (len(set(coords.ravel().tolist())) != coords.size or np.any(coords < 0) or np.any(actions >= n) or
            np.any(np.diff(actions.astype(np.int16), axis=2) <= 0) or
            len(set(keys)) != m or sorted(keys[1:]) != keys[1:] or
            not np.isfinite(pred).all() or not np.all(pred[0] == 0.) or
            np.max(abs(pred)) > 1. + TOL or len(chosen) != 2 or not all(0 <= i < m for i in chosen)):
        raise ValueError("Unique feasible ordered menu, exact static anchor and finite scores")
    return coords, actions, pred, chosen


def preference(truth, estimates):
    """Fraction of correctly ordered true unequal pairs. Predicted ties count one half."""
    i, j = np.triu_indices(len(truth), k=1)
    signs = np.sign(truth[i] - truth[j])
    v = signs != 0
    if not np.any(v):
        return 0, None
    gap = (estimates[i] - estimates[j])[v]
    order = np.where(np.abs(gap) > TOL, np.sign(gap), 0.0)
    direction = signs[v, None] * order
    accuracy = (np.count_nonzero(direction > 0, axis=0) + .5 * np.count_nonzero(direction == 0, axis=0)) / len(direction)
    return len(direction), accuracy


def measure(values, actions, estimates, selected, k):
    """Read selected outcome and both heads' preferences on identical truth menu actions."""
    exact = differences(values, actions).min(axis=-1)
    pairs, accuracy = preference(exact, estimates)
    direct = sorted({0, *selected})
    direct_pairs, direct_accuracy = preference(exact[direct], estimates[direct])
    outcome = exact[selected]
    return dict(full_pairs=pairs, full_accuracy=accuracy, direct_pairs=direct_pairs,
                direct_accuracy=direct_accuracy, minimum_raw=outcome, minimum_normalized=outcome/(2*k),
                best_menu_regret_raw=exact.max()-outcome,
                nonstatic=np.asarray(selected) != 0, harmful=(outcome < 0) & (np.asarray(selected) != 0))


def summarize(rows):
    """Equal decoded-group mechanism summaries with paired 4,096 draws."""
    out = {}
    for arm in ("full", "direct", "minimum_raw", "minimum_normalized", "best_menu_regret_raw"):
        grouped = defaultdict(lambda: defaultdict(list))
        n = pairs = 0
        for iid, g, v in rows:
            score = v.get(arm + "_accuracy") if arm in ("full", "direct") else v[arm]
            if score is not None:
                grouped[g][iid].append(score)
                pairs += v.get(arm + "_pairs", 0)
                n += 1
        if not n:
            out[arm] = dict(candidates=0, images=0, groups=0)
            continue
        table = np.stack([np.mean([np.mean(v, axis=0) for v in grouped[g].values()], axis=0)
                          for g in sorted(grouped)])
        delta = table[:,1] - table[:,0]
        draws = np.random.Generator(np.random.PCG64(5297)).integers(len(table), size=(4096,len(table)))
        out[arm] = dict(candidates=n, images=sum(map(len, grouped.values())), groups=len(grouped),
                        regression_ranking_group_means=table.mean(axis=0).tolist(),
                        rank_minus_regression=float(delta.mean()),
                        percentile95=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
        if arm in ("full", "direct"):
            out[arm]["true_ordered_pairs"] = pairs
    chose = np.sum([v["nonstatic"] for _,_,v in rows], axis=0)
    harmful = np.sum([v["harmful"] for _,_,v in rows], axis=0)
    out["nonstatic_and_harmful_hulls_regression_ranking"] = dict(nonstatic=chose.tolist(), harmful=harmful.tolist())
    return out
