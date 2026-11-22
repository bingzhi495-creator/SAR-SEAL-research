"""Frozen signed disagreement pairs and grouped source ridge calibration."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np


WIDTH = 16
PENALTY = 0.05
SIDES = ("add", "drop")
SOURCE = "ATTRACTION_LOCAL_CORRECTION_FIT109_PAIRS_COMPLETE"
FIT = "ATTRACTION_LOCAL_CORRECTION_SOURCE_RIDGE_COMPLETE"
PROTOCOL = "7e5c54f9e0ea3489d158daaa114a643d1100175ce4f818696ead95f816d7b933"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pinned(path, fingerprint):
    data = path.read_bytes()
    require(digest(data) == fingerprint, "Digest changed for " + path.name)
    return data


def observed(directory, terminal, phase, status):
    event_raw = terminal.read_bytes()
    event = json.loads(event_raw)
    require(event["phase"] == phase and event["actual_exit"] == 0 and event["output_footer_verified"] is True,
            "A successful observed " + phase + " result is required")
    result = json.loads(pinned(directory/"RESULT_SAFE.json", event["result_sha256"]))
    require(result["status"] == status and result["protocol_sha256"] == PROTOCOL
            and event["worker_sha256"] == result["worker_sha256"], "Observed worker or result identity differs")
    return result, event["result_sha256"], digest(event_raw)


def pair_rows(coordinates, features, q, attraction, fixed_choices, learned_choices):
    """Return [static index, learned index, oriented feature difference] per disagreement."""
    coords = np.asarray(coordinates)
    raw = np.asarray(features, dtype=np.float64)
    probability = np.asarray(q, dtype=np.float64)
    prior = np.asarray(attraction, dtype=np.float64)
    n, k = len(coords), len(fixed_choices)
    require(coords.shape == probability.shape == prior.shape == (n, 2)
            and raw.shape == (n, 2, WIDTH-2) and len(learned_choices) == k
            and np.issubdtype(coords.dtype, np.integer) and np.all(coords >= 0)
            and np.all(np.isfinite(raw)) and np.all(np.isfinite(probability)) and np.all(np.isfinite(prior)),
            "Malformed signed endpoint features")
    if not k:
        return dict.fromkeys(SIDES, [])
    rows = {}
    fixed = np.asarray(fixed_choices)
    learned = np.asarray(learned_choices)
    require(fixed.shape == learned.shape == (k, 2) and np.issubdtype(fixed.dtype, np.integer)
            and np.issubdtype(learned.dtype, np.integer) and np.all((0 <= fixed) & (fixed < n))
            and np.all((0 <= learned) & (learned < n)), "Original selection ordinal differs")
    for column, side in enumerate(SIDES):
        a, b = set(fixed[:, column]), set(learned[:, column])
        require(len(a) == len(b) == k, "Repeated native endpoint choice")
        s_rows = sorted(a - b, key=lambda i: coords[i, column])
        l_rows = sorted(b - a, key=lambda i: coords[i, column])
        factor = 1 if column == 0 else -1
        values = np.column_stack((prior[:, column], probability[:, column], raw[:, column]))
        rows[side] = [
            [int(coords[s, column]), int(coords[l, column]), (factor * (values[l] - values[s])).tolist()]
            for s, l in zip(s_rows, l_rows, strict=True)
        ]
    return rows


def ridge(x, y, weight):
    scale = np.sqrt(np.sum(weight[:, None] * np.square(x), axis=0))
    scale = np.where(scale == 0, 1., scale)
    normalized = x / scale
    gram = (normalized.T * weight) @ normalized
    gram += PENALTY * np.eye(x.shape[1])
    target = normalized.T @ (weight*y)
    coefficient = np.linalg.solve(gram, target)
    require(np.max(np.abs(gram @ coefficient - target)) <= 1e-10, "Ridge normal equations did not close")
    error = y - normalized @ coefficient
    return dict(weight=coefficient.tolist(), rms=scale.tolist()), float(np.sum(weight * error**2))


def scores(x, model):
    scale, weight = np.asarray(model["rms"]), np.asarray(model["weight"])
    return np.asarray(x, dtype=np.float64)[:, :len(weight)] @ (weight / scale)


def fit(a):
    source, result_sha, terminal_sha = observed(a.source, a.source_terminal, "source", SOURCE)
    worker = digest(Path(__file__).read_bytes())
    require(source["gate_sha256"] == worker and source["FIT"] and not source["DEV"] and not source["TEST"]
            and not source["target_labels_opened"] and source["point_optimizer_updates"] == 1000
            and source["detector_forwards"] == 0, "Source pairs must be the original FIT outside-fold outputs")
    stream = pinned(a.source/"SOURCE_PAIRS_PRIVATE.jsonl", source["source_pairs_sha256"])
    samples, targets, groups = [], [], []
    per_side, group_to_fold, candidates = Counter(), {}, set()
    for line in stream.splitlines():
        row = json.loads(line)
        assert row["group"] and type(row["fold"]) is int
        group, fold = row["group"], row["fold"]
        require(group_to_fold.setdefault(group, fold) == fold, "Source group crosses folds")
        key = row["image_id"], row["native_rank"]
        require(key not in candidates, "Duplicate FIT109 original match")
        candidates.add(key)
        for side in SIDES:
            pairs = row[side]
            require(len(pairs) <= 8 and not ({p[0] for p in pairs} & {p[1] for p in pairs}), "Repeated FIT correction pair")
            for old, new, x, y in pairs:
                require(old != new and len(x) == WIDTH and -1 <= y <= 1, "Fractional source pair differs")
                samples.append(x)
                targets.append(y)
                groups.append(group)
                per_side[side] += 1
    require(samples and len(candidates) == 412 and len(set(group_to_fold.values())) == 5, "Source census differs")
    per_group = Counter(groups)
    weights = np.asarray([1/(len(per_group) * per_group[group]) for group in groups], dtype=np.float64)
    x, y = np.asarray(samples, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    require(x.shape == (len(y), WIDTH) and np.isfinite(x).all() and np.isfinite(y).all()
            and abs(weights.sum() - 1.) < 1e-12, "Invalid grouped source fit")
    models, losses = {}, {}
    for name, dims in (("C", 2), ("Q", WIDTH)):
        models[name], losses[name] = ridge(x[:, :dims], y, weights)
    model = dict(status=FIT, protocol_sha256=PROTOCOL, model=models)
    a.output.mkdir()
    raw = encoded(model)
    (a.output/"MODEL_PRIVATE.json").write_bytes(raw)
    return dict(status=FIT, protocol_sha256=PROTOCOL, worker_sha256=worker, source_result_sha256=result_sha,
                source_terminal_sha256=terminal_sha, source_pairs_sha256=source["source_pairs_sha256"],
                point_optimizer_updates=source["point_optimizer_updates"], ridge_solves=2, penalty=PENALTY,
                fit_rows_by_side=dict(per_side), fit_rows=len(y), fit_groups=len(per_group),
                original_source_matches=len(candidates), fit_group_weighted_MSE=losses,
                model_sha256=digest(raw), detector_forwards=0, target_labels_opened=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "source-terminal", "protocol", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    a = p.parse_args()
    require(a.protocol_sha256 == PROTOCOL and not a.output.exists(), "Require frozen protocol and fresh fit")
    pinned(a.protocol, PROTOCOL)
    start = perf_counter()
    result = fit(a)
    result["wall_seconds_inside_worker"] = perf_counter() - start
    raw = encoded(result)
    (a.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()

