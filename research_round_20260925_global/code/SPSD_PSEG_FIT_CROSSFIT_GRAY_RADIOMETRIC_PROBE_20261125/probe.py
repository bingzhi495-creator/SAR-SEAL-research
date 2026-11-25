"""Probe incremental grayscale information in source FIT action gains."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from scipy.stats import rankdata


LOCK = "0e8c080a6c6cabfb7900c4cde4e876496fa58a8eca4e4285c878b6c28f9a9a38"
BANK = "5419c9179f1ca82bf21436049450e623725c7e45b1cb2b7b19c2541e57e26fb7"
METHODS = ("native", "raw", "pooled", "per_side")
FEATURES = ((0,), (0, 1), (0, 2), (0, 3))
RIDGE = .001


def sha(raw):
    return sha256(raw).hexdigest()


def pin(path, digest):
    raw = path.read_bytes()
    if sha(raw) != digest:
        raise ValueError("Unexpected input digest: " + path.name)
    return raw


def load(path, digest):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), module.__dict__)
    return module


def midrank(x, *, axis=None):
    if axis is None:
        width = x.size
    else:
        width = x.shape[axis]
    return (rankdata(x, method="average", axis=axis).reshape(x.shape) - .5) / width - .5


def vectors(packet):
    feature = packet.features
    gray = feature[:, :, 9].astype(np.float64)
    candidate = np.stack((feature[:, :, 0], gray, midrank(gray), midrank(gray, axis=0)), axis=-1)
    delta = np.einsum("aks,ksf->af", packet.action_weights, candidate) / (2 * packet.k)
    truth = np.einsum("aks,ks->a", packet.action_weights, packet.targets.mean(axis=-1)) / (2 * packet.k)
    return delta, truth


def ridge(x, y, w):
    w = w / w.sum()
    rms = np.sqrt(w @ np.square(x))
    rms[rms < 1e-12] = 1
    z = x / rms
    covariance = z.T @ (w[:, None] * z)
    return np.linalg.solve(covariance + RIDGE * np.eye(x.shape[1]), z.T @ (w * y)) / rms


def summarize(group_metric, draws):
    return dict(points=float(group_metric.mean()),
                percentile95_points=np.percentile(group_metric[draws].mean(axis=1), [2.5, 97.5]).tolist())


def main(a):
    started = perf_counter()
    pin(a.protocol, LOCK)
    bank = load(a.deployment_fit / "fit_banks.py", BANK)
    terminal = json.loads(pin(a.previous / "observed_fit/TERMINAL.json", bank.ORIGINAL_TERMINAL))
    bank.need(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
              and terminal["result_sha256"] == bank.ORIGINAL_RESULT
              and terminal["worker_sha256"] == bank.ORIGINAL, "Exact closed original FIT authority")
    previous = json.loads(pin(a.previous / "actual/RESULT_SAFE.json", bank.ORIGINAL_RESULT))
    original = load(a.previous / "fit_action.py", bank.ORIGINAL)
    action = load(a.action_core, original.ACTION_CORE)
    attraction = load(a.attraction, original.ATTRACTION)
    packets, counts, packet_sha, _ = bank.original_packets(a, original, action, attraction, previous)
    group = [packet.group_id for packet in packets]
    image = [packet.image_id for packet in packets]
    folds = bank.fixed_folds(group)
    group_ids = sorted(set(group))
    bank.need(len(packets) == 848 and len(group_ids) == 439
              and len(set(zip(group, image))) == 447, "Exact group and candidate counts")

    group_index = {value: i for i, value in enumerate(group_ids)}
    images = defaultdict(set)
    image_counts = Counter(zip(group, image, strict=True))
    for g, i in image_counts:
        images[g].add(i)
    comparisons = [vectors(packet) for packet in packets]
    lengths = np.array([len(y) for _, y in comparisons])
    x = np.vstack([features for features, _ in comparisons])
    y = np.concatenate([target for _, target in comparisons])
    weights = np.repeat([1 / (len(images[g]) * image_counts[g, i] * n)
                         for g, i, n in zip(group, image, lengths, strict=True)], lengths)
    fold_per_action = np.repeat(folds, lengths)
    group_per_action = np.repeat([group_index[g] for g in group], lengths)
    same_total = np.bincount(group_per_action, weights=weights)
    np.testing.assert_allclose(same_total, np.ones_like(same_total), atol=1e-12)
    bank.need(len(x) == len(y) == 5042 and np.isfinite(x).all() and np.isfinite(y).all(), "Exact action population")
    errors = np.empty((len(METHODS), len(y)))
    coefficients = {}
    for name, columns, error in zip(METHODS, FEATURES, errors, strict=True):
        xx = x[:, columns]
        each = []
        for fold in range(5):
            test = fold_per_action == fold
            train = ~test
            beta = ridge(xx[train], y[train], weights[train])
            error[test] = xx[test] @ beta - y[test]
            each.append(beta.tolist())
        coefficients[name] = each
    group_abs = np.stack([
        np.bincount(group_per_action, weights=weights * np.abs(err)) / same_total for err in errors
    ]) * 100
    r = np.random.Generator(np.random.PCG64(5297))
    draws = r.integers(len(group_ids), size=(4096, len(group_ids)))
    status = "SPSD_PSEG_FIT_GRAY_CROSSFIT_PROBE_COMPLETE"
    receipt = dict(status=status, protocol_sha256=LOCK, original_fit_result_sha256=bank.ORIGINAL_RESULT,
                   worker_sha256=sha(Path(__file__).read_bytes()), exact_packet_sha256=packet_sha,
                   counts=counts, folds=5, bootstrap=4096, groups=len(group_ids), candidates=len(packets),
                   action_count=len(y), source_gate_labels_opened=False, historical_test_opened=False,
                   extra_detector_forwards=0, linear_solutions=20, fold_coefficients=coefficients,
                   elapsed_seconds=perf_counter() - started,
                   by_method={name: dict(mae_budget_pct=summarize(group_abs[i], draws),
                                         rmse_budget_pct=float(100 * np.sqrt(weights @ np.square(errors[i]) / weights.sum())))
                              for i, name in enumerate(METHODS)},
                   contrasts={title: summarize(group_abs[i] - group_abs[j], draws)
                              for title, i, j in (("pooled_minus_raw", 2, 1), ("pooled_minus_per_side", 2, 3),
                                                  ("raw_minus_native", 1, 0), ("per_side_minus_raw", 3, 1))})
    a.out.mkdir()
    payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (a.out / "RESULT_SAFE.json").open("xb") as result:
        result.write(payload)
    print(json.dumps(dict(status=status, result_sha256=sha(payload))), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("protocol", "deployment-fit", "capture", "points", "labels", "prior-fit", "previous",
                 "action-core", "source-core", "attraction", "selector", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
