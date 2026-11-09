"""Paired group sensitivity for the fixed, independently closed HRSID DEV705 predictions."""

from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from types import ModuleType
import math

import numpy as np


BOOTSTRAP_DRAWS = 2048
DRAW_SEED = 5213
SEEDS = (5201, 5202)
METRICS = ("AP", "AP75")
ADAPTER_PATH = (Path(__file__).resolve().parent.parent /
                "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py")
ADAPTER_SHA = "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_official_adapter():
    with ADAPTER_PATH.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == ADAPTER_SHA,
            "Exact original official COCO adapter required")
    adapter = ModuleType("_dev705_group_original_official_coco")
    adapter.__file__ = str(ADAPTER_PATH)
    exec(compile(raw, str(ADAPTER_PATH), "exec"), adapter.__dict__)
    return adapter


def prepare_closed_arm(gt, predictions, groups_by_image_id, expected):
    """Repeat the original evaluator once and check it before accepting its matching cache."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from dev705_group_ap import assert_all_ones, prepare_group_ap

    require(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Exact original DEV705 CPU COCO runtime required")
    require(type(expected) is dict and set(expected) == set(METRICS)
            and all(type(x) is float and math.isfinite(x) and 0. <= x <= 1. for x in expected.values()),
            "Exact unrounded accepted arm reference AP and AP75 required")
    adapter = load_official_adapter()
    image_ids = [image["id"] for image in gt["images"]]
    require(len(image_ids) == len(set(image_ids)) == len(groups_by_image_id) == 705
            and set(groups_by_image_id) == set(image_ids) and len(set(groups_by_image_id.values())) == 28
            and len(gt["annotations"]) == 1782 and [row["id"] for row in gt["categories"]] == [1],
            "Exact original DEV705 population and proxy groups required")
    coco = COCO()
    coco.dataset = deepcopy(gt)
    captured = []

    def capture(*args):
        evaluator = COCOeval(*args)
        captured.append(evaluator)
        return evaluator

    with redirect_stdout(StringIO()):
        coco.createIndex()
        stats = adapter.evaluate_population(dict(COCO=COCO, COCOeval=capture), coco, predictions, image_ids)
    original = dict(AP=stats[0], AP75=stats[2])
    require(all(math.isclose(original[m], expected[m], rel_tol=0., abs_tol=1e-12) for m in METRICS),
            "Independent canonical COCO scores must reproduce the original closed arm")
    evaluator, = captured
    cache = prepare_group_ap(evaluator, groups_by_image_id)
    assert_all_ones(cache, expected)
    return cache


def metric_pairs(caches, counts):
    results = {}
    for seed, (native, center) in caches.items():
        left, right = native.scores(counts), center.scores(counts)
        require(all(math.isfinite(x) and (x == -1. or 0. <= x <= 1.)
                    for m in METRICS for x in (left[m], right[m])),
                "Finite original COCO score or no-GT sentinel required")
        require(all((left[m] < 0) == (right[m] < 0) for m in METRICS),
                "Both arms must have the same valid GT population")
        results[seed] = {m: None if left[m] < 0 else (left[m], right[m]) for m in METRICS}
    require(all((results[5201][m] is None) == (results[5202][m] is None) for m in METRICS),
            "Both downstream seeds must share the valid GT population")
    return results


def gain(pair):
    return 100. * (pair[1] - pair[0])


def summarize_caches(caches, *, progress=None):
    """Use the same group multiplicities for both arms of both completed seeds."""
    require(type(caches) is dict and set(caches) == set(SEEDS), "Exactly the closed 5201 and 5202 pairs required")
    groups = caches[5201][0].groups
    require(len(groups) >= 2 and all(cache.groups == groups for pair in caches.values() for cache in pair),
            "Exact common group ordering required")
    count = len(groups)
    all_ones = np.ones(count, dtype=np.int64)
    fixed = metric_pairs(caches, all_ones)
    require(all(fixed[s][m] is not None for s in SEEDS for m in METRICS), "Valid fixed primary AP required")
    draws = {s: {m: [] for m in METRICS} for s in SEEDS}
    omissions = {s: {m: [] for m in METRICS} for s in SEEDS}
    rng = np.random.Generator(np.random.PCG64(DRAW_SEED))
    sampled = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    for ordinal, picks in enumerate(sampled, 1):
        scores = metric_pairs(caches, np.bincount(picks, minlength=count))
        for s in SEEDS:
            for m in METRICS:
                if scores[s][m] is not None:
                    draws[s][m].append(gain(scores[s][m]))
        if progress is not None and ordinal % 256 == 0:
            progress(ordinal)
    for omitted in range(count):
        weights = all_ones.copy()
        weights[omitted] = 0
        scores = metric_pairs(caches, weights)
        for s in SEEDS:
            for m in METRICS:
                if scores[s][m] is not None:
                    omissions[s][m].append(gain(scores[s][m]))

    results = {}
    for s in SEEDS:
        per_seed = {}
        for m in METRICS:
            draws_m = np.asarray(draws[s][m])
            loo_m = np.asarray(omissions[s][m])
            require(draws_m.size > 0 and loo_m.size > 0, "At least one valid proxy resample and omission required")
            lower, median, upper = np.quantile(draws_m, (.025, .5, .975), method="linear")
            a, b = fixed[s][m]
            per_seed[m] = dict(
                fixed_native_fraction=a, fixed_center_fraction=b, fixed_gain_points=gain((a, b)),
                bootstrap=dict(valid_draws=draws_m.size, no_gt_draws=BOOTSTRAP_DRAWS - draws_m.size,
                               positive_draws=int((draws_m > 0).sum()),
                               percentile_025_gain_points=float(lower), median_gain_points=float(median),
                               percentile_975_gain_points=float(upper)),
                leave_one_out=dict(valid_omissions=loo_m.size, no_gt_omissions=count - loo_m.size,
                                   positive_omissions=int((loo_m > 0).sum()),
                                   min_gain_points=float(loo_m.min()), max_gain_points=float(loo_m.max())),
            )
        results[str(s)] = per_seed
    return dict(groups=count, bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=DRAW_SEED,
                generator="numpy_PCG64", seeds=results)

