"""Paired DEV705 composition diagnostics from original area-specific COCO matching.

The caller admits already closed predictions and the exact real DEV705 roster.
The portable core also accepts smaller single-category COCO fixtures. Add the
directory containing dev705_group_ap.py and dev705_group_sensitivity.py to
sys.path before importing this module.
"""

from contextlib import redirect_stdout
from copy import copy, deepcopy
from importlib.metadata import version
from io import StringIO
import math
from types import SimpleNamespace

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dev705_group_ap import MAX_GROUPS, prepare_group_ap, require
from dev705_group_sensitivity import BOOTSTRAP_DRAWS, DRAW_SEED, load_official_adapter


METRIC_INDEX = {"AP": 0, "AP75": 2, "APl": 5}
METRIC_AREA = {"AP": "all", "AP75": "all", "APl": "large"}
AREAS = ((0, 1e10), (0, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, 1e10))


def prepare_large_cache(evaluator, image_groups):
    """Pass only the official large-area matched rows to the frozen rank kernel."""
    params = evaluator._paramsEval
    require(tuple(params.areaRngLbl) == ("all", "small", "medium", "large")
            and tuple(map(tuple, params.areaRng)) == AREAS,
            "Expected canonical COCO size ranges")
    image_count = len(params.imgIds)
    require(len(evaluator.evalImgs) == len(AREAS) * image_count,
            "Expected the complete original area-by-image evaluation cache")
    large_rows = evaluator.evalImgs[3 * image_count:]
    require(all(row is None or tuple(row["aRng"]) == AREAS[3] for row in large_rows),
            "Expected original large-area evaluation rows")
    # The frozen accumulator accepts an all-area metadata label. Only adapt the
    # metadata for this view. Original dtMatches, gtIgnore and dtIgnore are kept.
    rank_params = copy(params)
    rank_params.areaRngLbl = ["all"]
    rank_params.areaRng = [list(AREAS[0])]
    rows = [None if row is None else dict(row, aRng=rank_params.areaRng[0]) for row in large_rows]
    view = SimpleNamespace(_paramsEval=rank_params, evalImgs=rows, eval={})
    return prepare_group_ap(view, image_groups)


def prepare_arm(adapter, gt, predictions, image_groups, expected):
    coco = COCO()
    coco.dataset = deepcopy(gt)
    captured = []

    def capture(*args):
        evaluator = COCOeval(*args)
        captured.append(evaluator)
        return evaluator

    with redirect_stdout(StringIO()):
        coco.createIndex()
        stats = adapter.evaluate_population(dict(COCO=COCO, COCOeval=capture), coco, predictions,
                                            sorted(image_groups))
    evaluator, = captured
    caches = {
        "all": prepare_group_ap(evaluator, image_groups),
        "large": prepare_large_cache(evaluator, image_groups),
    }
    values = score_arm(caches, np.ones(len(caches["all"].groups), dtype=np.int32))
    for name, index in METRIC_INDEX.items():
        require(math.isclose(stats[index], expected[name], rel_tol=0., abs_tol=1e-12)
                and math.isclose(values[name], stats[index], rel_tol=0., abs_tol=1e-12),
                f"Closed {name}, original COCO and full-composition cache must agree")
    return caches


def score_arm(caches, weights):
    values = caches["all"].scores(weights)
    values["APl"] = caches["large"].scores(weights)["AP"]
    return values


def paired_scores(arms, weights):
    native, center = (score_arm(caches, weights) for caches in arms)
    for name in METRIC_INDEX:
        a, b = native[name], center[name]
        require((a == b == -1.) or (0. <= a <= 1. and 0. <= b <= 1.),
                "Paired metrics must share a valid or undefined COCO population")
    return {
        name: dict(native=None, center=None, delta_points=None) if native[name] == -1.
        else dict(native=native[name], center=center[name], delta_points=100. * (center[name] - native[name]))
        for name in METRIC_INDEX
    }


def sign_summary(values):
    valid = [value for value in values if value is not None]
    return valid, dict(valid_draws=len(valid), no_gt_draws=len(values) - len(valid),
                       positive_draws=sum(value > 0. for value in valid),
                       negative_draws=sum(value < 0. for value in valid),
                       zero_draws=sum(value == 0. for value in valid))


def analyze_group_composition(gt, native_rows, center_rows, groups_by_image_id, expected):
    """Return paired AP, AP75 and APl sensitivity, with deltas in AP points.

    `gt` is the original COCO dataset mapping and rows are original mapped
    native-resolution COCO RLE predictions. `groups_by_image_id` maps every
    original integer image ID to a nonempty group label. `expected` maps
    "native" and "center" to exact original AP, AP75 and APl fractions.
    Omission rows use one-based ordinals in sorted group-label order.
    No-GT scores are undefined and counted separately.
    """
    require(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Expected exact original DEV705 CPU COCO runtime")
    require(type(expected) is dict and set(expected) == {"native", "center"}
            and all(type(arm) is dict and set(arm) == set(METRIC_INDEX)
                    and all(type(value) is float and 0. <= value <= 1. for value in arm.values())
                    for arm in expected.values()), "Expected original unrounded AP, AP75 and APl fractions")
    image_ids = [image["id"] for image in gt["images"]]
    require(len(image_ids) == len(set(image_ids)) and set(image_ids) == set(groups_by_image_id)
            and 2 <= len(set(groups_by_image_id.values())) <= MAX_GROUPS
            and [row["id"] for row in gt["categories"]] == [1],
            "Expected exact original image membership and a single ship category")

    adapter = load_official_adapter()
    arms = tuple(prepare_arm(adapter, gt, rows, groups_by_image_id, expected[name])
                 for name, rows in (("native", native_rows), ("center", center_rows)))
    native, center = arms
    for area in native:
        require(native[area].image_groups == center[area].image_groups
                and native[area].image_gt_counts == center[area].image_gt_counts,
                "Paired area caches must share original image groups and eligible GT")

    count = len(native["all"].groups)
    ones = np.ones(count, dtype=np.int32)
    fixed = paired_scores(arms, ones)
    draws = {name: [] for name in METRIC_INDEX}
    omissions = {name: [] for name in METRIC_INDEX}
    rng = np.random.Generator(np.random.PCG64(DRAW_SEED))
    for picks in rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count)):
        scores = paired_scores(arms, np.bincount(picks, minlength=count))
        for name, score in scores.items():
            draws[name].append(score["delta_points"])
    for ordinal in range(count):
        weights = ones.copy()
        weights[ordinal] = 0
        scores = paired_scores(arms, weights)
        for name, score in scores.items():
            counts = native[METRIC_AREA[name]].gt_by_group
            omissions[name].append(dict(omitted_group_ordinal=ordinal + 1,
                                        eligible_gt_remaining=int(counts @ weights), **score))

    metrics = {}
    for name in METRIC_INDEX:
        draw_values, bootstrap = sign_summary(draws[name])
        loo_values, leave_one = sign_summary([row["delta_points"] for row in omissions[name]])
        if draw_values:
            lower, median, upper = map(float, np.quantile(draw_values, (.025, .5, .975), method="linear"))
        else:
            lower = median = upper = None
        bootstrap.update(p2_5_points=lower, median_points=median, p97_5_points=upper)
        leave_one.update(min_points=min(loo_values, default=None), max_points=max(loo_values, default=None),
                         rows=omissions[name])
        metrics[name] = dict(fixed=fixed[name], bootstrap=bootstrap, leave_one_group_out=leave_one)

    return dict(images=len(image_ids), annotations=len(gt["annotations"]), groups=count,
                eligible_gt={name: int(cache.gt_by_group.sum()) for name, cache in native.items()},
                groups_with_eligible_gt={name: int(np.count_nonzero(cache.gt_by_group))
                                         for name, cache in native.items()},
                bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=DRAW_SEED, generator="numpy_PCG64",
                metrics=metrics)
