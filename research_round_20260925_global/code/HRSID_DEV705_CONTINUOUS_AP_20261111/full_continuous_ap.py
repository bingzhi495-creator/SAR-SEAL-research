"""Closed paired DEV705 score sensitivity using vector ground-truth overlaps."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np


POLICY = {
    "population": "705 original DEV images, identical full raw prediction streams",
    "positive_support": "valid finite noncrowd source polygon union clipped to canvas; exact original rerasterization",
    "neutral": "invalid, empty, unsupported, and crowd rows retained as ignored with original raster overlap in both arms",
    "area": "original annotation and prediction pixel area fields, original stock bins",
    "matching": "unchanged COCOeval sorting, matching, ignores, accumulation, IoU thresholds and maxDets",
    "pair_pruning": "noncrowd area and bbox overlap bound provably below .50 with conservative relative tolerance",
    "scope": "same annotation source; no official TEST, image opening, model inference or tuning",
}
STATUS = "HRSID_COMMON_DEV705_FULL_CONTINUOUS_AP_SENSITIVITY_ACTUAL_V1"
ARMS = ("native", "center")
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "APm", "APl", "AR100")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def boxes_from_counts(counts, height, width):
    """Return disjoint exact foreground rectangles in pixel-edge coordinates."""
    boxes = []
    start = 0
    for index, count in enumerate(counts):
        stop = start + count
        if index % 2 and count:
            x0, y0 = divmod(start, height)
            x1, last = divmod(stop - 1, height)
            if x0 == x1:
                boxes.append((x0, y0, x0 + 1, last + 1))
            else:
                boxes.append((x0, y0, x0 + 1, height))
                if x0 + 1 < x1:
                    boxes.append((x0 + 1, 0, x1, height))
                boxes.append((x1, 0, x1 + 1, last + 1))
        start = stop
    require(start == height * width, "Original prediction RLE must span the entire canvas")
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def overlap_area(bounds, geometry):
    import shapely
    x0, y0, x1, y1 = geometry.bounds
    kept = bounds[(bounds[:, 0] < x1) & (bounds[:, 2] > x0) &
                  (bounds[:, 1] < y1) & (bounds[:, 3] > y0)]
    if not len(kept):
        return 0.
    rectangles = shapely.box(*kept.T)
    return float(shapely.area(shapely.intersection(rectangles, geometry)).sum())


def bound_area(dt_bounds, dt_area, gt_bounds, gt_area):
    extent = np.maximum(np.minimum(dt_bounds[:, 2:], gt_bounds[2:]) -
                        np.maximum(dt_bounds[:, :2], gt_bounds[:2]), 0)
    return np.minimum(np.minimum(dt_area, gt_area), extent.prod(axis=1))


def plausible(dt_bounds, dt_area, gt_bounds, gt_area):
    upper = bound_area(dt_bounds, dt_area, gt_bounds, gt_area)
    combined = dt_area + gt_area
    return (dt_area > 0) & (3 * upper >= combined - 1e-12 * np.maximum(1., combined))


def load_source(path, expected, name):
    raw = path.read_bytes()
    require(path.resolve(strict=True) == path and sha256(raw).hexdigest() == expected, "Pinned ordinary analysis source differs")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def stratum(area):
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def polygon_support(gt, vector):
    import shapely
    from pycocotools import mask
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
    polygons, neutral = {}, set()
    excluded, supported = Counter(), Counter()
    for ann in gt["annotations"]:
        aid = ann["id"]
        area_bin = stratum(ann["area"])
        rle = coco.annToRLE(ann)
        reason = None
        if ann.get("iscrowd") or ann.get("ignore"):
            reason = "original_ignore_or_crowd"
        elif not int(mask.area(rle)):
            reason = "empty_original_raster"
        else:
            try:
                rings = vector.polygon_rings(ann["segmentation"])
                shape = shapely.union_all([shapely.Polygon(ring) for ring in rings])
                h, w = rle["size"]
                shape = shape.intersection(shapely.box(0, 0, w, h))
                if not shape.is_valid or not shape.area > 0:
                    reason = "invalid_or_empty_clipped_union"
                elif not np.array_equal(vector.polygon_raster(rings, (h, w)), mask.decode(rle)):
                    reason = "original_rerasterization_differs"
            except vector.UnsupportedPolygon:
                reason = "invalid_or_unsupported_source_components"
        if reason:
            neutral.add(aid)
            excluded[(area_bin, reason)] += 1
        else:
            polygons[aid] = (shape, rings)
            supported[area_bin] += 1
    require(len(polygons) + len(neutral) == len(gt["annotations"]), "Unique support and ignore partition required")
    return polygons, neutral, dict(supported_by_size=dict(supported), excluded=[
        dict(size=size, reason=reason, unique_instances=count) for (size, reason), count in sorted(excluded.items())
    ])


def score(gt, rows, adapter, neutral=(), polygons=None, runs=None, footprint=None):
    from pycocotools import mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    counters = Counter()
    max_abs_error = 0.

    class PairedEvaluator(COCOeval):
        def _prepare(self):
            super()._prepare()
            for annotations in self._gts.values():
                for ann in annotations:
                    if ann["id"] in neutral:
                        ann["ignore"] = 1

        def computeIoU(self, image, category):
            nonlocal max_abs_error
            raster = super().computeIoU(image, category)
            if polygons is None or not len(raster):
                return raster
            gts = self._gts[image, category]
            dts = self._dts[image, category]
            order = np.argsort([-dt["score"] for dt in dts], kind="mergesort")[: self.params.maxDets[-1]]
            dts = [dts[i] for i in order]
            require(raster.shape == (len(dts), len(gts)), "Native score and overlap matrix layouts differ")
            if not dts or not gts:
                return raster
            rles = [dt["segmentation"] for dt in dts]
            dt_area = np.asarray(mask.area(rles), dtype=np.float64)
            dt_bounds = np.asarray(mask.toBbox(rles), dtype=np.float64)
            dt_bounds[:, 2:] += dt_bounds[:, :2]
            cache = {}
            for j, ann in enumerate(gts):
                if ann["id"] in neutral:
                    counters["neutral_original_overlap_pairs"] += len(dts)
                    continue
                shape, rings = polygons[ann["id"]]
                area = shape.area
                gt_bounds = np.asarray(shape.bounds)
                possible = plausible(dt_bounds, dt_area, gt_bounds, area)
                counters["supported_pairs"] += len(dts)
                counters["provably_below_half_pairs"] += int((~possible).sum())
                # Neutral/crowd columns remain original. Here .50 is the lowest stock threshold.
                raster[:, j] = 0.
                for i in np.flatnonzero(possible):
                    if i not in cache:
                        rle = rles[i]
                        compressed = rle["counts"]
                        if isinstance(compressed, bytes):
                            compressed = compressed.decode("ascii")
                        cache[i] = boxes_from_counts(runs(compressed), *rle["size"])
                    intersection = overlap_area(cache[i], shape)
                    require(-1e-10 <= intersection <= min(dt_area[i], area) + 1e-7, "Continuous intersection outside area bounds")
                    actual = intersection / (dt_area[i] + area - intersection)
                    raster[i, j] = actual
                    counters["continuous_pairs_computed"] += 1
                    if counters["dense_geometry_spots"] < 16:
                        dense = mask.decode(rles[i]).astype(bool)
                        expected = footprint(rings, dense.shape, {"prediction": dense})["by_arm"]["prediction"]["iou"]
                        error = abs(expected - actual)
                        max_abs_error = max(max_abs_error, error)
                        require(error < 1e-10, "RLE rectangle overlap differs from independent real dense footprint")
                        counters["dense_geometry_spots"] += 1
                if counters["pruned_dense_spots"] < 8:
                    upper = bound_area(dt_bounds, dt_area, gt_bounds, area)
                    for i in np.flatnonzero((~possible) & (upper > 0))[:8 - counters["pruned_dense_spots"]]:
                        dense = mask.decode(rles[i]).astype(bool)
                        exact = footprint(rings, dense.shape, {"prediction": dense})["by_arm"]["prediction"]["iou"]
                        require(exact < .5, "Conservative pruning dropped an eligible real overlap")
                        counters["pruned_dense_spots"] += 1
            return raster

    coco = COCO()
    coco.dataset = deepcopy(gt)
    ids = [x["id"] for x in gt["images"]]
    captured = []

    def observe(*args):
        evaluator = PairedEvaluator(*args)
        captured.append(evaluator)
        return evaluator

    with redirect_stdout(StringIO()):
        coco.createIndex()
        all12 = adapter.evaluate_population(dict(COCO=COCO, COCOeval=observe), coco, rows, ids)
    evaluator, = captured
    params = evaluator.params
    require(params.maxDets == [1, 10, 100] and np.isclose(params.iouThrs[0], .5), "Exact stock COCO settings required")
    area = list(params.areaRngLbl).index("all")
    index90, = np.flatnonzero(np.isclose(params.iouThrs, .90))
    values = evaluator.eval["precision"][index90, :, 0, area, -1]
    valid = values[values > -1]
    require(len(all12) == 12 and np.isfinite(all12).all() and valid.size, "Complete finite stock COCO scores required")
    metrics = dict(zip(("AP", "AP50", "AP75", "APs", "APm", "APl"), all12[:6], strict=True))
    metrics.update(AP90=float(valid.mean()), AR100=all12[8], all12=list(all12))
    begin = area * len(ids)
    layout = tuple(None if e is None else (e["image_id"], tuple(e["dtIds"]), tuple(e["dtScores"]))
                   for e in evaluator.evalImgs[begin:begin + len(ids)])
    return metrics, layout, dict(counters, independent_dense_max_abs_iou_error=max_abs_error)


def run(a):
    started = perf_counter()
    sys.path.append(str(a.shapely_site))
    import shapely
    require(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11"
            and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
            "Exact previously used CPU geometry and scoring packages required")
    root = a.code_root
    shift_path = root/"HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py"
    shift = load_source(shift_path, a.shift_source_sha256, "_p374_closed_gate")
    p311 = shift.load_p311(root)
    require((a.pair_sha256, a.training_protocol, a.accepted_result_sha256)
            == (p311.PAIR_SHA, p311.TRAINING_PROTOCOL, p311.PRIMARY_SHA), "Exact closed common pair required")
    closed = p311.load_closed(root)
    driver = closed.load_gate(root).load_driver()
    accepted = driver.read_json(a.accepted_result, a.accepted_result_sha256)
    require(accepted["status"] == p311.PRIMARY_STATUS and accepted["training_seed"] == 5202
            and accepted["official_TEST_opened"] is False, "Exact independent original DEV acceptance required")
    gt, native, center, provenance = closed.verified_inputs(root, a.pair, a.pair_sha256,
        native_arm_receipt=a.native_arm_receipt, center_arm_receipt=a.center_arm_receipt,
        training_protocol=a.training_protocol)
    require(len(gt["images"]) == 705 and len(gt["annotations"]) == 1782 and
            provenance["training_pair_receipt_sha256"] == accepted["independent_training_pair_sha256"] and
            provenance["original_dev_gt_sha256"] == accepted["original_dev_gt_sha256"],
            "Exact original DEV population required")
    adapter = closed.load_diagnostics(root)["dev705_paired_mask_failure_scope"].load_official_adapter()
    vector_root = root/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    vector = load_source(vector_root/"vector_label_geometry.py",
        "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09", "_p374_polygon_rings")
    math_api = load_source(vector_root/"vector_population_math.py",
        "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e", "_p374_independent_footprint")
    streams = dict(native=native, center=center)
    raw_results, ranks, results, audits = {}, {}, {}, {}
    for arm, rows in streams.items():
        raw_results[arm], ranks[arm], _ = score(gt, rows, adapter)
        prior = accepted["scores"][arm]["primary"]
        require(np.allclose(raw_results[arm]["all12"], prior["all12"], rtol=0., atol=1e-12)
                and abs(raw_results[arm]["AP90"] - prior["AP90"]) <= 1e-12,
                "Full original raster AP must reproduce before shared-support scoring")
    print(json.dumps(dict(event="P374_EXACT_ORIGINAL_FULL_RASTER_REPRODUCED", images=705, annotations=1782)), flush=True)
    polygons, neutral, census = polygon_support(gt, vector)
    print(json.dumps(dict(event="P374_FROZEN_COMMON_VALID_SUPPORT", supported=len(polygons), neutral=len(neutral))), flush=True)
    for arm, rows in streams.items():
        raster, raster_ranks, _ = score(gt, rows, adapter, neutral)
        continuous, continuous_ranks, audit = score(gt, rows, adapter, neutral, polygons, shift.runs,
                                                    math_api.run_footprint_metrics)
        require(ranks[arm] == raster_ranks == continuous_ranks, "Original scores, ties and evaluated candidates must stay exact")
        require(audit["dense_geometry_spots"] == 16 and audit["pruned_dense_spots"] == 8,
                "Expected independent real geometry and pruning checks")
        results[arm] = dict(original_full_raster=raw_results[arm], shared_support_raster=raster,
                            shared_support_continuous=continuous)
        audits[arm] = audit
        print(json.dumps(dict(event="P374_PAIRED_OVERLAP_ARM_COMPLETE", arm=arm,
                              exact_pairs=audit["continuous_pairs_computed"])), flush=True)
    deltas = {mode: {metric: 100 * (results["center"][mode][metric] - results["native"][mode][metric])
                     for metric in METRICS} for mode in results["native"]}
    deltas["continuous_minus_raster_common_effect_points"] = {
        metric: deltas["shared_support_continuous"][metric] - deltas["shared_support_raster"][metric]
        for metric in METRICS
    }
    report = dict(status=STATUS, policy=POLICY, FIT=False, DEV=True, official_TEST_opened=False, optimizer_updates=0,
                  model_forward_calls=0, training_seed=5202, images=705, original_unique_annotations=1782,
                  supported_unique_annotations=len(polygons), neutral_unique_annotations=len(neutral), support=census,
                  raw_prediction_rows={arm: len(rows) for arm, rows in streams.items()}, original_order_unchanged=True,
                  verified_original_raster_scores=True, results=results, delta_points=deltas, geometry_audits=audits,
                  provenance=provenance, elapsed_seconds=perf_counter()-started)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with a.output.open("xb") as dst:
        dst.write(raw)
        dst.flush()
        os.fsync(dst.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "pair", "native-arm-receipt", "center-arm-receipt", "accepted-result", "shapely-site", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256", "shift-source-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(os.name == "nt" and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == 1
            and sys.flags.utf8_mode == 1 and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Exact isolated CPU execution required")
    run(args)


if __name__ == "__main__":
    main()
