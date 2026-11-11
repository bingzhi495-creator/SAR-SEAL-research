"""Measure oracle continuous hull-axis co-centering of original SSDD matches."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

FIXED_SOURCE = "86f2eecbac232c31d71d1a3f01f160b670dec7a4d89fcc263b3fe30a0d47782a"
FIXED_RESULT = "dc3cd6bd6b38fc465e9784cbf611b18aa78b935268f9e97dec136039c453be48"
FIXED_WAIT = "f4da22295118a35a49aa2222b5bd72ef9bb55403e7059482c6c0e888c54f5500"
MODES = ("raw", "short", "long", "both")
ARMS = ("native", "center")
METRICS = ("continuous_iou_points", "fp_per_gt_area_points", "fn_per_gt_area_points")
TOL = 1e-9


def require(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(path, expected):
    require(path == path.resolve(strict=True) and 0 < path.stat().st_size <= 2**20, "Canonical bounded input")
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Pinned closed input: " + path.name)
    return data


def admit(args):
    prior = args.score_dir.parent / "SSDD_FLOAT_COMMON_DEV187_FIXED_MATCH_GEOMETRY_ACTUAL_20261112_01"
    result = json.loads(pinned(prior / "RESULT_SAFE.json", FIXED_RESULT))
    wait = json.loads(pinned(prior / "ORIGINAL_WAIT_PRIVATE.json", FIXED_WAIT))
    tail = json.loads(pinned(prior / "ORIGINAL.stdout", wait["original_stdout_sha256"]).splitlines()[-1])
    require(wait["original_os_wait"] and wait["original_os_exit"] == 0 and wait["process_group_closed"]
            and wait["stop_reason"] is None and wait["score_source_sha256"] == FIXED_SOURCE
            and tail["result_sha256"] == FIXED_RESULT and result["DEV"] and not result["TEST"]
            and result["seed"] == 5201 and result["original_AP_and_AP90_exact"] and result["match_iou"] == .5,
            "Exact original closed fixed match diagnostic")
    path = args.code_root / "SSDD_FLOAT_COMMON_DEV187_FIXED_MATCH_GEOMETRY_20261112/float_fixed_geometry.py"
    mod = ModuleType("_closed_float_match_authority")
    mod.__file__ = str(path)
    exec(compile(pinned(path, FIXED_SOURCE), str(path), "exec"), mod.__dict__)
    return mod, result


def translations(rectangles, reference, tangent, normal):
    weights = (rectangles[:, 2:] - rectangles[:, :2]).prod(axis=1)
    center = np.average((rectangles[:, :2] + rectangles[:, 2:]) / 2, axis=0, weights=weights)
    offset = np.array([reference.centroid.x, reference.centroid.y]) - center
    short, long = float(normal @ offset) * normal, float(tangent @ offset) * tangent
    require(np.max(np.abs(short + long - offset)) < TOL, "Orthonormal hull components reconstruct centroid translation")
    require(abs(float(normal @ (offset-short))) < TOL and abs(float(tangent @ (offset-long))) < TOL,
            "Requested centroid projection must be zero after translation")
    return dict(raw=np.zeros(2), short=short, long=long, both=short+long)


def outside(rectangles, displacement, height, width):
    low = rectangles[:, :2].min(axis=0) + displacement
    high = rectangles[:, 2:].max(axis=0) + displacement
    return bool(np.any(low < -TOL) or np.any(high > np.array([width, height]) + TOL))


def readout(rectangles, area, reference, overlap):
    shifted_area = float((rectangles[:, 2:] - rectangles[:, :2]).prod(axis=1).sum())
    require(abs(area - shifted_area) < 1e-7, "Rigid translation must conserve original foreground area")
    inter = overlap(rectangles, reference)
    ref = float(reference.area)
    require(0 <= inter <= min(area, ref) + 1e-7 and area > 0 and ref > 0, "Valid nonempty continuous overlap")
    inter = min(inter, area, ref)
    return dict(continuous_iou_points=100 * inter / (area + ref - inter),
                fp_per_gt_area_points=100 * (area - inter) / ref,
                fn_per_gt_area_points=100 * (ref - inter) / ref)


def dense_witness(rle, reference, displacement, expected):
    from pycocotools import mask
    import shapely
    from shapely.affinity import translate

    y, x = np.nonzero(mask.decode(rle))
    geometry = shapely.union_all(shapely.box(x, y, x+1, y+1))
    moved = translate(geometry, xoff=float(displacement[0]), yoff=float(displacement[1]))
    joint = moved.intersection(reference).area
    ref, area = reference.area, moved.area
    actual = dict(continuous_iou_points=100 * joint / moved.union(reference).area,
                  fp_per_gt_area_points=100 * (area - joint) / ref,
                  fn_per_gt_area_points=100 * (ref - joint) / ref)
    error = max(abs(actual[key] - expected[key]) for key in METRICS)
    require(error < 1e-8, "Independent dense pixel-union witness failed")
    return float(error)


def cohort_summary(geometry, footprint, draws, full, retained, boundaries):
    require(full and retained, "Nonempty full and paired retained cohorts")
    def summary(rows, mode):
        pairs = [(chip, a[mode], b[mode]) for chip, a, b in rows]
        return {metric: geometry.summarize(pairs, metric, footprint, draws) for metric in METRICS}
    change = {}
    for mode in MODES[1:]:
        change[mode] = {}
        for metric in METRICS:
            contrasts = [(chip, (b[mode][metric]-a[mode][metric])-(b["raw"][metric]-a["raw"][metric]))
                         for chip, a, b in retained]
            boot = footprint.paired_cluster_stats(contrasts, draws)
            change[mode][metric] = dict(delta_change_from_raw=boot["row_weighted_mean_delta"],
                paired_chip_cluster_percentile95=boot["row_weighted_image_cluster_percentile95"])
    return dict(full_pairs=len(full), full_logical_chips=len({c for c, _, _ in full}), retained_pairs=len(retained),
                retained_logical_chips=len({c for c, _, _ in retained}), paired_excluded=len(full)-len(retained),
                boundaries=boundaries, raw_full=summary(full, "raw"),
                retained_modes={mode:summary(retained,mode) for mode in MODES}, retained_change_from_raw=change)


def run(args):
    start = perf_counter()
    pinned(Path(__file__).resolve().parent / "PROTOCOL.md", args.protocol_sha)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask
    require(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11"
            and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact original CPU versions")
    authority, prior = admit(args)
    worker = authority.module(args.code_root / "SSDD_FLOAT_PRIMARY_DEV187_20261112/float_dev187.py",
                              authority.WORKER_SHA, "_original_float_worker")
    geometry = authority.module(args.code_root / "MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                                authority.GEOMETRY_SHA, "_original_fixed_geometry")
    admission = authority.module(args.code_root / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                                 authority.ADMISSION_SHA, "_original_continuous_admission")
    stage, mods = worker.load_stock(args.code_root)
    stage.admit_runtime("SCORE")
    gt, streams, accepted, official = authority.load_score(args, worker, stage, mods["ssdd_phase_dev_admission"],
                                                           mods["ssdd_phase_dev_score"])
    wrap = geometry.load_module(args.code_root / geometry.MATCH_WRAPPER, geometry.MATCH_WRAPPER_SHA, "_original_match_bridge")
    _, _, template, _, _, matching = wrap.load_helpers(args.code_root, args.code_root / geometry.MATCH_GEOMETRY)
    helpers = SimpleNamespace(template=template, adapter=official, match=matching.match_official)
    matches = geometry.original_matches(gt, streams, accepted["primary"], helpers, admission)
    modules = admission.geometry_modules(args.code_root)
    polygons, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    require((len(polygons), len(neutral), len(gt["images"])) == (380, 0, 187), "Exact original float support")
    images = {image["id"]: (ordinal, image) for ordinal, image in enumerate(gt["images"])}
    full = {group: [] for group in ("small_elongated", "thin")}
    retained = {group: [] for group in full}
    excluded = {group: {arm: Counter({mode:0 for mode in MODES[1:]}) for arm in ARMS} for group in full}
    centroids = {arm: [] for arm in ARMS}
    dense_checks = 0
    dense_error = 0.
    for ann in sorted(gt["annotations"], key=lambda row: row["id"]):
        ann_id = ann["id"]
        if any(ann_id not in matches[arm] for arm in ARMS):
            continue
        reference, _ = polygons[ann_id]
        aspect, width, tangent, normal = geometry.hull(reference)
        is_small = aspect >= 2 and modules.geometry.stratum(ann["area"]) == "small"
        is_thin = aspect >= 2 and width <= 8
        groups = [group for group, active in (("small_elongated", is_small), ("thin", is_thin)) if active]
        if not groups:
            continue
        chip, image = images[ann["image_id"]]
        height, width = image["height"], image["width"]
        data, values = {}, {}
        any_outside = False
        for arm in ARMS:
            rle = matches[arm][ann_id]
            require(list(rle["size"]) == [height, width], "Preserve native raster canvas")
            counts = rle["counts"]
            text = counts.decode("ascii") if isinstance(counts, bytes) else counts
            rectangles = modules.geometry.boxes_from_counts(modules.shift.runs(text), height, width)
            area = float(mask.area(rle))
            shifts = translations(rectangles, reference, tangent, normal)
            data[arm] = (rle, rectangles, area, shifts)
            raw = readout(rectangles, area, reference, modules.geometry.overlap_area)
            values[arm] = {"raw": raw}
            require(not outside(rectangles, shifts["raw"], height, width), "Original footprint must stay in image")
            for mode in MODES[1:]:
                if outside(rectangles, shifts[mode], height, width):
                    any_outside = True
                    for group in groups:
                        excluded[group][arm][mode] += 1
            if is_small:
                centroids[arm].append(float(np.linalg.norm(shifts["short"])))
        raw_pair = (chip, values["native"], values["center"])
        for group in groups:
            full[group].append(raw_pair)
        if any_outside:
            continue
        witness_pair = is_small and dense_checks < 2
        for arm in ARMS:
            rle, rectangles, area, shifts = data[arm]
            for mode in MODES[1:]:
                moved = rectangles + np.tile(shifts[mode], 2)
                values[arm][mode] = readout(moved, area, reference, modules.geometry.overlap_area)
            if witness_pair:
                for mode in MODES:
                    dense_error = max(dense_error, dense_witness(rle, reference, shifts[mode], values[arm][mode]))
        if witness_pair:
            dense_checks += 1
        for group in groups:
            retained[group].append(raw_pair)
    require((len(full["small_elongated"]), len(full["thin"]), dense_checks) == (238, 30, 2), "Exact original matched cohorts")
    for arm in ARMS:
        key = f"{arm}_mean"
        require(abs(float(np.mean(centroids[arm])) - prior["paired_geometry"]["small"]["centroid_abs_short_px"][key]) < 1e-10,
                "Reproduce original small elongated centroid support")
    draws = modules.footprint.bootstrap_draws(len(images))
    summaries = {group:cohort_summary(geometry, modules.footprint, draws, full[group], retained[group], excluded[group])
                 for group in full}
    report = dict(status="SSDD_FLOAT_COMMON_DEV187_HULL_COCENTER_ACTUAL", DEV=True, FIT=False, TEST=False,
                  training=False, model_forward_calls=0, optimizer_updates=0, GT_informed=True, images=187,
                  seed=5201, original_all12_and_AP90_exact=True, match_iou=.5, bootstrap_draws=len(draws),
                  fixed_geometry_result_sha256=FIXED_RESULT, fixed_geometry_worker_sha256=FIXED_SOURCE,
                  protocol_sha256=args.protocol_sha, dense_pairs=dense_checks, dense_checks=16,
                  dense_max_abs_metric_error=dense_error, groups=summaries, worker_seconds=perf_counter()-start)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        require(out.write(raw) == len(raw), "Fresh complete co-centering report")
    print(json.dumps(dict(status=report["status"],result_sha256=sha256(raw).hexdigest())), flush=True)


def main():
    parser = ArgumentParser(allow_abbrev=False)
    for name in ("code-root", "score-dir", "score-wait", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and os.environ.get("OMP_NUM_THREADS")==os.environ.get("OPENBLAS_NUM_THREADS")=="1", "Isolated CPU geometry")
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Closed DEV geometry remains offline")
    sys.addaudithook(offline)
    run(args)


if __name__ == "__main__":
    main()
