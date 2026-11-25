"""Evaluate native-grid Boundary IoU on the exact archived Gaofen-3 comparison."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


LOCK = "92c645e3233de9e69ce645d60d20377376a19639079c2231c373205e1f9642ee"
BOUNDARY_STATS = "7e0e8e29018f6a9c7e19b03f17976a5de78ef09bfc35edfc88e1f9e914cb141f"
INPUTS = "8d5d6b28279bd492de9585f570ca0a8f23919cdfadc0e4214602681ebc94e554"
OLD_SCORER = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
AUTHOR_TEST = "ed5be06ac39b4cd5c373c00fdba132bc8f0ad1260023e2001c5e2d24a33f0760"
BOUNDARY_METRIC = "b477c918a1168eb6b734313b18c4df8e338a591305c164a444eb61a560e873c5"
SEEDS = {5202: (5217, 5218), 5203: (5217, 5218, 5219)}
NATIVE_MATCHES = {5202: 629, 5203: 628}
ELIGIBLE_GROUPS = {5202: 68, 5203: 67}
HEIGHT = WIDTH = 512
STATUS = "RSDD_GF3_NATIVE_BOUNDARY_IOU_COMPLETE"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    require(re.fullmatch("[0-9a-f]{64}", expected), "Expected an exact SHA256 pin")
    raw = path.read_bytes()
    require(digest(raw) == expected, "Original input fingerprint differs: " + path.name)
    return raw


def module(path, expected):
    value = ModuleType(path.stem)
    value.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), vars(value))
    return value


def arm(seed, variant):
    return f"{seed}/{variant}"


def matched_masks(codec, native, edits, seeds):
    """Read the two signed writers or their exact zero-quota native fallbacks."""
    native_area = int(codec.area(native))
    masks, quotas = {}, {}
    for seed in seeds:
        names = [arm(seed, "fixed"), arm(seed, "free_matched_count")]
        before, after = (edits.get(name) for name in names)
        require((before is None) == (after is None), "Orphan signed writer at the frozen count")
        if before is None:
            masks.update({name: native for name in names})
            quotas[seed] = 0
            continue
        quota = before["swaps"]
        require(type(quota) is int and 1 <= quota <= 8 and type(after["swaps"]) is int
                and after["swaps"] == quota, "Different signed budgets on the same candidate")
        for name, variant in zip(names, (before, after), strict=True):
            rle = variant["rle"]
            require(rle["size"] == native["size"] == [HEIGHT, WIDTH], "Frozen native raster dimensions differ")
            intersection = int(codec.area(codec.merge([native, rle], intersect=True)))
            require(native_area - intersection == int(codec.area(rle)) - intersection == quota,
                    "Written signed changes differ from the admitted equal budget")
            masks[name] = rle
        quotas[seed] = quota
    return masks, quotas


def scoped_blocks(blocks, detector):
    keep = {}
    arms = {arm(s, v) for s in SEEDS[detector] for v in ("fixed", "free_matched_count")}
    for iid, row in blocks.items():
        if row["sensor"] != "GF-3":
            continue
        changes = {}
        for rank, edits in row["changes"].items():
            retained = {policy: edit for policy, edit in edits.items() if policy in arms}
            if retained:
                changes[rank] = retained
        keep[iid] = dict(row, changes=changes)
    require(len(keep) == 583 and len({row["group"] for row in keep.values()}) == 76
            and len({row["rgb"] for row in keep.values()}) == 583,
            "Expect the complete historical Gaofen-3 population without decoded duplicates")
    return keep


def history(code, inputs, root, detector):
    blocks, raster, lineage = inputs.load_inputs(code, root, detector, 740)
    keep = scoped_blocks(blocks, detector)
    require(tuple(map(int, lineage["seeds"])) == SEEDS[detector],
            "The original head fits or their order differ")
    if detector == 5202:
        views = {}
        for seed in SEEDS[detector]:
            expected = lineage["phase_result_sha256"][f"score_{seed}"]
            report = json.loads(pin(root / f"score{seed}/RESULT_SAFE.json", expected))
            require(report["gt_sha256"] == AUTHOR_TEST and report["detector_forwards"] == report["optimizer_updates"] == 0,
                    "Historical scorer reference or frozen scope differs")
            views[seed] = report["views"]["gf3"]["geometry"]
    else:
        report = json.loads(pin(root / "score/RESULT_SAFE.json", lineage["score_result_sha256"]))
        require(report["annotation_sha256"] == AUTHOR_TEST and report["detector_forwards"] == report["optimizer_updates"] == 0,
                "Historical second-carrier scoring reference differs")
        views = report["views"]["gf3"]["geometry"]
    del blocks
    return keep, raster["gf3"], views, lineage


def author_labels(path, original, roster):
    # Rekey the complete, pinned historical file before subsetting, preserving original COCO match IDs.
    data = {"info": {}, **json.loads(pin(path, AUTHOR_TEST))}
    require(len(data["images"]) == 740 and len(data["annotations"]) == 872,
            "Author test release population differs")
    canonical, rekeyed = original.coco_annotation_ids(data)
    images = [row for row in canonical["images"] if row["id"] in roster]
    require(len(images) == len(roster) == 583 and {row["id"] for row in images} == set(roster)
            and all(row["width"] == WIDTH and row["height"] == HEIGHT
                    and row["file_name"] == roster[row["id"]]["file"] for row in images),
            "Bound historical author file-to-Gaofen-3 image join differs")
    annotations = [row for row in canonical["annotations"] if row["image_id"] in roster]
    require(len(annotations) == 631 and all(row["category_id"] == 1 and len(row["segmentation"]) == 1
                                         for row in annotations),
            "Original 631 Gaofen-3 single-ring author hulls required")
    groups = Counter(roster[row["image_id"]]["group"] for row in annotations)
    return dict(canonical, images=images, annotations=annotations), groups, rekeyed


def agree_native_metrics(actual, expected):
    require(set(actual) == set(expected), "Published native COCO metric names differ")
    for name, value in actual.items():
        prior = expected[name]
        if prior is None:
            require(not np.isfinite(value) or value < 0, "Prior unavailable native COCO metric differs")
        else:
            require(abs(value - prior) <= 1e-12, "Exact original native COCO metric parity failed: " + name)


def geometry_expected(hist, detector, seed):
    """Obtain original continuous groups and fixed/free arm statistics."""
    if detector == 5202:
        geo = hist[seed]
        arms = geo["stats"]["group_equal_arms"]
        paired = geo["stats"]["paired_contrasts_points"]["free_matched_count minus fixed"]["continuous_iou_points"]
        return (arms["raw"]["continuous_iou_points"], arms["fixed"]["continuous_iou_points"],
                arms["free_matched_count"]["continuous_iou_points"], paired)
    arms = hist["group_equal"]
    paired = hist["effects"]["independent_minus_fixed"][str(seed)]["continuous_iou_points"]
    return (arms["raw"]["continuous_iou_points"], arms[arm(seed, "fixed")]["continuous_iou_points"],
            arms[arm(seed, "free_matched_count")]["continuous_iou_points"], paired)


def geometric_parity(rows, history_geometry, detector, arms, seeds):
    keys = sorted(rows)
    require(len(keys) == ELIGIBLE_GROUPS[detector], "Prespecified eligible native acquisition count differs")
    matrix = np.stack([np.mean(rows[group], axis=0) for group in keys])
    means = dict(zip(arms, matrix.mean(axis=0).tolist(), strict=True))
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(keys), size=(4096, len(keys)))
    effects, group_delta = {}, []
    for seed in seeds:
        fixed, free = arm(seed, "fixed"), arm(seed, "free_matched_count")
        delta = matrix[:, arms.index(free)] - matrix[:, arms.index(fixed)]
        interval = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]).tolist()
        earlier_raw, earlier_fixed, earlier_free, earlier_pair = geometry_expected(history_geometry, detector, seed)
        require(np.allclose([means["native"], means[fixed], means[free], delta.mean(), *interval],
                            [earlier_raw, earlier_fixed, earlier_free, earlier_pair["delta"], *earlier_pair["interval_95"]],
                            rtol=0, atol=1e-9), "Original fixed-match continuous geometry or interval differs")
        effects[str(seed)] = {"delta_points": float(delta.mean()), "interval_95": interval}
        group_delta.append(delta)
    delta = np.mean(group_delta, axis=0)
    interval = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]).tolist()
    if detector == 5203:
        expected = history_geometry["effects"]["independent_minus_fixed"]["head_mean"]["continuous_iou_points"]
        require(np.allclose([delta.mean(), *interval], [expected["delta"], *expected["interval_95"]],
                            rtol=0, atol=1e-9), "Original pooled second detector continuous effect differs")
    return dict(group_equal_arm_points=means, per_head=effects,
                head_mean=dict(delta_points=float(delta.mean()), interval_95=interval), original_exact=True)


def primary_decision(boundary):
    row = boundary["widths"]["1"]
    subset = row["native_reference_non_saturated"]
    main = row["contrasts"]["independent_vs_fixed"]["paired_95_interval_points"][0]
    detail = subset["contrasts"].get("independent_vs_fixed")
    checks = {
        "one_pixel_lower_positive": main > 0,
        "non_saturated_coverage": subset["anchors"] >= 20 and subset["acquisition_groups"] >= 6,
        "non_saturated_lower_positive": bool(detail and detail["paired_95_interval_points"][0] > 0),
    }
    return {**checks, "boundary_specific_support": all(checks.values())}


def measure(detector, blocks, truth, true_masks, author_groups, baseline_raster, baseline_geometry, evaluator, core, stats, metric):
    seeds = SEEDS[detector]
    pair_arms = tuple(arm(s, v) for s in seeds for v in ("fixed", "free_matched_count"))
    terms = tuple((arm(s, "free_matched_count"), arm(s, "fixed")) for s in seeds)
    contrasts = {"independent_vs_fixed": terms}
    contrasts.update({f"independent_vs_fixed_{seed}": (pair,) for seed, pair in zip(seeds, terms, strict=True)})
    audit = stats.Audit(metric, pair_arms, contrasts)
    raw, anchors = evaluator.score_one(truth, blocks, "raw", anchors=True)
    agree_native_metrics(raw, baseline_raster["raw"])
    require(len(anchors) == NATIVE_MATCHES[detector], "Original nonignored COCO raster IoU .50 native matches differ")
    polygons = {v["id"]: v for v in truth["annotations"]}
    rows, positive, changed = defaultdict(list), {s: Counter() for s in seeds}, {s: Counter() for s in seeds}
    eligible, outside = 0, 0
    for gid, (iid, rank, record) in anchors.items():
        label = polygons[gid]
        require(label["image_id"] == iid and record["image_id"] == iid, "The native matched candidate owner differs")
        frame = blocks[iid]
        native = record["segmentation"]
        if not 32 <= int(core.codec.area(native)) < 1024:
            outside += 1
            continue
        shape = shapely.Polygon(np.asarray(label["segmentation"][0], dtype=float).reshape(-1, 2))
        require(shape.is_valid and shape.area > 0, "Unexpected invalid Gaofen-3 continuous hull")
        labels_rle = true_masks[gid]
        require(int(core.codec.area(labels_rle)) > 0, "Native matched author mask is empty")
        masks, quotas = matched_masks(core.codec, native, frame["changes"].get(rank, {}), seeds)
        parent = frame["group"]
        for seed in seeds:
            if quotas[seed] > 0:
                positive[seed][parent] += 1
            if masks[arm(seed, "fixed")] != masks[arm(seed, "free_matched_count")]:
                changed[seed][parent] += 1
        audit.add_rles(f"{parent:03d}", labels_rle, native, masks)
        geometry = {}
        entries = {"native": native, **masks}
        for rle in entries.values():
            key = rle["counts"]
            if key not in geometry:
                geometry[key] = 100 * core.geometric_iou(rle, shape, shapely)
        rows[parent].append([geometry[entries[name]["counts"]] for name in ("native", *pair_arms)])
        eligible += 1
    require(eligible == 425 and eligible + outside == len(anchors), "Exact archived native-area denominator differs")
    if detector == 5202:
        for s in seeds:
            old = baseline_geometry[s]
            require(sum(positive[s].values()) == old["counts"]["fixed_positive_anchors"]
                    and len(positive[s]) == old["positive_exposure_groups"],
                    "Original positive signed-budget matched exposure differs")
    else:
        for s in seeds:
            old = baseline_geometry["primary_exposure"][str(s)]
            require(sum(changed[s].values()) == old["changed_matches"] and len(changed[s]) == old["changed_groups"],
                    "Original differing writeback matched exposure differs")
    parity = geometric_parity(rows, baseline_geometry, detector, ("native", *pair_arms), seeds)
    boundary = audit.report({f"{parent:03d}": count for parent, count in author_groups.items()})
    return {
        "detector_seed": detector, "head_seeds": list(seeds), "images": len(blocks),
        "native_mapped_candidates": sum(len(frame["stock"]) for frame in blocks.values()),
        "author_hulls": len(truth["annotations"]), "author_parent_groups": len(author_groups),
        "native_matches": len(anchors), "native_misses": len(truth["annotations"]) - len(anchors),
        "area_excluded_matches": outside, "eligible_native_matches": eligible,
        "eligible_parent_groups": len(rows), "native_AP_points": 100 * raw["AP"],
        "historical_independent_vs_fixed_AP_points": {
            str(s): 100 * (baseline_raster[arm(s, "free_matched_count")]["AP"] - baseline_raster[arm(s, "fixed")]["AP"])
            for s in seeds
        },
        "positive_budget_exposure": {str(s): dict(hulls=sum(positive[s].values()), groups=len(positive[s])) for s in seeds},
        "different_mask_exposure": {str(s): dict(hulls=sum(changed[s].values()), groups=len(changed[s])) for s in seeds},
        "continuous_fixed_geometry_parity": parity,
        "boundary": boundary, "primary": primary_decision(boundary),
    }


def run(args):
    start = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and sys.version_info[:3] == (3, 10, 13) and np.__version__ == "1.26.4"
            and version("pycocotools") == "2.0.11" and shapely.__version__ == "2.0.2"
            and shapely.geos_version_string == "3.11.2" and not args.output.exists(),
            "Fresh isolated CPU phase in the qualified Gaofen-3 runtime")
    require(args.protocol_sha256 == LOCK and args.stats_sha256 == BOUNDARY_STATS,
            "Exact published lock and boundary summary implementation required")
    pin(args.protocol, LOCK)
    require(args.core_sha256 == BOUNDARY_METRIC, "Exact agreed Boundary IoU implementation required")
    metric = module(args.core, BOUNDARY_METRIC)
    stats = module(args.stats, args.stats_sha256)
    inputs = module(args.code / "RSDD_STRICT_CONTINUOUS_20261123/inputs.py", INPUTS)
    original = module(args.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", OLD_SCORER)
    require(original.AUTHOR_TEST == AUTHOR_TEST, "Shared exact historical author annotation")
    evaluator = original.module(args.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", original.SCORE_SOURCE)
    adaptor = original.module(args.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", original.GEOMETRY_SOURCE)
    core = adaptor.load_geometry(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    archives = {detector: history(args.code, inputs, root, detector)
                for detector, root in ((5202, args.run_5202), (5203, args.run_5203))}
    identities = lambda frames: {i: (r["group"], r["file"], r["rgb"]) for i, r in frames.items()}
    require(identities(archives[5202][0]) == identities(archives[5203][0]),
            "Gaofen-3 image IDs, parent grouping, filenames and decoded pixels differ across carriers")
    print("Both frozen native masks and writer lineages verified on the identical Gaofen-3 cohort.", flush=True)

    truth, groups, rekeyed = author_labels(args.author_labels, original, archives[5202][0])
    true_masks = {row["id"]: core.codec.merge(core.codec.frPyObjects(row["segmentation"], HEIGHT, WIDTH))
                  for row in truth["annotations"]}
    empty_author = sum(int(core.codec.area(rle)) == 0 for rle in true_masks.values())
    results = {}
    for detector, (frames, raster, geometry, lineage) in archives.items():
        results[str(detector)] = measure(detector, frames, truth, true_masks, groups, raster, geometry, evaluator, core, stats, metric)
        results[str(detector)]["input_lineage"] = lineage
        print(f"Completed frozen Gaofen-3 boundary scoring for carrier {detector}.", flush=True)
    result = dict(
        status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()),
        protocol_sha256=args.protocol_sha256, boundary_metric_sha256=BOUNDARY_METRIC,
        boundary_stats_sha256=args.stats_sha256, original_geometry_loader_sha256=INPUTS,
        author_reference_sha256=AUTHOR_TEST, evaluator_ids_rekeyed=rekeyed,
        empty_author_rasterizations=empty_author,
        previously_evaluated_historical_cohort=True, only_gf3_view_scored=True,
        matches_reused_within_each_detector=True, no_target_model_selection=True,
        scored_held_or_prospective_labels=False, optimizer_updates=0, detector_forwards=0,
        results=results, elapsed_seconds=perf_counter() - start,
    )
    args.output.mkdir()
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "run-5202", "run-5203", "author-labels", "protocol", "core", "stats", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha256", "core-sha256", "stats-sha256"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
