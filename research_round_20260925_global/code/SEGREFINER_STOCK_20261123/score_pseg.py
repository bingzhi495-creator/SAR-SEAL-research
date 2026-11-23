"""Score a closed stock SegRefiner-LR inference on the two original PSeg views."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


EXPOSURE_WORKER = "0ef0bb5f6e0780680f417d3e8ed85824eb83e49a4893fdf51ff0e91905fdd1b3"
EXPOSURE_SAFE = "9d6aea9df9aaf0c8f79a8ec48c27f9dfcd77b89256645e85cdcfbaceff652f7b"
REQUEST_PAYLOAD = "c02de790d0634afdc950ecc4252ccac792daeaec969a60f0f8d933691a0f4c29"
REQUEST_RESULT = "55d1a8f95d9008fc00b609503942f022e0e393407a478d50ce7cea3d4dc3c7f8"
REQUEST_TERMINAL = "bf5b06188a2e18c4c420de53616e3c2944f8adab51273f21a9da99dc9d7ffa4d"
AUTHOR_COMMIT = "53419a2d38ea3da0b6e2be77e5b45e139195a0b3"
STATUS = "PSEG232_SEGREFINER_STOCK_SCORE_COMPLETE"
INFERENCE_STATUS = "PSEG232_SEGREFINER_STOCK_INFERENCE_COMPLETE"
STOCK = "segrefiner_stock"
RAW = "raw"
LOCAL = "free_natural"
STATIC = "attraction_static_at_dynamic_count"
ARMS = (RAW, LOCAL, STATIC, STOCK)
GATE = 512


def exposure_module():
    path = Path(__file__).with_name("exposure_pseg.py")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != EXPOSURE_WORKER:
        raise ValueError("The original PSeg exposure code is required")
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def verify_receipt(source, args, capture, protocol):
    inference, result_sha, terminal_sha = source.observed(args.inference, args.inference_terminal, "inference",
                                                         INFERENCE_STATUS, args.inference_worker_sha256)
    source.require(inference["phase"] == "inference" and inference["protocol_sha256"] == protocol
                   and inference["runtime_sha256"] == args.runtime_sha256
                   and inference["checkpoint_sha256"] == args.checkpoint_sha256
                   and inference["author_commit"] == AUTHOR_COMMIT and inference["exposure_safe_sha256"] == EXPOSURE_SAFE
                   and inference["request_result_sha256"] == REQUEST_RESULT
                   and inference["request_terminal_sha256"] == REQUEST_TERMINAL and inference["request_sha256"] == REQUEST_PAYLOAD
                   and inference["manifest_sha256"] == capture["manifest_sha256"]
                   and inference["native_predictions_sha256"] == capture["input_predictions_sha256"]
                   and inference["native_audit_sha256"] == capture["candidate_audit_sha256"]
                   and inference["stock_min_original_mask_area_pixels"] == GATE and inference["runtime"]["seed"] == 5217
                   and inference["target_labels_opened"] is False and inference["bypass_verified_native_exact"] is True
                   and inference["detector_forwards"] == inference["optimizer_updates"] == 0,
                   "The observed inference must bind the exact stock release and historical inputs")
    return inference, result_sha, terminal_sha


def qualify_edits(source, records, roster, blocks, rows):
    totals = Counter()
    per_image = {}
    for ordinal, (item, payload) in enumerate(zip(roster, rows, strict=True), 1):
        name = item["image_ref"]["path"]
        rec = records[name]
        native, edits, size = blocks[name]
        source.require(set(payload) == {"image_id", "ordinal", "changes"} and type(payload["ordinal"]) is int
                       and payload["ordinal"] == ordinal and payload["image_id"] == item["image_id"],
                       "Inference image identity and original row order required")
        changes = payload["changes"]
        expected = {str(rank) for rank, area in enumerate(rec["areas"], 1) if area >= GATE}
        source.require(isinstance(changes, dict) and set(changes) == expected, "Exactly the expected gate-positive native slots may write")
        changed = empty = 0
        for rank, change in changes.items():
            source.require(set(change) == {"rle"} and STOCK not in edits.get(rank, {}), "One stock mask per accepted slot required")
            new = source.decoded(change["rle"], size)
            empty += int(not new.any())
            changed += int(change["rle"] != native[int(rank)-1]["segmentation"])
            edits.setdefault(rank, {})[STOCK] = change
        per_image[name] = dict(entered_gate=len(changes), changed=changed, empty=empty)
        rec["edits"] = edits
        totals.update(images=1, native_candidates=len(native), area_admitted=len(changes),
                      exact_native_bypass=len(native)-len(changes), changed_area_admitted=changed,
                      unchanged_area_admitted=len(changes)-changed)
    return dict(totals), per_image


def mask_at(record, rank, arm):
    raw = record["native"][rank-1]["segmentation"]
    return record["edits"].get(str(rank), {}).get(arm, {}).get("rle", raw)


def adjusted_records(records, ids, arm):
    return {iid: dict(native=[dict(det, segmentation=mask_at(records[iid], rank, arm))
                              for rank, det in enumerate(records[iid]["native"], 1)]) for iid in ids}


def calculate_coverage(source, geom, gt, ids, records, original, metrics, arm):
    modified = adjusted_records(records, ids, arm)
    anchors = {ann["id"]: ann for ann in gt["annotations"]}
    retained = 0
    for aid, (iid, rank) in original.items():
        prediction = modified[iid]["native"][rank-1]["segmentation"]
        raster = source.codec.merge(source.codec.frPyObjects(anchors[aid]["segmentation"], *prediction["size"]))
        retained += int(source.codec.iou([prediction], [raster], [False])[0, 0] >= .50)
    rematched, diagnostics = geom.evaluate_raw(gt, modified, ids, metrics[arm])
    return dict(original_pairs=len(original), original_pairs_retained_50=retained,
                rematched_hulls_50=len(rematched), ground_truth_hulls=diagnostics["gt_instances"])


def measure_candidate(source, base, modules, truth, masks):
    cache = {}
    for rle in masks.values():
        key = rle["counts"]
        if key not in cache:
            area = int(source.codec.area(rle))
            value = base.continuous(rle, area, truth, modules) if area else (0., 0., 100.)
            cache[key] = (*value, 100.-value[2])
    return np.asarray([cache[masks[arm]["counts"]] for arm in ARMS])


def geometry_views(source, geom, base, modules, truth, records, anchors, cache):
    full, exposed = [], []
    altered = {arm: Counter() for arm in ARMS[:-1]}
    for aid, (iid, rank) in anchors.items():
        record = records[iid]
        area = record["areas"][rank-1]
        if not 32 <= area < 1024:
            continue
        group = record["rgb"]
        masks = {arm: mask_at(record, rank, arm) for arm in ARMS}
        if aid not in cache:
            cache[aid] = measure_candidate(source, base, modules, truth[aid][0], masks)
        row = (group, cache[aid])
        full.append(row)
        if area >= GATE:
            exposed.append(row)
            if masks[STOCK] != masks[RAW]:
                for arm in altered:
                    if masks[STOCK] != masks[arm]:
                        altered[arm][group] += 1
    geom.ARMS = ARMS
    geom.PAIRS = tuple((STOCK, arm) for arm in ARMS[:-1])
    geom.FIELDS = ("continuous_iou_points", "fp_pixels_per_gt_area_pct", "fn_pixels_per_gt_area_pct", "intersection_per_gt_area_pct")
    census = {arm: dict(altered_differing_eligible_hulls=sum(groups.values()),
                        altered_differing_decoded_groups=len(groups)) for arm, groups in altered.items()}
    return geom.summarize(full), geom.summarize(exposed), census


def decision(ap, iou, exposure):
    lower, upper = iou["percentile95"]
    exposed = exposure["altered_differing_eligible_hulls"] >= 20 and exposure["altered_differing_decoded_groups"] >= 8
    return dict(absolute_AP_threshold=0.30, exposure_supported=exposed,
                supported_advantage=bool(exposed and ap >= .30 and lower > 0),
                supported_loss=bool(exposed and ap <= -.30 and upper < 0))


def evaluate(args):
    start = perf_counter()
    exposure = exposure_module()
    source = exposure.load_source(args.factorial_source)
    protocol = source.digest(args.protocol.read_bytes())
    source.require(args.protocol.name == exposure.PROTOCOL_NAME and protocol == args.protocol_sha256 == exposure.PROTOCOL_SHA256
                   and not args.output.exists() and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
                   "Exact published protocol, historical geometry engine and fresh output required")
    established = json.loads(source.pin(args.exposure_safe, EXPOSURE_SAFE))
    source.require(established["worker_sha256"] == EXPOSURE_WORKER and established["status"] == exposure.STATUS
                   and established["protocol_sha256"] == protocol, "Exact historical area-exposure result required")
    _, scorer, geom, prior, capture, roster, audits, original, blocks = source.historical(args)
    records_by_name = exposure.native_records(source, blocks, roster, audits, original)
    infer, result_sha, terminal_sha = verify_receipt(source, args, capture, protocol)
    data = source.pin(args.inference / "EDITS_PRIVATE.jsonl", infer["edits_sha256"])
    totals, per_image = qualify_edits(source, records_by_name, roster, blocks, map(json.loads, data.splitlines()))
    source.require(totals == infer["counts"] and tuple(totals[key] for key in ("images", "native_candidates", "area_admitted"))
                   == (232, 23200, 5067), "The actual stock writes do not reproduce the observed inference and exposure counts")
    print(json.dumps(dict(phase="verified_inference_before_labels", **totals)), flush=True)

    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    source.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
                   and {im["file_name"] for im in gt["images"]} == set(records_by_name)
                   and source.digest(("\n".join(scorer.EXCLUDED) + "\n").encode("ascii")) == scorer.EXCLUSION_SHA,
                   "Original author reference and sensitivity roster required")
    records = {im["id"]: records_by_name[im["file_name"]] for im in gt["images"]}
    source.require(len(records) == 232 and all(records[im["id"]]["size"] == [im["height"], im["width"]] for im in gt["images"]),
                   "Author reference identity and native mask canvas differ")
    base = geom.load(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    source.require(len(truth) == 546 and not neutral, "Unchanged floating author polygon reference required")
    views, cache, full_anchors = {}, {}, None
    for view, expected in exposure.VIEWS.items():
        names = set(records_by_name) if view == "official" else set(records_by_name) - set(scorer.EXCLUDED)
        ids = sorted(iid for iid, row in records.items() if row["name"] in names)
        ids_set = set(ids)
        subset = dict(gt, images=[im for im in gt["images"] if im["id"] in ids_set],
                      annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids_set])
        original_order = [im["id"] for im in subset["images"]]
        metrics = {}
        for arm in ARMS:
            candidates = [dict(image_id=iid, category_id=0, score=det["score"], segmentation=mask_at(records[iid], rank, arm))
                          for iid in original_order for rank, det in enumerate(records[iid]["native"], 1)]
            metrics[arm] = scorer.evaluate(subset, candidates)
        source.require(all(metrics[arm] == prior["views"][view]["metrics_fraction"][arm] for arm in ARMS[:-1]),
                       "Original native and historical independent comparator COCO values differ")
        anchors, diagnostics = geom.evaluate_raw(gt, records, ids, metrics[RAW])
        if full_anchors is None:
            full_anchors = anchors
        else:
            source.require(anchors == {aid: value for aid, value in full_anchors.items() if value[0] in ids_set},
                           "Sensitivity set must retain the original raw .50 anchors")
        fixed, gate_exposed, effects = geometry_views(source, geom, base, modules, truth, records, anchors, cache)
        observed = (len(ids), len(subset["annotations"]), len(anchors), fixed["eligible_pairs"], fixed["decoded_image_groups_with_pairs"])
        exposed = established["views"][view]["native_fixed_matches_iou50"]
        source.require(observed == expected and diagnostics["gt_instances"] == expected[1]
                       and gate_exposed["eligible_pairs"] == exposed["stock_admitted_within_research_area"]
                       and gate_exposed["decoded_image_groups_with_pairs"] == exposed["stock_admitted_within_research_area_decoded_groups"],
                       "Historical anchor or stock gate-exposed geometry populations differ")
        for arm in (RAW, STATIC):
            source.require(all(abs(fixed["arms"][arm][key]-v) < 1e-9 for key, v in prior["geometry"][view]["arms"][arm].items()),
                           "Historical fixed polygon geometry differs")
        coverage = {RAW: dict(original_pairs=len(anchors), original_pairs_retained_50=len(anchors),
                              rematched_hulls_50=len(anchors), ground_truth_hulls=diagnostics["gt_instances"])}
        coverage.update({arm: calculate_coverage(source, geom, gt, ids, records, anchors, metrics, arm) for arm in ARMS[1:]})
        contrasts = {}
        for arm, effect in effects.items():
            pair = STOCK + " minus " + arm
            coco = source.difference(metrics[STOCK], metrics[arm])
            geom_all = fixed["pairwise"][pair]
            contrasts[STOCK + "_minus_" + arm] = dict(COCO_points=coco, altered_exposure=effect,
                                                    criteria=decision(coco["AP"], geom_all["continuous_iou_points"], effect),
                                                    full_fixed_geometry=geom_all, gate_exposed_geometry=gate_exposed["pairwise"][pair])
        counts = {key: sum(per_image[name][key] for name in names) for key in ("entered_gate", "changed", "empty")}
        source.require(counts["entered_gate"] == established["views"][view]["original_candidates"]["stock_area_admitted"],
                       "Stock gate count by original cohort differs")
        views[view] = dict(images=len(ids), annotations=len(subset["annotations"]), candidates=len(ids)*100, stock_counts=counts,
                           metrics_fraction=metrics, coverage=coverage, geometry=fixed, gate_exposed_geometry=gate_exposed, contrasts=contrasts)
        print(json.dumps(dict(phase="scored_view", view=view, images=len(ids))), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=source.digest(Path(__file__).read_bytes()), protocol_sha256=protocol,
                inference_result_sha256=result_sha, inference_terminal_sha256=terminal_sha, inference_worker_sha256=args.inference_worker_sha256,
                runtime_sha256=args.runtime_sha256, checkpoint_sha256=args.checkpoint_sha256, edits_sha256=infer["edits_sha256"],
                exposure_score_sha256=EXPOSURE_SAFE, historical_score_sha256=source.PHASES["score"][0],
                historical_score_terminal_sha256=source.PHASES["score"][1], author_labels_sha256=scorer.GT_SHA,
                counts=totals, stock_min_original_mask_area_pixels=GATE, historical_COCO_and_geometry_parity=True,
                inference_verified_before_labels=True, detector_forwards=0, segrefiner_forwards=0, optimizer_updates=0,
                bootstrap_draws=4096, bootstrap_seed=5217, primary_view="official", sensitivity_view="exclude_exact_fit_overlap",
                primary_comparator=RAW, frozen_local_comparator=LOCAL, views=views, score_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("factorial-source", "code", "owner", "capture", "classical-root", "exposure-safe", "inference", "inference-terminal",
                 "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("inference-worker-sha256", "runtime-sha256", "checkpoint-sha256", "protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--original-readback", type=Path)
    args = parser.parse_args()
    result = evaluate(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete safe score receipt")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
