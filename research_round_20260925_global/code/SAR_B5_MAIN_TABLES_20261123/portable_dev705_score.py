"""Score the original center5202, three-head, eight-policy HRSID DEV705 experiment."""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("_clean_dev705_replay", HERE / "portable_dev705_replay.py")
replay = module_from_spec(spec)
spec.loader.exec_module(replay)
cap = replay.cap
require = cap.require
REFERENCE = "RFDETR_CENTER5202_DEV705_MATCHED_SCORE_PUBLIC_SAFE_20261120.json"
REFERENCE_SHA = "c0be05b24ca1bb23283ac19043b2ad96bbf2db39bc0ef525e1ea41a3dc9316e9"
ENTRY_SHA = "3a77e660fd31570cf1f4b44e818b28cc6743d414591216a5f274a11139955b05"
FORMAL = "RFDETR_MATCHED_POINT_FORMAL_20261120"
GEOMETRY_WITNESS = "SHAPE_MATCH_WITNESS_PRIVATE.json"
# Public binding for RF_WIT in DEV705_MATCHED_PAIRING_GEOMETRY_MODIFIERS_20261121/study.py.
GEOMETRY_WITNESS_SHA = "cdd238a5bea193635a063b7ff8ce22c522097bf30f76e033648006fc7b26eded"
TOLERANCE_POINTS = 1e-6


def operators(a):
    import shapely

    shared = cap.loader(a.code)
    original = shared.module(a.code / FORMAL / "run_score_dev.py", ENTRY_SHA)
    spot_dir = a.code / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120"
    fit = original.load_fit_step(spot_dir / "fit_step.py")
    spot = fit.load(spot_dir / "run_spot.py", original.SOURCE_SPOT, "_frozen_dev_spot")
    native = shared.module(a.code / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
                           original.NATIVE_SOURCE)
    native.DEV = a.intake
    native.OFFICIAL = a.code / "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"
    return SimpleNamespace(spot=spot, fit=fit, native=native,
                           readout=shared.module(a.code / FORMAL / "run_dev_readout.py", original.SOURCE_READOUT),
                           score=shared.module(a.code / FORMAL / "score_dev.py", original.MATHEMATICS),
                           geometry=shared.module(a.code / FORMAL / "evaluate_geometry.py", original.GEOMETRY),
                           helpers=fit.load_geometry(a.code, Path(shapely.__file__).resolve().parent.parent))


def admission(a, ops):
    """Require fresh label-free outputs, their original hashes and the observed probability correction."""
    cap_sha, repair_sha, _ = replay.capture_inputs(a)
    capture_raw = (a.capture / "RESULT_SAFE.json").read_bytes()
    capture = json.loads(capture_raw)
    require(cap.sha(capture_raw) == cap_sha and capture["independent_CPU_raster_disagree_rows"] == 1
            and capture["independent_CPU_raster_disagree_pixels"] == 1
            and capture["same_device_cuda_raster_exact"] == replay.EXPECTED["eligible"],
            "Observed native interpolation and repair counts differ")
    raw = (a.replay / "RESULT_SAFE.json").read_bytes()
    readout = json.loads(raw)
    require(readout["status"] == "B5_RFDETR_CENTER5202_DEV705_REPLAY_COMPLETE"
            and readout["images"] == 705 and readout["native_mapped_rows"] == 67386
            and readout["counts"] == replay.EXPECTED and readout["seeds"] == list(ops.score.SEEDS)
            and readout["arms"] == list(ops.score.ARMS) and readout["capture_result_sha256"] == cap_sha
            and readout["point_heads_sha256"] == {str(s): h for s, h in replay.HEADS.items()}
            and readout["checkpoint_sha256"] == replay.CHECKPOINT
            and readout["stock_predictions_sha256"] == replay.STOCK_SHA and readout["queries_sha256"] == cap.QUERIES_SHA
            and readout["edits_sha256"] == replay.EDITS_SHA and readout["probability_repair_sha256"] == repair_sha
            and readout["probability_repair_consumed"] == 1
            and all(readout[k] is True for k in ("exact_native_support", "exact_candidate_identity",
                    "actual_XOR_area_RLE_verified", "exact_per_candidate_budgets", "exact_historical_sparse_bytes", "DEV"))
            and all(readout[k] is False for k in ("FIT", "TEST", "AP", "GT_opened"))
            and all(readout[k] == 0 for k in ("optimizer_updates", "point_optimizer_updates", "model_forward_calls")),
            "Fresh label-free edit, head identity or full comparison roster differs")
    intake_raw = (a.intake / "RESULT_SAFE.json").read_bytes()
    intake = json.loads(intake_raw)
    require(intake["status"] == "B5_HRSID_DEV705_INPUTS_REBUILT" and intake["images"] == 705
            and intake["annotations"] == 1782 and intake["empty_rasters"] == 4 and intake["fit_images_redecoded"] == 2937
            and intake["dev_images_matching_fit_fingerprints"] == {"encoded_sha256": 0, "decoded_rgb_sha256": 0}
            and intake["image_manifest_sha256"] == cap.INPUT_MANIFEST
            and all(intake["artifacts_sha256"][name] == expected for name, expected in ops.native.FILES.items()
                    if name in ("DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json", "DEV_ORIGINAL_GT_PRIVATE.json",
                                "DEV_EMPTY_RASTERS_PRIVATE.json")), "Fresh author TRAIN-derived split or GT differs")
    return dict(intake_sha256=cap.sha(intake_raw), capture_sha256=cap_sha, replay_sha256=cap.sha(raw),
                edits_sha256=readout["edits_sha256"], queries_sha256=cap.QUERIES_SHA,
                input_manifest_sha256=cap.INPUT_MANIFEST, image_manifest_sha256=cap.OUTPUT_MANIFEST,
                probability_repair_sha256=repair_sha, probability_repair_consumed=1)


def read_rows(a, ops, np, torch, codec):
    manifest = ops.native.dev_file("DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json")
    pools, sparse = [], []
    digest = sha256()
    with (a.capture / cap.QUERIES).open("rb") as queries, (a.replay / "SPARSE_EDITS_PRIVATE.jsonl").open("rb") as edits:
        for ordinal, (item, query, line) in enumerate(zip(manifest, queries, edits, strict=True), 1):
            block, delta = json.loads(query), json.loads(line)
            digest.update(line)
            group = Path(item["file_name"]).name.split("_", 1)[0]
            require(item["image_id"] == block["image_id"] == delta["image_id"] and delta["ordinal"] == ordinal
                    and group == delta["group"] == block["source_prefix"] and delta["original_hw"] == [800,800]
                    and item["encoded_sha256"] == block["encoded_sha256"]
                    and item["decoded_rgb_sha256"] == block["image_content_sha256"], "Author image, prefix or edit order differs")
            rows = ops.spot.native_candidate_rows(block)
            boxes = ops.spot.author_boxes(block["raw_boxes_cxcywh"], torch)
            for row in rows:
                row["area"] = int(codec.area(row["rle"]))
                row["box_xyxy"] = boxes[row["query_id"]].tolist()
            pools.append(dict(native_block=ops.readout.stock_block(item["image_id"], rows), pool=rows, source_prefix=group))
            sparse.append(delta)
    require(digest.hexdigest() == replay.EDITS_SHA and len(manifest) == len({x["image_id"] for x in manifest}) == 705
            and sum(len(p["pool"]) for p in pools) == 67386, "Sparse digest, mapped population or roster differs")
    print(json.dumps(dict(event="VERIFY_MASKS_AND_QUOTAS", images=len(pools))), flush=True)
    prepared = ops.score.prepare_reconstruction(np, codec, pools, sparse)
    fresh_native = (a.capture / cap.STOCK).read_bytes()
    parity = ops.score.verify_native_baseline(ops.native, pools, fresh_native, replay.STOCK_SHA)
    return prepared, parity


def compute_coco(ops, np, codec, prepared, gt, empties, workers):
    # Reconstruct immutable inputs concurrently within the supplied bound. COCO and its
    # stdout redirection run on the calling thread in the original seed and method order.
    def reconstruct(seed, arm):
        return ops.score._reconstruct(np, codec, prepared, seed, arm)

    seeds, arms = ops.score.SEEDS, ops.score.ARMS
    baseline = ops.score.score_official(ops.native, gt, reconstruct(seeds[0], "N"), empties)
    results = {s: {"N": baseline} for s in seeds}
    print(json.dumps(dict(event="SCORED", arm="N")), flush=True)
    jobs = iter((s, arm) for s in seeds for arm in arms if arm != "N")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for seed, arm in jobs:
            pending.append((seed, arm, pool.submit(reconstruct, seed, arm)))
            if len(pending) == workers:
                break
        while pending:
            seed, arm, future = pending.popleft()
            predictions = future.result()
            del future
            next_job = next(jobs, None)
            if next_job is not None:
                pending.append((*next_job, pool.submit(reconstruct, *next_job)))
            results[seed][arm] = ops.score.score_official(ops.native, gt, predictions, empties)
            del predictions
            print(json.dumps(dict(event="SCORED", seed=seed, arm=arm)), flush=True)
    return results


def position_only(witness, coco, geometry, np):
    """Same-head, per-candidate matched budget supplement on the original frozen valid matches."""
    by_seed = {s:{} for s in geometry.SEEDS}
    for row in witness:
        key = (row["group"], row["image_id"], row["rank"], row["query_id"], row["matched_gt_id"])
        values = row["arms"]
        table = by_seed[row["seed"]]
        require(key not in table and values["SF"]["edits"] == values["SU=F"]["edits"], "Matched supplement support")
        table[key] = values["SU=F"]["iou"] - values["SF"]["iou"]
    keys = tuple(by_seed[geometry.SEEDS[0]])
    require(len(keys) == 1461 and all(tuple(t) == keys for t in by_seed.values()), "Frozen valid matches shared across seeds")
    grouped = defaultdict(list)
    for key in keys:
        grouped[key[0]].append(np.mean([by_seed[s][key] for s in geometry.SEEDS]))
    group_deltas = np.asarray([100*np.mean(grouped[key]) for key in sorted(grouped)])
    require(len(group_deltas) == 27, "Fixed-match prefix population")
    indices = np.random.Generator(np.random.PCG64(geometry.BOOTSTRAP_SEED)).integers(
        len(group_deltas), size=(geometry.DRAW_COUNT, len(group_deltas)))
    ap = [100*(coco[s]["SU=F"]["primary"]["AP"] - coco[s]["SF"]["primary"]["AP"]) for s in geometry.SEEDS]
    return dict(comparison="SU=F minus SF", AP_points=float(np.mean(ap)), AP_seeds=ap,
                IoU_points=float(group_deltas.mean()),
                source_interval95_points=np.quantile(group_deltas[indices].mean(axis=1), [.025,.975]).tolist())


def compare(expected, actual, path, report):
    if isinstance(expected, dict) and isinstance(actual, dict) and expected.keys() == actual.keys():
        for k, v in expected.items():
            compare(v, actual[k], (*path, k), report)
        return
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for i, v in enumerate(expected):
            compare(v, actual[i], (*path, str(i)), report)
        return
    report["leaves"] += 1
    if type(expected) is float and type(actual) in (int, float) and math.isfinite(actual):
        scale = 100 if "coco" in path or "mean_COCO" in path or path[-1].endswith("_fraction") else 1
        difference = abs(expected - actual)*scale
        report["maximum_absolute_difference_scaled"] = max(report["maximum_absolute_difference_scaled"], difference)
        if (expected == actual if expected == -1.0 else difference <= TOLERANCE_POINTS):
            return
    elif type(expected) is type(actual) and expected == actual:
        return
    name = ".".join(path)
    report["mismatch_count"] += 1
    if len(report["first_mismatches"]) < 20:
        report["first_mismatches"].append(name)


SCIENTIFIC_FIELDS = ("DEV", "FIT", "TEST", "arms", "seeds", "images", "model_forwards", "optimizer_updates",
                     "native_mapped_candidates", "primary_annotations", "sensitivity_annotations", "coco", "mean_COCO",
                     "contrasts_points", "geometry_summary", "stock_matching_counts", "budgets", "gate", "baseline_parity",
                     "original_CPU_disagree_pixels", "original_CPU_disagree_rows")


def qualify(result, ref):
    # Round trip integer seed keys to the published JSON representation.
    normalized = json.loads(json.dumps(result, allow_nan=False))
    report = dict(leaves=0, first_mismatches=[], mismatch_count=0, maximum_absolute_difference_scaled=0.0,
                  tolerance_points=TOLERANCE_POINTS, reference_sha256=REFERENCE_SHA)
    for name in SCIENTIFIC_FIELDS:
        compare(ref[name], normalized[name], (name,), report)
    compare(ref["independent_audit"]["position_only_supplement"], normalized["position_only_supplement"],
            ("position_only_supplement",), report)
    compare(GEOMETRY_WITNESS_SHA, normalized["private_shape_witness_sha256"], ("private_shape_witness_sha256",), report)
    report["passed"] = report["mismatch_count"] == 0
    return report


def run(a):
    import numpy as np
    import torch
    from pycocotools import mask as codec

    require(not torch.cuda.is_initialized(), "CPU scoring only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    ops = operators(a)
    print(json.dumps(dict(event="VERIFY_FRESH_INPUTS")), flush=True)
    lineage = admission(a, ops)
    a.output.mkdir()
    prepared, parity = read_rows(a, ops, np, torch, codec)
    seal = ops.spot.write(a.output / "PRELABEL_SAFE.json", dict(lineage, status="B5_DEV705_SCORE_PRELABEL",
        images=705, native_mapped_candidates=67386, AP=False, GT_opened=False))
    gt = ops.native.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    findings = ops.native.dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    empties = {r["annotation_id"] for r in findings if r["kind"] == "EMPTY_COCO_RASTER"}
    require([r["id"] for r in gt["images"]] == [r["native"]["image_id"] for r in prepared]
            and len(gt["annotations"]) == len({v["id"] for v in gt["annotations"]}) == 1782
            and [c["id"] for c in gt["categories"]] == [1] and len(empties) == len(findings) == 4,
            "Original DEV labels, order and empty-only sensitivity")
    print(json.dumps(dict(event="BEGIN_COCO", arms=list(ops.score.ARMS), workers=a.workers)), flush=True)
    coco = compute_coco(ops, np, codec, prepared, gt, empties, a.workers)
    print(json.dumps(dict(event="FIXED_MATCH_GEOMETRY")), flush=True)
    geometry = ops.score.fixed_match_geometry(np, gt, prepared, ops.fit, ops.helpers, ops.geometry)
    witness = geometry.pop("private_witness")
    witness_sha = ops.spot.write(a.output / GEOMETRY_WITNESS, witness)
    supplement = position_only(witness, coco, ops.geometry, np)
    mean_coco, contrasts = ops.score.summarize_scores(coco)
    gate = ops.geometry.locked_gate({s:{arm:v["primary"] for arm,v in r.items()} for s,r in coco.items()}, geometry["summary"])
    print(json.dumps(dict(event="COMPARE_PUBLISHED_TABLE")), flush=True)
    ref_path = a.code.parent / "results" / REFERENCE
    ref = json.loads(ops.spot.pinned(ref_path, REFERENCE_SHA, 2*ops.spot.MIB))
    require(parity.pop("predictions_sha256") == replay.STOCK_SHA, "Native export identity")
    parity["exact_historical_scores"] = all(coco[s]["N"] == ref["coco"][str(s)]["N"] for s in ops.score.SEEDS)
    result = dict(lineage, arms=list(ops.score.ARMS), seeds=list(ops.score.SEEDS), coco=coco, mean_COCO=mean_coco,
                  contrasts_points=contrasts, budgets=ops.score.edit_statistics(prepared), gate=gate, baseline_parity=parity,
                  geometry_summary=geometry["summary"], stock_matching_counts=geometry["stock_matching_counts"],
                  position_only_supplement=supplement, private_shape_witness_sha256=witness_sha,
                  images=len(prepared), native_mapped_candidates=parity["mapped_candidates"],
                  primary_annotations=len(gt["annotations"]), sensitivity_annotations=len(gt["annotations"])-len(empties),
                  original_CPU_disagree_pixels=1, original_CPU_disagree_rows=1, prelabel_sha256=seal,
                  checkpoint_sha256=replay.CHECKPOINT, point_heads_sha256=replay.HEADS, stock_predictions_sha256=replay.STOCK_SHA,
                  original_dev_gt_sha256=ops.native.FILES["DEV_ORIGINAL_GT_PRIVATE.json"],
                  original_empty_rasters_sha256=ops.native.FILES["DEV_EMPTY_RASTERS_PRIVATE.json"],
                  DEV=True, FIT=False, TEST=False, AP=True, model_forwards=0, optimizer_updates=0,
                  elapsed_seconds=perf_counter()-start, worker_sha256=cap.file_hash(Path(__file__)))
    qualification = qualify(result, ref)
    result.update(qualification=qualification, status=("B5_RFDETR_CENTER5202_DEV705_SCORE_COMPLETE" if qualification["passed"]
                                                       else "B5_RFDETR_CENTER5202_DEV705_SCORE_MISMATCH"))
    stamp = ops.spot.write(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=stamp, qualification=qualification)), flush=True)
    require(qualification["passed"] and not torch.cuda.is_initialized(), "Published full-precision table or CPU evidence differs")


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("intake", "capture", "replay", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--code", type=Path, default=HERE.parent)
    p.add_argument("--workers", type=int, choices=range(1, 9), default=1,
                   help="Maximum simultaneous original mask reconstructions; COCO evaluates in frozen order")
    a = p.parse_args()
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve(strict=name != "output"))
    require(sys.version_info[:3] == (3,10,13) and not a.output.exists(), "Fresh output and locked CPython required")
    expected = {"numpy":"1.26.4", "torch":"2.10.0+cu130", "pycocotools":"2.0.11", "shapely":"2.0.2"}
    require(all(version(k) == v for k,v in expected.items()), "Locked DEV scoring dependencies")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    run(a)


if __name__ == "__main__":
    main()

