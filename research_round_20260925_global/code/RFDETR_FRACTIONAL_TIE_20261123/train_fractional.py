"""Refit the original RF-DETR center5202 full-FIT109 endpoint heads with four paired targets.

Input is restricted to the original source FIT capture, FIT annotations, training
code and historical training receipts. All seeds first reproduce their archived
fractional checkpoints before any alternative-supervision fits are admitted.
"""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np


STATUS = "RFDETR_FRACTIONAL_TIE_FIT109_COMPLETE"
ARMS = ("F", "T", "H", "C")
SEEDS = (5217, 5218, 5219)
CAPTURE_ACCEPTED = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
FIT_ACCEPTED = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
FIT_CLOSED = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
FIT_FEATURES = "1cea85eb8c30b11bfa847292c55d2231fb4d16dd2a551c3da52ddffdd8d3830e"
SPOT = "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/"
FORMAL = "RFDETR_MATCHED_POINT_FORMAL_20261120/"
CODE = {
    "spot": (SPOT + "run_spot.py", "75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee"),
    "fit": (SPOT + "fit_step.py", "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1"),
    "feature": (SPOT + "feature_adapter.py", "747c4c4b43561f8d073e01854adf241f1cb66c4efa1d29f304ae8837ecd67b73"),
    "loader": (FORMAL + "run_fit109.py", "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e"),
    "plan": (FORMAL + "fit_plan.py", "1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Original training component mismatch: " + path.name)
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def historical(args, spot):
    root = args.historical_fit

    def read(name, digest):
        return json.loads(spot.pinned(root / name, digest, 8 * spot.MIB))

    accepted = read("INDEPENDENT_SAFE.json", FIT_ACCEPTED)
    old = read("RESULT_SAFE.json", FIT_CLOSED)
    terminal = read("ORIGINAL_TOOL_WAIT_PRIVATE.json", accepted["original_tool_wait_sha256"])
    pre = read("PRELABEL_SAFE.json", old["prelabel_sha256"])
    require(accepted["status"] == "RFDETR_MATCHED_POINTS_FIT109_INDEPENDENT_ACCEPTED"
            and accepted["fit_result_sha256"] == FIT_CLOSED and accepted["exact_schedule"]
            and accepted["run_natural_exit"] == terminal["terminal"]["exit_code"] == 0
            and accepted["checkpoint_bundles_verified"] == 18 and accepted["head_updates_verified"] == 7200
            and old["status"] == "RFDETR_MATCHED_POINTS_FIT109_RUN_COMPLETE" and old["images"] == 109
            and not any(old[key] for key in ("DEV", "TEST", "AP")) and old["FIT"]
            and pre["accepted_capture_sha256"] == CAPTURE_ACCEPTED and pre["feature_sha256"] == FIT_FEATURES,
            "Historical source run and independent closure required")
    all_fit = [entry for entry in old["all_initializations"] if entry["endpoint"] == "all_fit"]
    require(tuple(entry["seed"] for entry in all_fit) == SEEDS and all(entry["fold"] is None for entry in all_fit),
            "Three original full-FIT109 seed records required")
    return old, {entry["seed"]: entry for entry in all_fit}


def hard_targets(q):
    """Make exact tie decisions before the original trainer's float32 BCE conversion."""
    require(q.dtype == np.float64 and np.isfinite(q).all()
            and (q >= 0).all() and (q <= 1 + 1e-12).all(), "Expected original FP64 unit-cell intersections")
    threshold = (q >= 0.5).astype(np.float32)
    tie_only = np.where(q == 0.5, np.float32(0.5), threshold).astype(np.float32)
    return tie_only, threshold


def source_targets(records, source, codec):
    from pycocotools.coco import COCO

    selected_images = {row["image_id"] for row in records}
    annotations = [a for a in source["annotations"] if a["image_id"] in selected_images]
    coco = COCO()
    coco.dataset = dict(images=[im for im in source["images"] if im["id"] in selected_images],
                        annotations=annotations, categories=source["categories"])
    with redirect_stdout(StringIO()):
        coco.createIndex()
    populations = {arm: [] for arm in ARMS}
    counts = Counter()
    for row in records:
        if row["truth"] is None or not row["features"].size:
            continue
        q = row["truth"]
        image_id = row["image_id"]
        ann = coco.anns[row["matched_gt_id"]]
        coords = row["coordinates"]
        require(row["role"] == "valid" and ann["image_id"] == image_id and coords.shape == q.shape
                and coords.shape == row["features"].shape[:2] and coords.shape[1] == 2
                and len(coords) <= 32 and np.issubdtype(coords.dtype, np.integer),
                "Pair endpoints must retain native image, matched annotation, feature and row identities")
        rle = coco.annToRLE(ann)
        require(coco.imgs[image_id]["width"] == coco.imgs[image_id]["height"] == 800
                and list(rle["size"]) == [800, 800] and np.all((coords >= 0) & (coords < 800**2)),
                "Native COCO annotation dimensions and in-image endpoints")
        raster = codec.decode(rle).ravel(order="C")[coords].astype(np.float32)
        tie_only, threshold = hard_targets(q)
        for arm, target in zip(ARMS, (q, tie_only, threshold, raster), strict=True):
            populations[arm].append(row if arm == "F" else dict(row, truth=target))
        partial = (q > 0) & (q < 1)
        ties = q == 0.5
        fp32 = q.astype(np.float32)
        counts.update(matches=1, pairs=len(q), endpoints=q.size, exact_zero=int((q == 0).sum()),
                      exact_one=int((q == 1).sum()), greater_one_roundoff=int((q > 1).sum()),
                      partial=int(partial.sum()), exact_half=int(ties.sum()),
                      partial_non_half=int((partial & ~ties).sum()),
                      count_half_only_after_fp32=int(((fp32 == 0.5) & ~ties).sum()),
                      threshold_different_after_fp32=int(((fp32 >= .5) != (q >= .5)).sum()),
                      fractional_tie_only_differences_after_fp32=int((fp32 != tie_only).sum()),
                      tie_only_threshold_differences=int((tie_only != threshold).sum()),
                      threshold_raster_disagreements=int((threshold != raster).sum()),
                      threshold_raster_disagreements_at_tie=int(((threshold != raster) & ties).sum()),
                      raster_positive_below_half=int(((raster == 1) & (q < .5)).sum()),
                      raster_negative_at_least_half=int(((raster == 0) & (q >= .5)).sum()))
    require(counts["matches"] == 392 and counts["pairs"] == 12537 and counts["endpoints"] == 25074
            and len({r["group"] for r in populations["F"]}) == 92
            and counts["exact_zero"] + counts["exact_one"] + counts["greater_one_roundoff"] + counts["partial"] == counts["endpoints"]
            and counts["tie_only_threshold_differences"] == counts["exact_half"], "Source target population mismatch")
    return populations, counts


def refit(args, arm, entry, records, trainer, plan, fit):
    seed, sample_seed = entry["seed"], 20261112 + entry["seed"] - SEEDS[0]
    archived_name = f"{seed}_all_fit.npz"
    require(entry["relative_weights"] == archived_name, "Unexpected historical archive name")
    original = args.historical_fit / "POINT_HEADS_PRIVATE" / archived_name
    require(sha256(original.read_bytes()).hexdigest() == entry["weights_sha256"], "Historical weights changed")
    relpath = f"HEADS_PRIVATE/{seed}_{arm}.npz"
    started = perf_counter()
    fitted = fit.fit_heads(records, trainer, steps=plan.STEPS, batch=plan.BATCH, seed=seed, sample_seed=sample_seed)
    audit = fitted["audit"]
    for field in ("shared_pairs", "source_groups", "steps", "batch", "sampled_pair_rows", "sampled_point_rows",
                  "common_batch_sha256"):
        require(audit[field] == entry[field], "Group-balanced schedule differs: " + field)
    for name, stats in audit["heads"].items():
        baseline = entry["heads"][name]
        require(stats["parameters"] == baseline["parameters"] and stats["optimizer_steps"] == baseline["optimizer_steps"] == [200]
                and stats["initial_tensor_sha256"] == baseline["initial_tensor_sha256"] and stats["finite_parameters"],
                "Historical seed, initialization or optimizer step changed")
        if arm == "F":
            require(stats["final_tensor_sha256"] == baseline["final_tensor_sha256"], "Fractional tensor digest differs")
    arrays = plan.frozen_arrays(fitted)
    with np.load(original, allow_pickle=False) as sealed:
        require(set(sealed.files) == set(arrays), "Historical model or scaler schema differs")
        for key in (arrays.keys() if arm == "F" else ("center", "scale")):
            actual, old = arrays[key], sealed[key]
            require(actual.dtype == old.dtype and actual.shape == old.shape and actual.tobytes() == old.tobytes(),
                    "Bit-exact original tensor or standardization required: " + key)
    path = args.output / relpath
    weight_sha = plan.seal_bundle(path, fitted)
    if arm == "F":
        require(weight_sha == entry["weights_sha256"], "Reconstructed entire fractional NPZ differs")
    restored, center, scale = plan.restore_bundle(path, weight_sha, trainer, fit)
    for name, model in restored.items():
        readback = fit.predict(model, records[0]["features"], center, scale)
        require(np.array_equal(readback, fitted["prediction"][name][0]), "Restored endpoint predictions changed")
    heads = {name: {key: stats[key] for key in ("parameters", "optimizer_steps", "initial_tensor_sha256", "final_tensor_sha256")}
             for name, stats in audit["heads"].items()}
    return dict(relative_path=relpath, sha256=weight_sha), dict(
        seed=seed, sampler_seed=sample_seed, heads=heads, batch_sha256=audit["common_batch_sha256"],
        sampled_pair_rows=audit["sampled_pair_rows"], sampled_point_rows=audit["sampled_point_rows"],
        archived_fractional_sha256=entry["weights_sha256"], fit_and_predict_seconds=audit["cpu_fit_and_predict_seconds"],
        fit_write_verify_seconds=perf_counter()-started)


def run(args):
    import torch
    from pycocotools import mask as codec

    started = perf_counter()
    require(not args.output.exists() and args.output.parent.is_dir() and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Use a fresh output under an existing parent and disable GPU visibility")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    modules = {k: load(args.code / path, pin) for k, (path, pin) in CODE.items()}
    spot, fit, plan = modules["spot"], modules["fit"], modules["plan"]
    require(tuple(plan.SEEDS) == SEEDS and (plan.STEPS, plan.BATCH) == (200, 64) and not torch.cuda.is_initialized(),
            "Original deterministic CPU training required")
    spot.pinned(args.protocol, args.protocol_sha256, spot.MIB)
    old, history = historical(args, spot)
    args.acceptance_sha256 = CAPTURE_ACCEPTED
    roster, accepted = modules["loader"].read_population(args, spot)
    posterior = fit.load(args.code / "RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py", spot.ADAPTER_SHA, "native_posterior")
    trainer = fit.load_trainer(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    data, prepared, feature_digest = spot.prepare(args, roster, fit, modules["feature"], posterior, np, torch, codec)
    require(feature_digest == FIT_FEATURES and prepared == old["preparation"]
            and accepted["same_device_raster_exact"] == prepared["eligible"] == prepared["exact_original_masks"]
            and accepted["native_mapped_rows"] == prepared["mapped"], "Historical point population or feature bytes differ")
    args.output.mkdir()
    (args.output / "HEADS_PRIVATE").mkdir()
    source_lineage = dict(protocol_sha256=args.protocol_sha256, historical_fit_sha256=FIT_CLOSED,
                          historical_fit_acceptance_sha256=FIT_ACCEPTED, source_capture_acceptance_sha256=CAPTURE_ACCEPTED,
                          feature_sha256=feature_digest, source_cohort_sha256=modules["loader"].SOURCE_COHORT_SHA,
                          source_checkpoint_sha256=modules["loader"].CHECKPOINT_SHA,
                          source_manifest_sha256=accepted["image_manifest_sha256"], source_queries_sha256=accepted["queries_sha256"],
                          source_fit_gt_sha256=spot.GT_SHA)
    prelabel_sha = spot.write(args.output / "PRELABEL_SAFE.json", dict(status="RFDETR_FRACTIONAL_TIE_FEATURES_ACCEPTED",
                              FIT=True, GT_opened=False, target_inputs_opened=False, preparation=prepared, **source_lineage))
    records, supervision = spot.attach(args, data, fit)
    del data
    require(supervision == old["supervision"], "Original stock match identities and counts required")
    source = json.loads(spot.pinned(args.fit_gt, spot.GT_SHA, 32 * spot.MIB))
    populations, counts = source_targets(records, source, codec)
    del records, source
    bundles, audit, timings, updates = {}, {}, {}, {}
    for arm in ARMS:
        arm_started = perf_counter()
        for seed in SEEDS:
            key = f"{seed}/{arm}"
            bundles[key], audit[key] = refit(args, arm, history[seed], populations[arm], trainer, plan, fit)
            print(json.dumps(dict(event="FULL_FIT_HEAD_VERIFIED", seed=seed, arm=arm, sha256=bundles[key]["sha256"])), flush=True)
        timings[arm] = perf_counter() - arm_started
        updates[arm] = sum(sum(audit[f"{s}/{arm}"]["heads"][h]["optimizer_steps"]) for s in SEEDS for h in ("S", "L"))
    require(set(updates.values()) == {1200} and len(bundles) == 12 and not torch.cuda.is_initialized(),
            "All twelve matched full-source bundles and 4,800 head updates required")
    return dict(status=STATUS, FIT=True, FIT_GT_opened=True, DEV=False, TEST=False, target_inputs_opened=False,
                historical_fractional_exact=True, images=len(roster), source_groups_with_supervision=92,
                primary_head="S", bundles=bundles, fit_audit=audit, arm_fit_write_verify_seconds=timings,
                head_optimizer_updates_per_arm=updates, head_optimizer_updates=sum(updates.values()),
                update_split_per_arm=dict(S=600, L=600), detector_forwards=0, detector_optimizer_updates=0,
                prelabel_sha256=prelabel_sha, preparation=prepared, supervision=supervision, target_counts=counts,
                seeds=list(SEEDS), target_definitions=dict(F="exact polygon-cell area", T="hard except exact FP64 half retained",
                                                         H="inclusive FP64 half threshold", C="native COCO raster"),
                caps=dict(native_area_min_inclusive=32, native_area_max_exclusive=1024, maximum_pairs_per_candidate=32,
                          prediction_swaps="min(8, floor(area/10), number of pairs) in the closed reader"),
                preparation_and_fit_seconds=perf_counter()-started, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                **source_lineage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "capture", "acceptance", "source-cohort", "fit-images", "fit-gt", "shapely-site",
                 "historical-fit", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    result = run(args)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as handle:
        handle.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
