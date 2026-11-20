"""Run the predeclared source-only 109-image point-head refits from an accepted RF capture."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

SOURCE_COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
CHECKPOINT_SHA = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"


def read_population(args, spot):
    """Use sealed capture plus its original 109-source FIT roster."""
    original = json.loads(spot.pinned(args.source_cohort, SOURCE_COHORT_SHA, 4 * spot.MIB))
    core = [row for row in original["images"] if row["core"]]
    spot.need(len(core) == len({row["group"] for row in core}) == 109, "Exactly one original image per FIT prefix")
    core.sort(key=lambda row: (row["decoded_rgb_sha256"], row["image_id"]))
    close = json.loads(spot.pinned(args.acceptance, args.acceptance_sha256, 64 * 1024))
    spot.need(close["status"] == "RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1" and close["mode"] == "fit"
              and close["FIT"] and not any(close[key] for key in ("DEV", "TEST", "GT_opened"))
              and close["images"] == 109 and close["model_forwards"] == 0
              and close["original_os_exit"] == 0 and close["job_drained"] and close["cache_hits"] == 36
              and close["source_checkpoint_sha256"] == CHECKPOINT_SHA
              and close["independent_CPU_raster_disagree_rows"] == 0,
              "Independent FIT-only carrier verification")
    manifest = json.loads(spot.pinned(args.capture / "IMAGE_MANIFEST_PRIVATE.json",
                                     close["image_manifest_sha256"], 1 * spot.MIB))
    lines = spot.pinned(args.capture / "QUERIES_PRIVATE.jsonl", close["queries_sha256"], 24 * spot.MIB).splitlines()
    predicted = [json.loads(line) for line in lines]
    keys = ("image_id", "native_basename", "encoded_bytes", "encoded_sha256", "decoded_rgb_sha256", "group")
    images = [{key: row[key] for key in keys} for row in core]
    captured = [{key: row[key] for key in ("image_id", "encoded_sha256", "decoded_rgb_sha256")}
                for row in manifest]
    expected = [{key: row[key] for key in ("image_id", "encoded_sha256", "decoded_rgb_sha256")}
                for row in images]
    spot.need(expected == captured and len(predicted) == len(images) and
              [row["image_id"] for row in predicted] == [item["image_id"] for item in images] and
              [row["origin"] for row in manifest] == ["accepted_CPU_CORE36"] * 36 + ["new_same_CPU_FP32"] * 73,
              "Exact 36 cached plus 73 new image identities and order")
    return [(item, i, block) for i, (item, block) in enumerate(zip(images, predicted, strict=True), 1)], close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "acceptance", "source-cohort", "fit-images", "fit-gt", "code",
                 "posterior", "spot-source", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    args = parser.parse_args()

    sys.path[0:0] = [str(Path(__file__).resolve().parent), str(args.spot_source.resolve(strict=True))]
    import fit_plan
    import fit_step
    import feature_adapter
    import run_spot as spot
    import numpy as np
    import torch
    from pycocotools import mask as codec

    spot.need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Explicit CPU-only calibration")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    spot.need(not torch.cuda.is_initialized(), "No GPU")
    start = perf_counter()
    roster, accepted = read_population(args, spot)
    post = fit_step.load(args.posterior, spot.ADAPTER_SHA, "_rfd_native_posterior")
    data, prepared, feature_digest = spot.prepare(args, roster, fit_step, feature_adapter, post, np, torch, codec)
    spot.need(prepared["exact_original_masks"] == prepared["eligible"] == accepted["same_device_raster_exact"]
              and prepared["mapped"] == accepted["native_mapped_rows"],
              "All eligible masks and mapped candidates agree with independent capture")
    args.output.mkdir()
    prelabel = dict(status="RFDETR_MATCHED_POINTS_FIT109_PRELABEL", features=prepared,
                    feature_sha256=feature_digest, accepted_capture_sha256=args.acceptance_sha256,
                    FIT=True, DEV=False, TEST=False, GT_opened=False)
    pre = spot.write(args.output / "PRELABEL_SAFE.json", prelabel)
    print(json.dumps(prelabel), flush=True)

    records, label_counts = spot.attach(args, data, fit_step)
    del data
    trainer = fit_step.load_trainer(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    original_groups = [item["group"] for item, _, _ in roster]
    (args.output / "POINT_HEADS_PRIVATE").mkdir()
    audit = fit_plan.fit_fixed_schedule(records, original_groups, args.output / "POINT_HEADS_PRIVATE", fit_step, trainer)
    elapsed = perf_counter() - start
    spot.need(not torch.cuda.is_initialized() and len(audit) == len(fit_plan.SEEDS) * 6, "All five folds and full refits")
    result = dict(status="RFDETR_MATCHED_POINTS_FIT109_RUN_COMPLETE", FIT=True, DEV=False, TEST=False, AP=False,
                  images=109, source_seed_values=list(fit_plan.SEEDS), prelabel_sha256=pre, preparation=prepared,
                  supervision=label_counts, detector_checkpoint_sha256=CHECKPOINT_SHA, detector_forward_calls=0,
                  head_updates=2 * len(audit) * fit_plan.STEPS, all_initializations=audit, seconds=elapsed)
    digest = spot.write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], sha256=digest, images=109, head_updates=result["head_updates"])), flush=True)


if __name__ == "__main__":
    main()
