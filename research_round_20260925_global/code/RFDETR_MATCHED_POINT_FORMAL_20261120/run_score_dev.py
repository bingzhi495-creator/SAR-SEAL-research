"""Score the independently closed RF-DETR matched DEV705 outputs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

STOCK = "da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e"
OLD_SCORE = "f980babe7cc6b3e701885ec136fb359fa2e2514e49fc08d3faf14995602270fa"
NATIVE_SOURCE = "1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"
MATHEMATICS = "271554ea01f5e5e24e48ff2c53c3229755615aa514624d7d89e5c9ec559fb79c"
GEOMETRY = "fdee52489ba2df0e9db147f26d41432a3276638360f2858fb5b8c5b739953a44"
SOURCE_READOUT = "543bc7ffae51cb0de5b247fef59eeae6309a155440021d3ed50a7b3716ec8e9f"
SOURCE_FIT_STEP = "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1"
SOURCE_SPOT = "75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee"


def load_fit_step(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_FIT_STEP:
        raise ValueError("Exact original fit-step source")
    module = ModuleType("_rf_original_fit_step")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def verify_readout_pair(accepted, stage, readout):
    if not (
        accepted["readout_source_sha256"] == stage["worker_sha256"] == SOURCE_READOUT
        and accepted["stock_predictions_sha256"] == stage["stock_predictions_sha256"] == STOCK
        and tuple(stage["seeds"]) == readout.SEEDS and tuple(stage["arms"]) == readout.ARMS
    ):
        raise ValueError("Exact source, native stock, and locked seed and comparison plan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "capture-accepted", "image-manifest", "readout-root", "readout-accepted",
                 "historical-native", "historical-score", "original-dev-dir", "code", "spot-source", "shapely-site", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("capture-accepted-sha", "readout-accepted-sha"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--probability-repair", type=Path)
    ap.add_argument("--probability-repair-sha")
    args = ap.parse_args()
    spot_source = args.spot_source.resolve(strict=True)
    fit_step = load_fit_step(spot_source / "fit_step.py")
    spot = fit_step.load(spot_source / "run_spot.py", SOURCE_SPOT, "_rf_original_slots")
    import numpy as np
    import torch
    from pycocotools import mask as codec

    spot.need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only fixed DEV scoring")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    here = Path(__file__).resolve().parent
    readout = fit_step.load(here / "run_dev_readout.py", SOURCE_READOUT, "_rf_label_free_reader")
    scorer = fit_step.load(here / "score_dev.py", MATHEMATICS, "_rf_fixed_dev_scoring")
    geometry = fit_step.load(here / "evaluate_geometry.py", GEOMETRY, "_rf_matched_geometry")
    native = fit_step.load(args.code / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
                           NATIVE_SOURCE, "_original_native_rfd")
    native.DEV = args.original_dev_dir
    native.OFFICIAL = args.code / "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"

    approved = json.loads(spot.pinned(args.readout_accepted, args.readout_accepted_sha, 1 * spot.MIB))
    spot.need(approved["status"] == "RFDETR_CENTER5202_DEV705_MATCHED_READOUT_INDEPENDENT_ACCEPTED"
              and approved["DEV"] and not any(approved[k] for k in ("FIT", "TEST", "AP", "GT_opened"))
              and approved["images"] == 705 and approved["original_exit"] == 0
              and approved["capture_accepted_sha256"] == args.capture_accepted_sha
              and all(approved[key] for key in ("exact_candidate_identity", "exact_native_support", "exact_per_candidate_budgets",
                                                "actual_XOR_area_RLE_verified")),
              "Only an independently verified label-free 705-image readout")
    stage = json.loads(spot.pinned(args.readout_root / "RESULT_SAFE.json", approved["readout_result_sha256"], spot.MIB))
    spot.need(stage["status"] == readout.STATUS and stage["capture_accepted_sha256"] == args.capture_accepted_sha
              and stage["edits_sha256"] == approved["edits_sha256"]
              and not any(stage[k] for k in ("FIT", "TEST", "GT_opened", "AP")),
              "Exact original prelabel sparse materialization")
    verify_readout_pair(approved, stage, readout)
    original_prelabel = json.loads(spot.pinned(args.readout_root / "PRELABEL_SAFE.json", stage["prelabel_sha256"], 64*1024))
    spot.need(original_prelabel["stock_predictions_sha256"] == STOCK and
              original_prelabel["capture_accepted_sha256"] == args.capture_accepted_sha and
              not original_prelabel["GT_opened"], "Same original native stock before labels")

    images, blocks, capacity = readout.original_population(args, spot)
    had_repair = args.probability_repair_sha is not None
    original_rows = capacity["independent_CPU_raster_disagree_rows"]
    original_pixels = capacity["independent_CPU_raster_disagree_pixels"]
    if had_repair:
        for receipt in (approved, stage):
            spot.need(receipt["probability_repair_sha256"] == args.probability_repair_sha
                      and receipt["probability_repair_consumed"] == 1
                      and receipt["original_CPU_disagree_rows"] == original_rows == 1
                      and receipt["original_CPU_disagree_pixels"] == original_pixels == 1
                      and receipt["effective_unresolved_CPU_disagree_rows"] == 0
                      and receipt["effective_unresolved_CPU_disagree_pixels"] == 0,
                      "Same verified CUDA probability correction consumed once before labels")
    else:
        spot.need(original_rows == original_pixels == 0, "Zero unresolved original threshold differences")
    sparse = [json.loads(line) for line in spot.pinned(args.readout_root / "SPARSE_EDITS_PRIVATE.jsonl",
                                                       approved["edits_sha256"], 512*spot.MIB).splitlines()]
    old_stream = spot.pinned(args.historical_native, STOCK, 512*spot.MIB)
    old_score = json.loads(spot.pinned(args.historical_score, OLD_SCORE, 64*1024))
    pool = []
    for item, block in zip(images, blocks, strict=True):
        rows = spot.native_candidate_rows(block)
        xyxy = spot.author_boxes(block["raw_boxes_cxcywh"], torch)
        for row in rows:
            row["area"] = int(codec.area(row["rle"]))
            row["box_xyxy"] = xyxy[row["query_id"]].tolist()
        pool.append(dict(native_block=readout.stock_block(item["image_id"], rows),
                         pool=rows, source_prefix=block["source_prefix"]))
    spot.need([row["image_id"] for row in sparse] == [row["image_id"] for row in images]
              and len(sparse) == 705, "Exact source mask and sparse row order")
    args.output.mkdir()
    seal = spot.write(args.output / "PRELABEL_SAFE.json", dict(status="RFDETR_CENTER5202_DEV705_SCORE_PRELABEL",
        verified_readout_sha256=args.readout_accepted_sha, sparse_sha256=approved["edits_sha256"],
        target_native_sha256=STOCK, previous_native_score_sha256=OLD_SCORE, images=705,
        DEV=True, FIT=False, TEST=False, GT_opened=False, AP=False))
    gt = native.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    findings = native.dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    empties = {row["annotation_id"] for row in findings if row["kind"] == "EMPTY_COCO_RASTER"}
    spot.need(len(empties) == len(findings) == 4, "Original four empty raster IDs")
    helpers = fit_step.load_geometry(args.code, args.shapely_site)
    result = scorer.score_dev(np, codec, native, gt=gt, empty_ids=empties, capture_rows=pool, sparse_rows=sparse,
                              historical_native_bytes=old_stream, historical_scores=old_score,
                              interpolation_mismatches=0,
                              fit_step=fit_step, geometry_helpers=helpers, geometry_module=geometry)
    private_rows = result["geometry"].pop("private_witness")
    private_digest = spot.write(args.output / "SHAPE_MATCH_WITNESS_PRIVATE.json", private_rows)
    spot.need(not torch.cuda.is_initialized(), "No accelerator context while scoring")
    artifact = dict(result)
    artifact.update(status="RFDETR_CENTER5202_DEV705_MATCHED_SCORE_ACTUAL",
                    DEV_GT_opened=True, prelabel_sha256=seal,
                    original_CPU_disagree_rows=original_rows, original_CPU_disagree_pixels=original_pixels,
                    probability_repair_sha256=args.probability_repair_sha,
                    private_shape_witness_sha256=private_digest, wall_seconds=perf_counter()-start)
    digest = spot.write(args.output / "RESULT_SAFE.json", artifact)
    print(json.dumps(dict(status=artifact["status"], result_sha256=digest, gate=artifact["gate"])), flush=True)


if __name__ == "__main__":
    main()
