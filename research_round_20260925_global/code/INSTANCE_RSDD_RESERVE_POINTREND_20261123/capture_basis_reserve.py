"""Capture frozen RF-DETR spatial bases on the 74 Instance-RSDD reserve chips."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


PORTABLE_SHA = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
ORIGINAL_BASIS_SHA = "476a8ff4a9c705df3fa6530fca2c4f58d020dc7995b444480476eeac1e98b8a1"
MANIFEST_SHA = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
PREDICTIONS_SHA = "41163bd418d13a05f9c52b28e7821458bfb5eafca0fcf761682436fbff75f71c"
AUDIT_SHA = "bb10ca305fe08f8fb15c85db96e9b9553d3ed24949c1edcaef03739be041d5dd"
COUNTS = dict(mapped=7346, eligible=4137, overrides=0, rank_gaps=54)
RUNTIME = {"torch": "2.10.0+cu130", "rfdetr": "1.12.0", "Pillow": "12.3.0",
           "pycocotools": "2.0.11", "numpy": "1.26.4"}
STATUS = "INSTANCE_RSDD_RESERVE74_POINTREND_BASIS_CAPTURED_V1"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def pinned_module(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, f"Source identity differs for {path.name}")
    mod = ModuleType("_reserve_basis_" + path.stem)
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def capture(args):
    portable = pinned_module(args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", PORTABLE_SHA)
    original = pinned_module(args.code / "INSTANCE_RSDD_POINTREND_TRANSFER_20261122/capture_basis.py",
                             ORIGINAL_BASIS_SHA)
    native = portable.module(args.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    items, manifest_sha = portable.load_intake(args, native)
    require(manifest_sha == MANIFEST_SHA, "Image manifest differs from the qualified reserve")
    native.install_guard(args.intake, items)
    receipt, pairs = portable.read_capture(args, items, manifest_sha)
    require(receipt["worker_sha256"] == PORTABLE_SHA and receipt["optimizer_updates"] == 0
            and receipt["source_partition"] == "author train" and receipt["counts"] == COUNTS
            and receipt["predictions_sha256"] == PREDICTIONS_SHA and receipt["audit_sha256"] == AUDIT_SHA
            and len(items) == len(pairs) == 74, "Qualified reserve native streams and candidate census required")
    capture_sha = sha256((args.capture / portable.RESULT).read_bytes()).hexdigest()
    common = native.reference(args.code / portable.UEIKAP / "capture.py")

    import numpy as np
    import torch
    from pycocotools import mask as codec

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    # Import order is part of the native 74-image CUDA capture recipe.
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    require(torch.get_float32_matmul_precision() == "high" and torch.backends.cuda.matmul.allow_tf32,
            "Original RF-DETR FP32 CUDA matmul policy required")
    hooks = original.module(args.code / "HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py",
                            original.HOOK_SHA)
    carrier, core, base, post = portable.detector_components(args.code, args.checkpoint)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA device required")
    free, total = torch.cuda.mem_get_info()
    require(free >= base.CAP, "Original CUDA resource floor")
    torch.cuda.set_per_process_memory_fraction(min(0.35, base.CAP / total))
    torch.backends.cudnn.allow_tf32 = False
    model_args = SimpleNamespace(root=args.code, output=args.output, mode="dev")
    model, transform, config = carrier.model_and_transform(model_args, core, base, torch)
    require(not model.training and model.num_queries == 164 and model.group_detr == 13, "Frozen Q164 detector")
    geometry = SimpleNamespace(native_block=carrier.native_block, native_xyxy=native.native_xyxy)
    args.output.mkdir()
    (args.output / "spatial").mkdir()
    index = []
    with torch.inference_mode():
        for ordinal, (item, (row, prediction)) in enumerate(zip(items, pairs, strict=True), 1):
            image = native.decode_native(args.intake, item)
            tensor, target = transform(image, None)
            require(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                    "Original val576 tensor required")
            batch = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            require(batch.mask is not None and not bool(batch.mask.any()), "Unexpected transform padding")
            with hooks.capture_final_query(model.segmentation_head) as observation:
                raw = model(batch.to("cuda"))
            require(raw["pred_masks"].shape == (1, 164, 144, 144), "Full bank of native coarse mask logits required")
            # Seal first: the projected basis must reconstruct every query logit exactly.
            features = observation.seal(raw["pred_masks"]).spatial[0].contiguous().cpu().numpy()
            require(features.shape == (256, 144, 144) and features.dtype == np.float32 and np.isfinite(features).all(),
                    "Unexpected final spatial features")
            planes = common.read_array(args.capture, row["logits_ref"])
            # Independently compare every retained mask, all scores and boxes, and every eligible plane.
            original.compare_carrier(raw, geometry, post, row, prediction, planes, tuple(item["original_hw"]),
                                     np, torch, codec, PostProcess)
            ref = common.write_array(args.output, f"spatial/{ordinal:06d}.npy", features)
            index.append(dict(ordinal=ordinal, image_id=item["image_id"], rgb_sha256=item["rgb_sha256"],
                              eligible_rows=len(planes), spatial=ref))
            if ordinal == 1 or ordinal % 20 == 0:
                print(json.dumps(dict(images=ordinal, total=len(items))), flush=True)
    require(sum(r["eligible_rows"] for r in index) == COUNTS["eligible"], "Full eligible cohort required")
    index_sha = common.write(args.output / "INDEX_PRIVATE.json", common.encoded(index))
    return dict(status=STATUS, phase="capture", images=len(index), counts=COUNTS, model_forwards=len(index),
                optimizer_updates=0, exact_original_candidates_and_masks=True, exact_original_eligible_logits=True,
                exact_all_query_basis_contraction=True, checked_all_query_planes=len(index)*164,
                checkpoint_sha256=portable.CHECKPOINT, roster_sha256=portable.ROSTER, manifest_sha256=manifest_sha,
                index_sha256=index_sha, capture_receipt_sha256=capture_sha, predictions_sha256=PREDICTIONS_SHA,
                audit_sha256=AUDIT_SHA, candidate_audit_sha256=AUDIT_SHA, GT_opened=False, AP=False,
                target_labels_opened=False, source_partition="author train", operator_versions=RUNTIME,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "intake", "roster", "capture", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for key, path in vars(args).items():
        setattr(args, key, path.resolve(strict=key != "output"))
    require(not args.output.exists(), "Supply a fresh output directory")
    require(sys.version_info[:3] == (3, 10, 13) and {k: version(k) for k in RUNTIME} == RUNTIME,
            "Frozen capture runtime required")
    start = perf_counter()
    report = capture(args)
    report.update(seconds=perf_counter()-start, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
