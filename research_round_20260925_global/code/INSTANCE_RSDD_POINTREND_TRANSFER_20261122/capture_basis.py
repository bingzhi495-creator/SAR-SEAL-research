"""Recapture detached RF-DETR spatial bases against the original TEST masks."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


LOCK_SHA = "b4bfdeab8d78c0b7b6c68e3615c3f2c217bb568237749bf43508ea4bf74fa4ad"
LOCK_COMMIT = "11da49604bcc8a7bb871329821754297c159905b"
BASE_SHA = "6549dd151178a17146de518fbe16a3a045d90c5b4ed8345d4d8e272db0e51c2d"
REPLAY_SHA = "38f1c726c7d59c171e21e3943722896ce1b3ca11dcd4a1dd90ef904db8d92aa6"
EDIT_SHA = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
HOOK_SHA = "bc6bf64a7ff01a716fcd4fe97da9fa292a892bbc5699ff57b46c615581ce4adc"
CAPTURE_RECEIPT = "29357a70ceeff135e02ddb98b400b9f3695a8fe1f808f4a2af5e0ed86e4769ec"
CAPTURE_TERMINAL = "30c5bd77c5703621a4e3c4e2a555b97ca4df6aebb8538080a14a8bc3b7a4b275"
MANIFEST_SHA = "16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184"
STATUS = "INSTANCE_RSDD_POINTREND_BASIS_CAPTURED_V1"


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, f"Pinned bytes: {path.name}")
    return raw


def module(path, expected):
    loaded = ModuleType("_rsdd_point_" + path.stem)
    loaded.__file__ = str(path)
    sys.modules[loaded.__name__] = loaded
    exec(compile(pin(path, expected), str(path), "exec"), loaded.__dict__)
    return loaded


def input_modules(code):
    native_dir = code / "INSTANCE_RSDD_FROZEN_HRSID_20261122"
    base = module(native_dir / "capture_instance_rsdd.py", BASE_SHA)
    replay = module(native_dir / "replay_rsdd.py", REPLAY_SHA)
    edit = module(code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", EDIT_SHA)
    return base, replay, edit


def read_authority(payload):
    paths = {"code", "intake", "manifest", "roster", "reference", "carrier", "private_root",
             "capture", "capture_receipt", "capture_terminal", "protocol"}
    need(set(payload) == paths and all(Path(v).is_absolute() for v in payload.values()), "Complete absolute phase inputs")
    return SimpleNamespace(**{k: Path(v).resolve(strict=True) for k, v in payload.items()},
                           capture_receipt_sha=CAPTURE_RECEIPT, capture_terminal_sha=CAPTURE_TERMINAL)


def compare_carrier(raw, source, post, row, original, planes, hw, np, torch, codec, processor):
    query = post.aligned_query_rows(raw, hw, torch=torch, np=np, codec=codec, processor=processor)
    need(source.native_block(row["image_id"], query) == original, "Exact archived scores, candidate ranks and native raw masks")
    boxes = raw["pred_boxes"][0].cpu().numpy()
    need(np.array_equal(boxes, np.asarray(row["raw_boxes_cxcywh"], dtype=np.float32))
         and np.array_equal(raw["pred_logits"][0].sigmoid().cpu().numpy(),
                            np.asarray(row["raw_query_probabilities"], dtype=np.float32)),
         "Exact archived full query boxes and probabilities")
    need(np.array_equal(source.native_xyxy(boxes, hw)[row["query_ids"]],
                        np.asarray(row["native_boxes_xyxy"], dtype=np.float32).reshape(-1, 4)),
         "Exact mapped native clipped boxes")
    logit_ids = torch.as_tensor(row["logit_query_ids"], dtype=torch.long, device=raw["pred_masks"].device)
    sampled = raw["pred_masks"][0].index_select(0, logit_ids).cpu().numpy()
    need(planes.shape == (len(logit_ids), 144, 144) and np.array_equal(sampled, planes),
         "Exact archived eligible FP32 coarse planes")


def capture(a, output):
    base, replay, edit = input_modules(a.code)
    items = base.roster(json.loads(pin(a.roster, base.AUTHOR_ROSTER_SHA)), json.loads(pin(a.manifest, MANIFEST_SHA)))
    need(a.intake == a.manifest.parent, "Verified image root")
    base.install_guard(a.intake, items)
    common = base.reference(a.reference)
    source = common.exact_source(a)
    rows, receipt = replay.read_capture(a, edit)
    need(receipt["intake_sha256"] == MANIFEST_SHA and receipt["counts"] ==
         dict(mapped=73226, eligible=43252, rank_gaps=774, overrides=0), "Exact accepted original carrier census")

    import numpy as np
    import torch
    from pycocotools import mask as codec

    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
         and version("Pillow") == "12.3.0" and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4",
         "Original CUDA numerical runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA device for parity")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    # The original capture initializes torch before importing RF-DETR, whose
    # package import then switches matmul precision to high (enabling CUDA TF32).
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    need(torch.get_float32_matmul_precision() == "high" and torch.backends.cuda.matmul.allow_tf32,
         "Original effective FP32 matmul policy after RF-DETR import")
    hooks = module(a.code / "HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py", HOOK_SHA)
    core = source.load_core(a.private_root)
    args = SimpleNamespace(root=a.private_root, output=output, mode="dev")
    native, post = source.setup(args, core)
    free, capacity = torch.cuda.mem_get_info()
    need(free >= native.CAP, "Original CUDA resource floor")
    torch.cuda.set_per_process_memory_fraction(min(0.35, native.CAP / capacity))
    torch.backends.cudnn.allow_tf32 = False
    model, transform, config = source.model_and_transform(args, core, native, torch)
    need(not model.training and model.num_queries == 164 and model.group_detr == 13, "Original detector in eval mode")
    output.mkdir()
    (output / "spatial").mkdir()

    index = []
    with torch.inference_mode():
        for ordinal, (item, (row, original)) in enumerate(zip(items, rows, strict=True), 1):
            hw = tuple(item["original_hw"])
            need(item["image_id"] == row["image_id"] and list(hw) == row["original_hw"]
                 and item["image_ref"]["sha256"] == row["image_sha256"] and item["rgb_sha256"] == row["rgb_sha256"],
                 "Same accepted native RGB chip and dimensions")
            image = base.decode_native(a.intake, item)
            tensor, target = transform(image, None)
            need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32, "Original val576")
            batch = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            need(batch.mask is not None and not bool(batch.mask.any()), "Original unpadded image tensor")
            with hooks.capture_final_query(model.segmentation_head) as observation:
                raw = model(batch.to("cuda"))
            features = observation.seal(raw["pred_masks"]).spatial[0].contiguous().cpu().numpy()
            need(features.shape == (256, 144, 144) and features.dtype == np.float32 and np.isfinite(features).all(),
                 "Exact finite detached FP32 projected spatial basis")
            planes = common.read_array(a.capture, row["logits_ref"])
            # Compare after sealing the basis, including all original final-query boxes and masks.
            geometry = SimpleNamespace(native_block=source.native_block, native_xyxy=base.native_xyxy)
            compare_carrier(raw, geometry, post, row, original, planes, hw, np, torch, codec, PostProcess)
            ref = common.write_array(output, f"spatial/{ordinal:06d}.npy", features)
            index.append(dict(ordinal=ordinal, image_id=item["image_id"], rgb_sha256=item["rgb_sha256"],
                              eligible_rows=len(planes), spatial=ref))
            if ordinal % 20 == 0 or ordinal == 1:
                print(json.dumps(dict(images=ordinal, total=len(items))), flush=True)
    return dict(status=STATUS, images=len(index), model_forwards=len(index), optimizer_updates=0,
                counts=receipt["counts"], index_sha256=common.write(output / "INDEX_PRIVATE.json", common.encoded(index)),
                exact_original_candidates_and_masks=True, exact_original_eligible_logits=True,
                exact_all_query_basis_contraction=True, candidate_audit_sha256=receipt["audit_sha256"],
                predictions_sha256=receipt["predictions_sha256"], checkpoint_sha256=base.SOURCE_CHECKPOINT,
                capture_receipt_sha256=CAPTURE_RECEIPT, intake_sha256=MANIFEST_SHA, GT_opened=False,
                author_TEST=True, TEST=True, FIT=False, DEV=False, AP=False,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("authority", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--authority-sha", required=True)
    cfg = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and cfg.output.is_absolute()
         and cfg.output.name == "capture" and not cfg.output.exists(), "Fresh isolated capture")
    a = read_authority(json.loads(pin(cfg.authority, cfg.authority_sha)))
    pin(a.protocol, LOCK_SHA)
    start = perf_counter()
    result = capture(a, cfg.output)
    result.update(protocol_sha256=LOCK_SHA, prelock_commit=LOCK_COMMIT, seconds=perf_counter()-start,
                  worker_sha256=digest(Path(__file__).read_bytes()), input_authority_sha256=cfg.authority_sha)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    target = cfg.output / "RESULT_SAFE.json"
    with target.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
