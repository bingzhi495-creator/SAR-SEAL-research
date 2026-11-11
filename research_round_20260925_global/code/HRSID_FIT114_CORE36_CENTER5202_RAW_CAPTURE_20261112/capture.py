"""Export the original frozen RF-DETR Q164 candidates for the fixed HRSID FIT large-positive roster."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

ROOT = Path(os.environ["SAR_SEAL_CLOSED_FIT_ROOT"]).resolve(strict=True)
CODE = ROOT / "github_release_20261011/research_round_20260925_global/code"
LAYOUT = ROOT / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
RESTART = ROOT / "HRSID_MASK_COMMON_FIT_E24_5202_CENTER_RESTART_PRIVATE_20261110_01"
RECEIPTS = RESTART / "INDEPENDENT_ACCEPTANCE_PRIVATE"
CHECKPOINTS = {
    "native": ROOT / "HRSID_MASK_COMMON_FIT_E24_5202_PRIVATE_20261110_01/native/NATIVE_ACTUAL_PRIVATE/checkpoint_23.ckpt",
    "center": RESTART / "center/NATIVE_ACTUAL_PRIVATE/checkpoint_23.ckpt",
}
PINS = {
    "native": "d2d7c584c75a4bd4397f523ce9d7c827ae7d0aaf9bec61f5ee02eaf33c718c03",
    "center": "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11",
}
INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
LAYOUT_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
PAIR_SHA = "ec6c36973a763344d0f063727e7e07c697a132573e6235eaa8525482e8d8da9f"
TRAIN_PROTOCOL = "04308c52bd2b4730f1474335c6316f6b8a11762e"
DESIGN_SHA = '62b9d59c14628e7166b6b38dbbd4ee59ce6d08c3f3ead56e096b96f2cba7cbab'
SOURCES = {
    "admission": ("HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/common_fit_eval_admission.py",
                  "7a38fa028c07bf16d2da42935c5ee117f9e9ee5a88ede4337a43b1199c5ced13"),
    "phase": ("HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py",
              "488f9ed73943cc842162c0d45822b6bac42aecabf37c8ecf0d4dde8e96c0959b"),
    "filtered": ("HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py",
                 "9a69500d10d5b0a371ea504f1ba89b7335aca130c1a28d96c4f93a87b2a4b9c0"),
    "base": ("HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
             "1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"),
    "fit_audit": ("HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py",
                  "eb5f91a333ea7c7a1c12a87574d0c192f678aa80a742107a510654e3d5512df5"),
    "post": ("HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py",
             "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"),
}
ROSTER_FIELDS = ("image_id", "native_basename", "encoded_bytes", "encoded_sha256", "decoded_rgb_sha256", "group")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned(path, digest, limit):
    need(path.is_absolute() and path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit,
         "Expected a canonical bounded input")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Pinned input differs: " + path.name)
    return raw


def source(key):
    name, digest = SOURCES[key]
    path = CODE / name
    return path, pinned(path, digest, 1024**2)


def load(key):
    path, data = source(key)
    module = ModuleType("_p378_" + key)
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, value):
    raw = encoded(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def infer(args):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    start = perf_counter()
    need(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
         and version("pycocotools") == "2.0.11" and version("Pillow") == "12.3.0"
         and np.__version__ == "1.26.4" and not torch.cuda.is_initialized(), "Exact isolated native runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    arm = args.arm
    admission = load("admission")
    phase = admission.load_phase(source("phase")[0])
    accepted = admission.read_pair(
        phase, RECEIPTS / "PAIR_READY_PRIVATE.json", PAIR_SHA,
        native_path=RECEIPTS / "NATIVE_ARM_ACCEPTED_PRIVATE.json",
        center_path=RECEIPTS / "CENTER_ARM_ACCEPTED_PRIVATE.json", mode=arm, training_protocol=TRAIN_PROTOCOL,
    )
    need(accepted["checkpoint_sha256"] == PINS[arm], "Fixed independently accepted arm")
    filtered, base = phase.load_evaluator(source("filtered")[0], source("base")[0])
    admission.bind_evaluator(phase, base, filtered, accepted)
    base.FIT_AUDIT = source("fit_audit")[0]
    base.RECIPE = CODE.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    for name, digest in base.MODEL_SOURCES.items():
        pinned(Path(import_module(name).__file__), digest, 1024**2)
    post = load("post")
    manifest = json.loads(pinned(args.manifest, args.manifest_sha, 64*1024))
    need(manifest["status"] == "HRSID_FIT114_CORE36_CENTER5202_IMAGE_ONLY_MANIFEST_V1" and manifest["FIT"] is True
         and manifest["DEV"] is manifest["TEST"] is False
         and manifest["closed_roster_design_sha256"] == DESIGN_SHA
         and manifest["source_p316_cohort_sha256"] == "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e", "Fixed image-only FIT roster")
    rows = manifest["images"]
    need(len(rows) == len({row["image_id"] for row in rows}) == len({row["group"] for row in rows}) == 36
         and len({row["decoded_rgb_sha256"] for row in rows}) == 36
         and rows == sorted(rows, key=lambda row: row["decoded_rgb_sha256"])
         and all(set(row) == set(ROSTER_FIELDS) for row in rows), "36 unique locked image-only rows")
    need(CHECKPOINTS[arm].name == admission.CHECKPOINT_NAME, "Exact E24 archive filename")
    model, config, train, _ = base.load_model(CHECKPOINTS[arm], PINS[arm], torch,
                                              args.output.parent, TRAIN_PROTOCOL)
    model.eval()
    need(all(value.device.type == "cpu" for value in model.parameters())
         and model.num_queries == 164 and model.group_detr == 13, "Frozen CPU Q164 EMA")
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
        scale_jitter=train.scale_jitter, gpu_postprocess=False,
    )
    args.output.mkdir()
    predictions_sha = sha256()
    logits_dir = args.output / "RAW_LOGITS"
    logits_dir.mkdir()
    mapped = empty = logits_total = logits_bytes = 0
    with (args.output / "QUERIES_PRIVATE.jsonl").open("xb") as stream, torch.inference_mode():
        for ordinal, row in enumerate(rows, 1):
            name = row["native_basename"]
            need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name), "Exact FIT PNG basename")
            data = pinned(LAYOUT / "train" / name, row["encoded_sha256"], 4*1024**2)
            image, scales = decode_pil_image_bytes(data, draft_size=None)
            need(image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.)
                 and len(data) == row["encoded_bytes"]
                 and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
                 "Exact closed FIT RGB input")
            tensor, target = transform(image, None)
            need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                 "Stock GT-free val576")
            sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            need(sample.mask is not None and not bool(sample.mask.any()), "No stock canvas padding")
            raw = model(sample)
            query = post.aligned_query_rows(raw, (800, 800), torch=torch, np=np, codec=codec, processor=PostProcess)
            probs = raw["pred_logits"][0].sigmoid()
            joint = torch.argsort(probs.flatten(), descending=True, stable=True)[:100]
            areas = codec.area(query["all_rles"])
            retained = [(qid, rank, int(areas[qid])) for qid, rank in
                        zip(query["native_ids"], query["native_ranks"], strict=True) if 32 <= areas[qid] < 1024]
            qids = torch.tensor([item[0] for item in retained], dtype=torch.long)
            values = raw["pred_masks"][0].index_select(0, qids).contiguous().numpy().astype("<f4", copy=False)
            need(values.shape == (len(retained), 144, 144) and bool(np.isfinite(values).all()),
                 "Exact raw finite FP32 logit plane per admitted native candidate")
            logit_name = f"{ordinal:03d}.npy"
            logit_path = logits_dir / logit_name
            with logit_path.open("xb") as target:
                np.save(target, values, allow_pickle=False)
                target.flush()
                os.fsync(target.fileno())
            raw_file = logit_path.read_bytes()
            logits_total += len(retained)
            logits_bytes += len(raw_file)
            captured = dict(relative_path="RAW_LOGITS/" + logit_name, sha256=sha256(raw_file).hexdigest(),
                            bytes=len(raw_file), shape=list(values.shape), candidates=[
                                dict(logit_row=i, query_id=qid, native_rank=rank,
                                     score=query["ship_scores"][qid], predicted_area=area)
                                for i, (qid, rank, area) in enumerate(retained)])
            block = dict(image_id=row["image_id"], source_prefix=row["group"], image_content_sha256=row["decoded_rgb_sha256"],
                         encoded_sha256=row["encoded_sha256"], native_basename=row["native_basename"],
                         logit_capture=captured, **query, raw_boxes_cxcywh=raw["pred_boxes"][0].tolist(),
                         raw_classifier_logits=raw["pred_logits"][0].tolist(),
                         joint_query_ids=(joint//2).tolist(), joint_labels=(joint%2).tolist(),
                         joint_scores=probs.flatten().index_select(0, joint).tolist())
            line = encoded(block)
            need(stream.write(line) == len(line), "Complete original candidate row")
            predictions_sha.update(line)
            mapped += len(query["native_ids"])
            empty += int(np.count_nonzero(codec.area(query["all_rles"]) == 0))
            if ordinal % 6 == 0:
                print(json.dumps(dict(event="FIT114_CORE36_CENTER5202_Q164_PROGRESS", arm=arm, images=ordinal)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    need(not torch.cuda.is_initialized() and not model.training, "CPU-only frozen inference")
    result = dict(status="FIT114_CORE36_CENTER5202_Q164_CPU_ARM_COMPLETE_V1", arm=arm, FIT=True, DEV=False,
                  TEST=False, AP=False, GT_opened=False, images=36, model_forward_calls=36,
                  optimizer_updates=0, all_ship_queries=36*164, native_joint_rows=36*100,
                  raw_logit_rows=logits_total, raw_logit_bytes=logits_bytes, raw_logit_files=36,
                  native_mapped_rows=mapped, all_empty=empty, native_all_query_rles_equal_on_joint=True,
                  manifest_sha256=args.manifest_sha, checkpoint_sha256=PINS[arm], pair_acceptance_sha256=PAIR_SHA,
                  prediction_sha256=predictions_sha.hexdigest(), device="cpu", precision="float32",
                  evaluation_shape=[576, 576], output_mask_shape=[800, 800],
                  original_cuda_bitwise_equivalence_asserted=False, elapsed_seconds=perf_counter()-start)
    result_sha = write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], arm=arm, result_sha256=result_sha,
                          predictions_sha256=result["prediction_sha256"], images=36)), flush=True)


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
        raise PermissionError("Offline FIT worker")
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = Path(os.fsdecode(args[0]))
        if name.name.lower().endswith(".coco.json") or any(part.lower() in ("dev", "test", "valid") for part in name.parts):
            raise PermissionError("FIT capture opens no FIT GT, DEV or TEST data")


def main():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode", choices=("infer",))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha")
    parser.add_argument("--arm", choices=tuple(PINS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(sys.platform == "win32" and sys.version_info[:3] == (3, 10, 13)
         and sys.flags.isolated and sys.flags.utf8_mode and sys.dont_write_bytecode,
         "Exact isolated worker Python")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("HF_HUB_OFFLINE") == "1"
         and all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
         and args.arm == "center" and args.manifest_sha and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists(), "Bounded CPU and fresh arm output required")
    sys.addaudithook(offline)
    infer(args)


if __name__ == "__main__":
    main()
