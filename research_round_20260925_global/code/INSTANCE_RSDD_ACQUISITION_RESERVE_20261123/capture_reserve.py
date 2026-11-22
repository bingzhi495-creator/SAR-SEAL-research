"""Capture and independently verify frozen native RF-DETR predictions on the 74-chip reserve."""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


PROTOCOL = "3cc00d35d65318c3ae15c041c0f0b71f0c0dc743e0286c70955b9c3740009727"
ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
NATIVE = "6549dd151178a17146de518fbe16a3a045d90c5b4ed8345d4d8e272db0e51c2d"
CAPTURED = "INSTANCE_RSDD_RESERVE74_RFDETR_CAPTURE_COMPLETE_V1"
VERIFIED = "INSTANCE_RSDD_RESERVE74_RFDETR_READBACK_COMPLETE_V1"
SENSORS = {"GF-3": 42, "TerraSAR-X": 32}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, f"Digest mismatch for {path.name}")
    return raw


def read_authority(path, expected, phase):
    values = json.loads(pin(path, expected))
    paths = {"code", "private_root", "protocol", "intake", "manifest", "author_roster",
             "intake_result", "intake_terminal"}
    hashes = {"manifest_sha", "author_roster_sha", "intake_result_sha", "intake_terminal_sha"}
    if phase == "verify-capture":
        paths |= {"capture", "capture_terminal"}
        hashes |= {"capture_result_sha", "capture_terminal_sha"}
    need(set(values) == paths | hashes | {"prelock_commit"}, "Exact phase authority fields")
    need(all(re.fullmatch("[0-9a-f]{64}", values[k]) for k in hashes)
         and re.fullmatch("[0-9a-f]{40}", values["prelock_commit"]), "Full SHA identifiers")
    for key in paths:
        location = Path(values[key])
        need(location.is_absolute(), f"Absolute path required for {key}")
        values[key] = location.resolve(strict=True)
    return SimpleNamespace(**values)


def native_module(code):
    path = code / "INSTANCE_RSDD_FROZEN_HRSID_20261122" / "capture_instance_rsdd.py"
    module = ModuleType("_published_native_capture")
    module.__file__ = str(path)
    exec(compile(pin(path, NATIVE), str(path), "exec"), module.__dict__)
    return module


def observe(terminal_path, terminal_sha, result_path, result_sha):
    terminal = json.loads(pin(terminal_path, terminal_sha))
    result = json.loads(pin(result_path, result_sha))
    need(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
         and terminal["result_sha256"] == result_sha
         and terminal["worker_sha256"] == result["worker_sha256"], "Observed clean worker and bound result")
    return terminal, result


def read_intake(a, native):
    pin(a.protocol, PROTOCOL)
    need(a.author_roster_sha == ROSTER and a.intake == a.manifest.parent == a.intake_result.parent,
         "Prespecified roster and common intake root")
    _, result = observe(a.intake_terminal, a.intake_terminal_sha, a.intake_result, a.intake_result_sha)
    need(result["status"] == "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1"
         and result["image_manifest_sha256"] == a.manifest_sha and result["roster_sha256"] == ROSTER
         and result["source_split"] == "author train" and result["sensor_counts"] == SENSORS
         and result["acquisition_groups"] == 13 and result["common_test_parents"] == 0
         and result["images"] == result["member_crc32_verified"] == result["exact_qualified_rf_rgb"]
         == result["rgb_decode_verified"] == 74
         and result["target_polygon_rows_opened"] == result["model_forwards"] == result["optimizer_updates"] == 0,
         "Complete image-only admission for the reserved parent scenes")
    reader = csv.DictReader(StringIO(pin(a.author_roster, ROSTER).decode("utf-8")))
    need(reader.fieldnames == ["author_train_image_id", "file_name", "paper_scene", "sensor"],
         "Original reserve CSV columns")
    rows = list(reader)
    images = [dict(id=int(r["author_train_image_id"]), file_name=r["file_name"], height=512, width=512)
              for r in rows]
    items = native.roster({"images": images}, json.loads(pin(a.manifest, a.manifest_sha)), size=74)
    need(all((item["acquisition_group"], item["sensor"]) == (int(row["paper_scene"]), row["sensor"])
             for item, row in zip(items, rows, strict=True)), "Per-image parent and sensor from the locked CSV")
    return items


def verify(a, items, native, common, torch, shared):
    """Derive ranks, scores and eligible mask planes anew from the stored query tensors."""
    from pycocotools import mask as codec

    terminal, previous = observe(a.capture_terminal, a.capture_terminal_sha, a.capture / "RESULT_SAFE.json",
                                 a.capture_result_sha)
    need(terminal["phase"] == "capture" and previous["status"] == CAPTURED
         and previous["model_forward_calls"] == len(items) and previous["GT_opened"] is False
         and all(previous[key] == value for key, value in shared.items()),
         "Same observed reserve capture, source, cohort and protocol")
    preds = [json.loads(x) for x in pin(a.capture / native.PRED, previous["predictions_sha256"]).splitlines()]
    audit = [json.loads(x) for x in pin(a.capture / native.AUDIT, previous["audit_sha256"]).splitlines()]
    totals = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    for item, pred, row in zip(items, preds, audit, strict=True):
        native.decode_native(a.intake, item)
        num, hw = item["image_id"], item["original_hw"]
        need(num == row["image_id"] == pred["image_id"] and row["original_hw"] == hw
             and row["image_sha256"] == item["image_ref"]["sha256"]
             and all(row[k] == item[k] for k in ("rgb_sha256", "sensor", "acquisition_group", "source_image_name")),
             "Exact image provenance")
        probs = np.asarray(row["raw_query_probabilities"], np.float32)
        boxes = np.asarray(row["raw_boxes_cxcywh"], np.float32)
        need(probs.shape == (164, 2) and np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all()
             and boxes.shape == (164, 4) and np.isfinite(boxes).all(), "Original full query probabilities and boxes")
        best = np.argsort(-probs.ravel(), kind="stable")[:100]
        retained = [(rank, int(k) // 2) for rank, k in enumerate(best, 1) if k % 2 == 0]
        ranks, ids = [r for r, _ in retained], [q for _, q in retained]
        xyxy = native.native_xyxy(boxes, hw)
        need(pred["native_ranks"] == row["native_ranks"] == ranks and row["query_ids"] == ids
             and row["native_boxes_xyxy"] == xyxy[ids].tolist() and pred["raw_query_count"] == 164
             and pred["native_count"] == 100 and pred["unmapped_slot1_count"] == 100-len(ranks),
             "Independent joint top-100 mapping")
        pool = []
        for pos, ((rank, q), co, cand) in enumerate(zip(retained, pred["coco"], row["candidates"], strict=True), 1):
            need(co["image_id"] == num and co["category_id"] == 1 and co["segmentation"]["size"] == hw
                 and co["score"] == float(probs[q, 0]), "Native COCO score and identity")
            area = int(codec.area(co["segmentation"]))
            need(cand == dict(position=pos, native_rank=rank, query_id=q, mask_area=area, score=co["score"],
                              box_xyxy=xyxy[q].tolist()), "Exact candidate record")
            if 32 <= area < 1024:
                pool.append((rank, q, co["segmentation"]))
        need(row["logit_ranks"] == [r for r, _, _ in pool] and row["logit_query_ids"] == [q for _, q, _ in pool],
             "Eligible query/plane identity")
        values = common.read_array(a.capture, row["logits_ref"])
        need(values.shape == (len(pool), 144, 144), "Original eligible plane layout")
        overrides = {v["native_rank"]: v for v in row["posterior_overrides"]}
        need(len(overrides) == len(row["posterior_overrides"]), "Unique posterior overrides")
        for i, (rank, q, rle) in enumerate(pool):
            stock = codec.decode(rle).astype(bool)
            cpu_sign = native.rasterize(torch.from_numpy(values[i:i+1]), hw, torch)[0].numpy() > 0
            if np.array_equal(cpu_sign, stock):
                need(rank not in overrides, "Unnecessary CUDA posterior override")
                continue
            override = overrides.pop(rank)
            prob = common.read_array(a.capture, override)
            need(override["query_id"] == q and prob.shape == stock.shape and ((0 <= prob) & (prob <= 1)).all()
                 and not ((prob > 0.5) & ~stock).any() and not ((prob < 0.5) & stock).any(),
                 "CUDA posterior fallback and native signs")
        need(not overrides, "Unconsumed posterior overrides")
        native.count_rows(totals, retained, pool, row["posterior_overrides"])
    need(totals == previous["counts"] and not torch.cuda.is_initialized(), "Independent CPU readback totals")
    a.output.mkdir()
    return dict(status=VERIFIED, counts=totals, model_forward_calls=0, exact_native_parity=True,
                capture_sha256=a.capture_result_sha, capture_terminal_sha256=a.capture_terminal_sha,
                predictions_sha256=previous["predictions_sha256"], audit_sha256=previous["audit_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("capture", "verify-capture"))
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--authority-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    cfg = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and cfg.output.is_absolute()
         and cfg.output.name == cfg.phase and not cfg.output.exists(), "Fresh isolated phase output")
    a = read_authority(cfg.authority, cfg.authority_sha, cfg.phase)
    a.output = cfg.output.resolve()
    native = native_module(a.code)
    items = read_intake(a, native)
    native.install_guard(a.intake, items)
    a.carrier = a.code / "RFDETR_CENTER5202_FULL_CAPTURE_20261120" / "capture.py"
    common = native.reference(a.code / "UEIKAP_RFDETR_CARRIER_20261122" / "capture.py")
    import torch

    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
         and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
         and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4", "Qualified native runtime")
    shared = dict(images=len(items), worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_source_sha256=NATIVE, protocol_sha256=PROTOCOL, prelock_commit=a.prelock_commit,
                  intake_sha256=a.manifest_sha, intake_result_sha256=a.intake_result_sha,
                  intake_terminal_sha256=a.intake_terminal_sha, author_roster_sha256=ROSTER,
                  checkpoint_sha256=native.SOURCE_CHECKPOINT, carrier_sha256=common.CARRIER_SHA,
                  source_split="author train", evaluation_only=True, author_TEST=False, GT_opened=False,
                  AP=False, FIT=False, DEV=False, optimizer_updates=0)
    torch.set_num_threads(2 if cfg.phase == "capture" else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    if cfg.phase == "capture":
        from rfdetr.models.postprocess import PostProcess  # apply original import-time CUDA policy

        need(torch.get_float32_matmul_precision() == "high" and torch.backends.cuda.matmul.allow_tf32,
             "Historical RF-DETR CUDA matmul policy")
        result = native.capture(a, items, torch, common)
        need(not torch.backends.cudnn.allow_tf32, "Original cuDNN TF32 policy")
        result.update(status=CAPTURED, cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    else:
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only independent verification")
        result = verify(a, items, native, common, torch, shared)
    result.update(shared, input_authority_sha256=cfg.authority_sha)
    result_sha = common.write(a.output / "RESULT_SAFE.json", common.encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()
