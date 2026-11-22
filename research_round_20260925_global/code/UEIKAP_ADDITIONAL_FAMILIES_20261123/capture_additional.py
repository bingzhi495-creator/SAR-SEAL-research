"""Capture the frozen HRSID detector on the additional UEIKAP families and verify its native masks."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


PROTOCOL = "fbcc4d067351a7a7d2db226e6261a13c2f399eed59be414226e6ae29b6553bf2"
COHORT = "0d78b4611e7db88a146f5bad489b6301e64e1d1eb499d536fd414e7b134c456d"
PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
ORIGINAL = "89c3294d79e1d69530401a4bf08e3400d52fbc6b77bc7967029cbd1fbbdbb116"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
CAPTURE = "UEIKAP_ADDITIONAL_RFDETR_CAPTURE_COMPLETE"
VERIFIED = "UEIKAP_ADDITIONAL_RFDETR_CAPTURE_VERIFIED"
IMAGES, GROUPS = 342, 13


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_module(path, expected):
    source = path.read_bytes()
    require(sha256(source).hexdigest() == expected, "Published source identity differs: " + path.name)
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    exec(compile(source, str(path), "exec"), result.__dict__)
    return result


def observed(directory, path, phase, status):
    terminal_bytes = path.read_bytes()
    terminal = json.loads(terminal_bytes)
    raw = (directory / "RESULT_SAFE.json").read_bytes()
    result = json.loads(raw)
    require(type(terminal["actual_exit"]) is int and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["phase"] == phase
            and terminal["result_sha256"] == sha256(raw).hexdigest()
            and terminal["worker_sha256"] == result["worker_sha256"] and result["status"] == status,
            "Require the observed successful " + phase + " and its matching footer")
    return result, terminal["result_sha256"], sha256(terminal_bytes).hexdigest()


def image_population(args, common):
    receipt, receipt_sha, terminal_sha = observed(args.intake, args.intake_terminal, "intake",
                                                 "UEIKAP_ADDITIONAL_AUTHOR_INTAKE_COMPLETE")
    require(receipt["cohort_roster_sha256"] == COHORT and receipt["protocol_sha256"] == PROTOCOL
            and receipt["images"] == IMAGES and receipt["acquisition_groups"] == GROUPS,
            "Locked complete 342-chip intake required")
    manifest_sha = receipt["images_roster_sha256"]
    manifest = json.loads(common.pinned(args.intake / "images_roster.json", manifest_sha))
    items = manifest["items"]
    require(manifest["expected_ids"] == list(range(1, IMAGES + 1)) == [r["image_id"] for r in items]
            and len({r["acquisition_group"] for r in items}) == GROUPS, "Image roster order or acquisition census differs")
    image_root = (args.intake / "images").resolve(strict=True)
    for row in items:
        ref = row["image_ref"]
        name = f"{row['image_id']:06d}.png"
        require(row["original_hw"] == [512, 512] and ref["path"] == "images/" + name
                and (args.intake / ref["path"]).resolve(strict=True) == image_root / name
                and ref["bytes"] > 0 and len(row["rgb_sha256"]) == len(ref["sha256"]) == 64,
                "Invalid opaque native PNG reference")
    return items, dict(manifest_sha256=manifest_sha, intake_sha256=manifest_sha, intake_result_sha256=receipt_sha,
                       intake_terminal_sha256=terminal_sha, cohort_roster_sha256=COHORT)


def perform_capture(args, items, portable, common, torch):
    # Only the model/config lookup is adapted. The publisher-PNG decoder, Q164 joint
    # top-100 export, GPU strict-sign check and native candidate writer are the original UEIKAP implementation.
    from rfdetr.models.postprocess import PostProcess

    require(torch.get_float32_matmul_precision() == "high" and torch.backends.cuda.matmul.allow_tf32,
            "Original RF-DETR CUDA import precision policy")
    carrier, core, base, post = portable.detector_components(args.code, args.checkpoint)
    carrier.load_core = lambda _: core
    carrier.setup = lambda _args, _core: (base, post)
    common.exact_source = lambda _: carrier
    result = common.capture(SimpleNamespace(private_root=args.code, intake=args.intake, output=args.output), items, torch)
    require(result["model_forward_calls"] == IMAGES and result["checkpoint_sha256"] == CHECKPOINT,
            "Exact native forward count and checkpoint required")
    return result


def verify_planes(args, common, native, torch, row, prediction, eligible):
    hw = tuple(row["original_hw"])
    values = common.read_array(args.capture, row["logits_ref"])
    require(values.shape == (len(eligible), 144, 144), "Unexpected native FP32 logit grid")
    overrides = {v["native_rank"]: v for v in row["posterior_overrides"]}
    require(len(overrides) == len(row["posterior_overrides"]), "Repeated native posterior override")
    for slot, cand in enumerate(eligible):
        rank = cand["native_rank"]
        stock = codec.decode(prediction["coco"][cand["position"] - 1]["segmentation"]).astype(bool)
        cpu = native.rasterize(torch.from_numpy(values[slot:slot+1]), hw, torch)[0].numpy()
        if np.array_equal(cpu > 0, stock):
            require(rank not in overrides, "Unnecessary GPU posterior override")
            continue
        ref = overrides.pop(rank)
        posterior = common.read_array(args.capture, ref)
        require(ref["query_id"] == cand["query_id"] and posterior.shape == hw
                and np.all((0 <= posterior) & (posterior <= 1))
                and not np.any((posterior > 0.5) & ~stock) and not np.any((posterior < 0.5) & stock),
                "GPU posterior override fails to reproduce native threshold decisions")
    require(not overrides, "Unmatched GPU posterior override")


def verify(args, items, lineage, worker, portable, common, torch):
    capture, cap_sha, term_sha = observed(args.capture, args.capture_terminal, "capture", CAPTURE)
    require(all(capture[k] == v for k, v in lineage.items()) and capture["protocol_sha256"] == PROTOCOL
            and capture["worker_sha256"] == worker and capture["checkpoint_sha256"] == CHECKPOINT
            and capture["model_forward_calls"] == capture["images"] == IMAGES and capture["groups"] == GROUPS
            and capture["target_labels_opened"] is capture["GT_opened"] is False and capture["optimizer_updates"] == 0,
            "Observed capture differs from the current admitted cohort")
    native = portable.module(args.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    rows = (json.loads(line) for line in common.pinned(args.capture / common.AUDIT, capture["audit_sha256"]).splitlines())
    preds = (json.loads(line) for line in common.pinned(args.capture / common.PRED, capture["predictions_sha256"]).splitlines())
    totals = dict(mapped=0, eligible=0, rank_gaps=0, overrides=0)
    for item, row, pred in zip(items, rows, preds, strict=True):
        image_id = item["image_id"]
        _, rgb = common.decode_image(args, item)
        require(row["image_id"] == pred["image_id"] == image_id and row["source_image_name"] == f"{image_id:06d}.png"
                and row["original_hw"] == item["original_hw"] and rgb == row["rgb_sha256"] == item["rgb_sha256"]
                and row["image_sha256"] == item["image_ref"]["sha256"], "Native image and original RGB identity differ")
        probabilities = np.asarray(row["raw_query_probabilities"], dtype=np.float32)
        boxes = np.asarray(row["raw_boxes_cxcywh"], dtype=np.float32)
        require(probabilities.shape == (164, 2) and np.all((0 <= probabilities) & (probabilities <= 1))
                and boxes.shape == (164, 4) and np.isfinite(boxes).all(), "Invalid original query tensors")
        slots = np.argsort(-probabilities.ravel(), kind="stable")[:100]
        selected = [(r, int(slot) // 2) for r, slot in enumerate(slots, 1) if slot % 2 == 0]
        ranks, queries = [r for r, _ in selected], [q for _, q in selected]
        xyxy = native.native_xyxy(boxes, row["original_hw"])[queries].tolist()
        require(ranks == row["native_ranks"] == pred["native_ranks"] and queries == row["query_ids"]
                and row["native_boxes_xyxy"] == xyxy and pred["raw_query_count"] == 164 and pred["native_count"] == 100
                and pred["unmapped_slot1_count"] == 100 - len(ranks), "Original top-100 query ranks or boxes differ")
        eligible = []
        for pos, ((rank, query), cand, det, box) in enumerate(
                zip(selected, row["candidates"], pred["coco"], xyxy, strict=True), 1):
            area = int(codec.area(det["segmentation"]))
            require(det["image_id"] == image_id and det["category_id"] == 1 and det["segmentation"]["size"] == row["original_hw"]
                    and float(probabilities[query, 0]) == det["score"]
                    and cand == dict(position=pos, native_rank=rank, query_id=query, score=det["score"], mask_area=area,
                                     box_xyxy=box), "Native score, area, box or query identity differs")
            if 32 <= area < 1024:
                eligible.append(cand)
        require(row["logit_ranks"] == [c["native_rank"] for c in eligible]
                and row["logit_query_ids"] == [c["query_id"] for c in eligible], "Eligible native query order differs")
        verify_planes(args, common, native, torch, row, pred, eligible)
        native.count_rows(totals, selected, eligible, row["posterior_overrides"])
    require(totals == capture["counts"] and not torch.cuda.is_initialized(), "Independent CPU readback census differs")
    args.output.mkdir()
    return dict(counts=totals, capture_sha256=cap_sha, capture_terminal_sha256=term_sha,
                predictions_sha256=capture["predictions_sha256"], audit_sha256=capture["audit_sha256"],
                model_forward_calls=0, exact_native_parity=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("capture", "verify-capture"):
        sub = commands.add_parser(phase, allow_abbrev=False)
        for key in ("code", "intake", "intake-terminal", "protocol", "output"):
            sub.add_argument("--" + key, required=True, type=Path)
        sub.add_argument("--protocol-sha256", required=True)
        if phase == "capture":
            sub.add_argument("--checkpoint", required=True, type=Path)
        else:
            sub.add_argument("--capture", required=True, type=Path)
            sub.add_argument("--capture-terminal", required=True, type=Path)
    args = parser.parse_args()
    for k, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, v.resolve(strict=k != "output"))
    require(not args.output.exists() and sys.flags.isolated and sys.dont_write_bytecode
            and sys.version_info[:3] == (3, 10, 13), "Qualified isolated Python and fresh output required")
    require(args.protocol_sha256 == PROTOCOL == sha256(args.protocol.read_bytes()).hexdigest(), "Protocol changed")
    portable = source_module(args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", PORTABLE)
    common = source_module(args.code / "UEIKAP_RFDETR_CARRIER_20261122/capture.py", ORIGINAL)
    common.install_guard(args)
    items, lineage = image_population(args, common)
    if args.phase == "verify-capture":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    require(torch.__version__ == "2.10.0+cu130" and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and version("Pillow") == "12.3.0" and version("rfdetr") == "1.12.0", "Qualified detector runtime required")
    torch.set_num_threads(2 if args.phase == "capture" else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    worker = sha256(Path(__file__).read_bytes()).hexdigest()
    if args.phase == "capture":
        result = perform_capture(args, items, portable, common, torch)
    else:
        result = verify(args, items, lineage, worker, portable, common, torch)
    result.update(lineage, status=CAPTURE if args.phase == "capture" else VERIFIED, phase=args.phase,
                  images=IMAGES, groups=GROUPS, protocol_sha256=PROTOCOL, checkpoint_sha256=CHECKPOINT,
                  target_labels_opened=False, GT_opened=False, author_TEST=False, optimizer_updates=0, AP=False,
                  worker_sha256=worker, carrier_operator_sha256=ORIGINAL, portable_source_sha256=PORTABLE)
    output = common.write(args.output / "RESULT_SAFE.json", common.encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=output)), flush=True)


if __name__ == "__main__":
    main()
