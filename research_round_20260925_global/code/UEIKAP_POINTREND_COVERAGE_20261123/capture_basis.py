"""Recapture projected mask features against the closed 342-image UEIKAP carrier."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


ADDITIONAL = "a4f6942094f8f722db085d7de88c8627cae89c3c1d1173837a37b46f2054d390"
PROTOCOL = "e0cad345351d1cfc23b6777d84abfeffd9fc7e2c82d69242813356af33715225"
BASIS = "476a8ff4a9c705df3fa6530fca2c4f58d020dc7995b444480476eeac1e98b8a1"
NATIVE_CAPTURE = "d7f99ff75782172fcfd263c8af43293b375b58b4ee2a2129139f892fc38ab5b9"
NATIVE_CAPTURE_TERMINAL = "ad5ba3d2320699df213a83aa1128f39d89a71bcf3cc13d56c538e1c945f51a9e"
NATIVE_READBACK = "1f7790e9858a19fadafc495b38592e93858fc00626bd92bcca0e548ed3b49ff6"
NATIVE_READBACK_TERMINAL = "8462bfc621fea88412ccee213e9c2c80e049dca0ce8fc67717b61d56f21e45e9"
COUNTS = dict(mapped=33975, eligible=17537, overrides=0, rank_gaps=225)
RUNTIME = {"torch": "2.10.0+cu130", "rfdetr": "1.12.0", "Pillow": "12.3.0",
           "pycocotools": "2.0.11", "numpy": "1.26.4"}
STATUS = "UEIKAP_POINTREND_COVERAGE_BASIS_COMPLETE"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_source(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Published source identity differs for " + path.name)
    mod = ModuleType("_ueikap_basis_" + path.stem)
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def native_inputs(args, additional, common):
    root = args.native_root
    paths = SimpleNamespace(intake=root / "intake", intake_terminal=root / "observed_intake/TERMINAL.json",
                            capture=root / "capture")
    items, intake = additional.image_population(paths, common)
    cap, cap_sha, cap_term = additional.observed(
        paths.capture, root / "observed_capture_02/TERMINAL.json", "capture", additional.CAPTURE)
    verified, verify_sha, verify_term = additional.observed(
        root / "verify-capture", root / "observed_verify-capture/TERMINAL.json", "verify-capture", additional.VERIFIED)
    require((cap_sha, cap_term, verify_sha, verify_term) ==
            (NATIVE_CAPTURE, NATIVE_CAPTURE_TERMINAL, NATIVE_READBACK, NATIVE_READBACK_TERMINAL)
            and verified["capture_sha256"] == cap_sha and verified["capture_terminal_sha256"] == cap_term
            and verified["exact_native_parity"] is True, "Original observed capture and independent readback required")
    for receipt, forwards in ((cap, 342), (verified, 0)):
        require(all(receipt[k] == v for k, v in intake.items())
                and receipt["worker_sha256"] == ADDITIONAL and receipt["checkpoint_sha256"] == additional.CHECKPOINT
                and receipt["protocol_sha256"] == additional.PROTOCOL and receipt["counts"] == COUNTS
                and receipt["images"] == len(items) == 342 and receipt["groups"] == 13
                and receipt["model_forward_calls"] == forwards and receipt["optimizer_updates"] == 0
                and receipt["GT_opened"] is receipt["target_labels_opened"] is False,
                "Original image cohort, native stream census or provenance differs")
    lineage = dict(**intake, original_capture_sha256=cap_sha, original_capture_terminal_sha256=cap_term,
                   verified_capture_sha256=verify_sha, verified_capture_terminal_sha256=verify_term,
                   predictions_sha256=cap["predictions_sha256"], audit_sha256=cap["audit_sha256"])
    require(all(verified[k] == lineage[k] for k in ("predictions_sha256", "audit_sha256")),
            "Independent readback must bind the same native streams")
    # Verify each complete stream before consuming it. Parse one image's rows at a time.
    audit = map(json.loads, common.pinned(paths.capture / common.AUDIT, lineage["audit_sha256"]).splitlines())
    predictions = map(json.loads, common.pinned(paths.capture / common.PRED, lineage["predictions_sha256"]).splitlines())
    return paths, items, zip(audit, predictions, strict=True), lineage


def check_candidates(row, pred, np, codec):
    probabilities = np.asarray(row["raw_query_probabilities"], dtype=np.float32)
    order = np.argsort(-probabilities.ravel(), kind="stable")[:100]
    selected = [(rank, int(slot) // 2) for rank, slot in enumerate(order, 1) if slot % 2 == 0]
    require([r for r, _ in selected] == row["native_ranks"] == pred["native_ranks"]
            and [q for _, q in selected] == row["query_ids"]
            and pred["raw_query_count"] == 164 and pred["native_count"] == 100
            and pred["unmapped_slot1_count"] == 100 - len(selected), "Native top-100 carrier mapping differs")
    eligible = []
    for pos, (pair, box, cand, det) in enumerate(
            zip(selected, row["native_boxes_xyxy"], row["candidates"], pred["coco"], strict=True), 1):
        rank, query_id = pair
        area = int(codec.area(det["segmentation"]))
        require(det["image_id"] == row["image_id"] and det["category_id"] == 1
                and det["segmentation"]["size"] == [512, 512] and det["score"] == float(probabilities[query_id, 0])
                and cand == dict(position=pos, native_rank=rank, query_id=query_id, score=det["score"],
                                 mask_area=area, box_xyxy=box), "Native compact candidate metadata differs")
        if 32 <= area < 1024:
            eligible.append((rank, query_id))
    require([r for r, _ in eligible] == row["logit_ranks"] and [q for _, q in eligible] == row["logit_query_ids"]
            and row["posterior_overrides"] == [], "Native area eligibility or posterior override differs")
    return len(selected), len(eligible)


def run(args):
    additional = load_source(args.code / "UEIKAP_ADDITIONAL_FAMILIES_20261123/capture_additional.py", ADDITIONAL)
    common = load_source(args.code / "UEIKAP_RFDETR_CARRIER_20261122/capture.py", additional.ORIGINAL)
    portable = load_source(args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", additional.PORTABLE)
    original = load_source(args.code / "INSTANCE_RSDD_POINTREND_TRANSFER_20261122/capture_basis.py", BASIS)
    common.install_guard(args)
    paths, items, pairs, lineage = native_inputs(args, additional, common)

    import numpy as np
    import torch
    from pycocotools import mask as codec

    # Original UEIKAP342 initialization order: RF-DETR changes the matmul policy on import.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    initial_precision = torch.get_float32_matmul_precision()
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    post_import_precision = torch.get_float32_matmul_precision()
    require(initial_precision == "highest" and post_import_precision == "high" and torch.backends.cuda.matmul.allow_tf32,
            "Original UEIKAP342 RF-DETR import policy required")
    carrier, core, base, post = portable.detector_components(args.code, args.checkpoint)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA device required")
    free, total = torch.cuda.mem_get_info()
    require(free >= base.CAP, "Native GPU resource floor")
    torch.cuda.set_per_process_memory_fraction(min(0.35, base.CAP / total))
    torch.backends.cudnn.allow_tf32 = False
    model_args = SimpleNamespace(root=args.code, output=args.output, mode="dev")
    model, transform, config = carrier.model_and_transform(model_args, core, base, torch)
    require(not model.training and model.num_queries == 164 and model.group_detr == 13, "Frozen Q164 detector")
    native = portable.module(args.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    geometry = SimpleNamespace(native_block=carrier.native_block, native_xyxy=native.native_xyxy)
    forward_precision = torch.get_float32_matmul_precision()
    forward_tf32 = torch.backends.cuda.matmul.allow_tf32
    hooks = original.module(args.code / "HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py", original.HOOK_SHA)
    args.output.mkdir()
    (args.output / "spatial").mkdir()
    counts = dict.fromkeys(COUNTS, 0)
    timings = dict(decode_transform_prepare=0.0, native_forward_with_hook=0.0, basis_verify_and_transfer=0.0,
                   native_comparison=0.0, basis_write=0.0)
    index = []
    with torch.inference_mode():
        for ordinal, (item, (row, pred)) in enumerate(zip(items, pairs, strict=True), 1):
            start = perf_counter()
            image, rgb = common.decode_image(paths, item)
            require(ordinal == item["image_id"] == row["image_id"] == pred["image_id"]
                    and row["source_image_name"] == f"{ordinal:06d}.png"
                    and item["original_hw"] == row["original_hw"] == [512, 512]
                    and item["image_ref"]["sha256"] == row["image_sha256"]
                    and item["rgb_sha256"] == row["rgb_sha256"] == rgb, "Original PNG, RGB or dimensions differ")
            tensor, target = transform(image, None)
            require(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                    "Original unpadded val576 tensor required")
            batch = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            require(batch.mask is not None and not bool(batch.mask.any()), "Unexpected transform padding")
            batch = batch.to("cuda")
            torch.cuda.synchronize()
            tick = perf_counter()
            timings["decode_transform_prepare"] += tick - start
            require(torch.get_float32_matmul_precision() == forward_precision and
                    torch.backends.cuda.matmul.allow_tf32 == forward_tf32 and not torch.backends.cudnn.allow_tf32,
                    "Numerical policy changed before native forward")
            with hooks.capture_final_query(model.segmentation_head) as observation:
                raw = model(batch)
            torch.cuda.synchronize()
            end = perf_counter()
            timings["native_forward_with_hook"] += end - tick
            features = observation.seal(raw["pred_masks"]).spatial[0].contiguous().cpu().numpy()
            require(raw["pred_masks"].shape == (1, 164, 144, 144) and features.shape == (256, 144, 144)
                    and features.dtype == np.float32 and np.isfinite(features).all(), "Expected native finite FP32 basis")
            tick = perf_counter()
            timings["basis_verify_and_transfer"] += tick - end
            planes = common.read_array(paths.capture, row["logits_ref"])
            original.compare_carrier(raw, geometry, post, row, pred, planes, (512, 512), np, torch, codec, PostProcess)
            mapped, eligible = check_candidates(row, pred, np, codec)
            # In addition to the original numerical comparison, preserve signed zero in archived plane bytes.
            ids = torch.as_tensor(row["logit_query_ids"], dtype=torch.long, device=raw["pred_masks"].device)
            gathered = raw["pred_masks"][0].index_select(0, ids).contiguous().cpu().numpy()
            require(gathered.shape == planes.shape == (eligible, 144, 144)
                    and np.array_equal(gathered.view(np.uint32), planes.view(np.uint32)), "Native coarse plane bytes differ")
            counts["mapped"] += mapped
            counts["eligible"] += eligible
            counts["rank_gaps"] += 100 - mapped
            end = perf_counter()
            timings["native_comparison"] += end - tick
            ref = common.write_array(args.output, f"spatial/{ordinal:06d}.npy", features)
            ref.update(shape=list(features.shape), dtype=str(features.dtype))
            index.append(dict(ordinal=ordinal, image_id=ordinal, rgb_sha256=rgb, eligible_rows=eligible, spatial=ref))
            timings["basis_write"] += perf_counter() - end
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == len(items):
                print(json.dumps(dict(images=ordinal, total=len(items), eligible=counts["eligible"])), flush=True)
    require(len(index) == 342 and counts == COUNTS, "Recapture differs from the independently closed native census")
    index_sha = common.write(args.output / "INDEX_PRIVATE.json", common.encoded(index))
    return dict(**lineage, status=STATUS, phase="basis", images=len(index), groups=13, counts=counts,
                index_sha256=index_sha, spatial_bytes=sum(e["spatial"]["bytes"] for e in index),
                phase_seconds=timings, model_forwards=len(index), model_forward_calls=len(index), optimizer_updates=0,
                checked_all_query_planes=len(index) * 164, exact_original_candidates_and_masks=True,
                exact_original_eligible_logits=True, exact_all_query_basis_contraction=True,
                checkpoint_sha256=additional.CHECKPOINT, capture_receipt_sha256=NATIVE_CAPTURE,
                candidate_audit_sha256=lineage["audit_sha256"], original_basis_worker_sha256=BASIS,
                original_capture_worker_sha256=ADDITIONAL, final_query_hook_sha256=original.HOOK_SHA,
                operator_versions=RUNTIME, GT_opened=False, target_labels_opened=False, AP=False,
                initial_matmul_precision=initial_precision, post_import_matmul_precision=post_import_precision,
                matmul_precision=forward_precision, cuda_matmul_allow_tf32=forward_tf32,
                cudnn_allow_tf32=False, cuda_peak_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "native-root", "checkpoint", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve(strict=key != "output"))
    require(not args.output.exists() and not args.output.is_relative_to(args.native_root)
            and sys.flags.isolated and sys.dont_write_bytecode and sys.version_info[:3] == (3, 10, 13)
            and {k: version(k) for k in RUNTIME} == RUNTIME, "Fresh isolated output and original runtime required")
    require(args.protocol_sha256 == PROTOCOL == sha256(args.protocol.read_bytes()).hexdigest(),
            "Expected published UEIKAP PointRend coverage protocol")
    start = perf_counter()
    result = run(args)
    result.update(seconds=perf_counter()-start, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=args.protocol_sha256)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw) == len(raw), "Complete safe receipt required")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
