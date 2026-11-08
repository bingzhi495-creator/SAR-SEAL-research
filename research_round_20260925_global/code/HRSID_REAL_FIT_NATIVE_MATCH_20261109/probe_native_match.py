"""Read actual native grouped matching and backward on two frozen HRSID FIT views."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_native_fit import (
    AUDIT_REPORT_SHA, DENSE_COUNT, INPUT_PACKET, PRIVATE, file_digest, file_identity,
    need, read_pinned_module, write_json,
)


STOCK = PRIVATE / "RFDETR_E24_OPTIMIZER_PRESERVATION_TRAIN_CANDIDATE_20261108_01"
STOCK_SHA = "f870227ecce5da591c1bdc9b975591fff720b36540e0f01818277105d3c2176c"
STATE_SHA = "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678"
SELECTOR_SHA = "59e6129f8f10ccfb0e83fe816f6b41c3b5cd9bdeccfae6ebb7cb719c9c4df34f"
E24_SHA = "30d6b22104122e6a38f8ec7127e30ab1ae46117d898a1131c202bc5b673530b5"
FIT_SHA = "c7629de9afaa398d0b943fba8e4cec75c8b65a9864fd795c1465c9b4dec3c271"
ARCHIVE_SHA = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
MODEL_SOURCES = {
    "rfdetr.models.lwdetr": "248faf15d673bc18f51f60bb55245f64eee40eb9c8adacea398aebdadad2243b",
    "rfdetr.models.transformer": "279c0af7e999ff4ce58845c94c87bd9211356d62cb1eeaeb626fe2680fbb8f4d",
    "rfdetr.models.criterion": "fe3a1b05acc7edc4476c03954d75e2b9c91d5612da333782aecd23d0d1cea5c3",
    "rfdetr.models.matcher": "2521d58fd5c3270f8f352c14ae299364347bd27792e18696de8bb20c637765f6",
    "rfdetr.utilities.tensors": "122f44a151db786ef4845c5d89e5cb2209624651d3638f007e3c0867f7312827",
}
GROUPS = 13
WIDTH_BASE = 100
WIDTH_LARGE = 164
INIT_SEED = 5201
EXPANSION_SEED = 5201164
PROCESS_CAP = 6 * 1024**3
ZERO_TARGET_LOSS_NAMES = ("loss_bbox", "loss_giou", "loss_mask_ce", "loss_mask_dice")


def expand_groups(model, width):
    need(width in {WIDTH_BASE, WIDTH_LARGE}
         and model.num_queries == model.transformer.num_queries == WIDTH_BASE
         and model.group_detr == model.transformer.group_detr == GROUPS, "Expected original grouped native width")
    if width == WIDTH_BASE:
        return
    generator = torch.Generator().manual_seed(EXPANSION_SEED)
    for name in ("refpoint_embed", "query_feat"):
        table = getattr(model, name)
        original = table.weight.detach()
        need(original.device.type == "cpu" and original.shape[0] == GROUPS * WIDTH_BASE,
             "Exact original CPU query table")
        dim = original.shape[1]
        old_groups = original.reshape(GROUPS, WIDTH_BASE, dim)
        shape = GROUPS, width - WIDTH_BASE, dim
        if name == "refpoint_embed":
            extra = torch.zeros(shape, dtype=original.dtype)
        else:
            extra = torch.randn(shape, dtype=original.dtype, generator=generator)
        combined = torch.cat((old_groups, extra), dim=1)
        need(torch.equal(combined[:, :WIDTH_BASE], old_groups), "Every original per-group query is byte-preserved")
        setattr(model, name, nn.Embedding.from_pretrained(
            combined.reshape(GROUPS * width, dim), freeze=not table.weight.requires_grad))
    model.num_queries = model.transformer.num_queries = width


def match_counts(indices, width, targets):
    need(len(indices) == 1, "One actual image must be matched")
    src, dst = (index.detach().cpu() for index in indices[0])
    count = min(width, targets)
    need(src.shape == dst.shape == (GROUPS * count,) and src.dtype == dst.dtype == torch.int64
         and bool(((0 <= src) & (src < GROUPS * width)).all())
         and bool(((0 <= dst) & (dst < targets)).all()) and src.unique().numel() == src.numel(),
         "Matched query and target indexes must be bounded and unique by source")
    result = []
    group_index = src // width
    for group in range(GROUPS):
        matched = dst[group_index == group]
        need(matched.numel() == matched.unique().numel() == count,
             "Native Hungarian matching must assign each group's targets uniquely up to its capacity")
        result.append(matched.numel())
    return result


def load_packet(path, expected_digest, input_protocol):
    need(path.name == INPUT_PACKET, "Expected private CPU selector packet")
    receipt_path = path.parent / "CPU_SELECTION_SAFE.json"
    need(receipt_path.stat().st_size <= 16 * 1024, "Bounded CPU receipt")
    receipt = json.loads(receipt_path.read_bytes())
    need(receipt["status"] == "TWO_REAL_HRSID_FIT_BATCHES_READY" and receipt["protocol"] == input_protocol
         and receipt["p109_sha256"] == AUDIT_REPORT_SHA and receipt["selector_sha256"] == SELECTOR_SHA
         and receipt["fit_view_sha256"] == FIT_SHA and receipt["archive_sha256"] == ARCHIVE_SHA
         and receipt["private_packet_sha256"] == expected_digest
         and receipt["FIT_only"] is True and receipt["DEV_image_opened"] is receipt["TEST_member_opened"] is False,
         "Exact CPU provenance and closed packet receipt")
    start = path.stat()
    need(0 < start.st_size == receipt["private_packet_bytes"] <= 128 * 1024**2, "Bounded CPU packet")
    with path.open("rb") as stream:
        need(file_identity(start) == file_identity(os.fstat(stream.fileno())), "Exact CPU packet handle")
        digest = sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        need(digest.hexdigest() == expected_digest, "Exact packet digest")
        stream.seek(0)
        packet = torch.load(stream, weights_only=True, map_location="cpu")
        need(file_identity(start) == file_identity(os.fstat(stream.fileno())) == file_identity(path.stat()),
             "CPU packet unchanged during readback")
    need(packet["audit_sha256"] == AUDIT_REPORT_SHA and packet["fit_view_sha256"] == FIT_SHA
         and packet["original_archive_sha256"] == ARCHIVE_SHA and set(packet["cases"]) == {"empty", "dense"},
         "Exact input identity and the two predetermined roles")
    return packet["cases"]


def inspect_case(case, n, width, device, model, criterion):
    from rfdetr.utilities.tensors import NestedTensor

    pixels = case["sample"]["pixels"]
    target = case["sample"]["target"]
    need(case["after"] == {"labels": n, "empty_masks": 0}
         and pixels.shape == (1, 3, 576, 576) and pixels.dtype == torch.float32
         and bool(torch.isfinite(pixels).all()) and target["masks"].shape == (n, 576, 576)
         and target["masks"].dtype == torch.bool and bool(target["masks"].flatten(1).any(1).all())
         and target["boxes"].shape == (n, 4) and bool(torch.isfinite(target["boxes"]).all())
         and target["labels"].shape == (n,) and target["labels"].dtype == torch.int64
         and bool((target["labels"] == 0).all()), "Actual post-augmentation pixel and target readback")
    torch.manual_seed(case["seed"])
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    samples = NestedTensor(pixels.to(device), torch.zeros((1, 576, 576), dtype=torch.bool, device=device),
                           no_padding=True)
    targets = [{key: value.to(device) for key, value in target.items()}]
    original_match = criterion.matcher.forward
    assignments = []

    def observe(layer, current_targets, **kwargs):
        indices = original_match(layer, current_targets, **kwargs)
        assignments.append((layer["pred_logits"].shape, kwargs["group_detr"], indices))
        return indices

    # The native class method remains unchanged, so the criterion's fast-path test is unaffected.
    # Single-image segmentation uses the native per-layer path and must produce one trace per head.
    with torch.autocast("cuda", dtype=torch.bfloat16), patch.object(criterion.matcher, "forward", observe):
        outputs = model(samples, targets)
        losses = criterion(outputs, targets)
        loss = torch.stack([value * criterion.weight_dict[key] for key, value in losses.items()
                            if key in criterion.weight_dict]).sum()
    suffixes = [""] + [f"_{i}" for i in range(len(outputs["aux_outputs"]))] + ["_enc"]
    need("enc_outputs" in outputs and len(suffixes) == len(assignments) and len(suffixes) >= 3,
         "Read actual final, all auxiliary and encoder assignments on the native per-layer path")
    per_head = {}
    for suffix, (shape, groups, indices) in zip(suffixes, assignments, strict=True):
        need(shape[:2] == (1, width * GROUPS) and groups == GROUPS, "Every native head uses the declared group width")
        matches = match_counts(indices, width, n)
        per_head[suffix.removeprefix("_") or "final"] = dict(
            queries=int(shape[1]), matches_per_group=matches, unassigned_per_group=[n - m for m in matches])
    values = {key: float(value.detach().float()) for key, value in losses.items()}
    need(bool(torch.stack([torch.isfinite(value) for value in losses.values()]).all()) and bool(torch.isfinite(loss))
         and all(f"loss_ce{s}" in values for s in suffixes), "Finite native classification and all loss terms")
    if n == 0:
        need(all(values[f"{name}{s}"] == 0.0 for s in suffixes for name in ZERO_TARGET_LOSS_NAMES)
             and all(values[f"loss_ce{s}"] > 0 for s in suffixes), "Empty view has zero box/mask and positive negative-class loss")
    loss.backward()
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    need(bool(grads), "Native backward produced gradients")
    finite = torch.stack([torch.isfinite(grad).all() for grad in grads]).all()
    max_grad = torch.stack([grad.abs().amax() for grad in grads]).amax()
    need(bool(finite) and float(max_grad) > 0, "All observed gradients are finite and at least one is nonzero")
    torch.cuda.synchronize(device)
    return dict(targets=n, heads=per_head, losses=values, weighted_loss=float(loss.detach().float()),
                gradient_tensor_count=len(grads), finite_gradients=True, max_absolute_gradient=float(max_grad),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--input-protocol", required=True)
    parser.add_argument("--width", type=int, choices=(WIDTH_BASE, WIDTH_LARGE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(all(re.fullmatch(r"[0-9a-f]{40}", commit) for commit in (args.protocol, args.input_protocol))
         and bool(re.fullmatch(r"[0-9a-f]{64}", args.input_sha256)) and not args.output.exists(),
         "Committed GPU and CPU protocols, exact input digest and fresh private output")
    need(tuple(sys.version_info[:3]) == (3, 10, 13) and version("rfdetr") == "1.12.0"
         and torch.__version__ == "2.10.0+cu130" and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact native offline isolated Windows GPU runtime")
    need(file_digest(STOCK / "state_preservation.py") == STATE_SHA, "Exact closed E24 state helper")
    stock = read_pinned_module(STOCK / "run_state_arm.py", STOCK_SHA, "_closed_e24_initializer")
    sys.addaudithook(stock.offline)
    from importlib import import_module
    need(all(file_digest(Path(import_module(name).__file__)) == digest for name, digest in MODEL_SOURCES.items()),
         "Exact native matching, loss and model code")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
         "One visible BF16 GPU")
    free, total = torch.cuda.mem_get_info(device)
    need(free >= PROCESS_CAP, "GPU memory floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, PROCESS_CAP / total), device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(INIT_SEED)
    started = perf_counter()
    cases = load_packet(args.input, args.input_sha256, input_protocol=args.input_protocol)
    _, model, criterion, optimizer, ema, _, initialization = stock.initialize(
        "F", torch.device("cpu"), stock.p77_source())
    need(initialization["archive_start_sha256"] == E24_SHA, "Exact independently closed original E24 archive")
    del optimizer, ema
    expand_groups(model, args.width)
    state_before = stock.tensor_dict_sha256(model.state_dict())
    if args.width == WIDTH_BASE:
        need(state_before == initialization["initial_online_sha256"], "Baseline is the exact unmodified E24 state")
    args.output.mkdir()
    model.to(device).train()
    criterion.to(device).train()
    results = {}
    for role, n in (("empty", 0), ("dense", DENSE_COUNT)):
        results[role] = inspect_case(cases[role], n, args.width, device, model, criterion)
        need(stock.tensor_dict_sha256(model.state_dict()) == state_before, "All online parameters and buffers unchanged")
        print(json.dumps(dict(event="HRSID_FIT_NATIVE_BACKWARD_CHECKED", role=role, width=args.width,
                              weighted_loss=results[role]["weighted_loss"],
                              peak_allocated_bytes=results[role]["peak_allocated_bytes"]), allow_nan=False), flush=True)
    result = dict(status="TWO_REAL_HRSID_FIT_NATIVE_NO_UPDATE_BACKWARDS_COMPLETE", protocol=args.protocol,
                  input_protocol=args.input_protocol, p109_sha256=AUDIT_REPORT_SHA,
                  input_sha256=args.input_sha256, fit_view_sha256=FIT_SHA,
                  initializer_archive_sha256=E24_SHA, initializer_online_sha256=initialization["initial_online_sha256"],
                  state_unchanged_sha256=state_before, query_width=args.width, query_groups=GROUPS,
                  new_queries_per_group=args.width - WIDTH_BASE, precision="BF16 autocast", image_batch_size=1,
                  allocator_limit_bytes=PROCESS_CAP, result=results, elapsed_seconds=perf_counter() - started,
                  FIT_only=True, DEV_image_opened=False, TEST_member_opened=False, optimizer_updates=0,
                  ema_updates=0, AP=False, training_qualification=False)
    write_json(args.output / "NATIVE_MATCH_BACKWARD_SAFE.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
