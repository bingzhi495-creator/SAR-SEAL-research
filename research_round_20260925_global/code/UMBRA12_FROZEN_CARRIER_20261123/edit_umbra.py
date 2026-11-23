"""Prepare and independently verify the frozen, annotation-blind Umbra edit ledger."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType


PROTOCOL_NAME = "UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md"
PROTOCOL = "ff70161af804adc9f52c044771b504884ea80b7b2242ffc38fab51b55f2839ad"
MANIFEST = "24d08f3bf1ecb08212f78e5a3cfa8c709ef9a2194e9d345f5cb04a1f9f48c29e"
ROSTER = "e626c38ea53727a2d235a80e675735898bbce5defce698f04fb602d674a87a56"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
POINT_HEAD = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
NATIVE = "6549dd151178a17146de518fbe16a3a045d90c5b4ed8345d4d8e272db0e51c2d"
EDITOR = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
BASE = "be1516489123153a018ebc721806190d83f05285617e7c108e9b3c1064f66655"
CAPTURE = "UMBRA12_FROZEN_CAPTURE_COMPLETE"
CAPTURE_VERIFIED = "UMBRA12_FROZEN_CAPTURE_READBACK_COMPLETE"
PREPARED = "UMBRA12_FROZEN_EDITS_PREPARED"
VERIFIED = "UMBRA12_FROZEN_EDITS_READBACK_COMPLETE"
EDITS = "EDITS_PRIVATE.jsonl"
NATIVE_HW = (1024, 1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "Receipt digest must be SHA-256")
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Identity differs for {path.name}")
    return raw


def module(path, expected):
    source = ModuleType(path.stem)
    source.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), source.__dict__)
    return source


def observed(run, phase, status, worker_sha):
    terminal_bytes = (run / ("observed_" + phase) / "TERMINAL.json").read_bytes()
    terminal = json.loads(terminal_bytes)
    require(terminal["phase"] == phase and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["worker_sha256"] == worker_sha, f"Observed successful {phase} required")
    result_sha = terminal["result_sha256"]
    result = json.loads(pin(run / phase / "RESULT_SAFE.json", result_sha))
    require(result["phase"] == phase and result["status"] == status and result["worker_sha256"] == worker_sha
            and result["protocol_sha256"] == PROTOCOL and result["checkpoint_sha256"] == CHECKPOINT
            and result["anonymous_roster_sha256"] == ROSTER and result["intake_sha256"] == MANIFEST
            and result["images"] == 12 and result["target_labels_opened"] is False and result["optimizer_updates"] == 0,
            f"Frozen label-free {phase} lineage required")
    return result, result_sha, digest(terminal_bytes)


def capture_lineage(run, worker):
    worker_sha = digest(worker.read_bytes())
    capture, cap_sha, cap_term_sha = observed(run, "capture", CAPTURE, worker_sha)
    checked, verify_sha, verify_term_sha = observed(run, "verify-capture", CAPTURE_VERIFIED, worker_sha)
    shared = ("counts", "predictions_sha256", "audit_sha256", "intake_result_sha256", "intake_terminal_sha256",
              "intake_readback_sha256", "reviewer_receipt_sha256")
    require(checked["capture_sha256"] == cap_sha
            and checked["exact_native_parity"] is True and capture["model_forward_calls"] == 12
            and checked["model_forward_calls"] == 0 and all(capture[key] == checked[key] for key in shared),
            "Accepted CPU readback must belong to the observed native capture")
    return capture, dict(capture_result_sha256=cap_sha, capture_terminal_sha256=cap_term_sha,
                         capture_readback_result_sha256=verify_sha, capture_readback_terminal_sha256=verify_term_sha,
                         capture_worker_sha256=worker_sha, predictions_sha256=capture["predictions_sha256"],
                         audit_sha256=capture["audit_sha256"])


def fit_predictor(code, base, methods):
    fit_step = base.read_source(code / base.SPOT / "fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(code / base.RF / "fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = fit_step.load_trainer(code / base.BASE / "fit109_paired_point/train_point.py")
    return fit_step, plan, trainer


def replay_image(run_capture, image_root, ordinal, pred, audit, base, editor, methods,
                 pair, sampler, predict, *, independent):
    import numpy as np

    name = f"UMB-{ordinal:03d}"
    require(audit["image_id"] == pred["image_id"] == ordinal and audit["source_image_name"] == name + ".png"
            and audit["acquisition_group"] == name and audit["sensor"] == "Umbra"
            and audit["original_hw"] == list(NATIVE_HW), "Original admitted capture order and canvas required")
    rgb = base.load_rgb(image_root, audit)
    planes = base.load_array(run_capture, audit["logits_ref"])
    ranks = audit["logit_ranks"]
    logits = dict(zip(ranks, planes, strict=True))
    overrides = {item["native_rank"]: item for item in audit["posterior_overrides"]}
    require(len(logits) == len(ranks) and planes.shape == (len(ranks), 144, 144)
            and len(overrides) == len(audit["posterior_overrides"]) and set(overrides) <= set(logits),
            "Original eligible logits and native probability fallback required")
    features_on_grid = methods.pair.image_features(rgb) if logits else None
    counts = Counter(images=1, mapped_candidates=len(pred["coco"]), eligible_planes=len(logits))
    changes = {}
    for position, (item, candidate) in enumerate(zip(audit["candidates"], pred["coco"], strict=True), 1):
        rank, query = item["native_rank"], item["query_id"]
        stock_rle = candidate["segmentation"]
        area = int(base.codec.area(stock_rle))
        require(item["position"] == position and candidate["image_id"] == ordinal and candidate["category_id"] == 1
                and pred["native_ranks"][position-1] == rank and audit["query_ids"][position-1] == query
                and stock_rle["size"] == list(NATIVE_HW) and item["score"] == candidate["score"]
                and item["mask_area"] == area and (rank in logits) == (32 <= area < 1024),
                "Original candidate, category, rank, score and area gate required")
        if rank not in logits:
            continue
        stock = base.codec.decode(stock_rle).astype(bool)
        override = overrides.get(rank)
        require(override is None or override["query_id"] == query, "Original posterior override owner required")
        replacement = None if override is None else base.load_array(run_capture, override)
        probability = base.native_probability(logits[rank], stock, replacement)
        box = np.asarray(item["box_xyxy"], dtype=np.float32)
        require(box.shape == (4,) and np.isfinite(box).all() and np.all((0 <= box) & (box <= NATIVE_HW[0]))
                and np.all(box[:2] <= box[2:]), "Valid frozen native ROI required")
        details = None if np.any(box[:2] == box[2:]) else pair.paired_points(
            stock, box, area, candidate["score"], probability, features_on_grid, sampler)
        coords = np.empty((0, 2), dtype=np.int32) if details is None else details[0]
        if independent:
            base.check_support(methods, stock, box, probability, coords)
        if not len(coords):
            continue
        require(coords.ndim == 2 and coords.shape[1] == 2 and len(coords) <= 32
                and np.unique(coords).size == coords.size and np.all((0 <= coords) & (coords < stock.size))
                and not stock.flat[coords[:, 0]].any() and stock.flat[coords[:, 1]].all(),
                "Distinct in-bounds original addition and removal sites required")
        q = predict(details[1])
        policies = editor.policies(methods, stock, coords, q, probability, independent=independent)
        if independent:
            gain = q[:, 0] - q[:, 1]
            cap = min(8, area // 10, len(coords))
            expected_fixed = [(i, i) for i in sorted(range(len(gain)), key=lambda j: (-float(gain[j]), j))
                              if gain[i] > 0][:cap]
            require(policies["fixed"] == expected_fixed, "Independent fixed distance-pair decision differs")
        edits = {arm: base.mask_edit(stock, coords, actions) for arm, actions in policies.items() if actions}
        editor.validate_edits(stock, coords, edits)
        counts["editable_supports"] += 1
        if edits:
            changes[str(position)] = edits
        for arm, entry in edits.items():
            counts[arm + "_swaps"] += entry["swaps"]
    return dict(image_id=ordinal, changes=changes), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    for name in ("code", "intake", "protocol", "point-head", "capture-worker", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    started = perf_counter()
    independent = args.phase == "verify-edits"
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and args.output.is_absolute() and args.output.name == args.phase and not args.output.exists()
            and args.protocol.name == PROTOCOL_NAME and args.capture_worker.name == "capture_umbra.py",
            "Expected a fresh isolated CPU edit stage and exact named inputs")
    run = args.output.parent.resolve(strict=True)
    args.code = args.code.resolve(strict=True)
    args.intake = args.intake.resolve(strict=True)
    pin(args.protocol, PROTOCOL)
    capture, lineage = capture_lineage(run, args.capture_worker)
    worker_sha = digest(Path(__file__).read_bytes())
    previous = None
    if independent:
        prepared, prepared_sha, prepared_term_sha = observed(run, "prepare-edits", PREPARED, worker_sha)
        require(prepared["point_head_sha256"] == POINT_HEAD and all(prepared[key] == value for key, value in lineage.items()),
                "Same closed capture and frozen point head required for independent edit verification")
        previous = pin(run / "prepare-edits" / EDITS, prepared["edits_sha256"])
        lineage.update(prepared_result_sha256=prepared_sha, prepared_terminal_sha256=prepared_term_sha)

    native = module(args.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/capture_instance_rsdd.py", NATIVE)
    native.install_guard(args.intake, [{"source_image_name": f"UMB-{i:03d}.png"} for i in range(1, 13)])
    import torch

    require(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
            and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
            and version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4"
            and not torch.cuda.is_initialized(), "Expected the native CPU replay runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    from rfdetr.datasets.io_utils import decode_pil_image_bytes

    require(decode_pil_image_bytes is not None and torch.get_float32_matmul_precision() == "high",
            "Expected the original source occupancy inference precision")
    editor = module(args.code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", EDITOR)
    base = module(args.code / editor.BASE / "replay.py", BASE)
    methods = base.methods(args.code)
    pair, sampler = base.canvas(methods, NATIVE_HW)
    step, plan, trainer = fit_predictor(args.code, base, methods)
    heads, center, scale = plan.restore_bundle(args.point_head, POINT_HEAD, trainer, step)

    def predict(features):
        return step.predict(heads["S"], features, center, scale)

    run_capture = run / "capture"
    stocks = [json.loads(line) for line in pin(run_capture / native.PRED, capture["predictions_sha256"]).splitlines()]
    audits = [json.loads(line) for line in pin(run_capture / native.AUDIT, capture["audit_sha256"]).splitlines()]
    require(len(stocks) == len(audits) == 12, "All 12 originally captured native windows required")
    totals = Counter()
    rows = []
    for ordinal, (pred, audit) in enumerate(zip(stocks, audits, strict=True), 1):
        edits, counts = replay_image(run_capture, args.intake / "images", ordinal, pred, audit,
                                     base, editor, methods, pair, sampler, predict, independent=independent)
        rows.append(base.encoded(edits))
        totals.update(counts)
    payload = b"".join(rows)
    require(totals["images"] == 12 and totals["mapped_candidates"] == capture["counts"]["mapped"]
            and totals["eligible_planes"] == capture["counts"]["eligible"]
            and totals["fixed_swaps"] == totals["free_matched_count_swaps"]
            and totals["attraction_dynamic_swaps"] == totals["attraction_static_at_dynamic_count_swaps"]
            == totals["learned_at_attraction_count_swaps"] and not torch.cuda.is_initialized()
            and (previous is None or previous == payload), "Exact native writeback replay and original matching budgets required")
    status = VERIFIED if independent else PREPARED
    receipt = dict(status=status, phase=args.phase, worker_sha256=worker_sha, protocol_sha256=PROTOCOL,
                   checkpoint_sha256=CHECKPOINT, point_head_sha256=POINT_HEAD, anonymous_roster_sha256=ROSTER,
                   intake_sha256=MANIFEST, images=12, edits_sha256=digest(payload), exact_support_and_xor=True,
                   exact_independent_replay=independent, detector_forwards=0, optimizer_updates=0,
                   target_labels_opened=False, seconds=perf_counter() - started, **lineage)
    args.output.mkdir()
    if not independent:
        with (args.output / EDITS).open("xb") as stream:
            require(stream.write(payload) == len(payload), "Complete private ledger write required")
            stream.flush()
            os.fsync(stream.fileno())
    data = base.encoded(receipt)
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(data) == len(data), "Complete edit receipt write required")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=status, result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()
