"""Run or independently read back the frozen RF-DETR and editor on admitted supplemental Umbra windows."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


HERE = Path(__file__).resolve().parent
HW = (1024, 1024)
ORIGINAL = "UMBRA12_FROZEN_CARRIER_20261123"
CAPTURE_SOURCE = "a67678cd0c9666a8c432bcacddcc9f17f18a7ce1ba98426ffe6f9fa47d59cc38"
EDIT_SOURCE = "43387b43ded0e01dfd04870640142db7582f14676a3ccba3d9953ac2c086b517"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
POINT_HEAD = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
PHASES = {
    "capture": "UMBRA_SUPPLEMENT_FROZEN_CAPTURE_COMPLETE",
    "verify-capture": "UMBRA_SUPPLEMENT_FROZEN_CAPTURE_READBACK_COMPLETE",
    "prepare-edits": "UMBRA_SUPPLEMENT_FROZEN_EDITS_PREPARED",
    "verify-edits": "UMBRA_SUPPLEMENT_FROZEN_EDITS_READBACK_COMPLETE",
}
RECEIPTS = {
    "intake_result": "intake/RESULT_SAFE.json",
    "intake_terminal": "observed_intake/TERMINAL.json",
    "readback_result": "readback/RESULT_SAFE.json",
    "readback_terminal": "observed_readback/TERMINAL.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def pin(path, expected):
    require(valid_sha(expected), "SHA-256 binding required")
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Digest mismatch: {path.name}")
    return raw


def module(path, expected):
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), vars(result))
    return result


def roster_image_ids(roster, requested):
    require(type(requested) is int and 18 <= requested <= 24
            and roster.get("schema") == "sar-hull-roster-v1"
            and set(roster) == {"schema", "images"}
            and isinstance(roster["images"], list)
            and 18 <= len(roster["images"]) <= requested, "Admitted supplemental roster size and schema")
    ids = []
    digests = set()
    for row in roster["images"]:
        require(set(row) == {"image_id", "height", "width", "png_sha256"}
                and isinstance(row["image_id"], str) and re.fullmatch(r"UMB-1[0-2][0-9]", row["image_id"])
                and (row["height"], row["width"]) == HW and valid_sha(row["png_sha256"]),
                "Canonical anonymous supplemental PNG identifier and canvas")
        ids.append(int(row["image_id"][4:]))
        digests.add(row["png_sha256"])
    require(ids == sorted(set(ids)) and 101 <= ids[0] and ids[-1] < 101 + requested and len(digests) == len(ids),
            "Exact unique order and prior slot identity are required")
    return ids


def validate_native_lineage(intake, intake_term, readback, readback_term, bindings, n):
    requested = intake["requested_windows"]
    failed = requested - n
    partial = failed != 0
    expected_status = "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_PARTIAL" if partial else "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_COMPLETE"
    expected_exit = int(partial)
    require(intake_term["phase"] == "native-window-intake"
            and intake_term["actual_exit"] == expected_exit
            and intake_term["result_sha256"] == bindings["intake_result"]
            and intake_term["output_footer_verified"] is True
            and intake_term["worker_sha256"] == intake["worker_sha256"]
            and intake["status"] == expected_status
            and intake["native_window_shape_yx"] == list(HW)
            and intake["failed_window_count"] == failed
            and intake["verified_native_windows"] == intake["decoded_unique_windows"] == n
            and intake["annotation_files"] == intake["detector_forwards"] == intake["optimizer_updates"] == 0,
            "Observed native intake with exact withheld-window accounting")
    require(readback_term["phase"] == "native-window-readback"
            and readback_term["actual_exit"] == 0 and readback_term["output_footer_verified"] is True
            and readback_term["result_sha256"] == bindings["readback_result"]
            and readback_term["worker_sha256"] == readback["worker_sha256"]
            and readback["status"] == "UMBRA_SUPPLEMENT_NATIVE_READBACK_COMPLETE"
            and readback["eligible_for_detector"] is True
            and readback["extraction_result_sha256"] == bindings["intake_result"]
            and readback["extraction_terminal_sha256"] == bindings["intake_terminal"]
            and readback["extraction_actual_exit"] == expected_exit
            and readback["manifest_sha256"] == intake["manifest_sha256"] == bindings["manifest_sha256"]
            and readback["metadata_result_sha256"] == intake["metadata_result_sha256"] == bindings["metadata_result_sha256"]
            and readback["metadata_terminal_sha256"] == intake["metadata_terminal_sha256"] == bindings["metadata_terminal_sha256"]
            and readback["requested_windows"] == requested
            and readback["failed_window_count"] == failed
            and readback["native_window_shape_yx"] == list(HW)
            and readback["readback_windows"] == readback["unique_pixel_arrays"] == n
            and readback["matching_uncompressed_ranges"] == 4 * n,
            "Observed independent native pixel and source-range readback")
    return requested


def load_admission(args, bindings):
    native = {key: json.loads(pin(args.intake_root / path, bindings[key])) for key, path in RECEIPTS.items()}
    packet = json.loads(pin(args.packet_result, bindings["packet_result"]))
    terminal = json.loads(pin(args.packet_terminal, bindings["packet_terminal"]))
    roster = json.loads(pin(args.roster, bindings["anonymous_roster"]))
    ids = roster_image_ids(roster, native["intake_result"]["requested_windows"])
    validate_native_lineage(native["intake_result"], native["intake_terminal"], native["readback_result"],
                            native["readback_terminal"], bindings, len(ids))
    require(terminal["phase"] == packet["phase"] == "packet-build"
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == bindings["packet_result"] and terminal["worker_sha256"] == packet["worker_sha256"]
            and packet["status"] == "TWO_BLIND_SUPPLEMENT_REVIEW_PACKETS_VERIFIED"
            and packet["intake_result_sha256"] == bindings["intake_result"]
            and packet["readback_result_sha256"] == bindings["readback_result"]
            and packet["anonymous_roster_sha256"] == bindings["anonymous_roster"]
            and packet["frozen_manifest_sha256"] == bindings["manifest_sha256"]
            and packet["intake_terminal_sha256"] == bindings["intake_terminal"]
            and packet["readback_terminal_sha256"] == bindings["readback_terminal"]
            and packet["requested_windows"] == native["intake_result"]["requested_windows"]
            and packet["windows_per_packet"] == len(ids)
            and packet["unavailable_windows"] == packet["requested_windows"] - len(ids)
            and packet["labels_present"] == packet["model_outputs_present"] == packet["source_coordinates_present"] == 0,
            "Exact roster shared by both prediction-blind review packets")
    return roster, ids


def load_items(intake, roster, ids):
    from PIL import Image

    image_root = (intake / "images").resolve(strict=True)
    require(image_root == intake / "images", "Exact admitted image directory")
    items = []
    rgb_digests = set()
    for row, numeric in zip(roster["images"], ids, strict=True):
        name = row["image_id"] + ".png"
        path = image_root / name
        require(path.resolve(strict=True) == path, "Exact admitted image file")
        raw = pin(path, row["png_sha256"])
        with Image.open(BytesIO(raw)) as image:
            image.load()
            require(image.format == "PNG" and image.mode == "L" and image.size == HW and not image.info,
                    "Original native metadata-free 8-bit grayscale PNG")
            rgb_sha = digest(image.convert("RGB").tobytes())
        rgb_digests.add(rgb_sha)
        items.append(dict(image_id=numeric, original_hw=list(HW), sensor="Umbra",
                          acquisition_group=row["image_id"], source_image_name=name, rgb_sha256=rgb_sha,
                          image_ref=dict(path="images/" + name, sha256=row["png_sha256"], bytes=len(raw))))
    require(len(rgb_digests) == len(items), "No repeated decoded native windows")
    return items


def capture_modules(code, old):
    operator = module(code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", old.PORTABLE)
    native = operator.module(code / operator.NATIVE / "capture_instance_rsdd.py", operator.CAPTURE_SOURCE)
    common = native.reference(code / operator.UEIKAP / "capture.py")
    native.decode_native = old.decode_native
    native.CAPTURE_STATUS = PHASES["capture"]
    native.VERIFY_STATUS = PHASES["verify-capture"]
    native.__file__ = str(Path(__file__).resolve(strict=True))
    return operator, native, common


def runtime(capture, edit=False):
    import torch

    require(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
            and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
            and version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4"
            and (capture or not torch.cuda.is_initialized()), "Original accepted frozen RF-DETR runtime")
    torch.set_num_threads(2 if capture else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    if edit:
        from rfdetr.datasets.io_utils import decode_pil_image_bytes

        require(decode_pil_image_bytes is not None and torch.get_float32_matmul_precision() == "high",
                "Original frozen occupancy inference precision")
    return torch


def observed(run, phase, worker_sha, bound):
    raw = (run / ("observed_" + phase) / "TERMINAL.json").read_bytes()
    terminal = json.loads(raw)
    require(terminal["phase"] == phase and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["worker_sha256"] == worker_sha, f"Observed successful stage required: {phase}")
    result_sha = terminal["result_sha256"]
    receipt = json.loads(pin(run / phase / "RESULT_SAFE.json", result_sha))
    require(receipt["phase"] == phase and receipt["status"] == PHASES[phase]
            and receipt["worker_sha256"] == worker_sha and all(receipt[key] == value for key, value in bound.items())
            and receipt["target_labels_opened"] is False and receipt["optimizer_updates"] == 0,
            "Exact closed supplemental cohort and science lineage")
    return receipt, result_sha, digest(raw)


def model_stage(args, items, adapter, bound, worker):
    operator, native, common = adapter
    capture = args.phase == "capture"
    torch = runtime(capture)
    intake = args.intake_root / "intake"
    if capture:
        from rfdetr.models.postprocess import PostProcess

        require(PostProcess is not None and torch.get_float32_matmul_precision() == "high"
                and torch.backends.cuda.matmul.allow_tf32, "Original detector precision")
        carrier, core, base, post = operator.detector_components(args.code, args.checkpoint.resolve(strict=True))
        carrier.load_core = lambda _: core
        carrier.setup = lambda _args, _core: (base, post)
        common.exact_source = lambda _: carrier
        request = SimpleNamespace(intake=intake, output=args.output, private_root=args.code)
        result = native.capture(request, items, torch, common)
        require(result["model_forward_calls"] == len(items) and torch.backends.cudnn.allow_tf32 is False,
                "Original detector and exact supplemental forward count")
    else:
        original, original_sha, _ = observed(args.output.parent, "capture", worker, bound)
        request = SimpleNamespace(intake=intake, output=args.output, protocol_sha=bound["protocol_sha256"],
                                  manifest_sha=bound["intake_sha256"])
        result = native.verify(request, items, torch, common)
        require(result["capture_sha256"] == original_sha and result["model_forward_calls"] == 0
                and result["exact_native_parity"] is True
                and all(result[key] == original[key] for key in ("predictions_sha256", "audit_sha256", "counts")),
                "Independent readback reproduces exact supplemental capture")
    return result


def edit_stage(args, items, adapter, bound, worker):
    _, native, _ = adapter
    run = args.output.parent
    capture, capture_sha, capture_terminal = observed(run, "capture", worker, bound)
    verified, verify_sha, verify_terminal = observed(run, "verify-capture", worker, bound)
    require(verified["capture_sha256"] == capture_sha and verified["exact_native_parity"] is True
            and capture["model_forward_calls"] == len(items) and verified["model_forward_calls"] == 0
            and all(capture[key] == verified[key] for key in ("counts", "predictions_sha256", "audit_sha256")),
            "Same capture and CPU readback before frozen action replay")
    lineage = dict(capture_result_sha256=capture_sha, capture_terminal_sha256=capture_terminal,
                   capture_readback_result_sha256=verify_sha, capture_readback_terminal_sha256=verify_terminal,
                   predictions_sha256=capture["predictions_sha256"], audit_sha256=capture["audit_sha256"],
                   point_head_sha256=POINT_HEAD)
    independent = args.phase == "verify-edits"
    old = module(args.code / ORIGINAL / "edit_umbra.py", EDIT_SOURCE)
    previous = None
    if independent:
        prepared, prepared_sha, prepared_terminal = observed(run, "prepare-edits", worker, bound)
        require(all(prepared[key] == value for key, value in lineage.items()) and prepared["detector_forwards"] == 0,
                "Same frozen point head and capture for independent edit verification")
        previous = pin(run / "prepare-edits" / old.EDITS, prepared["edits_sha256"])
        lineage.update(prepared_result_sha256=prepared_sha, prepared_terminal_sha256=prepared_terminal)
    torch = runtime(False, edit=True)
    editor = old.module(args.code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", old.EDITOR)
    base = old.module(args.code / editor.BASE / "replay.py", old.BASE)
    methods = base.methods(args.code)
    pair, sampler = base.canvas(methods, HW)
    step, plan, trainer = old.fit_predictor(args.code, base, methods)
    heads, center, scale = plan.restore_bundle(args.point_head, POINT_HEAD, trainer, step)

    def predict(features):
        return step.predict(heads["S"], features, center, scale)

    run_capture = run / "capture"
    stocks = [json.loads(line) for line in pin(run_capture / native.PRED, lineage["predictions_sha256"]).splitlines()]
    audits = [json.loads(line) for line in pin(run_capture / native.AUDIT, lineage["audit_sha256"]).splitlines()]
    totals = Counter()
    rows = []
    for item, pred, audit in zip(items, stocks, audits, strict=True):
        require(all(audit[key] == item[key] for key in ("source_image_name", "image_id", "rgb_sha256")),
                "Same anonymous source bytes for action replay")
        row, counts = old.replay_image(run_capture, args.intake_root / "intake" / "images", item["image_id"],
                                       pred, audit, base, editor, methods, pair, sampler, predict,
                                       independent=independent)
        rows.append(base.encoded(row))
        totals.update(counts)
    payload = b"".join(rows)
    require(totals["images"] == len(items) and totals["mapped_candidates"] == capture["counts"]["mapped"]
            and totals["eligible_planes"] == capture["counts"]["eligible"]
            and totals["fixed_swaps"] == totals["free_matched_count_swaps"]
            and totals["attraction_dynamic_swaps"] == totals["attraction_static_at_dynamic_count_swaps"]
            == totals["learned_at_attraction_count_swaps"] and not torch.cuda.is_initialized()
            and (previous is None or previous == payload), "Exact original signed budgets and independently replayed masks")
    args.output.mkdir()
    if not independent:
        write(args.output / old.EDITS, payload)
    return dict(status=PHASES[args.phase], edits_sha256=digest(payload), exact_support_and_xor=True,
                exact_independent_replay=independent, detector_forwards=0, **lineage)


def write(path, payload):
    with path.open("xb") as handle:
        require(handle.write(payload) == len(payload), "Complete exclusive write")
        handle.flush()
        os.fsync(handle.fileno())
    return digest(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=PHASES)
    for name in ("code", "intake-root", "packet-result", "packet-terminal", "roster", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--point-head", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    capture = args.phase == "capture"
    edit = args.phase in {"prepare-edits", "verify-edits"}
    require(sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ("0" if capture else "-1")
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == HERE
            and args.output.name == args.phase and not args.output.exists()
            and (args.checkpoint is not None) == capture and (args.point_head is not None) == edit,
            "Fresh isolated phase under the new supplemental execution root")
    require(shutil.disk_usage(HERE).free >= 3 * 1024 ** 3, "Room for the frozen supplemental receipts and planes")
    binding = json.loads(pin(HERE / "ADMISSION_PRIVATE.json", args.admission_sha256))
    worker = digest(Path(__file__).read_bytes())
    require(binding["schema"] == "umbra-supplement-frozen-carrier-v1" and binding["worker_sha256"] == worker,
            "Exact supplemental pre-execution gate")
    pin(HERE / "PRELOCK.md", binding["protocol_sha256"])
    args.code = args.code.resolve(strict=True)
    args.intake_root = args.intake_root.resolve(strict=True)
    roster, ids = load_admission(args, binding)
    old = module(args.code / ORIGINAL / "capture_umbra.py", CAPTURE_SOURCE)
    adapter = capture_modules(args.code, old)
    _, native, common = adapter
    native.install_guard(args.intake_root / "intake", [{"source_image_name": f"UMB-{i:03d}.png"} for i in ids])
    items = load_items(args.intake_root / "intake", roster, ids)
    bound = dict(admission_sha256=args.admission_sha256, protocol_sha256=binding["protocol_sha256"],
                 intake_sha256=binding["manifest_sha256"], intake_result_sha256=binding["intake_result"],
                 intake_terminal_sha256=binding["intake_terminal"], intake_readback_sha256=binding["readback_result"],
                 intake_readback_terminal_sha256=binding["readback_terminal"], reviewer_receipt_sha256=binding["packet_result"],
                 reviewer_terminal_sha256=binding["packet_terminal"], metadata_result_sha256=binding["metadata_result_sha256"],
                 metadata_terminal_sha256=binding["metadata_terminal_sha256"],
                 anonymous_roster_sha256=binding["anonymous_roster"], checkpoint_sha256=CHECKPOINT, images=len(items))
    operation = edit_stage if edit else model_stage
    result = operation(args, items, adapter, bound, worker)
    result.update(bound, phase=args.phase, worker_sha256=worker, target_labels_opened=False,
                  optimizer_updates=0, seconds=perf_counter() - start)
    checksum = write(args.output / "RESULT_SAFE.json", common.encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=checksum)), flush=True)


if __name__ == "__main__":
    main()

