"""Capture and independently verify a frozen detector on the 12 admitted Umbra windows."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
PROTOCOL_NAME = "UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md"
PROTOCOL_SHA = "ff70161af804adc9f52c044771b504884ea80b7b2242ffc38fab51b55f2839ad"
MANIFEST = "24d08f3bf1ecb08212f78e5a3cfa8c709ef9a2194e9d345f5cb04a1f9f48c29e"
NATIVE_RESULT = "52482b1bb53d3ec8ca2ee4eebd9984aa5cacbb156873177841a5a2e2c4cdb7ec"
NATIVE_TERMINAL = "1aa3cb27c6f5205503c51b48f6a3a03104194365b3914639da507b5629f296b3"
NATIVE_READBACK = "11a99f558de5ebc9bf38bb70f0ec4516e68a37887b91f5825e3a6f41e7db784c"
REVIEWER_RECEIPT = "004c0557b4e7ff7e1299227b601c438039fa69ea4a15c291f8964a763779cd81"
ROSTER = "e626c38ea53727a2d235a80e675735898bbce5defce698f04fb602d674a87a56"
CAPTURE = "UMBRA12_FROZEN_CAPTURE_COMPLETE"
VERIFIED = "UMBRA12_FROZEN_CAPTURE_READBACK_COMPLETE"
NATIVE_HW = (1024, 1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Identity differs for {path.name}")
    return raw


def load_items(args):
    from PIL import Image

    terminal = json.loads(pin(args.intake_terminal, NATIVE_TERMINAL))
    result = json.loads(pin(args.intake / "RESULT_SAFE.json", NATIVE_RESULT))
    readback = json.loads(pin(args.intake_readback, NATIVE_READBACK))
    reviewer = json.loads(pin(args.reviewer_receipt, REVIEWER_RECEIPT))
    roster = json.loads(pin(args.roster, ROSTER))

    require(terminal["phase"] == "native-window-intake" and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["result_sha256"] == NATIVE_RESULT,
            "Observed native intake closure required")
    require(result["status"] == "UMBRA_NATIVE_WINDOWS_COMPLETE"
            and result["manifest_sha256"] == readback["manifest_sha256"] == reviewer["frozen_manifest_sha256"] == MANIFEST
            and result["requested_windows"] == result["decoded_unique_windows"] == result["verified_native_windows"] == 12
            and result["native_window_shape_yx"] == list(NATIVE_HW) and result["failed_windows"] == []
            and result["annotation_files"] == result["detector_forwards"] == result["optimizer_updates"] == 0,
            "Complete original label-free native-window cohort required")
    require(readback["status"] == "INDEPENDENT_NATIVE_WINDOW_READBACK_COMPLETE"
            and readback["extraction_actual_exit"] == 0 and readback["extraction_result_sha256"] == NATIVE_RESULT
            and readback["readback_windows"] == readback["unique_pixel_arrays"] == 12
            and readback["matching_uncompressed_ranges"] == 48 and readback["annotations_read"] == 0,
            "Exact independent native-window readback required")
    require(reviewer["status"] == "TWO_BLIND_REVIEW_PACKETS_VERIFIED"
            and reviewer["verified_source_result_sha256"] == NATIVE_RESULT and reviewer["anonymous_roster_sha256"] == ROSTER
            and reviewer["windows_per_packet"] == 12 and reviewer["labels_present"] == reviewer["model_outputs_present"] == 0
            and set(roster) == {"schema", "images"} and roster["schema"] == "sar-hull-roster-v1"
            and [row["image_id"] for row in roster["images"]] == [f"UMB-{i:03d}" for i in range(1, 13)],
            "Expected the exact anonymous roster shared with both reviewers")
    image_root = (args.intake / "images").resolve(strict=True)
    require(image_root == args.intake / "images", "Expected the original intake image directory")
    items = []
    for i, row in enumerate(roster["images"], 1):
        name = row["image_id"] + ".png"
        path = image_root / name
        require(path.resolve(strict=True) == path and (row["height"], row["width"]) == NATIVE_HW,
                "Expected an original anonymous image path and native canvas")
        raw = pin(path, row["png_sha256"])
        with Image.open(BytesIO(raw)) as image:
            image.load()
            require(image.format == "PNG" and image.mode == "L" and image.size == NATIVE_HW and not image.info,
                    "Expected the observed native grayscale PNG")
            rgb = image.convert("RGB").tobytes()
        items.append(dict(image_id=i, original_hw=list(NATIVE_HW), sensor="Umbra",
                          acquisition_group=row["image_id"], source_image_name=name, rgb_sha256=digest(rgb),
                          image_ref=dict(path="images/" + name, sha256=row["png_sha256"], bytes=len(raw))))
    require(len({item["rgb_sha256"] for item in items}) == 12, "Expected 12 distinct decoded windows")
    return items


def decode_native(root, item):
    from PIL import Image
    from rfdetr.datasets.io_utils import decode_pil_image_bytes

    ref = item["image_ref"]
    require(ref["path"] == "images/" + item["source_image_name"], "Exact native PNG identity")
    raw = pin(root / ref["path"], ref["sha256"])
    image, scale = decode_pil_image_bytes(raw, draft_size=None)
    with Image.open(BytesIO(raw)) as reference:
        require(reference.format == "PNG" and reference.mode == "L" and reference.size == NATIVE_HW
                and not reference.info and image.tobytes() == reference.convert("RGB").tobytes(),
                "Independent lossless grayscale replication")
    require(len(raw) == ref["bytes"] and image.mode == "RGB" and image.size == NATIVE_HW
            and scale == (1.0, 1.0) and digest(image.tobytes()) == item["rgb_sha256"],
            "Unscaled RGB with three copies of the original grayscale")
    return image


def compare_capture_receipt(args, native):
    run = args.output.parent
    terminal = json.loads((run / "observed_capture" / "TERMINAL.json").read_bytes())
    receipt = json.loads(pin(run / "capture" / "RESULT_SAFE.json", terminal["result_sha256"]))
    require(terminal["phase"] == receipt["phase"] == "capture"
            and receipt["checkpoint_sha256"] == native.SOURCE_CHECKPOINT and receipt["anonymous_roster_sha256"] == ROSTER
            and receipt["intake_result_sha256"] == NATIVE_RESULT and receipt["intake_readback_sha256"] == NATIVE_READBACK
            and receipt["reviewer_receipt_sha256"] == REVIEWER_RECEIPT
            and receipt["target_labels_opened"] is False and receipt["optimizer_updates"] == 0,
            "Original label-free, frozen Umbra capture lineage required")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("capture", "verify-capture"))
    for name in ("code", "intake", "intake-terminal", "intake-readback", "reviewer-receipt",
                 "roster", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    started = perf_counter()

    capture = args.phase == "capture"
    require(sys.flags.isolated and sys.dont_write_bytecode and args.output.is_absolute() and not args.output.exists()
            and args.output.name == args.phase and (args.checkpoint is not None) == capture
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ("0" if capture else "-1")
            and re.fullmatch("[0-9a-f]{64}", args.protocol_sha256), "Expected the isolated, locked capture or CPU readback stage")
    args.intake = args.intake.resolve(strict=True)
    args.code = args.code.resolve(strict=True)
    require(args.protocol.name == PROTOCOL_NAME and args.protocol_sha256 == PROTOCOL_SHA,
            "Expected the preregistered Umbra scoring protocol")
    pin(args.protocol, PROTOCOL_SHA)
    items = load_items(args)

    source = args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    operator = ModuleType("_umbra_portable_eval")
    operator.__file__ = str(source)
    exec(compile(pin(source, PORTABLE), str(source), "exec"), operator.__dict__)
    native = operator.module(args.code / operator.NATIVE / "capture_instance_rsdd.py", operator.CAPTURE_SOURCE)
    common = native.reference(args.code / operator.UEIKAP / "capture.py")
    native.install_guard(args.intake, items)
    native.decode_native = decode_native
    native.CAPTURE_STATUS = CAPTURE
    native.VERIFY_STATUS = VERIFIED
    # The source readback ties its original ledger to this exact worker.
    native.__file__ = str(Path(__file__).resolve(strict=True))

    import torch

    require(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
            and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
            and version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Expected the original accepted capture runtime")
    torch.set_num_threads(2 if capture else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    if capture:
        from rfdetr.models.postprocess import PostProcess

        require(PostProcess is not None and torch.get_float32_matmul_precision() == "high"
                and torch.backends.cuda.matmul.allow_tf32, "Expected original detector precision")
        carrier, core, base, post = operator.detector_components(args.code, args.checkpoint.resolve(strict=True))
        carrier.load_core = lambda _: core
        carrier.setup = lambda _args, _core: (base, post)
        common.exact_source = lambda _: carrier
        request = SimpleNamespace(intake=args.intake, output=args.output, private_root=args.code)
        result = native.capture(request, items, torch, common)
        require(result["model_forward_calls"] == 12 and torch.backends.cudnn.allow_tf32 is False,
                "Exact forward count and original cuDNN policy")
    else:
        compare_capture_receipt(args, native)
        request = SimpleNamespace(intake=args.intake, output=args.output,
                                  protocol_sha=args.protocol_sha256, manifest_sha=MANIFEST)
        result = native.verify(request, items, torch, common)

    result.update(phase=args.phase, worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=args.protocol_sha256,
                  intake_sha256=MANIFEST, intake_result_sha256=NATIVE_RESULT, intake_terminal_sha256=NATIVE_TERMINAL,
                  intake_readback_sha256=NATIVE_READBACK, reviewer_receipt_sha256=REVIEWER_RECEIPT,
                  anonymous_roster_sha256=ROSTER, checkpoint_sha256=native.SOURCE_CHECKPOINT, images=len(items),
                  target_labels_opened=False, optimizer_updates=0, seconds=perf_counter() - started)
    result_sha = common.write(args.output / "RESULT_SAFE.json", common.encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()
