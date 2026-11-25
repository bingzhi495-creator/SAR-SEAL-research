"""Capture frozen RF-DETR outputs on the exact label-free SAR-SSP validation roster."""

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
from zipfile import ZipFile


ARCHIVE_SHA = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"
ARCHIVE_BYTES = 866_821_756
MEMBER = re.compile(r"datasets/SAR-SSP/val2017/([0-9]{12})\.jpg")
COUNT = 232
HW = (500, 500)
MANIFEST = "IMAGE_MANIFEST_PRIVATE.json"
INTAKE_STATUS = "SAR_SSP_VAL232_IMAGES_ONLY_V1"
MANIFEST_SHA = "16c8e6b7655e51f208c3a9708acd419a60126e55a9075cdea7d4b9e4edbb9425"
PORTABLE_SHA = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
CHECKPOINT_SHA = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
QUALIFIED = "SAR_SSP_VAL232_DECODE_QUALIFIED_V1"
CAPTURED = "SAR_SSP_VAL232_RFDETR_NATIVE_CAPTURED_V1"
VERIFIED = "SAR_SSP_VAL232_RFDETR_NATIVE_VERIFIED_V1"
RESULT = "RESULT_SAFE.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return digest(raw)


def extract_images(archive, output):
    from PIL import Image

    require(output.name == "intake" and not output.exists(), "Fresh private intake directory required")
    require(archive.stat().st_size == ARCHIVE_BYTES, "Locked author archive size differs")
    with archive.open("rb") as stream:
        archived_digest = sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            archived_digest.update(block)
        require(archived_digest.hexdigest() == ARCHIVE_SHA, "Locked author archive SHA-256 differs")
        with ZipFile(stream) as source:
            members = [(match, member) for member in source.infolist() if (match := MEMBER.fullmatch(member.filename))]
            members.sort(key=lambda entry: entry[0][1])
            require(len(members) == len({entry[0][1] for entry in members}) == COUNT,
                    "Expected the 232 unique author validation JPEGs")
            output.mkdir()
            images = output / "images"
            images.mkdir()
            rows = []
            for match, member in members:
                raw = source.read(member)
                name = match[1] + ".jpg"
                with Image.open(BytesIO(raw)) as image:
                    require(image.format == "JPEG" and image.size == HW[::-1], "Expected native 500x500 author JPEG")
                    pixel_digest = digest(image.convert("RGB").tobytes())
                with (images / name).open("xb") as image_file:
                    image_file.write(raw)
                rows.append(dict(image_id=int(match[1]), original_hw=list(HW), sensor="unassigned",
                                 acquisition_group="unassigned", source_image_name=name,
                                 rgb_sha256=pixel_digest,
                                 image_ref=dict(path=f"images/{name}", bytes=len(raw), sha256=digest(raw))))
    roster = dict(status=INTAKE_STATUS, archive_sha256=ARCHIVE_SHA, split="val2017",
                  target_labels_opened=False, images=rows)
    manifest_sha = write_json(output / MANIFEST, roster)
    receipt = dict(status=INTAKE_STATUS, archive_sha256=ARCHIVE_SHA, manifest_sha256=manifest_sha,
                   images=COUNT, target_labels_opened=False)
    result_sha = write_json(output / "RESULT_SAFE.json", receipt)
    print(json.dumps(dict(status=INTAKE_STATUS, result_sha256=result_sha, manifest_sha256=manifest_sha)), flush=True)


def pinned(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned SHA-256 differs: " + path.name)
    return raw


def load_items(intake):
    manifest = json.loads(pinned(intake / MANIFEST, MANIFEST_SHA))
    require(manifest["status"] == INTAKE_STATUS and manifest["archive_sha256"] == ARCHIVE_SHA
            and manifest["split"] == "val2017" and manifest["target_labels_opened"] is False,
            "Closed label-free author validation roster")
    items = manifest["images"]
    names = [item["source_image_name"] for item in items]
    require(len(names) == len(set(names)) == COUNT and names == sorted(names)
            and {entry.name for entry in intake.joinpath("images").iterdir()} == set(names),
            "Complete ordered validation image roster")
    return items


def setup_capture(code, intake, capture):
    source = code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    operator = ModuleType("_sar_ssp_b5_portable")
    operator.__file__ = str(source)
    exec(compile(pinned(source, PORTABLE_SHA), str(source), "exec"), operator.__dict__)
    native = operator.module(code / operator.NATIVE / "capture_instance_rsdd.py", operator.CAPTURE_SOURCE)
    common = native.reference(code / operator.UEIKAP / "capture.py")
    require(operator.CHECKPOINT == native.SOURCE_CHECKPOINT == CHECKPOINT_SHA, "Frozen center5202 E24 lineage")
    items = load_items(intake)
    native.install_guard(intake, items)

    def reject_author_archives(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            require(Path(os.fsdecode(args[0])).suffix.casefold() != ".zip",
                    "No source archives during frozen image stages")

    sys.addaudithook(reject_author_archives)
    native.CAPTURE_STATUS = CAPTURED
    native.VERIFY_STATUS = VERIFIED
    # Native readback ties its receipt to this delegated worker, not its generic library.
    native.__file__ = str(Path(__file__).resolve(strict=True))

    import torch

    require(sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ("0" if capture else "-1")
            and sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
            and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
            and version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Exact original isolated B5 runtime required")
    return operator, native, common, items, torch


def infer(args):
    started = perf_counter()
    require(args.output.name == args.phase and args.output.parent == args.intake.parent
            and not args.output.exists() and args.intake.name == "intake",
            "Fresh phase destination beside closed images-only intake")
    pinned(args.protocol, args.protocol_sha)
    capture = args.phase == "capture"
    worker_sha = digest(Path(__file__).read_bytes())
    operator, native, common, items, torch = setup_capture(args.code, args.intake, capture)
    request = SimpleNamespace(intake=args.intake, output=args.output, private_root=args.code,
                              protocol_sha=args.protocol_sha, manifest_sha=MANIFEST_SHA)
    torch.set_num_threads(2 if capture else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")

    if args.phase == "qualify":
        for item in items:
            native.decode_native(args.intake, item)
        require(not torch.cuda.is_initialized(), "Image qualification uses no CUDA")
        args.output.mkdir()
        result = dict(status=QUALIFIED, detector_forwards=0, exact_jpeg_rgb_readback=True)
    else:
        qualified = json.loads(pinned(args.output.parent / "qualify" / RESULT, args.qualification_sha))
        require(qualified["status"] == QUALIFIED and qualified["images"] == COUNT
                and qualified["intake_sha256"] == MANIFEST_SHA and qualified["worker_sha256"] == worker_sha
                and qualified["protocol_sha256"] == args.protocol_sha
                and qualified["exact_jpeg_rgb_readback"] is True and qualified["target_labels_opened"] is False,
                "Original independent image decode qualification required")
        if capture:
            from rfdetr.models.postprocess import PostProcess

            require(PostProcess is not None and torch.get_float32_matmul_precision() == "high"
                    and torch.backends.cuda.matmul.allow_tf32, "Original B5 CUDA precision required")
            carrier, core, base, post = operator.detector_components(args.code, args.checkpoint)
            carrier.load_core = lambda _: core
            carrier.setup = lambda _args, _core: (base, post)
            common.exact_source = lambda _: carrier
            result = native.capture(request, items, torch, common)
            require(result["model_forward_calls"] == COUNT and torch.backends.cudnn.allow_tf32 is False,
                    "One forward per author validation JPEG and original cuDNN precision")
        else:
            result = native.verify(request, items, torch, common)
        result["qualification_sha256"] = args.qualification_sha
    result.update(phase=args.phase, worker_sha256=worker_sha, protocol_sha256=args.protocol_sha,
                  archive_sha256=ARCHIVE_SHA, intake_sha256=MANIFEST_SHA, checkpoint_sha256=CHECKPOINT_SHA,
                  author_partition="val2017", images=COUNT, target_labels_opened=False, optimizer_updates=0,
                  seconds=perf_counter() - started)
    result_sha = write_json(args.output / RESULT, result)
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="phase", required=True)
    intake = commands.add_parser("intake")
    intake.add_argument("--archive", type=Path, required=True)
    intake.add_argument("--output", type=Path, required=True)
    for phase in ("qualify", "capture", "verify-capture"):
        command = commands.add_parser(phase)
        for name in ("code", "intake", "protocol", "output"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--protocol-sha", required=True)
        if phase != "qualify":
            command.add_argument("--qualification-sha", required=True)
        if phase == "capture":
            command.add_argument("--checkpoint", type=Path, required=True)

    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase == "intake":
        extract_images(args.archive.resolve(strict=True), args.output)
    else:
        for name in ("code", "intake", "protocol", "checkpoint"):
            if hasattr(args, name):
                setattr(args, name, getattr(args, name).resolve(strict=True))
        infer(args)


if __name__ == "__main__":
    main()
