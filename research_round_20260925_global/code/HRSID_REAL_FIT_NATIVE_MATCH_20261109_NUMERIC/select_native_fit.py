"""Materialize the first seeded empty crop and first densest direct resize from closed P109 FIT."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from zipfile import ZipFile


PRIVATE = Path(__file__).resolve().parent.parent
AUDIT_SOURCE = PRIVATE / "HRSID_PNG_FIT_NATIVE_AUG_AUDIT_CANDIDATE_20261109_01/audit_hrsid_fit_native_aug.py"
AUDIT_SOURCE_SHA = "6d1b1b94c14cc8f6c3cf7d5eb70ea8db83a6c0f4c0219c68b4dbf24d773aca14"
AUDIT_REPORT = PRIVATE / "HRSID_PNG_FIT_NATIVE_AUG_AUDIT_ACTUAL_PRIVATE_20261109_01/NATIVE_AUG_AUDIT_SAFE.json"
AUDIT_REPORT_SHA = "53d2af6e17fca303ce93c913a4be5409e59ef04e0a0807b169897bbc7ee9c51b"
ARCHIVE_ROOT = Path("D:/SAR_DATASETS/HRSID_PNG_OFFICIAL_INTAKE_20261015/ARCHIVE_STANDARD_HTTP_TRANSFER_20261109_01")
VIEW_ROOT = PRIVATE / "HRSID_PNG_FIT_MASK_TRAIN_VIEW_ACTUAL_PRIVATE_20261109_01"
FINDINGS = PRIVATE / "HRSID_OFFICIAL_PNG_FIT_AUDIT_ACTUAL_PRIVATE_20261109_01/FIT_FINDINGS_PRIVATE.json"
DENSE_COUNT = 163
INPUT_PACKET = "TWO_NATIVE_BATCHES_PRIVATE.pt"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_pinned_module(path, digest, name):
    source = path.read_bytes()
    need(sha256(source).hexdigest() == digest, f"Exact closed source required for {name}")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def file_digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def select_cases(images, groups, base_seed, is_direct, materialize):
    need(max(len(v) for v in groups.values()) == DENSE_COUNT, "Exact FIT source maximum")
    selected = {}
    for index, image in enumerate(images):
        seed = base_seed + index
        annotations = groups[image["id"]]
        direct = is_direct(seed)
        seek_empty = "empty" not in selected and not direct
        seek_dense = "dense" not in selected and direct and len(annotations) == DENSE_COUNT
        if seek_empty or seek_dense:
            observed, sample = materialize(image, annotations, seed)
            if seek_dense:
                need(observed["labels"] == DENSE_COUNT, "The direct resize must retain exactly 163 targets")
                selected["dense"] = dict(sample=sample, seed=seed, after=observed)
            if seek_empty and observed["labels"] == 0:
                selected["empty"] = dict(sample=sample, seed=seed, after=observed)
            if len(selected) == 2:
                break
    need(set(selected) == {"empty", "dense"}, "Both predetermined real FIT roles are required")
    return selected


def acquire():
    import torch

    audit = read_pinned_module(AUDIT_SOURCE, AUDIT_SOURCE_SHA, "_closed_p109_native_input_audit")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_available()
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
         and tuple(sys.version_info[:3]) == (3, 10, 13)
         and {key: version(key) for key in audit.VERSIONS} == audit.VERSIONS
         and all(file_digest(Path(module.__file__)) == digest for module, digest in audit.SOURCE_SHA.items()),
         "Exact offline P109 CPU input runtime")
    sys.addaudithook(audit.offline)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = audit.pinned_json(AUDIT_REPORT, AUDIT_REPORT_SHA, 1024**2)
    direct = report["by_native_resize_branch"]["direct_resize"]["augmented"]
    cropped = report["by_native_resize_branch"]["resize_crop"]["augmented"]
    need(report["complete_FIT_images"] == 2937 and report["no_empty_mask_target_gate"] is True
         and report["one_view_per_FIT_image"] is True and report["resolution"] == 576
         and report["TEST_member_opened"] is report["original_DEV_image_opened"] is False
         and direct["zero_target_images"] == 0 and direct["max_targets_per_image"] == DENSE_COUNT
         and cropped["zero_target_images"] == 99, "Exact closed P109 selection prerequisites")
    receipt = audit.pinned_json(ARCHIVE_ROOT / "ROOT_ARCHIVE_INTEGRITY_SAFE.json", audit.ARCHIVE_RECEIPT_SHA, 1024**2)
    verified = audit.pinned_json(VIEW_ROOT / "ROOT_VERIFIED_TRAIN_VIEW_SAFE.json", audit.FIT_VIEW_RECEIPT_SHA, 1024**2)
    need(receipt["archive_sha256"] == audit.ARCHIVE_SHA and receipt["all_member_CRC_verified"] is True
         and receipt["expected_central_directory_sha256"] == audit.CENTRAL_SHA
         and verified["derived_sha256"] == audit.FIT_VIEW_SHA and verified["only_five_original_exclusions"] is True
         and verified["independent_closed_annotation_copy_field_readback"] is True, "Exact archive and FIT view receipts")
    view = audit.pinned_json(VIEW_ROOT / "FIT_MASK_TRAINING_VIEW_PRIVATE.json", audit.FIT_VIEW_SHA, 16 * 1024**2)
    findings = audit.pinned_json(FINDINGS, audit.FINDINGS_SHA, 1024**2)
    images, groups, _ = audit.ordered_fits(view, findings)
    native = audit.NativeInputAudit()
    archive = ARCHIVE_ROOT / "HRSID_PNG_ARCHIVE_COMPLETE_PRIVATE.zip"
    original_stat = archive.stat()
    need(original_stat.st_size == 5698302315, "Exact closed official PNG archive length")
    with archive.open("rb") as stream:
        need(file_identity(original_stat) == file_identity(os.fstat(stream.fileno())), "Closed PNG handle identity")
        stream.seek(5697629273)
        need(sha256(stream.read(672944)).hexdigest() == audit.CENTRAL_SHA, "Exact original ZIP central directory")
        with ZipFile(stream) as source:
            need(len(source.infolist()) == 5612, "Exact archive member census")

            def materialize(record, annotations, seed):
                raw = audit.read_member(source, audit.png_member(record), 8 * 1024**2)
                image, scale = audit.io_utils.decode_pil_image_bytes(raw, draft_size=None)
                need(raw.startswith(b"\x89PNG\r\n\x1a\n") and image.mode == "RGB"
                     and image.size == (800, 800) and scale == (1., 1.), "Full native 800x800 PNG decode")
                before, after, _, target, pixels = native.view(image, record["id"], annotations, seed)
                need(before["labels"] == len(annotations) and before["empty_masks"] == after["empty_masks"] == 0,
                     "Exact source target count and no empty accepted mask rows")
                return after, dict(pixels=pixels.clone(), target={k: v.clone() for k, v in target.items()})

            def is_direct(seed):
                return bool(torch.rand((), generator=torch.Generator().manual_seed(seed)) < .5)

            cases = select_cases(images, groups, audit.BASE_SEED, is_direct, materialize)
        need(file_identity(original_stat) == file_identity(os.fstat(stream.fileno())),
             "Closed PNG archive unchanged during member selection")
    need(file_identity(original_stat) == file_identity(archive.stat()), "Closed archive still unchanged")
    return dict(audit_sha256=AUDIT_REPORT_SHA, fit_view_sha256=audit.FIT_VIEW_SHA,
                original_archive_sha256=audit.ARCHIVE_SHA, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol)), "Prospectively committed protocol")
    need(not args.output.exists(), "Fresh private output directory")
    packet = acquire()
    import torch

    args.output.mkdir()
    path = args.output / INPUT_PACKET
    with path.open("xb") as stream:
        torch.save(packet, stream)
    result = dict(status="TWO_REAL_HRSID_FIT_BATCHES_READY", protocol=args.protocol,
                  p109_sha256=AUDIT_REPORT_SHA, selector_sha256=file_digest(Path(__file__)),
                  fit_view_sha256=packet["fit_view_sha256"], archive_sha256=packet["original_archive_sha256"],
                  private_packet_bytes=path.stat().st_size, private_packet_sha256=file_digest(path),
                  empty={"branch": "resize_crop", "targets": 0}, dense={"branch": "direct_resize", "targets": DENSE_COUNT},
                  resolution=576, FIT_only=True, DEV_image_opened=False, TEST_member_opened=False, model_executed=False)
    write_json(args.output / "CPU_SELECTION_SAFE.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
