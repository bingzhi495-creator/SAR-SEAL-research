"""Materialize only the closed official HRSID FIT PNG view in native RF-DETR COCO layout."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
from shutil import disk_usage
import sys
from types import ModuleType
from zipfile import ZipFile

from PIL import Image


PRIVATE = Path(__file__).resolve().parent.parent
SELECTOR = PRIVATE / "HRSID_FIT_REAL_NATIVE_MATCH_CANDIDATE_20261109_01/select_native_fit.py"
SELECTOR_SHA = "59e6129f8f10ccfb0e83fe816f6b41c3b5cd9bdeccfae6ebb7cb719c9c4df34f"
ANNOTATIONS = "_annotations.coco.json"
MEMBER_LIMIT = 8 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def exclusive_write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def sha(data):
    return sha256(data).hexdigest()


def output_is_fresh(path):
    return (path.is_absolute() and not path.exists() and not path.is_symlink()
            and path.parent.is_dir() and path.parent.resolve(strict=True) == path.parent)


def image_record(source):
    name = PurePosixPath(source["file_name"])
    need(name.name == source["file_name"] and name.suffix.lower() in {".jpg", ".png"}
         and re.fullmatch(r"P\d+_[A-Za-z0-9_]+", name.stem)
         and source["height"] == source["width"] == 800, "Safe official FIT image basename required")
    return dict(source, file_name=name.stem + ".png")


def materialize(archive, view, output):
    need(output_is_fresh(output), "Fresh, absolute and resolved output directory required")
    images = [image_record(row) for row in view["images"]]
    need(bool(images) and len({row["id"] for row in images}) == len(images)
         and len({row["file_name"].casefold() for row in images}) == len(images),
         "Duplicate FIT image IDs or Windows-equivalent basenames")
    members = Counter(info.filename for info in archive.infolist())
    pairs = []
    for row in images:
        name = f"images/{row['file_name']}"
        need(members[name] == 1, "Every selected FIT member must occur exactly once")
        info = archive.getinfo(name)
        need(not info.is_dir() and 0 < info.file_size <= MEMBER_LIMIT, "Bounded PNG ZIP member required")
        pairs.append((row, info))
    index = json_bytes(dict(view, images=images))
    pixel_bytes = sum(info.file_size for _, info in pairs)
    need(disk_usage(output.parent).free >= pixel_bytes + len(index) + 64 * 1024**2,
         "Free space required for the exact selected FIT PNGs and receipts")
    output.mkdir()
    train = output / "train"
    train.mkdir()
    manifest = []
    for row, info in pairs:
        encoded = archive.read(info)
        need(len(encoded) == info.file_size and encoded.startswith(b"\x89PNG\r\n\x1a\n"),
             "Exact ZIP CRC readback and real PNG signature")
        with Image.open(BytesIO(encoded)) as image:
            need(image.format == "PNG" and image.size == (800, 800), "Real PNG with original 800-pixel geometry")
            rgb_sha = sha(image.convert("RGB").tobytes())
        destination = train / row["file_name"]
        encoded_sha = sha(encoded)
        exclusive_write(destination, encoded)
        need(sha(destination.read_bytes()) == encoded_sha, "On-disk PNG readback must match the original ZIP member")
        manifest.append(dict(image_id=row["id"], member=info.filename, native_basename=destination.name,
                             zip_header_offset=info.header_offset, zip_crc32=f"{info.CRC:08x}",
                             compressed_bytes=info.compress_size, encoded_bytes=info.file_size,
                             encoded_sha256=encoded_sha, decoded_rgb_sha256=rgb_sha))
    annotations = train / ANNOTATIONS
    exclusive_write(annotations, index)
    need(annotations.read_bytes() == index and {p.name for p in train.iterdir()} ==
         {ANNOTATIONS, *(row["file_name"] for row in images)}, "Exact native annotation fields and FIT file census")
    return dict(native_index_sha256=sha(index), native_index_bytes=len(index),
                selected_encoded_png_bytes=pixel_bytes, images=manifest)


def pinned_selector():
    source = SELECTOR.read_bytes()
    need(sha(source) == SELECTOR_SHA, "Exact previously accepted FIT selector source")
    module = ModuleType("_closed_p110_fit_selector")
    module.__file__ = str(SELECTOR)
    exec(compile(source, str(SELECTOR), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol)) and output_is_fresh(args.output),
         "Prospectively committed protocol and fresh absolute output")
    need(tuple(sys.version_info[:3]) == (3, 10, 13) and version("Pillow") == "12.3.0"
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact pinned offline CPU materialization runtime")
    parent = pinned_selector()
    audit = parent.read_pinned_module(parent.AUDIT_SOURCE, parent.AUDIT_SOURCE_SHA, "_closed_p109_native_layout")
    sys.addaudithook(audit.offline)
    report = audit.pinned_json(parent.AUDIT_REPORT, parent.AUDIT_REPORT_SHA, 1024**2)
    receipt = audit.pinned_json(parent.ARCHIVE_ROOT / "ROOT_ARCHIVE_INTEGRITY_SAFE.json", audit.ARCHIVE_RECEIPT_SHA, 1024**2)
    fit_receipt = audit.pinned_json(parent.VIEW_ROOT / "ROOT_VERIFIED_TRAIN_VIEW_SAFE.json", audit.FIT_VIEW_RECEIPT_SHA, 1024**2)
    need(report["complete_FIT_images"] == 2937 and report["fit_view_sha256"] == audit.FIT_VIEW_SHA
         and report["original_archive_sha256"] == audit.ARCHIVE_SHA
         and report["TEST_member_opened"] is report["original_DEV_image_opened"] is False
         and receipt["archive_sha256"] == audit.ARCHIVE_SHA and receipt["all_member_CRC_verified"] is True
         and receipt["expected_central_directory_sha256"] == audit.CENTRAL_SHA
         and fit_receipt["derived_sha256"] == audit.FIT_VIEW_SHA
         and fit_receipt["only_five_original_exclusions"] is True
         and fit_receipt["independent_closed_annotation_copy_field_readback"] is True,
         "Exact accepted original archive and derived FIT annotation receipts")
    view = audit.pinned_json(parent.VIEW_ROOT / "FIT_MASK_TRAINING_VIEW_PRIVATE.json", audit.FIT_VIEW_SHA, 16 * 1024**2)
    findings = audit.pinned_json(parent.FINDINGS, audit.FINDINGS_SHA, 1024**2)
    images, _, _ = audit.ordered_fits(view, findings)
    need(all(f"images/{image_record(row)['file_name']}" == audit.png_member(row) for row in images),
         "Native output names must be the exact audited PNG members")
    source = parent.ARCHIVE_ROOT / "HRSID_PNG_ARCHIVE_COMPLETE_PRIVATE.zip"
    start = source.stat()
    need(start.st_size == 5698302315, "Exact closed official archive length")
    with source.open("rb") as handle:
        need(parent.file_identity(start) == parent.file_identity(os.fstat(handle.fileno())),
             "Exact closed archive handle identity")
        handle.seek(5697629273)
        need(sha(handle.read(672944)) == audit.CENTRAL_SHA, "Exact official ZIP central directory")
        with ZipFile(handle) as archive:
            need(len(archive.infolist()) == 5612, "Exact original archive member census")
            manifest = materialize(archive, view, args.output)
        need(parent.file_identity(start) == parent.file_identity(os.fstat(handle.fileno())),
             "Closed source archive unchanged during materialization")
    need(parent.file_identity(start) == parent.file_identity(source.stat()) and len(manifest["images"]) == 2937,
         "Exact output population and unchanged closed archive")
    archive_info = dict(archive_sha256_receipt=audit.ARCHIVE_SHA, archive_bytes=start.st_size,
                        archive_central_sha256=audit.CENTRAL_SHA, fit_source_index_sha256=audit.FIT_VIEW_SHA)
    private_manifest = json_bytes(dict(manifest, protocol=args.protocol, **archive_info))
    exclusive_write(args.output / "NATIVE_FIT_LAYOUT_MANIFEST_PRIVATE.json", private_manifest)
    result = dict(status="NATIVE_HRSID_FIT_COCO_LAYOUT_COMPLETE", protocol=args.protocol,
                  p109_sha256=parent.AUDIT_REPORT_SHA, materializer_sha256=sha(Path(__file__).read_bytes()),
                  **archive_info, native_index_sha256=manifest["native_index_sha256"],
                  native_index_bytes=manifest["native_index_bytes"], native_relative_index=f"train/{ANNOTATIONS}",
                  private_manifest_sha256=sha(private_manifest), fit_images=2937,
                  selected_encoded_png_bytes=manifest["selected_encoded_png_bytes"],
                  retained_annotations=len(view["annotations"]), only_filename_suffix_changed=True,
                  FIT_only=True, DEV_image_opened=False, TEST_member_opened=False, model_executed=False)
    exclusive_write(args.output / "NATIVE_FIT_LAYOUT_SAFE.json", json_bytes(result))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
