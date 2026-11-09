"""Independently verify the completed SSDD FIT-only layout against its closed archive."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from zipfile import ZIP_STORED, ZipFile

from ssdd_fit_stage import FIT_ARCHIVE_BYTES, FIT_ARCHIVE_SHA, FIT_INDEX_SHA
from ssdd_phase_init import require


FIT_IMAGES = 741
FIT_ANNOTATIONS = 1661
UNIQUE_IMAGES = 730
INDEX_BYTES = 498760
MANIFEST_BYTES = 861009
MANIFEST_SHA = "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"
MANIFEST_SCHEMA = "SSDD-FIT741-portable-original-byte-input-manifest-v1"
JPEG_NAME = re.compile(r"[A-Za-z0-9_.-]+\.jpg")
DIGEST = re.compile(r"[0-9a-f]{64}")


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def canonical_file(path):
    return path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def file_sha256(path, expected_bytes):
    require(canonical_file(path), "Expected exact canonical closed file")
    digest = sha256()
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        require(before[2] == expected_bytes, "Expected exact file byte count")
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        after = identity(os.fstat(stream.fileno()))
    require(before == after == identity(path.stat()), "Closed file changed during independent readback")
    return digest.hexdigest()


def read_member(archive, name, expected_bytes, expected_sha):
    info = archive.getinfo(name)
    require(info.file_size == expected_bytes and info.compress_type == ZIP_STORED,
            "Exact stored archive member size required")
    raw = archive.read(info)
    require(len(raw) == expected_bytes and sha256(raw).hexdigest() == expected_sha,
            "Archive member digest differs from fixed FIT input")
    return raw


def readback_fit_layout(archive_path, layout):
    """Compare each logical SSDD FIT image to its manifest reference, with no decoding."""
    require(layout.is_absolute() and layout.is_dir() and layout.resolve(strict=True) == layout,
            "Expected a canonical FIT-only layout")
    train = layout / "train"
    require(list(layout.iterdir()) == [train] and train.is_dir() and not train.is_symlink()
            and train.resolve(strict=True) == train, "Expected only a canonical train directory in FIT-only layout")
    require(file_sha256(archive_path, FIT_ARCHIVE_BYTES) == FIT_ARCHIVE_SHA, "Fixed SSDD FIT archive digest differs")
    archive_identity = identity(archive_path.stat())
    with ZipFile(archive_path) as archive:
        manifest = json.loads(read_member(archive, "manifest.json", MANIFEST_BYTES, MANIFEST_SHA))
        index = json.loads(read_member(archive, "annotations/fit.json", INDEX_BYTES, FIT_INDEX_SHA))
        archive_names = archive.namelist()
    require(archive_identity == identity(archive_path.stat()), "Closed FIT archive changed during readback")
    require(file_sha256(train / "_annotations.coco.json", INDEX_BYTES) == FIT_INDEX_SHA,
            "Staged FIT annotation digest differs from original bytes")

    images, annotations = index["images"], index["annotations"]
    ids = [row["id"] for row in images]
    require(manifest["schema"] == MANIFEST_SCHEMA and len(manifest["rows"]) == len(images) == len(set(ids)) == FIT_IMAGES
            and all(type(i) is int and i >= 0 for i in ids) and len(annotations) == FIT_ANNOTATIONS
            and [row["id"] for row in index["categories"]] == [1], "Exact FIT population required")
    id_set = set(ids)
    require(len({row["id"] for row in annotations}) == FIT_ANNOTATIONS
            and all(row["image_id"] in id_set and row["category_id"] == 1 for row in annotations),
            "FIT annotations must belong to the exact admitted image set")
    entries = []
    names = {"_annotations.coco.json"}
    expected_zip = {"manifest.json", "annotations/fit.json"}
    encoded_images = set()
    for position, (row, image) in enumerate(zip(manifest["rows"], images, strict=True)):
        name = image["file_name"]
        ref = row["original_image_ref"]
        expected_bytes, expected_sha = ref["bytes"], ref["sha256"]
        zip_name = f"images/{expected_sha}.jpg"
        require(type(name) is str and JPEG_NAME.fullmatch(name) and name not in names
                and row["dataset_index"] == position and row["image_id"] == image["id"]
                and row["original_basename"] == name
                and row["original_shape"] == [image["height"], image["width"]]
                and type(expected_bytes) is int and 0 < expected_bytes <= 2 * 1024**2
                and type(expected_sha) is str and DIGEST.fullmatch(expected_sha)
                and row["archive_image_entry"] == zip_name, "Exact original FIT manifest and safe JPEG name required")
        require(file_sha256(train / name, expected_bytes) == expected_sha,
                "Staged FIT JPEG content differs from manifest")
        names.add(name)
        encoded_images.add(expected_sha)
        expected_zip.add(zip_name)
        entries.append([image["id"], name, expected_bytes, expected_sha])
    require({path.name for path in train.iterdir()} == names and len(encoded_images) == UNIQUE_IMAGES
            and len(archive_names) == len(set(archive_names)) and set(archive_names) == expected_zip,
            "Exact staged and original FIT rosters required")
    return dict(status="SSDD_FIT741_ONLY_LAYOUT_INDEPENDENT_BYTES_VERIFIED_V1",
                images=FIT_IMAGES, annotations=FIT_ANNOTATIONS, unique_encoded_images=UNIQUE_IMAGES,
                fit_archive_sha256=FIT_ARCHIVE_SHA, fit_index_sha256=FIT_INDEX_SHA, portable_manifest_sha256=MANIFEST_SHA,
                original_annotation_order_sha256=sha256(compact(ids)).hexdigest(),
                native_sorted_id_order_sha256=sha256(compact(sorted(ids))).hexdigest(),
                content_roster_sha256=sha256(compact(entries)).hexdigest(), FIT=True, DEV=False, TEST=False,
                model_executed=False, training=False, AP=False)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(args.receipt.is_absolute() and args.receipt.parent.resolve(strict=True) == args.receipt.parent
            and not args.receipt.exists() and not args.receipt.is_symlink()
            and args.receipt != args.layout and args.layout not in args.receipt.parents,
            "Expected a fresh readback receipt outside the FIT layout")
    result = readback_fit_layout(args.archive, args.layout)
    with args.receipt.open("xb") as stream:
        stream.write(compact(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()

