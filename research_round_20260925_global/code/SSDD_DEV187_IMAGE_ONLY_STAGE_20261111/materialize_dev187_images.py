"""Materialize the fixed SSDD DEV187 image-only directory from verified JPEG objects."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from tempfile import TemporaryDirectory


MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
MANIFEST_BYTES = 49477
DEV_IMAGES = 187
UNIQUE_IMAGES = 183
LOGICAL_BYTES = 7274234
UNIQUE_BYTES = 7118253
OBJECT_LIMIT = 16 * 1024**2
STATUS = "P287_SSDD_DEV187_IMAGE_ONLY_STAGE_READY_V1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_fields(pairs):
    document = dict(pairs)
    require(len(document) == len(pairs), "Duplicate image-only manifest JSON fields")
    return document


def reject_constant(_):
    raise ValueError("Nonfinite image-only manifest field")


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def checked_bytes(path, size, digest):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical image-only source file required")
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        require(before[2] == size, "Pinned image-only input byte length mismatch")
        raw = stream.read(size + 1)
        require(before == identity(os.fstat(stream.fileno())) == identity(path.stat()),
                "Image-only source changed during read")
    require(len(raw) == size and sha256(raw).hexdigest() == digest, "Pinned image-only input SHA-256 mismatch")
    return raw


def manifest_objects(ref):
    require(type(ref) is dict and ref.keys() == {"path", "bytes", "sha256"} and type(ref["path"]) is str
            and type(ref["bytes"]) is int and ref["bytes"] == MANIFEST_BYTES and ref["sha256"] == MANIFEST_SHA,
            "Exact historical image-only manifest byte length and digest required")
    document = json.loads(checked_bytes(Path(ref["path"]), MANIFEST_BYTES, MANIFEST_SHA),
                          object_pairs_hook=unique_fields, parse_constant=reject_constant)
    require(type(document) is dict and document.keys() == {
        "schema", "population", "model_execution", "AP", "TEST", "expected_ids", "items",
    } and document["schema"] == "Root-RFDETR-DEV187-image-only-manifest-v1"
        and document["population"] == "internal_DEV187"
        and document["model_execution"] is document["AP"] is document["TEST"] is False,
        "Expected the fixed image-only DEV187 manifest schema")
    ids, rows = document["expected_ids"], document["items"]
    require(type(ids) is type(rows) is list and len(ids) == len(rows) == DEV_IMAGES
            and all(type(value) is int and value >= 0 for value in ids) and len(set(ids)) == DEV_IMAGES,
            "Expected the fixed ordered DEV187 image IDs")

    objects, names = {}, set()
    for image_id, row in zip(ids, rows, strict=True):
        require(type(row) is dict and row.keys() == {"image_id", "original_hw", "image_ref"}
                and type(row["image_id"]) is int and row["image_id"] == image_id,
                "Original DEV187 logical image identity and order required")
        hw, image = row["original_hw"], row["image_ref"]
        require(type(hw) is list and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw)
                and type(image) is dict and image.keys() == {"path", "bytes", "sha256"}
                and type(image["path"]) is str and type(image["bytes"]) is int and 0 < image["bytes"] <= OBJECT_LIMIT
                and type(image["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", image["sha256"]),
                "Exact bounded original JPEG reference and dimensions required")
        original = PurePosixPath(image["path"])
        name = original.name
        require(original.is_absolute() and original.parent.name == "valid" and ".." not in original.parts
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.jpg", name) and not PureWindowsPath(name).is_reserved()
                and name.casefold() not in names, "Distinct portable original JPEG basenames required")
        names.add(name.casefold())
        size, digest = image["bytes"], image["sha256"]
        recorded_size, recorded_hw, aliases = objects.setdefault(digest, (size, hw, []))
        require((size, hw) == (recorded_size, recorded_hw), "A JPEG byte alias must retain its original size and dimensions")
        aliases.append(name)
    require(len(objects) == UNIQUE_IMAGES and sorted(len(names) for _, _, names in objects.values())
            == [1] * 179 + [2] * 4
            and sum(size for size, _, _ in objects.values()) == UNIQUE_BYTES
            and sum(size * len(aliases) for size, _, aliases in objects.values()) == LOGICAL_BYTES,
            "Exact 187 logical JPEG names and 183 byte-object census required")
    return objects


def materialize(manifest_ref, objects_dir, output_root):
    """Publish a fresh image-only root and return digest/count-only staging evidence."""
    source, output = Path(objects_dir), Path(output_root)
    require(source.is_absolute() and source.resolve(strict=True) == source and source.is_dir()
            and source.name == "sha256" and source.parent.name == "objects",
            "Canonical source objects/sha256 directory required")
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and output.parent.is_dir()
            and output.resolve() == output and not output.exists() and not output.is_symlink()
            and not output.is_relative_to(source.parent) and not source.is_relative_to(output),
            "Fresh canonical image-only output outside the object directory required")
    objects = manifest_objects(manifest_ref)
    with TemporaryDirectory(dir=output.parent, prefix=".dev187-stage-") as temporary:
        staging = Path(temporary)
        require(staging.resolve(strict=True) == staging and staging.parent == output.parent,
                "Canonical sibling temporary directory required")
        valid = staging / "valid"
        valid.mkdir()
        expected = {}
        for digest, (size, _, aliases) in objects.items():
            raw = checked_bytes(source / (digest + ".jpg"), size, digest)
            for name in aliases:
                path = valid / name
                with path.open("xb") as stream:
                    require(stream.write(raw) == size, "Complete logical image copy required")
                    stream.flush()
                    os.fsync(stream.fileno())
                expected[name] = size
        require({path.name: path.stat().st_size for path in valid.iterdir()} == expected
                and not output.exists() and not output.is_symlink(),
                "Exact complete image-only output and fresh destination required")
        staging.rename(output)
    return dict(status=STATUS, manifest_sha256=MANIFEST_SHA, manifest_bytes=MANIFEST_BYTES,
                images=DEV_IMAGES, distinct_jpeg_sha256=UNIQUE_IMAGES, image_aliases=DEV_IMAGES - UNIQUE_IMAGES,
                unique_encoded_bytes=UNIQUE_BYTES, logical_encoded_bytes=LOGICAL_BYTES,
                FIT=False, DEV=True, TEST=False, GT_read=False, AP=False, model_execution=False)


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-bytes", type=int, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    ref = dict(path=args.manifest.as_posix(), bytes=args.manifest_bytes, sha256=args.manifest_sha256)
    print(json.dumps(materialize(ref, args.objects, args.output_root), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
