"""Receive the closed HRSID FIT2937 native layout from an uncompressed USTAR stream."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from shutil import disk_usage
import sys
from tarfile import AREGTYPE, BLOCKSIZE, DIRTYPE, HeaderError, RECORDSIZE, REGTYPE, TarInfo
from tempfile import TemporaryDirectory


MANIFEST_NAME = "NATIVE_FIT_LAYOUT_MANIFEST_PRIVATE.json"
MANIFEST_BYTES = 1386385
MANIFEST_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
INDEX_NAME = "train/_annotations.coco.json"
INDEX_BYTES = 30076663
INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
FIT_IMAGES, FIT_ANNOTATIONS, PNG_BYTES = 2937, 9260, 3698804355
ORIGINAL_ARCHIVE_BYTES = 5698302315
ORIGINAL_ARCHIVE_SHA = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
ORIGINAL_CENTRAL_SHA = "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce"
FIT_SOURCE_INDEX_SHA = "c7629de9afaa398d0b943fba8e4cec75c8b65a9864fd795c1465c9b4dec3c271"
STATUS = "HRSID_FIT2937_DGX_STREAM_BYTES_VERIFIED_V1"
MIB = 1024**2
ZERO_BLOCK = bytes(BLOCKSIZE)
MANIFEST_FIELDS = {"archive_bytes", "archive_central_sha256", "archive_sha256_receipt", "fit_source_index_sha256",
                   "images", "native_index_bytes", "native_index_sha256", "protocol", "selected_encoded_png_bytes"}
IMAGE_FIELDS = {"compressed_bytes", "decoded_rgb_sha256", "encoded_bytes", "encoded_sha256",
                "image_id", "member", "native_basename", "zip_crc32", "zip_header_offset"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_at_most(stream, count):
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(count - len(result))
        if not chunk:
            break
        result.extend(chunk)
    return bytes(result)


def read_exact(stream, count):
    raw = read_at_most(stream, count)
    require(len(raw) == count, "Incomplete FIT USTAR stream")
    return raw


def read_header(stream):
    raw = read_exact(stream, BLOCKSIZE)
    if raw == ZERO_BLOCK:
        return None
    require(raw[257:263] == b"ustar\0" and not any(raw[345:500]),
            "Expected plain USTAR header without long-name extensions")
    try:
        info = TarInfo.frombuf(raw, "ascii", "strict")
        original_name = raw[:100].split(b"\0", 1)[0].decode("ascii")
    except (HeaderError, UnicodeError) as error:
        raise ValueError("Invalid FIT USTAR header") from error
    require(not info.linkname and (info.type in (REGTYPE, AREGTYPE) and original_name == info.name
                                   or info.type == DIRTYPE and original_name in ("train", "train/")
                                   and info.name == "train" and info.size == 0),
            "Only original FIT regular files and one train directory are admitted")
    return info


def read_padding(stream, size):
    require(not any(read_exact(stream, (-size) % BLOCKSIZE)), "Nonzero FIT USTAR member padding")


def end_of_stream(stream):
    require(read_exact(stream, BLOCKSIZE) == ZERO_BLOCK, "Expected two zero USTAR terminator blocks")
    tail = read_at_most(stream, RECORDSIZE + 1)
    require(len(tail) <= RECORDSIZE and len(tail) % BLOCKSIZE == 0 and not any(tail),
            "Unexpected data after the FIT USTAR terminator")


def manifest_roster(raw):
    require(len(raw) == MANIFEST_BYTES and sha256(raw).hexdigest() == MANIFEST_SHA,
            "Exact original FIT2937 manifest bytes required")
    document = json.loads(raw)
    require(type(document) is dict and document.keys() == MANIFEST_FIELDS
            and document["archive_bytes"] == ORIGINAL_ARCHIVE_BYTES
            and document["archive_central_sha256"] == ORIGINAL_CENTRAL_SHA
            and document["archive_sha256_receipt"] == ORIGINAL_ARCHIVE_SHA
            and document["fit_source_index_sha256"] == FIT_SOURCE_INDEX_SHA
            and document["native_index_bytes"] == INDEX_BYTES and document["native_index_sha256"] == INDEX_SHA
            and document["selected_encoded_png_bytes"] == PNG_BYTES and type(document["images"]) is list
            and len(document["images"]) == FIT_IMAGES, "Fixed HRSID FIT2937 native source identity required")
    roster = {INDEX_NAME: (INDEX_BYTES, INDEX_SHA)}
    names, ids, digests = set(), set(), set()
    for image in document["images"]:
        require(type(image) is dict and image.keys() == IMAGE_FIELDS, "Exact original FIT PNG manifest row required")
        name, iid = image["native_basename"], image["image_id"]
        size, digest = image["encoded_bytes"], image["encoded_sha256"]
        require(type(name) is str and re.fullmatch(r"P[0-9]+_[A-Za-z0-9_]+\.png", name)
                and image["member"] == "images/" + name and name.casefold() not in names
                and type(iid) is int and iid >= 0 and iid not in ids and type(size) is int and 0 < size <= 8 * MIB
                and type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) and digest not in digests,
                "Unique exact FIT PNG names, IDs, sizes and content digests required")
        roster["train/" + name] = size, digest
        names.add(name.casefold())
        ids.add(iid)
        digests.add(digest)
    require(sum(size for size, _ in roster.values()) == PNG_BYTES + INDEX_BYTES, "Exact FIT byte census required")
    return roster


def write_member(stream, path, size, expected_digest):
    hasher = sha256()
    remaining = size
    with path.open("xb") as target:
        while remaining:
            block = read_exact(stream, min(remaining, MIB))
            hasher.update(block)
            require(target.write(block) == len(block), "Incomplete FIT file write")
            remaining -= len(block)
        require(hasher.hexdigest() == expected_digest, "FIT member SHA-256 mismatch")
        target.flush()
        os.fsync(target.fileno())
    require(path.stat().st_size == size, "FIT staged byte length mismatch")
    read_padding(stream, size)


def manifest_sidecar(output):
    return output.with_name(output.name + ".manifest.private.json")


def fresh(path):
    return path.is_absolute() and path.parent.resolve(strict=True) == path.parent and path.parent.is_dir()         and path.resolve() == path and not path.exists() and not path.is_symlink()


def receive(stream, output):
    """Publish the verified train-only root after the entire stream closes."""
    output = Path(output)
    sidecar = manifest_sidecar(output)
    require(fresh(output) and fresh(sidecar), "Fresh canonical FIT directory and manifest sibling required")
    require(disk_usage(output.parent).free >= PNG_BYTES + INDEX_BYTES + MANIFEST_BYTES + 64 * MIB,
            "Insufficient disk for the complete FIT layout and safety reserve")
    header = read_header(stream)
    require(header is not None and header.name == MANIFEST_NAME and header.type in (REGTYPE, AREGTYPE)
            and header.size == MANIFEST_BYTES, "Expected the exact original manifest as first FIT USTAR member")
    manifest = read_exact(stream, MANIFEST_BYTES)
    roster = manifest_roster(manifest)
    read_padding(stream, MANIFEST_BYTES)
    folder = read_header(stream)
    require(folder is not None and folder.type == DIRTYPE and folder.name == "train",
            "Expected exactly one train directory after the FIT manifest")

    with TemporaryDirectory(dir=output.parent, prefix=".fit2937-stage-") as temporary:
        scratch = Path(temporary)
        require(scratch.parent == output.parent and scratch.resolve(strict=True) == scratch,
                "Canonical sibling FIT staging directory required")
        payload = scratch / "layout"
        train = payload / "train"
        train.mkdir(parents=True)
        names = {name.split("/", 1)[1] for name in roster}
        while (header := read_header(stream)) is not None:
            pair = roster.pop(header.name, None)
            require(header.type in (REGTYPE, AREGTYPE) and pair is not None and header.size == pair[0],
                    "Unexpected, duplicate or wrong-sized FIT USTAR member")
            write_member(stream, train / header.name.split("/", 1)[1], *pair)
        end_of_stream(stream)
        require(not roster and {path.name for path in train.iterdir()} == names and fresh(output) and fresh(sidecar),
                "Incomplete FIT roster or occupied output")
        staged_manifest = scratch / MANIFEST_NAME
        with staged_manifest.open("xb") as target:
            require(target.write(manifest) == MANIFEST_BYTES, "Incomplete original FIT manifest write")
            target.flush()
            os.fsync(target.fileno())
        os.link(staged_manifest, sidecar)
        try:
            require(fresh(output), "FIT destination became occupied")
            payload.rename(output)
        except BaseException:
            if sidecar.exists() and sidecar.samefile(staged_manifest):
                sidecar.unlink()
            raise
    return dict(status=STATUS, transport="raw_ustar", manifest_sha256=MANIFEST_SHA,
                native_index_sha256=INDEX_SHA, images=FIT_IMAGES, annotations=FIT_ANNOTATIONS,
                png_bytes=PNG_BYTES, native_index_bytes=INDEX_BYTES, payload_bytes=PNG_BYTES + INDEX_BYTES,
                manifest_bytes=MANIFEST_BYTES, FIT=True, DEV=False, TEST=False, AP=False, model_executed=False)


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(receive(sys.stdin.buffer, args.output), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
