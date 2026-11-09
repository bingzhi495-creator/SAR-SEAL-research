"""Materialize only the independently closed SSDD FIT archive into a fresh layout."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ssdd_phase_init import RESEARCH, load_api, require


BUILDER = RESEARCH / "RFDETR_SSDD_NATIVE_INPUT_LAYOUT_DRAFT_20261107_01/build_native_view.py"
BUILDER_SHA = "f6dd54b414bc7661bf6b791268d2dc6f278f01f76417c66646f2ebdedc4b0ad1"
FIT_ARCHIVE_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
FIT_ARCHIVE_BYTES = 33079338
FIT_INDEX_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def stage_fit_layout(archive, destination):
    require(destination.is_absolute() and destination.parent.resolve(strict=True) == destination.parent
            and not destination.exists() and not destination.is_symlink(),
            "Expected a fresh canonical SSDD FIT-only destination")
    require(archive.is_absolute() and archive.resolve(strict=True) == archive
            and archive.is_file() and archive.stat().st_size == FIT_ARCHIVE_BYTES,
            "Expected the canonical closed SSDD FIT archive")
    builder = load_api(BUILDER, BUILDER_SHA)
    destination.mkdir()
    train = destination / "train"
    summary, _ = builder.assemble_split("train", archive, train)
    ids = summary["image_ids"]
    require(list(destination.iterdir()) == [train] and summary["images"] == len(ids) == len(set(ids)) == 741
            and all(type(image_id) is int and image_id >= 0 for image_id in ids)
            and summary["instances"] == 1661 and summary["unique_encoded_images"] == 730
            and summary["category_ids"] == [1] and summary["annotation_bytes"] == 498760
            and summary["annotation_sha256"] == FIT_INDEX_SHA,
            "Exact FIT741 annotations and isolated training layout required")
    return dict(status="SSDD_FIT741_ONLY_LAYOUT_COMPLETE_PENDING_INDEPENDENT_READBACK",
                fit_archive_sha256=FIT_ARCHIVE_SHA, fit_archive_bytes=FIT_ARCHIVE_BYTES,
                fit_index_sha256=FIT_INDEX_SHA, layout_builder_sha256=BUILDER_SHA,
                original_annotation_order_sha256=sha256(encode(ids)).hexdigest(),
                native_sorted_id_order_sha256=sha256(encode(sorted(ids))).hexdigest(),
                images=741, annotations=1661, unique_encoded_images=730, FIT=True, DEV=False, TEST=False,
                model_executed=False, training=False, AP=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(args.receipt.is_absolute() and args.receipt.parent.resolve(strict=True) == args.receipt.parent
            and not args.receipt.exists() and not args.receipt.is_symlink()
            and args.receipt != args.output and args.output not in args.receipt.parents,
            "Expected a fresh receipt outside the isolated FIT layout")
    result = stage_fit_layout(args.archive, args.output)
    with args.receipt.open("xb") as stream:
        stream.write(encode(result) + b"\n")


if __name__ == "__main__":
    main()

