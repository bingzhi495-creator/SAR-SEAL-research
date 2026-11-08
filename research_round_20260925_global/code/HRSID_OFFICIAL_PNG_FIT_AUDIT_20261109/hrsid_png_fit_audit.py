"""Check the locked FIT subset against original polygons and official HRSID PNG pixels."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import math
from os import fstat
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

import numpy as np
from PIL import Image, ImageFile
from pycocotools import mask

ARCHIVE_BYTES = 5698302315
ARCHIVE_SHA = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
CENTRAL_OFFSET = 5697629273
CENTRAL_BYTES = 672944
CENTRAL_SHA = "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce"
RECEIPT_SHA = "4de89dff4cbf13a856e714b8952503a7e11ed2c9ef4405d72b06acdb425d52db"
SPLIT_SHA = "80bfd66f3f11270339ee53a1176468215c33abe908000b7a27316bec53671355"
ASSIGNMENT_SHA = "c92b85dc278bc49cc8e34634fd7d354018844ae2a2692ea451b014a082bbaa5a"
FIT_SHA = "d81e541811af9e1e0483a96f5a09efb91a7e67e3f761a58dc9ba4d1ba26549b8"
TRAIN_SHA = "3e240c38014526e9b93206a946105fb41c808272238932b0d4c2422978d5116f"
TRAIN_MEMBER = "annotations/train2017.json"
COUNTS = (3642, 11047, 2937, 9265, 109, 28)
SIDE = 800
PNG_LIMIT = 8 * 1024**2
TRAIN_LIMIT = 24 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def checked_json(path, digest, limit):
    need(path.is_file() and 0 < path.stat().st_size <= limit, "Pinned local metadata size")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Pinned local metadata SHA-256")
    return json.loads(raw)


def checked_member(z, member, limit):
    info = z.getinfo(member)
    need(0 < info.file_size <= limit, "Pinned ZIP member size")
    raw = z.read(info)  # ZipFile verifies CRC after complete payload read.
    need(len(raw) == info.file_size, "Complete ZIP member read")
    return raw


def image_stem(value):
    name = PurePosixPath(value)
    need(name.name == value and name.suffix.lower() in {".jpg", ".png"}
         and re.fullmatch(r"P\d+_[A-Za-z0-9_]+", name.stem), "Original image name and P prefix")
    return name.stem


def fit_roster(assignment, train, expected):
    train_n, ann_n, fit_n, fit_ann_n, fit_groups_n, dev_groups_n = expected
    need(len(assignment) == len(train["images"]) == train_n and len(train["annotations"]) == ann_n,
         "Exact official TRAIN and locked roster sizes")
    need([c["id"] for c in train["categories"]] == [1], "Original single ship category")
    images = {r["id"]: r for r in train["images"]}
    need(len(images) == train_n and all(type(k) is int for k in images), "Unique original TRAIN image IDs")
    roles = {}
    groups = defaultdict(set)
    stems = set()
    for r in assignment:
        image_id = r["image_id"]
        stem = image_stem(r["file_name"])
        need(type(image_id) is int and image_id in images and image_id not in roles and stem not in stems
             and r["split"] in {"fit", "dev"} and r["group"] == stem.split("_", 1)[0]
             and image_stem(images[image_id]["file_name"]) == stem, "Locked role, image ID and stem identity")
        roles[image_id] = (r["split"], stem)
        groups[r["split"]].add(r["group"])
        stems.add(stem)
    need(len(groups["fit"]) == fit_groups_n and len(groups["dev"]) == dev_groups_n
         and not (groups["fit"] & groups["dev"]), "Disjoint locked P-prefix proxy groups")
    fit = {i: s for i, (r, s) in roles.items() if r == "fit"}
    need(len(fit) == fit_n, "Exact FIT population")
    anns = {i: [] for i in fit}
    ann_ids = set()
    for a in train["annotations"]:
        image_id = a["image_id"]
        need(type(image_id) is int and image_id in roles, "TRAIN annotation/image association")
        if image_id in fit:
            need(type(a["id"]) is int and a["id"] not in ann_ids, "Unique original FIT annotation ID")
            ann_ids.add(a["id"])
            anns[image_id].append(a)
    need(len(ann_ids) == fit_ann_n, "Exact FIT original instance count")
    return images, fit, anns


def decode_rgb(raw):
    need(raw.startswith(b"\x89PNG\r\n\x1a\n") and raw[12:16] == b"IHDR"
         and raw[24:26] == b"\x08\x02", "Original PNG IHDR RGB8 encoding")
    with Image.open(BytesIO(raw)) as im:
        need(im.format == "PNG" and im.mode == "RGB" and im.getbands() == ("R", "G", "B")
             and im.size == (SIDE, SIDE) and im.n_frames == 1, "Original PNG RGB800 storage shape")
        orientation = im.getexif().get(274, 1)
        need(type(orientation) is int and orientation == 1, "Original pixel orientation")
        im.load()
        array = np.asarray(im)
        need(array.dtype == np.uint8 and array.shape == (SIDE, SIDE, 3), "Actual RGB8 PNG decode shape")
        return sha256(array.tobytes(order="C")).hexdigest()


def finite_number(v):
    return type(v) in (int, float) and math.isfinite(v)


def raster_instance(a):
    polygons = a["segmentation"]
    area, bbox, crowd = a["area"], a["bbox"], a["iscrowd"]
    need(type(a["category_id"]) is int and a["category_id"] == 1
         and type(crowd) is int and crowd in (0, 1), "Original category and crowd flags")
    need(finite_number(area) and area >= 0 and isinstance(bbox, list) and len(bbox) == 4
         and all(finite_number(v) for v in bbox) and bbox[2] > 0 and bbox[3] > 0,
         "Original continuous area and bbox")
    need(isinstance(polygons, list) and polygons and all(isinstance(p, list) and len(p) >= 6
         and len(p) % 2 == 0 and all(finite_number(v) for v in p) for p in polygons),
         "Original finite COCO polygon components")
    outside = sum(v < 0 or v > SIDE for p in polygons for v in p)
    rles = mask.frPyObjects(polygons, SIDE, SIDE)
    need(isinstance(rles, list) and len(rles) == len(polygons), "Official per-component COCO RLE")
    rle = mask.merge(rles)
    bitmap = mask.decode(rle)
    need(rle["size"] == [SIDE, SIDE] and bitmap.dtype == np.uint8 and bitmap.shape == (SIDE, SIDE)
         and int(bitmap.max()) <= 1, "Actual full-canvas binary raster")
    pixels = int(bitmap.sum(dtype=np.int64))
    need(int(mask.area(rle)) == pixels, "COCO RLE area equals decoded sum")
    if pixels:
        x, y, w, h = mask.toBbox(rle)
        need(0 <= x < x + w <= SIDE and 0 <= y < y + h <= SIDE, "Nonempty raster bounds in actual canvas")
    return pixels, outside, len(polygons), crowd


def fingerprint_counts(counts):
    return {
        "unique_fingerprints": len(counts),
        "duplicate_groups": sum(n > 1 for n in counts.values()),
        "images_in_duplicate_groups": sum(n for n in counts.values() if n > 1),
        "excess_duplicate_images": sum(n - 1 for n in counts.values()),
    }


def audit_fit(z, assignment, *, expected=COUNTS, train_sha=None, diagnostics=None, progress=None):
    need(ImageFile.LOAD_TRUNCATED_IMAGES is False, "Truncated image acceptance must remain disabled")
    members = z.namelist()
    need(len(members) == len(set(members)), "Unique ZIP member names")
    train_raw = checked_member(z, TRAIN_MEMBER, TRAIN_LIMIT)
    actual_sha = sha256(train_raw).hexdigest()
    need(train_sha is None or actual_sha == train_sha, "Exact original TRAIN polygon bytes")
    train = json.loads(train_raw)
    images, fit, anns = fit_roster(assignment, train, expected)
    del train_raw, train
    errors = [] if diagnostics is None else diagnostics
    encoded, decoded = Counter(), Counter()
    image_errors, raster_errors, empty, nonempty, attempted = 0, 0, 0, 0, 0
    outside_anns, outside_coordinates, multi, crowd = 0, 0, 0, 0
    zero_labels = sum(not rows for rows in anns.values())
    no_nonempty_labels = 0
    nonempty_density, pixel_areas = [], []
    for visited, (image_id, stem) in enumerate(fit.items(), 1):
        member = "images/" + stem + ".png"
        record = images[image_id]
        raw = None
        try:
            need(type(record["width"]) is int and type(record["height"]) is int
                 and (record["width"], record["height"]) == (SIDE, SIDE), "Original annotation canvas dimensions")
            raw = checked_member(z, member, PNG_LIMIT)
            encoded[sha256(raw).hexdigest()] += 1
            decoded[decode_rgb(raw)] += 1
        except (KeyError, OSError, ValueError, SyntaxError, BadZipFile, ZlibError, Image.DecompressionBombError) as error:
            image_errors += 1
            errors.append({"image_id": image_id, "kind": "IMAGE_CHECK_FAILED", "detail": str(error)})
            continue
        finally:
            del raw
        positives = 0
        for a in anns[image_id]:
            attempted += 1
            try:
                pixels, outside, components, iscrowd = raster_instance(a)
                outside_anns += outside > 0
                outside_coordinates += outside
                multi += components > 1
                crowd += iscrowd
                if pixels:
                    nonempty += 1
                    positives += 1
                    pixel_areas.append(pixels)
                else:
                    empty += 1
                    errors.append({"image_id": image_id, "annotation_id": a["id"], "kind": "EMPTY_COCO_RASTER"})
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raster_errors += 1
                errors.append({"image_id": image_id, "annotation_id": a["id"],
                               "kind": "ANNOTATION_CHECK_FAILED", "detail": str(error)})
        no_nonempty_labels += positives == 0
        nonempty_density.append(positives)
        if progress is not None and visited % 250 == 0:
            progress(visited, attempted)
    decoded_count = sum(decoded.values())
    full = decoded_count == expected[2] and attempted == expected[3]
    strict = full and image_errors == raster_errors == empty == 0
    status = ("COMPLETED_FULL_FIT_NO_STRUCTURAL_FAILURE" if strict else
              "COMPLETED_FULL_FIT_WITH_EMPTY_RASTERS" if full and not raster_errors else
              "COMPLETED_FIT_INVENTORY_WITH_CHECK_FAILURES")
    return {
        "status": status, "strict_input_qualified": strict, "full_fit_image_and_raster_attempt_coverage": full,
        "train_annotation_sha256": actual_sha, "official_train_member_opened": TRAIN_MEMBER,
        "fit_images_expected": expected[2], "original_fit_annotations_expected": expected[3],
        "fit_proxy_groups": expected[4], "excluded_internal_dev_proxy_groups": expected[5],
        "decoded_fit_images": decoded_count, "image_failures": image_errors,
        "expected_image_shape": [SIDE, SIDE, 3], "decoded_dtype": "uint8",
        "attempted_fit_rasters": attempted, "nonempty_fit_rasters": nonempty,
        "empty_fit_rasters": empty, "fit_raster_check_failures_excluding_empty": raster_errors,
        "fit_rasters_not_attempted": expected[3] - attempted,
        "fit_original_zero_annotation_images": zero_labels, "fit_images_without_nonempty_instance": no_nonempty_labels,
        "fit_multi_component_rastered_annotations": multi, "fit_crowd_rastered_annotations": crowd,
        "fit_annotations_with_outside_coordinates": outside_anns, "fit_outside_coordinates": outside_coordinates,
        "fit_min_nonempty_raster_pixel_area": min(pixel_areas, default=None),
        "fit_max_nonempty_raster_pixel_area": max(pixel_areas, default=None),
        "fit_max_nonempty_instances_per_decoded_image": max(nonempty_density, default=0),
        "fit_decoded_images_above_100_nonempty_instances": sum(n > 100 for n in nonempty_density),
        "encoded_fit_png": fingerprint_counts(encoded), "decoded_fit_RGB8": fingerprint_counts(decoded),
        "separate_DEV_or_TEST_annotation_member_opened": False, "DEV_or_TEST_image_member_opened": False,
        "annotation_scope": "One original official TRAIN member includes FIT and internal DEV metadata. FIT geometry only.",
        "semantic_alignment_proven": False, "parent_scene_independence_proven": False,
        "model_executed": False, "AP": False, "formal_dataset_or_TEST_qualification": False,
    }


def archive_identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def runtime_versions():
    return {"python": ".".join(map(str, sys.version_info[:3])),
            "Pillow": version("Pillow"), "numpy": np.__version__, "pycocotools": version("pycocotools")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    start = perf_counter()
    rv = runtime_versions()
    need(rv == {"python": "3.10.13", "Pillow": "12.3.0", "numpy": "1.26.4", "pycocotools": "2.0.11"},
         "Pinned CPU image/raster environment")
    receipt = checked_json(args.receipt, RECEIPT_SHA, 1024**2)
    need(receipt["archive_sha256"] == ARCHIVE_SHA and receipt["archive_bytes"] == ARCHIVE_BYTES
         and receipt["expected_central_directory_sha256"] == CENTRAL_SHA and receipt["all_member_CRC_verified"] is True,
         "Closed official PNG archive identity")
    split = checked_json(args.split_manifest, SPLIT_SHA, 1024**2)
    hrsid = split["datasets"]["hrsid"]
    need(split["split_selection_locked_before_model_results"] is True and hrsid["status"] == "frozen"
         and hrsid["image_group_assignment"]["sha256"] == ASSIGNMENT_SHA
         and hrsid["splits"]["fit"]["sha256"] == FIT_SHA, "Closed locked split identity")
    roster = checked_json(args.assignment, ASSIGNMENT_SHA, 2 * 1024**2)
    identity = archive_identity(args.archive.stat())
    need(identity[2] == ARCHIVE_BYTES, "Exact closed ZIP length")
    args.output.mkdir(parents=False, exist_ok=False)
    diagnostics = []
    try:
        with args.archive.open("rb") as handle:
            need(archive_identity(fstat(handle.fileno())) == identity, "Same closed ZIP handle identity")
            handle.seek(CENTRAL_OFFSET)
            need(sha256(handle.read(CENTRAL_BYTES)).hexdigest() == CENTRAL_SHA, "Exact official ZIP central directory")
            with ZipFile(handle) as z:
                need(len(z.infolist()) == 5612, "Exact known official central-directory inventory")
                report = audit_fit(z, roster, train_sha=TRAIN_SHA, diagnostics=diagnostics, progress=lambda n, a:
                    print(json.dumps({"event": "FIT_PROGRESS", "visited_images": n, "raster_attempts": a}), flush=True))
        need(archive_identity(args.archive.stat()) == identity, "Archive identity stable during selected reads")
    except Exception as error:
        diagnostics.append({"kind": "FATAL_FIT_AUDIT_GATE", "detail": str(error)})
        report = {"status": "FAILED_FIT_AUDIT_INPUT_OR_EXECUTION_GATE", "error_type": type(error).__name__,
                  "strict_input_qualified": False, "model_executed": False, "AP": False}
    report.update(runtime=rv, verified_historical_archive_sha256=ARCHIVE_SHA, central_directory_sha256=CENTRAL_SHA,
                  closed_archive_receipt_sha256=RECEIPT_SHA, locked_split_manifest_sha256=SPLIT_SHA,
                  locked_assignment_sha256=ASSIGNMENT_SHA, seconds=perf_counter() - start)
    for filename, obj in (("FIT_FINDINGS_PRIVATE.json", diagnostics), ("REPORT_SAFE.json", report)):
        with (args.output / filename).open("x", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report.get("full_fit_image_and_raster_attempt_coverage") and not report.get(
        "fit_raster_check_failures_excluding_empty") else 2


if __name__ == "__main__":
    raise SystemExit(main())

