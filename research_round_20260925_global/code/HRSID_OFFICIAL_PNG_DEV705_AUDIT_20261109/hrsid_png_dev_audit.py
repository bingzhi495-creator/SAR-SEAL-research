"""Audit locked HRSID DEV PNGs and preserve their exact original TRAIN annotations."""
import argparse
from collections import Counter
from functools import cache
from hashlib import sha256
import json
from os import fstat
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile


FIT_SOURCE = Path(__file__).resolve().parent.parent / (
    "HRSID_OFFICIAL_PNG_FIT_AUDIT_CANDIDATE_20261109_01/hrsid_png_fit_audit.py"
)
FIT_SOURCE_SHA = "eb5f91a333ea7c7a1c12a87574d0c192f678aa80a742107a510654e3d5512df5"
DEV_COUNTS = (705, 1782)
RUNTIME = {"python": "3.10.13", "Pillow": "12.3.0", "numpy": "1.26.4", "pycocotools": "2.0.11"}


@cache
def fit_tools():
    source = FIT_SOURCE.read_bytes()
    if sha256(source).hexdigest() != FIT_SOURCE_SHA:
        raise ValueError("Prior official FIT audit source must be byte-exact")
    module = ModuleType("_hrsid_closed_fit_png_audit")
    module.__file__ = str(FIT_SOURCE)
    exec(compile(source, str(FIT_SOURCE), "exec"), module.__dict__)
    return module


def audit_dev(archive, assignment, *, expected=None, train_sha=None, fit_fingerprints=None):
    """Return report, image-only manifest, unchanged DEV GT subset and raster findings."""
    fit = fit_tools()
    expected = fit.COUNTS if expected is None else expected
    members = archive.namelist()
    fit.need(len(members) == len(set(members)) and not fit.ImageFile.LOAD_TRUNCATED_IMAGES,
             "Unique archive names and strict PNG decoder")
    train_raw = fit.checked_member(archive, fit.TRAIN_MEMBER, fit.TRAIN_LIMIT)
    actual_train_sha = sha256(train_raw).hexdigest()
    fit.need(train_sha is None or actual_train_sha == train_sha, "Exact official TRAIN bytes")
    train = json.loads(train_raw)
    _, fit_ids, _ = fit.fit_roster(assignment, train, expected)
    dev_images = [image for image in train["images"] if image["id"] not in fit_ids]
    dev_ids = {image["id"] for image in dev_images}
    annotations = [row for row in train["annotations"] if row["image_id"] in dev_ids]
    fit.need((len(dev_images), len(annotations)) == (expected[0] - expected[2], expected[1] - expected[3])
             and len({row["id"] for row in train["annotations"]}) == expected[1]
             and all(type(row["id"]) is int for row in annotations),
             "Exact complementary DEV image population and unique original annotation IDs")

    manifest = []
    for image in dev_images:
        fit.need((image["width"], image["height"]) == (fit.SIDE, fit.SIDE),
                 "Original DEV annotation dimensions")
        member = f"images/{fit.image_stem(image['file_name'])}.png"
        info = archive.getinfo(member)
        raw = fit.checked_member(archive, member, fit.PNG_LIMIT)
        manifest.append(dict(
            image_id=image["id"], file_name=image["file_name"], member=member,
            width=fit.SIDE, height=fit.SIDE, bytes=len(raw), crc32=f"{info.CRC:08x}",
            encoded_sha256=sha256(raw).hexdigest(), decoded_rgb_sha256=fit.decode_rgb(raw),
        ))

    findings = []
    fractional = outside = multicomponent = crowd = 0
    for annotation in annotations:
        pixels, outside_coords, components, crowd_flag = fit.raster_instance(annotation)
        fractional += any(value != int(value) for polygon in annotation["segmentation"] for value in polygon)
        outside += outside_coords > 0
        multicomponent += components > 1
        crowd += crowd_flag
        if pixels == 0:
            findings.append(dict(image_id=annotation["image_id"], annotation_id=annotation["id"],
                                 kind="EMPTY_COCO_RASTER"))

    fingerprints = {
        field: Counter(row[field] for row in manifest)
        for field in ("encoded_sha256", "decoded_rgb_sha256")
    }
    cross_fit = None if fit_fingerprints is None else {
        field: sum(count for digest, count in counts.items() if digest in fit_fingerprints[field])
        for field, counts in fingerprints.items()
    }
    gt = dict(train, images=dev_images, annotations=annotations)
    report = dict(
        status="DEV_INPUT_AUDIT_COMPLETE", full_dev_input_audit=True, original_gt_preserved=True,
        original_train_sha256=actual_train_sha, opened_annotation_member=fit.TRAIN_MEMBER,
        decoded_dev_images=len(manifest), member_crc_checked_images=len(manifest),
        attempted_dev_rasters=len(annotations), empty_dev_rasters=len(findings),
        nonempty_dev_rasters=len(annotations) - len(findings), fractional_coordinate_annotations=fractional,
        out_of_canvas_annotations=outside, multicomponent_annotations=multicomponent, crowd_annotations=crowd,
        fit_proxy_groups=expected[4], dev_proxy_groups=expected[5],
        dev_duplicate_counts={field: fit.fingerprint_counts(counts) for field, counts in fingerprints.items()},
        dev_images_matching_fit_fingerprints=cross_fit,
        scene_independence_proven=False, fit_png_member_opened=False, official_test_member_opened=False,
        model_executed=False, AP=False,
    )
    return report, manifest, gt, findings


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "receipt", "split-manifest", "assignment", "fit-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fit-manifest-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    fit = fit_tools()
    start = perf_counter()
    fit.need(fit.runtime_versions() == RUNTIME, "Pinned CPU PNG and COCO runtime")
    fit.need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol))
             and bool(re.fullmatch(r"[0-9a-f]{64}", args.fit_manifest_sha256)),
             "Prelocked protocol commit and independently closed FIT manifest SHA-256")
    fit.need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
             and not args.output.exists() and not args.output.is_symlink(), "Fresh isolated output path")
    receipt = fit.checked_json(args.receipt, fit.RECEIPT_SHA, 1024**2)
    fit.need(receipt["archive_sha256"] == fit.ARCHIVE_SHA and receipt["archive_bytes"] == fit.ARCHIVE_BYTES
             and receipt["expected_central_directory_sha256"] == fit.CENTRAL_SHA
             and receipt["all_member_CRC_verified"] is True, "Closed official PNG archive receipt")
    split = fit.checked_json(args.split_manifest, fit.SPLIT_SHA, 1024**2)
    hrsid = split["datasets"]["hrsid"]
    fit.need(split["split_selection_locked_before_model_results"] is True and hrsid["status"] == "frozen"
             and hrsid["image_group_assignment"]["sha256"] == fit.ASSIGNMENT_SHA
             and hrsid["splits"]["fit"]["sha256"] == fit.FIT_SHA, "Closed FIT/DEV split identity")
    roles = fit.checked_json(args.assignment, fit.ASSIGNMENT_SHA, 2 * 1024**2)
    fit_manifest = fit.checked_json(args.fit_manifest, args.fit_manifest_sha256, 4 * 1024**2)
    fit_rows = fit_manifest["images"]
    fit_ids = {row["image_id"] for row in roles if row["split"] == "fit"}
    fit.need(len(fit_rows) == len(fit_ids) == fit.COUNTS[2]
             and {row["image_id"] for row in fit_rows} == fit_ids
             and fit_manifest["archive_sha256_receipt"] == fit.ARCHIVE_SHA
             and fit_manifest["archive_central_sha256"] == fit.CENTRAL_SHA, "Exact closed FIT PNG manifest")
    fingerprints = {
        field: {row[field] for row in fit_rows}
        for field in ("encoded_sha256", "decoded_rgb_sha256")
    }
    fit.need(all(len(digests) == len(fit_ids) and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in digests)
                 for digests in fingerprints.values()), "Closed FIT PNG digest coverage")

    identity = fit.archive_identity(args.archive.stat())
    fit.need(args.archive.is_absolute() and args.archive.resolve(strict=True) == args.archive
             and identity[2] == fit.ARCHIVE_BYTES, "Exact canonical closed archive identity")
    args.output.mkdir()
    try:
        with args.archive.open("rb") as handle:
            fit.need(fit.archive_identity(fstat(handle.fileno())) == identity, "Closed archive handle")
            handle.seek(fit.CENTRAL_OFFSET)
            fit.need(sha256(handle.read(fit.CENTRAL_BYTES)).hexdigest() == fit.CENTRAL_SHA,
                     "Exact archive central-directory bytes")
            with ZipFile(handle) as archive:
                fit.need(len(archive.infolist()) == 5612, "Exact official central-directory inventory")
                report, images, gt, findings = audit_dev(
                    archive, roles, train_sha=fit.TRAIN_SHA, fit_fingerprints=fingerprints,
                )
            fit.need(fit.archive_identity(fstat(handle.fileno())) == identity, "Open archive identity unchanged")
        fit.need(fit.archive_identity(args.archive.stat()) == identity
                 and (len(images), len(gt["annotations"])) == DEV_COUNTS, "Closed complete DEV705")
        report.update(
            image_manifest_sha256=write_json(args.output / "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json", images),
            original_dev_gt_sha256=write_json(args.output / "DEV_ORIGINAL_GT_PRIVATE.json", gt),
            empty_raster_findings_sha256=write_json(args.output / "DEV_EMPTY_RASTERS_PRIVATE.json", findings),
        )
        if any(report["dev_images_matching_fit_fingerprints"].values()):
            report["status"] = "DEV_INPUT_AUDIT_COMPLETE_WITH_EXACT_FIT_OVERLAP"
    except Exception as error:
        report = dict(status="DEV_INPUT_AUDIT_FAILED", full_dev_input_audit=False,
                      error_type=type(error).__name__, model_executed=False, AP=False)
        write_json(args.output / "FAILURE_PRIVATE.json", {"detail": str(error)})
    report.update(protocol=args.protocol, runtime=RUNTIME, official_archive_receipt_sha256=fit.ARCHIVE_SHA,
                  central_directory_sha256=fit.CENTRAL_SHA, assignment_sha256=fit.ASSIGNMENT_SHA,
                  closed_fit_manifest_sha256=args.fit_manifest_sha256, seconds=perf_counter() - start)
    write_json(args.output / "DEV_INPUT_AUDIT_SAFE.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["status"] == "DEV_INPUT_AUDIT_COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())

