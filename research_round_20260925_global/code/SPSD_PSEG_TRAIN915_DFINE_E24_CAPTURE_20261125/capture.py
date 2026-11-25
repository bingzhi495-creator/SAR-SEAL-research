"""Capture admitted original PSeg TRAIN rectangles with the exact frozen D-FINE author path."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from types import ModuleType

N = 915
MANIFEST = "701c89d7350f6a078337d6074d69960799ade868a4f64e546897e07137c92bf4"
INVENTORY = "2a7aaf3013c77ddd59804a446140153234a4279807ef92f1b3eb3c014534ed58"
INTAKE = "190d206a1b4d3690323d7fbc4db6c0cd6398a1f53f993ab0c3819c3b0396ffde"
STAGER = "9dbffacca9a64067903efa747d4bcef5e1bedc6949ae5f350b8a0715a86ed460"
ADMISSION = "e40636ec2cfb4c42718673a553f069f5f4e2833e3e81264c9d44a43ece7b9927"
ROSTER = "f4e23e5eb7f58a881db48e4a89d2902a253aa945e7a92ab7d1674c8b02b3fb17"
ARCHIVE = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"
UPSTREAM = "11df18cb1eb0dbcff66b576d0c082b45896e5e254adb66b98fdfaff366d7391a"
LOCK = "0565109236be7fd538bca5e35259fbf109a132fcd8fea65208f4c6fdb2350c56"
STATUS = "DFINE_HRSID_E24_PSEG_TRAIN915_PROBABILITY_EXPORT_COMPLETE_V1"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Pinned input: " + path.name)
    return raw


def bind_intake(directory):
    need(directory.is_absolute() and directory.resolve(strict=True) == directory,
         "Canonical closed image-only intake parent")
    term = json.loads((directory.parent / "observed_intake_retry/TERMINAL.json").read_bytes())
    record = json.loads(pinned(directory / "RESULT_SAFE.json", INTAKE))
    need(term["actual_exit"] == 0 and term["output_footer_verified"] is True and term["result_sha256"] == INTAKE
         and term["worker_sha256"] == STAGER and term["phase"] == "prepare-edits"
         and record["status"] == "SPSD_PSEG_TRAIN915_IMAGE_ONLY_READY"
         and record["admission_sha256"] == ADMISSION and record["roster_sha256"] == ROSTER
         and record["source_archive_sha256"] == ARCHIVE
         and record["image_manifest_sha256"] == MANIFEST and record["image_inventory_sha256"] == INVENTORY
         and record["images"] == N and record["source_groups"] == 901
         and all(record[v] == 0 for v in ("exact_cross_test_encoded", "exact_cross_test_decoded",
                    "test_image_members_read", "test_label_members_read", "train_label_members_read",
                    "detector_forwards", "optimizer_updates")), "Closed verified source-image admission")


def manifest_rows(path, manifest_sha, base):
    need(manifest_sha == MANIFEST, "Expected admitted train scope")
    doc = json.loads(pinned(path, MANIFEST))
    rows = doc["items"]
    need(doc["schema"] == "SPSD_PSEG_DUAL_REFERENCE_TRAIN915_IMAGE_ONLY_V1"
         and doc["population"] == "clean_author_train_915" and doc["FIT"] is True
         and doc["DEV"] is doc["TEST"] is doc["GT_opened"] is doc["AP"] is doc["model_execution"] is False
         and doc["source_admission_sha256"] == ADMISSION and doc["source_roster_sha256"] == ROSTER
         and doc["source_archive_sha256"] == ARCHIVE and doc["image_inventory_sha256"] == INVENTORY
         and doc["images"] == len(rows) == len(set(doc["expected_ids"])) == N
         and doc["unique_rgb_groups"] == 901 and doc["cross_test_encoded"] == doc["cross_test_decoded"] == 0
         and [row["image_id"] for row in rows] == doc["expected_ids"], "Whole admitted train manifest")
    names = [row["image_ref"]["path"] for row in rows]
    need(names == sorted(set(names)), "Complete distinct original JPEG order")
    listing = sha256()
    for row, name in zip(rows, names, strict=True):
        ref = row["image_ref"]
        need(re.fullmatch(r"[0-9]{6}\.jpg",name) and int(name[:6]) % 10 not in (1,9)
             and type(row["image_id"]) is int and all(type(x) is int and x > 0 for x in row["original_hw"])
             and len(row["original_hw"]) == 2 and base.digest(ref["sha256"])
             and base.digest(row["rgb_sha256"]) and base.digest(row["rgb_size_prefixed_sha256"])
             and base.digest(row["source_group"])
             and type(ref["bytes"]) is int and 0 < ref["bytes"] <= 4*1024**2,
             "Admitted train native JPEG identification")
        listing.update(f"{name}\t{ref['bytes']}\t{ref['sha256']}\n".encode("ascii"))
    need(listing.hexdigest() == INVENTORY and len({x["source_group"] for x in rows}) == 901,
         "Exact encoded inventory and source grouping")
    return rows


def upstream_capture(source):
    body = pinned(source, UPSTREAM).decode("utf-8")
    chunks = (
        ('args.test_root', 'args.fit_root', 5),
        ('("test-root", "reference-dir",', '("fit-root", "reference-dir",', 1),
        ('"Exactly the 232 JPEGs in the image-only root"', '"Exactly the admitted TRAIN JPEGs in the image-only root"', 1),
        ('"SSDD_TEST232_NATIVE_CAPTURE_PROGRESS"', '"SPSD_PSEG_TRAIN915_NATIVE_CAPTURE_PROGRESS"', 1),
        ('stats["probability_files"] == 232', 'stats["probability_files"] == EXPECTED_N', 1),
        ('images=232, model_forwards=232', 'images=EXPECTED_N, model_forwards=EXPECTED_N', 1),
        ('raw_queries=232*300, internal_top_rows=232*300', 'raw_queries=EXPECTED_N*300, internal_top_rows=EXPECTED_N*300', 1),
        ('native_dimensions_verified=232', 'native_dimensions_verified=EXPECTED_N', 1),
        ('optimizer_updates=0, historical_target_exposure=True, FIT=False, DEV=False, TEST=True, GT_opened=False, AP=False, **stats)',
         'optimizer_updates=0, FIT=True, DEV=False, TEST=False, GT_opened=False, AP=False, upstream_worker_sha256=UPSTREAM_SHA, **stats)', 1),
    )
    for old, new, count in chunks:
        need(body.count(old) == count, "Exact old source population/report seam: " + old)
        body = body.replace(old,new)
    original=ModuleType("_exact_train_rectangular_capture")
    original.__file__=__file__
    original.EXPECTED_N=N
    original.UPSTREAM_SHA=UPSTREAM
    exec(compile(body,str(source),"exec"), original.__dict__)
    original.INVENTORY_SHA=INVENTORY
    original.MANIFEST_SHA=MANIFEST
    original.STATUS=STATUS
    original.manifest_rows=manifest_rows
    return original


def main():
    source=Path(sys.argv.pop(1))
    intake=Path(sys.argv.pop(1))
    bind_intake(intake)
    original=upstream_capture(source)
    read_rgb=original.read_rgb
    run=original.run

    def exact_rgb(root,row,base):
        rgb=read_rgb(root,row,base)
        need(sha256(rgb.tobytes(order="C")).hexdigest()==row["rgb_sha256"], "Exact independently admitted decoded RGB")
        return rgb

    def exact_run(args):
        need(args.manifest == intake / "TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json"
             and args.fit_root == intake / "FIT_IMAGES" and args.protocol_sha == LOCK,
             "Exact new TRAIN image endpoints and bound protocol")
        pinned(args.protocol,LOCK)
        run(args)

    original.read_rgb=exact_rgb
    original.run=exact_run
    original.main()


if __name__=="__main__":
    main()
