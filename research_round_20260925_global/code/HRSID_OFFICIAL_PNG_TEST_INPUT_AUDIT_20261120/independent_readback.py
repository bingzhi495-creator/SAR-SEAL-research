"""Independently verify closed official TEST input audit outputs. Writes aggregates only."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
from time import perf_counter
from zipfile import ZipFile
from zlib import crc32

import numpy as np
import PIL
from PIL import Image
from pycocotools import mask


ARCHIVE_SHA = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
CENTRAL_SHA = "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce"
CENTER_SHA = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
PHASE = "HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_ONLY_V1"
FIELDS = ("encoded_sha256", "decoded_rgb_sha256")
NAMES = {
    "manifest": "TEST_IMAGE_ONLY_MANIFEST_PRIVATE.json",
    "annotations": "TEST_ORIGINAL_ANNOTATIONS_PRIVATE.json",
    "findings": "TEST_RASTER_FINDINGS_PRIVATE.json",
    "matches": "TEST_IDENTITY_BRIDGES_PRIVATE.json",
}


def check(condition, label):
    if not condition:
        raise ValueError(label)


def digest(raw):
    return sha256(raw).hexdigest()


def normalized(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_pinned(path, expected):
    raw = path.read_bytes()
    check(digest(raw) == expected, "Pinned input metadata identity")
    return json.loads(raw)


def archive_and_images(path, expected, annotations_raw, manifest):
    with path.open("rb") as stream:
        initial = os.fstat(stream.fileno())
        digest_all = sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest_all.update(block)
        check(initial.st_size == 5698302315 and digest_all.hexdigest() == ARCHIVE_SHA, "Whole archive")
        check(expected["whole_archive"] == dict(bytes=initial.st_size, sha256=ARCHIVE_SHA, exact=True), "Whole archive receipt")
        stream.seek(5697629273)
        check(digest(stream.read(672944)) == CENTRAL_SHA, "Central directory")
        with ZipFile(stream) as archive:
            entries = archive.infolist()
            check(len(entries) == len(set(archive.namelist())) == 5612, "Unique inventory")
            inventory = [dict(name=i.filename, bytes=i.file_size, crc32=format(i.CRC, "08x")) for i in entries]
            census = dict(png_members=sum(i.filename.lower().endswith(".png") for i in entries),
                          json_members=sum(i.filename.lower().endswith(".json") for i in entries),
                          directory_members=sum(i.is_dir() for i in entries),
                          central_directory_sha256=CENTRAL_SHA, ordered_member_inventory_sha256=digest(normalized(inventory)),
                          exact=True)
            check(census == expected["member_list"], "Complete inventory and all 5612 member CRC declarations")
            ann_name = "annotations/test2017.json"
            raw = archive.read(ann_name)
            check(raw == annotations_raw and crc32(raw) == archive.getinfo(ann_name).CRC, "Exact original TEST member")
            gt = json.loads(raw)
            check(expected["original_annotations"] == dict(member=ann_name, bytes=len(raw), sha256=digest(raw),
                  images=len(gt["images"]), annotations=len(gt["annotations"]), original_fields_preserved=True, exact=True),
                  "Annotation member report")
            check(len(manifest) == len(gt["images"]) == 1962 and len(gt["annotations"]) == 5922, "Original role census")
            ids = {i["id"] for i in gt["images"]}
            ann_ids = {a["id"] for a in gt["annotations"]}
            check(len(ids) == 1962 and len(ann_ids) == 5922 and all(type(i) is int for i in ids | ann_ids),
                  "Unique integer IDs within each role")
            check(all(type(a["image_id"]) is int and a["image_id"] in ids for a in gt["annotations"]), "Annotation references")
            check(len({i["file_name"] for i in gt["images"]}) == 1962 and [c["id"] for c in gt["categories"]] == [1], "Original names/category")
            for i, row in zip(gt["images"], manifest):
                stem = PurePosixPath(i["file_name"]).stem
                member = "images/" + stem + ".png"
                png = archive.read(member)
                info = archive.getinfo(member)
                check(crc32(png) == info.CRC and len(png) == info.file_size, "Selected image content CRC and size")
                check(png[24:26] == bytes([8, 2]), "PNG RGB8 IHDR")
                with Image.open(BytesIO(png)) as image:
                    check(image.format == "PNG" and image.mode == "RGB" and image.size == (800, 800)
                          and image.n_frames == 1 and image.getexif().get(274, 1) == 1, "Original RGB dimensions/orientation")
                    pixels = image.tobytes()
                check(i["width"] == i["height"] == 800 and len(pixels) == 800 * 800 * 3, "Decoded byte count")
                rebuilt = dict(image_id=i["id"], file_name=i["file_name"], member=member, bytes=len(png),
                               width=800, height=800, crc32=format(info.CRC, "08x"), source_prefix=stem.split("_", 1)[0],
                               encoded_sha256=digest(png), decoded_rgb_sha256=digest(pixels))
                check(rebuilt == row, "Exact independent manifest row")
        final = os.fstat(stream.fileno())
        check((initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns), "Archive descriptor stable")
    return gt


def raster_check(gt, findings):
    empty = []
    outside = multi = crowds = 0
    referenced = set()
    nonempty_images = set()
    for a in gt["annotations"]:
        referenced.add(a["image_id"])
        segments = a["segmentation"]
        check(type(a["category_id"]) is int and a["category_id"] == 1 and type(a["iscrowd"]) is int and
              a["iscrowd"] in (0, 1) and len(a["bbox"]) == 4 and min(a["bbox"][2:]) > 0 and a["area"] >= 0,
              "Category, crowd flag, area and bounding box")
        check(all(type(v) in (int, float) and math.isfinite(v) for v in a["bbox"] + [a["area"]]), "Finite area and box")
        check(isinstance(segments, list) and segments and all(isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0
              and all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in segments), "Finite original polygons")
        outside += any(v < 0 or v > 800 for p in segments for v in p)
        multi += len(segments) > 1
        crowds += a["iscrowd"]
        rles = mask.frPyObjects(segments, 800, 800)
        planes = mask.decode(rles)
        check(planes.shape == (800, 800, len(segments)) and planes.dtype == np.uint8 and planes.max() <= 1,
              "Original component bitmap planes")
        count = int(np.count_nonzero(np.any(planes, axis=2))) if len(rles) > 1 else int(np.count_nonzero(planes))
        check(count == int(mask.area(mask.merge(rles))), "Bitmap area and merged COCO area")
        if count:
            nonempty_images.add(a["image_id"])
        else:
            empty.append(dict(kind="EMPTY_COCO_RASTER", image_id=a["image_id"], annotation_id=a["id"]))
    check(empty == findings, "Exact findings including each empty instance identity")
    return dict(empty_rasters=len(empty), nonempty_rasters=len(gt["annotations"])-len(empty),
                invalid_rasters=0, out_of_canvas_annotations=outside, crowd_annotations=crowds,
                multicomponent_annotations=multi), dict(images_without_annotations=len(gt["images"])-len(referenced),
                images_with_only_empty_annotations=len(referenced - nonempty_images),
                images_with_any_empty_annotation=len({a["image_id"] for a in empty}))


def duplicates(manifest, bridges, stored_matches):
    within, cross, recomputed = {}, {}, []
    check(set(bridges) == {"FIT", "DEV"}, "Reference roles")
    for field in FIELDS:
        counts = Counter(r[field] for r in manifest)
        repeated = [n for n in counts.values() if n > 1]
        within[field] = dict(unique_fingerprints=len(counts), duplicate_groups=len(repeated),
                             images_in_duplicate_groups=sum(repeated), excess_duplicate_images=sum(repeated)-len(repeated))
    for role, size in (("FIT", 2937), ("DEV", 705)):
        rows = bridges[role]
        check(len(rows) == len(set(r["image_id"] for r in rows)) == size and all(type(r["image_id"]) is int for r in rows),
              "Complete original reference role")
        cross[role] = {}
        for field in FIELDS:
            index = defaultdict(list)
            for r in rows:
                check(re.fullmatch("[0-9a-f]{64}", r[field]) is not None, "Original reference fingerprint format")
                index[r[field]].append(r["image_id"])
            matched = [dict(field=field, reference_role=role, reference_image_id=i,
                            test_image_id=t["image_id"], fingerprint=t[field])
                       for t in manifest for i in index.get(t[field], ())]
            cross[role][field] = dict(identity_pairs=len(matched),
                                      reference_images_matched=len({r["reference_image_id"] for r in matched}),
                                      test_images_matched=len({r["test_image_id"] for r in matched}),
                                      shared_fingerprint_groups=len({r["fingerprint"] for r in matched}))
            recomputed.extend(matched)
    check(Counter(map(normalized, recomputed)) == Counter(map(normalized, stored_matches)), "Exact identity bridge links")
    return within, cross


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("closed-output", "freeze", "admission", "bridges", "archive", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("freeze-sha256", "admission-sha256", "bridge-sha256", "owner-handle"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--owner-exit", required=True, type=int)
    a = p.parse_args()
    check(not a.output.exists(), "Fresh safe report path")
    started = perf_counter()
    freeze = load_pinned(a.freeze, a.freeze_sha256)
    admission = load_pinned(a.admission, a.admission_sha256)
    bridges = load_pinned(a.bridges, a.bridge_sha256)
    runtime = dict(python=platform.python_version(), numpy=np.__version__, Pillow=PIL.__version__, pycocotools=version("pycocotools"))
    check(freeze["phase"] == admission["phase"] == PHASE and freeze["status"] == "HRSID_TEST_ENDPOINTS_FROZEN_FOR_INPUT_AUDIT_V1"
          and freeze["endpoint_freeze_complete"] is True and freeze["endpoints_sha256"] == {"center_e24": CENTER_SHA}
          and freeze["seed"] == 5202 and freeze["completed_epochs"] == 24, "Frozen baseline")
    check(admission["status"] == "ROOT_ADMITTED_HRSID_TEST_INPUT_AUDIT_V1" and admission["input_audit_only"] is True and
          admission["endpoint_freeze_sha256"] == a.freeze_sha256 and admission["reference_bridge_manifest_sha256"] == a.bridge_sha256
          and admission["expected_runtime_versions"] == runtime, "Pinned original admission and runtime")
    raw_report = (a.closed_output / "RESULT_SAFE.json").read_bytes()
    original = json.loads(raw_report)
    check(original["endpoint_freeze_sha256"] == a.freeze_sha256 and original["admission_sha256"] == a.admission_sha256
          and original["reference_bridge_manifest_sha256"] == a.bridge_sha256 and original["runtime_versions"] == runtime
          and original["phase"] == PHASE and original["status"] == "HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_COMPLETE", "Worker scope and bindings")
    check(set(original["private_artifacts"]) == set(NAMES.values()), "Exact private artifact roster")
    payload = {}
    for role, name in NAMES.items():
        raw = (a.closed_output / name).read_bytes()
        check(original["private_artifacts"][name] == dict(bytes=len(raw), sha256=digest(raw)), "Private artifact byte identity")
        payload[role] = raw if role == "annotations" else json.loads(raw)
    gt = archive_and_images(a.archive, original["qualification"], payload["annotations"], payload["manifest"])
    check(original["image_manifest_sha256"] == digest(normalized(payload["manifest"])), "Image manifest binding")
    rasters, image_counts = raster_check(gt, payload["findings"])
    check(all(original[k] == v for k, v in rasters.items()), "Independent raster count comparison")
    within, cross = duplicates(payload["manifest"], bridges, payload["matches"])
    check(original["within_test_identity"] == within and original["cross_split_identity"] == cross, "Duplicate counts")
    check(original["structural_input_qualified"] is True and original["decoded_images"] == 1962 and original["decoded_storage_shape"] == [800,800,3]
          and original["raster_attempts"] == 5922 and original["image_check_failures"] == 0 and original["empty_instances_removed"] == 0
          and original["original_gt_preserved"] is True and original["AP"] is False and original["model_forward_calls"] == 0
          and original["optimizer_updates"] == 0 and original["TEST"] is True and original["FIT"] is False and original["DEV"] is False,
          "Structural qualification and execution scope")
    check(a.owner_exit == 0, "Owner-reported original exit")
    safe = dict(status="INDEPENDENT_HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_READBACK_ACCEPTED", phase=PHASE,
                scope="Input only. Historical official TEST selection exposure; acquisition/scene independence and annotation geometry not established.",
                original_process=dict(tool_session_handle=a.owner_handle, exit_code=a.owner_exit, evidence="Root owner's observation of original process completion"),
                readback_source_sha256=digest(Path(__file__).read_bytes()), readback_seconds=round(perf_counter()-started,3),
                runtime_versions=runtime, endpoint_sha256=CENTER_SHA, freeze_sha256=a.freeze_sha256,
                admission_sha256=a.admission_sha256, bridge_sha256=a.bridge_sha256, closed_result_sha256=digest(raw_report),
                closed_artifact_bindings={k:original["private_artifacts"][v] for k,v in NAMES.items()},
                archive_sha256=ARCHIVE_SHA, whole_archive_bytes=5698302315, central_directory_sha256=CENTRAL_SHA,
                full_inventory_crc_declarations=5612, separately_read_member_crcs=1963,
                original_annotation_member_byte_equality=True, independently_redecoded_test_images=1962,
                original_test_annotations=5922, reference_counts=dict(FIT=2937,DEV=705),
                within_test_identity=within, cross_split_identity=cross, exact_private_bridge_link_multiset=True,
                exact_private_raster_findings=True, original_annotations_removed=0,
                structural_input_qualified=True, model_forward_calls=0, optimizer_updates=0, AP=False, **rasters, **image_counts)
    with a.output.open("xb") as out:
        out.write((json.dumps(safe, indent=2, sort_keys=True) + "\n").encode())
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({"status":safe["status"],"seconds":safe["readback_seconds"],"safe_sha256":digest(a.output.read_bytes())}),flush=True)


if __name__ == "__main__":
    main()
