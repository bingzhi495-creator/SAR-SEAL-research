"""Input-only audit of frozen HRSID official PNG TEST; no model or metric code."""
from collections import Counter, defaultdict
import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import UnsupportedOperation
import json
from importlib.metadata import version
import os
from os import fstat
from pathlib import Path, PurePosixPath
import platform
import re
from types import ModuleType
from time import perf_counter
from zipfile import ZipFile

PHASE = "HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_ONLY_V1"
FREEZE_STATUS = "HRSID_TEST_ENDPOINTS_FROZEN_FOR_INPUT_AUDIT_V1"
ADMISSION_STATUS = "ROOT_ADMITTED_HRSID_TEST_INPUT_AUDIT_V1"
ANNOTATION_MEMBER = "annotations/test2017.json"
MIB = 1024**2


@dataclass(frozen=True)
class Contract:
    archive_bytes: int
    archive_sha256: str
    central_offset: int
    central_bytes: int
    central_sha256: str
    png_members: int
    json_members: int
    directory_members: int
    images: int
    annotations: int
    bridge_counts: dict


OFFICIAL = Contract(5698302315, "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159",
                    5697629273, 672944,
                    "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce",
                    5604, 5, 3, 1962, 5922, {"FIT":2937, "DEV":705})


def need(ok, why):
    if not ok:
        raise ValueError(why)


def digest_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_scope(frozen, admission, frozen_sha256):
    need(frozen.get("phase") == admission.get("phase") == PHASE, "Exact input-only TEST phase")
    need(frozen.get("status") == FREEZE_STATUS and frozen.get("endpoint_freeze_complete") is True,
         "Actual endpoint freeze required")
    refs = frozen.get("endpoints_sha256")
    need(isinstance(refs, dict) and bool(refs) and all(isinstance(k, str) and k and digest_string(v)
                                                    for k, v in refs.items()), "Frozen endpoint digests")
    need(admission.get("status") == ADMISSION_STATUS and admission.get("input_audit_only") is True
         and digest_string(frozen_sha256) and admission.get("endpoint_freeze_sha256") == frozen_sha256,
         "Exact Root admission to this endpoint freeze")


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def load_helper(path, expected_sha256):
    """Load only the explicitly pinned, ordinary published FIT helper source."""
    need(digest_string(expected_sha256), "Pinned helper digest")
    raw = path.read_bytes()
    need(0 < len(raw) <= MIB and sha256(raw).hexdigest() == expected_sha256, "Exact FIT helper source")
    module = ModuleType("_hrsid_test_input_fit_tools")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    need(module.SIDE == 800 and module.ImageFile.LOAD_TRUNCATED_IMAGES is False,
         "Original strict RGB800 helper")
    return module


def stream_identity(stream):
    try:
        s = fstat(stream.fileno())
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns
    except (AttributeError, UnsupportedOperation):
        return None  # In-memory synthetic integration fixtures only.


def qualify_archive(stream, contract):
    start = stream_identity(stream)
    stream.seek(0)
    total, digest = 0, sha256()
    while True:
        block = stream.read(4*MIB)
        if not block:
            break
        total += len(block)
        digest.update(block)
    need(total == contract.archive_bytes and digest.hexdigest() == contract.archive_sha256,
         "Exact whole official archive bytes and SHA-256")
    stream.seek(contract.central_offset)
    central = stream.read(contract.central_bytes)
    need(len(central) == contract.central_bytes and sha256(central).hexdigest() == contract.central_sha256,
         "Exact official central directory")
    stream.seek(0)
    return start, dict(bytes=total, sha256=digest.hexdigest(), exact=True)


def qualify_members(z, contract):
    infos = z.infolist()
    names = [r.filename for r in infos]
    need(len(names) == len(set(names)), "Unique ZIP member list")
    for row in infos:
        p = PurePosixPath(row.filename)
        need(not p.is_absolute() and "\\" not in row.filename and ".." not in p.parts
             and not row.flag_bits & 1 and (row.external_attr >> 16) & 0o170000 != 0o120000,
             "Safe unencrypted ordinary ZIP members")
    png = sum(n.lower().endswith(".png") for n in names)
    js = sum(n.lower().endswith(".json") for n in names)
    dirs = sum(r.is_dir() for r in infos)
    need((png, js, dirs) == (contract.png_members, contract.json_members, contract.directory_members)
         and len(infos) == png + js + dirs, "Exact original member-list census")
    need(ANNOTATION_MEMBER in names, "Official TEST annotation member present")
    inventory = [dict(name=r.filename, bytes=r.file_size, crc32=f"{r.CRC:08x}") for r in infos]
    return dict(png_members=png, json_members=js, directory_members=dirs,
                central_directory_sha256=contract.central_sha256,
                ordered_member_inventory_sha256=sha256(encoded(inventory)).hexdigest(), exact=True)


def qualify_annotations(z, helper, contract):
    raw = helper.checked_member(z, ANNOTATION_MEMBER, 24*MIB)
    gt = json.loads(raw)
    images, annotations = gt["images"], gt["annotations"]
    need(len(images) == contract.images and len(annotations) == contract.annotations,
         "Exact original TEST image and annotation counts")
    need([c["id"] for c in gt["categories"]] == [1], "Original single category")
    ids = [i["id"] for i in images]
    ann_ids = [a["id"] for a in annotations]
    need(len(set(ids)) == len(ids) and all(type(i) is int for i in ids)
         and len(set(ann_ids)) == len(ann_ids) and all(type(i) is int for i in ann_ids),
         "Unique original integer image and annotation IDs")
    image_ids = set(ids)
    need(all(type(a["image_id"]) is int and a["image_id"] in image_ids for a in annotations),
         "Original annotation/image association")
    stems = [helper.image_stem(r["file_name"]) for r in images]
    need(len(set(stems)) == len(stems), "Unique original image stems")
    for record, stem in zip(images, stems):
        need(type(record["width"]) is type(record["height"]) is int
             and record["width"] == record["height"] == 800, "Original annotation size is RGB800")
        need("images/" + stem + ".png" in z.namelist(), "Missing original TEST PNG member")
    qualification = dict(member=ANNOTATION_MEMBER, bytes=len(raw), sha256=sha256(raw).hexdigest(),
                         images=len(images), annotations=len(annotations), original_fields_preserved=True,
                         exact=True)
    return gt, raw, qualification


def identity_bridges(manifest, references, expected, helper):
    need(set(references) == set(expected), "Exact closed FIT and DEV bridge roles")
    summaries, links = {}, []
    for role, rows in references.items():
        need(isinstance(rows, list) and len(rows) == expected[role]
             and len({r["image_id"] for r in rows}) == len(rows)
             and all(type(r["image_id"]) is int and
                     all(digest_string(r[k]) for k in ("encoded_sha256", "decoded_rgb_sha256")) for r in rows),
             "Exact closed reference roster and fingerprints")
        summaries[role] = {}
        for field in ("encoded_sha256", "decoded_rgb_sha256"):
            by_digest = defaultdict(list)
            for row in rows:
                by_digest[row[field]].append(row["image_id"])
            test_ids, reference_ids, groups, pairs = set(), set(), set(), 0
            for row in manifest:
                matched = by_digest.get(row[field], [])
                if matched:
                    test_ids.add(row["image_id"])
                    reference_ids.update(matched)
                    groups.add(row[field])
                    pairs += len(matched)
                    links.extend(dict(field=field, reference_role=role, test_image_id=row["image_id"],
                                      reference_image_id=i, fingerprint=row[field]) for i in matched)
            summaries[role][field] = dict(test_images_matched=len(test_ids), reference_images_matched=len(reference_ids),
                                          shared_fingerprint_groups=len(groups), identity_pairs=pairs)
    within = {field:helper.fingerprint_counts(Counter(r[field] for r in manifest))
              for field in ("encoded_sha256", "decoded_rgb_sha256")}
    return summaries, within, links


def audit_archive(stream, helper, contract, frozen, admission, frozen_sha256, references):
    """No defaults for payloads. Contract overrides exist only for synthetic fixtures."""
    validate_scope(frozen, admission, frozen_sha256)  # Before any archive access.
    initial, whole = qualify_archive(stream, contract)
    with ZipFile(stream) as z:
        members = qualify_members(z, contract)
        gt, original_bytes, annotations = qualify_annotations(z, helper, contract)
        manifest, findings = [], []
        image_failures = invalid = empty = nonempty = outside = multicomponent = crowd = 0
        for record in gt["images"]:
            stem = helper.image_stem(record["file_name"])
            member = "images/" + stem + ".png"
            info = z.getinfo(member)
            try:
                raw = helper.checked_member(z, member, helper.PNG_LIMIT)
                rgb_sha = helper.decode_rgb(raw)
                manifest.append(dict(image_id=record["id"], file_name=record["file_name"], member=member,
                                     source_prefix=stem.split("_",1)[0], width=800, height=800,
                                     bytes=len(raw), crc32=f"{info.CRC:08x}",
                                     encoded_sha256=sha256(raw).hexdigest(), decoded_rgb_sha256=rgb_sha))
            except (KeyError, OSError, ValueError, SyntaxError) as error:
                image_failures += 1
                findings.append(dict(kind="IMAGE_CHECK_FAILED", image_id=record["id"], detail=str(error)))
        for row in gt["annotations"]:
            try:
                pixels, out, components, iscrowd = helper.raster_instance(row)
                outside += out > 0
                multicomponent += components > 1
                crowd += iscrowd
                if pixels:
                    nonempty += 1
                else:
                    empty += 1
                    findings.append(dict(kind="EMPTY_COCO_RASTER", image_id=row["image_id"], annotation_id=row["id"]))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                invalid += 1
                findings.append(dict(kind="ANNOTATION_CHECK_FAILED", image_id=row["image_id"],
                                     annotation_id=row["id"], detail=str(error)))
    need(stream_identity(stream) == initial, "Original archive descriptor unchanged during audit")
    cross, within, links = identity_bridges(manifest, references, contract.bridge_counts, helper)
    qualified = len(manifest) == contract.images and image_failures == invalid == 0
    report = dict(status="HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_COMPLETE", phase=PHASE,
                  structural_input_qualified=qualified, qualification=dict(whole_archive=whole,
                  member_list=members, original_annotations=annotations), decoded_images=len(manifest),
                  image_check_failures=image_failures, decoded_storage_shape=[800,800,3],
                  raster_attempts=len(gt["annotations"]), nonempty_rasters=nonempty, empty_rasters=empty,
                  invalid_rasters=invalid, out_of_canvas_annotations=outside,
                  multicomponent_annotations=multicomponent, crowd_annotations=crowd,
                  empty_instances_removed=0, original_gt_preserved=True, within_test_identity=within,
                  cross_split_identity=cross, endpoint_freeze_sha256=frozen_sha256,
                  image_manifest_sha256=sha256(encoded(manifest)).hexdigest(),
                  TEST=True, FIT=False, DEV=False, AP=False, model_forward_calls=0, optimizer_updates=0,
                  geometric_annotation_accuracy_validated=False, scene_independence_proven=False)
    return dict(report=report, image_manifest=manifest, original_gt=gt, original_annotation_bytes=original_bytes,
                raster_findings=findings, identity_bridge_links=links)


def pinned_metadata(path, digest, limit):
    need(digest_string(digest) and path.is_file() and not path.is_symlink()
         and 0 < path.stat().st_size <= limit, "Bounded explicit metadata input")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact metadata SHA-256")
    return json.loads(raw)


def runtime_versions():
    import numpy
    import PIL
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                Pillow=PIL.__version__, pycocotools=version("pycocotools"))


def durable_new(path, raw):
    with path.open("xb") as f:
        need(f.write(raw) == len(raw), "Complete output write")
        f.flush()
        os.fsync(f.fileno())
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "admission", "archive", "fit-helper", "bridges", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("freeze-sha256", "admission-sha256", "fit-helper-sha256", "bridges-sha256"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    frozen = pinned_metadata(args.freeze, args.freeze_sha256, MIB)
    admission = pinned_metadata(args.admission, args.admission_sha256, MIB)
    validate_scope(frozen, admission, args.freeze_sha256)
    # No payload/helper/reference lookup or output creation before this gate.
    actual_runtime = runtime_versions()
    need(admission.get("expected_runtime_versions") == actual_runtime, "Explicit admitted CPU runtime versions")
    helper = load_helper(args.fit_helper, args.fit_helper_sha256)
    references = pinned_metadata(args.bridges, args.bridges_sha256, 8*MIB)
    need(args.archive.is_file() and not args.archive.is_symlink(), "Explicit regular archive input")
    need(args.output.is_absolute() and not args.output.exists() and args.output.parent.is_dir(),
         "Fresh absolute output under an existing ordinary parent")
    started = perf_counter()
    with args.archive.open("rb") as stream:
        result = audit_archive(stream, helper, OFFICIAL, frozen, admission, args.freeze_sha256, references)
    args.output.mkdir()
    files = {
        "TEST_IMAGE_ONLY_MANIFEST_PRIVATE.json":encoded(result["image_manifest"]),
        "TEST_ORIGINAL_ANNOTATIONS_PRIVATE.json":result["original_annotation_bytes"],
        "TEST_RASTER_FINDINGS_PRIVATE.json":encoded(result["raster_findings"]),
        "TEST_IDENTITY_BRIDGES_PRIVATE.json":encoded(result["identity_bridge_links"]),
    }
    artifacts = {name:durable_new(args.output/name, raw) for name, raw in files.items()}
    report = result["report"]
    report.update(source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=args.fit_helper_sha256, admission_sha256=args.admission_sha256,
                  reference_bridge_manifest_sha256=args.bridges_sha256, runtime_versions=actual_runtime,
                  private_artifacts=artifacts, audit_worker_seconds=perf_counter()-started)
    durable_new(args.output/"RESULT_SAFE.json", encoded(report))
    print(json.dumps(dict(status=report["status"], structural_input_qualified=report["structural_input_qualified"],
                          decoded_images=report["decoded_images"], raster_attempts=report["raster_attempts"],
                          empty_rasters=report["empty_rasters"], invalid_rasters=report["invalid_rasters"],
                          TEST=True, AP=False, model_forward_calls=0), sort_keys=True), flush=True)
    return 0 if report["structural_input_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
