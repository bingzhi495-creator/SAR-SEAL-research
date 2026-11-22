"""Rebuild the image-only HRSID official manifest and frozen scoring populations from the author ZIP."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from zipfile import ZipFile


MANIFEST = "IMAGE_MANIFEST_PRIVATE.json"
PLAN = "COHORTS_PRIVATE.json"
MANIFEST_SHA = "dd42cbf5ef415dc152bab9160c5eb385ea33255d697fba2dddf49917eda3ee1b"
COHORT_SHA = "1afb4b95632d3c301320eb07711bb48aa468d3a154f33ad84e7b28c1af110827"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
GT_SHA = "eb611754afc74c34222fa52cc04a500d1b84f956782fcbcc0e91d9766c4ac937"
ARCHIVE_SHA = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
AUDIT_REFERENCE = "15aabe26b7580c55a5b5f3e4ae1558cac84294ae950f283c826de5b0720e53ee"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, name, digest=None):
    raw = path.read_bytes()
    check(digest is None or sha256(raw).hexdigest() == digest, "Published operator differs: " + path.name)
    module = module_from_spec(spec_from_file_location(name, str(path)))
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def encode(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_digest(path):
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
        return digest.hexdigest()


def prepare(args):
    helper = load(args.code / "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py", "_images",
                  "eb5f91a333ea7c7a1c12a87574d0c192f678aa80a742107a510654e3d5512df5")
    partition = load(args.code / "HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_20261120/freeze_score_cohorts.py", "_cohorts",
                     "d0351e2867076471f4380c12b1ff40c8ea859e8def3654b491b0fd3af2d8c466")
    check(not args.output.exists(), "Use a fresh output directory")
    check(args.archive.stat().st_size == helper.ARCHIVE_BYTES and file_digest(args.archive) == ARCHIVE_SHA,
          "Original author ZIP digest differs")
    with ZipFile(args.archive) as archive:
        gt_raw = archive.read("annotations/test2017.json")
        train_raw = archive.read(helper.TRAIN_MEMBER)
        check(sha256(gt_raw).hexdigest() == GT_SHA and sha256(train_raw).hexdigest() == helper.TRAIN_SHA,
              "Original annotation files differ")
        gt = json.loads(gt_raw)
        manifest = []
        for record in gt["images"]:
            stem = helper.image_stem(record["file_name"])
            member = "images/" + stem + ".png"
            info = archive.getinfo(member)
            raw = helper.checked_member(archive, member, helper.PNG_LIMIT)
            manifest.append(dict(image_id=record["id"], file_name=record["file_name"], member=member,
                                 source_prefix=stem.split("_", 1)[0], width=record["width"], height=record["height"],
                                 bytes=len(raw), crc32=f"{info.CRC:08x}", encoded_sha256=sha256(raw).hexdigest(),
                                 decoded_rgb_sha256=helper.decode_rgb(raw)))
        data = encode(manifest)
        check(sha256(data).hexdigest() == MANIFEST_SHA, "Regenerated ordered 1,962-image manifest differs")
        # Both original FIT overlaps are the only TRAIN overlaps. Reconstruct from the
        # full author train roster, then verify every image and annotation ID against
        # the historical cohort digest before publishing this intake.
        train_rgb = set()
        for record in json.loads(train_raw)["images"]:
            name = "images/" + helper.image_stem(record["file_name"]) + ".png"
            train_rgb.add(helper.decode_rgb(helper.checked_member(archive, name, helper.PNG_LIMIT)))
    links = [dict(test_image_id=r["image_id"], reference_role="FIT", field="decoded_rgb_sha256")
             for r in manifest if r["decoded_rgb_sha256"] in train_rgb]
    findings = [dict(kind="EMPTY_COCO_RASTER", annotation_id=a["id"]) for a in gt["annotations"]
                if helper.raster_instance(a)[0] == 0]
    check(len(links) == 2 and len(findings) == 4, "TRAIN/TEST overlap or empty-raster population differs")
    views = partition.partition(gt, manifest, findings, links)
    plan = dict(status="HRSID_TEST_SCORING_COHORTS_FROZEN_BEFORE_PREDICTIONS", endpoint_sha256=ENDPOINT,
                independent_input_readback_sha256=AUDIT_REFERENCE,
                image_identity_rule="Exclude decoded-RGB FIT overlaps, retain smallest original ID per remaining decoded-RGB key.",
                views=views)
    # The original freeze used text-mode newlines on its host.
    raw = encode(plan)
    if sha256(raw).hexdigest() != COHORT_SHA:
        raw = raw[:-1] + b"\r\n"
    check(sha256(raw).hexdigest() == COHORT_SHA, "Reconstructed frozen evaluation IDs differ")
    result = dict(status="B5_HRSID1962_PREPARE_COMPLETE", archive_sha256=ARCHIVE_SHA, gt_sha256=GT_SHA,
                  manifest_sha256=MANIFEST_SHA, cohorts_sha256=COHORT_SHA, images=len(manifest),
                  annotations=len(gt["annotations"]), named_cohorts={k:{f:len(ids) for f, ids in v.items()}
                  for k, v in views.items()}, model_forward_calls=0)
    args.output.mkdir()
    (args.output / MANIFEST).write_bytes(data)
    (args.output / PLAN).write_bytes(raw)
    (args.output / "RESULT_SAFE.json").write_bytes(encode(result))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parent.parent)
    prepare(parser.parse_args())

