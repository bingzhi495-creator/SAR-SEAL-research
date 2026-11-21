"""Freeze official and two sensitivity cohorts before any score is computed."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def partition(gt, manifest, findings, links):
    primary_images = [record["id"] for record in gt["images"]]
    primary_annotations = [record["id"] for record in gt["annotations"]]
    man = {row["image_id"]: row["decoded_rgb_sha256"] for row in manifest}
    if set(primary_images) != set(man) or len(primary_images) != len(manifest):
        raise ValueError("Exact original image population")
    fit_overlap = {row["test_image_id"] for row in links if
                   row["reference_role"] == "FIT" and row["field"] == "decoded_rgb_sha256"}
    if not fit_overlap <= set(primary_images):
        raise ValueError("FIT overlap refers outside the test population")
    remaining = {}
    for image in sorted(set(primary_images) - fit_overlap):
        remaining.setdefault(man[image], image)
    keep = set(remaining.values())
    empty_labels = {row["annotation_id"] for row in findings if row["kind"] == "EMPTY_COCO_RASTER"}
    if not empty_labels <= set(primary_annotations):
        raise ValueError("Empty-raster IDs outside original annotations")
    return {
        "official": {"images": primary_images, "annotations": primary_annotations},
        "empty_raster_sensitivity": {
            "images": primary_images,
            "annotations": [item for item in primary_annotations if item not in empty_labels],
        },
        "image_identity_sensitivity": {
            "images": [item for item in primary_images if item in keep],
            "annotations": [ann["id"] for ann in gt["annotations"] if ann["image_id"] in keep],
        },
    }


def read(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Digest mismatch: " + path.name)
    return json.loads(raw)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--audit", type=Path, required=True)
    args.add_argument("--readback", type=Path, required=True)
    args.add_argument("--readback-sha256", required=True)
    args.add_argument("--output", type=Path, required=True)
    a = args.parse_args()
    closure = read(a.readback, a.readback_sha256)
    if closure["status"] != "INDEPENDENT_HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_READBACK_ACCEPTED":
        raise ValueError("The input audit is not independently accepted")
    artifacts = closure["closed_artifact_bindings"]
    gt, manifest, findings, links = (
        read(a.audit / name, artifacts[key]["sha256"])
        for key, name in (
            ("annotations", "TEST_ORIGINAL_ANNOTATIONS_PRIVATE.json"),
            ("manifest", "TEST_IMAGE_ONLY_MANIFEST_PRIVATE.json"),
            ("findings", "TEST_RASTER_FINDINGS_PRIVATE.json"),
            ("matches", "TEST_IDENTITY_BRIDGES_PRIVATE.json")
        )
    )
    views = partition(gt, manifest, findings, links)
    totals = {k: {name: len(seq) for name, seq in v.items()} for k, v in views.items()}
    if totals["official"] != {"images": 1962, "annotations": 5922}:
        raise ValueError("The official cohort changed")
    if totals["empty_raster_sensitivity"] != {"images": 1962, "annotations": 5918}:
        raise ValueError("Expected the independently verified four empty masks")
    omitted = (set(views["official"]["images"]) - set(views["image_identity_sensitivity"]["images"]))
    annotations_per_image = Counter(ann["image_id"] for ann in gt["annotations"])
    plan = {
        "status": "HRSID_TEST_SCORING_COHORTS_FROZEN_BEFORE_PREDICTIONS",
        "endpoint_sha256": closure["endpoint_sha256"],
        "independent_input_readback_sha256": a.readback_sha256,
        "image_identity_rule": "Exclude decoded-RGB FIT overlaps, retain smallest original ID per remaining decoded-RGB key.",
        "views": views,
    }
    with a.output.open("x", encoding="utf-8") as fh:
        json.dump(plan, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
    print(json.dumps(dict(named_cohorts=totals, omitted_unlabeled_images=sum(
        annotations_per_image[i] == 0 for i in omitted), model_forward_calls=0, AP=False), sort_keys=True))


if __name__ == "__main__":
    main()
