"""Score two closed Phoenix FIT images against fixed original native assignments."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec


BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
INDEX_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
INDEX = "FIT114_COCO_PRIVATE.json"
REQUEST_STATUS = "PHOENIX_HRSID_FIT114_FIXED_CANDIDATE_REQUEST_V1"
RULES = ("stock_coco", "all_nonempty")
THRESHOLDS = (.50, .75, .90)
IMAGE_KEYS = {"image_id", "image_file", "encoded_sha256", "decoded_rgb_sha256", "candidate_rows"}
ROW_KEYS = ("native_rank", "query_id", "score", "category_id")
SIZE = 800
MIB = 1024**2


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def exact(obj, keys):
    return isinstance(obj, dict) and obj.keys() == set(keys)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique(items):
    result = dict(items)
    require(len(result) == len(items), "Duplicate JSON keys")
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique)


def load_json(path, expected):
    require(digest(expected) and path.is_absolute() and path.resolve(strict=True) == path,
            "Canonical SHA-bound input file required")
    with path.open("rb") as stream:
        raw = stream.read(8*MIB + 1)
    require(len(raw) <= 8*MIB and sha256(raw).hexdigest() == expected, "JSON input digest differs")
    return parse(raw)


def load_index(path):
    require(path.is_absolute() and path.resolve(strict=True) == path, "Canonical FIT archive required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= 512*MIB, "FIT archive size out of bounds")
        hash_value = sha256()
        for block in iter(lambda: stream.read(MIB), b""):
            hash_value.update(block)
        require(hash_value.hexdigest() == BUNDLE_SHA, "FIT archive digest differs")
        stream.seek(0)
        with ZipFile(stream) as archive:
            require(archive.namelist().count(INDEX) == 1 and archive.getinfo(INDEX).file_size <= 4*MIB,
                    "Exact single bounded FIT index member required")
            raw = archive.read(INDEX)
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                and sha256(raw).hexdigest() == INDEX_SHA, "FIT archive changed or member digest differs")
    index = parse(raw)
    require(len(index["images"]) == 114, "Exact FIT114 index expected")
    return index


def prediction_rle(value):
    require(exact(value, ("size", "counts")) and value["size"] == [SIZE, SIZE]
            and isinstance(value["counts"], str) and value["counts"].isascii(),
            "Native original-grid compressed RLE required")
    return dict(size=value["size"], counts=value["counts"].encode("ascii"))


def row_identity(row):
    require(exact(row, (*ROW_KEYS, "segmentation")) and type(row["native_rank"]) is int
            and 1 <= row["native_rank"] <= 100 and type(row["query_id"]) is int
            and 0 <= row["query_id"] < 164 and type(row["category_id"]) is int
            and row["category_id"] == 1 and type(row["score"]) in (int, float)
            and math.isfinite(row["score"]) and 0 <= row["score"] <= 1, "Candidate identity schema differs")
    return row["native_rank"], row["query_id"], float(row["score"]).hex(), row["category_id"]


def validate_predictions(request, request_sha, variants):
    require(exact(request, ("status", "provenance", "images")) and request["status"] == REQUEST_STATUS
            and isinstance(request["provenance"], dict) and request["provenance"].get("fit_bundle_sha256") == BUNDLE_SHA
            and type(request["images"]) is list and len(request["images"]) == 2 and digest(request_sha),
            "Exact frozen two-image request required")
    images = request["images"]
    require(all(exact(im, IMAGE_KEYS) and type(im["image_id"]) is int and im["image_id"] >= 0
                and digest(im["encoded_sha256"]) and digest(im["decoded_rgb_sha256"]) for im in images)
            and [im["image_file"] for im in images] == ["image_001.png", "image_002.png"]
            and len({im["image_id"] for im in images}) == len({im["encoded_sha256"] for im in images}) == 2,
            "Exact distinct requested images required")
    ids = [im["image_id"] for im in images]
    require(exact(variants, RULES), "Both predeclared rules required")
    for rule, variant in variants.items():
        require(exact(variant, ("rule", "request_sha256", "images")) and variant["rule"] == rule
                and variant["request_sha256"] == request_sha and type(variant["images"]) is list
                and all(exact(im, ("image_id", "candidate_rows")) and type(im["image_id"]) is int
                        for im in variant["images"])
                and [im["image_id"] for im in variant["images"]] == ids, "Closed variant image/lock differs")
    all_rles = []
    counts = {rule: dict(eligible=0, changed=0, newly_empty=0) for rule in RULES}
    for pos, image in enumerate(images):
        orig = image["candidate_rows"]
        require(type(orig) is list and 1 <= len(orig) <= 100, "Candidate row count differs")
        keys = [row_identity(row) for row in orig]
        ranks, queries, _, _ = zip(*keys)
        require(tuple(sorted(set(ranks))) == ranks and len(set(queries)) == len(queries),
                "Original ordered native ranks/unique queries required")
        original_rles = [prediction_rle(row["segmentation"]) for row in orig]
        areas = codec.area(original_rles)
        by_rule = dict(original=original_rles)
        for rule, variant in variants.items():
            rows = variant["images"][pos]["candidate_rows"]
            require(type(rows) is list and [row_identity(row) for row in rows] == keys,
                    "Closed variant candidate identity/order/scores differ")
            rles = [prediction_rle(row["segmentation"]) for row in rows]
            output_areas = codec.area(rles)
            for orig_rle, variant_rle, area, output_area in zip(original_rles, rles, areas, output_areas, strict=True):
                eligible = area > 0 and (rule == "all_nonempty" or 100 * int(area) > SIZE * SIZE)
                changed = orig_rle != variant_rle
                require(eligible or not changed, "Closed rule changed an unselected original row")
                counts[rule]["eligible"] += int(eligible)
                counts[rule]["changed"] += int(changed)
                counts[rule]["newly_empty"] += int(area > 0 and output_area == 0)
            by_rule[rule] = rles
        all_rles.append(by_rule)
    return ids, all_rles, counts


def truth_by_image(index, ids):
    by_id = {im["id"]: im for im in index["images"] if im["id"] in ids}
    require(set(by_id) == set(ids) and all((im["height"], im["width"]) == (SIZE, SIZE)
                for im in by_id.values()), "Exact FIT original-grid images required")
    truth = {image: [] for image in ids}
    exclusions = dict(crowd_or_ignore=0, empty=0)
    seen = set()
    for ann in sorted(index["annotations"], key=lambda row: row["id"]):
        image = ann["image_id"]
        if image not in truth:
            continue
        require(ann["id"] not in seen and ann["category_id"] == 1, "Unique ship FIT annotation required")
        seen.add(ann["id"])
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            exclusions["crowd_or_ignore"] += 1
            continue
        seg = ann["segmentation"]
        rle = prediction_rle(seg) if isinstance(seg, dict) else codec.merge(codec.frPyObjects(seg, SIZE, SIZE))
        if codec.area(rle) == 0:
            exclusions["empty"] += 1
        else:
            truth[image].append(rle)
    return truth, exclusions


def score_validated(ids, rles, counts, index):
    truth, exclusions = truth_by_image(index, ids)
    labels = ("original", *RULES)
    fixed = {rule: [] for rule in labels}
    maxima = {rule: [] for rule in labels}
    for image, rows in zip(ids, rles, strict=True):
        gt = truth[image]
        if not gt:
            continue
        matrices = {rule: codec.iou(rows[rule], gt, [0] * len(gt)) for rule in labels}
        for rule, mat in matrices.items():
            maxima[rule].extend(mat.max(axis=0).tolist())
        free = np.ones(len(gt), dtype=bool)
        for row, scores in enumerate(matrices["original"]):
            allowed = np.where(free, scores, -1)
            best = int(np.argmax(allowed))
            if allowed[best] >= .5:
                free[best] = False
                for rule in labels:
                    fixed[rule].append(float(matrices[rule][row, best]))
    baseline = np.asarray(fixed["original"])
    output = {}
    for rule in labels:
        values = np.asarray(fixed[rule])
        output[rule] = dict(mean_fixed_iou=float(values.mean()) if len(values) else None,
            mean_fixed_iou_delta=float((values-baseline).mean()) if len(values) else None,
            fixed_thresholds={f"{t:.2f}": dict(
                qualified=int((values >= t).sum()),
                recovered=int(((baseline < t) & (values >= t)).sum()),
                lost=int(((baseline >= t) & (values < t)).sum())) for t in THRESHOLDS},
            independent_max_coverage={f"{t:.2f}": sum(value >= t for value in maxima[rule]) for t in THRESHOLDS})
    return dict(images=len(ids), eligible_gt=sum(map(len, truth.values())), exclusions=exclusions,
                candidate_rows=sum(len(by_rule["original"]) for by_rule in rles), rule_rows=counts,
                original_050_matches=len(baseline), metrics=output)


def evaluate(request, request_sha, variants, index):
    return score_validated(*validate_predictions(request, request_sha, variants), index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("request", "bundle", "stock", "all-nonempty", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    for arg in ("request", "stock", "all-nonempty"):
        parser.add_argument("--" + arg + "-sha256", required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical score output required")
    request = load_json(args.request, args.request_sha256)
    variants = {"stock_coco": load_json(args.stock, args.stock_sha256),
                "all_nonempty": load_json(args.all_nonempty, args.all_nonempty_sha256)}
    admitted = validate_predictions(request, args.request_sha256, variants)
    result = dict(status="PHOENIX_HRSID_FIT114_CLOSED_TWO_IMAGE_GEOMETRY_COMPLETE_V1",
                  request_sha256=args.request_sha256, fit_bundle_sha256=BUNDLE_SHA, fit_index_sha256=INDEX_SHA,
                  stock_coco_sha256=args.stock_sha256, all_nonempty_sha256=args.all_nonempty_sha256,
                  fixed_assignment="original_native_rank_greedy_mask_iou_ge_0.50_then_gt_id",
                  FIT=True, DEV=False, TEST=False, AP=False, tuning=False, model_forward_calls=0,
                  **score_validated(*admitted, load_index(args.bundle)))
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
