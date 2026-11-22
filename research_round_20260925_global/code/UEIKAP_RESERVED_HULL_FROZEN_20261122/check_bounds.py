"""Verify overlap counts directly on bitmaps and compute geometry-only ceilings."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as codec


def checked(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("Input digest mismatch")
    return raw


def ratio(intersection, area, gt):
    return intersection / (area + gt - intersection)


def geometry_bounds(scope, expected):
    count = Counter()
    thresholds = sorted(float(t) for t in expected)
    for image in scope:
        for truth in image:
            for t in thresholds:
                for name, observed in truth.items():
                    if observed >= t:
                        count[str(t), name] += 1
    result = {}
    for t in thresholds:
        key = str(t)
        row = {name: count[key, name] for name in (
            "box", "mask", "area_only", "crop_only", "area_and_crop",
            "up_to_eight_anywhere", "area_guard_and_up_to_eight_anywhere",
        )}
        if row["box"] != expected[key]["any_box"] or row["mask"] != expected[key]["any_mask"]:
            raise ValueError("Independent direct overlap counts disagree")
        if not (row["mask"] <= row["area_guard_and_up_to_eight_anywhere"] <= row["up_to_eight_anywhere"]
                <= row["area_only"] and row["mask"] <= row["area_and_crop"] <= min(row["area_only"], row["crop_only"])):
            raise ValueError("Oracle bound consistency failure")
        result[key] = row
    return result


def execute(args):
    primary = json.loads(checked(args.primary, "9ba3d8fc3c92d6169a9ccd693c1ca85156cb13d18dc757a6f4950bfb18770a90"))
    digests = primary["input_sha256"]
    labels = json.loads(checked(args.intake / "labels_roster.json", digests["labels_roster"]))["items"]
    predictions = checked(args.capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", digests["predictions_sha256"]).splitlines()
    audits = checked(args.capture / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", digests["candidate_audit_sha256"]).splitlines()
    outputs = {"top100": [], "all_post_nms": []}
    total_gt = 0
    for item, pred_raw, audit_raw in zip(labels, predictions, audits):
        ref = item["label_ref"]
        pred, audit = json.loads(pred_raw), json.loads(audit_raw)
        if pred["image_id"] != item["image_id"] or audit["image_id"] != item["image_id"]:
            raise ValueError("Image identities differ")
        truth_rows = []
        for line in checked(args.intake / ref["path"], ref["sha256"]).decode().splitlines():
            words = line.split()
            if not words or words[0] != "0":
                continue
            poly = np.array(words[1:], dtype=np.float64).reshape(-1, 2) * 512
            gt = codec.decode(codec.frPyObjects([poly.reshape(-1).tolist()], 512, 512)[0]).astype(bool)
            truth_rows.append((gt, gt.sum(), poly.min(axis=0), poly.max(axis=0)))
        total_gt += len(truth_rows)
        best = [Counter() for _ in truth_rows]
        for rank, (det, row) in enumerate(zip(audit["candidates"], pred["coco"]), 1):
            left, top, right, bottom = det["box_xyxy"]
            x0, y0, x1, y1 = np.ceil(det["box_xyxy"]).astype(int)
            pred_area = det["mask_area"]
            # In an empty original crop no foreground is allowed, including reversed endpoints.
            mask = codec.decode(row["segmentation"]).astype(bool) if pred_area else None
            if mask is not None and (mask.sum() != pred_area or mask[y0:y1, x0:x1].sum() != pred_area):
                raise ValueError("Native foreground count or original crop mismatch")
            box_area = max(0.0, right-left) * max(0.0, bottom-top)
            for j, (gt, gt_area, low, high) in enumerate(truth_rows):
                in_crop = np.count_nonzero(gt[y0:y1, x0:x1])
                hit = np.count_nonzero(mask[y0:y1, x0:x1] & gt[y0:y1, x0:x1]) if mask is not None else 0
                overlap_box = max(0.0, min(right, high[0]) - max(left, low[0])) * max(
                    0.0, min(bottom, high[1]) - max(top, low[1]))
                oracle = min(pred_area, gt_area)
                max_swaps = 8 if 32 <= pred_area < 1024 else 0
                for name, val in {
                    "box": ratio(overlap_box, box_area, np.prod(high-low)),
                    "mask": ratio(hit, pred_area, gt_area),
                    "area_only": ratio(oracle, pred_area, gt_area),
                    "crop_only": in_crop / gt_area,
                    "area_and_crop": ratio(min(pred_area, in_crop), pred_area, gt_area),
                    "up_to_eight_anywhere": ratio(min(oracle, hit+8), pred_area, gt_area),
                    "area_guard_and_up_to_eight_anywhere": ratio(min(oracle, hit+max_swaps), pred_area, gt_area),
                }.items():
                    best[j][name] = max(best[j][name], val)
            if rank == 100:
                outputs["top100"].append([dict(v) for v in best])
        outputs["all_post_nms"].append(best)
    if total_gt != 139 or any(len(v) != 108 for v in outputs.values()):
        raise ValueError("Cohort did not match the frozen population")
    return {
        "status": "independent_dense_overlap_and_ceiling_complete", "labeled_hulls": total_gt, "images": 108,
        "scopes": {s: geometry_bounds(rows, primary["scopes"][s]["thresholds"]) for s, rows in outputs.items()},
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_aggregate_sha256": "9ba3d8fc3c92d6169a9ccd693c1ca85156cb13d18dc757a6f4950bfb18770a90",
        "meaning": {
            "box_and_mask": "Observed analytic-box and native-raster opportunity, independently counted by rectangle arithmetic and dense bitmap intersection per annotated hull.",
            "area_only": "Preserve one candidate's foreground count A, freely redraw anywhere on that chip. Best intersection is min(A,G), where G is GT area. Perfect oracle, unconstrained by location.",
            "crop_only": "Arbitrary binary mask inside the original integer pixel crop. Maximum IoU is K/G, K counting GT pixels in the half-open crop.",
            "area_and_crop": "Best achievable IoU when a candidate is freely redrawn within its own original box at unchanged area. Best intersection is min(A,K).",
            "up_to_eight_anywhere": "Keep area, replace up to eight foreground pixels by eight background pixels anywhere on chip. Best intersection is min(A,G,I+8) with native intersection I.",
            "area_guard_and_up_to_eight_anywhere": "Use zero swaps outside original [32,1024) area guard, else up to eight swaps. Ignores all other policy restrictions, including rank, candidate pairing, support and confidence; reported independently within each named rank population.",
            "interpretation": "Post-hoc mathematical hull coverage ceilings, with ideal knowledge of each label. Ceilings are per hull and may reuse candidates. No prediction edits, training, rematching of published primary result, or realized performance claims.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("primary", "intake", "capture", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    record = execute(args)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": record["status"], "sha256": sha256(args.output.read_bytes()).hexdigest()}))
