"""Validate anonymous human hull polygon returns against an image roster."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.errors import ShapelyError
from shapely.geometry import Polygon


ROSTER_SCHEMA = "sar-hull-roster-v1"
REVIEW_SCHEMA = "sar-hull-review-v1"
STATUSES = ("complete", "complete_empty", "incomplete", "unreviewed")
COMPLETE = ("complete", "complete_empty")
LABELS = ("ship_hull", "uncertain", "truncated")
MODES = {"native", "bright"}
IOU_THRESHOLD = 0.50


def require(condition, context, reason):
    if not condition:
        raise ValueError(f"{context}: {reason}")


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON", f"duplicate object key {key!r}")
        result[key] = value
    return result


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{path.name}: {error}") from error


def fields(obj, names, context):
    require(isinstance(obj, dict) and set(obj) == set(names), context,
            f"expected exactly these fields: {', '.join(names)}")


def positive_integer(value):
    return type(value) is int and value > 0


def read_roster(path):
    roster = read_json(path)
    fields(roster, ("schema", "images"), "roster")
    require(roster["schema"] == ROSTER_SCHEMA, "roster", f"expected schema {ROSTER_SCHEMA}")
    require(isinstance(roster["images"], list) and bool(roster["images"]), "roster", "expected a nonempty image list")
    items = {}
    for index, row in enumerate(roster["images"]):
        context = f"roster image {index + 1}"
        fields(row, ("image_id", "width", "height", "png_sha256"), context)
        image_id = row["image_id"]
        require(isinstance(image_id, str) and bool(image_id.strip()), context, "image_id must be a nonempty string")
        require(image_id not in items, context, "duplicate image_id")
        require(positive_integer(row["width"]) and positive_integer(row["height"]), context,
                "width and height must be positive integers")
        checksum = row["png_sha256"]
        require(isinstance(checksum, str) and re.fullmatch(r"[0-9a-fA-F]{64}", checksum), context,
                "png_sha256 must contain 64 hexadecimal digits")
        items[image_id] = (row["width"], row["height"])
    return items


def in_bounds(value, limit):
    if type(value) is int:
        return 0 <= value <= limit
    return type(value) is float and math.isfinite(value) and 0 <= value <= limit


def read_polygon(points, width, height, context):
    require(isinstance(points, list) and len(points) >= 3, context, "expected at least three vertices")
    for point in points:
        require(isinstance(point, list) and len(point) == 2, context, "each vertex must be an [x, y] pair")
        require(in_bounds(point[0], width) and in_bounds(point[1], height), context,
                "vertex outside native image-edge bounds or not finite")
    try:
        polygon = Polygon(points)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{context}: cannot interpret vertices as a polygon") from error
    require(polygon.is_valid and math.isfinite(polygon.area) and polygon.area > 0, context,
            "polygon is degenerate or invalid, including self-intersection")
    return polygon


def read_review(path, roster):
    review = read_json(path)
    fields(review, ("schema", "reviewer", "images"), "review")
    require(review["schema"] == REVIEW_SCHEMA, "review", f"expected schema {REVIEW_SCHEMA}")
    reviewer = review["reviewer"]
    require(reviewer in ("A", "B"), "review", "reviewer must be A or B")
    require(isinstance(review["images"], list), "review", "expected an image list")
    items = {}
    for index, row in enumerate(review["images"]):
        context = f"review image {index + 1}"
        fields(row, ("image_id", "width", "height", "status", "notes", "modes", "shapes"), context)
        image_id = row["image_id"]
        require(isinstance(image_id, str) and image_id in roster, context, "image_id is outside the roster")
        context = f"review image {image_id}"
        require(image_id not in items, context, "duplicate image_id")
        require(positive_integer(row["width"]) and positive_integer(row["height"]) and
                (row["width"], row["height"]) == roster[image_id], context, "dimensions do not match the native roster")
        status = row["status"]
        require(status in STATUSES, context, "unknown review status")
        require(isinstance(row["notes"], str), context, "notes must be a string")
        modes = row["modes"]
        require(isinstance(modes, list) and all(isinstance(mode, str) and mode in MODES for mode in modes),
                context, "modes must be a list of native or bright")
        shapes = row["shapes"]
        require(isinstance(shapes, list), context, "shapes must be a list")
        if status == "complete":
            require(bool(shapes), context, "complete requires at least one shape")
        if status == "complete_empty":
            require(not shapes, context, "complete_empty requires zero shapes")
        polygons = {}
        for ordinal, shape in enumerate(shapes):
            shape_context = f"{context}, shape {ordinal + 1}"
            fields(shape, ("id", "label", "points"), shape_context)
            shape_id = shape["id"]
            require(positive_integer(shape_id) and shape_id not in polygons, shape_context,
                    "shape id must be a distinct positive integer within the image")
            label = shape["label"]
            require(label in LABELS, shape_context, "unknown polygon label")
            polygons[shape_id] = (label, read_polygon(shape["points"], *roster[image_id], shape_context))
        items[image_id] = (status, polygons)
    require(set(items) == set(roster), "review", f"missing {len(roster) - len(items)} roster image(s)")
    return reviewer, items


def review_summary(reviewer, items):
    windows = Counter(status for status, _ in items.values())
    shapes = Counter(label for _, polygons in items.values() for label, _ in polygons.values())
    complete_with_hulls = sum(status in COMPLETE and any(label == "ship_hull" for label, _ in polygons.values())
                              for status, polygons in items.values())
    return {
        "reviewer": reviewer,
        "images": len(items),
        "images_by_status": {status: windows[status] for status in STATUSES},
        "shapes_by_label": {label: shapes[label] for label in LABELS},
        "complete_windows_with_hulls": complete_with_hulls,
    }


def continuous_iou(a, b):
    intersection = a.intersection(b).area
    return min(1.0, intersection / (a.area + b.area - intersection))


def matched_hulls(a_items, b_items):
    a = sorted((identifier, polygon) for identifier, (label, polygon) in a_items.items() if label == "ship_hull")
    b = sorted((identifier, polygon) for identifier, (label, polygon) in b_items.items() if label == "ship_hull")
    pairs = []
    if a and b:
        overlaps = np.array([[continuous_iou(left, right) for _, right in b] for _, left in a])
        accepted = overlaps >= IOU_THRESHOLD
        # The bonus exceeds every possible total IoU difference, so pair count comes first.
        bonus = min(len(a), len(b)) + 1
        weights = np.where(accepted, bonus + overlaps, 0.0)
        a_indices, b_indices = linear_sum_assignment(-weights)
        pairs = [{"a_shape_id": a[i][0], "b_shape_id": b[j][0], "continuous_iou": float(overlaps[i, j])}
                 for i, j in zip(a_indices, b_indices) if accepted[i, j]]
    a_used = {pair["a_shape_id"] for pair in pairs}
    b_used = {pair["b_shape_id"] for pair in pairs}
    return pairs, [identifier for identifier, _ in a if identifier not in a_used], [
        identifier for identifier, _ in b if identifier not in b_used
    ]


def agreement_report(roster, a, b):
    status_pairs = {status: Counter() for status in STATUSES}
    ious = []
    unmatched_a = unmatched_b = scored_count = 0
    windows = []
    for image_id in roster:
        a_status, a_items = a[image_id]
        b_status, b_items = b[image_id]
        status_pairs[a_status][b_status] += 1
        window = {
            "image_id": image_id, "a_status": a_status, "b_status": b_status, "pairs": None,
            "a_unmatched_shape_ids": None, "b_unmatched_shape_ids": None,
        }
        if a_status in COMPLETE and b_status in COMPLETE:
            pairs, a_left, b_left = matched_hulls(a_items, b_items)
            window.update(pairs=pairs, a_unmatched_shape_ids=a_left, b_unmatched_shape_ids=b_left)
            ious.extend(pair["continuous_iou"] for pair in pairs)
            unmatched_a += len(a_left)
            unmatched_b += len(b_left)
            scored_count += 1
        windows.append(window)
    pair_count = len(ious)
    a_count, b_count = pair_count + unmatched_a, pair_count + unmatched_b
    return {
        "schema": "sar-hull-review-agreement-v1",
        "a_review": review_summary("A", a),
        "b_review": review_summary("B", b),
        "status_pairs": {left: {right: counts[right] for right in STATUSES} for left, counts in status_pairs.items()},
        "agreement": {
            "minimum_continuous_iou": IOU_THRESHOLD,
            "jointly_completed_windows": scored_count,
            "other_windows": len(roster) - scored_count,
            "a_ship_hulls": a_count,
            "b_ship_hulls": b_count,
            "accepted_pairs": pair_count,
            "a_unmatched_hulls": unmatched_a,
            "b_unmatched_hulls": unmatched_b,
            "symmetric_instance_f1": 2 * pair_count / (a_count + b_count) if a_count + b_count else None,
            "matched_continuous_iou_mean": mean(ious) if ious else None,
            "matched_continuous_iou_median": median(ious) if ious else None,
        },
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate one return and count its reviewed windows and polygons")
    validate.add_argument("--annotation", type=Path, required=True)
    agreement = commands.add_parser("agreement", help="validate both returns and compare their polygons")
    agreement.add_argument("--a", type=Path, required=True)
    agreement.add_argument("--b", type=Path, required=True)
    for command in (validate, agreement):
        command.add_argument("--roster", type=Path, required=True)
        command.add_argument("--output", type=Path, help="write a new report file in place of standard output")
    args = parser.parse_args()
    try:
        roster = read_roster(args.roster)
        if args.command == "validate":
            reviewer, items = read_review(args.annotation, roster)
            report = {"schema": "sar-hull-review-validation-v1", "review": review_summary(reviewer, items)}
        else:
            a_role, a = read_review(args.a, roster)
            b_role, b = read_review(args.b, roster)
            require((a_role, b_role) == ("A", "B"), "agreement", "--a must be reviewer A and --b reviewer B")
            report = agreement_report(roster, a, b)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
    except (OSError, ValueError, ShapelyError) as error:
        parser.exit(2, f"validation failed: {error}\n")


if __name__ == "__main__":
    main()
