"""Recheck closed FIT empty masks without opening a DEV or TEST image or geometry."""
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
from sys import version_info

from pycocotools import mask


D = Path(__file__).resolve().parent.parent
OWNER = D / "FIT_BILATERAL_AND_HRSID_OWNER_PRIVATE_20261109_01"
AUDIT = D / "HRSID_OFFICIAL_PNG_FIT_AUDIT_ACTUAL_PRIVATE_20261109_01"
TRAIN = Path("E:/SAR/seal/tmp/HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01/ROUND5_CURL_FINAL_TRAIN_20261015_01/PNG_OFFICIAL_annotations_train2017_ROUND5_COMPLETE.json")
ROSTER = Path("E:/SAR/seal/research_round_20260925_global/experiments/public_baseline_20260925/protocol/hrsid/image_group_assignment.json")
TRAIN_SHA = "3e240c38014526e9b93206a946105fb41c808272238932b0d4c2422978d5116f"
ROSTER_SHA = "c92b85dc278bc49cc8e34634fd7d354018844ae2a2692ea451b014a082bbaa5a"
PROTOCOL = "ca068f9258665b29337a5fb18d4da2e21c5fa729"
SIDE = 800
HALF = Decimal("0.5")


def need(check, message):
    if not check:
        raise ValueError(message)


def data(path, expected=None, size=None, decimal=False):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    need(expected is None or digest == expected, "Pinned input hash mismatch")
    need(size is None or len(raw) == size, "Pinned input size mismatch")
    return json.loads(raw, parse_float=Decimal if decimal else float), digest


def verify_closure():
    birth, _ = data(OWNER / "OWNER_BIRTH_PRIVATE.json")
    worker, _ = data(OWNER / "CONTROL/hrsid/LIVE_PRIVATE.json")
    final, _ = data(OWNER / "ORIGINAL_TERMINAL_PRIVATE.json")
    done, _ = data(OWNER / "CONTROL/hrsid/TERMINAL_PRIVATE.json")
    need(birth["pid"] == 49920 and worker["pid"] == 13492
         and birth["protocol_commit"] == final["protocol"] == PROTOCOL, "Original worker and owner identity")
    need(final["original_OS_waits"] is True and final["exits"]["hrsid"] == 0
         and final["automatic_retry"] is final["DEV"] is final["TEST"] is False
         and done["original_OS_wait"] is True and done["exit"] == 0 and done["guard"] is None,
         "Verified normal closure of original FIT audit")
    for name in ("stdout", "stderr"):
        ref = done[name + "_ref"]
        path = OWNER / "CONTROL/hrsid" / ("WORKER." + name)
        raw = path.read_bytes()
        need(Path(ref["path"]) == path and len(raw) == ref["bytes"] and sha256(raw).hexdigest() == ref["sha256"],
             "Exact original worker output evidence")
    report, digest = data(AUDIT / "REPORT_SAFE.json")
    last = json.loads((OWNER / "CONTROL/hrsid/WORKER.stdout").read_bytes().splitlines()[-1])
    need(report == last and report["status"] == "COMPLETED_FULL_FIT_WITH_EMPTY_RASTERS"
         and report["full_fit_image_and_raster_attempt_coverage"] is True
         and report["strict_input_qualified"] is report["model_executed"] is False
         and report["image_failures"] == report["fit_raster_check_failures_excluding_empty"] == 0
         and report["DEV_or_TEST_image_member_opened"] is report["separate_DEV_or_TEST_annotation_member_opened"] is False,
         "Same closed complete FIT census")
    findings, findings_sha = data(AUDIT / "FIT_FINDINGS_PRIVATE.json")
    need(len(findings) == report["empty_fit_rasters"] == 5 and all(x["kind"] == "EMPTY_COCO_RASTER" for x in findings),
         "Only five originally observed FIT empty rasters")
    return report, findings, digest, findings_sha


def edges(points):
    return list(zip(points, points[1:] + points[:1], strict=True))


def cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def on_edge(a, b, p):
    return (cross(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segment_contact(a, b, c, d):
    return (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
            or on_edge(a, b, c) or on_edge(a, b, d) or on_edge(c, d, a) or on_edge(c, d, b))


def continuous_area(points):
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in edges(points))) / 2


def center_counts(points):
    xlo, xhi = min(p[0] for p in points), max(p[0] for p in points)
    ylo, yhi = min(p[1] for p in points), max(p[1] for p in points)
    x_range = range(max(0, int((xlo - HALF).to_integral_value(rounding=ROUND_CEILING))),
                    min(SIDE - 1, int((xhi - HALF).to_integral_value(rounding=ROUND_FLOOR))) + 1)
    y_range = range(max(0, int((ylo - HALF).to_integral_value(rounding=ROUND_CEILING))),
                    min(SIDE - 1, int((yhi - HALF).to_integral_value(rounding=ROUND_FLOOR))) + 1)
    interior = boundary = 0
    for x in x_range:
        for y in y_range:
            p = (Decimal(x) + HALF, Decimal(y) + HALF)
            if any(on_edge(a, b, p) for a, b in edges(points)):
                boundary += 1
            elif sum((a[1] > p[1]) != (b[1] > p[1])
                     and (cross(a, b, p) > 0) == (b[1] > a[1]) for a, b in edges(points)) % 2:
                interior += 1
    return interior, boundary


def interval(values):
    return [float(min(values)), float(max(values))]


def inspect_one(row):
    polygon = row["segmentation"]
    need(len(polygon) == 1 and row["category_id"] == 1 and row["iscrowd"] == 0, "Original single ship polygon")
    poly = [Decimal(x) for x in polygon[0]]
    need(len(poly) >= 6 and len(poly) % 2 == 0 and all(x.is_finite() and 0 <= x <= SIDE for x in poly),
         "Finite original polygon on the native canvas")
    pts = list(zip(poly[::2], poly[1::2], strict=True))
    raw_vertices = len(pts)
    if pts[0] == pts[-1]:
        pts.pop()
    segments = edges(pts)
    contacts = sum(segment_contact(a, b, c, d) for i, (a, b) in enumerate(segments)
                   for j, (c, d) in enumerate(segments) if j > i + 1 and (i, j) != (0, len(segments) - 1))
    area = continuous_area(pts)
    x, y = (min(p[k] for p in pts) for k in (0, 1))
    x1, y1 = (max(p[k] for p in pts) for k in (0, 1))
    bbox = (x, y, x1 - x, y1 - y)
    upstream_box = tuple(Decimal(v) for v in row["bbox"])
    upstream_area = Decimal(row["area"])
    need(len(upstream_box) == 4 and all(v.is_finite() for v in upstream_box) and upstream_area.is_finite(),
         "Finite official bbox and metadata area")
    pixels = {}
    for scale in (1, 2, 4, 8):
        rles = mask.frPyObjects([[float(scale * v) for v in poly]], SIDE * scale, SIDE * scale)
        rle = mask.merge(rles)
        pixels[scale] = int(mask.area(rle))
        if scale == 1:
            need(int(mask.decode(rle).sum()) == pixels[scale] == 0, "Exact native COCO empty reproduced")
    interior, boundary = center_counts(pts)
    return dict(
        area=area, official_area=upstream_area, width=bbox[2], height=bbox[3],
        official_w=upstream_box[2], official_h=upstream_box[3], bbox_exact=bbox == upstream_box,
        bbox_max_abs_error=max(abs(a-b) for a,b in zip(bbox, upstream_box, strict=True)),
        official_area_abs_error=abs(area - upstream_area),
        duplicate_vertices=len(pts) != len(set(pts)), raw_vertices=raw_vertices,
        nonadjacent_edge_contacts=contacts, interior_centers=interior, boundary_centers=boundary,
        touches_canvas_boundary=0 in (x, y) or SIDE in (x1, y1),
        within_one_of_canvas_boundary=min(x, y, SIDE-x1, SIDE-y1) < 1,
        pixels=pixels,
    )


def main():
    need(version("pycocotools") == "2.0.11" and version_info[:3] == (3, 10, 13), "Pinned CPU raster runtime")
    report, findings, report_sha, findings_sha = verify_closure()
    train, train_sha = data(TRAIN, TRAIN_SHA, 15641078, decimal=True)
    assignment, assignment_sha = data(ROSTER, ROSTER_SHA)
    fit = {a["image_id"] for a in assignment if a["split"] == "fit"}
    selected = {(a["image_id"], a["annotation_id"]) for a in findings}
    need(len(selected) == 5 and {i for i, _ in selected} <= fit and len(fit) == 2937, "Only locked FIT scope")
    rows = [a for a in train["annotations"] if a["image_id"] in fit]
    counts = Counter(a["image_id"] for a in rows)
    need(len(rows) == 9265 and len(counts) == 2937, "Original FIT roster and annotation count")
    exact_rows = [a for a in rows if (a["image_id"], a["id"]) in selected]
    need(len(exact_rows) == len(selected) == 5, "Each originally empty FIT annotation exactly once")
    inspected = [inspect_one(a) for a in exact_rows]
    for i, _ in selected:
        counts[i] -= 1
    excess = [n - 100 for n in counts.values() if n > 100]
    need(sum(counts.values()) == report["nonempty_fit_rasters"] == 9260
         and max(counts.values()) == report["fit_max_nonempty_instances_per_decoded_image"] == 163
         and len(excess) == report["fit_decoded_images_above_100_nonempty_instances"] == 5,
         "Closed density counts reproduced with original five empty raster records")
    result = {
        "status": "ACTUAL_FIVE_OFFICIAL_HRSID_FIT_EMPTY_RASTERS_GEOMETRY_REVIEW_COMPLETE",
        "protocol_commit": PROTOCOL, "closed_audit_REPORT_SHA256": report_sha,
        "closed_audit_private_findings_SHA256": findings_sha,
        "official_TRAIN_SHA256": train_sha, "locked_FIT_assignment_SHA256": assignment_sha,
        "fit_images": len(fit), "original_fit_annotations": len(rows), "selected_empty_instances": len(inspected),
        "affected_fit_images": len({i for i, _ in selected}), "native_empty_reproduced": len(inspected),
        "positive_continuous_area_count": sum(r["area"] > 0 for r in inspected),
        "continuous_shoelace_area_px2_min_max": interval([r["area"] for r in inspected]),
        "official_area_px2_min_max": interval([r["official_area"] for r in inspected]),
        "official_area_vs_polygon_max_abs_difference_px2": float(max(r["official_area_abs_error"] for r in inspected)),
        "positive_official_bbox_count": sum(r["official_w"] > 0 and r["official_h"] > 0 for r in inspected),
        "official_bbox_min_side_px_min_max": interval([min(r["official_w"],r["official_h"]) for r in inspected]),
        "polygon_bbox_min_side_px_min_max": interval([min(r["width"],r["height"]) for r in inspected]),
        "official_bbox_matches_polygon_exact": sum(r["bbox_exact"] for r in inspected),
        "official_bbox_vs_polygon_max_abs_difference_px": float(max(r["bbox_max_abs_error"] for r in inspected)),
        "duplicate_nonclosing_vertex_instances": sum(r["duplicate_vertices"] for r in inspected),
        "nonadjacent_edge_contact_instances": sum(r["nonadjacent_edge_contacts"] > 0 for r in inspected),
        "contains_native_pixel_center_strict_interior": sum(r["interior_centers"] > 0 for r in inspected),
        "native_pixel_center_on_polygon_boundary": sum(r["boundary_centers"] > 0 for r in inspected),
        "touches_canvas_boundary": sum(r["touches_canvas_boundary"] for r in inspected),
        "within_one_pixel_of_canvas_boundary": sum(r["within_one_of_canvas_boundary"] for r in inspected),
        "positive_finer_grid_counts": {str(s): sum(r["pixels"][s] > 0 for r in inspected) for s in (2, 4, 8)},
        "finer_grid_scaled_back_area_px2_min_max": {str(s): interval([r["pixels"][s] / s**2 for r in inspected])
                                                     for s in (2, 4, 8)},
        "capacity_100_per_matching_group": {
            "nonempty_instances": sum(counts.values()), "images_above_cap": len(excess),
            "max_nonempty_instances_per_image": max(counts.values()),
            "minimum_unassigned_per_group_over_FIT": sum(excess),
            "minimum_unassigned_over_affected_images_min_max": [min(excess), max(excess)],
        },
        "separate_DEV_or_TEST_member_opened": False, "DEV_or_TEST_geometry_computed": False,
        "archive_or_image_bytes_opened": False, "source_records_modified": False,
        "higher_resolution_is_diagnostic_only": True, "model_executed": False, "AP": False,
        "runtime": {"pycocotools": version("pycocotools"), "python": ".".join(map(str, version_info[:3]))},
    }
    output = Path(__file__).with_name("EMPTY_CAUSE_SAFE.json")
    with output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

