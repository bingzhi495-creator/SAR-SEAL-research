"""Score frozen edits against released UEIKAP hull polygons already matched by the native detector."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np
from pycocotools import mask as codec


HEIGHT = WIDTH = 512
IMAGES, LABELS, GROUPS = 108, 139, 18
ARMS = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count",
        "attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
CONTRASTS = (("free_matched_count", "fixed"), ("free_natural", "raw"),
             ("free_natural", "native_matched_count"),
             ("learned_at_attraction_count", "attraction_dynamic"),
             ("learned_at_attraction_count", "attraction_static_at_dynamic_count"))
PRIMARY = "free_matched_count minus fixed"
MATCH_IOU = 0.50
BOOTSTRAP_DRAWS = 4096
BOOTSTRAP_SEED = 5217
MIN_EXPOSED_HULLS, MIN_EXPOSED_GROUPS = 20, 8
RESERVATION_SHA = "1f3b4d3bf0ff8f0ed1767cfdd796b541e60f2a4c7d76b03c316f0b1e81dbded1"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read_pinned(path, expected):
    check(isinstance(expected, str) and len(expected) == 64, "Expected SHA-256 required")
    data = path.read_bytes()
    check(sha256(data).hexdigest() == expected, f"SHA-256 mismatch for {path.name}")
    return data


def json_pinned(path, expected):
    return json.loads(read_pinned(path, expected))


def stream_pinned(path, expected):
    return [json.loads(line) for line in read_pinned(path, expected).splitlines()]


def label_path(root, reference):
    root = root.resolve(strict=True)
    path = (root / reference["path"]).resolve(strict=True)
    check(root in path.parents, "Label member must be contained in the accepted intake")
    raw = read_pinned(path, reference["sha256"])
    check(len(raw) == reference["bytes"], "Label member byte count mismatch")
    return raw


def parse_yolo(data, height, width, shapely):
    """Use every class-0 polygon in release order and leave wake instances out."""
    hulls = []
    counts = Counter()
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        label, *coordinates = line.split()
        check(label in ("0", "1") and len(coordinates) >= 6 and len(coordinates) % 2 == 0,
              "Unexpected UEIKAP YOLO polygon row")
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
        check(np.all(np.isfinite(points)) and np.all((points >= 0) & (points <= 1)),
              "Normalized polygon outside the author PNG")
        counts[label] += 1
        if label == "1":
            continue
        scaled = points * (width, height)
        polygon = shapely.Polygon(scaled)
        check(not polygon.is_empty and polygon.is_valid and polygon.area > 0,
              "Invalid native class-0 polygon. Preserve the cohort and investigate the label")
        raster = codec.merge(codec.frPyObjects([scaled.ravel().tolist()], height, width))
        check(int(codec.area(raster)) > 0, "A released hull rasterized to an empty mask")
        hulls.append((polygon, raster))
    return hulls, dict(counts)


def checked_rle(rle, height, width):
    check(set(rle) == {"size", "counts"} and rle["size"] == [height, width] and
          isinstance(rle["counts"], str) and bool(rle["counts"]), "Invalid native RLE geometry")
    return rle


def match_native(rasters, truth_masks, threshold=MATCH_IOU):
    """One COCO-style greedy matching pass at the frozen threshold and native rank."""
    if not rasters or not truth_masks:
        return {}
    overlaps = codec.iou(rasters, truth_masks, [False] * len(truth_masks))
    matches = {}
    for rank, row in enumerate(overlaps, start=1):
        best_index, best_iou = None, threshold
        for index, overlap in enumerate(row):
            if index not in matches and overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None:
            matches[best_index] = rank
    return matches


def geometric_iou(rle, polygon, shapely):
    """Intersect the released polygon with an exact union of disjoint unit-cell runs."""
    bitmap = codec.decode(rle).astype(np.int8)
    boundaries = np.diff(bitmap, axis=1, prepend=0, append=0)
    rows, starts = np.where(boundaries == 1)
    ending_rows, ends = np.where(boundaries == -1)
    check(np.array_equal(rows, ending_rows) and np.all(ends > starts), "Invalid foreground runs")
    area = int(codec.area(rle))
    check(int((ends - starts).sum()) == area, "RLE and unit-cell areas differ")
    if not area:
        return 0.0
    rectangles = shapely.box(starts, rows, ends, rows + 1)
    intersection = float(shapely.area(shapely.intersection(rectangles, polygon)).sum())
    check(0 <= intersection <= min(area, polygon.area) + 1e-8, "Invalid polygon intersection area")
    return intersection / (area + polygon.area - intersection)


def verify_native(prediction, audit, image_id, height, width):
    check(prediction["image_id"] == audit["image_id"] == image_id, "Native stream image order mismatch")
    rows, candidates = prediction["coco"], audit["candidates"]
    check(len(rows) == len(candidates), "Invalid full native candidate audit population")
    masks = []
    for rank, (row, item) in enumerate(zip(rows, candidates), start=1):
        check(row["image_id"] == image_id and row["category_id"] == 1 and item["native_rank"] == rank and
              math.isfinite(row["score"]) and row["score"] == item["score"], "Frozen native candidate identity mismatch")
        mask = checked_rle(row["segmentation"], height, width)
        check(int(codec.area(mask)) == item["mask_area"], "Native candidate area mismatch")
        masks.append(mask)
    check(all(x["score"] >= y["score"] for x, y in zip(rows, rows[1:])), "Native ranks must follow stable descending scores")
    return masks[:100]


def verify_edits(changes, masks, height, width):
    """Recheck the conditionally scored edit budget without predictor access."""
    check(isinstance(changes, dict), "Invalid edit rows")
    verified = {}
    for rank_text, updates in changes.items():
        rank = int(rank_text)
        check(rank_text == str(rank) and 1 <= rank <= len(masks) and set(updates) <= set(ARMS),
              "Edit rank or arm outside frozen support")
        stock = masks[rank - 1]
        area = int(codec.area(stock))
        verified[rank] = {}
        for arm, edit in updates.items():
            swaps = edit["swaps"]
            edited = checked_rle(edit["rle"], height, width)
            check(type(swaps) is int and 0 <= swaps <= 8 and (arm != "raw" or swaps == 0),
                  "Invalid released edit count")
            intersection = int(codec.area(codec.merge([stock, edited], intersect=True)))
            check(int(codec.area(edited)) == area and area - intersection == swaps,
                  "Edit does not preserve exact foreground area and balanced XOR")
            check(not swaps or 32 <= area < 1024, "Executed edit outside the frozen native area rule")
            verified[rank][arm] = (swaps, edited)
        counts = {arm: verified[rank].get(arm, (0,))[0] for arm in ARMS}
        check(counts["fixed"] == counts["free_matched_count"] and
              counts["free_natural"] == counts["native_matched_count"] and
              counts["attraction_dynamic"] == counts["learned_at_attraction_count"] ==
              counts["attraction_static_at_dynamic_count"], "Paired edit counts differ")
    return verified


def paired(groups):
    """Group means followed by paired bootstrap over the observed source groups."""
    scene_values = np.asarray([np.mean(groups[key], axis=0) for key in sorted(groups)], dtype=np.float64)
    check(scene_values.ndim == 2 and scene_values.shape[1] == len(ARMS), "Invalid group effects")
    differences = np.column_stack([scene_values[:, ARMS.index(left)] - scene_values[:, ARMS.index(right)]
                                   for left, right in CONTRASTS])
    keys = [left + " minus " + right for left, right in CONTRASTS]
    n_groups = len(scene_values)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(n_groups, size=(BOOTSTRAP_DRAWS, n_groups))
    intervals = np.percentile(differences[draws].mean(axis=1), [2.5, 97.5], axis=0) * 100
    return {
        "source_groups": n_groups,
        "arm_mean_iou_points": dict(zip(ARMS, (scene_values.mean(axis=0) * 100).tolist())),
        "contrasts": {key: {"mean_points": float(100 * differences[:, index].mean()),
                             "paired_95_interval_points": intervals[:, index].tolist(),
                             "groups_improved": int(np.sum(differences[:, index] > 0)),
                              "groups_worse": int(np.sum(differences[:, index] < 0)),
                              "groups_equal": int(np.sum(differences[:, index] == 0))}
                      for index, key in enumerate(keys)},
    }


def verify_reservation(reservation, label_rows, members):
    """Bind the admitted metadata to the old ordered TIFF-based reservation before reading labels."""
    records = reservation["items"]
    check(reservation["held_group_count"] == GROUPS and reservation["held_image_count"] == IMAGES and
          reservation["held_class0_candidate_polygon_count"] == LABELS and len(records) == len(label_rows) == IMAGES,
          "Unexpected prior reservation scope")
    original_groups = [row["acquisition_group"] for row in records]
    check(set(original_groups) == set(reservation["held_groups"]) and len(set(original_groups)) == GROUPS,
          "Unexpected prior acquisition family partition")
    check(isinstance(members, list) and len(members) == 2 * IMAGES, "Exact reserved member provenance count")
    label_members = [row for row in members if "/labels/" in row["zip_member"] and row["zip_member"].endswith(".txt")]
    check(len(label_members) == IMAGES and len({row["image_id"] for row in label_members}) == IMAGES,
          "Missing or duplicate author label member provenance")
    by_image = {row["image_id"]: row for row in label_members}
    for image_id, (old, new) in enumerate(zip(records, label_rows), start=1):
        member = by_image[image_id]
        expected = PurePosixPath(old["tiff_zip_name"].replace("/images_tifs/", "/labels/")).with_suffix(".txt")
        check(new["image_id"] == image_id and new["acquisition_group"] == old["acquisition_group"] and
              new["count_by_class"].get("0", 0) == old["class0_candidate_polygons"] and
              member["zip_member"] == str(expected) and member["sha256"] == new["label_ref"]["sha256"] and
              member["bytes"] == new["label_ref"]["bytes"], "Accepted label roster differs from old reservation")


def area_summary(areas):
    names = ("minimum", "first_quartile", "median", "third_quartile", "maximum")
    return dict(zip(names, np.percentile(areas, [0, 25, 50, 75, 100]).tolist())) if areas else None


def execute(intake, terminal_sha, reservation_path, capture, replay, receipt_path, receipt_sha, shapely):
    intake_terminal = json_pinned(intake / "intake_terminal.json", terminal_sha)
    check(intake_terminal["status"] == "completed_verified" and intake_terminal["actual_exit"] == 0,
          "Original author input intake did not close")
    label_manifest = json_pinned(intake / "labels_roster.json", intake_terminal["labels_roster_sha256"])
    image_manifest = json_pinned(intake / "images_roster.json", intake_terminal["images_roster_sha256"])
    members = json_pinned(intake / "member_provenance.json", intake_terminal["member_provenance_sha256"])
    reservation = json_pinned(reservation_path, RESERVATION_SHA)
    verify_reservation(reservation, label_manifest["items"], members)
    receipt = json_pinned(receipt_path, receipt_sha)
    check(receipt["status"] == "UEIKAP_DFINE_RESERVED_EDIT_REPLAY_ACCEPTED_V1" and
          receipt["exact_replayed_policy_choices"] is True and receipt["exact_support_xor_area"] is True and
          receipt["model_forwards"] == 0 and receipt["optimizer_updates"] == 0 and
          receipt["manifest_sha256"] == intake_terminal["images_roster_sha256"] and
          receipt["author_TEST"] is False, "Independent closed image-only edit replay required")
    predictions = stream_pinned(capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", receipt["predictions_sha256"])
    audits = stream_pinned(capture / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", receipt["candidate_audit_sha256"])
    edits = stream_pinned(replay / "EDITS_PRIVATE.jsonl", receipt["edits_sha256"])
    label_rows, image_rows = label_manifest["items"], image_manifest["items"]
    roster = list(range(1, IMAGES + 1))
    check(label_manifest["expected_ids"] == image_manifest["expected_ids"] == roster and
          all(len(rows) == IMAGES for rows in (label_rows, image_rows, predictions, audits, edits)),
          "Exact reserved image population required")
    counts = Counter({"labeled_hulls": 0, "native_matched": 0, "native_unmatched": 0,
                      "native_area_below_32": 0, "native_area_at_least_1024": 0,
                      "eligible_area_sf_zero": 0, "sf_positive": 0})
    all_groups, matched_groups = defaultdict(list), defaultdict(list)
    exposed_groups = set()
    native_areas = []
    transitions = {str(threshold): {arm: Counter(retained=0, lost=0, gained=0, both_below=0) for arm in ARMS}
                   for threshold in (0.75, 0.90)}
    seen_per_arm = Counter({arm: 0 for arm in ARMS})
    for image_id, im, label, pred, audit, edit in zip(roster, image_rows, label_rows, predictions, audits, edits):
        check(im["image_id"] == label["image_id"] == edit["image_id"] == image_id and
              im["original_hw"] == [HEIGHT, WIDTH], "Image, label and edit roster mismatch")
        group = label["acquisition_group"]
        check(isinstance(group, str) and bool(group), "Missing source family")
        truth, label_counts = parse_yolo(label_path(intake, label["label_ref"]), HEIGHT, WIDTH, shapely)
        check(label_counts == label["count_by_class"] and truth, "Exact author label count mismatch")
        masks = verify_native(pred, audit, image_id, HEIGHT, WIDTH)
        updates = verify_edits(edit["changes"], masks, HEIGHT, WIDTH)
        matches = match_native(masks, [raster for _, raster in truth])
        counts.update(labeled_hulls=len(truth), wake_rows_excluded=label_counts.get("1", 0),
                      native_matched=len(matches), native_unmatched=len(truth)-len(matches))
        for index, (polygon, _) in enumerate(truth):
            if index not in matches:
                all_groups[group].append([0.0] * len(ARMS))
                continue
            rank = matches[index]
            original = masks[rank - 1]
            area = int(codec.area(original))
            native_areas.append(area)
            variants = updates.get(rank, {})
            if area < 32:
                counts["native_area_below_32"] += 1
            elif area >= 1024:
                counts["native_area_at_least_1024"] += 1
            elif variants.get("fixed", (0,))[0] > 0:
                counts["sf_positive"] += 1
                exposed_groups.add(group)
            else:
                counts["eligible_area_sf_zero"] += 1
            cache = {}
            values = []
            for arm in ARMS:
                swaps, mask = variants.get(arm, (0, original))
                if swaps:
                    seen_per_arm[arm] += 1
                key = mask["counts"]
                if key not in cache:
                    cache[key] = geometric_iou(mask, polygon, shapely)
                values.append(cache[key])
            for threshold, arms in transitions.items():
                cutoff = float(threshold)
                was = values[ARMS.index("raw")] >= cutoff
                for arm, value in zip(ARMS, values):
                    now = value >= cutoff
                    event = ("retained" if now else "lost") if was else ("gained" if now else "both_below")
                    arms[arm][event] += 1
            all_groups[group].append(values)
            matched_groups[group].append(values)
    check(counts["labeled_hulls"] == LABELS and len(all_groups) == GROUPS and
          counts["native_unmatched"] + counts["native_matched"] == LABELS and
          sum(counts[key] for key in ("native_area_below_32", "native_area_at_least_1024",
                                      "eligible_area_sf_zero", "sf_positive")) == counts["native_matched"],
          "Reserved hull, acquisition group or eligibility totals differ")
    main = paired(all_groups)
    effect = main["contrasts"][PRIMARY]
    coverage = counts["sf_positive"] >= MIN_EXPOSED_HULLS and len(exposed_groups) >= MIN_EXPOSED_GROUPS
    return {
        "status": "UEIKAP_RESERVED_ANNOTATED_HULL_SCORE_COMPLETE_V1", "whole_scene_AP": False,
        "optimizer_updates": 0, "model_forwards": 0, "images": IMAGES,
        "labeled_hulls_and_coverage": dict(counts, sf_positive_source_groups=len(exposed_groups),
                                           originally_matched_source_groups=len(matched_groups)),
        "native_matched_mask_area_px_five_number_summary": area_summary(native_areas),
        "primary_estimand": PRIMARY, "primary_all_labeled_hulls": main,
        "originally_matched_hulls_sensitivity": paired(matched_groups) if matched_groups else None,
        "edited_among_original_matches_by_arm": dict(seen_per_arm),
        "continuous_iou_threshold_transitions_on_original_matches": transitions,
        "coverage_gate_met": coverage, "frozen_geometry_advance": coverage and effect["paired_95_interval_points"][0] > 0,
        "inputs": {"intake_terminal_sha256": terminal_sha, "image_roster_sha256": receipt["manifest_sha256"],
                   "labels_roster_sha256": intake_terminal["labels_roster_sha256"],
                   "member_provenance_sha256": intake_terminal["member_provenance_sha256"],
                   "old_reservation_sha256": RESERVATION_SHA,
                   "accepted_image_only_replay_sha256": receipt_sha, "scorer_sha256": sha256(Path(__file__).read_bytes()).hexdigest()},
        "geometry_backend": {"shapely": shapely.__version__, "geos": shapely.geos_version_string},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("intake", "reservation", "capture", "replay", "readback", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("intake-terminal-sha", "readback-sha"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--shapely-site", type=Path)
    args = parser.parse_args()
    if args.shapely_site:
        sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely

    check(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
          "Use the original frozen geometry runtime")
    check(not args.output.exists(), "Use a fresh output directory")
    result = execute(args.intake, args.intake_terminal_sha, args.reservation, args.capture, args.replay,
                     args.readback, args.readback_sha, shapely)
    args.output.mkdir()
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (args.output / "RESULT_SAFE.json").open("xb") as handle:
        handle.write(data)
    print(json.dumps({"status": result["status"], "result_sha256": sha256(data).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
