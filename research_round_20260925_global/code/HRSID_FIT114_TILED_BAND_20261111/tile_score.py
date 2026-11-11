"""Score frozen FIT tile mask edits on the original full-candidate assignment."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import ModuleType

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

QUOTA = 2
SIZE = 800
CROP = 400
IMAGES = 16
MIB = 1024**2
SELECTION_SALT = b"P326-HRSID-FIT114-TILED-BAND-v1\0"
RANDOM_SALT = b"P327-HRSID-FIT114-TILED-RANDOM-v1\0"
P317_SHA256 = "261f533956651b6cd3868ebfa91624ed5a6b8efa7b40b69cb26254d3b09dd05d"
PRIMARY = ("raw", "native_crop", "lowpass_crop", "abs_full_logit", "random", "gt_oracle")
SECONDARY = ("native_crop", "lowpass_crop")
CANDIDATE_FIELDS = ("native_rank", "query_id", "score", "category_id", "segmentation")
POINT_FIELDS = ("y", "x", "inside", "full", "native_crop", "lowpass_crop")
WEIGHTS = {75: (4, 3), 90: (10, 9)}
EDIT_FIELDS = (
    "edited_candidates", "add", "delete", "edited_unmatched", "unmatched_add",
    "unmatched_delete", "matched_correct_add", "matched_correct_delete",
    "matched_wrong_add", "matched_wrong_delete",
)


def equal_edits(priority, inside, pixels):
    """Take up to two actions on each side, independently of score sign."""
    priority = np.asarray(priority)
    chosen = []
    for side in (False, True):
        available = np.flatnonzero(inside == side)
        ranked = np.lexsort((pixels[available], -priority[available]))
        chosen.extend(available[ranked[:QUOTA]])
    return np.asarray(chosen, dtype=np.intp)


def oracle_edits(inside, truth, pixels):
    """Place all possible correct edits first, maximizing I and minimizing U."""
    return equal_edits((inside != truth).astype(np.int8), inside, pixels)


def candidate_geometry(mask):
    """Recover the frozen prediction-only crop and complete in-canvas band."""
    if mask.shape != (SIZE, SIZE):
        raise ValueError("Exact original-grid prediction required")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0 = int(np.clip((xs.min() + xs.max() + 1 - CROP)//2, 0, SIZE-CROP))
    y0 = int(np.clip((ys.min() + ys.max() + 1 - CROP)//2, 0, SIZE-CROP))
    mask = mask.astype(bool, copy=False)
    band = (maximum_filter(mask, size=5, mode="constant", cval=0)
            & ~minimum_filter(mask, size=5, mode="constant", cval=0))
    points = np.column_stack(np.nonzero(band))
    if (not len(points) or np.any(points < [y0, x0])
            or np.any(points >= [y0+CROP, x0+CROP])):
        return None
    return [x0, y0, x0+CROP, y0+CROP], points


def score_candidate(counter, selected, inside, truth, intersection=0, union=0):
    """Update exact error and threshold counters for a fixed mask/target pair."""
    sides = inside[selected]
    adds, deletes = int((~sides).sum()), int(sides.sum())
    counter["edited_candidates"] += bool(len(selected))
    counter["add"] += adds
    counter["delete"] += deletes
    if truth is None:
        counter["edited_unmatched"] += bool(len(selected))
        counter["unmatched_add"] += adds
        counter["unmatched_delete"] += deletes
        return None
    target = truth[selected]
    correct_add = int((~sides & target).sum())
    correct_delete = int((sides & ~target).sum())
    wrong_add, wrong_delete = adds - correct_add, deletes - correct_delete
    counter["matched_correct_add"] += correct_add
    counter["matched_correct_delete"] += correct_delete
    counter["matched_wrong_add"] += wrong_add
    counter["matched_wrong_delete"] += wrong_delete
    delta_i = correct_add - wrong_delete
    delta_u = wrong_add - correct_delete
    new_i, new_u = intersection + delta_i, union + delta_u
    if not 0 <= new_i <= new_u or new_u <= 0:
        raise ValueError("Edited exact mask counts are inconsistent")
    counter["fixed_iou_sum"] += new_i / new_u
    for threshold, (q, p) in WEIGHTS.items():
        before = q * intersection >= p * union
        after = q * new_i >= p * new_u
        counter[f"margin_gain_{threshold}"] += q * delta_i - p * delta_u
        counter[f"recover_{threshold}"] += not before and after
        counter[f"lose_{threshold}"] += before and not after
    return new_i, new_u


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_p317(path):
    source = path.read_bytes()
    require(sha256(source).hexdigest() == P317_SHA256, "Exact P317 source required")
    module = ModuleType("p317_fit")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def read_tile_inputs(report_path, report_sha, stream_path, capture_sha):
    require(report_path.stat().st_size <= MIB, "Bounded capture report required")
    report_bytes = report_path.read_bytes()
    require(sha256(report_bytes).hexdigest() == report_sha, "Exact tile capture report required")
    report = json.loads(report_bytes)
    expected = {
        "status": "P326_HRSID_FIT114_TILED_BAND_CAPTURE_COMPLETE",
        "p316_candidate_stream_sha256": capture_sha, "images": IMAGES,
        "native_full_replay_exact": True, "GT_opened": False, "DEV": False,
        "TEST": False, "training": False, "optimizer_updates": 0, "forward_calls": 48,
    }
    require(all(report.get(k) == v and type(report.get(k)) is type(v) for k, v in expected.items()),
            "Exact complete FIT-only P326 capture scope required")
    for name in ("p316_report_sha256", "p316_original_wait_sha256", "tiled_band_stream_sha256"):
        require(re.fullmatch("[0-9a-f]{64}", report.get(name, "")) is not None,
                "Missing capture provenance digest")
    count = report.get("tiled_band_stream_bytes")
    require(type(count) is int and 0 < count <= 32*MIB and stream_path.stat().st_size == count,
            "Bounded exact tile band stream size required")
    data = stream_path.read_bytes()
    require(len(data) == count and sha256(data).hexdigest() == report["tiled_band_stream_sha256"]
            and data.endswith(b"\n"), "Closed tile band stream differs")
    lines = data.splitlines()
    require(len(lines) == IMAGES and all(0 < len(line) <= 4*MIB for line in lines),
            "Exact bounded tile row count required")
    return report, [json.loads(line) for line in lines]


def select_candidates(p317, capture_path, capture_sha, cohort):
    """Reproduce P326 image and candidate selection without opening GT."""
    eligible = []
    with capture_path.open("rb") as stream:
        require(0 < os.fstat(stream.fileno()).st_size <= 256*MIB, "Bounded P316 capture required")
        for image in p317.stream_capture(stream, cohort, capture_sha):
            if not image["core"] or image["stress"]:
                continue
            candidates = image["candidates"]
            predictions, areas = p317.image_masks(candidates)
            for i, area in enumerate(areas):
                if not 20 <= area <= 1023:
                    continue
                mask = p317.codec.decode(predictions[i]).astype(bool, copy=False)
                geometry = candidate_geometry(mask)
                if geometry is None:
                    continue
                crop, points = geometry
                side = mask[points[:, 0], points[:, 1]]
                identity = {key: candidates[i][key] for key in CANDIDATE_FIELDS}
                eligible.append(dict(
                    image_id=image["image_id"], group=image["group"], encoded_sha256=image["encoded_sha256"],
                    candidate=identity, crop_xyxy=crop, points=points, inside=side,
                    predictions=predictions, areas=areas, index=i,
                ))
                break
    eligible.sort(key=lambda row: (
        sha256(SELECTION_SALT + bytes.fromhex(row["encoded_sha256"])).digest(), row["group"], row["image_id"],
    ))
    selected, groups, encodings = [], set(), set()
    for row in eligible:
        group, encoded = row["group"], row["encoded_sha256"]
        if group in groups or encoded in encodings:
            continue
        selected.append(row)
        groups.add(group)
        encodings.add(encoded)
        if len(selected) == IMAGES:
            break
    require(len(selected) == IMAGES, "Exact 16 prediction-selected core images required")
    return selected


def tile_fields(row, selected):
    """Verify the complete ordered tile band against the original prediction."""
    identity = ("image_id", "group", "encoded_sha256", "crop_xyxy", "candidate")
    require(row.keys() == {*identity, "points"} and all(row[k] == selected[k] for k in identity),
            "Tile image, crop, rank, query, score or original RLE differs")
    raw = row["points"]
    require(type(raw) is list and len(raw) == len(selected["points"]), "Exact full tile band required")
    require(all(point.keys() == set(POINT_FIELDS) and type(point["inside"]) is bool
                and type(point["y"]) is type(point["x"]) is int
                and all(type(point[k]) in (float, int) for k in POINT_FIELDS[3:]) for point in raw),
            "Exact finite signed tile point schema required")
    points = np.asarray([(point["y"], point["x"]) for point in raw], dtype=np.intp)
    inside = np.asarray([point["inside"] for point in raw], dtype=bool)
    values = np.asarray([[point[k] for k in POINT_FIELDS[3:]] for point in raw], dtype=np.float64)
    require(np.array_equal(points, selected["points"]) and np.array_equal(inside, selected["inside"])
            and np.isfinite(values).all() and np.array_equal(values[:, 0] > 0, inside),
            "Tile coordinate order, band support or original signed logits differ")
    return points, inside, values


def random_priority(encoded, query, pixels):
    seed = RANDOM_SALT + bytes.fromhex(encoded) + int(query).to_bytes(2, "big")
    return np.asarray([int.from_bytes(sha256(seed + int(pixel).to_bytes(4, "big")).digest()[:6], "big")
                       for pixel in pixels], dtype=np.int64)


def edit_arms(inside, values, pixels, selected, truth):
    full, native, lowpass = values.T
    primary = {
        "raw": np.empty(0, dtype=np.intp),
        "native_crop": equal_edits(np.where(inside, -native, native), inside, pixels),
        "lowpass_crop": equal_edits(np.where(inside, -lowpass, lowpass), inside, pixels),
        "abs_full_logit": equal_edits(-np.abs(full), inside, pixels),
        "random": equal_edits(random_priority(selected["encoded_sha256"], selected["candidate"]["query_id"], pixels),
                              inside, pixels),
    }
    if truth is not None:
        primary["gt_oracle"] = oracle_edits(inside, truth, pixels)
    secondary = {
        "native_crop": np.flatnonzero((native > 0) != inside),
        "lowpass_crop": np.flatnonzero((lowpass > 0) != inside),
    }
    return primary, secondary


def score_tiles(p317, selected_rows, tile_rows, ground_truth):
    primary = {name: Counter() for name in PRIMARY}
    secondary = {name: Counter() for name in SECONDARY}
    census = Counter(images=len(selected_rows))
    paired = {name: Counter() for name in PRIMARY if name != "native_crop"}
    for selected, tile in zip(selected_rows, tile_rows, strict=True):
        points, inside, values = tile_fields(tile, selected)
        ys, xs = points.T
        pixels = ys * SIZE + xs
        index = selected["index"]
        predictions = selected["predictions"]
        gt = ground_truth[selected["image_id"]]
        matches = p317.original_matches(predictions, gt)
        target = matches.get(index)
        intersection = union = 0
        truth = None
        census.update(points=len(points), original_all_candidates=len(predictions),
                      original_all_fixed_matches=len(matches), eligible_gt=len(gt),
                      selected_fixed_matches=int(target is not None),
                      selected_unmatched=int(target is None))
        if target is not None:
            prediction = predictions[index]
            truth = p317.codec.decode(target)[ys, xs].astype(bool, copy=False)
            intersection = int(p317.codec.area(p317.codec.merge([prediction, target], intersect=True)))
            union = int(selected["areas"][index]) + int(p317.codec.area(target)) - intersection
            require(2*intersection >= union > 0, "Original fixed mask match below .50")
            for threshold, (q, p) in WEIGHTS.items():
                census[f"original_at_{threshold}"] += q * intersection >= p * union
        equal, signed = edit_arms(inside, values, pixels, selected, truth)
        outcomes = {name: score_candidate(primary[name], edits, inside, truth, intersection, union)
                    for name, edits in equal.items()}
        for name, edits in signed.items():
            score_candidate(secondary[name], edits, inside, truth, intersection, union)
        if target is not None:
            native_i, native_u = outcomes["native_crop"]
            oracle_i, oracle_u = outcomes["gt_oracle"]
            for name, outcome in outcomes.items():
                i, u = outcome
                if name != "raw":
                    require(oracle_i >= i and oracle_u <= u, "Fixed-quota oracle must dominate on raster counts")
                if name != "native_crop":
                    diff = native_i * u - i * native_u
                    paired[name]["native_iou_win"] += diff > 0
                    paired[name]["native_iou_tie"] += diff == 0
                    paired[name]["native_iou_loss"] += diff < 0
    matched = census["selected_fixed_matches"]
    for arm in ("lowpass_crop", "abs_full_logit", "random"):
        require(all(primary[arm][key] == primary["native_crop"][key]
                    for key in ("add", "delete", "edited_candidates", "unmatched_add", "unmatched_delete")),
                "Primary candidate action budgets differ")
    return dict(
        census=dict(census),
        primary={name: summarize_arm(counter, matched) for name, counter in primary.items()},
        secondary_direct_sign={name: summarize_arm(counter, matched) for name, counter in secondary.items()},
        primary_native_paired_iou={name: dict(value) for name, value in paired.items()},
    )


def summarize_arm(counter, matched):
    result = {field: counter[field] for field in EDIT_FIELDS}
    result["pixels_changed"] = counter["add"] + counter["delete"]
    result["pixels_changed_unmatched"] = counter["unmatched_add"] + counter["unmatched_delete"]
    result["mean_fixed_iou"] = counter["fixed_iou_sum"] / matched if matched else None
    for threshold in WEIGHTS:
        for field in ("recover", "lose", "margin_gain"):
            key = f"{field}_{threshold}"
            result[key] = counter[key]
        result[f"net_{threshold}"] = counter[f"recover_{threshold}"] - counter[f"lose_{threshold}"]
    return result


def run(args):
    for digest in (args.capture_sha256, args.tile_report_sha256):
        require(re.fullmatch("[0-9a-f]{64}", digest) is not None, "Exact input digest required")
    p317 = load_p317(args.p317_source)
    report, tiles = read_tile_inputs(args.tile_report, args.tile_report_sha256, args.tile_stream, args.capture_sha256)
    cohort, index = p317.load_scope(args.bundle)
    selected = select_candidates(p317, args.capture, args.capture_sha256, cohort)
    truth = p317.ground_truth(index, cohort)
    return dict(
        status="P327_HRSID_FIT114_TILED_BAND_FIXED_MATCH_SCORE_COMPLETE",
        p317_source_sha256=P317_SHA256, p326_report_sha256=args.tile_report_sha256,
        p326_stream_sha256=report["tiled_band_stream_sha256"], p316_candidate_stream_sha256=args.capture_sha256,
        fit_bundle_sha256=p317.BUNDLE_SHA, fit_cohort_sha256=p317.COHORT_SHA, fit_index_sha256=p317.INDEX_SHA,
        scope=dict(FIT=True, DEV=False, TEST=False, AP=False, training=False, model_forward_calls=0,
                   optimizer_updates=0, exploratory_prediction_selected=True, fixed_original_match_iou=0.50,
                   candidate_query_box_score_order_frozen=True, primary_quota_per_side=QUOTA,
                   secondary_equal_budget=False, gt_oracle_matched_only=True, independent_efficacy_claim=False),
        **score_tiles(p317, selected, tiles, truth),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("p317-source", "bundle", "capture", "tile-stream", "tile-report", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--capture-sha256", required=True)
    cli.add_argument("--tile-report-sha256", required=True)
    args = cli.parse_args()
    for path in (args.p317_source, args.bundle, args.capture, args.tile_stream, args.tile_report):
        require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
                "Canonical input file required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output file required")
    result = run(args)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
