"""Admit paired FIT source polygons in 36 closed, exact RGB-overlap crops."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


PROTOCOL_SHA = "56c0d7949fbffad3cfa0b0f77a47bfbdc89a491c10c33c15d596b2606d3c7e01"
PAIR_SHA = "1ba947c401098e92737300d8715d523a890c414fc2e1795e7443530149b81024"
RGB_SHA = "622481efd470b5c2549dc07f11acb0e584b14d6cc8eb45399bdf9379b8a6cbb8"
LABEL_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
CROP = box(0, 0, 800, 800)


def need(valid, message):
    if not valid:
        raise ValueError(message)


def read_exact(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Unexpected closed digest: " + path.name)
    return json.loads(raw)


def write_new(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with path.open("xb") as file:
        file.write(raw)
    return sha256(raw).hexdigest()


def distance(polygon, window):
    top, bottom, left, right = window
    x0, y0, x1, y1 = polygon.bounds
    return min(x0 - left, y0 - top, right - x1, bottom - y1)


def exact_polygon(segmentation):
    if not isinstance(segmentation, list) or not segmentation:
        return None
    parts = []
    for component in segmentation:
        if (not isinstance(component, list) or len(component) < 6 or len(component) % 2 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in component)):
            return None
        geom = Polygon(list(zip(component[::2], component[1::2], strict=True)))
        if geom.is_empty or not geom.is_valid or geom.area <= 0:
            return None
        parts.append(geom)
    union = unary_union(parts)
    return union if union.is_valid and union.area > 0 and union.geom_type in ("Polygon", "MultiPolygon") else None


def eligible_polygons(annotations, ship_category, window, shared):
    counts = Counter()
    retained = []
    for ann in annotations:
        if ann["category_id"] != ship_category:
            counts["nonship"] += 1
            continue
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            counts["crowd_or_ignore"] += 1
            continue
        source = exact_polygon(ann.get("segmentation"))
        if source is None:
            counts["invalid_source_polygon"] += 1
            continue
        clipped = source.intersection(CROP)
        if not clipped.equals(source) or distance(clipped, (0, 800, 0, 800)) <= 0:
            counts["touches_or_crosses_own_canvas"] += 1
            continue
        mother = translate(clipped, xoff=window[2], yoff=window[0])
        if distance(mother, shared) <= 0:
            counts["not_fully_inside_shared_viewport"] += 1
            continue
        retained.append(mother)
        counts["full_hull_candidate"] += 1
    return retained, counts


def overlaps(a, b):
    values = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            if p.intersects(q):
                common = p.intersection(q).area
                values[i, j] = common / (p.area + q.area - common)
    return values


def reciprocal(matrix):
    """Classify row endpoints and return admitted reciprocal best matches."""
    counts = Counter()
    if matrix.shape[1] == 0:
        counts["no_opposite_full_hull"] += matrix.shape[0]
        return [], counts
    retained = []
    for i, row in enumerate(matrix):
        j = int(np.argmax(row))
        score = row[j]
        if score < .95:
            counts["below_iou_95"] += 1
            continue
        column = matrix[:, j]
        if i != np.argmax(column):
            counts["not_reciprocal"] += 1
            continue
        second = max(max(np.delete(row, j), default=0.), max(np.delete(column, i), default=0.))
        if score - second < .10:
            counts["ambiguous_gap_below_10"] += 1
            continue
        retained.append((i, j))
        counts["admitted"] += 1
    return retained, counts


def label_pair(row, images, index, ship):
    shared = tuple(row["overlap"])
    pair = row["pair"]
    per_view = []
    label_counts = Counter()
    for member in pair:
        img = images[member["native_basename"]]
        need(img["width"] == img["height"] == 800, "Native label canvas differs")
        polys, counts = eligible_polygons(index[img["id"]], ship, member["window"], shared)
        per_view.append(polys)
        label_counts.update(counts)

    scores = overlaps(*per_view)
    matches, left_failures = reciprocal(scores)
    reverse, right_failures = reciprocal(scores.T)
    need(set(matches) == {(j, i) for i, j in reverse}, "Reciprocal orientation differs")
    match_counts = left_failures + right_failures
    result = Counter()
    for i, j in matches:
        hull = per_view[0][i].union(per_view[1][j])
        a, b = sorted(distance(hull, member["window"]) for member in pair)
        need(a > 0, "Matched union crossed a crop margin")
        result["full_hull"] += 1
        result["seam_asymmetric" if a <= 16 and b >= 64 else
               "both_interior" if a >= 64 else "other"] += 1
    return result, label_counts, match_counts


def lattice(strata, pair, counts):
    dy, dx = map(abs, pair["shift_yx"])
    offset = f"dy={dy},dx={dx}"
    record = strata.setdefault(offset, {
        "crop_pairs": 0, "full_hull": 0, "seam_asymmetric": 0, "both_interior": 0,
        "mask_grid_shift_yx": [dy * 144 // 800, dx * 144 // 800],
        "network_input_shift_yx": [dy * 576 // 800, dx * 576 // 800],
        "patch_grid_shift_yx": [dy * 576 // (800 * 12), dx * 576 // (800 * 12)],
        "mask_grid_aligned": all(v * 144 % 800 == 0 for v in (dy, dx)),
        "input_grid_aligned": all(v * 576 % 800 == 0 for v in (dy, dx)),
        "patch_grid_aligned": all(v * 576 % (800 * 12) == 0 for v in (dy, dx)),
    })
    record["crop_pairs"] += 1
    for key in ("full_hull", "seam_asymmetric", "both_interior"):
        record[key] += counts[key]


def main(args):
    started = perf_counter()
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Pinned exact CPU geometry")
    protocol = read_exact(args.prelock, PROTOCOL_SHA)
    roster = read_exact(args.pair_lock, PAIR_SHA)
    rgb = read_exact(args.rgb_result, RGB_SHA)
    pairs = roster["pairs"]
    need(protocol["GT_opened"] is False and protocol["labels_sha256"] == LABEL_SHA and
         roster["GT_opened"] is False and rgb["GT_opened"] is False and rgb["lock_sha256"] == PAIR_SHA and
         len(pairs) == len({p["group"] for p in pairs}) == rgb["exact_RGB_pairs"] == 36,
         "Prior exact label-free pair lock or RGB acceptance differs")
    need([p["ordinal"] for p in pairs] == list(range(1, 37)), "Pair order differs")
    names = {r["native_basename"] for p in pairs for r in p["pair"]}
    need(len(names) == 72, "Expected 72 unique FIT crops")
    receipt_sha = write_new(args.receipt, dict(
        status="HRSID_FIT_OVERLAP_TARGET_SCOPE_ADMITTED_BEFORE_LABELS", GT_opened=False,
        FIT=True, DEV=False, TEST=False, forwards=0, pairs=36, crops=72,
        protocol_sha256=PROTOCOL_SHA, pair_sha256=PAIR_SHA, rgb_result_sha256=RGB_SHA,
    ))

    labels = read_exact(args.fit_index, LABEL_SHA)
    need(len(labels["images"]) == 2937 and len(labels["annotations"]) == 9260, "Exact native FIT census")
    images = {Path(img["file_name"]).name: img for img in labels["images"] if Path(img["file_name"]).name in names}
    need(set(images) == names and len({img["id"] for img in images.values()}) == 72, "Locked crop identity mismatch")
    ids = {img["id"] for img in images.values()}
    index = defaultdict(list)
    for ann in labels["annotations"]:
        if ann["image_id"] in ids:
            index[ann["image_id"]].append(ann)
    ship_ids = [cat["id"] for cat in labels["categories"] if cat["name"].lower().strip() == "ship"]
    need(len(ship_ids) == 1, "Expected exactly one ship category")
    ship = ship_ids[0]

    counts = Counter({k: 0 for k in ("full_hull", "seam_asymmetric", "both_interior", "other")})
    groups = Counter(counts)
    label_counts, match_counts, strata = Counter(), Counter(), {}
    for row in pairs:
        per_pair, label_states, match_states = label_pair(row, images, index, ship)
        counts.update(per_pair)
        groups.update(k for k, v in per_pair.items() if v > 0)
        label_counts.update(label_states)
        match_counts.update(match_states)
        lattice(strata, row, per_pair)
    need(match_counts["admitted"] == 2 * counts["full_hull"], "Endpoint and object match census differs")
    need(counts["full_hull"] == counts["seam_asymmetric"] + counts["both_interior"] + counts["other"],
         "Matched margin strata do not partition the cohort")
    passed = (counts["full_hull"] >= 40 and groups["full_hull"] >= 12 and
              counts["seam_asymmetric"] >= 10 and groups["seam_asymmetric"] >= 8)
    result = dict(
        status="HRSID_FIT_OVERLAP_TARGET_ADMISSION_ACTUAL",
        gate="TARGET_ADMISSION_PASS" if passed else "TARGET_ADMISSION_STOP",
        FIT=True, DEV=False, TEST=False, GT_opened=True, PNG_opened=False,
        model_forwards=0, optimizer_updates=0, pairs=36, crops=72,
        crop_annotations=sum(map(len, index.values())),
        label_endpoint_states=dict(label_counts), reciprocal_endpoint_states=dict(match_counts),
        unique_full_hull_counts=dict(counts), source_group_counts=dict(groups), offset_strata=strata,
        geometry_version="Shapely 2.0.2 / GEOS 3.11.2",
        protocol_sha256=PROTOCOL_SHA, pair_sha256=PAIR_SHA, rgb_result_sha256=RGB_SHA,
        label_sha256=LABEL_SHA, prelabel_receipt_sha256=receipt_sha,
        worker_seconds=perf_counter() - started,
    )
    digest = write_new(args.output, result)
    print(json.dumps(dict(result, result_sha256=digest), sort_keys=True))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("prelock", "pair-lock", "rgb-result", "fit-index", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args()
    need(sys.flags.isolated, "Use isolated CPU invocation")
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("Local closed labels only")
    sys.addaudithook(offline)
    main(options)
