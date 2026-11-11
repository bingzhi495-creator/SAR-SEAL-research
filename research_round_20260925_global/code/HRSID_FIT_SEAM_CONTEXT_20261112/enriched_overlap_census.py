"""Run a label-enriched, exact-RGB HRSID overlap feasibility census."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


PROTOCOL = "4cc14564b49e788d9c8d5da86f038402aab8ced77db6b88ab8a7535aada5a0e6"
NATIVE = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
LABEL = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
OLD_PAIR = "1ba947c401098e92737300d8715d523a890c414fc2e1795e7443530149b81024"
OLD_ADMISSION = "12fae2b7265884ef552eb3ef54a81cef737bca50c15554156a8065263536942d"
CROP_CODE = "0088d1a339ec89c4dd8fdaa668b5ad4e3b13127bece26dd8f1f087378a4df4c4"
LABEL_CODE = "72df4ca6488a7377d574000310f4c40dd46dabeca9d37850b9a1c96f15cd7730"
SALT = "HRSID-fit-overlap-enriched-v1|"


def need(valid, reason):
    if not valid:
        raise ValueError(reason)


def exact(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Unexpected input digest: " + path.name)
    return raw


def load_module(path, expected):
    module = ModuleType(path.stem)
    exec(compile(exact(path, expected), str(path), "exec"), module.__dict__)
    return module


def read_exact(path, expected):
    return json.loads(exact(path, expected))


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with path.open("xb") as file:
        file.write(raw)
    return sha256(raw).hexdigest()


def ranked(*parts):
    token = "|".join(map(str, parts))
    return sha256((SALT + token).encode()).digest(), token


def offset(shift):
    return tuple(map(abs, shift))


def choice_key(candidate):
    a, b = candidate["pair"]
    return ranked("row", a["native_basename"], b["native_basename"], *candidate["annotation_ids"])


def choose(seam, interior):
    if not seam:
        return None
    by_shift = defaultdict(list)
    for control in interior:
        by_shift[offset(control["shift_yx"])].append(control)
    selected = min(seam, key=lambda row: (not by_shift[offset(row["shift_yx"])], choice_key(row)))
    matches = by_shift[offset(selected["shift_yx"])]
    return selected, min(matches, key=choice_key) if matches else None


def metadata_pairs(layout, crop):
    windows = defaultdict(list)
    for row in layout:
        parsed = crop.parse_window(row["native_basename"])
        if parsed is not None:
            group, extent = parsed
            windows[group].append(dict({k: row[k] for k in crop.FIELDS}, window=extent))
    result = {}
    for group, records in windows.items():
        candidates = []
        ordered = sorted(records, key=lambda row: row["native_basename"])
        for a, b in combinations(ordered, 2):
            info = crop.pair_geometry(a["window"], b["window"])
            if info is None:
                continue
            shared, shift, mask_shift = info
            need(all(v * 576 % (800 * 12) == 0 for v in shift), "Unexpected RF patch-lattice phase")
            candidates.append(dict(pair=[a, b], overlap=shared, shift_yx=shift, mask_shift_yx=mask_shift))
        if candidates:
            result[group] = candidates
    return result


def original_geometry(pairs, labels, geometry):
    member_rows = {
        row["native_basename"]: row for group_pairs in pairs.values() for p in group_pairs for row in p["pair"]
    }
    images = {Path(img["file_name"]).name: img for img in labels["images"] if Path(img["file_name"]).name in member_rows}
    need(set(images) == set(member_rows), "Native label and crop census differ")
    by_image = defaultdict(list)
    for ann in labels["annotations"]:
        by_image[ann["image_id"]].append(ann)
    ships = [cat["id"] for cat in labels["categories"] if cat["name"].lower().strip() == "ship"]
    need(len(ships) == 1, "Unexpected native ship category")
    polys = defaultdict(list)
    for basename, img in images.items():
        need(img["width"] == img["height"] == 800, "Native FIT geometry differs")
        window = member_rows[basename]["window"]
        for ann in by_image[img["id"]]:
            admitted, _ = geometry.eligible_polygons([ann], ships[0], window, window)
            if admitted:
                polys[basename].append((ann["id"], admitted[0]))
    return polys


def witnesses(row, polys, geometry):
    pair, shared = row["pair"], row["overlap"]
    a, b = [
        [(key, poly) for key, poly in polys[r["native_basename"]] if geometry.distance(poly, shared) > 0]
        for r in pair
    ]
    scores = geometry.overlaps([p for _, p in a], [p for _, p in b])
    matches, _ = geometry.reciprocal(scores)
    reverse, _ = geometry.reciprocal(scores.T)
    need(set(matches) == {(j, i) for i, j in reverse}, "Reciprocal polygon matching differs")
    categories = Counter()
    seam, interior = [], []
    for i, j in matches:
        aid, ap = a[i]
        bid, bp = b[j]
        hull = ap.union(bp)
        margins = [geometry.distance(hull, member["window"]) for member in pair]
        near, far = min(margins), max(margins)
        need(near > 0, "Matched polygon union exceeds shared crop")
        categories["full_hull"] += 1
        entry = dict(row, annotation_ids=[aid, bid], margins=margins)
        if near <= 16 and far >= 64:
            seam.append(entry)
            categories["seam_asymmetric"] += 1
        elif near >= 64:
            interior.append(entry)
            categories["both_interior"] += 1
    return seam, interior, categories


def geometry_phase(args):
    started = perf_counter()
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact Shapely/GEOS versions")
    prelock = read_exact(args.prelock, PROTOCOL)
    old_pairs = read_exact(args.original_pair_lock, OLD_PAIR)
    old_actual = read_exact(args.original_admission, OLD_ADMISSION)
    need(prelock["GT_opened"] is False and old_actual["gate"] == "TARGET_ADMISSION_STOP", "Original stop must remain closed")
    excluded = {x["group"] for x in old_pairs["pairs"]}
    need(len(excluded) == 36, "Original 36-prefix exclusion differs")
    crop = load_module(args.crop_code, CROP_CODE)
    geometry = load_module(args.label_code, LABEL_CODE)
    layout = read_exact(args.native_layout, NATIVE)["images"]
    pair_pool = metadata_pairs(layout, crop)
    need(len(layout) == 2937 and len(pair_pool) == 108 and sum(map(len, pair_pool.values())) == 4112
         and excluded <= pair_pool.keys(), "Exact earlier geometric pair pool")
    receipt = write_new(args.receipt, dict(status="HRSID_FIT_OVERLAP_ENRICHED_PRELABEL_RECEIPT",
        FIT=True, DEV=False, TEST=False, GT_opened=False, PNG_opened=False, pair_count=4112,
        protocol_sha256=PROTOCOL, old_admission_sha256=OLD_ADMISSION))
    labels = read_exact(args.fit_index, LABEL)
    need(len(labels["images"]) == 2937 and len(labels["annotations"]) == 9260, "Exact FIT labels")
    polys = original_geometry(pair_pool, labels, geometry)

    all_counts, fresh_counts = Counter(), Counter()
    selected = []
    offset_support = defaultdict(Counter)
    for group in sorted(pair_pool, key=lambda g: ranked("source", g)):
        fresh = group not in excluded
        scope = fresh_counts if fresh else Counter()
        cohort_counts = Counter()
        near, inside = [], []
        for candidate in pair_pool[group]:
            a, b, kinds = witnesses(candidate, polys, geometry)
            near.extend(a)
            inside.extend(b)
            cohort_counts.update(kinds)
            if fresh:
                offset_support[crop.stratum(candidate["shift_yx"])].update(kinds)
        picked = choose(near, inside)
        for counters in (all_counts, scope):
            counters.update(cohort_counts)
            counters["groups_full_hull"] += bool(cohort_counts["full_hull"])
            counters["groups_seam_asymmetric"] += bool(near)
            counters["groups_both_interior"] += bool(inside)
            counters["groups_same_offset_control"] += bool(picked is not None and picked[1] is not None)
        if fresh and picked is not None:
            selected.append(dict(ordinal=len(selected) + 1, group=group, near_far=picked[0], matched_interior=picked[1]))
    need(len(selected) == len({x["group"] for x in selected}) == fresh_counts["groups_seam_asymmetric"],
         "Selection must preserve one near/far witness per positive fresh prefix")
    selected_sha = write_new(args.selected, dict(status="HRSID_FIT_OVERLAP_ENRICHED_RGB_ROSTER_LOCKED",
        FIT=True, DEV=False, TEST=False, GT_opened=True, PNG_opened=False,
        protocol_sha256=PROTOCOL, original_excluded_groups=36, records=selected))
    safe = dict(status="HRSID_FIT_OVERLAP_ENRICHED_GEOMETRY_ACTUAL",
        FIT=True, DEV=False, TEST=False, GT_opened=True, PNG_opened=False, model_forwards=0,
        geometric_pairs=4112, geometric_groups=108,
        fresh_geometric_pairs=sum(len(v) for g,v in pair_pool.items() if g not in excluded),
        fresh_geometric_groups=len(pair_pool) - len(excluded), original_excluded_groups=len(excluded),
        all_label_support=dict(all_counts), fresh_label_support=dict(fresh_counts),
        fresh_offset_support={k:dict(v) for k,v in offset_support.items() if any(v.values())},
        selected_near_far_groups=len(selected), selected_matched_controls=sum(r["matched_interior"] is not None for r in selected),
        protocol_sha256=PROTOCOL, roster_sha256=selected_sha, prelabel_receipt_sha256=receipt,
        worker_seconds=perf_counter() - started)
    digest = write_new(args.output, safe)
    print(json.dumps(dict(safe, result_sha256=digest), sort_keys=True))


def compare_pixels(crop, item, cache, fit_train):
    a, b = item["pair"]
    key = tuple(row["native_basename"] for row in (a, b))
    rgb = []
    for member in (a, b):
        name = member["native_basename"]
        if name not in cache:
            cache[name] = crop.load_image(fit_train, member)
        rgb.append(crop.overlap_pixels(cache[name], member["window"], item["overlap"]))
    need(rgb[0].shape == rgb[1].shape and rgb[0].size > 0, "Selected overlap dimensions differ")
    different = int(np.count_nonzero(np.any(rgb[0] != rgb[1], axis=2)))
    return key, different, rgb[0].shape[0] * rgb[0].shape[1]


def rgb_phase(args):
    started = perf_counter()
    selected = read_exact(args.selected, args.roster_sha256)
    prior = read_exact(args.geometry_result, args.geometry_sha256)
    need(selected["PNG_opened"] is prior["PNG_opened"] is False and
         selected["protocol_sha256"] == prior["protocol_sha256"] == PROTOCOL and
         prior["roster_sha256"] == args.roster_sha256, "Roster must close before PNG access")
    crop = load_module(args.crop_code, CROP_CODE)
    exact_near = exact_controls = 0
    decoded, outcomes = set(), {}
    offsets = defaultdict(Counter)
    for group in selected["records"]:
        cache = {}
        near, control = group["near_far"], group["matched_interior"]
        key, diff, pixels = compare_pixels(crop, near, cache, args.fit_train)
        outcomes[key] = (diff, pixels)
        exact_near += not diff
        stratum = crop.stratum(near["shift_yx"])
        offsets[stratum]["exact_near_far" if not diff else "failed_near_far"] += 1
        if control is not None:
            need(offset(near["shift_yx"]) == offset(control["shift_yx"]), "Matched control stratum changed")
            key_c, diff_c, pixels_c = compare_pixels(crop, control, cache, args.fit_train)
            outcomes[key_c] = (diff_c, pixels_c)
            control_pass = not diff and not diff_c
            exact_controls += control_pass
            offsets[stratum]["exact_matched_interior" if control_pass else "failed_matched_interior"] += 1
        decoded.update(cache)
    passed = exact_near >= 12 and exact_controls >= 8
    safe = dict(status="HRSID_FIT_OVERLAP_ENRICHED_RGB_ACTUAL",
        gate="FUTURE_FORWARD_DATA_FEASIBLE" if passed else "FUTURE_FORWARD_DATA_INSUFFICIENT",
        FIT=True, DEV=False, TEST=False, GT_opened=False, prior_geometry_used_FIT_GT=True,
        model_forwards=0, optimizer_updates=0, fresh_prefix_only=True,
        exact_near_far_source_groups=exact_near, exact_matched_control_source_groups=exact_controls,
        selected_near_far_groups=prior["selected_near_far_groups"],
        selected_matched_control_groups=prior["selected_matched_controls"], decoded_PNG_count=len(decoded),
        unique_crop_pairs_checked=len(outcomes), nonidentical_crop_pairs=sum(bool(x[0]) for x in outcomes.values()),
        shared_pixel_count=sum(x[1] for x in outcomes.values()), nonidentical_pixel_count=sum(x[0] for x in outcomes.values()),
        offset_strata={k:dict(v) for k,v in offsets.items()}, roster_sha256=args.roster_sha256,
        geometry_sha256=args.geometry_sha256, protocol_sha256=PROTOCOL,
        worker_seconds=perf_counter() - started)
    digest = write_new(args.output, safe)
    print(json.dumps(dict(safe, result_sha256=digest), sort_keys=True))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    g = commands.add_parser("geometry")
    for name in ("prelock", "original-pair-lock", "original-admission", "crop-code", "label-code",
                 "native-layout", "fit-index", "receipt", "selected", "output"):
        g.add_argument("--" + name, type=Path, required=True)
    g.set_defaults(run=geometry_phase)
    r = commands.add_parser("rgb")
    for name in ("selected", "geometry-result", "crop-code", "fit-train", "output"):
        r.add_argument("--" + name, type=Path, required=True)
    for name in ("roster-sha256", "geometry-sha256"):
        r.add_argument("--" + name, required=True)
    r.set_defaults(run=rgb_phase)
    args = parser.parse_args()
    args.run(args)
