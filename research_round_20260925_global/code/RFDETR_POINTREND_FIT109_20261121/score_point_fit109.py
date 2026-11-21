"""Score six frozen-carrier FIT109 policies under the same geometry and COCO rules."""

from collections import Counter, defaultdict
from math import isfinite

import numpy as np
from pycocotools import mask as codec


ARMS = ("N", "C", "P", "F", "B", "J")
MATCHED = ("F", "B", "J")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def natural_geometry(rle, reference, modules):
    """Exact overlap with the visible polygon on the native image grid."""
    counts = rle["counts"]
    counts = counts.decode("ascii") if isinstance(counts, bytes) else counts
    area = int(codec.area(rle))
    if area == 0:
        return 0.0
    rectangles = modules.geometry.boxes_from_counts(modules.shift.runs(counts), *rle["size"])
    intersection = modules.geometry.overlap_area(rectangles, reference)
    truth = float(reference.area)
    need(truth > 0 and 0 <= intersection <= min(area, truth) + 1e-7, "Physical overlap")
    return float(intersection / (area + truth - intersection))


def paired(values, geometry):
    """Equal weight for the original source prefixes. Single point-head seed."""
    need(bool(values), "At least one contributing source group")
    means = 100*np.stack([np.asarray(values[g]).mean(axis=0) for g in sorted(values)])
    rng = np.random.Generator(np.random.PCG64(geometry.BOOTSTRAP_SEED))
    draws = rng.integers(len(means), size=(geometry.DRAW_COUNT, len(means)))
    result = {}
    for a, b in (("B", "F"), ("B", "J"), ("B", "P"), ("P", "C"), ("B", "N"), ("P", "N")):
        delta = means[:, ARMS.index(a)] - means[:, ARMS.index(b)]
        result[a+"_minus_"+b] = dict(geometry_delta_points=float(delta.mean()),
                                     geometry_ci95_points=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist())
    return dict(zip(ARMS, map(float, means.mean(axis=0)), strict=True)), result


def evaluate(gt, pools, attached, outputs, budgets, helpers, pooled, protocol):
    geom = helpers.geometry
    rows = {geom.identity(r): r for r in attached}
    need(len(rows) == protocol["data"]["area_eligible_candidates"]
         and len(rows) == len(attached) and set(rows) == set(outputs) == set(budgets),
         "Exact complete area-eligible prediction and edited-candidate roster")
    need(len(gt["images"]) == protocol["data"]["images"] == len(pools)
         and len(gt["annotations"]) == protocol["data"]["annotations"], "FIT-only COCO population")
    reports = {a: Counter() for a in ARMS}
    strict = {a: Counter(dict.fromkeys(geom.TRANSITIONS, 0)) for a in ARMS}
    streams = {a: [] for a in ARMS}
    valid = defaultdict(list)
    counts = Counter()
    groups = defaultdict(list)
    for ann in gt["annotations"]:
        groups[ann["image_id"]].append(ann)
    modules = helpers.truth.modules

    for image in gt["images"]:
        iid = image["id"]
        image_gt = dict(images=[image], annotations=groups[iid], categories=gt["categories"])
        polygons, _, _ = modules.geometry.polygon_support(image_gt, modules.vector)
        for stock in pools[iid]:
            source, area = stock["rle"], stock["area"]
            key = (iid, stock["rank"], stock["query_id"])
            counts["mapped"] += 1
            views = {a: source for a in ARMS}
            row = rows.get(key)
            if row is not None:
                packet, k = outputs[key], budgets[key]["k"]
                counts["eligible"] += 1
                counts[row["role"]] += 1
                counts["zero_budget"] += k == 0
                views.update(C=packet["C"], P=packet["P"])
                measured = {"N": dict(flipped=[], adds=0, drops=0, edits=0, area_delta=0)}
                for a in ("C", "P"):
                    result = views[a]
                    if result is source:
                        continue
                    result_area = int(codec.area(result))
                    common = int(codec.area(codec.merge([source, result], intersect=True)))
                    report = reports[a]
                    report["edits"] += area + result_area - 2*common
                    report["adds"] += result_area - common
                    report["drops"] += area - common
                for a in MATCHED:
                    indices = packet[a]
                    diag = packet["diagnostics"][a]
                    add, drop = diag["adds"], diag["drops"]
                    need(len(indices) == add+drop == 2*k and diag["area_delta"] == add-drop
                         and (a == "J" or add == drop == k), "Exact per-candidate realized budget")
                    measured[a] = dict(flipped=indices, adds=add, drops=drop, area_delta=add-drop, edits=2*k)
                    geom.sparse_edit(row, measured[a])
                    views[a] = helpers.sparse.xor_rle(source, indices)
                    if k:
                        common = int(codec.area(codec.merge([source, views[a]], intersect=True)))
                        need(int(codec.area(views[a])) == area + add - drop and common == area - drop,
                             "RLE area and intersection confirm the exported edit counts")
                    else:
                        need(views[a] is source, "Zero budget keeps exact original mask")
                    reports[a].update(edits=2*k, adds=add, drops=drop,
                                      nonpositive_flips=diag["nonpositive_flips"])

                if row["role"] == "valid":
                    ref = polygons[row["matched_gt_id"]][0]
                    before = geom.candidate_geometry(row, measured["N"])["raw_iou"]
                    ious = dict(N=before)
                    for a in MATCHED:
                        ious[a] = geom.candidate_geometry(row, measured[a])["iou"]
                    for a in ("C", "P"):
                        ious[a] = natural_geometry(views[a], ref, modules)
                    need(all(isfinite(x) for x in ious.values()), "Finite fixed-identity geometry")
                    valid[row["group"]].append([ious[a] for a in ARMS])
                    for a in ARMS:
                        changed = measured[a]["edits"] if a in measured else views[a]["counts"] != source["counts"]
                        strict[a][geom.strict_transition(before, ious[a], changed, .90)] += 1

            for a in ARMS:
                streams[a].append(dict(image_id=iid, category_id=1, score=stock["score"], segmentation=views[a]))

    expected_edits = 2*protocol["matched_swaps_per_arm"]
    need(counts["eligible"] == len(rows) and counts["mapped"] == protocol["data"]["mapped_candidates"]
         and all(reports[a]["edits"] == expected_edits for a in MATCHED),
         "Locked complete pooled population and total edit counts")
    means, contrasts = paired(valid, geom)
    metrics = {a: dict(**pooled(gt, streams.pop(a), helpers.official), strict90=dict(strict[a]),
                       source_equal_iou_points=means[a], edits={k: reports[a][k] for k in
                       ("edits","adds","drops","nonpositive_flips")})
               for a in ARMS}
    for contrast in contrasts:
        a, b = contrast.split("_minus_")
        contrasts[contrast]["AP_delta_points"] = 100*(metrics[a]["AP"]-metrics[b]["AP"])
    checks = {}
    gate = protocol["gate"]
    for b in ("F", "J"):
        d = contrasts["B_minus_"+b]
        checks[b] = dict(AP=d["AP_delta_points"] >= gate["ap_min_points"],
                         geometry=d["geometry_delta_points"] >= gate["geometry_min_points"],
                         geometry_lower=d["geometry_ci95_points"][0] > 0,
                         AP75=metrics["B"]["AP75"] >= metrics[b]["AP75"],
                         strict=metrics["B"]["strict90"]["lost"] <= metrics[b]["strict90"]["lost"])
    return dict(arms=metrics, contrasts=contrasts, counts=dict(counts), contributing_groups=len(valid),
                bootstrap_draws=geom.DRAW_COUNT, bootstrap_seed=geom.BOOTSTRAP_SEED,
                advance=all(all(row.values()) for row in checks.values()), checks=checks)
