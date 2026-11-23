"""Pure fixed-match and sensitivity metrics for the locked Umbra prospective score."""

from collections import Counter, defaultdict

import numpy as np
import shapely


def rasterize(core, polygon, height, width):
    """COCO rasterization uses the same continuous pixel coordinates as the polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    core.check(polygon.geom_type == "Polygon" and polygon.is_valid and polygon.area > 0
               and 0 <= min_x < max_x <= width and 0 <= min_y < max_y <= height,
               "Expected a valid confirmed hull in native continuous image coordinates")
    core.check(not polygon.interiors, "Expected a single ring from the frozen review format")
    points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    core.check(np.all(np.isfinite(points)), "Confirmed hull has non-finite coordinates")
    return core.codec.merge(core.codec.frPyObjects([points.ravel().tolist()], height, width))


def geometry(core, mask, polygon):
    """Reuse the validated native unit-cell geometry. Recover intersection from IoU and areas."""
    area = int(core.codec.area(mask))
    truth = polygon.area
    iou = core.geometric_iou(mask, polygon, shapely)
    intersection = iou * (area + truth) / (1.0 + iou)
    core.check(0 <= intersection <= min(area, truth) + 1e-8, "Expected a bounded polygon intersection")
    intersection = min(intersection, area, truth)
    return iou, (area - intersection) / truth, (truth - intersection) / truth


def score_reference(core, refs, frames, hw):
    """Score one validated review on the jointly completed acquisitions.

    refs maps acquisition IDs to confirmed shapely Polygons in ascending shape ID.
    frames maps acquisition IDs to (native masks, edits), with edits already
    verified by core.verify_edits. Missing arms and ranks use the native mask.
    The core supplies the released native matching and paired bootstrap functions.
    """
    height, width = hw
    core.check(type(height) is int and type(width) is int and min(height, width) > 0,
               "Expected a positive native image shape")
    core.check(set(refs) <= set(frames), "Missing frozen candidate frame for a completed review")
    counts = Counter(jointly_complete_source_groups=len(refs), source_groups_without_reference_hulls=0,
                     labeled_hulls=0, reference_hulls_empty_raster=0, native_matched=0, native_unmatched=0,
                     native_area_below_32=0, native_area_at_least_1024=0, eligible_area_sf_zero=0,
                     sf_positive=0)
    all_groups, matched_groups, area_errors = defaultdict(list), defaultdict(list), defaultdict(list)
    exposed_groups = set()
    edited = Counter({arm: 0 for arm in core.ARMS})
    rematched = Counter({arm: 0 for arm in core.ARMS})
    transitions = {str(threshold): {arm: Counter(retained=0, lost=0, gained=0, both_below=0)
                                    for arm in core.ARMS} for threshold in (0.75, 0.90)}

    for group in sorted(refs):
        polygons = refs[group]
        if not polygons:
            counts["source_groups_without_reference_hulls"] += 1
            continue
        masks, edits = frames[group]
        masks = list(masks)
        truth = [rasterize(core, polygon, height, width) for polygon in polygons]
        matches = core.match_native(masks, truth)
        counts.update(labeled_hulls=len(polygons), native_matched=len(matches),
                      native_unmatched=len(polygons) - len(matches),
                      reference_hulls_empty_raster=sum(int(core.codec.area(mask)) == 0 for mask in truth))
        for arm in core.ARMS:
            updated = [edits.get(rank, {}).get(arm, (0, mask))[1]
                       for rank, mask in enumerate(masks, start=1)]
            rematched[arm] += len(core.match_native(updated, truth))

        for index, polygon in enumerate(polygons):
            if index not in matches:
                all_groups[group].append([0.0] * len(core.ARMS))
                continue
            rank = matches[index]
            original = masks[rank - 1]
            area = int(core.codec.area(original))
            variants = edits.get(rank, {})
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
            error_pairs = []
            for arm in core.ARMS:
                swaps, mask = variants.get(arm, (0, original))
                edited[arm] += swaps > 0
                key = mask["counts"]
                if key not in cache:
                    cache[key] = geometry(core, mask, polygon)
                iou, fp, fn = cache[key]
                values.append(iou)
                error_pairs.append((fp, fn))
            for label, per_arm in transitions.items():
                cutoff = float(label)
                was = values[core.ARMS.index("raw")] >= cutoff
                for arm, value in zip(core.ARMS, values, strict=True):
                    passes = value >= cutoff
                    event = ("retained" if passes else "lost") if was else ("gained" if passes else "both_below")
                    per_arm[arm][event] += 1
            area_errors[group].append(error_pairs)
            all_groups[group].append(values)
            matched_groups[group].append(values)

    core.check(rematched["raw"] == counts["native_matched"], "Raw rematch disagrees with frozen native matching")
    main = core.paired(all_groups) if all_groups else None
    effect = main["contrasts"][core.PRIMARY] if main else None
    coverage = counts["sf_positive"] >= core.MIN_EXPOSED_HULLS and len(exposed_groups) >= core.MIN_EXPOSED_GROUPS
    lower, upper = effect["paired_95_interval_points"] if effect else (None, None)
    errors = None
    if area_errors:
        means = np.mean([np.mean(area_errors[group], axis=0) for group in sorted(area_errors)], axis=0) * 100
        errors = {arm: {"mean_fp_pct_of_reference_area": float(fp), "mean_fn_pct_of_reference_area": float(fn)}
                  for arm, (fp, fn) in zip(core.ARMS, means, strict=True)}
    return {
        "primary_estimand": core.PRIMARY,
        "primary_all_labeled_hulls": main,
        "originally_matched_hulls_sensitivity": core.paired(matched_groups) if matched_groups else None,
        "labeled_hulls_and_coverage": dict(counts, source_groups_with_reference_hulls=len(all_groups),
                                           sf_positive_source_groups=len(exposed_groups),
                                           originally_matched_source_groups=len(matched_groups)),
        "edited_among_original_matches_by_arm": dict(edited),
        "continuous_iou_threshold_transitions_on_original_matches": transitions,
        "fixed_match_continuous_fp_fn_group_equal": errors,
        "descriptive_raster_iou_0_50_rematched_hulls_by_arm": dict(rematched),
        "coverage_gate_met": coverage,
        "frozen_geometry_advance": bool(coverage and lower is not None and lower > 0),
        "frozen_geometry_adverse": bool(coverage and upper is not None and upper < 0),
    }


def combine(reference_a, reference_b):
    """Apply the prelocked same-direction rule across both frozen references."""
    both_covered = reference_a["coverage_gate_met"] and reference_b["coverage_gate_met"]
    decision = "inconclusive"
    if both_covered:
        if reference_a["frozen_geometry_advance"] and reference_b["frozen_geometry_advance"]:
            decision = "positive"
        elif reference_a["frozen_geometry_adverse"] and reference_b["frozen_geometry_adverse"]:
            decision = "adverse"
    return {"decision": decision, "both_coverage_gates_met": both_covered}
