"""Unbound E24 fixed-pair consensus-region localization."""

from copy import deepcopy
import json
from math import fsum
from types import SimpleNamespace


INTEGER = "INTEGER_PRIMARY"
FLOAT = "LOCAL_CANDIDATE_FLOAT"
RADIUS = 2
THRESHOLD = .90
PAIRS = 377
QUERIES = 100
ACCEPTED_COUNTS = {
    INTEGER: {"0.90": (104, 103, 60), "0.75": (351, 351, 331)},
    FLOAT: {"0.90": (128, 123, 86), "0.75": (359, 358, 341)},
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def binary_mask(np, value):
    mask = np.asarray(value)
    need(mask.ndim == 2 and all(mask.shape), "Positive two-dimensional original-grid mask")
    if mask.dtype != np.bool_:
        need(np.issubdtype(mask.dtype, np.integer) and bool(np.all((mask == 0) | (mask == 1))),
             "Binary original-grid mask")
        mask = mask.astype(np.bool_, copy=False)
    return mask


def dilate_chebyshev(np, mask):
    """A clipped square neighborhood on the original image grid."""
    height, width = mask.shape
    vertical = np.pad(mask, ((RADIUS, RADIUS), (0, 0)), constant_values=False)
    intermediate = np.zeros_like(mask)
    for index in range(2 * RADIUS + 1):
        intermediate |= vertical[index:index + height, :]
    horizontal = np.pad(intermediate, ((0, 0), (RADIUS, RADIUS)), constant_values=False)
    expanded = np.zeros_like(mask)
    for index in range(2 * RADIUS + 1):
        expanded |= horizontal[:, index:index + width]
    return expanded


def localize_pair(np, integer_mask, floating_mask, prediction):
    """Disjoint consensus regions for one original-grid prediction and both GTs."""
    gi, gf, pred = (binary_mask(np, value) for value in (integer_mask, floating_mask, prediction))
    need(gi.shape == gf.shape == pred.shape and bool(gi.any()) and bool(gf.any()),
         "Same original grid and two nonempty GT masks")
    union = gi | gf
    disagreement = gi ^ gf
    foreground = gi & gf
    background = ~union
    # The complement of clipped erosion is clipped dilation of the complement.
    band = dilate_chebyshev(np, union) & dilate_chebyshev(np, ~foreground) & ~disagreement
    fg_band, fg_outside = foreground & band, foreground & ~band
    bg_band, bg_outside = background & band, background & ~band
    count = lambda value: int(np.count_nonzero(value))
    areas = dict(GT_union=count(union), GT_disagreement=count(disagreement),
                 FG_band=count(fg_band), FG_outside=count(fg_outside),
                 BG_band=count(bg_band), BG_outside=count(bg_outside))
    errors = dict(FN=dict(band=count(fg_band & ~pred), outside=count(fg_outside & ~pred)),
                  FP=dict(band=count(bg_band & pred), outside=count(bg_outside & pred)))
    return dict(areas=areas, errors=errors)


def summarize_localization(rows):
    """Keep normalized error burden separate from at-risk exposure and concentration."""
    need(rows, "Nonempty locked dual-miss cohort")
    unions = [row["areas"]["GT_union"] for row in rows]
    need(all(value > 0 for value in unions), "Each pair has a positive common denominator")
    area = {key: sum(row["areas"][key] for row in rows) for key in rows[0]["areas"]}
    total_union = sum(unions)

    def measure(values):
        pixels = sum(values)
        return dict(pixels=pixels, instance_mean_pp=100 * fsum(v/u for v, u in zip(values, unions))/len(rows),
                    pooled_pp=100 * pixels/total_union)

    def ratio(numerator, denominator):
        return numerator/denominator if denominator else None

    errors, concentrations, counts = {}, {}, {}
    for direction, region in (("FN", "FG"), ("FP", "BG")):
        band = [row["errors"][direction]["band"] for row in rows]
        outside = [row["errors"][direction]["outside"] for row in rows]
        counts[direction] = dict(band=band, outside=outside, total=[a+b for a, b in zip(band, outside)])
        errors[direction] = {name: measure(values) for name, values in counts[direction].items()}
        band_area, outside_area = area[region + "_band"], area[region + "_outside"]
        band_error, outside_error = sum(band), sum(outside)
        exposure = ratio(band_area, band_area+outside_area)
        concentration = ratio(band_error, band_error+outside_error)
        concentrations[direction] = dict(
            band_exposure_fraction=exposure, error_band_fraction=concentration,
            concentration_minus_exposure_pp=(100*(concentration-exposure)
                if concentration is not None and exposure is not None else None),
            band_error_rate=ratio(band_error, band_area), outside_error_rate=ratio(outside_error, outside_area))
    signed = {name: measure([fn-fp for fn, fp in zip(values, counts["FP"][name])])
              for name, values in counts["FN"].items()}
    return dict(areas_pixels=area, errors=errors, exposure_and_concentration=concentrations, FN_minus_FP=signed)


def integer_best_query(values, accepted_max):
    """The first maximum in natural query order, independently of mapped rank."""
    need(len(values) == QUERIES, "All 100 true natural queries")
    query = max(range(QUERIES), key=values.__getitem__)
    need(float(values[query]).hex() == float(accepted_max).hex(), "Exact accepted integer raw maximum")
    return query


def signature(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evaluate_localization(np, control, *, integer_gt, floating_gt, accepted_private_GT_records,
                          accepted_raw_private_pairs, primary_manifest, primary_predictions, admitted_items,
                          raw_records, modules):
    """Use original accepted pair identities and raw overlap matrices, with no new matching."""
    expected = {"output_bridge", "cpu_adapter", "gt_sensitivity", "coco_loader", "mask_api", "raw_geometry"}
    need(type(modules) is dict and set(modules) == expected, "Exact original localization dependency roles")
    need(type(accepted_raw_private_pairs) is list and len(accepted_raw_private_pairs) == PAIRS,
         "All original accepted raw-query fixed pairs")
    control.check()
    raw_geometry = modules["raw_geometry"]
    accepted_totals = raw_geometry.decompose(accepted_raw_private_pairs)
    need(all(tuple(accepted_totals[variant][threshold][name] for name in ("raw", "retained", "assigned")) == counts
             for variant, thresholds in ACCEPTED_COUNTS.items() for threshold, counts in thresholds.items()),
         "Exact previously accepted E24 raw, retained and assigned aggregate anchors")
    modules["gt_sensitivity"].validate_GT_pair(integer_gt, floating_gt)
    grouped = modules["output_bridge"].group_canonical_rows(primary_manifest, primary_predictions, admitted_items)
    _, blocks = modules["cpu_adapter"].normalize_export(grouped, integer_gt["images"])
    images = [block["image_id"] for block in blocks]
    need(len(images) == 187, "Original complete DEV187")
    cocos = {}
    for variant, document in ((INTEGER, integer_gt), (FLOAT, floating_gt)):
        coco = modules["coco_loader"].COCO()
        coco.dataset = deepcopy(document)
        coco.createIndex()
        need(list(coco.imgs) == images and len(coco.anns) == 380 and set(coco.cats) == {1},
             "Original ordered DEV187 and all 380 GT identities")
        cocos[variant] = coco
    pairs, _, _ = raw_geometry.select_pairs(accepted_private_GT_records, cocos[INTEGER], blocks)
    mask_api = modules["mask_api"]
    records, checked = [], 0
    for raw, manifest, block in zip(raw_records, primary_manifest["images"], blocks, strict=True):
        control.check()
        retained_queries = raw_geometry.validate_membership(raw, manifest, block)
        image = block["image_id"]
        matrices = []

        def capture_overlap(*args):
            matrix = mask_api.iou(*args)
            matrices.append(matrix)
            return matrix

        image_rows = raw_geometry.image_pair_overlaps(
            cocos, pairs[image], raw, block, retained_queries, SimpleNamespace(iou=capture_overlap))
        need(len(matrices) == (2 if image_rows else 0), "Exactly the original integer and floating raw overlap matrices")
        for column, row in enumerate(image_rows):
            control.check()
            need(checked < PAIRS and signature(row) == signature(accepted_raw_private_pairs[checked]),
                 "Exact ordered accepted raw-query pair identity and all overlaps")
            checked += 1
            by_gt = row["by_GT"]
            if not (by_gt[INTEGER]["raw"] < THRESHOLD and by_gt[FLOAT]["raw"] < THRESHOLD):
                continue
            integer_values = [float(matrices[0][query, column]) for query in range(QUERIES)]
            chosen = integer_best_query(integer_values, by_gt[INTEGER]["raw"])
            aid = row["GT_annotation_id"]
            chosen_rle = raw["raw_masks"][chosen]
            pred = binary_mask(np, mask_api.decode(dict(size=chosen_rle["size"],
                                                       counts=chosen_rle["counts"].encode("ascii"))))
            shape = tuple(raw["original_hw"])
            need(pred.shape == shape, "Selected true query on the original image grid")
            gt_masks = []
            for matrix, variant in zip(matrices, (INTEGER, FLOAT)):
                coco, annotation = cocos[variant], cocos[variant].anns[aid]
                need(annotation.get("iscrowd", 0) == 0, "Fixed noncrowd mask-overlap interpretation")
                gt = binary_mask(np, mask_api.decode(coco.annToRLE(annotation)))
                need(gt.shape == shape and bool(gt.any()), "Nonempty admitted original GT grid")
                intersection = int(np.count_nonzero(gt & pred))
                union = int(np.count_nonzero(gt | pred))
                need((intersection/union).hex() == float(matrix[chosen, column]).hex(),
                     "Decoded original-grid pixels reproduce the selected-query overlap")
                gt_masks.append(gt)
            localization = localize_pair(np, *gt_masks, pred)
            records.append(dict(image_id=image, GT_annotation_id=aid, selected_raw_query_id=chosen,
                                prior_overlaps=by_gt, selected_IoU_by_GT={
                                    variant: float(matrix[chosen, column])
                                    for variant, matrix in zip((INTEGER, FLOAT), matrices)}, **localization))
    need(checked == PAIRS, "Every original fixed pair checked before defining the dual-miss subset")
    summary = summarize_localization(records)
    summary.update(schema="Root-RFDETR-E24-dual-GT-raw-mask-localization-v1",
                   status="LOCALIZATION_COMPLETE_PENDING_ROOT_ACCEPTANCE", qualified=False, TEST=False, new_training=False,
                   scope=dict(images=len(images), fixed_pairs=PAIRS, dual_raw_miss_pairs=len(records),
                              images_with_dual_miss_pairs=len({row["image_id"] for row in records}),
                              other_fixed_pairs=PAIRS-len(records)),
                   radius_pixels=RADIUS, raw_IoU_threshold=THRESHOLD,
                   morphology="Clipped Chebyshev neighborhood on the finite original image grid",
                   band_definition="dilate(Gi union Gf) minus erode(Gi intersection Gf), excluding Gi xor Gf",
                   shared_foreground="Gi intersection Gf", shared_background="outside Gi union Gf",
                   selection="Integer GT maximum among 100 natural raw queries, ascending query index on ties",
                   both_variants_use_same_query=True, original_raw_pairs_exact=True,
                   AP_recomputed=False, COCO_matching_recomputed=False,
                   interpretation="Descriptive fixed conditional cohort. Concentration versus exposure is not a causal test.")
    control.check()
    return dict(summary=summary, private_pair_records=records)
