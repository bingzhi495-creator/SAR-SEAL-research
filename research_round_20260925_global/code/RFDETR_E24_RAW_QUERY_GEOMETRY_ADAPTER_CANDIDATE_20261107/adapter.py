"""Unbound fixed-pair E24 raw-query/mapped/assigned overlap decomposition."""

from copy import deepcopy
from math import isfinite


INTEGER = "INTEGER_PRIMARY"
FLOAT = "LOCAL_CANDIDATE_FLOAT"
GT_VERSIONS = (INTEGER, FLOAT)
THRESHOLDS = (("0.90", .90), ("0.75", .75))
PAIRS = 377
QUERIES = 100
RAW_KEYS = {"image_id", "image_sha256", "original_hw", "primary_canonical_json_sha256",
            "raw_masks", "slot0_scores", "slot0_native_ranks", "retained"}
ACCEPTED_GT_KEYS = {"image_id", "GT_annotation_id", "retained_candidate_count", "integer_effectively_ignored",
                    "assigned_detection_ordinal", "assigned_native_rank", "by_GT"}
ACCEPTED_OVERLAP_KEYS = {"mapped_oracle_max_IoU", "mapped_oracle_best_native_rank", "fixed_assigned_IoU"}
INTEGER_ANCHORS = {"0.90": (103, 60), "0.75": (351, 331)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def unit(value):
    return type(value) in (int, float) and isfinite(value) and 0 <= value <= 1


def decompose(rows):
    """Count nested coverage losses for the same fixed pairs, without rematching."""
    need(rows, "Nonempty fixed pair cohort")
    result = {}
    for gt in GT_VERSIONS:
        variants = [row["by_GT"][gt] for row in rows]
        need(all(all(unit(item[key]) for key in ("raw", "retained", "assigned"))
                 and item["raw"] >= item["retained"] >= item["assigned"] for item in variants),
             "Nested raw, retained and assigned overlaps")
        result[gt] = {}
        for name, threshold in THRESHOLDS:
            raw, retained, assigned = (sum(item[key] >= threshold for item in variants)
                                       for key in ("raw", "retained", "assigned"))
            result[gt][name] = dict(pairs=len(rows), raw=raw, retained=retained, assigned=assigned,
                                    geometry=len(rows)-raw, selection=raw-retained, assignment=retained-assigned)
    return result


def validate_membership(raw, manifest, block):
    """Link all retained mapped rows to the actual indexed raw query masks."""
    need(type(raw) is dict and set(raw) == RAW_KEYS and type(raw["image_id"]) is int
         and raw["image_id"] == manifest["image_id"] == block["image_id"]
         and raw["image_sha256"] == manifest["image_sha256"]
         and raw["primary_canonical_json_sha256"] == manifest["canonical_json_sha256"],
         "Same admitted image and frozen primary export identity")
    hw = raw["original_hw"]
    need(type(hw) is list and len(hw) == 2 and all(type(value) is int and value > 0 for value in hw)
         and hw == manifest["original_hw"], "Same admitted original mask grid")
    masks, scores, ranks = (raw[key] for key in ("raw_masks", "slot0_scores", "slot0_native_ranks"))
    need(all(type(values) is list and len(values) == QUERIES for values in (masks, scores, ranks))
         and all(unit(score) for score in scores)
         and all(type(rank) is int and 1 <= rank <= 2*QUERIES for rank in ranks)
         and len(set(ranks)) == QUERIES, "All 100 original query masks with unique stock slot0 ranks")
    for rle in masks:
        need(type(rle) is dict and set(rle) == {"size", "counts"}
             and type(rle["size"]) is list and all(type(value) is int for value in rle["size"])
             and rle["size"] == hw and type(rle["counts"]) is str
             and rle["counts"] and rle["counts"].isascii(), "Original-grid compressed raw query mask")
    order = sorted(range(QUERIES), key=ranks.__getitem__)
    need(all(scores[left] >= scores[right] for left, right in zip(order, order[1:])),
         "Slot0 scores retain stock all-slot order")
    queries = [query for query in order if ranks[query] <= QUERIES]
    witnesses = raw["retained"]
    expected = [dict(native_rank=ranks[query], query_id=query) for query in queries]
    need(type(witnesses) is list and all(type(item) is dict and set(item) == {"native_rank", "query_id"}
         and all(type(value) is int for value in item.values()) for item in witnesses)
         and witnesses == expected and [ranks[query] for query in queries] == block["native_ranks"]
         and len(queries) == len(block["coco"]), "Exact slot0 membership from original stock top100")
    for query, original in zip(queries, block["coco"]):
        need(masks[query] == original["segmentation"]
             and float(scores[query]).hex() == float(original["score"]).hex(),
             "Same original exact retained score and RLE for each true query")
    return queries


def image_pair_overlaps(cocos, pairs, raw, block, retained_queries, mask_api):
    """One raw-query matrix per GT variant, with the retained and assigned subsets fixed."""
    if not pairs:
        return []
    image, ranks, start = (block[key] for key in ("image_id", "native_ranks", "start"))
    aids = [pair["GT_annotation_id"] for pair in pairs]
    crowds = [cocos[INTEGER].anns[aid].get("iscrowd", 0) for aid in aids]
    need(all(type(value) in (int, bool) and value in (0, 1) for value in crowds), "Original COCO crowd flags")
    matrices = {}
    for variant in GT_VERSIONS:
        coco = cocos[variant]
        annotations = [coco.anns[aid] for aid in aids]
        need(all(ann["image_id"] == image and ann["category_id"] == 1
                 and ann.get("iscrowd", 0) == flag for ann, flag in zip(annotations, crowds)),
             "Same fixed original annotation identity under both GTs")
        gts = [coco.annToRLE(ann) for ann in annotations]
        need(all(list(rle["size"]) == raw["original_hw"] for rle in gts), "Same original GT raster grid")
        matrix = mask_api.iou(raw["raw_masks"], gts, [int(value) for value in crowds])
        need(tuple(matrix.shape) == (QUERIES, len(pairs)), "One official all-query by fixed GT overlap matrix")
        matrices[variant] = matrix
    records = []
    for column, pair in enumerate(pairs):
        offset = pair["assigned_detection_ordinal"] - 1 - start
        need(0 <= offset < len(retained_queries) and ranks[offset] == pair["assigned_native_rank"],
             "Same exact accepted assigned mapped ordinal/rank")
        assigned_query = retained_queries[offset]
        by_gt = {}
        for variant, matrix in matrices.items():
            values = [float(matrix[query, column]) for query in range(QUERIES)]
            need(all(unit(value) for value in values), "Finite official raw-mask IoU")
            retained_values = [values[query] for query in retained_queries]
            best_retained = max(retained_values)
            best_rank = ranks[retained_values.index(best_retained)]
            assigned = values[assigned_query]
            expected = pair["by_GT"][variant]
            need(type(expected) is dict and set(expected) == ACCEPTED_OVERLAP_KEYS
                 and unit(expected["mapped_oracle_max_IoU"]) and unit(expected["fixed_assigned_IoU"])
                 and type(expected["mapped_oracle_best_native_rank"]) is int
                 and best_rank == expected["mapped_oracle_best_native_rank"]
                 and best_retained.hex() == float(expected["mapped_oracle_max_IoU"]).hex()
                 and assigned.hex() == float(expected["fixed_assigned_IoU"]).hex(),
                 "Exact accepted per-GT overlap and retained native rank reproduction")
            by_gt[variant] = dict(raw=max(values), retained=best_retained, assigned=assigned)
        need(by_gt[INTEGER]["assigned"] >= .50, "Original integer .50 assigned pair")
        records.append(dict(image_id=image, GT_annotation_id=pair["GT_annotation_id"],
                            assigned_native_rank=pair["assigned_native_rank"], assigned_query_id=assigned_query,
                            by_GT=by_gt))
    return records


def select_pairs(accepted, integer, blocks):
    """Use the accepted private assignments, without another matching operation."""
    need(type(accepted) is list and len(accepted) == 380, "All original E24 GT diagnostics")
    groups = {block["image_id"]: [] for block in blocks}
    by_image = {block["image_id"]: block for block in blocks}
    seen_gt, assigned_ordinals = set(), set()
    ignored = unmatched = 0
    for row in accepted:
        need(type(row) is dict and set(row) == ACCEPTED_GT_KEYS and type(row["GT_annotation_id"]) is int,
             "Exact accepted per-GT record schema")
        aid, image = row["GT_annotation_id"], row["image_id"]
        need(aid in integer.anns and aid not in seen_gt and type(image) is int
             and integer.anns[aid]["image_id"] == image and image in by_image
             and type(row["retained_candidate_count"]) is int
             and row["retained_candidate_count"] == len(by_image[image]["coco"])
             and type(row["integer_effectively_ignored"]) is bool
             and type(row["by_GT"]) is dict and set(row["by_GT"]) == set(GT_VERSIONS),
             "Exact accepted original annotation/image/retained population")
        seen_gt.add(aid)
        ordinal = row["assigned_detection_ordinal"]
        if ordinal is None:
            need(row["assigned_native_rank"] is None, "No fabricated rank on unassigned GT")
            ignored += int(row["integer_effectively_ignored"])
            unmatched += int(not row["integer_effectively_ignored"])
            continue
        need(type(ordinal) is int and ordinal > 0 and ordinal not in assigned_ordinals
             and not row["integer_effectively_ignored"] and type(row["assigned_native_rank"]) is int,
             "Unique original nonignored assigned detection row")
        assigned_ordinals.add(ordinal)
        groups[image].append(row)
    need(seen_gt == set(integer.anns) and len(assigned_ordinals) == PAIRS and ignored+unmatched == 380-PAIRS,
         "Same exact 377 accepted paired GT and complete original coverage")
    return groups, ignored, unmatched


def evaluate_raw_geometry(control, *, integer_gt, floating_gt, accepted_private_GT_records, primary_manifest,
                          primary_predictions, admitted_items, raw_records, modules):
    """Consume admitted parsed inputs and stream raw records in the original DEV order."""
    expected = {"output_bridge", "cpu_adapter", "gt_sensitivity", "coco_loader", "mask_api"}
    need(type(modules) is dict and set(modules) == expected, "Exact original evaluation and raw diagnostic helpers")
    control.check()
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
    pairs, ignored, unmatched = select_pairs(accepted_private_GT_records, cocos[INTEGER], blocks)
    records = []
    for raw, manifest, block in zip(raw_records, primary_manifest["images"], blocks, strict=True):
        control.check()
        queries = validate_membership(raw, manifest, block)
        records.extend(image_pair_overlaps(cocos, pairs[block["image_id"]], raw, block, queries, modules["mask_api"]))
    need(len(records) == PAIRS, "Every fixed pair contributes once to both GT variants")
    totals = decompose(records)
    for threshold, (retained, assigned) in INTEGER_ANCHORS.items():
        actual = totals[INTEGER][threshold]
        need((actual["retained"], actual["assigned"]) == (retained, assigned), "Exact published E24 threshold anchors")
    summary = dict(schema="Root-RFDETR-E24-raw-query-fixed-pair-overlap-v1",
                   status="GEOMETRY_COMPLETE_PENDING_ROOT_ACCEPTANCE", qualified=False, TEST=False, new_training=False,
                   AP_recomputed=False, COCO_matching_recomputed=False, primary_GT=INTEGER,
                   primary_IoU_threshold=.90, secondary_IoU_threshold=.75,
                   images=187, paired_GT=PAIRS, total_GT=380, integer_ignored_GT=ignored,
                   integer_unmatched_nonignored_GT=unmatched, raw_query_mask_records_checked=18700,
                   images_with_pairs=sum(bool(group) for group in pairs.values()), counts_by_GT=totals,
                   original_retained_membership_exact=True, original_assigned_and_retained_overlap_exact=True,
                   interpretation="Nested one-GT maxima on fixed original assignments. No one-to-one oracle or causal attribution.")
    control.check()
    return dict(summary=summary, private_pair_records=records)
