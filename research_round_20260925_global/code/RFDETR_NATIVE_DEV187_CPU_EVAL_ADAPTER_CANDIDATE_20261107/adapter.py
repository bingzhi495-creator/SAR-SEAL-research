"""Unbound RF-DETR CPU evaluation on the original ordered DEV population."""

from copy import deepcopy
from math import fsum, isfinite
from statistics import median


EXPORT_SCHEMA = "Root-RFDETR-DEV187-ordered-coco-export-v1"
IMAGE_FIELDS = {"image_id", "raw_query_count", "native_count", "unmapped_slot1_count", "native_ranks", "coco"}
COCO_FIELDS = {"image_id", "category_id", "score", "segmentation"}
INTEGER = "INTEGER_PRIMARY"
FLOAT = "LOCAL_CANDIDATE_FLOAT"
THRESHOLDS = (("0.50", .50), ("0.75", .75), ("0.90", .90), ("0.95", .95))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def normalize_export(document, original_images):
    """Preserve the exact ordered COCO rows and recover their native rank sidecar."""
    need(type(document) is dict and set(document) == {"schema", "images"}
         and document["schema"] == EXPORT_SCHEMA and type(document["images"]) is list
         and len(document["images"]) == len(original_images), "Exact original export population")
    flat, blocks = [], []
    for block, image in zip(document["images"], original_images):
        iid, height, width = (image[key] for key in ("id", "height", "width"))
        need(type(iid) is int and type(height) is type(width) is int and height > 0 and width > 0,
             "Canonical original image ID and dimensions")
        need(type(block) is dict and set(block) == IMAGE_FIELDS
             and type(block["image_id"]) is int and block["image_id"] == iid
             and type(block["raw_query_count"]) is type(block["native_count"]) is int
             and block["raw_query_count"] == block["native_count"] == 100, "Exact ordered native 100-query/top100 evidence")
        rows, ranks, unmapped = (block[key] for key in ("coco", "native_ranks", "unmapped_slot1_count"))
        need(type(rows) is type(ranks) is list and type(unmapped) is int and 0 <= unmapped <= 100
             and len(rows) == len(ranks) == 100 - unmapped
             and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
             and all(left < right for left, right in zip(ranks, ranks[1:])), "Exact retained stock all-slot ranks/counts")
        previous = 1.0
        for row in rows:
            need(type(row) is dict and set(row) == COCO_FIELDS
                 and type(row["image_id"]) is int and row["image_id"] == iid
                 and type(row["category_id"]) is int and row["category_id"] == 1
                 and type(row["score"]) in (int, float) and isfinite(row["score"]) and 0 <= row["score"] <= previous,
                 "Exact category1 row, original ID and unchanged native descending score order")
            rle = row["segmentation"]
            need(type(rle) is dict and set(rle) == {"size", "counts"}
                 and type(rle["size"]) is list and all(type(value) is int for value in rle["size"])
                 and rle["size"] == [height, width]
                 and type(rle["counts"]) is str and rle["counts"] and rle["counts"].isascii(),
                 "Original-grid compressed RLE JSON")
            previous = row["score"]
        blocks.append(dict(image_id=iid, start=len(flat), coco=rows, native_ranks=ranks))
        flat.extend(rows)
    need(len({block["image_id"] for block in blocks}) == len(blocks), "Unique original image IDs")
    need(flat, "Globally empty mapped export is outside this admitted legacy COCO loadRes path")
    return flat, blocks


def rle_identity(rle):
    counts = rle["counts"]
    return tuple(rle["size"]), counts.encode("ascii") if type(counts) is str else counts


def assigned_rows(np, evaluation, flat, image_ids, annotation_ids):
    """Extract actual integer-GT COCO assignments without manufacturing query IDs."""
    areas = []
    for params in (evaluation.params, evaluation._paramsEval):
        labels = list(params.areaRngLbl)
        need(params.iouType == "segm" and set(params.imgIds) == set(image_ids)
             and len(params.imgIds) == len(image_ids) and list(params.catIds) == [1]
             and list(params.maxDets) == [1, 10, 100]
             and np.array_equal(params.iouThrs, np.linspace(.5, .95, 10))
             and labels.count("all") == 1, "Actually evaluated original integer COCO parameters")
        areas.append(list(params.areaRng[labels.index("all")]))
    area = areas[0]
    need(area == areas[1], "Same original all-area COCO setting")
    detections = evaluation.cocoDt.anns
    need(set(detections) == set(range(1, len(flat)+1)), "Exact mapped loadRes ordinal population")
    for ordinal, row in enumerate(flat, 1):
        observed = detections[ordinal]
        need(observed["id"] == ordinal and observed["image_id"] == row["image_id"]
             and observed["category_id"] == row["category_id"] == 1
             and float(observed["score"]).hex() == float(row["score"]).hex()
             and rle_identity(observed["segmentation"]) == rle_identity(row["segmentation"]),
             "Exact original mapped detection identity at its COCO ordinal")
    assignments, seen_detections = {}, set()
    for item in evaluation.evalImgs:
        if item is None or list(item["aRng"]) != area or item["maxDet"] != 100 or item["category_id"] != 1:
            continue
        aids, ignores, matches = item["gtIds"], item["gtIgnore"], item["gtMatches"][0]
        image, valid_detections = item["image_id"], set(item["dtIds"])
        need(len(aids) == len(ignores) == len(matches), "Aligned original GT assignment metadata")
        for aid, ignore, value in zip(aids, ignores, matches):
            dt_id = int(value)
            need(aid not in assignments and ignore in (0, 1) and value == dt_id and dt_id >= 0,
                 "Unique original integer GT and effective ignore/assignment flags")
            assigned = dt_id if not ignore and dt_id else None
            if assigned is not None:
                need(assigned in valid_detections and assigned in detections and assigned not in seen_detections
                     and detections[assigned]["image_id"] == image, "One actual nonignored original COCO assignment")
                seen_detections.add(assigned)
            assignments[int(aid)] = dict(ignored=bool(ignore), dt_id=assigned)
    need(set(assignments) == set(annotation_ids), "Every original GT considered, including unmatched and ignored")
    return assignments


def collect_geometry(control, cocos, blocks, assignments, mask_api):
    """Measure each retained row against both GTs, fixing integer COCO assignment."""
    integer = cocos[INTEGER]
    records = []
    for block in blocks:
        control.check()
        image = block["image_id"]
        ids = integer.getAnnIds(imgIds=[image], catIds=[1])
        if not ids:
            continue
        rows, ranks, start = (block[key] for key in ("coco", "native_ranks", "start"))
        pred_rles = [row["segmentation"] for row in rows]
        hw = (integer.imgs[image]["height"], integer.imgs[image]["width"])
        flags = [integer.anns[aid].get("iscrowd", 0) for aid in ids]
        need(all(type(flag) in (int, bool) and flag in (0, 1) for flag in flags), "Original GT crowd flags")
        crowds = [int(flag) for flag in flags]
        by_gt = {}
        for version, coco in cocos.items():
            annotations = [coco.anns[aid] for aid in ids]
            need(all(ann["image_id"] == image and ann["category_id"] == 1
                     and ann.get("iscrowd", 0) == flag for ann, flag in zip(annotations, flags)),
                 "Same original annotation and crowd flags under both GTs")
            gt_rles = [coco.annToRLE(ann) for ann in annotations]
            need(all(tuple(rle["size"]) == hw for rle in gt_rles), "Same original RLE grids")
            if rows:
                matrix = mask_api.iou(pred_rles, gt_rles, crowds)
                need(tuple(matrix.shape) == (len(rows), len(ids)), "Official mapped rows by original GT overlap matrix")
                by_gt[version] = matrix
        for column, aid in enumerate(ids):
            assignment = assignments[aid]
            dt_id = assignment["dt_id"]
            offset = None if dt_id is None else dt_id - 1 - start
            need(offset is None or 0 <= offset < len(rows), "Assigned detection is the same mapped row on its original image")
            variants = {}
            for version in cocos:
                values = [float(by_gt[version][row, column]) for row in range(len(rows))]
                need(all(isfinite(value) and 0 <= value <= 1 for value in values), "Finite official mask overlap")
                best = max(values, default=0.0)
                first = values.index(best) if values else None
                variants[version] = dict(mapped_oracle_max_IoU=best,
                    mapped_oracle_best_native_rank=None if first is None else ranks[first],
                    fixed_assigned_IoU=None if offset is None else values[offset])
            need(offset is None or variants[INTEGER]["fixed_assigned_IoU"] >= .50,
                 "Same integer assigned pair reproduces IoU .50")
            records.append(dict(image_id=image, GT_annotation_id=aid, retained_candidate_count=len(rows),
                integer_effectively_ignored=assignment["ignored"], assigned_detection_ordinal=dt_id,
                assigned_native_rank=None if offset is None else ranks[offset], by_GT=variants))
    need(len(records) == len(assignments) and {row["GT_annotation_id"] for row in records} == set(assignments),
         "All original GT, including no-candidate images")
    return records


def overlap_summary(values):
    counts = {key: sum(value >= limit for value in values) for key, limit in THRESHOLDS}
    n = len(values)
    return dict(GT_denominator=n, mean_IoU=fsum(values)/n if n else None, median_IoU=median(values) if n else None,
                threshold_counts=counts, threshold_fractions={key: value/n if n else None for key, value in counts.items()})


def summarize_geometry(records):
    assigned = [row for row in records if row["assigned_detection_ordinal"] is not None]
    groups = dict(all_GT=records, integer_assigned_nonignored=assigned,
                  integer_unmatched_nonignored=[row for row in records if row["assigned_detection_ordinal"] is None
                                                and not row["integer_effectively_ignored"]],
                  integer_effectively_ignored=[row for row in records if row["integer_effectively_ignored"]])
    coverage, fixed = {}, {}
    for version in (INTEGER, FLOAT):
        coverage[version] = {name: overlap_summary([row["by_GT"][version]["mapped_oracle_max_IoU"] for row in group])
                             for name, group in groups.items()}
        fixed[version] = overlap_summary([row["by_GT"][version]["fixed_assigned_IoU"] for row in assigned])
    return dict(mapped_export_only_oracle=coverage, integer_assigned_same_rows=fixed,
                GT_without_mapped_candidates=sum(row["retained_candidate_count"] == 0 for row in records),
                mask_overlap_scope="Original COCO mask.iou with original per-GT iscrowd flags",
                oracle_scope="Max overlap within the retained mapped export, independently for each GT",
                oracle_is_AP=False, oracle_is_one_to_one_recall=False, float_assignment_recomputed=False)


def evaluate_dev187(np, control, *, integer_gt, floating_gt, export_document, modules, progress=None):
    """Use admitted inputs only. Optionally emit safe progress after each real COCO case."""
    expected = {"coco_loader", "cocoeval", "mask_api", "gt_sensitivity", "eval_body", "official"}
    need(type(modules) is dict and set(modules) == expected, "Exact original CPU evaluation dependency roles")
    need(progress is None or callable(progress), "Optional caller-owned progress callback")
    control.check()
    modules["gt_sensitivity"].validate_GT_pair(integer_gt, floating_gt)
    images = integer_gt["images"]
    flat, blocks = normalize_export(export_document, images)
    ids = [image["id"] for image in images]
    need(len(ids) == 187 and len(integer_gt["annotations"]) == 380, "Fixed original DEV187/380")
    cocos, metrics_by_gt, assignments = {}, {}, None
    for version, gt in ((INTEGER, integer_gt), (FLOAT, floating_gt)):
        control.check()
        coco = modules["coco_loader"].COCO()
        coco.dataset = deepcopy(gt)
        coco.createIndex()
        need(list(coco.imgs) == ids and len(coco.anns) == 380 and set(coco.cats) == {1},
             "Original ordered DEV187, complete annotations, category1")
        cocos[version] = coco
        metrics, per_iou, evaluation = modules["eval_body"].official_metrics(
            np, control, modules["official"], modules["cocoeval"].COCOeval, coco, flat, ids)
        metrics_by_gt[version] = dict(COCO=metrics, per_IoU=per_iou)
        if progress is not None:
            progress(dict(stage=version + "_COCO_COMPLETED", metrics_by_GT=deepcopy(metrics_by_gt)))
        if version == INTEGER:
            assignments = assigned_rows(np, evaluation, flat, ids, list(coco.anns))
        del evaluation
    records = collect_geometry(control, cocos, blocks, assignments, modules["mask_api"])
    summary = dict(schema="Root-RFDETR-DEV187-canonical-CPU-evaluation-v1",
                   status="COMPLETE_PENDING_ROOT_ACCEPTANCE", qualified=False, TEST=False, new_training=False,
                   formal_training_budget_credit=0, images=187, annotations=380, primary_GT=INTEGER,
                   metrics_by_GT=metrics_by_gt, raw_queries=18700, native_all_slot_topk_rows=18700,
                   mapped_rows=len(flat), unmapped_slot1_rows=18700-len(flat),
                   images_without_mapped_rows=sum(not block["coco"] for block in blocks),
                   float_GT_scope="Local candidate restoration sensitivity",
                   diagnostics=summarize_geometry(records))
    control.check()
    return dict(summary=summary, private_GT_records=records)
