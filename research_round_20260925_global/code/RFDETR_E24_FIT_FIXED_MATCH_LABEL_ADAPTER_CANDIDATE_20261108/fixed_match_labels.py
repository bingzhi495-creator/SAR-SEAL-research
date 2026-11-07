"""Attach fixed official COCO .50 flip labels to a frozen FIT image capture."""
from copy import deepcopy


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _coco_view(source, image, annotations):
    view = type(source)()
    view.dataset = {
        "images": [deepcopy(image)],
        "categories": [deepcopy(source.cats[1])],
        "annotations": deepcopy(annotations),
        "info": deepcopy(source.dataset.get("info", {})),
    }
    view.createIndex()
    return view


def _official_matches(np, coco_eval_type, gt, dt, image_id, gt_ids, dt_count):
    evaluation = coco_eval_type(gt, dt, "segm")
    p = evaluation.params
    area = list(p.areaRng[p.areaRngLbl.index("all")])
    p.imgIds, p.catIds, p.useCats = [image_id], [1], True
    p.areaRng, p.areaRngLbl, p.maxDets = [area], ["all"], [100]
    p.iouThrs = np.array([0.50], dtype=np.float64)
    evaluation.evaluate()
    require(len(evaluation.evalImgs) == 1, "Expected one official image evaluation")
    cell = evaluation.evalImgs[0]
    if not gt_ids and not dt_count:
        require(cell is None, "Expected official empty image evaluation")
        return [], [], 0
    require(cell is not None and cell["image_id"] == image_id
            and cell["category_id"] == 1 and list(cell["aRng"]) == area and cell["maxDet"] == 100
            and list(cell["dtIds"]) == list(range(1, dt_count + 1))
            and len(cell["gtIds"]) == len(gt_ids) and set(cell["gtIds"]) == gt_ids
            and cell["dtMatches"].shape == cell["dtIgnore"].shape == (1, dt_count)
            and cell["gtIgnore"].shape == (len(gt_ids),),
            "Official result must preserve the entire frozen candidate and GT populations")
    matches, ignored = cell["dtMatches"][0], cell["dtIgnore"][0]
    gt_ignored = set(g for g, flag in zip(cell["gtIds"], cell["gtIgnore"]) if flag)
    require(all(value == int(value) and (value == 0 or value in gt_ids)
                and (value == 0 or bool(ignore) == (value in gt_ignored))
                for value, ignore in zip(matches, ignored)), "Invalid official match identities")
    return matches, ignored, len(gt_ignored)


def build_fixed_labels(np, codec, coco_eval_type, *, integer_coco, rows_document, arrays, verify_capture):
    """Return all native rows, labeling only positive matches on unchanged support.

    The caller binds the accepted FIT-only integer COCO view and source artifacts.
    No candidate selection, reranking, aggregate AP, or cross-image labels occur.
    """
    verify_capture(np, codec, rows_document, arrays)
    image_id, records = rows_document["image_id"], rows_document["records"]
    require(image_id in integer_coco.imgs and list(integer_coco.getCatIds()) == [1],
            "Expected image in the caller's fixed single-category integer FIT view")
    image = integer_coco.imgs[image_id]
    require([image["height"], image["width"]] == rows_document["original_hw"],
            "Ground truth must use the exact original capture grid")
    annotations = integer_coco.imgToAnns.get(image_id, [])
    gt_ids = set()
    for ann in annotations:
        aid = ann["id"]
        require(ann["image_id"] == image_id and ann["category_id"] == 1
                and type(aid) is int and 0 < aid < 2**53 and aid not in gt_ids,
                "Expected unique COCO IDs in the admitted one-image GT roster")
        gt_ids.add(aid)

    # Official preparation converts segmentation and marks ignores in place.
    gt = _coco_view(integer_coco, image, annotations)
    detections = []
    for row in records:
        segmentation = row["segmentation"]
        detections.append({"image_id": image_id, "category_id": 1, "score": row["score"],
                           "segmentation": {"size": list(segmentation["size"]),
                                            "counts": segmentation["counts"].encode("ascii")}})
    dt = gt.loadRes(detections) if detections else _coco_view(gt, image, [])
    matches, ignored, ignored_gt = _official_matches(
        np, coco_eval_type, gt, dt, image_id, gt_ids, len(records))

    rows, matched_gt = [], set()
    for i, (record, match, ignore) in enumerate(zip(records, matches, ignored)):
        start, stop = int(arrays["offsets"][i]), int(arrays["offsets"][i + 1])
        aid = int(match) or None
        status = "ignored" if ignore else "matched" if aid is not None else "unmatched"
        target = None
        if status == "matched":
            require(aid not in matched_gt, "Regular GT cannot match multiple predictions")
            matched_gt.add(aid)
            gt_mask = integer_coco.annToMask(integer_coco.anns[aid])
            require(gt_mask.dtype == np.uint8 and gt_mask.shape == tuple(rows_document["original_hw"])
                    and bool((gt_mask <= 1).all()), "Expected original official binary integer GT raster")
            predicted = arrays["logits"][start:stop] > 0
            reference = gt_mask.ravel()[arrays["positions"][start:stop]] == 1
            target = np.logical_xor(predicted, reference)
        rows.append({
            "native_rank": record["native_rank"], "query_id": record["query_id"], "score": record["score"],
            "status": status, "GT_annotation_id": aid, "start": start, "stop": stop,
            "mask_empty": bool(dt.anns[i + 1]["area"] == 0), "flip_correct": target,
        })

    require(len(rows) == len(records), "Labeling must preserve every original candidate")
    targets = [row["flip_correct"] for row in rows if row["flip_correct"] is not None]
    return {"rows": rows, "summary": {
        "candidates": len(rows), "matched": len(matched_gt),
        "unmatched": sum(row["status"] == "unmatched" for row in rows),
        "ignored": sum(row["status"] == "ignored" for row in rows),
        "empty_masks": sum(row["mask_empty"] for row in rows),
        "empty_bands": sum(row["start"] == row["stop"] for row in rows),
        "labeled_pixels": sum(len(target) for target in targets),
        "flip_correct_pixels": sum(int(target.sum()) for target in targets),
        "GT": len(gt_ids), "ignored_GT": ignored_gt, "unmatched_GT": len(gt_ids) - ignored_gt - len(matched_gt),
    }}
