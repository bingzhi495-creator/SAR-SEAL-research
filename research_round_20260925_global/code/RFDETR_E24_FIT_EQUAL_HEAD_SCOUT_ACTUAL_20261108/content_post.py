"""Independent content audit of one materialized FIT scorer image."""


_SOURCE_DTYPES = {"positions": "uint32", "offsets": "int64", "logits": "float32"}
_MATCH_FIELDS = {"native_rank", "query_id", "score", "status", "GT_annotation_id",
                 "start", "stop", "mask_empty"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _typed(np, array, dtype, shape):
    return isinstance(array, np.ndarray) and array.dtype == np.dtype(dtype) and array.shape == shape


def _binary(np, mask, hw):
    require(_typed(np, mask, "uint8", hw) and bool((mask <= 1).all()),
            "Expected official original-grid binary mask")
    return mask


def _valid_neighbors(np, mask, positions, center):
    if not len(positions):
        return np.empty(0, dtype=np.bool_)
    h, w = mask.shape
    row, col = np.divmod(positions.astype(np.intp, copy=False), w)
    y0, x0 = np.maximum(row - 2, 0), np.maximum(col - 2, 0)
    y1, x1 = np.minimum(row + 3, h), np.minimum(col + 3, w)
    summed = np.pad(mask, ((1, 0), (1, 0))).cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    inside = summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0] - center
    outside = (y1 - y0) * (x1 - x0) - 1 - inside
    return (inside >= 2) & (outside >= 2)


def verify_materialized(np, *, rows_document, source_arrays, match_rows, feature_arrays, integer_coco):
    """Check a complete image against original artifacts and fixed match labels."""
    records = rows_document["records"]
    image_id = rows_document["image_id"]
    hw = tuple(rows_document["original_hw"])
    image = integer_coco.imgs[image_id]
    require(hw == (image["height"], image["width"]) and 0 < hw[0] * hw[1] < 2**31,
            "Expected exact original image geometry")
    require(type(match_rows) is list and len(match_rows) == len(records),
            "All original candidates must remain in the materialized population")
    require(set(source_arrays) == _SOURCE_DTYPES.keys()
            and set(feature_arrays) == _SOURCE_DTYPES.keys() | {"features", "edit_sign", "valid", "label_offsets", "y"},
            "Expected exact source and materialized array roles")
    source = {key: source_arrays[key] for key in _SOURCE_DTYPES}
    n = len(source["positions"])
    shapes = {"positions": (n,), "offsets": (len(records) + 1,), "logits": (n,)}
    for key, dtype in _SOURCE_DTYPES.items():
        original, output = source[key], feature_arrays[key]
        require(_typed(np, original, dtype, shapes[key]) and _typed(np, output, dtype, shapes[key])
                and np.array_equal(original.view(np.uint32) if key == "logits" else original,
                                   output.view(np.uint32) if key == "logits" else output),
                "Original sparse arrays must be preserved exactly")
    offsets, positions, logits = source["offsets"], source["positions"], source["logits"]
    require(offsets[0] == 0 and offsets[-1] == n and bool((offsets[1:] >= offsets[:-1]).all())
            and bool((positions < hw[0] * hw[1]).all()) and bool(np.isfinite(logits).all()),
            "Expected complete bounded source support")
    matrix, sign, valid, yoff, y = (feature_arrays[key] for key in
                                   ("features", "edit_sign", "valid", "label_offsets", "y"))
    require(_typed(np, matrix, "float32", (n, 3)) and _typed(np, sign, "int8", (n,))
            and _typed(np, valid, "bool", (n,)) and _typed(np, yoff, "int64", offsets.shape)
            and isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype == np.bool_
            and yoff[0] == 0 and yoff[-1] == len(y) and bool((yoff[1:] >= yoff[:-1]).all()),
            "Invalid typed feature or compact label arrays")
    positive = logits > 0
    require(np.array_equal(matrix[:, 0].view(np.uint32), np.abs(logits).view(np.uint32))
            and np.array_equal(sign, np.where(positive, np.int8(-1), np.int8(1))),
            "Feature absolute logit and edit direction must match the original sign")
    descriptors = matrix[:, 1:]
    require(bool(np.isfinite(descriptors).all()) and bool((np.abs(descriptors) <= 1).all())
            and bool((descriptors[~valid] == 0).all()),
            "Mean and rank must be bounded finite and neutral on invalid support")

    counts = dict(matched=0, unmatched=0, ignored=0, empty_masks=0, empty_bands=0)
    matched_ids = set()
    for index, (record, row) in enumerate(zip(records, match_rows)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        require(type(row) is dict and set(row) == _MATCH_FIELDS
                and all(type(row[key]) is int and row[key] == record[key] for key in ("native_rank", "query_id"))
                and type(row["score"]) is float and row["score"].hex() == record["score"].hex()
                and type(row["start"]) is int and type(row["stop"]) is int
                and (row["start"], row["stop"]) == (start, stop),
                "Candidate rank, query, score, order or span changed")
        span = slice(start, stop)
        segmentation = record["segmentation"]
        prediction = {"image_id": image_id, "segmentation": {
            "size": segmentation["size"], "counts": segmentation["counts"].encode("ascii")}}
        pred = _binary(np, integer_coco.annToMask(prediction), hw)
        center = pred.ravel()[positions[span]]
        empty = not bool(pred.any())
        require(type(row["mask_empty"]) is bool and row["mask_empty"] == empty
                and np.array_equal(center == 1, positive[span])
                and np.array_equal(valid[span], _valid_neighbors(np, pred, positions[span], center)),
                "Expected original mask sign, emptiness and independent clipped neighbor validity")

        status, aid = row["status"], row["GT_annotation_id"]
        require(status in ("matched", "unmatched", "ignored"), "Unknown fixed matching status")
        ann = None
        if aid is not None:
            require(type(aid) is int and aid in integer_coco.anns, "Unknown original GT identity")
            ann = integer_coco.anns[aid]
            require(ann["image_id"] == image_id and ann["category_id"] == 1,
                    "GT identity must be an original ship annotation on this image")
        label_span = y[int(yoff[index]):int(yoff[index + 1])]
        if status == "matched":
            require(ann is not None and not ann["iscrowd"] and aid not in matched_ids,
                    "Matched positives require unique regular original annotations")
            matched_ids.add(aid)
            gt = _binary(np, integer_coco.annToMask(ann), hw)
            require(np.array_equal(label_span, center != gt.ravel()[positions[span]]),
                    "Matched labels must equal original mask/GT XOR on exact support")
        else:
            require(len(label_span) == 0 and (status == "ignored" or aid is None),
                    "Unmatched and ignored candidates must remain unlabeled")
        counts[status] += 1
        counts["empty_masks"] += empty
        counts["empty_bands"] += start == stop

    return dict(images=1, candidates=len(records), **counts, support_pixels=n,
                valid_pixels=int(valid.sum()), labeled_pixels=len(y), flip_correct_pixels=int(y.sum()))
