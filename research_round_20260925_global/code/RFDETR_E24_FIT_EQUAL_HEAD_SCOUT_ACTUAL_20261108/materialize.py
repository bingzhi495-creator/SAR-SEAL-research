"""Build one private FIT scorer image from admitted frozen capture artifacts."""
from hashlib import sha256


FEATURE_COLUMNS = ("abs_logit", "mean_contrast", "rank")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _feature_arrays(np, codec, records, source, gray, extract_features):
    positions, offsets, logits = (source[key] for key in ("positions", "offsets", "logits"))
    features = np.empty((len(positions), len(FEATURE_COLUMNS)), dtype=np.float32)
    edit_sign = np.empty(len(positions), dtype=np.int8)
    valid = np.empty(len(positions), dtype=np.bool_)
    # Features depend only on the frozen image and the entire captured population.
    for index, record in enumerate(records):
        span = slice(int(offsets[index]), int(offsets[index + 1]))
        segmentation = record["segmentation"]
        rle = {"size": segmentation["size"], "counts": segmentation["counts"].encode("ascii")}
        mask = codec.decode(rle).astype(np.bool_)
        details = extract_features(gray, mask, positions[span], logits[span])
        columns = [details[key] for key in FEATURE_COLUMNS]
        require(np.array_equal(details["flat_indices"], positions[span])
                and all(col.shape == positions[span].shape and col.dtype == np.float32
                        and bool(np.isfinite(col).all()) for col in columns)
                and np.array_equal(columns[0], np.abs(logits[span])),
                "Features must retain every original sparse position and absolute FP32 logit")
        sign, usable = details["edit_sign"], details["valid"]
        require(sign.shape == usable.shape == positions[span].shape
                and sign.dtype == np.int8 and usable.dtype == np.bool_
                and np.array_equal(sign, np.where(logits[span] > 0, -1, 1)),
                "Expected exact native edit direction and aligned common validity")
        for j, column in enumerate(columns):
            features[span, j] = column
        edit_sign[span], valid[span] = sign, usable
    return {"features": features, "edit_sign": edit_sign, "valid": valid}


def materialize_image(np, codec, coco_eval_type, *, integer_coco, rows_document, arrays, gray,
                      verify_capture, build_fixed_labels, extract_features):
    """Return the complete fixed feature population and separate matched-only y."""
    require(isinstance(gray, np.ndarray) and gray.dtype == np.uint8
            and gray.ndim == 2 and tuple(rows_document["original_hw"]) == gray.shape
            and sha256(gray.tobytes(order="C")).hexdigest() == rows_document["gray_sha256"],
            "Gray pixels must reproduce the frozen original-grid capture digest")
    require(set(arrays) == {"positions", "offsets", "logits"}, "Expected original sparse array roles")
    # A loaded NpzFile materializes on indexing. Read each array once per image.
    source = {key: arrays[key] for key in ("positions", "offsets", "logits")}
    labels = build_fixed_labels(np, codec, coco_eval_type, integer_coco=integer_coco,
                                rows_document=rows_document, arrays=source, verify_capture=verify_capture)
    records = rows_document["records"]
    require(len(labels["rows"]) == len(records), "Labels must retain the entire native population")
    features = _feature_arrays(np, codec, records, source, gray, extract_features)

    match_rows, targets = [], []
    label_offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for index, (record, row) in enumerate(zip(records, labels["rows"])):
        start, stop = int(source["offsets"][index]), int(source["offsets"][index + 1])
        require(row["native_rank"] == record["native_rank"] and row["query_id"] == record["query_id"]
                and row["score"].hex() == record["score"].hex()
                and row["start"] == start and row["stop"] == stop,
                "Label identities must use the exact original candidate and sparse span")
        target = row["flip_correct"]
        label_offsets[index + 1] = label_offsets[index]
        if row["status"] == "matched":
            require(isinstance(target, np.ndarray) and target.dtype == np.bool_
                    and target.shape == (stop - start,),
                    "Matched labels must span the whole unchanged predicted support")
            targets.append(target)
            label_offsets[index + 1] += len(target)
        else:
            require(row["status"] in ("unmatched", "ignored") and target is None,
                    "Unmatched and ignored candidates are unlabeled")
        match_rows.append({key: value for key, value in row.items() if key != "flip_correct"})
    y = np.concatenate(targets) if targets else np.empty(0, dtype=np.bool_)
    summary = labels["summary"]
    require(len(y) == summary["labeled_pixels"] and int(y.sum()) == summary["flip_correct_pixels"],
            "Compact labels must agree with the fixed match summary")
    return {"rows_document": rows_document, "match_rows": match_rows,
            "arrays": {**source, **features, "label_offsets": label_offsets, "y": y},
            "summary": summary}
