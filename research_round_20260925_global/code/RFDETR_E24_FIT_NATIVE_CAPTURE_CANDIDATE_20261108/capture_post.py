"""Independently verify one native FIT image's sparse radius-two capture."""
import math
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _band_positions(np, mask):
    # Integral-image rectangle sums count both signs in each clipped 5x5 window.
    prefix = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)))
    prefix = prefix.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    y, x = (np.arange(length) for length in mask.shape)
    y0, x0 = np.maximum(y - 2, 0), np.maximum(x - 2, 0)
    y1, x1 = np.minimum(y + 3, mask.shape[0]), np.minimum(x + 3, mask.shape[1])
    foreground = (prefix[np.ix_(y1, x1)] - prefix[np.ix_(y0, x1)]
                  - prefix[np.ix_(y1, x0)] + prefix[np.ix_(y0, x0)])
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return np.flatnonzero((foreground > 0) & (foreground < area))


def _decode_mask(np, codec, segmentation, hw):
    require(type(segmentation) is dict and set(segmentation) == {"size", "counts"}
            and segmentation["size"] == hw
            and all(type(length) is int for length in segmentation["size"])
            and type(segmentation["counts"]) is str and bool(segmentation["counts"])
            and all("0" <= char <= "o" for char in segmentation["counts"]),
            "Expected original-grid compressed ASCII RLE")
    native_rle = {"size": hw, "counts": segmentation["counts"].encode("ascii")}
    mask = codec.decode(native_rle)
    require(isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and mask.shape == tuple(hw)
            and bool((mask <= 1).all())
            and codec.encode(np.asfortranarray(mask)) == native_rle,
            "Compressed RLE must roundtrip to the exact canonical binary mask")
    return mask


def verify_capture(np, codec, rows_document, arrays):
    """Check supplied ROWS and arrays in memory, returning aggregate evidence."""
    require(type(rows_document) is dict
            and set(rows_document) == {"image_id", "image_sha256", "gray_sha256", "original_hw", "records"}
            and type(rows_document["image_id"]) is int
            and all(type(rows_document[key]) is str and re.fullmatch(r"[0-9a-f]{64}", rows_document[key])
                    for key in ("image_sha256", "gray_sha256")),
            "Expected exact single-image ROWS schema")
    hw, records = rows_document["original_hw"], rows_document["records"]
    require(type(hw) is list and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw)
            and math.prod(hw) <= 2 * 1024**2 and type(records) is list and len(records) <= 100,
            "Expected bounded original grid and at most 100 retained records")
    require(set(arrays) == {"offsets", "positions", "logits"}, "Expected exact sparse array roles")
    offsets, positions, logits = (arrays[key] for key in ("offsets", "positions", "logits"))
    require(all(isinstance(a, np.ndarray) and a.ndim == 1 and a.dtype == dtype
                for a, dtype in zip((offsets, positions, logits), (np.int64, np.uint32, np.float32))),
            "Expected 1-D int64 offsets, uint32 positions and float32 logits")
    require(len(offsets) == len(records) + 1 and offsets[0] == 0
            and offsets[-1] == len(positions) == len(logits)
            and bool((offsets[1:] >= offsets[:-1]).all()),
            "Offsets must partition all sparse positions and logits")
    require(bool(np.isfinite(logits).all()), "Sparse logits must be finite")

    previous_rank, previous_query, previous_score = 0, -1, 1.0
    queries = set()
    for index, row in enumerate(records):
        require(type(row) is dict and set(row) == {"native_rank", "query_id", "score", "segmentation"},
                "Expected exact retained record schema")
        rank, query, score = row["native_rank"], row["query_id"], row["score"]
        require(type(rank) is int and previous_rank < rank <= 100
                and type(query) is int and 0 <= query < 100 and query not in queries
                and type(score) is float and math.isfinite(score) and 0 <= score <= previous_score
                and float(np.float32(score)) == score
                and (score < previous_score or query > previous_query),
                "Expected canonical native rank, unique query and stable sigmoid score order")
        previous_rank, previous_query, previous_score = rank, query, score
        queries.add(query)
        mask = _decode_mask(np, codec, row["segmentation"], hw)
        span = slice(int(offsets[index]), int(offsets[index + 1]))
        expected = _band_positions(np, mask)
        require(np.array_equal(positions[span], expected),
                "Sparse positions must equal the sorted clipped radius-two prediction band")
        require(np.array_equal(logits[span] > 0, mask.ravel()[expected] == 1),
                "Strictly positive sparse logits must reproduce the RLE foreground in the band")

    return {
        "images": 1, "retained": len(records), "empty_bands": int((offsets[1:] == offsets[:-1]).sum()),
        "support_pixels": len(positions), "positive_logits": int((logits > 0).sum()),
        "negative_logits": int((logits < 0).sum()), "zero_logits": int((logits == 0).sum()),
    }
