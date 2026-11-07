"""Pack lossless native FIT mask logits on predicted-only boundary pixels."""
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pack_candidates(rows, original_hw, *, np, codec, dilate):
    records, positions, logits = [], [], []
    offsets = [0]
    queries = set()
    previous_rank, previous_score = 0, 1.0
    for row in rows:
        rank, query, score = (row[key] for key in ("native_rank", "query_id", "score"))
        require(type(rank) is int and previous_rank < rank <= 100
                and type(query) is int and 0 <= query < 100 and query not in queries
                and type(score) is float and math.isfinite(score) and 0 <= score <= previous_score,
                "Preserve original native order, query and score")
        previous_rank, previous_score = rank, score
        queries.add(query)
        mask = np.asarray(row["mask"])
        plane = np.asarray(row["mask_logits"])
        require(mask.dtype == np.bool_ and plane.dtype == np.float32 and mask.shape == plane.shape == original_hw
                and bool(np.isfinite(plane).all()) and bool(np.array_equal(mask, plane > 0)),
                "Native binary mask must equal positive original-grid FP32 logits")
        support = dilate(np, mask) & dilate(np, ~mask)
        flat = np.flatnonzero(support).astype(np.uint32)
        selected = plane.reshape(-1)[flat]
        rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
        require(type(rle.get("counts")) is bytes and rle["size"] == list(original_hw)
                and np.array_equal(codec.decode(rle), mask),
                "Exact native compressed RLE roundtrip")
        records.append(dict(native_rank=rank,query_id=query,score=score,
                            segmentation=dict(size=rle["size"],counts=rle["counts"].decode("ascii"))))
        positions.append(flat)
        logits.append(selected)
        offsets.append(offsets[-1] + len(flat))
    return dict(records=records,offsets=np.asarray(offsets,dtype=np.int64),
                positions=np.concatenate(positions) if positions else np.empty(0,dtype=np.uint32),
                logits=np.concatenate(logits) if logits else np.empty(0,dtype=np.float32))
