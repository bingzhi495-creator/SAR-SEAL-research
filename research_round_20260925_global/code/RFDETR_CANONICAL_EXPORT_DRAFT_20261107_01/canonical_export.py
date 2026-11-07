"""Source-only canonical COCO mask export draft."""
import math


def export_coco_mask_predictions(predictions, *, image_id, original_hw,
                                 label2cat, encode_mask):
    """Convert a single-class, two-logit native result serialized with Tensor.tolist().

    Caller supplies original-size bool masks from native all-class top100,
    with score_threshold=None. The caller also supplies the qualified image
    identity, dimensions, fixed {0: 1} inverse mapping, and pinned RLE coder.
    Label 1 is the sole expected unmapped slot. Input order and scores survive.
    """
    if not isinstance(label2cat, dict):
        raise ValueError("label2cat must be the explicit integer mapping {0: 1}")
    if label2cat != {0: 1} or any(
            type(key) is not int or type(value) is not int
            for key, value in label2cat.items()):
        raise ValueError("label2cat must be the explicit integer mapping {0: 1}")
    if (not isinstance(original_hw, (tuple, list)) or len(original_hw) != 2
            or any(type(value) is not int or value <= 0 for value in original_hw)):
        raise ValueError("original_hw must contain positive integer height and width")
    height, width = original_hw
    if type(image_id) is not int:
        raise ValueError("image_id must preserve the canonical integer identity")
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be one native PostProcess dictionary")
    fields = [predictions.get(name) for name in ("labels", "scores", "masks")]
    if any(type(field) is not list for field in fields):
        raise ValueError("labels, scores and masks must be Tensor.tolist() lists")
    labels, scores, masks = fields
    if len(labels) != len(scores) or len(labels) != len(masks):
        raise ValueError("native prediction rows must have equal lengths")
    if len(labels) > 100:
        raise ValueError("native all-class num_select=100 permits at most 100 rows")
    if not callable(encode_mask):
        raise ValueError("encode_mask must be the injected fixed COCO coder wrapper")
    results = []
    for label, score, wrapped_mask in zip(labels, scores, masks):
        if type(label) is not int or label not in (0, 1):
            raise ValueError("the single-class two-logit layout permits only labels 0 and 1")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("native sigmoid scores must be finite numbers in [0, 1]")
        if label not in label2cat:
            continue
        if type(wrapped_mask) is not list or len(wrapped_mask) != 1:
            raise ValueError("each mask must have native shape [1, original_H, original_W]")
        mask = wrapped_mask[0]
        if type(mask) is not list or len(mask) != height:
            raise ValueError("mask height differs from original_hw")
        if any(type(row) is not list or len(row) != width for row in mask):
            raise ValueError("mask width differs from original_hw")
        if any(type(pixel) is not bool for row in mask for pixel in row):
            raise ValueError("PostProcess masks must already be boolean")
        rle = encode_mask(mask)
        if not isinstance(rle, dict):
            raise ValueError("the coder must return one compressed RLE dictionary")
        size = rle.get("size")
        if (type(size) is not list or size != [height, width]
                or any(type(value) is not int for value in size)):
            raise ValueError("RLE size must equal the original height and width")
        counts = rle.get("counts")
        if type(counts) is bytes:
            counts = counts.decode("ascii")
        if type(counts) is not str or not counts or not counts.isascii():
            raise ValueError("compressed RLE counts must be nonempty ASCII bytes or str")
        results.append({
            "image_id": image_id, "category_id": label2cat[label], "score": score,
            "segmentation": {"size": [height, width], "counts": counts},
        })
    return results
