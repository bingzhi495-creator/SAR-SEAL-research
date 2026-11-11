"""Validate author mask results before scoring without box-derived areas."""

from math import isfinite


def prepare_mask_rows(export, image_shapes, *, expected_per_image=100):
    """Return mask-only rows in native order for a sealed, ordered image roster."""
    if type(expected_per_image) is not int or expected_per_image <= 0 or not image_shapes:
        raise ValueError("An ordered image roster and positive fixed native row count are required")
    if type(export) is not dict or set(export) != {"bbox", "mask"}:
        raise ValueError("Expected separate native author box and mask streams")
    boxes, masks = export["bbox"], export["mask"]
    total = len(image_shapes) * expected_per_image
    if type(boxes) is not list or type(masks) is not list or len(boxes) != total or len(masks) != total:
        raise ValueError("Author streams do not match the fixed native candidate counts")

    mask_rows = []
    expected_ids = (image_id for image_id in image_shapes for _ in range(expected_per_image))
    for image_id, box, mask in zip(expected_ids, boxes, masks, strict=True):
        if (type(box) is not dict or set(box) != {"image_id", "category_id", "score", "bbox"}
                or type(mask) is not dict or set(mask) != {"image_id", "category_id", "score", "segmentation"}):
            raise ValueError("Unexpected fields in native author rows")
        score = mask["score"]
        if (type(mask["image_id"]) is not int or type(box["image_id"]) is not int
                or mask["image_id"] != image_id or box["image_id"] != image_id
                or type(mask["category_id"]) is not int or type(box["category_id"]) is not int
                or mask["category_id"] != 1 or box["category_id"] != 1
                or type(score) not in (int, float) or not isfinite(score) or not 0 <= score <= 1
                or type(box["score"]) not in (int, float) or box["score"] != score):
            raise ValueError("Author image, class, score, or order differs from its paired row")
        bbox = box["bbox"]
        if type(bbox) is not list or len(bbox) != 4 or any(type(x) not in (int, float) or not isfinite(x) for x in bbox):
            raise ValueError("Invalid native box row")
        rle = mask["segmentation"]
        if (type(rle) is not dict or set(rle) != {"size", "counts"}
                or rle["size"] != list(image_shapes[image_id])
                or type(rle["counts"]) is not str or not rle["counts"] or not rle["counts"].isascii()):
            raise ValueError("Mask RLE must have the original image canvas")
        mask_rows.append(dict(image_id=image_id, category_id=1, score=score,
                              segmentation=dict(size=list(rle["size"]), counts=rle["counts"])))
    return mask_rows
