"""Map native SRNet mask outputs and score the fixed integer SSDD DEV187."""

from importlib.metadata import version

import numpy as np
from pycocotools import mask as codec


def require(condition, message):
    if not condition:
        raise ValueError(message)


def map_native_image(image, box_results, mask_results):
    """Return paired COCO predictions and a separate native rotated-box sidecar.

    Accepts one original image-manifest row and the two elements returned by
    author simple_test for that image. The one ship class occupies native index
    0. Both streams preserve the author's emitted per-image row order.
    """
    image_id, hw = image["image_id"], tuple(image["original_hw"])
    require(type(image_id) is int and len(hw) == 2 and
            all(type(n) is int and n > 0 for n in hw),
            "Canonical logical image ID and original (height, width) required")
    require(isinstance(box_results, (list, tuple)) and isinstance(mask_results, (list, tuple))
            and len(box_results) == len(mask_results) == 1,
            "Expected only native class index 0 (ship)")
    boxes, masks = box_results[0], mask_results[0]
    require(isinstance(boxes, np.ndarray) and boxes.shape == (len(masks), 6)
            and np.isfinite(boxes).all(), "Expected aligned finite SRNet rbox and mask rows")
    predictions, rotated = [], []
    for rank, (box, mask) in enumerate(zip(boxes, masks, strict=True), start=1):
        require(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == hw,
                "Expected native boolean mask on the exact original image grid")
        rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
        score = float(box[5])
        require(0 <= score <= 1, "Native ship score outside [0, 1]")
        predictions.append(dict(image_id=image_id, category_id=1, score=score,
                                segmentation=dict(size=list(hw), counts=rle["counts"].decode("ascii"))))
        rotated.append(dict(image_id=image_id, category_id=1, native_rank=rank, score=score,
                            rbox=[float(value) for value in box[:5]]))
    return predictions, rotated


def score_original_dev187(scorer, official, image_manifest, integer_gt, image_results):
    """Score closed native exports with the existing DEV187 scorer.

    scorer is the pinned ssdd_phase_dev_score module and official its pinned
    COCO adapter. The two artifacts are scorer.Artifact(ref, raw) instances.
    image_results contains (logical image ID, flat prediction rows) in fixed
    manifest order, including an empty list for each zero-candidate image.
    """
    require(version("pycocotools") == "2.0.11", "Exact original COCO backend required")
    _, ids, items = scorer.fixed_images(image_manifest)
    require(integer_gt.ref["sha256"] == scorer.INTEGER_GT_SHA
            and integer_gt.ref["bytes"] == scorer.INTEGER_GT_BYTES,
            "Exact original DEV187 integer GT required")
    _, gt = scorer.read_artifact(integer_gt, 1024**2, dict)
    scorer.validate_gt(gt, ids, items)

    predictions = []
    for expected_id, (image_id, rows) in zip(ids, image_results, strict=True):
        require(type(image_id) is int and image_id == expected_id and isinstance(rows, list),
                "Prediction blocks must cover every fixed logical DEV187 image in order")
        for row in rows:
            require(isinstance(row, dict) and row.keys() == {"image_id", "category_id", "score", "segmentation"}
                    and type(row["image_id"]) is int and row["image_id"] == image_id,
                    "Native COCO rows must match the image block and exclude the rotated-box sidecar")
        predictions.extend(rows)

    image_sizes = [dict(image_id=item["image_id"], height=item["original_hw"][0],
                        width=item["original_hw"][1]) for item in items]
    official.validate_predictions(predictions, image_sizes)
    return scorer.score_one(official, gt, predictions, ids)
