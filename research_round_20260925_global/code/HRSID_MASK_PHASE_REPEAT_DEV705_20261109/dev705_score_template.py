"""A paired score-template diagnostic on frozen equal nonempty DEV705 prefixes."""

from contextlib import redirect_stdout
from copy import deepcopy
from importlib.metadata import version
from io import StringIO
import math

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dev705_group_sensitivity import load_official_adapter
from dev705_paired_coverage import ARMS, CANVAS, DEV_IMAGES, GT_COUNT, group_rows, require


AP_METRICS = ("AP", "AP50", "AP75", "AP90", "APs")
MATCH_FIELDS = ("dtIds", "dtMatches", "dtIgnore", "gtIds", "gtMatches", "gtIgnore")


def nonempty_by_image(rows, images):
    rles = group_rows(rows, images)
    offset, result = 0, []
    for block in rles:
        count = len(block)
        nonempty = mask.area(block) > 0 if block else ()
        result.append([row for row, keep in zip(rows[offset:offset + count], nonempty, strict=True) if keep])
        offset += count
    require(offset == len(rows), "Complete original prediction row accounting required")
    return result


def prefix_rows(native, center):
    a, b, sizes = [], [], []
    for rows_a, rows_b in zip(native, center, strict=True):
        count = min(len(rows_a), len(rows_b))
        sizes.append(count)
        a.extend(rows_a[:count])
        b.extend(rows_b[:count])
    return dict(native=a, center=b), sizes


def score_template(geometry, donor):
    result = []
    previous_image, previous_score = None, 1.
    for row, template in zip(geometry, donor, strict=True):
        image, score = row["image_id"], template["score"]
        require(image == template["image_id"] and type(score) in (int, float)
                and math.isfinite(score) and 0. <= score <= 1.
                and (image != previous_image or score <= previous_score),
                "Exact per-image native order and finite nonincreasing template scores required")
        result.append(dict(row, score=score))
        previous_image, previous_score = image, score
    return result


def evaluate_cell(gt, predictions, image_ids, adapter):
    coco = COCO()
    coco.dataset = deepcopy(gt)
    captured = []

    def capture(*args):
        evaluator = COCOeval(*args)
        captured.append(evaluator)
        return evaluator

    with redirect_stdout(StringIO()):
        coco.createIndex()
        values = adapter.evaluate_population(dict(COCO=COCO, COCOeval=capture), coco, predictions, image_ids)
    evaluator, = captured
    require(len(values) == 12 and all(math.isfinite(value) and -1. <= value <= 1. for value in values),
            "Complete finite original COCO result required")
    params = evaluator.params
    area = list(params.areaRngLbl).index("all")
    max_det = list(params.maxDets).index(100)
    t75, = np.flatnonzero(np.isclose(params.iouThrs, .75))
    t90, = np.flatnonzero(np.isclose(params.iouThrs, .90))
    precision = evaluator.eval["precision"][t90, :, 0, area, max_det]
    valid = precision[precision > -1]
    recall = evaluator.eval["recall"][:, 0, area, max_det]
    metrics = dict(AP=values[0], AP50=values[1], AP75=values[2], APs=values[3],
                   AP90=float(valid.mean()) if valid.size else -1.,
                   recall75=float(recall[t75]), recall90=float(recall[t90]))
    start = area * len(image_ids)
    rank_layout = tuple(None if item is None else (item["image_id"], tuple(item["dtIds"]), tuple(item["dtScores"]))
                        for item in evaluator.evalImgs[start:start + len(image_ids)])
    return metrics, evaluator.evalImgs, rank_layout


def require_same_matching(reference, alternate):
    for left, right in zip(reference, alternate, strict=True):
        if left is None or right is None:
            require(left is right, "Within-image matching evidence changed across score donors")
        else:
            require(all(left[key] == right[key] for key in ("image_id", "category_id", "aRng", "maxDet"))
                    and all(np.array_equal(left[key], right[key]) for key in MATCH_FIELDS),
                    "Within-image matching evidence changed across score donors")


def differences(a, b):
    return {name: 100. * (b[name] - a[name]) if min(a[name], b[name]) >= 0 else None for name in AP_METRICS}


def evaluate_pair(gt, native_rows, center_rows):
    """Evaluate geometry/content arm × donor score arm without changing within-image order."""
    require(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Exact original DEV705 CPU COCO runtime required")
    images, annotations = gt["images"], gt["annotations"]
    require(len(images) == DEV_IMAGES and len(annotations) == GT_COUNT
            and [category["id"] for category in gt["categories"]] == [1]
            and all(type(image["id"]) is int and (image["height"], image["width"]) == CANVAS for image in images),
            "Exact original DEV705 population and canvas required")
    image_ids = [image["id"] for image in images]
    known_ids = set(image_ids)
    require(len(known_ids) == DEV_IMAGES and len({row["id"] for row in annotations}) == GT_COUNT
            and all(type(row["id"]) is int and type(row["image_id"]) is int and row["image_id"] in known_ids
                    and type(row["category_id"]) is int and row["category_id"] == 1 for row in annotations),
            "Exact original DEV705 ship annotation identities required")

    selected, counts = prefix_rows(nonempty_by_image(native_rows, images), nonempty_by_image(center_rows, images))
    adapter = load_official_adapter()
    cells, donor_layouts = {}, {}
    for geometry in ARMS:
        cells[geometry] = {}
        reference = None
        for donor in ARMS:
            predictions = score_template(selected[geometry], selected[donor])
            metrics, matching, layout = evaluate_cell(gt, predictions, image_ids, adapter)
            if donor in donor_layouts:
                require(layout == donor_layouts[donor], "Same donor must produce identical global score rank order")
            else:
                donor_layouts[donor] = layout
            if reference is None:
                reference = matching
            else:
                require_same_matching(reference, matching)
                require(all(metrics[key] == cells[geometry]["native"][key] for key in ("recall75", "recall90")),
                        "Fixed-geometry recalls must be invariant across score donors")
            cells[geometry][donor] = metrics

    a, b = (cells[geometry] for geometry in ARMS)
    return dict(
        status="HRSID_DEV705_EQUAL_NONEMPTY_SCORE_TEMPLATE_DIAGNOSTIC_COMPLETE",
        images=len(images), annotations=len(annotations), retained_prefix_rows=sum(counts),
        images_with_prefix_rows=sum(count > 0 for count in counts),
        images_without_prefix_rows=sum(count == 0 for count in counts), cells=cells,
        contrasts_points=dict(
            score_on_native_geometry=differences(a["native"], a["center"]),
            score_on_center_geometry=differences(b["native"], b["center"]),
            geometry_with_native_scores=differences(a["native"], b["native"]),
            geometry_with_center_scores=differences(a["center"], b["center"]),
        ),
    )

