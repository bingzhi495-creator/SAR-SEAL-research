import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def require(ok, why):
    if not ok:
        raise ValueError(why)


def freeze_coco(image, annotations, pool):
    """Use stock COCOeval preparation and evaluateImg without metric accumulation."""
    gt = COCO()
    det = COCO()
    category = [dict(id=1, name="ship")]
    gt.dataset = dict(images=[image], annotations=[dict(a) for a in annotations], categories=category)
    image_id = image["id"]
    det.dataset = dict(images=[image], categories=category, annotations=[
        dict(id=p["rank"], image_id=image_id, category_id=1, segmentation=p["rle"], score=p["score"],
             area=p["area"], bbox=mask.toBbox(p["rle"]).tolist(), iscrowd=0)
        for p in pool
    ])
    gt.createIndex()
    det.createIndex()
    evaluator = COCOeval(gt, det, "segm")
    evaluator.params.iouThrs = np.asarray([.50])
    evaluator._prepare()
    overlaps = evaluator.computeIoU(image_id, 1)
    evaluator.ious = {(image_id, 1): overlaps}
    result = evaluator.evaluateImg(image_id, 1, evaluator.params.areaRng[0], 100)
    require(result is not None and result["dtIds"] == [p["rank"] for p in pool],
            "Stock score order and nonempty native pool must agree")
    matched = {int(a): int(d) for a, d, ignored in zip(result["gtIds"], result["gtMatches"][0],
                                                     result["gtIgnore"], strict=True)
               if d and not ignored}
    require(len(set(matched.values())) == len(matched), "Ordinary stock anchors are one to one")
    rles = {a["id"]: a["segmentation"] for a in evaluator._gts[image_id, 1]}
    if not pool:
        return matched, rles, {}
    ids = list(rles)
    require(overlaps.shape == (len(pool), len(ids)), "Exact COCO overlap layout")
    winners = {key: pool[int(np.argmax(overlaps[:, col]))]["rank"] for col, key in enumerate(ids)}
    return matched, rles, winners
