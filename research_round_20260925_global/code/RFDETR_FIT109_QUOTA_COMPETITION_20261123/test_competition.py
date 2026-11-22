"""Numerical parity checks against the official COCO evaluator."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import unittest

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from competition import FrozenImage, ap_curve, marginal_ap


def rect(x0,y0,x1,y1):
    data = np.zeros((12,12),dtype=np.uint8,order="F")
    data[y0:y1,x0:x1] = 1
    return data


def encoded(data):
    rle = mask.encode(data)
    return dict(size=rle["size"],counts=rle["counts"].decode("ascii"))


def official(truth, predictions):
    reference = COCO()
    reference.dataset = deepcopy(truth)
    with redirect_stdout(StringIO()):
        reference.createIndex()
        ev = COCOeval(reference,reference.loadRes(deepcopy(predictions)),"segm")
        ev.evaluate()
        ev.accumulate()
    return ev


def status_rows(ev, image):
    row = next(r for r in ev.evalImgs if r is not None and r["image_id"] == image
               and r["aRng"] == ev.params.areaRng[0] and r["maxDet"] == 100)
    j = {item["id"]:k for k,item in enumerate(ev._gts[(image,1)],1)}
    codes = np.array([j[k] for k in row["dtMatches"].ravel() if k],dtype=np.int64)
    labels = np.zeros(row["dtMatches"].shape,dtype=np.int64)
    labels[row["dtMatches"] > 0] = codes
    labels[row["dtIgnore"]] *= -1
    return labels, row["dtScores"]


def frames():
    # Equal score rows in image 7, interleaved crowd GT in image 3, two nearby
    # regular GTs, and a separate image with no ground-truth instance.
    gt_rects = [
        (3,0,rect(2,2,7,7)), (3,1,rect(8,8,12,12)), (3,0,rect(3,2,8,7)),
        (7,0,rect(0,1,5,6)), (7,0,rect(6,6,11,11)),
    ]
    truth = dict(info={},images=[dict(id=i,width=12,height=12) for i in (3,7,11)],
                 categories=[dict(id=1,name="ship")],
                 annotations=[dict(id=i,category_id=1,image_id=img,iscrowd=crowd,
                                   area=int(bitmap.sum()),segmentation=encoded(bitmap))
                              for i,(img,crowd,bitmap) in enumerate(gt_rects,1)])
    pred_rects = [
        (7,.8,rect(0,1,5,6)), (3,.99,rect(2,2,8,7)), (3,.83,rect(3,2,8,7)),
        (3,.7,rect(9,9,12,12)), (7,.8,rect(6,6,11,11)), (11,.82,rect(3,3,5,5)),
        (3,.65,rect(2,2,7,7)),
    ]
    return truth, [dict(image_id=i,category_id=1,score=s,segmentation=encoded(v))
                   for i,s,v in pred_rects]


class CompetitionTests(unittest.TestCase):
    def test_exact_official_mask_substitutions(self):
        truth, predictions = frames()
        original = official(truth,predictions)
        baseline_frames = {image:status_rows(original,image) for image in (3,7,11)}
        ranks = np.argsort(-np.concatenate([r[1] for r in baseline_frames.values()]),kind="mergesort")
        global_rows = np.concatenate([r[0] for r in baseline_frames.values()],axis=1)[:,ranks]
        idx = np.argsort(ranks)
        expected = original.eval["precision"][:,:,0,0,-1].mean(axis=1)
        np.testing.assert_allclose(ap_curve(global_rows,4),expected,atol=1e-15,rtol=0)
        start = 0
        for image,(statuses,scores) in baseline_frames.items():
            gt_rows = original._gts[(image,1)]
            crowd = np.array([r["iscrowd"] for r in gt_rows],dtype=bool)
            ious = np.asarray(original.ious[(image,1)]).reshape((len(scores),len(crowd)))
            state = FrozenImage(ious,crowd)
            np.testing.assert_array_equal(state.baseline,statuses)
            loc = [i for i,r in enumerate(predictions) if r["image_id"] == image]
            loc.sort(key=lambda i:-predictions[i]["score"])
            for rank, position in enumerate(loc):
                for alternative in (rect(2,2,7,7),rect(8,8,12,12),rect(0,0,0,0)):
                    changed = deepcopy(predictions)
                    changed[position]["segmentation"] = encoded(alternative)
                    counterfactual = official(truth,changed)
                    expected_status = status_rows(counterfactual,image)[0]
                    incoming = np.asarray(counterfactual.ious[(image,1)]).reshape(ious.shape)[rank]
                    substitute = state.substitute(rank,incoming)
                    np.testing.assert_array_equal(substitute,expected_status)
                    actual = marginal_ap(global_rows,idx[start:start+len(loc)],substitute,4)
                    expected_delta = counterfactual.eval["precision"][:,:,0,0,-1].mean(axis=1)-expected
                    np.testing.assert_allclose(actual,expected_delta,atol=1e-15,rtol=0)
            start += len(loc)


if __name__ == "__main__":
    unittest.main()
