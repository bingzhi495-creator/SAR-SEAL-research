from copy import deepcopy
import unittest

import numpy as np
from pycocotools import mask
import score_test232 as s


def encoded(pixels):
    value = mask.encode(np.asfortranarray(pixels, dtype=np.uint8))
    return dict(size=value["size"], counts=value["counts"].decode())


def fixture():
    gt = dict(info={}, categories=[dict(id=0,name="ship")],
              images=[dict(id=98,file_name="000001.jpg",height=8,width=8),dict(id=4,file_name="000009.jpg",height=8,width=8)],
              annotations=[dict(id=7,image_id=4,category_id=0,area=9.,bbox=[1.,1.,3.,3.],iscrowd=0,
                                segmentation=[[1.,1.,4.,1.,4.,4.,1.,4.]])])
    cells = np.zeros((8,8), dtype=np.uint8)
    cells[1:4,1:4] = 1
    good = encoded(cells)
    cells[3,3] = 0
    cells[4,4] = 1
    changed = encoded(cells)
    items, preds, audits, edits = [], [], [], []
    for i, image in enumerate(gt["images"]):
        ident = i + 1000
        score = 1. - .1*i
        items.append(dict(image_id=ident,image_ref=dict(path=image["file_name"]),original_hw=[8,8]))
        preds.append(dict(image_id=ident,coco=[dict(image_id=ident,category_id=1,score=score,segmentation=good)]))
        audits.append(dict(image_id=ident,post_nms=1,candidates=[dict(native_rank=1,score=score)]))
        change = {"1":{arm:dict(swaps=1,rle=changed) for arm in ("free_natural","native_matched_count")}} if i == 1 else {}
        edits.append(dict(image_id=ident,changes=change))
    return gt, items, preds, audits, edits


class TestScoring(unittest.TestCase):
    def test_category_zero_filename_join_and_sensitivity(self):
        gt, *streams = fixture()
        before = deepcopy((gt,streams))
        output = s.score_populations(gt,s.join_blocks(*streams),{"000001.jpg"})
        self.assertEqual(before,(gt,streams))
        full, subset = output["official"],output["exclude_exact_fit_overlap"]
        self.assertEqual((full["images"],subset["images"],full["candidates"],subset["candidates"]),(2,1,2,1))
        self.assertEqual(set(full["metrics_fraction"]),set(s.ARMS))
        self.assertAlmostEqual(full["metrics_fraction"]["raw"]["AP"],.5)
        self.assertAlmostEqual(subset["metrics_fraction"]["raw"]["AP"],1.)
        self.assertAlmostEqual(subset["metrics_fraction"]["free_natural"]["AP90"],0.)
        self.assertAlmostEqual(subset["paired_contrasts_points"]["free_natural minus raw"]["AP90"],-100.)
        self.assertIsNone(subset["paired_contrasts_points"]["free_natural minus raw"]["APl"])

    def test_frozen_raw_top100_order(self):
        gt, items, preds, audits, changes = fixture()
        preds[0]["coco"] *= 101
        audits[0] = dict(image_id=items[0]["image_id"], post_nms=101,
                         candidates=[dict(score=1.,native_rank=i) for i in range(1,102)])
        blocks = s.join_blocks(items,preds,audits,changes)
        self.assertEqual(len(blocks["000001.jpg"][0]),100)
        edits = {"free_natural":dict(swaps=1,rle=preds[0]["coco"][0]["segmentation"])}
        bad = deepcopy(changes)
        bad[0]["changes"] = {"101":edits}
        with self.assertRaises(ValueError):
            s.join_blocks(items,preds,audits,bad)
        bad[0]["changes"] = {"1":edits}
        with self.assertRaises(ValueError):
            s.join_blocks(items,preds,audits,bad)
        wrong_gt = deepcopy(gt)
        wrong_gt["categories"][0]["id"] = 1
        with self.assertRaises(ValueError):
            s.score_populations(wrong_gt,blocks,{"000001.jpg"})

    def test_receipt_and_ten_frozen_codes(self):
        good = dict(status=s.ACCEPTED,TEST=True,FIT=False,DEV=False,AP=False,GT_opened=False,
                    images=232,exact_native_support=True,optimizer_updates=0,ema_sha256=s.EMA,
                    refit_accepted_sha256=s.HEAD,modes=list(s.ARMS))
        for key in ("manifest_sha256","acceptance_sha256","replay_sha256","edits_sha256","input_predictions_sha256",
                    "candidate_audit_sha256","probability_index_sha256","original_tool_wait_sha256"):
            good[key] = "a"*64
        s.check_acceptance(good)
        for patch in (dict(AP=True),dict(images=222),dict(exact_native_support=False),dict(edits_sha256=None)):
            with self.assertRaises(ValueError):
                s.check_acceptance(dict(good,**patch))
        self.assertEqual(len(set(s.EXCLUDED)),10)
        self.assertEqual(s.digest(("\n".join(sorted(s.EXCLUDED))+"\n").encode()),s.EXCLUSION_SHA)


if __name__ == "__main__":
    unittest.main()
