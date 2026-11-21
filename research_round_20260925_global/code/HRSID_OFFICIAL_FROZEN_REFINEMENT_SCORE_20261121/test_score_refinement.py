from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pycocotools import mask

import score_refinement as runner


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = runner.score_module(Path(os.environ["OFFICIAL_SCORE_SOURCE"]))
        cls.adapter = cls.scorer.adapter_from(Path(os.environ["NATIVE_SCORE_ADAPTER"]))

    def images(self):
        pixels = np.zeros((8,8), dtype=np.uint8, order="F")
        pixels[1:4,1:4] = 1
        counts = mask.encode(pixels)["counts"].decode("ascii")
        row = dict(image_id=7, category_id=1, score=.79, segmentation=dict(size=[8,8], counts=counts))
        return [dict(image_id=7, raw_query_count=164, native_count=100, unmapped_slot1_count=98,
                     native_ranks=[2, 5], coco=[row, deepcopy(row)]),
                dict(image_id=9, raw_query_count=164, native_count=100, unmapped_slot1_count=100,
                     native_ranks=[], coco=[])]

    def read(self, blocks, signatures=None):
        raw = b"".join((json.dumps(b)+"\n").encode() for b in blocks)
        ref = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated.jsonl"
            path.write_bytes(raw)
            return runner.load_arm(path, ref, [7,9], (8,8), 2, self.scorer, signatures)

    def test_mask_only_changes_with_tied_scores(self):
        blocks = self.images()
        native, sums, identities = self.read(blocks)
        empty_rle = mask.encode(np.zeros((8,8), dtype=np.uint8, order="F"))["counts"].decode("ascii")
        for row in blocks[0]["coco"]:
            row["segmentation"]["counts"] = empty_rle
        change, sums2, _ = self.read(blocks, identities)
        gt = dict(info={}, categories=[dict(id=1,name="ship")],
                  images=[dict(id=i,width=8,height=8) for i in (7,9)],
                  annotations=[dict(id=1,image_id=7,category_id=1,segmentation=[[1,1,4,1,4,4,1,4]],area=9,
                                    bbox=[1,1,3,3],iscrowd=0)])
        main = dict(images=[7,9],annotations=[1])
        views = dict(official=main,empty_raster_sensitivity=main,image_identity_sensitivity=dict(images=[7],annotations=[1]))
        n = self.scorer.evaluate_views(gt, views, native, sums, self.adapter)
        changed = self.scorer.evaluate_views(gt, views, change, sums2, self.adapter)
        treatments = {s:{a:changed for a in runner.ARMS} for s in runner.SEEDS}
        effects = runner.contrasts(n, treatments)
        self.assertEqual(set(effects), {"SU minus N", "SU=F minus SF", "SU minus SU=F", "SU minus LW=S"})
        self.assertAlmostEqual(effects["SU minus N"]["official"]["arithmetic_mean_points"]["AP"], -100.)
        self.assertIsNone(effects["SU minus N"]["official"]["arithmetic_mean_points"]["APl"])
        self.assertEqual(effects["SU=F minus SF"]["official"]["per_head_seed_points"]["5217"]["AP"], 0.)
        self.assertNotEqual(sums[7], sums2[7])
        self.assertEqual(sums[9], sums2[9])

    def test_equal_count_order_or_score_change_rejected(self):
        original = self.images()
        _, _, identities = self.read(original)
        for changed in ("rank_gap", "score", "category", "image"):
            blocks = deepcopy(original)
            if changed == "rank_gap":
                blocks[0]["native_ranks"][1] = 7
            if changed == "score":
                blocks[0]["coco"][1]["score"] = .78
            if changed == "category":
                blocks[0]["coco"][0]["category_id"] = 3
            if changed == "image":
                blocks.reverse()
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.read(blocks, identities)

    def test_frozen_readout_qualification(self):
        refs = {s:{a:dict(bytes=12, sha256="a"*64) for a in runner.ARMS} for s in runner.SEEDS}
        good = dict(status=runner.ACCEPTED, original_exit=0, TEST=True, GT_opened=False, AP=False, optimizer_updates=0,
                    endpoint_sha256=self.scorer.ENDPOINT, fit_accepted_sha256=runner.FIT, native_predictions_sha256=runner.NATIVE,
                    images=1962, mapped_rows=187796, exact_candidate_identity=True, exact_native_per_image=True,
                    exact_per_candidate_budgets=True, actual_XOR_area_RLE_verified=True, streams=refs)
        self.assertIs(runner.validate_readout(good,self.scorer), refs)
        for exception in (dict(AP=True), dict(original_exit=False), dict(fit_accepted_sha256="0"*64),
                          dict(exact_candidate_identity=False), dict(streams={"5219":refs["5219"]})):
            with self.subTest(exception=exception), self.assertRaises(ValueError):
                runner.validate_readout(dict(good, **exception),self.scorer)


if __name__ == "__main__":
    unittest.main()
