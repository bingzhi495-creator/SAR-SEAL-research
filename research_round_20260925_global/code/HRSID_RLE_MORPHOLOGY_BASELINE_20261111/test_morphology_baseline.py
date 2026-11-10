import unittest

import numpy as np
from pycocotools import mask

import morphology_baseline as worker


class RleBaselineTest(unittest.TestCase):
    def test_fill_and_remove_share_fixed_sham_budget(self):
        original = np.zeros((12, 12), dtype=np.uint8)
        original[2:7, 2:7] = 1
        original[4, 4] = 0
        original[10, 10] = 1
        rle = worker.encode_mask(original)
        conservative, sham, counts = worker.edit_rle(rle, image_id=7, mapped_rank=2)
        changed = mask.decode(conservative)
        placebo = mask.decode(sham)
        self.assertEqual(counts, {"area": 25, "cap": 2, "add": 1, "delete": 1})
        self.assertEqual((changed[4, 4], changed[10, 10]), (1, 0))
        self.assertEqual(np.count_nonzero(changed != original), 2)
        self.assertEqual(np.count_nonzero(placebo > original), 1)
        self.assertEqual(np.count_nonzero(placebo < original), 1)
        self.assertEqual(sham, worker.edit_rle(rle, image_id=7, mapped_rank=2)[1])

    def test_empty_and_below_cap_masks_keep_original_rle(self):
        blank = np.zeros((4, 4), dtype=np.uint8)
        for canvas in (blank, np.eye(4, dtype=np.uint8)):
            rle = worker.encode_mask(canvas)
            conservative, sham, counts = worker.edit_rle(rle, image_id=1, mapped_rank=1)
            self.assertIs(conservative, rle)
            self.assertIs(sham, rle)
            self.assertEqual(counts["cap"], 0)

    def test_same_rank_and_frozen_original_assignment_with_official_coco(self):
        import os
        from pathlib import Path

        root = Path(os.environ["P314_ORDINARY_CODE"]).resolve()
        p311 = worker.load_p311(root)
        closed = p311.load_closed(root)
        diagnostics = closed.load_diagnostics(root)
        evaluator = diagnostics["dev705_score_template"]
        adapter = diagnostics["dev705_group_sensitivity"].load_official_adapter()
        matcher = diagnostics["dev705_paired_mask_geometry"]
        target = np.zeros((12, 12), dtype=np.uint8)
        target[2:7, 2:7] = 1
        original = target.copy()
        original[4, 4] = 0
        original[10, 10] = 1
        rows = [
            dict(image_id=1, category_id=1, score=.9, segmentation=worker.encode_mask(original)),
            dict(image_id=1, category_id=1, score=.1, segmentation=worker.encode_mask(np.zeros_like(target))),
        ]
        gt = dict(info={}, images=[dict(id=1, height=12, width=12)], categories=[dict(id=1, name="ship")],
                  annotations=[dict(id=1, image_id=1, category_id=1, iscrowd=0, area=25, bbox=[2, 2, 5, 5],
                                    segmentation=worker.encode_mask(target))])
        variants, counts = worker.build_predictions(rows)
        frozen = {name: worker.hash_rows(data) for name, data in variants.items()}
        identity_metrics, identity_evals, identity_ranks = evaluator.evaluate_cell(gt, list(variants["identity"]), [1], adapter)
        for name in ("morphology", "sham"):
            _, _, ranks = evaluator.evaluate_cell(gt, list(variants[name]), [1], adapter)
            self.assertEqual(ranks, identity_ranks)
        self.assertEqual(frozen, {name: worker.hash_rows(data) for name, data in variants.items()})
        self.assertEqual((counts["mapped_rows"], counts["empty_rows"], counts["rows_with_edits"]), (2, 1, 1))
        self.assertAlmostEqual(identity_metrics["AP90"], 1.)
        fixed = worker.strict_fixed_assignments(gt, variants, identity_evals, matcher)
        self.assertEqual(fixed["original_050_matches"], 1)
        self.assertEqual(fixed["variants"]["identity"]["kept"], 1)
        self.assertEqual(fixed["variants"]["morphology"]["kept"], 1)
