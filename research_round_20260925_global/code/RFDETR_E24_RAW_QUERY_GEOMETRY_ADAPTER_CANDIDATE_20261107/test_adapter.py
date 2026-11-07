"""Fabricated counts and native-rank witnesses only."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

from adapter import FLOAT, INTEGER, decompose


class DecompositionTests(unittest.TestCase):
    def test_nested_threshold_deficits_are_additive(self):
        scores = [(.89, .80, .65), (.95, .89, .80), (.96, .92, .89), (.95, .95, .90)]
        rows = [dict(by_GT={gt: dict(raw=raw, retained=retained, assigned=assigned)
                           for gt in (INTEGER, FLOAT)}) for raw, retained, assigned in scores]
        result = decompose(rows)
        self.assertEqual(result[INTEGER]["0.90"], dict(
            pairs=4, raw=3, retained=2, assigned=1, geometry=1, selection=1, assignment=1))
        self.assertEqual(result[FLOAT]["0.75"], dict(
            pairs=4, raw=4, retained=4, assigned=3, geometry=0, selection=0, assignment=1))

    def test_nonnested_maxima_are_rejected(self):
        row = dict(by_GT={gt: dict(raw=.6, retained=.7, assigned=.5) for gt in (INTEGER, FLOAT)})
        with self.assertRaisesRegex(ValueError, "Nested raw, retained and assigned overlaps"):
            decompose([row])

    def test_raw_witness_matches_the_original_slot0_rows_by_true_query(self):
        from adapter import validate_membership

        masks = [dict(size=[2, 3], counts=f"synthetic{qid}") for qid in range(100)]
        ranks = [100, 2, *range(103, 201)]
        scores = [.2, .8, *([.1] * 98)]
        manifest = dict(image_id=5, image_sha256="a"*64, original_hw=[2, 3], canonical_json_sha256="b"*64)
        block = dict(image_id=5, native_ranks=[2, 100],
                     coco=[dict(segmentation=masks[qid], score=scores[qid]) for qid in (1, 0)])
        raw = dict(image_id=5, image_sha256="a"*64, original_hw=[2, 3], primary_canonical_json_sha256="b"*64,
                   raw_masks=masks, slot0_scores=scores, slot0_native_ranks=ranks,
                   retained=[dict(native_rank=2, query_id=1), dict(native_rank=100, query_id=0)])
        self.assertEqual(validate_membership(raw, manifest, block), [1, 0])
        changed = deepcopy(raw)
        changed["retained"][1]["query_id"] = 2
        with self.assertRaisesRegex(ValueError, "Exact slot0 membership"):
            validate_membership(changed, manifest, block)
        changed = deepcopy(raw)
        changed["raw_masks"][0]["counts"] = "different"
        with self.assertRaisesRegex(ValueError, "exact retained score and RLE"):
            validate_membership(changed, manifest, block)

    def test_float_and_integer_reproduce_saved_retained_and_fixed_rows(self):
        from adapter import image_pair_overlaps

        class Matrix:
            shape = (100, 1)

            def __init__(self, values):
                self.values = values

            def __getitem__(self, indices):
                row, _ = indices
                return self.values.get(row, .1)

        coco = lambda version: SimpleNamespace(
            anns={7: dict(image_id=5, category_id=1, iscrowd=0)},
            annToRLE=lambda ann: dict(size=[2, 3], counts=version))
        cocos = {INTEGER: coco("integer"), FLOAT: coco("floating")}
        mask_api = SimpleNamespace(iou=lambda rows, gt, flags:
                                   Matrix({1: .91, 3: .80, 42: .99}) if gt[0]["counts"] == "integer"
                                   else Matrix({1: .88, 3: .69, 42: .93}))
        row = dict(GT_annotation_id=7, assigned_detection_ordinal=9, assigned_native_rank=99, by_GT={
            INTEGER: dict(mapped_oracle_max_IoU=.91, mapped_oracle_best_native_rank=2, fixed_assigned_IoU=.80),
            FLOAT: dict(mapped_oracle_max_IoU=.88, mapped_oracle_best_native_rank=2, fixed_assigned_IoU=.69)})
        block = dict(image_id=5, start=7, native_ranks=[2, 99])
        raw = dict(original_hw=[2, 3], raw_masks=[None]*100)
        result = image_pair_overlaps(cocos, [row], raw, block, [1, 3], mask_api)
        self.assertEqual(result[0]["assigned_query_id"], 3)
        self.assertEqual(result[0]["by_GT"], {
            INTEGER: dict(raw=.99, retained=.91, assigned=.80),
            FLOAT: dict(raw=.93, retained=.88, assigned=.69)})
        changed = deepcopy(row)
        changed["by_GT"][FLOAT]["fixed_assigned_IoU"] = .70
        with self.assertRaisesRegex(ValueError, "Exact accepted per-GT overlap"):
            image_pair_overlaps(cocos, [changed], raw, block, [1, 3], mask_api)
