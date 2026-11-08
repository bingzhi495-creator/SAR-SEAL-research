"""Synthetic native selection, GT preservation and canonical COCO fixtures."""
from copy import deepcopy
import unittest

from hrsid_native_dev705 import (
    BASE, COCO_SHA, ENDPOINT_KEY, FIT_INDEX_SHA, INITIAL_SHA, RELEASE, RECIPE_SHA,
    STEPS, check_endpoint, native_rows, pinned_module, score_variants,
)


class CanonicalScoreTest(unittest.TestCase):
    def test_same_predictions_primary_retains_empty_gt(self):
        from pycocotools import mask
        import numpy as np

        polygon = [[0, 0, 2, 0, 2, 2, 0, 2]]
        positive = mask.merge(mask.frPyObjects(polygon, 4, 4))
        rows = [{"image_id": 1, "category_id": 1, "score": 0.9,
                 "segmentation": {"size": [4, 4], "counts": positive["counts"].decode("ascii")}}]
        base = {"image_id": 1, "category_id": 1, "area": 4, "bbox": [0, 0, 2, 2], "iscrowd": 0}
        gt = {
            "images": [{"id": 1, "height": 4, "width": 4}],
            "categories": [{"id": 1, "name": "ship"}],
            "annotations": [
                dict(base, id=1, segmentation=polygon),
                dict(base, id=2, segmentation=[[0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1]]),
            ],
        }
        original_gt = deepcopy(gt)
        original_rows = deepcopy(rows)
        report = score_variants(gt, rows, {2})
        self.assertEqual(gt, original_gt)
        self.assertEqual(rows, original_rows)
        self.assertEqual(report["primary"]["annotations"], 2)
        self.assertEqual(report["sensitivity"]["annotations"], 1)
        self.assertAlmostEqual(report["primary"]["AR100"], 0.5)
        self.assertAlmostEqual(report["sensitivity"]["AP90"], 1.0)
        self.assertLess(report["primary"]["AP90"], report["sensitivity"]["AP90"])



    def test_global_empty_predictions_keep_full_gt_population(self):
        gt = {"images": [{"id": 1, "height": 4, "width": 4}],
              "categories": [{"id": 1, "name": "ship"}],
              "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "area": 4,
                               "bbox": [0, 0, 2, 2], "iscrowd": 0,
                               "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]]}]}
        result = score_variants(gt, [], set())
        self.assertEqual(result["primary"]["annotations"], 1)
        self.assertEqual(result["primary"]["AP"], 0)
        self.assertEqual(result["primary"]["AP90"], 0)


class NativeExportTest(unittest.TestCase):
    def test_vectorized_rle_matches_closed_native_adapter_and_retains_tie_order(self):
        import numpy as np
        from pycocotools import mask
        import torch
        from rfdetr.models.postprocess import PostProcess

        previous = pinned_module(
            BASE / "RFDETR_NATIVE_E1_EXPORT_CANDIDATE_20261107_01/e1_export.py",
            "4e1fe8e825f31152380ee4191adb92987a828669eb14189058622ac2d22c4039",
        )
        canonical = pinned_module(
            RELEASE / "code/RFDETR_CANONICAL_EXPORT_DRAFT_20261107_01/canonical_export.py",
            "b2e99e581acb0236f6cee5542321bb3e01b2c620f2c727062b28b71eedfd1820",
        )
        previous.QUERIES = 164
        logits = torch.full((1, 164, 2), -20.)
        logits[0, 0, :] = logits[0, 1, 0] = 10.
        logits[0, 1, 1] = 9.
        masks = torch.full((1, 164, 2, 2), -1.)
        masks[0, 0] = 1.
        masks[0, 1, 0, 0] = 1.
        raw = dict(pred_logits=logits, pred_boxes=torch.zeros((1, 164, 4)), pred_masks=masks)
        ref = previous.export_e1_image(
            raw, image_id=7, original_hw=(4, 4), torch_api=torch, postprocess_type=PostProcess,
            canonical_export=canonical.export_coco_mask_predictions,
            encode_mask=lambda bitmap: mask.encode(np.asfortranarray(bitmap, dtype=np.uint8)),
        )
        result = native_rows(raw, image_id=7, hw=(4, 4), torch=torch,
                             postprocess=PostProcess(num_select=100), np=np, codec=mask)
        self.assertEqual(result["coco"], ref["coco"])
        self.assertEqual(result["native_ranks"][:2], [1, 3])
        self.assertEqual(result["unmapped_slot1_count"], ref["unmapped_slot1_count"])
        self.assertEqual(result["raw_query_count"], 164)
        self.assertEqual(result["native_count"], 100)
        self.assertTrue(all("bbox" not in row for row in result["coco"]))

class EndpointTest(unittest.TestCase):
    def test_requires_exact_predeclared_fit_only_coco_lineage(self):
        protocol = "a" * 40
        marker = dict(protocol=protocol, epoch=24, actual_updates=STEPS,
                      native_index_sha256=FIT_INDEX_SHA, coco_sha256=COCO_SHA,
                      original_recipe_sha256=RECIPE_SHA, initial_expanded_sha256=INITIAL_SHA,
                      FIT_only=True, exact_restart_trajectory_verified=False, rng_state={})
        checkpoint = {"pytorch-lightning_version": "2.6.6", "epoch": 23,
                      "global_step": STEPS, ENDPOINT_KEY: marker}
        check_endpoint(checkpoint, protocol)
        marker["native_index_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "endpoint lineage"):
            check_endpoint(checkpoint, protocol)

if __name__ == "__main__":
    unittest.main()


