import unittest

import numpy as np
from pycocotools import mask
import torch
from rfdetr.models.postprocess import PostProcess

from fit_query_export import aligned_query_rows


class ExportParityTests(unittest.TestCase):
    def test_joint_order_and_rle_agree_for_ties_and_slot_one(self):
        logits = torch.full((1, 164, 2), -3.)
        logits[0, :4, :] = 0
        logits[0, 40, 1] = 3
        boxes = torch.full((1, 164, 4), .4)
        ramp = torch.arange(144, dtype=torch.float32)[None, None, None, :]
        masks = (ramp - torch.arange(164, dtype=torch.float32)[None, :, None, None]).expand(1, 164, 144, 144)
        with torch.inference_mode():
            result = aligned_query_rows(dict(pred_logits=logits, pred_boxes=boxes, pred_masks=masks),
                                        (16, 16), torch=torch, np=np, codec=mask, processor=PostProcess)
        self.assertEqual(result["native_ranks"][:4], [2, 4, 6, 8])
        self.assertEqual(result["native_ids"][:4], [0, 1, 2, 3])
        self.assertEqual(len(result["all_rles"]), 164)
        self.assertEqual(mask.area(result["all_rles"][-1]), 0)
        self.assertGreater(mask.area(result["all_rles"][0]), 0)


if __name__ == "__main__":
    unittest.main()
