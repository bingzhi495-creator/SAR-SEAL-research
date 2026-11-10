"""Check the shared original-grid geometry on a mixed-size COCO fixture."""
import os
from pathlib import Path
import unittest
import numpy as np
from pycocotools import mask

from ssdd_fixed_match_geometry import evaluate_pair, load_helpers


def rle(height, width, top, left):
    canvas = np.zeros((height, width), dtype=np.uint8, order="F")
    canvas[top:top+4, left:left+4] = 1
    result = mask.encode(canvas)
    return dict(size=result["size"], counts=result["counts"].decode("ascii"))


class FixedMatchTest(unittest.TestCase):
    def test_native_downward_offset_on_common_match(self):
        code = Path(os.environ["P259_CODE"]).resolve(strict=True)
        helpers = load_helpers(code, code / "HRSID_MASK_PHASE_REPEAT_DEV705_20261109" / "dev705_paired_mask_geometry.py")
        truth = rle(11, 13, 3, 2)
        gt = dict(info={}, images=[dict(id=1,height=11,width=13), dict(id=2,height=12,width=14)],
                  annotations=[dict(id=1,image_id=1,category_id=1,area=16,bbox=[2,3,4,4],iscrowd=0,
                                    segmentation=truth)], categories=[dict(id=1,name="ship")])
        def row(seg):
            return dict(image_id=1,category_id=1,score=.8,segmentation=seg)
        result = evaluate_pair(gt,[row(rle(11,13,4,2))],[row(truth)],helpers)
        self.assertEqual(result["prefix_rows_per_arm"],1)
        metrics=result["strata"]["all"]
        self.assertEqual(metrics["both_matched"],1)
        self.assertEqual(metrics["paired"]["native"]["mean_signed_dy_pixels"],1.)
        self.assertEqual(metrics["paired"]["center"]["mean_signed_dy_pixels"],0.)
        self.assertEqual(metrics["directional"]["native"]["up"]["gains_strict90"],1)


if __name__ == "__main__":
    unittest.main()
