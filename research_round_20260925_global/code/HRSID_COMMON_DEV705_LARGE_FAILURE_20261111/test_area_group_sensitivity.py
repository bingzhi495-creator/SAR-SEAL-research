"""Synthetic checks for original COCO area-aware paired composition."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import unittest

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from area_group_sensitivity import analyze_group_composition


def region(x, y, side):
    image = np.zeros((192, 192), dtype=np.uint8, order="F")
    image[y:y + side, x:x + side] = 1
    rle = mask.encode(image)
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def oracle(gt, predictions, image_ids):
    dataset = COCO()
    dataset.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        dataset.createIndex()
        scorer = COCOeval(dataset, dataset.loadRes(deepcopy(predictions)), "segm")
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
    return {name: float(scorer.stats[index]) for name, index in (("AP", 0), ("AP75", 2), ("APl", 5))}


class AreaGroupSensitivityTests(unittest.TestCase):
    def test_leave_one_group_uses_official_area_and_treats_no_large_gt_as_undefined(self):
        large = region(0, 0, 96)
        small = region(150, 150, 8)
        other_large = region(96, 0, 96)
        gt = dict(
            info={},
            images=[dict(id=i, width=192, height=192) for i in (2, 4, 7)],
            categories=[dict(id=1, name="ship")],
            annotations=[
                dict(id=1, image_id=4, category_id=1, segmentation=large, bbox=[0, 0, 96, 96],
                     area=96 ** 2, iscrowd=0),
                dict(id=2, image_id=2, category_id=1, segmentation=small, bbox=[150, 150, 8, 8],
                     area=8 ** 2, iscrowd=0),
                dict(id=3, image_id=7, category_id=1, segmentation=large, bbox=[0, 0, 96, 96],
                     area=96 ** 2, iscrowd=1),
            ],
        )
        def detection(image_id, segmentation):
            return dict(image_id=image_id, category_id=1, score=.9, segmentation=segmentation)

        native = [detection(4, large), detection(2, small), detection(7, large)]
        center = [detection(4, large), detection(2, other_large), detection(2, small), detection(7, large)]
        expected = {name: oracle(gt, rows, [2, 4, 7])
                    for name, rows in (("native", native), ("center", center))}
        original_gt = deepcopy(gt)
        actual = analyze_group_composition(gt, native, center, {2: "A", 4: "B", 7: "C"}, expected)
        self.assertEqual(gt, original_gt)
        self.assertEqual(actual["eligible_gt"], {"all": 2, "large": 1})
        self.assertEqual(actual["groups_with_eligible_gt"], {"all": 2, "large": 1})
        metric = actual["metrics"]["APl"]
        self.assertAlmostEqual(metric["fixed"]["native"], expected["native"]["APl"], places=12)
        self.assertAlmostEqual(metric["fixed"]["center"], expected["center"]["APl"], places=12)
        self.assertLess(metric["fixed"]["delta_points"], 0.)
        self.assertEqual(metric["leave_one_group_out"]["no_gt_draws"], 1)
        self.assertEqual(metric["leave_one_group_out"]["rows"][0]["delta_points"], 0.)
        self.assertGreater(metric["bootstrap"]["no_gt_draws"], 0)
        for row, remaining_ids in zip(metric["leave_one_group_out"]["rows"], ([4, 7], [2, 7], [2, 4])):
            direct = [oracle(gt, arm, remaining_ids)["APl"] for arm in (native, center)]
            if direct[0] == -1.:
                self.assertIsNone(row["delta_points"])
            else:
                self.assertAlmostEqual(row["delta_points"], 100. * (direct[1] - direct[0]), places=12)


if __name__ == "__main__":
    unittest.main()
