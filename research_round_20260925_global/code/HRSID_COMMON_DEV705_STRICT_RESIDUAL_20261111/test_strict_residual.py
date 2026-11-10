import unittest

from strict_residual import exact_edit_budget


class PixelBudget(unittest.TestCase):
    def test_matches_exhaustive_add_delete_enumeration(self):
        for gt in range(1, 13):
            for pred in range(1, 13):
                for intersection in range(min(gt, pred) + 1):
                    fp, fn = pred - intersection, gt - intersection
                    union = intersection + fp + fn
                    passing = [
                        (a, d) for a in range(fn + 1) for d in range(fp + 1)
                        if 10 * (intersection + a) >= 9 * (union - d)
                    ]
                    budget = exact_edit_budget(pred, gt, intersection)
                    self.assertEqual(budget["min_edits"], min(a + d for a, d in passing))
                    self.assertEqual(budget["add_only_edits"], min((a for a, d in passing if d == 0), default=None))
                    self.assertEqual(budget["delete_only_edits"], min((d for a, d in passing if a == 0), default=None))
                    self.assertEqual(budget["area_ratio_can_reach_090"], 10 * min(gt, pred) >= 9 * max(gt, pred))


class CoverageCensus(unittest.TestCase):
    def setUp(self):
        import os
        from pathlib import Path
        import strict_residual as worker
        self.worker = worker
        root = Path(os.environ["STRICT_RESIDUAL_CODE"]).resolve()
        closed = worker.load_closed(root)
        self.coverage = closed.load_source(root / worker.COVERAGE, worker.COVERAGE_SHA, "_strict_residual_fixture")

    def test_four_loss_stages_and_success_do_not_overlap(self):
        import numpy as np
        overlaps = np.array([
            [.5, .7, .95, .91, .0, .0],
            [.5, .82, .93, .0, .0, .0],
            [.5, .7, .0, .0, .94, .94],
        ])
        self.assertEqual(self.worker.strict_stages(overlaps, self.coverage.graph_counts_fn()), (1, 1, 1, 1, 2))

    def test_same_target_near_strict_and_switch_cohorts_on_original_rles(self):
        import numpy as np
        from pycocotools import mask

        def rle(row, start, stop):
            pixels = np.zeros((4, 128), dtype=np.uint8, order="F")
            pixels[row, start:stop] = 1
            result = mask.encode(pixels)
            result["counts"] = result["counts"].decode("ascii")
            return result

        def candidates(spans):
            return [
                dict(image_id=1, category_id=1, score=1 - i / 10, segmentation=rle(row, start, end))
                for i, (row, start, end) in enumerate(spans)
            ]

        gt = dict(
            images=[dict(id=1, height=4, width=128)], categories=[dict(id=1, name="ship")],
            annotations=[dict(id=i + 1, image_id=1, category_id=1, area=100, iscrowd=0, segmentation=rle(i, 0, 100))
                         for i in range(4)],
        )
        native = candidates([(0, 0, 0), (0, 0, 85), (1, 0, 95), (2, 0, 85), (3, 110, 120)])
        center = candidates([(0, 0, 90), (1, 0, 70), (2, 0, 87), (3, 0, 100)])
        result = self.worker.census(gt, native, center, self.coverage)
        self.assertEqual(result["equal_nonempty_prefix_rows_per_arm"], 4)
        self.assertEqual(result["strict_coverage"]["all"]["arms"]["native"]["native_order_greedy_090_matches"], 1)
        self.assertEqual(result["strict_coverage"]["all"]["arms"]["center"]["native_order_greedy_090_matches"], 2)
        common = result["common_near_strict"]["all"]
        self.assertEqual((common["same_gt_pairs"], common["center_fewer_min_edits"]), (1, 1))
        self.assertEqual(common["arms"]["native"]["min_edits_mean"], 5)
        self.assertEqual(common["arms"]["center"]["min_edits_mean"], 3)
        switches = result["directional_090_switches"]["all"]
        self.assertEqual(switches["center_only"]["targets"], 2)
        self.assertEqual(switches["center_only"]["failing_best_iou_bands"]["075_to_090"]["min_edits_mean"], 5)
        self.assertEqual(switches["center_only"]["failing_best_iou_bands"]["below_050"]["min_edits_mean"], 174)
        self.assertEqual(switches["native_only"]["failing_best_iou_bands"]["050_to_075"]["min_edits_mean"], 20)
