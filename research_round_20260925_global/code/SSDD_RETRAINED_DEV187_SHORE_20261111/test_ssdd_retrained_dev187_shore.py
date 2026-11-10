import importlib.util
import os
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("p302", Path(__file__).with_name("ssdd_retrained_dev187_shore.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PartitionTest(unittest.TestCase):
    def test_full_and_equal_prefix_splits_preserve_original_logical_order(self):
        gt = dict(images=[dict(id=3), dict(id=2), dict(id=1)],
                  annotations=[dict(image_id=3), dict(image_id=2), dict(image_id=1), dict(image_id=3)])
        rows = dict(native=[dict(image_id=i) for i in (3, 3, 2, 1)], center=[dict(image_id=i) for i in (3, 2, 1)])
        grids = {3: ("all_water", None), 2: ("land_containing", None), 1: ("all_water", None)}
        result = worker.split_population(gt, rows, grids)
        water_gt, water_rows = result["all_water"]
        land_gt, land_rows = result["land_containing"]
        self.assertEqual([i["id"] for i in water_gt["images"]], [3, 1])
        self.assertEqual([i["image_id"] for i in water_gt["annotations"]], [3, 1, 3])
        self.assertEqual([r["image_id"] for r in water_rows["native"]], [3, 3, 1])
        self.assertEqual(land_gt["images"], [dict(id=2)])
        self.assertEqual(land_rows["center"], [dict(image_id=2)])
        grids[3] = ("unknown", None)
        with self.assertRaisesRegex(ValueError, "Exact DEV"):
            worker.split_population(gt, rows, grids)


class ClosedCocoFixtureTest(unittest.TestCase):
    def test_real_coco_pair_coverage_and_ordered_donors_add_back(self):
        import numpy as np
        from pycocotools import mask

        code = Path(os.environ["P302_TEST_CODE"]).resolve(strict=True)
        shore = worker.source_module(code, "_p302_test_shore", worker.SHORE_SOURCE)
        rank = worker.source_module(code, "_p302_test_rank", (
            "SSDD_DEV187_POST_PRIMARY_RANK_20261110/p250_rank_diagnostic.py", shore.RANK_SHA))
        support = rank.load_support(code)
        ids = (1, 2, 3, 4)
        grids = {i: (worker.STRATA[(i - 1) // 2], None) for i in ids}
        images = [dict(id=i, height=16, width=16) for i in ids]
        gt = dict(images=images, annotations=[dict(id=i, image_id=i, category_id=1,
                  segmentation=[[2, 2, 6, 2, 6, 6, 2, 6]], bbox=[2, 2, 4, 4], area=16, iscrowd=0)
                  for i in ids], categories=[dict(id=1)])
        def rle(offset):
            bitmap = np.zeros((16, 16), order="F", dtype=np.uint8)
            bitmap[2:6, 2 + offset:6 + offset] = 1
            encoded = mask.encode(bitmap)
            return dict(size=encoded["size"], counts=encoded["counts"].decode("ascii"))
        rows = {}
        for mode, offsets in dict(native=(0, 0, 0, 5), center=(0, 5, 0, 0)).items():
            rows[mode] = [dict(image_id=i, category_id=1, segmentation=rle(offset), score=(5 - i) / 5)
                          for i, offset in zip(ids, offsets, strict=True)]
        grouped = worker.analyze_subsets(gt, rows, grids, rank, support)
        all_devs = rank.analyze_pair(gt, rows["native"], rows["center"], support)
        worker.add_back(grouped, all_devs)
        self.assertLess(grouped["all_water"]["raw_center_minus_native_points"]["AP75"], 0)
        self.assertGreater(grouped["land_containing"]["raw_center_minus_native_points"]["AP90"], 0)
        self.assertTrue(all(group["equal_nonempty"]["matching_witness"]["native"]["exact_across_score_donors"]
                            for group in grouped.values()))
        grouped["all_water"]["equal_nonempty"]["prefix_counts"]["retained_per_arm"] -= 1
        with self.assertRaisesRegex(ValueError, "P299 equal-prefix census"):
            worker.add_back(grouped, all_devs)


if __name__ == "__main__":
    unittest.main()
