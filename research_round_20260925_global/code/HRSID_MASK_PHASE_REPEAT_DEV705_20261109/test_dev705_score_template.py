"""Synthetic score-template checks using the exact original CPU COCO adapter."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

import dev705_score_template as subject


def rle(y=None, x=None):
    canvas = np.zeros((12, 12), dtype=np.uint8, order="F")
    if y is not None:
        canvas[y:y+2, x:x+2] = 1
    value = mask.encode(canvas)
    return dict(size=value["size"], counts=value["counts"].decode("ascii"))


def row(iid, score, segmentation):
    return dict(image_id=iid, category_id=1, score=score, segmentation=segmentation)


def fixture():
    images = [dict(id=i, height=12, width=12) for i in (7, 3, 11)]
    gt = dict(info={}, images=images, categories=[dict(id=1, name="ship")],
              annotations=[dict(id=j, image_id=i, category_id=1, area=4, bbox=[1, 1, 2, 2],
                                iscrowd=0, segmentation=rle(1, 1)) for j, i in enumerate((7, 3), 1)])
    true, false, empty = rle(1, 1), rle(8, 8), rle()
    native = [row(7, .99, empty), row(7, .90, true), row(7, .05, false), row(3, .10, false),
              row(11, .70, empty)]
    center = [row(7, .10, false), row(3, .90, true), row(11, .80, false)]
    return gt, native, center


def evaluate(gt, native, center):
    with patch.multiple(subject, DEV_IMAGES=len(gt["images"]), GT_COUNT=len(gt["annotations"]), CANVAS=(12, 12)):
        return subject.evaluate_pair(gt, native, center)


class ScoreTemplateTests(unittest.TestCase):
    def test_fixed_geometry_score_swap_changes_ap_but_not_recall(self):
        gt, native, center = fixture()
        original = deepcopy((gt, native, center))
        result = evaluate(gt, native, center)
        self.assertEqual((gt, native, center), original)
        self.assertEqual(result["images"], 3)
        self.assertEqual(result["retained_prefix_rows"], 2)
        self.assertEqual(result["images_with_prefix_rows"], 2)
        self.assertEqual(result["images_without_prefix_rows"], 1)
        cells = result["cells"]
        for mode in ("native", "center"):
            other = "center" if mode == "native" else "native"
            self.assertAlmostEqual(cells[mode][mode]["AP"], 51 / 101, delta=1e-12)
            self.assertAlmostEqual(cells[mode][other]["AP"], 51 / 202, delta=1e-12)
            self.assertEqual(cells[mode][mode]["recall75"], .5)
            self.assertEqual(cells[mode][other]["recall90"], .5)


    def test_tied_donor_scores_retain_each_geometrys_given_within_image_order(self):
        gt, _, _ = fixture()
        gt["images"] = gt["images"][:1]
        gt["annotations"] = gt["annotations"][:1]
        true, false = rle(1, 1), rle(8, 8)
        native = [row(7, .90, false), row(7, .80, true)]
        center = [row(7, .50, true), row(7, .50, false)]
        result = evaluate(gt, native, center)
        for donor in ("native", "center"):
            self.assertAlmostEqual(result["cells"]["native"][donor]["AP90"], .5, delta=1e-12)
            self.assertAlmostEqual(result["cells"]["center"][donor]["AP90"], 1., delta=1e-12)
            self.assertEqual(result["cells"]["native"][donor]["recall90"], 1.)
            self.assertEqual(result["cells"]["center"][donor]["recall90"], 1.)
        self.assertAlmostEqual(result["contrasts_points"]["geometry_with_center_scores"]["AP90"], 50.)

    def test_rejects_changed_matching_even_when_the_final_recalls_are_unchanged(self):
        gt, native, center = fixture()
        original = subject.evaluate_cell
        calls = 0

        def broken(*args):
            nonlocal calls
            metrics, matching, layout = original(*args)
            calls += 1
            if calls == 2:
                first = next(item for item in matching if item is not None and item["dtMatches"].size)
                first["dtMatches"][0, 0] = -1
            return metrics, matching, layout

        with patch.object(subject, "evaluate_cell", side_effect=broken):
            with self.assertRaisesRegex(ValueError, "matching evidence changed"):
                evaluate(gt, native, center)

    def test_rejects_a_nonfinite_or_reordered_native_input_before_scoring(self):
        gt, native, center = fixture()
        for score in (float("nan"), 1.0):
            bad = deepcopy(native)
            bad[1]["score"] = score
            with self.subTest(score=score), patch.object(subject, "load_official_adapter") as load:
                with self.assertRaisesRegex(ValueError, "nonincreasing native score order"):
                    evaluate(gt, bad, center)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

