"""Synthetic oracle tests against the pinned pycocotools segmentation scorer."""
from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dev705_group_ap import assert_all_ones, prepare_group_ap, score_pair


def rle(y, x, size=2):
    pixels = np.zeros((16, 16), dtype=np.uint8, order="F")
    pixels[y:y+size, x:x+size] = 1
    out = mask.encode(pixels)
    return dict(size=out["size"], counts=out["counts"].decode("ascii"))


def evaluate(image_ids, annotations, predictions):
    dataset = dict(
        info={}, images=[dict(id=i, width=16, height=16) for i in image_ids],
        categories=[dict(id=1, name="ship")],
        annotations=[dict(id=j, image_id=i, category_id=1, area=int(mask.area(seg)),
                          iscrowd=crowd, segmentation=seg, bbox=mask.toBbox(seg).tolist())
                     for j, (i, seg, crowd) in enumerate(annotations, 1)],
    )
    dt = [dict(image_id=i, category_id=1, score=score, segmentation=seg) for i, score, seg in predictions]
    gt = COCO()
    gt.dataset = dataset
    with redirect_stdout(StringIO()):
        gt.createIndex()
        scorer = COCOeval(gt, gt.loadRes(dt), "segm")
        scorer.params.imgIds = list(image_ids)
        scorer.params.catIds = [1]
        scorer.evaluate()
        scorer.accumulate()
    return scorer



def clone_case(images, annotations, predictions, memberships, multiplicities):
    groups = sorted(set(memberships.values()))
    weights = dict(zip(groups, multiplicities, strict=True))
    cloned_ids, cloned_gt, cloned_dt = [], [], []
    for original in sorted(images):
        for _ in range(weights[memberships[original]]):
            iid = len(cloned_ids) + 1
            cloned_ids.append(iid)
            cloned_gt.extend((iid, seg, crowd) for i, seg, crowd in annotations if i == original)
            cloned_dt.extend((iid, score, seg) for i, score, seg in predictions if i == original)
    return evaluate(cloned_ids, cloned_gt, cloned_dt)


def original_scores(scorer):
    rows = scorer.eval["precision"][:, :, 0, 0, 2]
    return {
        key: float(valid.mean()) if (valid := values[values > -1]).size else -1.0
        for key, values in (("AP", rows), ("AP75", rows[5]))
    }


class GroupAPTests(unittest.TestCase):
    def assertScores(self, actual, expected):
        for key in ("AP", "AP75"):
            self.assertAlmostEqual(actual[key], expected[key], delta=1e-12, msg=key)

    def test_same_image_tie_uses_copy_major_order(self):
        good, false = rle(0, 0), rle(10, 10)
        base = evaluate([10], [(10, good, 0)], [(10, .5, good), (10, .5, false)])
        cloned = evaluate([1, 2], [(1, good, 0), (2, good, 0)],
                          [(1, .5, good), (1, .5, false), (2, .5, good), (2, .5, false)])
        cache = prepare_group_ap(base, {10: "P1"})
        self.assertScores(assert_all_ones(cache, original_scores(base)), original_scores(base))
        self.assertScores(cache.scores([2]), original_scores(cloned))
        self.assertAlmostEqual(cache.scores([2])["AP"], .8349834983498348, delta=1e-12)


    def test_cross_image_ties_follow_original_image_ids(self):
        good, false = rle(0, 0), rle(10, 10)
        original = evaluate([20, 10], [(20, good, 0)], [(20, .5, good), (10, .5, false)])
        cache = prepare_group_ap(original, {20: "P2", 10: "P1"})
        self.assertEqual(cache.groups, ("P1", "P2"))
        self.assertScores(cache.scores([1, 1]), {"AP": .5, "AP75": .5})
        self.assertScores(cache.scores([2, 1]), {"AP": 1 / 3, "AP75": 1 / 3})

    def test_clone_oracle_with_tied_mixed_iou_crowd_and_empty_images(self):
        groups = {30: "P2", 10: "P1", 40: "P3", 20: "P1", 50: "P2"}
        a, b, c = rle(0, 0, 3), rle(6, 6, 3), rle(10, 10)
        approximate = rle(1, 0, 3)
        gt = [(10, a, 0), (10, b, 0), (20, b, 1), (30, a, 0), (30, c, 0)]
        dt = [(30, .9, approximate), (30, .9, c), (10, .9, approximate), (10, .9, b),
              (10, .9, c), (20, .9, b), (20, .9, c), (40, .9, a),
              (30, .5, a), (10, .4, a)]
        images = list(groups)
        native = prepare_group_ap(evaluate(images, gt, dt), groups)
        center_dt = [(i, score, a if i == 10 and seg == approximate else seg) for i, score, seg in dt]
        center = prepare_group_ap(evaluate(images, gt, center_dt), groups)
        for weights in ([1, 1, 1], [2, 2, 1], [0, 2, 1], [2, 0, 0], [0, 0, 1]):
            with self.subTest(weights=weights):
                actual = score_pair(native, center, np.asarray(weights, dtype=np.int32))
                a_original = original_scores(clone_case(images, gt, dt, groups, weights))
                b_original = original_scores(clone_case(images, gt, center_dt, groups, weights))
                self.assertScores(actual["native"], a_original)
                self.assertScores(actual["center"], b_original)
                delta = actual["center_minus_native"]
                if a_original["AP"] == -1.:
                    self.assertEqual(delta, dict(AP=None, AP75=None))
                else:
                    self.assertScores(delta, {key: b_original[key] - a_original[key] for key in ("AP", "AP75")})
        self.assertNotEqual(native.scores([2, 0, 0])["AP"], native.scores([2, 0, 0])["AP75"])

    def test_max_dets_is_per_copy_and_valid_gt_without_true_positives_scores_zero(self):
        good, false = rle(0, 0), rle(10, 10)
        groups, ids = {10: "P1"}, [10]
        annotations = [(10, good, 0)]
        predictions = [(10, 1., false)] * 100 + [(10, .5, good)]
        base = evaluate(ids, annotations, predictions)
        cache = prepare_group_ap(base, groups)
        self.assertScores(cache.scores([2]), {"AP": 0., "AP75": 0.})
        self.assertScores(cache.scores([2]), original_scores(clone_case(ids, annotations, predictions, groups, [2])))

    def test_a_cached_image_with_gt_and_no_detections_remains_in_the_denominator(self):
        good = rle(0, 0)
        source = evaluate([10, 20], [(10, good, 0), (20, good, 0)], [(20, .5, good)])
        cache = prepare_group_ap(source, {10: "P1", 20: "P2"})
        self.assertScores(cache.scores([2, 0]), {"AP": 0., "AP75": 0.})
        self.assertScores(cache.scores([0, 2]), {"AP": 1., "AP75": 1.})
        self.assertScores(cache.scores([1, 1]), {"AP": 51 / 101, "AP75": 51 / 101})

    def test_bad_multiplicities_and_pair_membership_or_baseline_are_refused(self):
        good = rle(0, 0)
        original = evaluate([10, 20], [(10, good, 0)], [(10, .5, good)])
        native = prepare_group_ap(original, {10: "P1", 20: "P2"})
        center = prepare_group_ap(original, {10: "P2", 20: "P1"})
        for weights in ([True, True], [1., 1.], [-1, 2], [0, 0], [29, 0], [14, 15], [1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                native.scores(weights)
        with self.assertRaisesRegex(ValueError, "membership"):
            score_pair(native, center, [1, 1])
        with self.assertRaisesRegex(ValueError, "All-ones"):
            assert_all_ones(native, dict(AP=.999, AP75=1.))


if __name__ == "__main__":
    unittest.main()

