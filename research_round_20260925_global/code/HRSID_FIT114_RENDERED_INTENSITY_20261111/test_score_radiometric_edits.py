from collections import Counter
import unittest

import numpy as np

from score_radiometric_edits import (MODELS, Candidate, fit_models, fixed_quota, five_folds, paired_values,
                                     permuted_radiometry, rank_points)


class RadiometricEditTest(unittest.TestCase):
    def test_group_balancing_and_reproducibility(self):
        images = {i: f"P{i % 109}" for i in range(112)}
        folds = five_folds(images)
        self.assertEqual(folds, five_folds(dict(reversed(list(images.items())))))
        self.assertEqual(sorted(Counter(folds.values()).values()), [21, 22, 22, 22, 22])

    def test_fixed_budget_matches_both_sides_and_ties_by_original_pixel(self):
        inside = np.array([False, False, True, True, True])
        pixels = np.array([7, 2, 3, 1, 9])
        self.assertEqual(fixed_quota(9, inside), {False: 0, True: 0})
        self.assertEqual(fixed_quota(10, inside), {False: 0, True: 1})
        self.assertEqual(fixed_quota(10, inside[:4]), {False: 1, True: 0})
        quota = fixed_quota(20, inside)
        self.assertEqual(rank_points(np.ones(5), inside, pixels, quota).tolist(), [1, 3])
        a = rank_points(np.arange(5), inside, pixels, quota)
        self.assertEqual(np.bincount(inside[a].astype(int)).tolist(), [1, 1])

    def test_side_permutation_preserves_joint_feature_rows(self):
        inside = np.array([False, False, True, True, True])
        radio = np.arange(20).reshape(5, 4)
        shuffled = permuted_radiometry(radio, inside, "fixture")
        self.assertTrue(np.array_equal(shuffled, permuted_radiometry(radio, inside, "fixture")))
        for side in (False, True):
            self.assertEqual(set(map(tuple, radio[inside == side])),
                             set(map(tuple, shuffled[inside == side])))

    def test_feature_merge_rejects_a_changed_base_value(self):
        point = dict(x=1, y=2, inside=True, logits=[1.]*9)
        base = dict(candidates=[dict(native_rank=1, points=[point])])
        feature = dict(candidates=[dict(native_rank=1, points=[dict(point, radiometric=[1., 0., 0., .4])])])
        values, = paired_values(base, feature)
        self.assertEqual(values.tolist(), [[1., 0., 0., .4]])
        feature = dict(candidates=[dict(native_rank=1, points=[dict(point, x=3, radiometric=[1., 0., 0., .4])])])
        with self.assertRaisesRegex(ValueError, "base stream"):
            paired_values(base, feature)

    def test_models_learn_each_fitting_label_with_common_points_and_group_count(self):
        inside = np.array([False, False, True, True])
        target = np.array([True, False, False, True])
        x = np.array([2., -2., 2., -2.])[:, None]
        data = {arm: np.column_stack((x, np.zeros((4, width-1))))
                for arm, width in zip(MODELS, (10, 14, 14), strict=True)}
        matched = Candidate("row", 100, inside, np.arange(4), data, target, 81, 119)
        unmatched = Candidate("unmatched", 100, inside, np.arange(4), data, None, 0, 0)
        models, counts = fit_models([[matched, unmatched], [matched]])
        self.assertEqual(counts, dict(fitting_groups=2, fitting_rows=2, fitting_points=8))
        for name, (center, scale, parameters) in models.items():
            z = (data[name]-center)/scale @ parameters[:-1] + parameters[-1]
            self.assertEqual((z > 0).tolist(), (target != inside).tolist())


if __name__ == "__main__":
    unittest.main()
