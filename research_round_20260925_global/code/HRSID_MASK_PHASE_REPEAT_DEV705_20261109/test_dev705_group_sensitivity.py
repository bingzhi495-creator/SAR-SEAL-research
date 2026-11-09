import unittest
from unittest.mock import patch

import dev705_group_sensitivity as subject


class CachedArm:
    def __init__(self, native, delta):
        self.groups = ("P1", "P2")
        self.native = native
        self.delta = delta
        self.calls = []

    def scores(self, weights):
        self.calls.append(tuple(weights))
        factor = float(weights[1] / sum(weights))
        return dict(AP=self.native + self.delta * factor, AP75=self.native + .1 + self.delta * factor)


class GroupSensitivityTests(unittest.TestCase):
    def test_same_bootstrap_and_omission_draws_drive_all_four_arms(self):
        seeds = {5201: (CachedArm(.5, 0), CachedArm(.5, .2)),
                 5202: (CachedArm(.6, 0), CachedArm(.6, .1))}
        with patch.object(subject, "BOOTSTRAP_DRAWS", 12):
            result = subject.summarize_caches(seeds)
        calls = seeds[5201][0].calls
        self.assertEqual(len(calls), 15)
        for arms in seeds.values():
            for arm in arms:
                self.assertEqual(arm.calls, calls)
        self.assertEqual(calls[0], (1, 1))
        self.assertTrue(all(sum(row) == 2 for row in calls[1:13]))
        self.assertEqual(calls[-2:], [(0, 1), (1, 0)])
        ap = result["seeds"]["5201"]["AP"]
        self.assertAlmostEqual(ap["fixed_gain_points"], 10.)
        self.assertAlmostEqual(ap["leave_one_out"]["min_gain_points"], 0.)
        self.assertAlmostEqual(ap["leave_one_out"]["max_gain_points"], 20.)
        self.assertEqual(ap["leave_one_out"]["positive_omissions"], 1)
        self.assertEqual(ap["bootstrap"]["valid_draws"], 12)

    def test_refuses_mismatched_group_order_before_scoring(self):
        seeds = {5201: (CachedArm(.5, 0), CachedArm(.5, .2)),
                 5202: (CachedArm(.6, 0), CachedArm(.6, .1))}
        seeds[5202][1].groups = ("P2", "P1")
        with self.assertRaisesRegex(ValueError, "group ordering"):
            subject.summarize_caches(seeds)
        self.assertTrue(all(not arm.calls for arms in seeds.values() for arm in arms))

