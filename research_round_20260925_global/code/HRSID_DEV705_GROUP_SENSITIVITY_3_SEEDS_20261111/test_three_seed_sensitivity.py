import unittest
from unittest.mock import patch

from test_dev705_group_sensitivity import CachedArm
import three_seed_sensitivity as subject


class ThreeSeedSensitivityTests(unittest.TestCase):
    def test_same_draws_drive_all_six_arms(self):
        seeds = {5201: (CachedArm(.5, 0), CachedArm(.5, .2)),
                 5202: (CachedArm(.6, 0), CachedArm(.6, .1)),
                 5203: (CachedArm(.4, 0), CachedArm(.4, .3))}
        with patch.object(subject, "BOOTSTRAP_DRAWS", 12):
            result = subject.summarize_caches(seeds)
        calls = seeds[5201][0].calls
        self.assertEqual(len(calls), 15)
        for arms in seeds.values():
            for arm in arms:
                self.assertEqual(arm.calls, calls)
        self.assertAlmostEqual(result["seeds"]["5203"]["AP"]["fixed_gain_points"], 15.)
        self.assertEqual(result["seeds"]["5203"]["AP"]["bootstrap"]["valid_draws"], 12)


if __name__ == "__main__":
    unittest.main()

