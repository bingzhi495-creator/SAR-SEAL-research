from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


SOURCE = Path(__file__).with_name("radiometric.py")


class RadiometricTests(unittest.TestCase):
    def test_joint_pool_is_distinct_from_side_control_with_exact_ties(self):
        transform = run_path(str(SOURCE))["transform_candidate"]
        f = np.zeros((3, 2, 14), dtype=np.float32)
        f[..., 9] = f[..., 10] = [[.25, .75], [.25, .75], [.25, .75]]
        j, s = [transform(f, arm) for arm in ("joint", "per_side")]
        np.testing.assert_array_equal(j[..., :14], f)
        np.testing.assert_array_equal(s[..., :14], f)
        np.testing.assert_array_equal(j[..., 23], [[-.5, .5]] * 3)
        np.testing.assert_array_equal(s[..., 23:], np.zeros((3, 2, 5)))
        self.assertEqual(j.shape, (3, 2, 28))


    def test_common_positive_affine_transform_preserves_cues(self):
        transform = run_path(str(SOURCE))["transform_candidate"]
        f = np.zeros((4, 2, 14), dtype=np.float32)
        for j in (9, 10, 11):
            f[..., j] = (np.arange(8).reshape(4, 2) % (j-5)) / 32
        changed = f.copy()
        changed[..., 9:12] = 2 * f[..., 9:12] + .125
        for arm in ("joint", "per_side"):
            np.testing.assert_array_equal(transform(f, arm)[..., 23:], transform(changed, arm)[..., 23:])

    def test_range_fallback_and_untied_permutation_do_not_use_matched_partner(self):
        funcs = run_path(str(SOURCE))
        np.testing.assert_array_equal(funcs["robust_z"]([0, 0, 0, 0, 1]), [0, 0, 0, 0, 1])
        f = np.zeros((4, 2, 14), dtype=np.float32)
        f[..., 9] = np.arange(8).reshape(4, 2) / 16
        swapped = f.copy()
        swapped[:, 1, :] = f[::-1, 1, :]
        for arm in ("joint", "per_side"):
            original = funcs["transform_candidate"](f, arm)
            updated = funcs["transform_candidate"](swapped, arm)
            np.testing.assert_array_equal(original[:, 0, :], updated[:, 0, :])
            np.testing.assert_array_equal(original[::-1, 1, :], updated[:, 1, :])


if __name__ == "__main__":
    unittest.main()
