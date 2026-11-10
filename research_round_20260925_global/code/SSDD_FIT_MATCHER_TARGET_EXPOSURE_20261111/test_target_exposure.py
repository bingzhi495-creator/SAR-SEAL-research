import unittest

import numpy as np

from target_exposure import exposure


class ExposureTest(unittest.TestCase):
    def test_signed_target_changes_and_zero_hit_survival(self):
        a = np.array([[0, 1, 0], [1, 0, 0]], dtype=bool)
        b = np.array([[0, 0, 0], [1, 0, 1]], dtype=bool)
        c = np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
        got = exposure(a, b, c)
        self.assertEqual(got["compared_target_points"], 6)
        self.assertEqual(got["b_to_c"]["added"], 1)
        self.assertEqual(got["b_to_c"]["removed"], 2)
        self.assertEqual(got["b_to_c"]["changed_rows"], 2)
        self.assertEqual(got["zero_hit_rows"], {"a": 0, "b": 1, "c": 1})
        self.assertEqual(got["b_to_c"]["gained_hit_rows"], 1)
        self.assertEqual(got["b_to_c"]["lost_hit_rows"], 1)

    def test_shape_mismatch_is_not_broadcast(self):
        one = np.zeros((1, 3), dtype=bool)
        with self.assertRaises(ValueError):
            exposure(one, np.zeros((2, 3), dtype=bool), one)

    def test_direct_equivalence_detects_no_b_to_c_change(self):
        a = np.array([[1, 0]], dtype=bool)
        b = np.array([[0, 1]], dtype=bool)
        got = exposure(a, b, b.copy())
        self.assertEqual(got["a_to_b"]["added"] + got["a_to_b"]["removed"], 2)
        self.assertEqual(got["b_to_c"]["added"] + got["b_to_c"]["removed"], 0)


if __name__ == "__main__":
    unittest.main()
