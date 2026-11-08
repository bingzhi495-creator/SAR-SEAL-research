import unittest

import numpy as np

from measure_geometry import measure


class CoverageTest(unittest.TestCase):
    def test_exact_threshold_and_one_to_one_capacity(self):
        inter = np.array([[9, 0], [9, 0]], dtype=np.int64)
        union = np.full(inter.shape, 10, dtype=np.int64)
        self.assertEqual(measure(inter, union, 9, 10), ([True, True], 1))


if __name__ == "__main__":
    unittest.main()
