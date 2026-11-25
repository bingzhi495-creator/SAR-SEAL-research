import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stride_geometry import coordinates


class GridGeometryTest(unittest.TestCase):
    def test_same_cell_can_keep_two_distinct_positions(self):
        anchors, exact = coordinates(np.array([1, 2]), (640, 640))
        self.assertEqual(anchors[0], anchors[1])
        self.assertNotEqual(exact[0], exact[1])

    def test_border_clipping_can_collapse_positions(self):
        anchors, exact = coordinates(np.array([0, 1]), (640, 640))
        np.testing.assert_array_equal(anchors, [0, 0])
        np.testing.assert_array_equal(exact, [0, 0])


if __name__ == "__main__":
    unittest.main()
