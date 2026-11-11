import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burst_geometry_readback import project_inclusive, containing


class CoordinateTests(unittest.TestCase):
    def test_inclusive_one_based_author_box(self):
        self.assertEqual(project_inclusive(8, 1, 512), (7, 518))
        self.assertEqual(project_inclusive(8008, 383, 399), (8389, 8405))
        self.assertEqual(project_inclusive(23986, 164, 210), (24148, 24194))

    def test_adjacent_burst_boundary_is_exclusive_at_end(self):
        rows = [(0, 1499), (1499, 1499), (2998, 1499)]
        self.assertEqual(containing(rows, (1499, 2997)), [1])
        self.assertEqual(containing(rows, (1498, 1499)), [])


if __name__ == "__main__":
    unittest.main()
