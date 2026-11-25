import unittest

import numpy as np

from pathlib import Path
from runpy import run_path

functions = run_path(str(Path(__file__).with_name("complement.py")))
match, rescue = functions["match"], functions["rescue"]


class DonorMatchTest(unittest.TestCase):
    def test_capacity_and_conflicts_are_separate(self):
        iou = np.array([[.6, .8, .7, 0], [0, .7, 0, 0], [.59, 0, .4, 0]])
        self.assertEqual(match(iou, .5), {0, 1, 2})
        result = rescue(iou, [True, False, False, False], .5)
        self.assertEqual(result, (2, 1, {1, 2}, {1}))


if __name__ == "__main__":
    unittest.main()

