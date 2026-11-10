"""Selection and boundary-bank witnesses for the fixed FIT probe."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fit_point_probe import boundary_bank, selected_pairs


class FitPointProbeTest(unittest.TestCase):
    def test_first_core_source_with_both_paths_and_first_view_per_path(self):
        rows = [dict(ordinal=o, layer=l, path=p) for o, l, p in (
            (0, "core", "direct"), (0, "core", "direct"),
            (1, "stress", "crop"), (1, "stress", "direct"),
            (2, "core", "crop"), (2, "core", "crop"), (2, "core", "direct"),
            (3, "core", "direct"), (3, "core", "crop"),
        )]
        self.assertEqual(selected_pairs(rows, 2), [(2, 4, 6), (3, 8, 7)])

    def test_boundary_bank_has_four_disjoint_hard_label_shells(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 2:7] = True
        xy, strata = boundary_bank(mask)
        bank = {tuple(point): int(group) for point, group in zip(xy, strata, strict=True)}
        self.assertEqual(len(bank), len(xy))
        self.assertEqual((bank[(2, 2)], bank[(3, 3)], bank[(1, 1)], bank[(0, 0)]), (0, 1, 2, 3))
        self.assertNotIn((4, 4), bank)


if __name__ == "__main__":
    unittest.main()
