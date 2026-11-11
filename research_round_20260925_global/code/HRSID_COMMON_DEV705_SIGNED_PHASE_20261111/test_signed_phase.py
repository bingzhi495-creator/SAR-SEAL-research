import unittest

import numpy as np
from signed_phase import phase_indices, stratified_alignment


class SignedPhaseTest(unittest.TestCase):
    def test_signed_null_keeps_both_polarities_and_within_stratum_counts(self):
        aligned = stratified_alignment(
            np.array([10, 0, -10, -9, 0]),
            np.array([1, 0, -1, 1, 0]),
            np.array([0, 0, 0, 1, 1]),
        )
        self.assertEqual(aligned["signed_observed"], 11)
        self.assertEqual(aligned["signed_expected"], -4.5)
        self.assertEqual(aligned["exposed"], 3)
        self.assertEqual(aligned["correct_observed"], 1)
        self.assertEqual(aligned["correct_expected"], -4.5)
        self.assertEqual(aligned["harmful_observed"], -10)


    def test_direct_label_paths_share_inverse_grid_but_keep_phase(self):
        legacy, center, inv = phase_indices()
        self.assertEqual((legacy[575], center[575], inv[799]), (798, 799, 575))
        self.assertEqual((legacy[inv[2]], center[inv[2]]), (1, 2))
        self.assertEqual(int(np.count_nonzero(legacy != center)), 384)


if __name__ == "__main__":
    unittest.main()
