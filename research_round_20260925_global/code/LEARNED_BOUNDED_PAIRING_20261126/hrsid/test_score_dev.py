"""Synthetic independent checks for cap validation and historical zero fallback."""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from pycocotools import mask

sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_dev


class ScoreChecks(unittest.TestCase):
    def test_original_noop(self):
        expected = {"adds": 0, "drops": 0, "area_delta": 0, "flipped": (), "edits": 0}
        ops = SimpleNamespace(score=SimpleNamespace(get_edit=lambda *unused: expected))
        self.assertEqual(score_dev.original_edit(ops, {}, 5217, 7, "SF"), ([], 0))

    def test_cap_and_true_mask_sign(self):
        coords = np.array([[12, 11], [24, 23]])
        valid = {"pairs": [[0, 0], [1, 1]], "max_fixed_distance_squared": 1}
        self.assertEqual(score_dev.bounded_flip(coords, valid, [12, 11, 24, 23], 2, 10),
                         [12, 11, 24, 23])
        with self.assertRaisesRegex(ValueError, "selected edge"):
            score_dev.bounded_flip(coords, dict(valid, pairs=[[0, 1], [1, 0]]),
                                   [12, 11, 24, 23], 2, 10)
        old = np.zeros((10,10),dtype=np.uint8,order="F")
        for index in (11,23):
            old[index//10,index%10] = 1
        module = SimpleNamespace(decode_runs=lambda unused: [11,1,9,1,10,1,67])
        # Verify the offset conversion against an independently encoded asymmetric case.
        runs = np.flatnonzero(np.diff(np.r_[0,old.reshape(-1,order="F"),1]))
        module.decode_runs = lambda unused: np.diff(np.r_[0,runs]).tolist()
        np.testing.assert_array_equal(score_dev.orientations(coords, mask.encode(old), module),
                                      [[0,1],[0,1]])


if __name__ == "__main__":
    unittest.main()
