import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from large_core_fn import best_shift, split_errors


class CoreTests(unittest.TestCase):
    def test_interior_and_band_partition(self):
        truth=np.ones((7,7),dtype=bool)
        pred=truth.copy()
        pred[0,0]=False
        pred[3,3]=False
        read=split_errors(truth,pred)
        self.assertEqual((read["fn"],read["core_fn"],read["band_fn"]),(2,1,1))
        self.assertEqual((read["core_area"],read["band_area"]),(9,40))

    def test_alignment_preserves_all_prediction_pixels(self):
        truth=np.zeros((8,8),dtype=bool)
        truth[0:2,0:2]=True
        pred=np.zeros_like(truth)
        pred[1:3,1:3]=True
        shift,out=best_shift(truth,pred)
        self.assertEqual(shift,(-1,-1))
        self.assertEqual(int(out.sum()),int(pred.sum()))
        self.assertTrue(np.array_equal(truth,out))


if __name__ == "__main__":
    unittest.main()
