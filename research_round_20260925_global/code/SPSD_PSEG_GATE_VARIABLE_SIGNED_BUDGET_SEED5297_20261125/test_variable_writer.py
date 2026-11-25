from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


MODULE = Path(__file__).with_name("variable_writer.py")


class VariableWriterTests(unittest.TestCase):
    def test_exact_reference_pixel_counts_and_add_only_choice(self):
        m = run_path(str(MODULE))
        posterior = np.array([[[1,1],[1,1]],[[1,1],[1,1]],[[0,0],[1,1]]],dtype=float)
        static = ((0,),(0,))
        solved = m["select"](posterior,static,1,10)
        self.assertEqual(solved["fixed_iou"]["action"],static)
        self.assertEqual(solved["variable_iou"]["action"],((0,1),()))
        self.assertEqual(solved["near_iou"]["action"],solved["variable_iou"]["action"])
        np.testing.assert_allclose(solved["variable_iou"]["iou_gain"], [1-10/12]*2)
        np.testing.assert_allclose(solved["variable_iou"]["intersection_gain"], [2,2])


    def test_static_first_when_all_estimated_scores_are_zero(self):
        m = run_path(str(MODULE))
        static = ((0,), (1,))
        solved = m["select"](np.zeros((5,2,2)),static,1,5)
        for arm in m["ARMS"]:
            self.assertEqual(solved[arm]["action"],static)
            self.assertEqual(solved[arm]["score"],0.)

    def test_exact_raster_for_unequal_side_write(self):
        from pycocotools import mask

        m = run_path(str(MODULE))
        native = np.zeros((4,4),dtype=bool)
        native[1:3,:] = True
        coords = np.array([[0,4],[1,5],[2,7],[3,9]],dtype=np.int32)
        rle,(adds,drops) = m["write_mask"](native,coords,((0,2,3),(1,)),2,mask)
        observed = mask.decode(rle).astype(bool)
        self.assertEqual((adds,drops),(3,1))
        self.assertEqual(int(observed.sum()),int(native.sum())+2)
        np.testing.assert_array_equal(np.flatnonzero(observed.ravel() & ~native.ravel()),[0,2,3])
        np.testing.assert_array_equal(np.flatnonzero(~observed.ravel() & native.ravel()),[5])
        with self.assertRaisesRegex(ValueError,"quota"):
            m["write_mask"](native,coords,((0,2,3),(1,)),1,mask)

if __name__ == "__main__":
    unittest.main()
