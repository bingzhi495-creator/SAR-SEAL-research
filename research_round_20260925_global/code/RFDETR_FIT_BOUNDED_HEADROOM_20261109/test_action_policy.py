import unittest
import numpy as np
from action_policy import Readout


class FixedCandidateActions(unittest.TestCase):
    def test_changes_only_allowed_matched_candidate(self):
        native=np.array([[[1, 0]], [[0, 1]], [[1, 0]]],dtype=bool)
        truth=np.array([[[1, 1]], [[0, 1]]],dtype=bool)
        readout=Readout()
        readout.start(dict(native=native),truth,[(0, 0), (1, 1)],["native-rles"])
        for i,strict in enumerate((False, True)):
            readout.replace(i,{"strict":strict},truth[i],((True, {}),(False, {})),
                            np.array([[False, False]]),np.array([[False, True]]))
        np.testing.assert_array_equal(readout.masks["native"],native)
        np.testing.assert_array_equal(readout.masks["offset"],native)
        np.testing.assert_array_equal(readout.masks["coeff"],
                                      np.array([[[0, 0]], [[0, 1]], [[1, 0]]],dtype=bool))
        np.testing.assert_array_equal(readout.masks["matched_gt"],
                                      np.array([[[1, 1]], [[0, 1]], [[1, 0]]],dtype=bool))
        np.testing.assert_array_equal(native,np.array([[[1, 0]], [[0, 1]], [[1, 0]]],dtype=bool))


if __name__=="__main__":
    unittest.main()

