from pathlib import Path
from runpy import run_path
import unittest

import numpy as np

M=run_path(str(Path(__file__).with_name("repeat_summary.py")))


class RepeatPoolingTest(unittest.TestCase):
    def test_common_group_draws_preserve_cross_seed_cancellation(self):
        x=np.array([-3.,6.])
        y=np.array([3.,-6.])
        draws=np.array([[0,0],[1,1],[0,1]])
        checked=M["paired_seed_pool"]([x,y],[.4,.8],draws)
        self.assertEqual(checked["iou_points"],0.)
        self.assertEqual(checked["percentile95_points"],[0.,0.])
        self.assertAlmostEqual(checked["ap_points"],.6)

    def test_disagreement_is_not_a_new_group_replicate(self):
        draws=np.array([[0,0],[1,1]])
        with self.assertRaisesRegex(ValueError,"group"):
            M["paired_seed_pool"]([np.array([1.,2.]),np.array([3.])],[.1,.2],draws)


if __name__=="__main__":
    unittest.main()
