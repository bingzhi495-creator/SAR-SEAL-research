import unittest

import numpy as np

from candidate_rank import midranks, ranked_inputs


class CandidateRankTest(unittest.TestCase):
    def test_column_ranks_with_exact_ties_and_singletons(self):
        x=np.array([[1,2,9],[1,5,9],[4,2,9],[9,7,9]],dtype=np.float32)
        target=np.array([[-.5,-.5,0],[-.5,.25,0],[.25,-.5,0],[.75,.75,0]],dtype=np.float32)
        np.testing.assert_array_equal(midranks(x),target)
        np.testing.assert_array_equal(midranks(x[:1]),np.zeros_like(x[:1]))
        self.assertEqual(midranks(x[:0]).shape,x[:0].shape)

    def test_side_reference_and_exact_original_six(self):
        a=np.arange(152,dtype=np.float32).reshape(4,38)
        side=np.array([False,True,True,False])
        ranks=ranked_inputs(a,side,True)
        np.testing.assert_array_equal(ranks[:,:6],a[:,:6])
        np.testing.assert_array_equal(ranks[:,6:],np.broadcast_to(np.array([-.5,-.5,.5,.5])[:,None],(4,32)))
        self.assertFalse(np.array_equal(ranks,ranked_inputs(a,side,False)))
        p=np.array([3,0,2,1])
        np.testing.assert_array_equal(ranked_inputs(a[p],side[p],True),ranks[p])

    def test_pooled_ranks_ignore_side_label_assignment(self):
        rng=np.random.default_rng(15)
        a=rng.standard_normal((7,38)).astype(np.float32)
        side=np.array([True]*3+[False]*4)
        np.testing.assert_array_equal(ranked_inputs(a,side,False),ranked_inputs(a,~side,False))


if __name__=="__main__":
    unittest.main()

