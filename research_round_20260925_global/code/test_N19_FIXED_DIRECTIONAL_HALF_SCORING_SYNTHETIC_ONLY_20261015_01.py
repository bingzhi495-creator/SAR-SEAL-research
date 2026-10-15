# UNBOUND_DRAFT: hand probabilities only; no actual task inputs.
import importlib.util
from pathlib import Path
import unittest
import numpy as np

p = Path(__file__).with_name('N19_FIXED_DIRECTIONAL_HALF_SCORING_UNBOUND_20261015_01.py')
s = importlib.util.spec_from_file_location('n19_half_synthetic', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class HandFixture(unittest.TestCase):
    def test_fixed_four_readouts(self):
        p = np.array([[.5,.25,.25], [.5,.25,.25], [.125,.5,.375],
                      [.25,.25,.5], [.5,.25,.25], [.5,.25,.25], [.5,.25,.25]], dtype=np.float64)
        logits = np.log(p)
        L = np.array([-2,2,-1,0,-1,1,-1], dtype=np.float32)
        C = np.arange(10,17,dtype=np.int64)
        r = m.score_and_select(logits,L,C,2)
        expected = {
            'SYMMETRIC': ([.25,.25,-.375,0,.25,.25,.25], [14,15], [4,5]),
            'DIR_HALF': ([.375,0,-.125,.125,.375,0,.375], [14,16], [4,6]),
            'GLOBAL_HALF': ([.375,.375,-.125,.125,.375,.375,.375], [14,15], [4,5]),
            'REVERSED_HALF': ([0,.375,-.4375,-.125,0,.375,0], [11,15], [1,5]),
        }
        self.assertEqual(tuple(r), tuple(expected))
        for policy,(scores,flat,pos) in expected.items():
            self.assertEqual(r[policy]['score'].dtype,np.float64)
            np.testing.assert_allclose(r[policy]['score'],scores,rtol=0,atol=1e-15)
            np.testing.assert_array_equal(r[policy]['selected_flat_sorted'],flat)
            np.testing.assert_array_equal(r[policy]['selected_C_positions'],pos)
        # strict positive only: actual zeros and negatives never pad atmost.
        full = m.score_and_select(logits,L,C,len(C))
        np.testing.assert_array_equal(full['SYMMETRIC']['selected_C_positions'],[0,1,4,5,6])
        np.testing.assert_array_equal(full['DIR_HALF']['selected_C_positions'],[0,3,4,6])
        for values in m.score_and_select(logits,L,C,0).values():
            self.assertEqual(values['selected_flat_sorted'].shape,(0,))
            self.assertEqual(values['selected_C_positions'].dtype,np.int64)
        empty = m.score_and_select(np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty(0,np.int64),0)
        self.assertTrue(all(v['score'].shape==(0,) for v in empty.values()))
        bad = [(logits[:,:2],L,C,2),(logits,L[:,None],C,2),(logits,L,C[::-1],2),
               (logits,L,C.astype(np.float32),2),(logits,L,C,True),(logits,L,C,8),
               (np.full_like(logits,np.nan),L,C,2),(logits,None,C,2),
               (logits,L,np.array([10,11,12,13,14,15,15],np.int64),2)]
        for args in bad:
            with self.assertRaises((TypeError,ValueError)):
                m.score_and_select(*args)


if __name__ == '__main__':
    unittest.main()
