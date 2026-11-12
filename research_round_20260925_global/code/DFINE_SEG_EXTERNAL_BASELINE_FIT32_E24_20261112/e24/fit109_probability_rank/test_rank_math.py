import unittest

import numpy as np

from rank_math import choose, editable_pools, sample_probability


class RankMath(unittest.TestCase):
    def test_same_counts_strata_and_author_box(self):
        hard = np.zeros((11, 13), dtype=bool)
        hard[2:9, 3:10] = True
        pools, quotas = editable_pools(hard, (2.2, 1.8, 10.1, 9.1))
        self.assertEqual(sum(quotas[k] for k in ("add1", "add2")), 4)
        self.assertEqual(sum(quotas[k] for k in ("drop1", "drop2")), 4)
        for key,(coords,_) in pools.items():
            yy,xx = np.divmod(coords, hard.shape[1])
            self.assertTrue(np.all((xx >= 3) & (xx < 11) & (yy >= 2) & (yy < 10)))
            self.assertTrue(np.all(hard[yy,xx] == key.startswith("drop")))

    def test_selection_is_posterior_by_sign_and_sham_is_repeatable(self):
        pools = {"add1": (np.array([1,3,5]),np.array([1.,1.,1.])),
                 "drop1": (np.array([2,4,6]),np.array([1.,1.,1.]))}
        scores = {"add1": np.array([.10,.40,.30]),"drop1": np.array([.90,.51,.65])}
        quotas = dict(add1=1,drop1=1)
        first = choose(pools,quotas,scores,"P1",9)
        self.assertEqual(first,choose(pools,quotas,scores,"P1",9))
        self.assertEqual(first["geometry"],[[1],[2]])
        self.assertEqual(first["probability"],[[3],[4]])
        self.assertEqual(len(first["permuted"]),16)
        self.assertTrue(all(len(a)==len(d)==1 for a,d in first["permuted"]))

    def test_no_background_capacity_means_no_edits(self):
        hard = np.ones((8,8),dtype=bool)
        _, quotas = editable_pools(hard,(0.,0.,8.,8.))
        self.assertEqual(sum(quotas.values()),0)

    def test_constant_probability_border_roundtrip(self):
        p = np.full((160,160),.49,dtype=np.float32)
        coords=np.array([0,799,799*800,799*800+799,800*111+123])
        out=sample_probability(p,coords,800)
        self.assertTrue(np.all(out==np.float16(.49)))


if __name__=="__main__":
    unittest.main()