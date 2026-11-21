import itertools
import unittest

import numpy as np

from ratio_writer import expected_ratio_writes


class RatioWriterTests(unittest.TestCase):
    def test_exact_agreement_with_exhaustive_selection(self):
        coords = np.array([[1,11],[2,12],[3,13],[4,14]])
        q = np.array([[.42,.36],[.46,.73],[.2,.4],[.55,.12]])
        for k in (1,2,3,4):
            with self.subTest(k=k):
                choices = expected_ratio_writes(coords,q,k,41,21,33)
                qid = dict(zip(coords.ravel(),q.ravel()))
                side = dict(zip(coords.ravel(),[1,-1]*len(coords)))
                def objective(indices):
                    i = 21+sum(side[idx]*qid[idx] for idx in indices)
                    p = 41+sum(side[idx] for idx in indices)
                    return i/(p+33-i)
                best = max(objective(subset) for subset in itertools.combinations(coords.ravel(),2*k))
                self.assertAlmostEqual(objective(choices["R"]),best,places=12)
                self.assertEqual(len(set(choices["T"])),2*k)

    def test_zero_and_invalid_counts(self):
        c,q=np.empty((0,2),dtype=np.int64),np.empty((0,2))
        self.assertEqual(len(expected_ratio_writes(c,q,0,32,0,0)["R"]),0)
        with self.assertRaises(ValueError):
            expected_ratio_writes(c,q,1,32,1,2)


if __name__ == "__main__":
    unittest.main()
