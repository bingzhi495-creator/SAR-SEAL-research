"""Synthetic parity checks for native heads, sparse metrics and donor rank."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('dev_donor', Path(__file__).with_name('dev_donor.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FrozenVote(unittest.TestCase):
    def test_original_cost_fallback(self):
        q = np.array([[.9,.1],[.1,.9]])
        donor = np.array([[-1.,1.],[1.,-1.]])
        np.testing.assert_array_equal(worker.blended(q, donor, 1., 0.), q)
        np.testing.assert_array_equal(worker.blended(q, donor, 0., 0.), q)

    def test_donor_signed_affine_vote(self):
        q = np.full((2,2), .5, dtype=np.float32)
        cue = np.array([[1., -1.], [0., .5]])
        np.testing.assert_allclose(worker.blended(q, cue, .25,.75), [[.875,.125],[.5,.6875]])


if __name__ == '__main__':
    unittest.main()
