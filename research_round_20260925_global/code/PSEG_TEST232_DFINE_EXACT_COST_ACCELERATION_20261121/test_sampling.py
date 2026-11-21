"""Numerical limits, ties and non-square native dimensions using generated probability planes."""
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


def module(path, name, digest=None):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec)
    if digest and sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Source identity")
    spec.loader.exec_module(m)
    return m


class SamplesTest(unittest.TestCase):
    def test_numeric_edges_and_repeated_queries(self):
        code = Path(os.environ["DFINE_CLOSED_INPUT_ROOT"])/"github_release_20261011/research_round_20260925_global/code"
        p = code/"DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120/native_points.py"
        src = module(p,"native","475ecbdd932a2129f160e48bd1d838653b971df6cd69b9edda103c037ee3fcdc")
        fast = module(Path(__file__).with_name("sparse_sampling.py"),"fast")
        e24 = code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
        pair,rank = src.load_old(e24/"fit109_paired_point/pair_math.py",e24/"fit109_probability_rank/rank_math.py")
        rng = np.random.default_rng(5217)
        planes = [rng.random((160,160),dtype=np.float32),
                  np.resize(np.array([0.,-0.,1.,.5,.5+2**-12,2**-20,2**-149],np.float32),(160,160))]
        for h,w in ((1,1),(1,401),(317,1),(27,15),(300,668),(494,294),(160,160),(800,800)):
            _,original = src.make_native(pair,rank,h,w)
            optimized = fast.native_sampler(h,w)
            points = np.r_[np.arange(h*w, dtype=np.int32)[::max(1,(h*w)//479)],0,h*w-1,h*w-1]
            for plane in planes:
                for q in (points,np.asarray([],dtype=np.int32)):
                    expected = original.sample_probability(plane,q,w)
                    actual = optimized(plane,q,w)
                    np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))


if __name__ == "__main__":
    unittest.main()
