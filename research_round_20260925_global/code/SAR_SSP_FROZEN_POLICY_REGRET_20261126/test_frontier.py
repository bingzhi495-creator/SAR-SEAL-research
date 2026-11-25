"""Verify the audit frontiers against exhaustive feasible subsets."""
from itertools import combinations
import importlib.util
import os
import unittest

import numpy as np

from frontier import audit

spec = importlib.util.spec_from_file_location("closed_signed_writer", os.environ["SAR_SEAL_VARIABLE_WRITER"])
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

class FrontierTest(unittest.TestCase):
    def test_full_support_bound_matches_brute_force(self):
        p = np.array([[[.8,.6],[.2,.4]],[[.7,.9],[.1,.2]],[[.3,.4],[.9,.7]]])
        truth_add = np.array([0,1,1])
        truth_drop = np.array([1,0,0])
        static = ((0,), (1,))
        observed = {key: value["action"] for key,value in policy.select(p,static,1,6).items()
                    if key in ("fixed_iou","variable_iou")}
        result = audit(policy,p,static,1,6,7,4,truth_add,truth_drop,observed)
        values = []
        for adds in range(3):
            for ia in combinations(range(3),adds):
                for idr in combinations(range(3),2-adds):
                    hit = 4 + sum(truth_add[list(ia)]) - sum(truth_drop[list(idr)])
                    values.append(hit/(6+adds-(2-adds)+7-hit))
        self.assertAlmostEqual(result["full_support"],max(values))
        self.assertGreaterEqual(result["full_support"],result["full_menu"])
        self.assertGreaterEqual(result["full_menu"],result["rank_only"])
        self.assertGreaterEqual(result["rank_only"],result["variable"])
        self.assertGreaterEqual(result["allocation_only"],result["variable"])
        self.assertGreaterEqual(result["full_menu"],result["allocation_only"])

if __name__ == "__main__":
    unittest.main()
