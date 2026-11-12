import itertools
import unittest
import numpy as np
from score_headroom import oracle_counts, confusion_iou


class ExactOracle(unittest.TestCase):
    def test_closed_form_equals_bruteforce(self):
        anchor = np.array([True, True, True, False, False, False])
        truth = np.array([True, False, False, True, True, False])
        for keep in range(1 << len(anchor)):
            support = np.array([(keep >> i) & 1 for i in range(len(anchor))], dtype=bool)
            add_sites = np.flatnonzero(support & ~anchor)
            drop_sites = np.flatnonzero(support & anchor)
            for a in range(len(add_sites) + 1):
                for d in range(len(drop_sites) + 1):
                    good_add, good_drop = oracle_counts(anchor, truth, support, a, d)
                    numeric = confusion_iou(int(anchor.sum())+a-d, int(truth.sum()),
                                            int((anchor & truth).sum())+good_add-d+good_drop)
                    brute = []
                    for aa in itertools.combinations(add_sites, a):
                        for dd in itertools.combinations(drop_sites, d):
                            v = anchor.copy()
                            v[list(aa)] = True
                            v[list(dd)] = False
                            brute.append(float((v & truth).sum() / (v | truth).sum()))
                    self.assertAlmostEqual(numeric, max(brute))
