from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


WORKER = Path(__file__).with_name("fit_banks.py")


class ActionBankTests(unittest.TestCase):
    def test_deployment_writes_same_budget_and_keeps_exact_robust_action(self):
        make = run_path(str(WORKER))["deployment_bank"]
        static = ((0,), (0,))
        old = np.zeros((2, 3, 2), dtype=np.int8)
        old[0, (0, 1), 0] = (-1, 1)
        old[1, (0, 1), 1] = (1, -1)
        estimates = np.asarray([
            [[.1, .1], [.9, .9]],
            [[.9, .9], [.1, .1]],
            [[.8, .8], [.2, .2]],
        ])

        class Selector:
            @staticmethod
            def _candidate_sets(values, incumbent, k, largest):
                self.assertEqual(k, 1)
                ranked = np.argsort(-values[:, 0] if largest else values[:, 0], kind="stable")
                return incumbent, (int(ranked[0]),)

            @staticmethod
            def select(add, drop, inc_add, inc_drop, *, k):
                return {"static": (inc_add, inc_drop), "robust": ((1,), (1,))}

        new, audit = make(estimates, static, old, Selector)
        np.testing.assert_array_equal(new[0], old[0] + old[1])
        np.testing.assert_array_equal(new.sum(axis=1), np.zeros((2, 2)))
        self.assertEqual(new.shape, old.shape)
        self.assertEqual(audit["menu_used"], 2)
        self.assertTrue(audit["robust_nonstatic"])


    def test_group_folds_keep_duplicate_and_are_order_invariant(self):
        folds = run_path(str(WORKER))["fixed_folds"]
        groups = [f"rgb_{idx}" for idx in range(27)]
        assigned = folds(groups + [groups[3], groups[7]])
        self.assertEqual(assigned[-2], assigned[3])
        self.assertEqual(assigned[-1], assigned[7])
        np.testing.assert_array_equal(assigned[:27], folds(list(reversed(groups)))[::-1])
        count = np.bincount(assigned[:27], minlength=5)
        self.assertEqual(int(count.max() - count.min()), 1)

    def test_static_only_menu_reuses_exact_old_proposals(self):
        make = run_path(str(WORKER))["deployment_bank"]
        static = ((0,), (0,))
        old = np.zeros((2, 2, 2), dtype=np.int8)
        old[0, :, 0] = (-1, 1)
        old[1, :, 1] = (1, -1)

        class Selector:
            @staticmethod
            def _candidate_sets(values, incumbent, k, largest):
                return (incumbent,)

            @staticmethod
            def select(add, drop, inc_add, inc_drop, *, k):
                return {"static": (inc_add, inc_drop), "robust": (inc_add, inc_drop)}

        new, audit = make(np.full((2, 2, 2), .5), static, old, Selector)
        np.testing.assert_array_equal(new, old)
        self.assertEqual(audit["menu_used"], 0)
        self.assertEqual(audit["shared_actions"], 2)
        self.assertFalse(audit["robust_nonstatic"])


if __name__ == "__main__":
    unittest.main()
