import unittest

import numpy as np

from dual_selector import select


class SelectTest(unittest.TestCase):
    def test_crosses_add_drop_sets_to_improve_the_worst_scenario(self):
        add = np.array([[.45, .45], [.95, .20], [.20, .95], [.50, .50]])
        drop = np.array([[.40, .40], [.68, .00], [.00, .83], [.31, .31]])
        result = select(add, drop, [0], [0], k=1)
        self.assertEqual(result["robust"], ((1,), (1,)))
        self.assertEqual(result["mean"], ((1,), (3,)))
        self.assertEqual(result["cell_pessimistic"], ((3,), (3,)))
        self.assertEqual(result["static"], ((0,), (0,)))

    def test_every_writer_preserves_the_signed_capacity(self):
        random = np.random.default_rng(5282)
        adds = random.random((12, 2))
        drops = random.random((9, 2))
        for k in (0, 1, 4, 8, 9):
            incumbent_add = tuple(range(12 - k, 12))
            incumbent_drop = tuple(range(9 - k, 9))
            result = select(adds, drops, incumbent_add, incumbent_drop, k=k)
            self.assertEqual(result["static"], (incumbent_add, incumbent_drop))
            for promoted, removed in result.values():
                self.assertEqual(len(set(promoted)), k)
                self.assertEqual(len(set(removed)), k)
                self.assertTrue(set(promoted) <= set(range(len(adds))))
                self.assertTrue(set(removed) <= set(range(len(drops))))
            chosen_add, chosen_drop = result["robust"]
            chosen = adds[list(chosen_add)].sum(axis=0) - drops[list(chosen_drop)].sum(axis=0)
            static = adds[list(incumbent_add)].sum(axis=0) - drops[list(incumbent_drop)].sum(axis=0)
            self.assertTrue(np.all(chosen >= static))

    def test_incumbent_wins_exact_worst_scenario_ties(self):
        add = np.ones((5, 2)) * .5
        drop = np.ones((4, 2)) * .25
        result = select(add, drop, [4, 3], [3, 1], k=2)
        self.assertEqual(result["robust"], ((3, 4), (1, 3)))
        self.assertEqual(result["mean"], ((0, 1), (0, 1)))
        self.assertEqual(result["cell_pessimistic"], result["mean"])

    def test_empty_support_and_invalid_incumbent(self):
        empty = np.empty((0, 2))
        self.assertEqual(select(empty, empty, [], [], k=0)["robust"], ((), ()))
        table = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            select(table, table, [0, 0], [0, 1], k=2)


if __name__ == "__main__":
    unittest.main()

