import unittest

from fit_plan import source_folds


class FoldTest(unittest.TestCase):
    def test_group_order_is_input_invariant_and_never_splits_a_prefix(self):
        groups = ["P12", "P05", "P6", "P20", "P33", "P6", "P44", "P19"]
        original = source_folds(groups)
        self.assertEqual(source_folds(reversed(groups)), original)
        self.assertEqual(set(original), set(groups))
        self.assertEqual(set(original.values()), set(range(5)))

    def test_requires_at_least_one_group_for_each_fold(self):
        with self.assertRaises(ValueError):
            source_folds(["P1", "P1", "P2"])


if __name__ == "__main__":
    unittest.main()
