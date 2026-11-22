"""Small invariant test for the extra matched-count arms."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).with_name("compare.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class BudgetTests(unittest.TestCase):
    def test_same_budget_for_all_three_arms(self):
        arms = {name: {"swaps": 3} for name in comparison.NEW_ARMS}
        self.assertEqual(comparison.validate_extra_budget(arms), 3)
        self.assertEqual(comparison.validate_extra_budget({}), 0)
        arms[comparison.NEW_ARMS[1]]["swaps"] = 2
        with self.assertRaises(ValueError):
            comparison.validate_extra_budget(arms)
        with self.assertRaises(ValueError):
            comparison.validate_extra_budget({comparison.NEW_ARMS[0]: {"swaps": 1}})


if __name__ == "__main__":
    unittest.main()

