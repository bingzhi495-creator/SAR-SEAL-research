"""Check the same-budget comparison and reference projection contract."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("replay", Path(__file__).with_name("replay.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class TestProjection(unittest.TestCase):
    def test_keeps_old_policies_and_requires_all_three_new_budgets(self):
        previous = {"24": {"fixed": {"swaps": 1}, "free_matched_count": {"swaps": 1}}}
        row = {key: dict(value) for key, value in previous.items()}
        row["24"].update({key: {"swaps": 3} for key in replay.NEW_ARMS})
        self.assertEqual(replay.check_projection(row, previous), (1, 3))
        wrong_budget = {key: dict(value) for key, value in row.items()}
        wrong_budget["24"][replay.NEW_ARMS[1]] = {"swaps": 2}
        with self.assertRaises(ValueError):
            replay.check_projection(wrong_budget, previous)
        wrong_projection = {key: dict(value) for key, value in row.items()}
        wrong_projection["24"]["fixed"] = {"swaps": 2}
        with self.assertRaises(ValueError):
            replay.check_projection(wrong_projection, previous)
        partial = {key: dict(value) for key, value in row.items()}
        del partial["24"][replay.NEW_ARMS[0]]
        with self.assertRaises(ValueError):
            replay.check_projection(partial, previous)
        self.assertEqual(replay.check_projection(previous, previous), (0, 0))


if __name__ == "__main__":
    unittest.main()

