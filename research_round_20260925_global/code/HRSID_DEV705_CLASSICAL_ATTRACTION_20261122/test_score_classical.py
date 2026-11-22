"""Guard matched per-candidate counts and preserve unedited cases."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("devscore", Path(__file__).with_name("score_classical.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class BudgetTests(unittest.TestCase):
    def test_matched_count_and_missing_arms(self):
        box = {name: {"rle": {}, "swaps": 3} for name in worker.NEW_ARMS}
        self.assertEqual(worker.new_budget(box, 32), 3)
        self.assertEqual(worker.new_budget({}, 32), 0)
        with self.assertRaises(ValueError):
            worker.new_budget(box, 31)
        with self.assertRaises(ValueError):
            worker.new_budget({worker.NEW_ARMS[0]: {"rle": {}, "swaps": 1}}, 100)
        box[worker.NEW_ARMS[1]]["swaps"] = 2
        with self.assertRaises(ValueError):
            worker.new_budget(box, 100)


if __name__ == "__main__":
    unittest.main()

