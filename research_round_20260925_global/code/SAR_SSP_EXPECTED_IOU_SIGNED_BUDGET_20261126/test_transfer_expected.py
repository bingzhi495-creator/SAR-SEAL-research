from pathlib import Path
from runpy import run_path
import unittest


WORKER = Path(__file__).with_name("transfer_expected.py")


class SourceAdaptationTests(unittest.TestCase):
    def test_named_source_seams_are_verified_before_replacement(self):
        f = run_path(str(WORKER))
        self.assertEqual(f["patch"]("a original c","original","updated"),"a updated c")
        with self.assertRaises(ValueError):
            f["patch"]("absent","original","updated")
        with self.assertRaises(ValueError):
            f["patch"]("original original","original","updated")


if __name__ == "__main__":
    unittest.main()
