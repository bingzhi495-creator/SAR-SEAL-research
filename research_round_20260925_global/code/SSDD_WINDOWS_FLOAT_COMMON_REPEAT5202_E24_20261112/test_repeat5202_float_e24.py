from pathlib import Path
from runpy import run_path
import unittest


class RepeatE24Test(unittest.TestCase):
    def test_exact_new_full_owner_identity_and_clock(self):
        entry = run_path(str(Path(__file__).with_name("repeat5202_float_e24.py")))
        full = entry["bridge"]()
        manifest = full.launch_manifest("native", {}, {}, "a" * 40)
        self.assertEqual(full.key_for("native"), "SSDD_WINDOWS_Q100_FLOAT_COMMON_E24_NATIVE_5202")
        self.assertEqual(full.key_for("center"), "SSDD_WINDOWS_Q100_FLOAT_COMMON_E24_CENTER_5202")
        self.assertEqual(manifest["initialization_seed"], 5201)
        self.assertEqual(manifest["training_seed"], 5202)
        self.assertEqual(manifest["complete_epochs"], 24)
        self.assertEqual(manifest["fit_visits"], 17856)
        self.assertEqual(manifest["optimizer_updates"], 4464)
        self.assertEqual(manifest["optimizer_plan_updates"], 18600)
        self.assertEqual(manifest["warmup_updates"], 558)
        self.assertTrue(manifest["FIT"])
        self.assertFalse(manifest["DEV"] or manifest["TEST"] or manifest["AP"])
        self.assertEqual(manifest["original_short_sha256"], entry["SHORT_PAIR_SHA"])


if __name__ == "__main__":
    unittest.main()
