from pathlib import Path
from runpy import run_path
import unittest


reader = run_path(str(Path(__file__).with_name("accept_repeat5202_float_e24.py")))


class RepeatE24ClosedReaderTest(unittest.TestCase):
    def test_split_initialization_without_changing_the_downstream_check(self):
        original = """
def expected_phase():
    return dict(seed=SEED, initialization_seed=SEED)
def expected_init():
    return dict(initialization_seed=SEED)
def check(phase):
    return phase.get("seed") == SEED and phase.get("initialization_seed") == SEED
"""
        namespace = {"SEED": 5202, "INITIALIZATION_SEED": 5201}
        exec(reader["split_stock_seed_source"](original), namespace)
        self.assertEqual(namespace["expected_phase"](), dict(seed=5202, initialization_seed=5201))
        self.assertEqual(namespace["expected_init"](), dict(initialization_seed=5201))
        self.assertTrue(namespace["check"](namespace["expected_phase"]()))
        self.assertFalse(namespace["check"](dict(seed=5202, initialization_seed=5202)))
        with self.assertRaisesRegex(ValueError, "immutable"):
            reader["split_stock_seed_source"](original.replace("initialization_seed=SEED", "initialization_seed=0"))

    def test_prior_closed_reader_keeps_exact_new_status_and_source(self):
        verifier = reader["closed_verifier"]()
        self.assertEqual(verifier.ARM_STATUS, reader["ARM_STATUS"])
        self.assertEqual(verifier.PAIR_STATUS, reader["PAIR_STATUS"])
        self.assertEqual(verifier.BRIDGE_SHA, reader["BRIDGE_SHA"])
        self.assertEqual(verifier.BRIDGE.name, "repeat5202_float_e24.py")


if __name__ == "__main__":
    unittest.main()
