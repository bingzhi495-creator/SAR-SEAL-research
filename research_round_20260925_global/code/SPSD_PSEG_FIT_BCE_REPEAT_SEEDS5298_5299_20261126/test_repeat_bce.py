from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
import unittest


WORKER = Path(__file__).with_name("fit_repeat_bce.py")


class SeedFitContractTests(unittest.TestCase):
    def test_pure_bce_override_is_seeded_and_restored(self):
        f = run_path(str(WORKER))
        core = SimpleNamespace(SEED=5297,MODES=("bce","bce_signed_action"))
        observed = []

        def run(packets, *, source_core, seed):
            self.assertEqual(seed, core.SEED)
            self.assertEqual(core.MODES,("bce",))
            observed.append(seed)
            return packets,source_core

        core.train_matched_action = run
        self.assertEqual(f["fit_seed"](core,"packets","source",5298),("packets","source"))
        self.assertEqual(f["fit_seed"](core,"packets","source",5299),("packets","source"))
        self.assertEqual(observed,[5298,5299])
        self.assertEqual(core.SEED,5297)
        self.assertEqual(core.MODES,("bce","bce_signed_action"))


if __name__ == "__main__":
    unittest.main()
