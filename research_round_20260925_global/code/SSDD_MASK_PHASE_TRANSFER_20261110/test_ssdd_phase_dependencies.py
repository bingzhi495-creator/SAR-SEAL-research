from hashlib import sha256
from pathlib import Path
import unittest

from ssdd_phase_dependencies import load_worker


DIR = Path(__file__).resolve().parent


class SSDDPhaseDependenciesTest(unittest.TestCase):
    def pins(self):
        return {name: sha256((DIR / name).read_bytes()).hexdigest()
                for name in ("ssdd_phase_init.py", "ssdd_fit_only.py", "ssdd_phase_train.py")}

    def test_ordinary_worker_assembles_original_native_dependencies(self):
        run, dependencies, manifest = load_worker(self.pins())
        self.assertEqual(run.__name__, "run_fit_arm")
        self.assertEqual(dependencies.e100.EPOCH_STEPS, 186)
        self.assertEqual(dependencies.e100.WARMUP_STEPS, 558)
        self.assertEqual(dependencies.fit_only_data.__name__, "FitOnlyData")
        self.assertEqual(dependencies.empty_guard.__name__, "drop_empty_augmented_masks")
        self.assertEqual(len(manifest), 64)

    def test_wrong_worker_hash_fails_before_any_scientific_inputs(self):
        pins = self.pins()
        pins["ssdd_phase_train.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Exact pinned source bytes"):
            load_worker(pins)

