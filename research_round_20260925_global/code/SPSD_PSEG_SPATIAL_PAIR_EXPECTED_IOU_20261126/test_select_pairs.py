import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from pair_iou import graphs
from paired_expectation import prepare
from select_pairs import choose

CODE = Path(os.environ.get("SAR_PAIR_TEST_CODE", Path(__file__).resolve().parents[1]))


class JointSelectTest(TestCase):
    def test_zero_kappa_preserves_historical_exact_menu(self):
        writer = SimpleNamespace(**runpy.run_path(str(CODE /
            "SPSD_PSEG_GATE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/variable_writer.py")))
        indep = SimpleNamespace(**runpy.run_path(str(CODE /
            "SPSD_PSEG_EXPECTED_IOU_SIGNED_BUDGET_20261126/expected_iou.py")))
        rng = np.random.default_rng(919)
        points = np.arange(10, dtype=np.intp).reshape(5, 2)
        gs = graphs(np.stack((points // 3, points % 3), axis=-1), "fixture:13")
        weights = {which: [0., 0.] for which in gs}
        for _ in range(10):
            p = rng.uniform(0.01, 0.99, size=(5, 2, 2))
            static = ((0, 2), (1, 3))
            old, expected_counts = indep.select(writer, p, static, 2, 22)
            new, counts = choose(writer, indep, prepare, p, static, 2, 22, gs, weights)
            for arm in ("local", "nonspatial"):
                for quota in ("exact", "fixed"):
                    x, y = new[f"{arm}_{quota}"], old[f"expectation_{quota}"]
                    self.assertEqual(x["action"], y["action"])
                    self.assertAlmostEqual(x["score"], y["score"], places=12)
                self.assertEqual(counts[arm]["menu_actions"], expected_counts["menu_actions"])
