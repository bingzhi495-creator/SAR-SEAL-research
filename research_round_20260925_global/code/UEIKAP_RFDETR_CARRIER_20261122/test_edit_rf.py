"""Synthetic tests against the released RF-DETR edit primitives."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np

path = Path(__file__).with_name("edit_rf.py")
spec = spec_from_file_location("ueikap_edit_test", path)
edit = module_from_spec(spec)
spec.loader.exec_module(edit)
CODE = Path(sys.argv.pop(1))


class FrozenActions(unittest.TestCase):
    def test_independent_choice_and_native_controls_keep_realized_counts(self):
        ref = edit.load_module(CODE / edit.BASE / "replay.py", edit.REPLAY_SHA)
        m = ref.methods(CODE)
        stock = np.zeros((30, 57), dtype=bool)
        stock[7:19, 8:22] = True
        stock[10:13, 10:14] = False
        rng = np.random.default_rng(88)
        p = rng.random(stock.shape, dtype=np.float32)
        rgb = rng.integers(256, size=(*stock.shape, 3), dtype=np.uint8)
        pair, sampler = ref.canvas(m, stock.shape)
        coords, feats, *_ = pair.paired_points(stock, (2., 2., 45., 27.), int(stock.sum()), .8, p,
                                                m.pair.image_features(rgb), sampler)
        q = np.round(feats[:, :, 0] * .5 + rng.random(coords.shape, dtype=np.float32) * .5, 1)
        a = edit.policies(m, stock, coords, q, p, independent=False)
        b = edit.policies(m, stock, coords, q, p, independent=True)
        self.assertEqual(a, b)
        self.assertGreater(len(a["fixed"]), 0)
        self.assertEqual(len(a["fixed"]), len(a["free_matched_count"]))
        self.assertEqual(len(a["free_natural"]), len(a["native_matched_count"]))
        expected = {arm: ref.mask_edit(stock, coords, seq) for arm,seq in a.items() if seq}
        edit.validate_edits(stock, coords, expected)
        arm = "free_matched_count"
        expected[arm]["swaps"] += 1
        with self.assertRaisesRegex(ValueError, "Matched per-candidate budgets"):
            edit.validate_edits(stock, coords, expected)


if __name__ == "__main__":
    unittest.main()
