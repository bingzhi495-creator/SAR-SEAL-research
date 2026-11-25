from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


SOURCE = Path(__file__).with_name("spatial_cues.py")


class CueTests(unittest.TestCase):
    def test_scramble_preserves_sign_multiset_without_native_coordinate_or_raw_changes(self):
        m = run_path(str(SOURCE))
        d = m["CUE_WIDTH"]
        raw = np.zeros((7, 2, 14), dtype=np.float32)
        raw[..., 0] = .25
        raw[..., 9] = np.arange(14).reshape((7, 2))
        cues = np.arange(14*d, dtype=np.float32).reshape(7, 2, d)
        j = m["transform_candidate"](raw, cues, "spatial", 18, 11)
        s = m["transform_candidate"](raw, cues, "scrambled", 18, 11)
        self.assertEqual(j.shape, (7,2,28))
        np.testing.assert_array_equal(j[...,:14], raw)
        np.testing.assert_array_equal(s[...,:14], raw)
        np.testing.assert_array_equal(j[...,-d:], cues)
        self.assertTrue(np.all(s[...,-d] != j[...,-d]))
        for side in (0,1):
            np.testing.assert_array_equal(np.sort(s[:,side,-d], axis=0), np.sort(cues[:,side,0], axis=0))
        np.testing.assert_array_equal(s, m["transform_candidate"](raw, cues, "scrambled", 18, 11))


    def test_all_sizes_are_bijective_and_small_sizes_remain_feasible(self):
        m = run_path(str(SOURCE))
        for n in range(33):
            for side in (0,1):
                a = m["scrambled_indices"](n, 3, 99, side)
                np.testing.assert_array_equal(np.sort(a), np.arange(n))
                if n > 1:
                    self.assertTrue(np.all(a != np.arange(n)))

    def test_baseline_prediction_uses_original_model_path(self):
        import torch

        m = run_path(str(SOURCE))
        raw = np.zeros((3,2,14), dtype=np.float32)
        raw[...,0] = .5
        center, scale = np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32)

        class Identity(torch.nn.Module):
            def forward(self, normalized, native):
                torch.testing.assert_close(normalized[...,0], native)
                return native.unsqueeze(-1).expand(-1,-1,2)

        actual = m["predict"](Identity(), raw, None, center, scale, "regression", 1, 4)
        np.testing.assert_array_equal(actual, np.repeat(torch.sigmoid(torch.full((3,2,1), .5)).numpy(), 2, axis=-1))


if __name__ == "__main__":
    unittest.main()
