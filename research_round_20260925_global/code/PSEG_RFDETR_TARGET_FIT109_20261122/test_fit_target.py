"""Synthetic parity check for the archived small-point-head training recipe."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


CODE = Path(sys.argv.pop(1))
spec = spec_from_file_location("fit_target", Path(__file__).with_name("fit_target.py"))
fit = module_from_spec(spec)
spec.loader.exec_module(fit)


class SmallHeadParity(unittest.TestCase):
    def test_compact_matches_archived_small_head_bitwise(self):
        old, trainer = fit.training_sources(CODE)
        rng = np.random.Generator(np.random.PCG64(17))
        records = []
        for r in range(3):
            x = rng.normal(size=(4 + r, 2, 14)).astype(np.float32)
            x[:, :, 0] = rng.uniform(.2, .8, size=x.shape[:2])
            q = rng.uniform(0, 1, size=x.shape[:2])
            records.append(dict(image_id=r + 1, rank=1, query_id=5, group=str(r % 2),
                                features=x, truth=q))
        torch.set_num_threads(1)
        original = old.fit_heads(records, trainer, steps=200, batch=64, seed=5217, sample_seed=20261112)
        compact = fit.fit_compact(records, old, trainer)
        for key in ('center', 'scale'):
            np.testing.assert_array_equal(compact[key], original[key])
        for expected, measured in zip(original['heads']['S'].parameters(),
                                      compact['model'].parameters(), strict=True):
            self.assertTrue(torch.equal(expected, measured))
        self.assertEqual(compact['audit']['common_batch_sha256'], original['audit']['common_batch_sha256'])
        self.assertEqual(compact['audit']['initial_tensor_sha256'],
                         original['audit']['heads']['S']['initial_tensor_sha256'])
        self.assertEqual(compact['audit']['optimizer_updates'], 200)


if __name__ == '__main__':
    unittest.main()
