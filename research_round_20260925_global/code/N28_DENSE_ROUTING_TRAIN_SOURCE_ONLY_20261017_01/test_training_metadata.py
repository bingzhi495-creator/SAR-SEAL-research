"""Pure standard-library tests; do not construct or mock a Torch network."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    'n28_train_core', str(Path(__file__).with_name('train_core.py')))
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


class TrainingMetadataTests(unittest.TestCase):
    def test_order_is_preserved_and_requires_every_image_once(self):
        self.assertEqual(CORE.validate_order([2, 0, 1], 3), (2, 0, 1))
        for order in ([0, 0, 2], [0, 1], [0, 1, 3], [False, 1, 2]):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    CORE.validate_order(order, 3)

    def test_microbatches_preserve_gt_order_and_image_mean_with_tail(self):
        plan = CORE.chunk_plan(9, 4)
        self.assertEqual([(start, stop) for start, stop, _ in plan],
                         [(0, 4), (4, 8), (8, 9)])
        visits = [gt for start, stop, _ in plan for gt in range(start, stop)]
        self.assertEqual(visits, list(range(9)))
        losses = [1.0, 8.0, 3.0, 6.0, 2.0, 7.0, 4.0, 5.0, 90.0]
        accumulated = math.fsum(
            math.fsum(losses[start:stop]) / (stop-start) * weight
            for start, stop, weight in plan)
        self.assertAlmostEqual(accumulated, math.fsum(losses)/9)
        self.assertEqual(CORE.chunk_plan(1, 4), ((0, 1, 1.0),))

    def test_invalid_microbatch_population_or_weight_rejected(self):
        for count, size in ((0, 4), (2, 0), (True, 4), (3, False)):
            with self.subTest(count=count, size=size):
                with self.assertRaises(ValueError):
                    CORE.chunk_plan(count, size)
        for count, total in ((0, 3), (4, 3), (1, 0), (True, 3)):
            with self.subTest(count=count, total=total):
                with self.assertRaises(ValueError):
                    CORE.chunk_weight(count, total)

    def test_real_epoch_rejects_invalid_order_before_model_or_reader_access(self):
        with self.assertRaises(ValueError):
            CORE.train_epoch(None, None, None, None, [0]*741, None)


if __name__ == '__main__':
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('This test must not import Torch')
    unittest.main()
