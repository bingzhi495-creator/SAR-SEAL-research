"""Deferred pure stdlib matrix fixture; no NumPy, COCO, GT or AP execution.

Authoring qualification is AST/static compile only. ROOT may later execute
these artificial boundary cases; they do not qualify actual prediction input.
"""
import importlib.util
from pathlib import Path
import unittest


def load_greedy():
    path = Path(__file__).with_name('run.py')
    spec = importlib.util.spec_from_file_location('N08_pure_matrix_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.greedy_keep


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.greedy = staticmethod(load_greedy())

    def test_exact_threshold_is_kept_and_strict_excess_is_suppressed(self):
        for iou, expected in ((.80, [0, 1]), (.80001, [0])):
            result = self.greedy([.9, .8], [[1., iou], [iou, 1.]])
            self.assertEqual(result['keep_indices'], expected)

    def test_stable_ties_and_suppressed_candidate_has_no_power(self):
        result = self.greedy([.9, .9, .8, .1],
                             [[1., .80001, .80, 0.],
                              [.80001, 1., .99, 0.],
                              [.80, .99, 1., 0.],
                              [0., 0., 0., 0.]])
        self.assertEqual(result['stable_score_order'], [0, 1, 2, 3])
        self.assertEqual(result['keep_indices'], [0, 2, 3])
        self.assertEqual(result['suppressed_indices'], [1])

    def test_return_order_is_original_and_self_does_not_suppress(self):
        result = self.greedy([.7, .9, .4], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(result['kept_score_order'], [1, 0, 2])
        self.assertEqual(result['keep_indices'], [0, 1, 2])
        self.assertEqual(result['suppressed_indices'], [])

    def test_all100_empty_masks_remain(self):
        result = self.greedy([.2] * 100, [[0.] * 100 for _ in range(100)])
        self.assertEqual(result['keep_indices'], list(range(100)))
        self.assertEqual(result['suppressed_indices'], [])

    def test_tuning_or_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.greedy([.8], [[1.]], .7)
        with self.assertRaises(RuntimeError):
            self.greedy([float('nan')], [[1.]])
        with self.assertRaises(RuntimeError):
            self.greedy([.8], [[float('nan')]])


if __name__ == '__main__':
    unittest.main()
