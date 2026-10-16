"""Only hand-written synthetic arrays; no project predictions or GT."""
import unittest
import numpy as np

from N25_FIXEDMASK_SCORE_CORE_UNBOUND_20261017_01 import quality_scores


class QualityScoreTests(unittest.TestCase):
    def test_all_GT_max_and_two_distinct_frozen_scoring_rules(self):
        ious = np.array([[.9, .1], [.9, .4], [.2, .6], [0., 0.]])
        original = np.array([.2, .8, .9, 0.])
        result = quality_scores(np, ious, original)
        np.testing.assert_array_equal(result['quality'], [.9, .9, .6, 0.])
        np.testing.assert_array_equal(result['GT_PRODUCT'], original * np.array([.9, .9, .6, 0.]))
        np.testing.assert_array_equal(result['GT_IOU_ONLY'], [.9, .9, .6, 0.])
        self.assertEqual(np.argsort(-result['GT_IOU_ONLY'], kind='mergesort').tolist(), [0, 1, 2, 3])
        self.assertEqual(np.argsort(-result['GT_PRODUCT'], kind='mergesort').tolist(), [1, 2, 0, 3])
        np.testing.assert_array_equal(original, [.2, .8, .9, 0.])

    def test_no_GT_and_FP64_product_without_extra_clipping(self):
        scores = np.array([.12345679, 1.], dtype=np.float32)
        empty = quality_scores(np, np.empty((2, 0)), scores)
        np.testing.assert_array_equal(empty['quality'], [0., 0.])
        np.testing.assert_array_equal(empty['GT_PRODUCT'], [0., 0.])
        actual = quality_scores(np, np.array([[.87654321], [.25]]), scores)
        self.assertEqual(actual['GT_PRODUCT'].dtype, np.dtype('float64'))
        self.assertEqual(actual['GT_PRODUCT'][0], float(scores[0]) * .87654321)
        self.assertEqual(actual['GT_PRODUCT'][1], .25)

    def test_bad_values_and_row_misalignment_are_rejected(self):
        for matrix, scores in [([[float('nan')]], [.5]), ([[1.1]], [.5]),
                               ([[-.1]], [.5]), ([[.5]], [float('inf')]),
                               ([[.5]], [1.01]), ([[.5], [.2]], [.5]),
                               ([.5], [.5])]:
            with self.subTest(matrix=matrix, scores=scores):
                with self.assertRaises((ValueError, AssertionError)):
                    quality_scores(np, matrix, scores)


if __name__ == '__main__':
    unittest.main()
