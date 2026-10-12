"""Hand-specified JSON/number fixtures; no data, Torch, COCO, or AP evaluator."""
import copy
import unittest

from crossed_predictions import assemble_crossed_predictions


def row(image, query, score, counts):
    return dict(image_id=image, query_id=query, category_id=1, score=score,
                segmentation=dict(size=[1, 2], counts=counts))


class CrossedPredictionsTests(unittest.TestCase):
    def setUp(self):
        self.native = [row(10, 1, .5, '011'), row(10, 0, .5, '2'),
                       row(20, 0, .3, '11'), row(20, 1, 0., '2')]
        self.posterior = [row(20, 1, .7, '02'), row(10, 0, .2, '2'),
                          row(20, 0, .1, '011'), row(10, 1, .6, '11')]
        self.budget = [row(10, 0, .8, '2'), row(20, 0, .9, '011'),
                       row(10, 1, .4, '11'), row(20, 1, 0., '02')]

    def test_default_all100_population_retains_empty_masks_and_image_shapes(self):
        native = [row(image, query, 0., '2')
                  for image in (20, 10) for query in reversed(range(100))]
        posterior = [row(r['image_id'], r['query_id'], .4, '02') for r in reversed(native)]
        budget = [row(r['image_id'], r['query_id'], 1., '02') for r in reversed(posterior)]
        for rows in (native, posterior, budget):
            for prediction in rows:
                if prediction['image_id'] == 20:
                    prediction['segmentation']['size'] = [2, 1]
        cells = assemble_crossed_predictions(native, posterior, budget, image_ids=[10, 20])
        for rows in cells.values():
            self.assertEqual(len(rows), 200)
            self.assertEqual([(r['image_id'], r['query_id']) for r in rows],
                             [(r['image_id'], r['query_id']) for r in native])
        self.assertTrue(all(r['segmentation']['counts'] == '2' and r['score'] == 1.
                            for r in cells['F01']))
        self.assertEqual(cells['F01'][0]['segmentation']['size'], [2, 1])
        self.assertEqual(cells['F01'][-1]['segmentation']['size'], [1, 2])

    def test_undefined_or_invalid_main_AP_never_generates_a_decomposition(self):
        from crossed_predictions import summarize_factorial_AP
        values = dict(F00=.4, F10=.5, F01=.35, F11=.6)
        for key in values:
            invalid = dict(values, **{key: -1.})
            with self.assertRaisesRegex(ValueError, '^undefined AP, decomposition unavailable$'):
                summarize_factorial_AP(invalid)
        for value in (float('nan'), float('inf'), True, 50., -.1, '0.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_factorial_AP(dict(values, F11=value))
        with self.assertRaises(ValueError):
            summarize_factorial_AP(dict(F00=.4, F10=.5, F01=.35))
        with self.assertRaises(ValueError):
            summarize_factorial_AP(dict(values, extra=.1))

    def test_numeric_summary_reports_both_conditional_paths_and_signed_interaction(self):
        from crossed_predictions import summarize_factorial_AP
        summary = summarize_factorial_AP(dict(F00=.4, F10=.5, F01=.35, F11=.6))
        effects = summary['effects']
        self.assertAlmostEqual(effects['geometry_at_S0'], .1)
        self.assertAlmostEqual(effects['score_at_G1'], .1)
        self.assertAlmostEqual(effects['score_at_G0'], -.05)
        self.assertAlmostEqual(effects['geometry_at_S1'], .25)
        self.assertAlmostEqual(summary['total_delta'], .2)
        self.assertAlmostEqual(summary['interaction'], .15)
        for residual in summary['decomposition_residuals'].values():
            self.assertAlmostEqual(residual, 0.)
        self.assertFalse(summary['unique_causal_attribution'])
        self.assertIsNone(summary['contribution_percentages'])
        negative = summarize_factorial_AP(dict(F00=.4, F10=.5, F01=.6, F11=.65))
        self.assertAlmostEqual(negative['interaction'], -.05)

    def test_scores_require_finite_probability_values_without_recomputation(self):
        for source in range(3):
            for score in (float('nan'), float('inf'), -float('inf'), -.01, 1.01, True, '0.4'):
                inputs = copy.deepcopy([self.native, self.posterior, self.budget])
                inputs[source][0]['score'] = score
                with self.subTest(source=source, score=score), self.assertRaises(ValueError):
                    assemble_crossed_predictions(*inputs, image_ids=[10, 20], query_count=2)
        inputs = copy.deepcopy([self.native, self.posterior, self.budget])
        inputs[2][0]['score'] = 1.
        cells = assemble_crossed_predictions(*inputs, image_ids=[10, 20], query_count=2)
        self.assertEqual(cells['F01'][1]['segmentation']['counts'], '2')
        self.assertEqual(cells['F01'][1]['score'], 1.)
        cells['F00'][0]['segmentation']['size'][0] = 9
        self.assertEqual(cells['F01'][0]['segmentation']['size'], [1, 2])
        self.assertEqual(inputs[0][0]['segmentation']['size'], [1, 2])

    def test_incompatible_geometry_or_prediction_schema_is_rejected(self):
        for defect in ('posterior_mask_changed', 'cross_dimensions', 'within_image_dimensions',
                       'category', 'bool_category', 'extra_field', 'non_ascii_rle', 'missing_counts'):
            inputs = copy.deepcopy([self.native, self.posterior, self.budget])
            if defect == 'posterior_mask_changed':
                inputs[1][0]['segmentation']['counts'] = '2'
            elif defect == 'cross_dimensions':
                for rows in inputs[1:]:
                    for prediction in rows:
                        prediction['segmentation']['size'] = [2, 1]
            elif defect == 'within_image_dimensions':
                inputs[0][0]['segmentation']['size'] = [2, 1]
            elif defect == 'category':
                inputs[2][0]['category_id'] = 2
            elif defect == 'bool_category':
                inputs[2][0]['category_id'] = True
            elif defect == 'extra_field':
                inputs[0][0]['bbox'] = [0, 0, 1, 1]
            elif defect == 'non_ascii_rle':
                inputs[2][0]['segmentation']['counts'] = '\u4e00'
            else:
                del inputs[0][0]['segmentation']['counts']
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                assemble_crossed_predictions(*inputs, image_ids=[10, 20], query_count=2)

    def test_incomplete_or_duplicate_query_populations_are_rejected(self):
        for source in range(3):
            for defect in ('duplicate', 'missing', 'unexpected_query', 'bool_query', 'unexpected_image'):
                inputs = copy.deepcopy([self.native, self.posterior, self.budget])
                if defect == 'duplicate':
                    inputs[source].append(copy.deepcopy(inputs[source][0]))
                elif defect == 'missing':
                    inputs[source].pop()
                elif defect == 'unexpected_query':
                    inputs[source][0]['query_id'] = 2
                elif defect == 'bool_query':
                    inputs[source][0]['query_id'] = True
                else:
                    inputs[source][0]['image_id'] = 30
                with self.subTest(source=source, defect=defect), self.assertRaises(ValueError):
                    assemble_crossed_predictions(*inputs, image_ids=[10, 20], query_count=2)
        with self.assertRaises(ValueError):
            assemble_crossed_predictions(self.native, self.posterior, self.budget,
                                         image_ids=[10, 20, 20], query_count=2)

    def test_crossing_joins_identity_and_preserves_native_tie_order(self):
        before = copy.deepcopy((self.native, self.posterior, self.budget))
        cells = assemble_crossed_predictions(self.native, self.posterior,
                    self.budget, image_ids=[10, 20], query_count=2)
        self.assertEqual(set(cells), {'F00', 'F10', 'F01', 'F11'})
        self.assertEqual(cells['F00'], self.native)
        native_keys = [(r['image_id'], r['query_id']) for r in self.native]
        for rows in cells.values():
            self.assertEqual([(r['image_id'], r['query_id']) for r in rows], native_keys)
            self.assertEqual(len(rows), 4)
        self.assertEqual([r['score'] for r in cells['F10']], [.5, .5, .3, 0.])
        self.assertEqual([r['score'] for r in cells['F01']], [.4, .8, .9, 0.])
        self.assertEqual([r['score'] for r in cells['F11']], [.4, .8, .9, 0.])
        self.assertEqual([r['segmentation']['counts'] for r in cells['F01']],
                         ['011', '2', '11', '2'])
        self.assertEqual([r['segmentation']['counts'] for r in cells['F10']],
                         ['11', '2', '011', '02'])
        self.assertEqual((self.native, self.posterior, self.budget), before)


if __name__ == '__main__':
    unittest.main()
