"""Small stdlib-only contract tests; no GT, scientific imports or AP."""
import copy
import unittest
import worker


def population():
    return [dict(image_id=7, query_id=q, category_id=1, score=0.2+0.1*q,
                 segmentation=dict(size=[2, 3], counts='6')) for q in range(2)]


class Contract(unittest.TestCase):
    def test_undefined_stats_are_retained_and_not_subtracted(self):
        left = [0.6, -1.0]+[0.4]*10
        right = [0.5, 0.8]+[0.3]*10
        result = worker.delta_stats(left, right)
        self.assertAlmostEqual(result['percentage_points'][0], 10)
        self.assertIsNone(result['percentage_points'][1])
        self.assertEqual(result['left_stats'], left)
        self.assertEqual(result['right_stats'], right)

    def test_mask_grid_cannot_change(self):
        native = population()
        refined = copy.deepcopy(native)
        refined[0]['segmentation']['size'] = [3,2]
        with self.assertRaises(ValueError):
            worker.validate_population(native, refined, [7], {7:[2,3]}, queries_per_image=2)

    def test_duplicate_native_query_is_rejected_even_when_both_lists_agree(self):
        native = population()
        native[1]['query_id'] = 0
        with self.assertRaises(ValueError):
            worker.validate_population(native, copy.deepcopy(native), [7], {7:[2,3]}, queries_per_image=2)

    def test_refinement_can_change_only_segmentation(self):
        native = population()
        refined = copy.deepcopy(native)
        refined[0]['segmentation']['counts'] = '06'
        worker.validate_population(native, refined, [7], {7: [2, 3]}, queries_per_image=2)
        refined[0]['score'] += 0.01
        with self.assertRaises(ValueError):
            worker.validate_population(native, refined, [7], {7: [2, 3]}, queries_per_image=2)
        refined = copy.deepcopy(native)
        refined[0]['category_id'] = True
        with self.assertRaises(ValueError):
            worker.validate_population(native, refined, [7], {7:[2,3]}, queries_per_image=2)


if __name__ == '__main__':
    unittest.main()
