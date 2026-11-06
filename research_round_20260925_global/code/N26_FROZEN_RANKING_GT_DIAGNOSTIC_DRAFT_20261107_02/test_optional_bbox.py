"""SYNTHETIC-only validator regression; no COCO, GT, audit, model or AP."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

path = Path(__file__).with_name('worker.py')
spec = importlib.util.spec_from_file_location('n26_optional_bbox_candidate', path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def synthetic_population():
    images = [{'id': i, 'height': 8, 'width': 9} for i in range(187)]
    reference = [
        {'image_id': im['id'], 'category_id': 1, 'query_id': q,
         'score': 0.5, 'SYNTHETIC': True,
         'segmentation': {'size': [8, 9], 'counts': 'SYNTHETIC_NOT_RLE'}}
        for im in images for q in range(100)
    ]
    heads = {str(seed): copy.deepcopy(reference) for seed in worker.SEEDS}
    for rows in heads.values():
        rows[0]['score'] = 0.25
    return images, reference, heads


class OptionalBBoxTests(unittest.TestCase):
    def test_full_segmentation_only_population_is_accepted_without_mutation(self):
        images, reference, heads = synthetic_population()
        before = copy.deepcopy((reference, heads))
        counts = worker.validate_prediction_pair(reference, heads, images)
        self.assertEqual(counts, {role: 18700 for role in worker.ROLES})
        self.assertEqual((reference, heads), before)
        self.assertFalse(any('bbox' in row for row in reference))
        self.assertFalse(any('bbox' in row for rows in heads.values() for row in rows))
        self.assertFalse(any(name.split('.')[0] in
                             ('numpy', 'torch', 'pycocotools') for name in sys.modules))

    def test_explicit_malformed_bbox_is_still_rejected(self):
        images, reference, heads = synthetic_population()
        for bad in (None, [], [0, 0, 1], [0, 0, 1, float('nan')],
                    [0, 0, 1, float('inf')], [0, 0, 1, True]):
            with self.subTest(bbox=bad):
                reference[0]['bbox'] = bad
                with self.assertRaisesRegex(ValueError, 'finite bbox'):
                    worker.validate_prediction_pair(reference, heads, images)

    def test_optional_bbox_must_match_every_original_non_score_field(self):
        images, reference, heads = synthetic_population()
        reference[0]['bbox'] = [0, 0, 1, 1]
        for rows in heads.values():
            rows[0]['bbox'] = [0, 0, 1, 1]
        before = copy.deepcopy((reference, heads))
        self.assertEqual(worker.validate_prediction_pair(reference, heads, images)['R'],
                         18700)
        self.assertEqual((reference, heads), before)
        row = heads[str(worker.SEEDS[0])][0]
        row['bbox'] = [0, 0, 2, 1]
        with self.assertRaisesRegex(ValueError, 'non-score'):
            worker.validate_prediction_pair(reference, heads, images)
        del row['bbox']
        with self.assertRaisesRegex(ValueError, 'non-score'):
            worker.validate_prediction_pair(reference, heads, images)


if __name__ == '__main__':
    unittest.main()
