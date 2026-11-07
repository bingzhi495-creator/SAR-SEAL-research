"""Official COCO contract checks using synthetic in-memory masks only."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from unittest import TestCase, main

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluate import evaluate_holdout


def compressed(bitmap):
    value = codec.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return dict(size=value['size'], counts=value['counts'].decode('ascii'))


def fixture():
    ids = list(range(111, 0, -1))
    bitmap = np.zeros((4, 4), dtype=np.uint8)
    empty = compressed(bitmap)
    bitmap[:2, :2] = 1
    hit = compressed(bitmap)
    gt = dict(images=[dict(id=i, height=4, width=4) for i in [*ids, 112]],
              categories=[dict(id=1, name='ship')],
              annotations=[dict(id=i, image_id=i, category_id=1, area=4, bbox=[0, 0, 2, 2],
                                segmentation=hit, iscrowd=0) for i in (1, 112)])
    baseline = {i: [] for i in ids}
    baseline[1] = [dict(image_id=1, category_id=1, score=.5, segmentation=rle) for rle in (empty, hit)]
    baseline[2] = [dict(image_id=2, category_id=1, score=.5, segmentation=hit)]
    edited = deepcopy(baseline)
    edited[1][0]['segmentation'] = hit
    return gt, baseline, edited, ids


def evaluate(gt, original, edited, ids):
    with redirect_stdout(StringIO()):
        return evaluate_holdout(gt, original, edited, ids, coco_class=COCO, cocoeval_class=COCOeval)


class HoldoutTests(TestCase):
    def test_stable_ties_retain_empty_and_unmatched_candidates(self):
        inputs = fixture()
        unchanged = deepcopy(inputs)
        result = evaluate(*inputs)
        self.assertEqual(inputs, unchanged)
        self.assertEqual(result['image_count'], 111)
        self.assertEqual(result['original']['prediction_count'], 3)
        self.assertEqual(result['original']['images_without_predictions'], 109)
        self.assertAlmostEqual(result['original']['AP'], .5)
        self.assertAlmostEqual(result['original']['AP75'], .5)
        self.assertAlmostEqual(result['edited']['AP'], 1.)
        self.assertAlmostEqual(result['edited']['AP75'], 1.)
        self.assertEqual(len(result['original']['stats']), 12)
        self.assertEqual(result['original']['stats'][4:6], [-1., -1.])
        self.assertAlmostEqual(result['delta']['AP'], .5)

    def test_empty_prediction_population_and_undefined_ground_truth_use_stock_values(self):
        gt, _, _, ids = fixture()
        empty = {image_id: [] for image_id in ids}
        result = evaluate(gt, empty, empty, ids)
        self.assertEqual(result['original']['prediction_count'], 0)
        self.assertEqual(result['original']['images_without_predictions'], 111)
        self.assertEqual(result['original']['stats'], [0., 0., 0., 0., -1., -1., 0., 0., 0., 0., -1., -1.])
        self.assertEqual(result['delta'], dict(AP=0., AP75=0.))
        gt['annotations'] = [row for row in gt['annotations'] if row['image_id'] == 112]
        result = evaluate(gt, empty, empty, ids)
        self.assertEqual(result['original']['stats'], [-1.] * 12)
        self.assertEqual(result['delta'], dict(AP=None, AP75=None))

    def test_rejects_score_changes_and_dropped_native_candidate(self):
        gt, original, edited, ids = fixture()
        edited[1][0]['score'] = .51
        with self.assertRaisesRegex(ValueError, 'Frozen score'):
            evaluate(gt, original, edited, ids)
        edited[1].pop(0)
        with self.assertRaisesRegex(ValueError, 'prediction count'):
            evaluate(gt, original, edited, ids)


if __name__ == '__main__':
    main()
