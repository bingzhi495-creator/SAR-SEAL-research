"""Synthetic metadata contracts only; no SAM/scientific library execution."""
import copy
import unittest

import worker


def population():
    images = []
    predictions = []
    for ordinal in range(187):
        image_id = 9000 - ordinal
        images.append(dict(ordinal=ordinal, image_id=image_id, shape=[2, 2],
                           jpeg_ref=dict(path='/images/%d.jpg' % ordinal,
                                         bytes=1, sha256='a' * 64)))
        for query_id in list(range(50, 100)) + list(range(50)):
            predictions.append(dict(image_id=image_id, query_id=query_id,
                                    category_id=1, score=0.01,
                                    segmentation=dict(size=[2, 2], counts='4')))
    return images, predictions


class PopulationContract(unittest.TestCase):
    def test_population_keeps_export_permutation_and_opaque_empty_rles(self):
        images, predictions = population()
        before = copy.deepcopy(predictions)
        groups = worker.validate_population(predictions, images)
        self.assertEqual(len(groups), 187)
        self.assertEqual(groups[0], list(range(100)))
        self.assertEqual(groups[-1], list(range(18600, 18700)))
        self.assertEqual([predictions[i]['query_id'] for i in groups[0]],
                         list(range(50, 100)) + list(range(50)))
        self.assertEqual(predictions, before)

    def test_output_changes_only_segmentation_without_reordering_metadata(self):
        rows = [dict(image_id=8, query_id=99, category_id=1, score=1e-9,
                     extra='untouched', segmentation=dict(size=[2, 2], counts='4')),
                dict(image_id=8, query_id=2, category_id=1, score=0.9,
                     segmentation=dict(size=[2, 2], counts='4'))]
        before = copy.deepcopy(rows)
        encoded = {1: dict(size=[2, 2], counts='121'),
                   0: dict(size=[2, 2], counts='04')}
        output = worker.replace_segmentations(rows, encoded)
        self.assertEqual(rows, before)
        self.assertEqual([r['query_id'] for r in output], [99, 2])
        for i in range(2):
            self.assertEqual(output[i]['segmentation'], encoded[i])
            self.assertEqual({k: v for k, v in output[i].items() if k != 'segmentation'},
                             {k: v for k, v in rows[i].items() if k != 'segmentation'})
        with self.assertRaises(ValueError):
            worker.replace_segmentations(rows, {0: encoded[0]})

    def test_author_receives_all100_including_empty_masks_in_one_five_round_call(self):
        masks = [[[0, 0], [0, 0]] for unused in range(100)]
        masks[3][0][0] = 1
        marker = object()
        calls = []

        def author(path, submitted, sam, **params):
            calls.append((path, submitted, sam, params))
            return submitted, 'aux1', 'aux2'

        result = worker.invoke_author('/original.png', masks, marker, author)
        self.assertIs(result[0], masks)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], masks)
        self.assertIs(calls[0][2], marker)
        self.assertEqual(len(calls[0][1]), 100)
        self.assertEqual(calls[0][3], dict(resize_transform=None, use_point=True,
            use_box=True, use_mask=True, add_neg=True, iters=5, margin=0.0,
            gamma=4.0, strength=30, use_samhq=False, ddp=False, is_train=False))


if __name__ == '__main__':
    unittest.main()
