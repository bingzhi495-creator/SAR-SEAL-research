import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest

from e24_dev_export_worker import ema_state_for_model, select_items, write_population


def encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def population():
    order = [777, 4, *range(10, 195)]
    return [(None, {'sha256': 'a' * 64}, image_id, (2, 2)) for image_id in order]


def image_result(item, *, mapped=False):
    _, _, image_id, _ = item
    rows = [dict(image_id=image_id, category_id=1, score=score,
                 segmentation=dict(size=[2, 2], counts='fictional')) for score in (0.9, 0.14)] if mapped else []
    count = len(rows)
    return rows, dict(raw_query_count=100, native_count=100, unmapped_slot1_count=100-count,
                      retained_native_ranks=[2, 91] if mapped else [], coco_rows=count,
                      codec_roundtrip_masks=count, nonempty_masks=1 if mapped else 0,
                      all_zero_masks=1 if mapped else 0,
                      canonical_json_sha256=hashlib.sha256(encode(rows)).hexdigest())


class PopulationTests(unittest.TestCase):
    def test_canonical_declared_order_zero_images_and_native_ranks_survive(self):
        items = population()
        calls = []

        def infer(item):
            calls.append(item[2])
            return image_result(item, mapped=item[2] == 4)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve() / 'private-export'
            outcome = write_population(items, infer, output_directory=output, checkpoint_sha256='b'*64,
                                       recipe_sha256='c'*64)
            prediction = (output/'PREDICTIONS.json').read_bytes()
            manifest = json.loads((output/'IMAGE_MANIFEST.json').read_bytes())
            self.assertEqual(calls, [item[2] for item in items])
            self.assertEqual(outcome['predictions_ref']['sha256'], hashlib.sha256(prediction).hexdigest())
            self.assertEqual(outcome['prediction_count'], 2)
            self.assertEqual(outcome['zero_prediction_images'], 186)
            self.assertEqual([row['score'] for row in json.loads(prediction)], [0.9, 0.14])
            self.assertEqual([row['image_id'] for row in manifest['images']], calls)
            self.assertEqual([(row['prediction_start'], row['prediction_count']) for row in manifest['images'][:3]],
                             [(0, 0), (0, 2), (2, 0)])
            self.assertEqual(manifest['images'][1]['retained_native_ranks'], [2, 91])
            self.assertEqual(manifest['native_count_total'], 18700)


    def test_failed_first_two_gate_preserves_partial_and_does_not_continue(self):
        calls = []

        def infer(item):
            calls.append(item[2])
            return image_result(item)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve() / 'private-export'
            with self.assertRaisesRegex(ValueError, 'First two fixed images'):
                write_population(population(), infer, output_directory=output, checkpoint_sha256='b'*64,
                                 recipe_sha256='c'*64)
            self.assertEqual(calls, [777, 4])
            self.assertEqual((output/'PREDICTIONS.json.partial').read_bytes(), b'[')
            self.assertFalse((output/'PREDICTIONS.json').exists())
            self.assertFalse((output/'IMAGE_MANIFEST.json').exists())

    def test_caller_order_is_checked_without_sorting_or_discovery(self):
        ids = [item[2] for item in population()]
        items = [dict(image_id=image_id, original_hw=[2, 2],
                      image_ref=dict(path='/synthetic/valid/{}.jpg'.format(image_id), bytes=1, sha256='a'*64))
                 for image_id in ids]
        api = SimpleNamespace(MAX_PIXELS=100, MAX_JPEG_BYTES=100,
                              original_ref=lambda ref, **kwargs: PurePosixPath(ref['path']))
        self.assertEqual([row[2] for row in select_items(items, ids, api)], ids)
        with self.assertRaisesRegex(ValueError, 'image ID/order'):
            select_items(items, sorted(ids), api)

    def test_e24_ema_must_have_4464_updates_and_matching_native_model_keys(self):
        checkpoint = {
            'global_step': 4464,
            'state_dict': {'model.class_embed.weight': 'raw'},
            'callbacks': {'RFDETREMACallback': {
                'latest_update_step': 4464,
                'average_model_state_dict': {'n_averaged': 4464, 'module.model.class_embed.weight': 'ema'},
            }},
        }
        mapped, raw = ema_state_for_model(checkpoint, {'class_embed.weight'})
        self.assertEqual(mapped, {'class_embed.weight': 'ema'})
        self.assertEqual(raw, 'raw')
        checkpoint['callbacks']['RFDETREMACallback']['average_model_state_dict']['n_averaged'] = 186
        with self.assertRaisesRegex(ValueError, '4464 native EMA updates'):
            ema_state_for_model(checkpoint, {'class_embed.weight'})


if __name__ == '__main__':
    unittest.main()
