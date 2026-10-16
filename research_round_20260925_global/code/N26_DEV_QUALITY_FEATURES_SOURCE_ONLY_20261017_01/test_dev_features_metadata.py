"""Pure hand-written identity metadata only; no tensors, GT, RLEs or Torch."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
source = Path(__file__).with_name('dev_features_body.py')
module_spec = importlib.util.spec_from_file_location('N26_DEV_METADATA_ONLY', str(source))
body = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(body)


def contexts():
    ids = list(range(1000, 1187))
    rows, order = [], []
    for index, image_id in enumerate(ids):
        shift = index % 100
        qids = list(range(shift, 100)) + list(range(shift))
        rows.append(dict(dataset_index=index, image_id=image_id,
                         qids=qids, original_shape=[3, 4]))
        order.extend([[image_id, query] for query in qids])
    newer = dict(path='SYNTHETIC_NEW.json', bytes=1, sha256='a'*64)
    older = dict(path='SYNTHETIC_OLD.json', bytes=2, sha256='b'*64)
    prediction = dict(schema='ROOT-N25-qualified-fixedmask-score-context-v1',
                      image_ids=ids, prediction_order=order, predictions={'R':newer})
    bundle = dict(schema='ROOT-N23-qualified-cache-DEV-context-v1',
                  image_ids=ids, rows=rows, R_predictions=older)
    return prediction, bundle


class PopulationMetadataTests(unittest.TestCase):
    def test_distinct_file_refs_do_not_invalidate_identity_metadata(self):
        prediction, bundle = contexts()
        before = copy.deepcopy((prediction, bundle))
        ids, rows, order = body._population(prediction, bundle)
        self.assertEqual((len(ids), len(rows), len(order)), (187, 187, 18700))
        self.assertNotEqual(prediction['predictions']['R'], bundle['R_predictions'])
        self.assertEqual(order, [tuple(pair) for pair in prediction['prediction_order']])
        self.assertEqual((prediction, bundle), before)

    def test_query_order_mismatch_is_rejected_without_a_ref_mismatch(self):
        prediction, bundle = contexts()
        bundle['R_predictions'] = copy.deepcopy(prediction['predictions']['R'])
        bundle['rows'][0]['qids'][0:2] = [1, 0]
        with self.assertRaisesRegex(AssertionError, 'exact prediction ordinals'):
            body._population(prediction, bundle)


if __name__ == '__main__':
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    unittest.main()
