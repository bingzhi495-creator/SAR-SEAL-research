"""Stdlib metadata tests only; fixtures contain no tensors, pixels or GT masks."""
from copy import deepcopy
import unittest

import matched_export as core


def fixture():
    def ref(name, digest='a'*64):
        return dict(path='E:/synthetic_only/'+name, bytes=1, sha256=digest)
    sources = {
        'rgb_matched_reader': ref('rgb.py', '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025'),
        'matched_reader': ref('matched.py', '82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870'),
        'image_geometry': ref('geometry.py', '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7'),
    }
    context = {key: ref(key+'.json') for key in ('FIT_context', 'N13_qualified', 'N13_report')}
    identity = dict(image_id=123, ann_ids=[88, 21], original_shape=[5, 7],
                    actual_img_shape=[571, 800], batch_input_shape=[800, 800], image=ref('image.jpg'))
    item = dict(G=2, query_ids=[19, 7], identity=identity,
                rgb_receipt=dict(image=identity['image'], original_shape=[5, 7], actual_img_shape=[571, 800],
                    canvas_size=800, branch_size=400, original_mode='RGB', decoded_mode='RGB',
                    exif_orientation=None, auto_orientation=False,
                    input_convention='RGB uint8 -> FP32 /255; common reviewed geometry',
                    teacher_photometric_reconstruction=False,
                    baseline='N16 native_L selected by unchanged original GT-matched query ids',
                    image_source_control='base'))
    record = dict(dataset_index=54, image_id=123, G=2, query_ids=[19, 7], identity=deepcopy(identity))
    provenance = dict(context=context, source_refs=sources,
                      input_refs=list(context.values())+[identity['image'], ref('carrier.pt'), ref('packet.pt'), ref('L.pt')])
    return item, record, provenance


class MetadataContract(unittest.TestCase):
    def test_only_preselected_indices_and_original_order_are_preserved_without_mutating_inputs(self):
        item, record, provenance = fixture()
        original = deepcopy((item, record, provenance))
        result = core.package_metadata(54, item, record, provenance)
        self.assertEqual(result['dataset_index'], 54)
        self.assertEqual(result['query_ids'], [19, 7])
        self.assertEqual(result['ann_ids'], [88, 21])
        self.assertEqual(result['identity'], item['identity'])
        self.assertEqual((item, record, provenance), original)
        result['identity']['ann_ids'].reverse()
        self.assertEqual(item['identity']['ann_ids'], [88, 21])
        with self.assertRaises(ValueError):
            core.package_metadata(53, item, record, provenance)

    def test_query_or_gt_order_and_duplicate_mapping_are_rejected(self):
        item, record, provenance = fixture()
        for field in ('query_ids', 'ann_ids'):
            altered = deepcopy(item)
            values = altered['query_ids'] if field == 'query_ids' else altered['identity']['ann_ids']
            values.reverse()
            with self.assertRaises(ValueError):
                core.package_metadata(54, altered, record, provenance)
        item['query_ids'] = record['query_ids'] = [19, 19]
        with self.assertRaises(ValueError):
            core.package_metadata(54, item, record, provenance)

    def test_geometry_receipt_and_admitted_source_cannot_be_substituted(self):
        item, record, provenance = fixture()
        for mutation in ('geometry', 'source', 'missing_image', 'not_primitive'):
            modified, proof = deepcopy(item), deepcopy(provenance)
            if mutation == 'geometry':
                modified['rgb_receipt']['actual_img_shape'] = [570, 800]
            elif mutation == 'source':
                proof['source_refs']['image_geometry']['sha256'] = 'b'*64
            elif mutation == 'missing_image':
                proof['input_refs'] = list(proof['context'].values())
            else:
                modified['identity']['unportable_object'] = object()
                record = deepcopy(record)
                record['identity'] = modified['identity']
            with self.assertRaises(ValueError):
                core.package_metadata(54, modified, record, proof)


if __name__ == '__main__':
    unittest.main()
