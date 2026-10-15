"""One pure metadata fixture; no torch, payload reads or scientific proof."""

import importlib.util
from pathlib import Path
import unittest


class ParentAssociationTest(unittest.TestCase):
    def test_original_bundle_member_and_natural_order(self):
        path = Path(__file__).with_name('N20_DEV_TRUE_FEATURE_NEIGHBOR_CAPTURE_BODY_UNBOUND_DRAFT_20261015_01.py')
        spec = importlib.util.spec_from_file_location('_n20_metadata_synthetic', path)
        body = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(body)
        # Synthetic strict-ref SHAPES only; these files do not exist or get read.
        bundle = dict(path='E:/SYNTHETIC_ONLY/bundle.pth', bytes=1, sha256='a' * 64)
        packet = dict(path='E:/SYNTHETIC_ONLY/native.pth', bytes=1, sha256='b' * 64)
        selector = dict(path='E:/SYNTHETIC_ONLY/selector.pth', bytes=1, sha256='c' * 64)
        identity = dict(image_id=17, original_shape=[2, 3])
        qids = list(reversed(range(100)))
        row = dict(dataset_index=4, image_id=17, identity=identity, native_packet=packet,
                   selector=selector, qids=qids, bundle=bundle,
                   shared_actual_image_inputs='i004/SHARED_ACTUAL_INPUTS.pth')
        shared = dict(schema='N16-actual-GPU-image-shared-inputs-v1', dataset_index=4,
                      image_id=17, identity=identity, native_packet=packet, selector=selector,
                      natural_query_ids=list(range(100)), native_export_query_ids=qids,
                      actual_GPU_F_captured=True, CPU_F_rerendered=False)
        actual = body._parent_metadata(row, 4, packet, shared)
        self.assertEqual(actual, dict(parent_bundle=bundle,
            parent_shared_member='i004/SHARED_ACTUAL_INPUTS.pth', native_packet=packet, identity=identity))
        self.assertEqual(shared['native_export_query_ids'], qids)
        with self.assertRaises(RuntimeError):
            body._parent_metadata(dict(row, shared_actual_image_inputs='i003/SHARED_ACTUAL_INPUTS.pth'), 4, packet, shared)
        with self.assertRaises(RuntimeError):
            body._parent_metadata(row, 4, dict(packet, sha256='d' * 64), shared)
        with self.assertRaises(RuntimeError):
            body._parent_metadata(row, 4, packet, dict(shared, native_export_query_ids=list(range(100))))


if __name__ == '__main__':
    unittest.main()
