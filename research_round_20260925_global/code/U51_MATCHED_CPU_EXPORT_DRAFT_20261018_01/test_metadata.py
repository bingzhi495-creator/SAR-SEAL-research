"""Tiny metadata contracts only; no ML import, tensors, GT or actual inputs."""
import unittest

import matched_export as core


class BatchMetadata(unittest.TestCase):
    def test_batch_preserves_prelocked_order_and_returns_an_independent_copy(self):
        requested = [740, 0, 31]
        result = core.validate_batch(requested)
        self.assertEqual(result, [740, 0, 31])
        result.reverse()
        self.assertEqual(requested, [740, 0, 31])

    def test_batch_rejects_empty_oversized_duplicate_or_invalid_original_indices(self):
        for bad in ([], tuple([0]), list(range(33)), [0, 0], [-1], [741], [True], [1.0], [0, None]):
            with self.subTest(batch=bad):
                with self.assertRaises(ValueError):
                    core.validate_batch(bad)
        self.assertEqual(core.validate_batch(list(range(32))), list(range(32)))

    def test_item_keeps_original_ordinal_and_requires_membership_in_the_locked_batch(self):
        def ref(name, sha='a'*64):
            return dict(path='E:/synthetic_metadata_only/'+name, bytes=1, sha256=sha)
        context = {key: ref(key) for key in ('FIT_context', 'N13_qualified', 'N13_report')}
        sources = {key: ref(key, sha) for key, sha in core.SOURCE_SHA256.items()}
        image = ref('image.jpg')
        identity = dict(image_id=123, ann_ids=[88], original_shape=[2, 4], actual_img_shape=[400, 800],
                        batch_input_shape=[800, 800], image=image)
        rgb = dict(image=image, original_shape=[2, 4], actual_img_shape=[400, 800], canvas_size=800,
            branch_size=400, original_mode='RGB', decoded_mode='RGB', exif_orientation=None,
            auto_orientation=False, teacher_photometric_reconstruction=False,
            input_convention='RGB uint8 -> FP32 /255; common reviewed geometry',
            baseline='N16 native_L selected by unchanged original GT-matched query ids', image_source_control='base')
        item = dict(G=1, query_ids=[83], identity=identity, rgb_receipt=rgb)
        provenance = dict(context=context, source_refs=sources, input_refs=list(context.values())+[image])
        batch = [740, 0]
        for index in batch:
            record = dict(dataset_index=index, image_id=123, G=1, query_ids=[83], identity=identity)
            result = core.package_metadata(index, item, record, provenance, batch_indices=batch)
            self.assertEqual(result['dataset_index'], index)
            self.assertEqual(result['query_ids'], [83])
            self.assertEqual(result['population_metadata'], dict(images=741, GT=1661, tensor_population_revalidated=False))
        record['dataset_index'] = 54
        with self.assertRaises(ValueError):
            core.package_metadata(54, item, record, provenance, batch_indices=batch)
        self.assertEqual(batch, [740, 0])


if __name__ == '__main__':
    unittest.main()
