"""Synthetic metadata only; no tensor/image/NN/COCO execution or qualification."""
import copy
import unittest

import dev_rgb_reader as bridge


def image_ref(index):
    return dict(path='E:/SYNTHETIC_ONLY/DEV/image_%03d.jpg' % index,
                bytes=index + 1, sha256=('%064x' % (index + 1)))


def records():
    return [dict(dataset_index=i, image_id=1000 + i,
                 qids=list(range(i % 100, 100)) + list(range(i % 100)))
            for i in range(187)]


def native_item(index):
    # Opaque JSON metadata carriers exercise composition, never pretend tensors.
    row = records()[index]
    item = {name: {'opaque_metadata_only': name} for name in ('F', 'e', 'M', 'L')}
    item.update(identity=dict(image=image_ref(index), image_id=row['image_id']),
                query_ids=list(row['qids']))
    return item


class MetadataContract(unittest.TestCase):
    def test_complete_187_original_image_mapping_preserves_records(self):
        rows = records()
        refs = [image_ref(i) for i in range(187)]
        before = copy.deepcopy((rows, refs))
        self.assertEqual(bridge.prepare_image_references(rows, refs), refs)
        self.assertEqual((rows, refs), before)
        with self.assertRaises(RuntimeError):
            bridge.prepare_image_references(rows, refs[:-1])
        duplicate = copy.deepcopy(refs)
        duplicate[-1] = duplicate[0]
        with self.assertRaises(RuntimeError):
            bridge.prepare_image_references(rows, duplicate)

    def test_rgb_fields_never_overwrite_native_or_accept_wrong_image(self):
        item, row, reference = native_item(2), records()[2], image_ref(2)
        arguments = dict(rgb400={'new_metadata': 'rgb'}, geometry={'new_metadata': 'geometry'},
                         rgb_receipt={'image': reference})
        before = copy.deepcopy(item)
        result = bridge.compose_rgb_item(item, row, reference, **arguments)
        self.assertEqual(set(result), set(item) | {'rgb400', 'geometry', 'rgb_receipt'})
        self.assertEqual(item, before)
        for collision in ('rgb400', 'geometry', 'rgb_receipt'):
            with self.subTest(collision=collision):
                conflicting = dict(item, **{collision: 'existing'})
                with self.assertRaises(RuntimeError):
                    bridge.compose_rgb_item(conflicting, row, reference, **arguments)
        wrong = copy.deepcopy(reference)
        wrong['sha256'] = 'f' * 64
        with self.assertRaises(RuntimeError):
            bridge.compose_rgb_item(item, row, wrong, **arguments)

    def test_natural_field_objects_and_export_query_order_stay_unchanged(self):
        item, row, reference = native_item(73), records()[73], image_ref(73)
        additions = dict(rgb400={'metadata_only': 'RGB'}, geometry={'metadata_only': 'geometry'},
                         rgb_receipt={'image': reference})
        result = bridge.compose_rgb_item(item, row, reference, **additions)
        for key in ('F', 'e', 'M', 'L', 'identity', 'query_ids'):
            self.assertIs(result[key], item[key])
        self.assertNotEqual(result['query_ids'], list(range(100)))
        reordered = dict(item, query_ids=list(range(100)))
        with self.assertRaises(RuntimeError):
            bridge.compose_rgb_item(reordered, row, reference, **additions)


if __name__ == '__main__':
    unittest.main()
