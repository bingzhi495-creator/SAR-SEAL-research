"""Synthetic JSON metadata only; never imports Torch, NumPy or real inputs."""
import copy
import unittest

import resource_probe as probe


def rows():
    return [dict(dataset_index=7, G=1, identity=dict(original_shape=[10, 20])),
            dict(dataset_index=3, G=3, identity=dict(original_shape=[20, 40])),
            dict(dataset_index=10, G=4, identity=dict(original_shape=[10, 10])),
            dict(dataset_index=2, G=1, identity=dict(original_shape=[10, 20]))]


def locked_pre():
    return dict(schema='U39-full-method-cross-image-resource-prelock-v1',
        image_order=[2, 3, 2], arms=['PLAIN', 'FULL'], updates_per_arm=3,
        total_pilot_updates=6, image_batch_size=1, query_chunk=1, branch_size=400,
        canvas_size=800, dtype='float32', initialization_seed=202610173101,
        optimizer=dict(name='AdamW', lr=3e-4, weight_decay=1e-4,
                       betas=[.9, .999], eps=1e-8, foreach=False, fused=False),
        training_encode_saved_tensor_mode='save_on_cpu_pin_memory_true',
        checkpointing=False, AMP=False, VJP=False, automatic_retry=False,
        main_training=False, no_AP=True, no_DEV=True, no_TEST=True)


class MetadataContract(unittest.TestCase):
    def test_small_max_G_area_small_uses_no_loss_and_stable_ties(self):
        original = rows()
        before = copy.deepcopy(original)
        self.assertEqual(probe.select_image_order(original), [2, 3, 2])
        self.assertEqual(original, before)

    def test_invalid_geometry_and_duplicate_ordinal_are_rejected(self):
        invalid = rows()
        invalid[0]['identity']['original_shape'] = [0, 20]
        with self.assertRaises(ValueError):
            probe.select_image_order(invalid)
        invalid = rows()+[rows()[0]]
        with self.assertRaises(ValueError):
            probe.select_image_order(invalid)
        for value in ([], [dict(dataset_index=True, G=1, identity=dict(original_shape=[2, 3]))],
                      [dict(dataset_index=0, G=0, identity=dict(original_shape=[2, 3]))],
                      [dict(dataset_index=741, G=1, identity=dict(original_shape=[2, 3]))]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.select_image_order(value)

    def test_fixed_recipe_and_order_reject_unlocked_changes(self):
        self.assertEqual(probe.validate_prelock(locked_pre(), rows()), [2, 3, 2])
        for change in ('lr', 'image_order', 'query_chunk', 'foreach', 'AMP'):
            pre = locked_pre()
            if change == 'lr':
                pre['optimizer']['lr'] = .001
            elif change == 'image_order':
                pre['image_order'] = [3, 2, 3]
            elif change == 'query_chunk':
                pre['query_chunk'] = True
            elif change == 'foreach':
                pre['optimizer']['foreach'] = True
            else:
                pre['AMP'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                probe.validate_prelock(pre, rows())


if __name__ == '__main__':
    unittest.main()
