import unittest
from training_core import validate_order

class VisitTests(unittest.TestCase):
    def test_order_is_a_complete_image_permutation(self):
        self.assertEqual(validate_order([2, 0, 1], 3), (2, 0, 1))
        for order in ([0, 0, 2], [0, 1], [0, 1, 3], [True, 0, 2]):
            with self.assertRaises(ValueError):
                validate_order(order, 3)

    def test_invalid_order_fails_before_model_or_torch_access(self):
        from training_core import train_one_epoch
        class Unusable:
            def __getattr__(self, name):
                raise AssertionError("unexpected scientific access: " + name)
        with self.assertRaises(ValueError):
            train_one_epoch(Unusable(), Unusable(), Unusable(), [[], []],
                            [0, 0], Unusable(), Unusable())

    def test_raw_GT_order_is_preserved_within_each_image(self):
        from training_core import iter_epoch_gt
        images = [[{"GT_annotation_id": 9}, {"GT_annotation_id": 2}],
                  [{"GT_annotation_id": 7}]]
        visits = [(i, gt["GT_annotation_id"])
                  for i, gt in iter_epoch_gt(images, [1, 0])]
        self.assertEqual(visits, [(1, 7), (0, 9), (0, 2)])

if __name__ == '__main__':
    unittest.main()
