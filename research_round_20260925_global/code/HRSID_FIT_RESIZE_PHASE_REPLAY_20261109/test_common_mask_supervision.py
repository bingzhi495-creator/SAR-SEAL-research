import unittest

import torch

from common_mask_supervision import ROW_ID, common_targets


def target(rows, live):
    masks = torch.zeros((len(rows), 2, 3), dtype=torch.bool)
    for i, column in enumerate(live):
        if column is not None:
            masks[i, 0, column] = True
    return {
        ROW_ID: torch.tensor(rows, dtype=torch.int64),
        "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]] * len(rows)).reshape(-1, 4),
        "labels": torch.zeros(len(rows), dtype=torch.int64),
        "area": torch.ones(len(rows)),
        "iscrowd": torch.zeros(len(rows), dtype=torch.int64),
        "masks": masks,
        "image_id": torch.tensor([7]),
        "orig_size": torch.tensor([4, 6]),
        "size": torch.tensor([2, 3]),
    }


class CommonMaskSupervisionTest(unittest.TestCase):
    def test_preserves_same_nonempty_ids_and_each_arms_mask(self):
        native = target([0, 1, 2, 3], [0, 1, 1, None])
        center = target([1, 2, 3, 4], [2, None, 0, 2])
        a, b = common_targets(native, center)
        self.assertEqual(a[ROW_ID].tolist(), [1])
        self.assertEqual(b[ROW_ID].tolist(), [1])
        self.assertTrue(torch.equal(a["masks"], native["masks"][1:2]))
        self.assertTrue(torch.equal(b["masks"], center["masks"][0:1]))
        self.assertEqual(native[ROW_ID].tolist(), [0, 1, 2, 3])
        self.assertEqual(center[ROW_ID].tolist(), [1, 2, 3, 4])


    def test_three_arms_use_one_joint_intersection(self):
        a, b, c = common_targets(
            target([0, 1, 2], [0, 1, 2]),
            target([0, 1, 2], [1, None, 0]),
            target([1, 2, 3], [2, 1, 0]),
        )
        self.assertEqual([arm[ROW_ID].tolist() for arm in (a, b, c)], [[2], [2], [2]])
        self.assertEqual([int(arm["masks"][0, 0].nonzero().item()) for arm in (a, b, c)], [2, 0, 1])

    def test_disjoint_nonempty_rows_are_valid_empty_targets(self):
        a, b = common_targets(target([0], [0]), target([1], [1]))
        for arm in (a, b):
            self.assertEqual(arm[ROW_ID].tolist(), [])
            self.assertEqual(tuple(arm["boxes"].shape), (0, 4))
            self.assertEqual(tuple(arm["masks"].shape), (0, 2, 3))
            self.assertEqual(arm["image_id"].item(), 7)

    def test_unequal_nonmask_fields_and_order_are_rejected(self):
        native, center = target([0, 1], [0, 1]), target([0, 1], [1, 2])
        center["boxes"][0, 2] = 0.25
        with self.assertRaisesRegex(ValueError, "nonmask fields"):
            common_targets(native, center)
        with self.assertRaisesRegex(ValueError, "order"):
            common_targets(native, target([1, 0], [1, 2]))

    def test_cross_image_and_duplicate_annotation_ids_are_rejected(self):
        native, center = target([0], [0]), target([0], [2])
        center["image_id"][0] = 8
        with self.assertRaisesRegex(ValueError, "different images"):
            common_targets(native, center)
        with self.assertRaisesRegex(ValueError, "unique"):
            common_targets(target([0, 0], [0, 1]), center)


if __name__ == "__main__":
    unittest.main()
