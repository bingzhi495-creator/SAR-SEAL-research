import unittest

import numpy as np

from box_shift_oof import predict_oof


class BoxShiftOOFTests(unittest.TestCase):
    def test_predictions_hold_out_every_row_from_the_same_fold(self):
        groups = [f"scene-{i}" for i in range(10) for _ in range(2)]
        box = np.tile([[-1., 1.], [1., -1.]], (10, 1))
        mask = np.zeros_like(box)
        target = (2 * box).astype(int)

        first = predict_oof(groups, box, mask, target)
        fold = first["folds"][0]
        held = first["folds"] == fold
        changed_target = target.copy()
        changed_target[held] = -changed_target[held]
        second = predict_oof(groups, box, mask, changed_target)

        self.assertEqual(set(first["folds"]), set(range(5)))
        self.assertTrue(np.array_equal(first["folds"][::2], first["folds"][1::2]))
        np.testing.assert_array_equal(first["identity"], np.zeros_like(target))
        np.testing.assert_array_equal(first["box"], target)
        np.testing.assert_array_equal(first["mask"], np.zeros_like(target))
        np.testing.assert_array_equal(first["constant"], np.zeros_like(target))
        for key in ("identity", "constant", "box", "mask"):
            np.testing.assert_array_equal(first[key][held], second[key][held])

    def test_pressure_rows_do_not_train_and_share_their_frozen_group_fold(self):
        groups = ["A", "B", "C", "D", "E", "A"]
        universe = list("ABCDEFG")
        box = np.array([[2., -1.], [1., 2.], [-2., 1.], [0., 1.], [3., 2.], [2., -1.]])
        mask = -box
        target = np.zeros_like(box, dtype=int)
        target[-1] = [2, -2]
        fit_mask = [True, True, True, True, True, False]

        output = predict_oof(groups, box, mask, target, all_groups=universe, fit_mask=fit_mask)

        np.testing.assert_array_equal(output["folds"], [1, 3, 0, 1, 0, 1])
        for key in ("identity", "constant", "box", "mask"):
            np.testing.assert_array_equal(output[key], np.zeros_like(target))

    def test_box_and_mask_share_penalty_after_independent_standardization(self):
        groups = [group for group in "ABCDE" for _ in range(2)] + ["A"]
        box = np.vstack((np.tile([[-1., 1.], [1., -1.]], (5, 1)), [.28, -.28]))
        mask = box * [4., 20.] + [70., -110.]
        target = np.vstack(((2 * box[:-1]).astype(int), [0, 0]))

        output = predict_oof(groups, box, mask, target, fit_mask=[True] * 10 + [False])

        # Standardized slopes have magnitude 2 / 1.25. At .28 their
        # predictions round to zero. Unregularized slopes round to one.
        np.testing.assert_array_equal(output["box"][-1], [0, 0])
        np.testing.assert_array_equal(output["box"], output["mask"])
        np.testing.assert_array_equal(output["box"][:-1], target[:-1])

    def test_constant_weights_groups_equally_and_half_steps_round_away(self):
        groups = ["A", "A", "B", "C", "C", "C", "D", "E"]
        box = np.zeros((8, 2))
        box[:3] = [[-1000., 1000.], [1000., -1000.], [1., -1.]]
        target = np.zeros((8, 2), dtype=int)
        target[2] = [2, -2]

        output = predict_oof(groups, box, np.zeros_like(box), target)

        # Group B has the only nonzero target in the four training groups.
        # Its mean contribution is exactly (+.5, -.5), despite unequal rows.
        np.testing.assert_array_equal(output["constant"][:2], [[1, -1], [1, -1]])
        np.testing.assert_array_equal(output["mask"][:2], output["constant"][:2])
        np.testing.assert_array_equal(output["box"][:2], [[-2, 2], [2, -2]])
