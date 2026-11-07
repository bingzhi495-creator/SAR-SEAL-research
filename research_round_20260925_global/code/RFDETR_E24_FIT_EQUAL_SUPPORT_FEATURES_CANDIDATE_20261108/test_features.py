import unittest

import numpy as np

from features import extract_features


class EqualSupportFeaturesTests(unittest.TestCase):
    def test_clipped_window_uses_same_support_and_zero_vote_for_ties(self):
        gray = np.array(
            [[250, 2, 1, 255], [4, 3, 2, 255], [3, 2, 3, 255], [255] * 4],
            dtype=np.uint8,
        )
        mask = np.zeros(gray.shape, dtype=bool)
        mask[0, 1] = mask[1, 0] = True
        result = extract_features(gray, mask, np.array([0]), np.array([-0.75]))

        self.assertEqual(result["flat_indices"].tolist(), [0])
        self.assertEqual(result["abs_logit"].tolist(), [0.75])
        self.assertEqual(result["edit_sign"].tolist(), [1])
        self.assertEqual(result["valid"].tolist(), [True])
        self.assertAlmostEqual(result["mean_contrast"][0], 1 / 8)
        self.assertAlmostEqual(result["rank"][0], 1 / 3)


    def test_invalid_rows_and_zero_gray_keep_their_identity(self):
        gray = np.zeros((7, 8), dtype=np.uint8)
        mask = np.zeros(gray.shape, dtype=bool)
        mask[0, 0] = mask[4, 5:8] = True
        indices = np.array([0, 36, 37], dtype=np.int32)
        logits = np.array([2.0, -3.0, 4.0], dtype=np.float32)
        result = extract_features(gray, mask, indices, logits)

        np.testing.assert_array_equal(result["flat_indices"], indices)
        np.testing.assert_array_equal(result["abs_logit"], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result["edit_sign"], [-1, 1, -1])
        np.testing.assert_array_equal(result["valid"], [False, True, True])
        np.testing.assert_array_equal(result["mean_contrast"], [0.0] * 3)
        np.testing.assert_array_equal(result["rank"], [0.0] * 3)

    def test_rank_matches_explicit_pairwise_vote_on_tied_gray(self):
        rng = np.random.default_rng(5201)
        gray = rng.integers(0, 7, size=(11, 9), dtype=np.uint8).T
        mask = (rng.random((11, 9)) < 0.45).T
        indices = np.arange(gray.size)
        result = extract_features(gray, mask, indices, np.zeros(gray.size))
        remapped = extract_features(gray * 3 + 5, mask, indices, np.zeros(gray.size))
        np.testing.assert_array_equal(result["rank"], remapped["rank"])

        for index in indices:
            row, col = divmod(index, gray.shape[1])
            inside, outside = [], []
            for yy in range(max(0, row - 2), min(gray.shape[0], row + 3)):
                for xx in range(max(0, col - 2), min(gray.shape[1], col + 3)):
                    if (yy, xx) != (row, col):
                        side = inside if mask[yy, xx] else outside
                        side.append(int(gray[yy, xx]))
            valid = min(len(inside), len(outside)) >= 2
            self.assertEqual(result["valid"][index], valid)
            expected = 0.0
            if valid:
                votes = sum((a > b) - (a < b) for a in inside for b in outside)
                expected = votes / (len(inside) * len(outside))
            self.assertAlmostEqual(result["rank"][index], expected)

    def test_rejects_misaligned_inputs_without_reordering(self):
        gray = np.zeros((2, 2), dtype=np.uint8)
        mask = np.zeros(gray.shape, dtype=bool)
        cases = [
            ([1, 1], [0.0, 1.0]),
            ([1, 0], [0.0, 1.0]),
            ([-1, 0], [0.0, 1.0]),
            ([0, 4], [0.0, 1.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([0, 1], [0.0]),
            ([0, 1], [0.0, np.nan]),
        ]
        for indices, logits in cases:
            with self.subTest(indices=indices, logits=logits):
                with self.assertRaises(ValueError):
                    extract_features(gray, mask, np.asarray(indices), np.asarray(logits))
        with self.assertRaises(ValueError):
            extract_features(gray, mask.astype(np.uint8), np.array([0]), np.array([0.0]))
        with self.assertRaises(ValueError):
            extract_features(gray.astype(float), mask, np.array([0]), np.array([0.0]))
        result = extract_features(gray, mask, np.array([], dtype=np.int64), np.array([]))
        self.assertTrue(all(value.size == 0 for value in result.values()))


if __name__ == "__main__":
    unittest.main()

