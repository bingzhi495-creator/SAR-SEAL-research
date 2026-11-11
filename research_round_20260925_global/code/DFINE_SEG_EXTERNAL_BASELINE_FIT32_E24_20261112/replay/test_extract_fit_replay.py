import unittest

from extract_fit_replay import validate_epoch


class FitReplayTest(unittest.TestCase):
    def test_exact_order_and_batches(self):
        ids = [8, 6, 1, 3, 6, 1, 2, 3]
        line = dict(epoch=2, visits=[[i, index] for index, i in enumerate(ids)])
        batches, repeated = validate_epoch(line, 2, {1, 2, 3, 6, 8}, padding=3)
        self.assertEqual(batches, [[8, 6, 1, 3], [6, 1, 2, 3]])
        self.assertEqual(repeated, {1: 1, 3: 1, 6: 1})
        with self.assertRaisesRegex(ValueError, "FIT identity"):
            validate_epoch(line, 2, {1, 2, 3, 6, 9}, padding=3)
        with self.assertRaisesRegex(ValueError, "chronological"):
            validate_epoch(line, 3, {1, 2, 3, 6, 8}, padding=3)


if __name__ == "__main__":
    unittest.main()

