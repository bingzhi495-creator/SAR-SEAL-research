import unittest

from p330_common_preflight import SEED, prospective_positions, compare_common


class PreflightTest(unittest.TestCase):
    def test_sample_repeats_seeded_order_and_does_not_use_global_rng(self):
        import random
        prior = random.getstate()
        ids = prospective_positions()
        self.assertEqual(random.getstate(), prior)
        self.assertEqual(ids, tuple(random.Random(SEED).sample(range(741), 32)))
        self.assertEqual(len(set(ids)), 32)

    def test_common_comparison_requires_same_consumed_ledger_and_real_masks(self):
        left = {"ledger":"same", "chosen_sha":"A", "changed":2, "crop":3, "visits":32}
        right = dict(left, chosen_sha="B")
        self.assertTrue(compare_common(left, right))
        with self.assertRaisesRegex(ValueError, "common.*ledger"):
            compare_common(left, dict(right, ledger="foreign"))
        with self.assertRaisesRegex(ValueError, "differing"):
            compare_common(left, dict(right, chosen_sha="A"))


if __name__ == "__main__":
    unittest.main()
