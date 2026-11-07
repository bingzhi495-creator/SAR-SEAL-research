"""Synthetic independent verifier contract."""
from unittest import TestCase, main

from post import replay_cell


class ReplayTests(TestCase):
    def test_joint_reaches_exact_threshold_at_k16_and_keeps_preexisting_pass(self):
        # Only the floating GT needs 150 points. Adding 6 and deleting 10
        # obtains 150 exactly at K=16. Its counterpart was already at 9/10.
        replay = replay_cell(((180, 200), (183, 220)), 6, 10)
        self.assertEqual(replay['baseline'], dict(integer=True, float=False, dual=False))
        self.assertFalse(replay['curve'][8]['dual'])
        self.assertTrue(replay['curve'][16]['dual'])
        self.assertTrue(replay['curve'][None]['dual'])
        self.assertEqual(replay['minimum_edits'], dict(integer=0, float=16, dual=16))


if __name__ == '__main__':
    main()
