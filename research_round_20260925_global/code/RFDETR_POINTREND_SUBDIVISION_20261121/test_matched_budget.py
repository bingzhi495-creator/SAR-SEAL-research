"""Focused checks for the fixed edit-count projection."""

import unittest

from matched_budget import Edits, project_matched_budget


class MatchBudgetTest(unittest.TestCase):
    def test_quotas_change_area_and_locations_at_same_count(self):
        add = [(6, .8), (7, .95), (9, .95)]
        remove = [(2, .6), (4, .45), (5, .42)]
        balanced, joint = project_matched_budget(add, remove, 2)
        self.assertEqual(balanced, Edits((7, 9), (5, 4), 0))
        self.assertEqual(joint, Edits((7, 9, 6), (5,), 0))
        self.assertEqual(balanced.area_delta, 0)
        self.assertEqual(joint.area_delta, 2)

    def test_zero_count_and_signed_ties(self):
        add, remove = [(9, .5)], [(2, .5), (4, .5)]
        self.assertEqual(project_matched_budget(add, remove, 0), (Edits((), (), 0),) * 2)
        self.assertEqual(project_matched_budget(add, remove, 1),
                         (Edits((9,), (2,), 2), Edits((), (2, 4), 2)))

    def test_bad_support(self):
        with self.assertRaisesRegex(ValueError, "appear once"):
            project_matched_budget([(2, .8)], [(2, .1)], 1)
        with self.assertRaisesRegex(ValueError, "swap count"):
            project_matched_budget([(2, .8)], [], 1)


if __name__ == "__main__":
    unittest.main()
