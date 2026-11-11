import unittest
from score_seam_context import summarize

class SeamSummaryTest(unittest.TestCase):
    def test_missing_zero_and_strict_hit_change(self):
        rows=[dict(offset=(0,600),near=0.,far=.81,inside_a=.82,inside_b=.78),
              dict(offset=(0,600),near=.84,far=.74,inside_a=.84,inside_b=.85)]
        for row in rows:
            for key in ("near","far","inside_a","inside_b"):
                row[key+"_matched"]=row[key]>0
        a=summarize(rows)
        self.assertAlmostEqual(a['far_minus_near_points'],35.5)
        self.assertEqual(a['strict_far_minus_near'],0)
        self.assertEqual(a['far_only_matches'],1)
        self.assertAlmostEqual(a['paired_matched_only_delta_points'],-10.0)

    def test_explicit_stock_matches_are_separate_from_geometry(self):
        a=summarize([dict(offset=(0,600),near=0.,far=.8,inside_a=0.,inside_b=0.,
                          near_matched=True,far_matched=True,inside_a_matched=True,inside_b_matched=False)])
        self.assertEqual(a["matched_both"],1)
        self.assertEqual(a["interior_a_only"],1)

if __name__ == '__main__':
    unittest.main()
