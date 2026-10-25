"""Synthetic metadata only; no actual GT, RLE pixels or scientific libraries."""
import unittest
import worker


def rows():
    return [dict(image_id=7,query_id=99-i,category_id=1,score=.2) for i in range(100)]


class Contract(unittest.TestCase):
    def test_all380_GT_remain_in_denominator_including_native_unmatched_six(self):
        scope = worker.coverage_scope(list(range(1,381)),set(range(1,375)))
        self.assertEqual(scope,dict(total_GT=380,native_matched_GT=374,native_unmatched_or_ignored_GT=6))
        with self.assertRaises(ValueError):
            worker.coverage_scope(list(range(1,375)),set(range(1,375)))

    def test_IoU_tie_uses_first_export_and_does_not_choose_by_score_rank(self):
        values = [.2]*100
        values[3] = values[8] = .8
        self.assertEqual(worker.best_export_position(values),3)
        self.assertEqual(values[8],.8)

    def test_S0_ranks_use_stable_export_order_for_ties_without_changing_scores(self):
        values = rows()
        values[0]['score'] = values[1]['score'] = .9
        before = [dict(row) for row in values]
        ranks = worker.stable_score_ranks(values)
        self.assertEqual(ranks[0:3],[1,2,3])
        self.assertEqual(values,before)

    def test_all100_queries_are_required_without_score_filter(self):
        values = rows()
        worker.require_all100(values)
        with self.assertRaises(ValueError):
            worker.require_all100(values[:-1])


if __name__ == '__main__':
    unittest.main()
