import unittest

import numpy as np

from phoenix_fit_cpu_pilot import refine_image


class FakeRefiner:
    def __init__(self):
        self.calls = []

    def refine_batch(self, masks, **settings):
        self.calls.append((len(masks), settings))
        return [~mask for mask in masks]


class FixedCandidateTest(unittest.TestCase):
    def test_rule_and_native_metadata_stay_independent(self):
        masks = []
        rows = []
        for rank, area in ((2, 0), (3, 1), (5, 2), (6, 3), (8, 4)):
            mask = np.zeros((10, 10), dtype=bool)
            mask.flat[:area] = True
            masks.append(mask)
            rows.append(dict(native_rank=rank, query_id=rank + 10, category_id=1,
                             score=rank / 100, segmentation={'size': [10, 10], 'counts': str(area)}))
        worker = FakeRefiner()
        encode = lambda mask: {'size': [10, 10], 'counts': str(int(mask.sum()))}
        stock, stock_summary = refine_image(worker, rows, masks, 'stock_coco', encode)
        broad, broad_summary = refine_image(worker, rows, masks, 'all_nonempty', encode)
        self.assertEqual([r['segmentation']['counts'] for r in stock], ['0', '1', '98', '97', '96'])
        self.assertEqual([r['segmentation']['counts'] for r in broad], ['0', '99', '98', '97', '96'])
        for source, left, right in zip(rows, stock, broad):
            for key in ('native_rank', 'query_id', 'category_id', 'score'):
                self.assertEqual(source[key], left[key])
                self.assertEqual(source[key], right[key])
        self.assertEqual([r['segmentation']['counts'] for r in rows], ['0', '1', '2', '3', '4'])
        self.assertEqual([count for count, _ in worker.calls], [2, 1, 2, 2])
        self.assertTrue(all(call['max_batch'] == 2 and call['refine_iters'] == 5
                            for _, call in worker.calls))
        self.assertEqual((stock_summary['eligible'], broad_summary['eligible']), (3, 4))
        self.assertEqual((stock_summary['empty_input'], broad_summary['empty_output']), (1, 1))


if __name__ == '__main__':
    unittest.main()
