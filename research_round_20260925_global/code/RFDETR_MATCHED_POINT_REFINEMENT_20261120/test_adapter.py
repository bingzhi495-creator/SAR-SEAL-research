import unittest
import torch
from adapter import native_probability


class AdapterTests(unittest.TestCase):
    def test_all_policy_counts_lw_area_and_immutable_identity(self):
        from adapter import readouts, replace_segmentation
        stock = [False, True, False, True, False, True]
        pairs = [(0, 1), (2, 3), (4, 5)]
        small = [( .9, .1), (.4, .6), (.3, .7)]
        large = [(.9, .55), (.85, .6), (.8, .65)]
        out = readouts(stock, pairs, small, large, [(.5, .5)] * 3, 2)
        self.assertEqual(set(out), {'N', 'SF', 'SU=F', 'SU', 'LF=S', 'LU=S', 'LW=S', 'NP=S'})
        for name, row in out.items():
            self.assertEqual(row['edits'], 0 if name == 'N' else 2)
            self.assertEqual(row['area_delta'], 2 if name == 'LW=S' else 0)
        self.assertEqual(out['LW=S']['flipped'], [0, 2])
        tied = readouts(stock, pairs, small, [(.5, .5)] * 3, [(.5, .5)] * 3, 2)
        self.assertEqual(tied['LW=S']['flipped'], [0, 1])
        self.assertEqual(tied['LW=S']['nonpositive_flips'], 2)
        reordered = readouts(stock, [(0,5),(2,1),(4,3)], small,
                             large, [(.5,.5)] * 3, 2)
        self.assertEqual(reordered['NP=S']['flipped'], [0,5])
        no_op = readouts(stock, pairs, [(.1, .9)] * 3, large, large, 2)
        self.assertTrue(all(row['edits'] == 0 for row in no_op.values()))
        record = {'score': .73, 'native_rank': 9, 'query_id': 31,
                  'category_id': 1, 'segmentation': {'counts': 'original'}}
        changed = replace_segmentation(record, {'counts': 'new'})
        self.assertEqual(record['segmentation']['counts'], 'original')
        self.assertEqual({k:v for k,v in changed.items() if k != 'segmentation'},
                         {k:v for k,v in record.items() if k != 'segmentation'})

    def test_author_half_pixel_and_zero_tie(self):
        x = torch.tensor([[[-1., 1.], [-1., 1.]]])
        p, mask = native_probability(torch, x, (3, 3))
        expected = torch.tensor([[[-1., 0., 1.]]]).expand(1, 3, 3)
        self.assertTrue(torch.equal(p, expected.sigmoid()))
        self.assertTrue(torch.equal(mask, expected > 0))
        self.assertTrue(torch.equal(p[:, :, 1], torch.full((1, 3), .5)))
        self.assertFalse(bool(mask[:, :, 1].any()))
        native_probability(torch, x, (3, 3), mask)
        with self.assertRaises(ValueError):
            native_probability(torch, x, (3, 3), ~mask)
        # Rounding sigmoid to .5 must never redefine original >0 stock.
        _, tiny = native_probability(torch, torch.full((1, 1, 1), 1e-10), (1, 1))
        self.assertTrue(bool(tiny.item()))


if __name__ == '__main__':
    unittest.main()
