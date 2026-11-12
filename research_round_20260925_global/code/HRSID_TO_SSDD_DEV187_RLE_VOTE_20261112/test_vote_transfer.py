import unittest

import numpy as np
from pycocotools import mask as codec

from vote_transfer import choose_pixels, exclusive_pairs, load_score


class Vote(unittest.TestCase):
    def test_cap_is_row_major_and_control_same_budget(self):
        anchor = np.zeros((12, 13), dtype=bool)
        anchor[3:9, 3:10] = True
        donor = anchor.copy()
        donor[1:3, 3:10] = True
        result = choose_pixels(anchor, donor, donor)
        vote, control, up, down = result
        self.assertEqual((up, down), (8, 0))
        self.assertEqual(np.flatnonzero(vote ^ anchor).tolist(), [16, 17, 18, 19, 20, 21, 22, 29])
        self.assertEqual(np.flatnonzero(control ^ anchor).tolist(), [28, 29, 30, 31, 32, 33, 34, 35])

    def test_frame_clip_and_drop_priority(self):
        anchor = np.ones((6, 6), dtype=bool)
        donor = anchor.copy()
        donor[:2, :2] = False
        vote, control, up, down = choose_pixels(anchor, donor, donor)
        self.assertEqual((up, down), (0, 4))
        self.assertEqual(np.flatnonzero(vote ^ anchor).tolist(), [0, 1, 6, 7])
        self.assertEqual(np.flatnonzero(control ^ anchor).tolist(), [0, 1, 2, 3])

    def test_exclusive_pair_uses_earlier_native_rank_tie(self):
        base = np.zeros((20, 20), dtype=np.uint8)
        base[3:10, 3:10] = 1
        other = np.roll(base, 1, axis=0)
        third = np.roll(base, -1, axis=0)
        def row(rank, mask):
            return (rank, int(mask.sum()), codec.encode(np.asfortranarray(mask)))
        pools = {5202: [row(2, base), row(3, other)],
                 5201: [row(4, other), row(5, third)],
                 5203: [row(3, base)]}
        self.assertEqual(list(exclusive_pairs(pools)), [(0, 0, 0)])

    def test_empty_rows_need_no_edit(self):
        anchor = np.ones((6, 6), dtype=bool)
        self.assertIsNone(choose_pixels(anchor, anchor, anchor))
        self.assertEqual(list(exclusive_pairs({5201: [], 5202: [], 5203: []})), [])

if __name__ == "__main__":
    unittest.main()