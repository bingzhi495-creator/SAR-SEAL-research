"""Regression tests for the same-count deletion-location controls."""

import unittest
import numpy as np

from depth_priority import deletion_location_masks


class DeletionLocationTests(unittest.TestCase):
    def test_interior_removals_shift_to_an_original_edge_without_budget_change(self):
        raw = np.zeros((11, 13), dtype=bool)
        raw[2:9, 3:10] = True
        stock = raw.copy()
        stock[4, 6] = False
        stock[5, 6] = False
        stock[5, 10] = True
        periphery, proximity, stats = deletion_location_masks(raw, stock)
        old_delete = raw & ~stock
        outer = raw.copy()
        outer[3:8, 4:9] = False
        new_p = raw & ~periphery
        new_q = raw & ~proximity
        self.assertEqual(2, stats["relocated_deletions"])
        self.assertTrue(np.all(new_p <= outer))
        self.assertTrue(np.any(new_q & ~outer))
        for arm, deletions in ((periphery, new_p), (proximity, new_q)):
            self.assertEqual(2, int(deletions.sum()))
            self.assertTrue(np.array_equal(arm & ~raw, stock & ~raw))
            self.assertFalse(np.any(deletions & old_delete))


    def test_peripheral_ties_retain_existing_removals(self):
        raw = np.ones((5, 5), dtype=bool)
        stock = raw.copy()
        stock[1, 0] = False
        stock[4, 3] = False
        periphery, proximity, stats = deletion_location_masks(raw, stock)
        self.assertTrue(np.array_equal(periphery, stock))
        self.assertTrue(np.array_equal(proximity, stock))
        self.assertEqual(0, stats["relocated_deletions"])

    def test_exact_rank_on_small_masks_with_holes(self):
        rng = np.random.default_rng(419)
        for _ in range(40):
            raw = rng.random((7, 8)) > 0.22
            dc = int(rng.integers(1, 1 + raw.sum()))
            fg = {tuple(v) for v in np.argwhere(raw)}
            bg = {(r, c) for r in range(-1, 8) for c in range(-1, 9)} - fg
            d = {tuple(pts) for pts in rng.choice(np.asarray(sorted(fg)), dc, replace=False)}
            preset = raw.copy()
            for y, x in d:
                preset[y, x] = False
            preset[~raw] = rng.random(np.count_nonzero(~raw)) > 0.7

            def dist(p, seeds):
                return min((p[0] - y) ** 2 + (p[1] - x) ** 2 for y, x in seeds)

            p = set(sorted(fg, key=lambda q: (dist(q, bg), dist(q, d), q))[:dc])
            kept = p & d
            moved = d - kept
            other = kept | set(sorted(fg - d, key=lambda q: (dist(q, moved), q))[:len(moved)]) if moved else d
            a, b, stats = deletion_location_masks(raw, preset)
            self.assertEqual(p, {tuple(v) for v in np.argwhere(raw & ~a)})
            self.assertEqual(other, {tuple(v) for v in np.argwhere(raw & ~b)})
            self.assertTrue(np.array_equal(a & ~raw, preset & ~raw))
            self.assertTrue(np.array_equal(b & ~raw, preset & ~raw))
            self.assertEqual(len(moved), stats["relocated_deletions"])

if __name__ == "__main__":
    unittest.main()
