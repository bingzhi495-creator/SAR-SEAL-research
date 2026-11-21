"""Synthetic readback tests. All masks, point coordinates and edit lists are generated here."""
from copy import deepcopy
import unittest

import numpy as np
from pycocotools import mask

from verify_point_fit109 import rebuild_candidate


def rle(values):
    obj = mask.encode(np.asfortranarray(values, dtype=np.uint8))
    return {"size": obj["size"], "counts": obj["counts"].decode("ascii")}


def diagnostic(adds, drops):
    return {"adds": adds, "drops": drops, "area_delta": adds - drops, "nonpositive_flips": 0}


class RebuildCandidateTests(unittest.TestCase):
    def setUp(self):
        # Rectangular shape and asymmetric coordinates distinguish C-order pixel IDs
        # used by the edit witness from Fortran-order storage in the compressed RLE.
        self.stock = np.zeros((7, 9), dtype=np.uint8)
        self.stock.flat[[1, 7, 9]] = 1
        self.stock.flat[20:50] = 1
        self.coords = np.array([[2, 1], [8, 7]], dtype=np.int64)
        self.coarse = self.stock.copy()
        self.coarse.flat[[0, 9]] ^= 1
        self.rendered = self.stock.copy()
        self.rendered.flat[[4, 9, 10]] ^= 1
        self.witness = {
            "key": [10, 3, 27], "C": rle(self.coarse), "P": rle(self.rendered),
            "F": [2, 1], "B": [8, 1], "J": [2, 8],
            "diagnostics": {"F": diagnostic(1, 1), "B": diagnostic(1, 1), "J": diagnostic(2, 0)},
        }

    def test_dense_reconstruction_and_signed_budget(self):
        masks, counts = rebuild_candidate(rle(self.stock), self.witness, self.coords, 1)
        self.assertEqual(set(masks), {"N", "C", "P", "F", "B", "J"})
        source = self.stock
        for arm, expected in (("N", source), ("C", self.coarse), ("P", self.rendered)):
            np.testing.assert_array_equal(mask.decode(masks[arm]), expected)
        for arm in ("F", "B", "J"):
            expected = source.copy()
            expected.flat[self.witness[arm]] ^= 1
            np.testing.assert_array_equal(mask.decode(masks[arm]), expected)
            add = int(np.count_nonzero((expected == 1) & (source == 0)))
            drop = int(np.count_nonzero((source == 1) & (expected == 0)))
            self.assertEqual((counts[arm]["adds"], counts[arm]["drops"], counts[arm]["area_delta"]),
                             (add, drop, add - drop))
            self.assertEqual(add + drop, 2)
        self.assertEqual(counts["J"]["area_delta"], 2)
        self.assertEqual(counts["B"]["area_delta"], 0)

    def test_zero_budget_keeps_natural_render_and_empty_edit_policies(self):
        witness = deepcopy(self.witness)
        for arm in ("F", "B", "J"):
            witness[arm] = []
            witness["diagnostics"][arm] = diagnostic(0, 0)
        masks, counts = rebuild_candidate(rle(self.stock), witness, np.empty((0, 2), np.int64), 0)
        np.testing.assert_array_equal(mask.decode(masks["C"]), self.coarse)
        np.testing.assert_array_equal(mask.decode(masks["P"]), self.rendered)
        for arm in ("F", "B", "J"):
            np.testing.assert_array_equal(mask.decode(masks[arm]), self.stock)
            self.assertEqual((counts[arm]["adds"], counts[arm]["drops"]), (0, 0))

    def test_invalid_support_and_inconsistent_observed_direction(self):
        for arm, flips, counts in (("F", [2, 2], diagnostic(2, 0)),
                                   ("F", [2, 7], diagnostic(1, 1)),
                                   ("J", [2, 4], diagnostic(2, 0)),
                                   ("B", [2, 8], diagnostic(2, 0)),
                                   ("J", [2, 8], diagnostic(0, 2))):
            witness = deepcopy(self.witness)
            witness[arm] = flips
            witness["diagnostics"][arm] = counts
            with self.subTest(arm=arm, flips=flips, counts=counts), self.assertRaises(ValueError):
                rebuild_candidate(rle(self.stock), witness, self.coords, 1)


if __name__ == "__main__":
    unittest.main()
