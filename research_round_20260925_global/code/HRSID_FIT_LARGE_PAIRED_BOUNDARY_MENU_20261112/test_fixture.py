import unittest
import numpy as np
from boundary_menu import proposal, oracle

class MenuFixture(unittest.TestCase):
    def test_equal_area_and_peer_vs_depth(self):
        anchor = np.zeros((9, 9), dtype=bool)
        anchor[2:7, 2:7] = True
        donor = anchor.copy()
        donor[:, 2] = False
        donor[2:7, 7] = True
        edit = np.ones_like(anchor)
        options = proposal(anchor, donor, edit)
        self.assertEqual(options["k"], 1)
        for key in ("peer", "contour"):
            self.assertEqual(options[key]["add"].size, 1)
            self.assertEqual(options[key]["trim"].size, 1)
        self.assertTrue(donor.ravel()[options["peer"]["add"][0]])
        self.assertFalse(donor.ravel()[options["peer"]["trim"][0]])
        self.assertFalse(anchor.ravel()[options["peer"]["add"][0]])
        self.assertTrue(anchor.ravel()[options["peer"]["trim"][0]])

    def test_equal_area_requires_both_sides(self):
        anchor = np.zeros((9, 9), dtype=bool)
        anchor[2:7, 2:7] = True
        donor = anchor.copy()
        donor[2:7, 7] = True
        self.assertEqual(proposal(anchor, donor, np.ones_like(anchor))["k"], 0)

    def test_constrained_selection(self):
        original = {"iou": .80, "core_recall": 1.0, "new_bridge_count": 0,
                    "max_fraction": 0., "splits": 0}
        add = dict(original, iou=.85, new_bridge_count=1)
        trim = dict(original, iou=.83, core_recall=.998)
        self.assertEqual(oracle(original, add, trim, False), "add")
        self.assertEqual(oracle(original, add, trim, True), "trim")
        self.assertEqual(oracle(original, original, original, True), "anchor")

if __name__ == "__main__":
    unittest.main()

