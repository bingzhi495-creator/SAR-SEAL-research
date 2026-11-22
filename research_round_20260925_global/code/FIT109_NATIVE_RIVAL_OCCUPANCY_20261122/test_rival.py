import unittest
from pathlib import Path
from runpy import run_path

import numpy as np
from pycocotools import mask as codec


rival = run_path(str(Path(__file__).with_name("rival.py")))
pilot = run_path(str(Path(__file__).with_name("run_rival_pilot.py")))


def row(cells, score):
    mask = np.zeros((5, 8), dtype=np.uint8, order="F")
    for y, x in cells:
        mask[y, x] = 1
    return {"segmentation": codec.encode(mask), "score": score}


class RivalNativeOccupancy(unittest.TestCase):
    def test_scores_use_other_nonduplicate_native_masks_only(self):
        rows = [
            row([(1, 1), (1, 2), (1, 3), (2, 1)], 0.99),
            row([(1, 1), (1, 2), (2, 2)], 0.96),
            row([(2, 2), (2, 3), (2, 4)], 0.73),
            row([(2, 3), (3, 3)], 0.81),
            row([(2, 4), (2, 5)], 0.49),
        ]
        scene = rival["NativeRivals"](rows)
        p = scene.penalties(0, np.array([18, 19, 20, 21], dtype=np.int32))
        np.testing.assert_array_equal(p, [0.73, 0.81, 0.73, 0.0])

    def test_rejects_original_foreground_and_duplicate_cells(self):
        scene = rival["NativeRivals"]([row([(1, 1)], 0.4)])
        for coordinates in ([9], [1, 1], [40], [-1]):
            with self.assertRaises(ValueError):
                scene.penalties(0, np.array(coordinates))

    def test_placebo_preserves_penalty_multiset_in_each_probability_quartile(self):
        q = np.array([0.2, 0.8, 0.1, 0.1, 0.8, 0.15, 0.95, 0.9, 0.9, 0.2, 0.7, 0.4])
        penalty = np.arange(len(q), dtype=np.float64) / len(q)
        permute = rival["placebo_penalties"]
        shuffled = permute(q, penalty, 23, 1)
        np.testing.assert_array_equal(shuffled, permute(q, penalty, 23, 1))
        self.assertFalse(np.array_equal(shuffled, penalty))
        order = np.argsort(q, kind="stable")
        for block in np.array_split(order, 4):
            np.testing.assert_array_equal(np.sort(shuffled[block]), np.sort(penalty[block]))

    def test_real_penalty_reroutes_additions_with_same_count_and_deletions(self):
        def choose(x, cap, fixed_count):
            self.assertEqual(cap, 2)
            self.assertEqual(fixed_count, 1)
            np.testing.assert_array_equal(x[:, 1], [0.4, 0.5, 0.3])
            return [(int(x[:, 0].argmax()), int(x[:, 1].argmin()))]

        p = np.array([[0.9, 0.4], [0.8, 0.5], [0.7, 0.3]])
        penalty = np.array([0.8, 0.0, 0.0])
        self.assertEqual(rival["rank_additions"](p, penalty, 2, 1, choose), [(1, 2)])
        self.assertEqual(rival["rank_additions"](p, np.zeros(3), 2, 1, choose), [(0, 2)])
        self.assertEqual(float(p[0, 0]), 0.9)

    def test_gate_needs_coverage_and_both_controls(self):
        independent, true, placebo = (pilot[k] for k in ("INDEPENDENT", "TRUE", "PLACEBO"))
        summarize = pilot["near_contrasts"]
        keys = ("true_changed", "placebo_changed", "placebo_displaced", "true_placebo_changed")
        rows = [dict(group=i % 8, exposed=True, arms={independent: (1, 50), true: (0.8, 50),
                     placebo: (0.9, 50)}, **{name: True for name in keys}) for i in range(12)]
        coco = {arm: {"AP": 50} for arm in (independent, true, placebo)}
        self.assertFalse(summarize(rows[:11], coco)["inference_enabled"])
        self.assertTrue(summarize(rows, coco)["advances"])
        coco[true]["AP"] = 49.69
        self.assertFalse(summarize(rows, coco)["advances"])

    def test_rejects_old_slot_factory_without_matched_identity(self):
        def old():
            return dict(area=32, role="valid", truth=None, raw=None)

        def patched():
            return dict(area=32, role="valid", truth=None, raw=None, gt_id=7)

        require = pilot["require_match_identity"]
        with self.assertRaises(ValueError):
            require(old)
        require(patched)


if __name__ == "__main__":
    unittest.main()
