from io import BytesIO
from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


ROOT = Path(__file__).parent
M = run_path(str(ROOT / "common_menu.py"))
S = run_path(str(ROOT / "score_gate.py"))
R = run_path(str(ROOT / "render_gate.py"))


class MatchedSpatialGateTest(unittest.TestCase):
    def test_same_union_removes_menu_support_advantage_and_keeps_static_ties(self):
        gains = np.array([
            [[0., 0.], [0., 0.], [0., 0.]],
            [[1., 1.], [-1., -1.], [0., 0.]],
            [[2., 2.], [2., 1.], [-1., 1.]],
            [[2., 2.], [1., 1.], [-1., 1.]],
        ])
        expected = [2, 2, 0]
        self.assertEqual(M["shared_selection"](gains), expected)
        truth = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
        actions = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
        chosen = [1, 2, 0]
        own = [[0, 1], [0, 1, 2], [0, 3]]
        result = M["measure"](truth, actions, gains, chosen, own, 1)
        np.testing.assert_array_equal(result["selected_minimum_raw"], [-1, -1, 0])
        np.testing.assert_array_equal(result["shared_selected_minimum_raw"], [-1, -1, 0])
        np.testing.assert_array_equal(result["changed_own_to_union"], [True, False, False])

    def test_cue_admission_rejects_misregistered_points(self):
        ranks = np.array([2, 8], dtype="<i2")
        offsets = np.array([0, 2, 3], dtype="<i4")
        coords = np.array([[2, 4], [8, 3], [9, 1]], dtype="<i4")
        observed = np.arange(36, dtype="<f4").reshape(3, 2, 6)
        expected_sha = "b" * 64

        def packet(coordinates):
            stream = BytesIO()
            np.savez(stream, native_archive_ordinal=np.array(7, dtype="<i4"),
                     native_ranks=ranks, point_offsets=offsets, signed_coordinates=coordinates,
                     cue=observed, point_sha256=np.array(expected_sha, dtype="S64"))
            return stream.getvalue()

        actual = R["paired_cue"](packet(coords), 7, expected_sha, ranks, offsets, coords)
        np.testing.assert_array_equal(actual, observed)
        with self.assertRaisesRegex(ValueError, "coordinates"):
            R["paired_cue"](packet(coords[:, ::-1]), 7, expected_sha, ranks, offsets, coords)

    def test_main_and_spatial_specificity_must_both_pass(self):
        row = {"iou_points": .16, "percentile95_points": [.01, .3], "ap_points": -.09}
        keys = ("regression_robust", "scrambled_robust", "spatial_mean",
                "spatial_cell_pessimistic", "static")
        suites = {ref: {"spatial_robust_minus": {name: row.copy() for name in keys}}
                  for ref in ("pseg", "spsd")}
        exposure = dict(changed_common_hulls=20, changed_common_groups=8,
                        changed_common_hulls_vs_scrambled=20, changed_common_groups_vs_scrambled=8)
        self.assertTrue(S["gate"](suites, exposure)["advances"])
        suites["spsd"]["spatial_robust_minus"]["scrambled_robust"]["percentile95_points"] = [-.01, .3]
        result = S["gate"](suites, exposure)
        self.assertTrue(result["main_pass"])
        self.assertFalse(result["specificity_pass"])
        self.assertFalse(result["advances"])


if __name__ == "__main__":
    unittest.main()
