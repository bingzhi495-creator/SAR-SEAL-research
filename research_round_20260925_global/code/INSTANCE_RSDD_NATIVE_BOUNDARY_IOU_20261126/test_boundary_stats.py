from unittest import TestCase

import numpy as np

import boundary_metric
from boundary_stats import Audit


class AuditTest(TestCase):
    def test_group_balancing_saturation_and_all_reference_denominator(self):
        audit = Audit(boundary_metric, ("better", "worse"), {"test": (("better", "worse"),)})
        ref = np.zeros((7, 7), dtype=bool)
        ref[2:5, 2:5] = True
        for group, worse in (("A", np.zeros_like(ref)), ("A", ref), ("B", ref)):
            audit.add(group, ref, ref, {"better": ref, "worse": worse})
        out = audit.report({"A": 4, "B": 1, "C": 2})
        first, second = out["widths"]["1"], out["widths"]["2"]
        self.assertAlmostEqual(first["contrasts"]["test"]["mean_points"], 25.0)
        self.assertAlmostEqual(first["all_author_zero_delta"]["contrasts"]["test"]["mean_points"], 100 / 12)
        self.assertEqual(first["saturation"]["reference_whole_mask"], 0)
        self.assertEqual(second["saturation"]["reference_whole_mask"], 3)
        self.assertEqual(first["native_reference_non_saturated"]["anchors"], 3)
        self.assertEqual(second["native_reference_non_saturated"]["anchors"], 0)
        self.assertEqual(first["mean_boundary_iou_points"]["worse"], 75)
