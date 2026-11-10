"""Paired screen guard witnesses."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_fit_point_updates import compare_held


class FitPointUpdatesTest(unittest.TestCase):
    def cases(self, values, c=0.70):
        return {arm: [{"polygon_iou": p, "center_iou": c if arm != "phase" else c-0.001,
                       "raster_iou": 0.60, "ring_bce": 0.24} for p in data]
                for arm,data in values.items()}

    def test_gate_needs_both_controls_and_unchanged_reference_floor(self):
        x=self.cases({"anchor":[0.70]*8,"phase":[0.72]*8,"random":[0.71]*8,"uncertainty":[0.71]*8})
        self.assertTrue(compare_held(x)["descriptive_gate"])
        x["uncertainty"][0]["polygon_iou"]=0.79
        self.assertFalse(compare_held(x)["descriptive_gate"])


if __name__ == "__main__":
    unittest.main()
