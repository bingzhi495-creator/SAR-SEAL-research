import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("cohort", Path(__file__).with_name("select_cohort.py"))
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


class CohortTest(unittest.TestCase):
    def test_midpoints_and_separation(self):
        self.assertEqual(cohort.midpoint_indices(9, 4), [1, 3, 5, 7])
        self.assertEqual(cohort.midpoint_indices(3, 64), [0, 1, 2])
        make = lambda i, lon, ipr: {
            "collect_id": str(i), "center_lon_lat": [lon, 0],
            "target_ipr_m": ipr, "footprint_lon_lat": [[[lon-.01, -.01], [lon+.01, -.01],
                [lon+.01, .01], [lon-.01, .01], [lon-.01, -.01]]],
        }
        candidates = [make(1, 10, .5), make(2, 0, 1.0), make(3, 3, 2.0),
                      make(4, 3.2, .35), make(5, 6, .5), make(6, 8, 2.0)]
        admitted, decisions = cohort.choose(candidates, [[0.1, 0]], quota=2)
        self.assertEqual([row["collect_id"] for row in admitted], ["3", "6", "1", "5"])
        self.assertEqual([row["window_id"] for row in admitted],
                         ["UMB-001", "UMB-002", "UMB-003", "UMB-004"])
        self.assertEqual(decisions["within_100km_historical"], 1)
        self.assertEqual(decisions["within_100km_selected"], 1)

    def test_footprint_check(self):
        good = {"type":"Polygon","coordinates":[[[1,1],[2,1],[2,2],[1,2],[1,1]]]}
        bad = {"type":"Polygon","coordinates":[[[1,1],[2,2],[2,1],[1,2],[1,1]]]}
        self.assertTrue(cohort.checked_polygon(good).is_valid)
        for other in [bad, {"type":"Point","coordinates":[1,1]}, {"type":"Polygon","coordinates":[[[181,1],[182,1],[182,2],[181,1]]]}]:
            with self.assertRaises(ValueError):
                cohort.checked_polygon(other)


if __name__ == "__main__":
    unittest.main()

