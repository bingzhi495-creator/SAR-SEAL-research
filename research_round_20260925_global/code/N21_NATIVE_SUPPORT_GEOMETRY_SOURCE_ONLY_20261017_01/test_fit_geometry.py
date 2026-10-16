"""UNIT_FIXTURE only. No real FIT metadata, scientific Source or model is read."""
import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

module_spec = importlib.util.spec_from_file_location(
    "unit_fixture_fit_geometry", Path(__file__).with_name("fit_geometry.py")
)
geometry = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(geometry)


def row(key, shape=(6, 8), **extra):
    return dict(image_key=key, native_shape=shape, canvas_shape=shape,
                crop_shape=shape, image_shape=shape, **extra)


class FitGeometryTests(unittest.TestCase):
    def test_image_selection_has_fixed_hash_order_and_preserves_metadata(self):
        rows = [row("alpha", dataset_index=1), row("beta", dataset_index=2),
                row("gamma", dataset_index=3)]
        before = copy.deepcopy(rows)
        selected = geometry.select_rows(rows, count=2)
        self.assertEqual([r["image_key"] for r in selected], ["gamma", "alpha"])
        self.assertEqual([r["dataset_index"] for r in selected], [3, 1])
        self.assertEqual(selected, geometry.select_rows(list(reversed(rows)), count=2))
        self.assertEqual(rows, before)

    def test_selection_rejects_invalid_rows_instead_of_filtering_them(self):
        with self.assertRaises(ValueError):
            geometry.select_rows([row("x")] * 32)
        with self.assertRaises(ValueError):
            geometry.select_rows([row(str(i)) for i in range(31)])
        missing = row("bad"); del missing["image_key"]
        with self.assertRaises(ValueError):
            geometry.select_rows([row("good"), missing], count=1)
        illegal = row("bad"); illegal["crop_shape"] = (7, 8)
        with self.assertRaises(ValueError):
            geometry.select_rows([row("good"), illegal], count=1)
        illegal = row("bad"); illegal["native_shape"] = (6.0, 8)
        with self.assertRaises(ValueError):
            geometry.select_rows([illegal], count=1)

    def test_center_hash_order_is_deterministic_without_replacement(self):
        selected = row("center-fixture", shape=(3, 4))
        self.assertEqual(geometry.unique_centers(selected, 202610170301, cap=5),
                         [(2, 2), (0, 1), (1, 0), (1, 2), (1, 3)])
        centers = geometry.unique_centers(selected, 202610170301)
        self.assertEqual(len(centers), 12)
        self.assertEqual(len(set(centers)), 12)
        self.assertEqual(centers, geometry.unique_centers(selected, 202610170301))
        self.assertEqual(geometry.unique_centers(row("one", (1, 1)), 3), [(0, 0)])
        with self.assertRaises(ValueError):
            geometry.unique_centers(selected, 3, cap=65)

    def test_center_candidate_exhaustion_hard_fails_without_seed_retry(self):
        class ZeroDigest:
            def digest(self):
                return bytes(32)
        with patch.object(geometry.hashlib, "sha256", return_value=ZeroDigest()) as digest:
            with self.assertRaises(RuntimeError):
                geometry.unique_centers(row("collision", (1, 2)), 7, cap=2)
        self.assertEqual(digest.call_count, 20000)

    def test_summary_includes_exact_half_coverage_and_serializable_supports(self):
        # Identity 6x8, all48 centers: exactly the24 interior centers are unclamped.
        result = geometry.analyze_rows([row("half", dataset_index=8)], 202610170301)
        self.assertEqual(set(result), {"cases", "summary", "decision"})
        summary = result["summary"]
        self.assertEqual(summary["all"]["denominator"], 48)
        self.assertEqual(summary["no_any_clamp"]["denominator"], 24)
        self.assertEqual(summary["clamped"]["denominator"], 24)
        self.assertEqual(summary["all"]["rank_histogram"], {"4": 4, "6": 20, "9": 24})
        self.assertEqual(summary["no_any_clamp"]["same_cell_count"], 0)
        self.assertEqual(result["decision"], "SHARED_FOUR_PREMISE_REJECTED")
        self.assertEqual(result["cases"][0]["row"]["dataset_index"], 8)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        for case in result["cases"]:
            for group in case["groups"]:
                self.assertEqual(len(group["supports"]), 9)
                self.assertEqual(len(group["clamp_flags"]), 9)
                for support in group["supports"]:
                    self.assertEqual(sum(Fraction(v["numerator"], v["denominator"])
                                         for v in support), Fraction(1))

    def test_coverage_below_half_is_inconclusive(self):
        result = geometry.analyze_rows([row("below-half", shape=(6, 6))], 202610170301)
        self.assertEqual(result["summary"]["all"]["denominator"], 36)
        self.assertEqual(result["summary"]["no_any_clamp"]["denominator"], 16)
        self.assertEqual(result["decision"], "INCONCLUSIVE_CLAMP_COVERAGE")

    def test_same_cell_four_fifths_boundary_uses_main_stratum_only(self):
        # Large integer shapes are tiny metadata fixtures: no image array exists.
        # Fixed64 centers per row are asserted exactly, never accepted as "likely".
        large = 10 ** 18
        def positive(key):
            value = row(key, shape=(large, large))
            value["native_shape"] = (10 ** 9, 10 ** 9)
            return value
        selected = [positive("support-" + str(i)) for i in range(4)]
        selected.append(row("identity-4", shape=(large, large)))
        result = geometry.analyze_rows(selected, 202610170301)
        main = result["summary"]["no_any_clamp"]
        self.assertEqual(result["summary"]["all"]["denominator"], 320)
        self.assertEqual(main["denominator"], 320)
        self.assertEqual(main["same_cell_count"], 256)
        self.assertEqual(result["decision"], "GEOMETRY_PREMISE_SUPPORTED")
        selected[3] = row("identity-3", shape=(large, large))
        below = geometry.analyze_rows(selected, 202610170301)
        self.assertEqual(below["summary"]["no_any_clamp"]["denominator"], 320)
        self.assertEqual(below["summary"]["no_any_clamp"]["same_cell_count"], 192)
        self.assertEqual(below["decision"], "SHARED_FOUR_PREMISE_REJECTED")

    def test_clamped_same_cells_never_enter_the_main_numerator(self):
        large = 10 ** 18
        clamped = row("all-clamped", shape=(large, large))
        clamped["native_shape"] = (1, 1)
        result = geometry.analyze_rows([clamped, row("all-main", shape=(large, large))],
                                       202610170301)
        self.assertEqual(result["summary"]["all"]["denominator"], 128)
        self.assertEqual(result["summary"]["all"]["same_cell_count"], 64)
        self.assertEqual(result["summary"]["no_any_clamp"]["denominator"], 64)
        self.assertEqual(result["summary"]["no_any_clamp"]["same_cell_count"], 0)
        self.assertEqual(result["decision"], "SHARED_FOUR_PREMISE_REJECTED")


if __name__ == "__main__":
    unittest.main()
