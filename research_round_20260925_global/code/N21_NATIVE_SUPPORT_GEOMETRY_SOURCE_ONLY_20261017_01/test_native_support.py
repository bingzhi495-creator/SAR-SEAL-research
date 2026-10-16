"""UNIT_FIXTURE only: exact geometry, no data, model or execution qualification."""
import importlib.util
from fractions import Fraction as F
from pathlib import Path
import unittest

module_spec = importlib.util.spec_from_file_location(
    "unit_fixture_native_support", Path(__file__).with_name("native_support.py")
)
geometry = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(geometry)


class NativeSupportTests(unittest.TestCase):
    def test_single_resize_degenerate_composition(self):
        # The first resize and crop are identities; 2 -> 4 at x=1 gives x=1/4.
        actual = geometry.point_support((1, 2), (1, 2), (1, 2), (1, 4), (0, 1))
        self.assertEqual(actual, {(0, 0): F(3, 4), (0, 1): F(1, 4)})

    def test_same_native_cell_after_two_resizes_and_actual_crop(self):
        # In 1D, the three image positions have weights (.9,.1),
        # (.75,.25), (.45,.55); the 2D center is their middle outer product.
        result = geometry.trace_neighborhood((2, 2), (4, 4), (3, 3), (5, 5), (2, 2))
        self.assertEqual(result["supports"][4], {
            (0, 0): F(9, 16), (0, 1): F(3, 16),
            (1, 0): F(3, 16), (1, 1): F(1, 16),
        })
        self.assertEqual(result["rank"], 4)
        self.assertEqual(result["union_support_count"], 4)
        self.assertTrue(result["same_native_2x2_cell"])
        self.assertEqual(len(result["coefficient_matrix"]), 9)
        for sparse in result["supports"]:
            self.assertEqual(sum(sparse.values()), F(1))
            self.assertTrue(all(type(v) is F and v > 0 for v in sparse.values()))

    def test_two_resizes_mix_intermediate_nodes_before_native_support(self):
        # 3 -> 4 -> crop3 -> 2. The first output mixes canvas0,1 as 3/4,1/4;
        # the second mixes canvas1,2 as 1/4,3/4. This is not one collapsed resize.
        arguments = ((1, 3), (1, 4), (1, 3), (1, 2))
        self.assertEqual(geometry.point_support(*arguments, (0, 0)), {
            (0, 0): F(27, 32), (0, 1): F(5, 32),
        })
        self.assertEqual(geometry.point_support(*arguments, (0, 1)), {
            (0, 0): F(3, 32), (0, 1): F(5, 8), (0, 2): F(9, 32),
        })

    def test_cross_native_cells_has_nine_independent_coefficient_rows(self):
        result = geometry.trace_neighborhood((3, 3), (6, 6), (6, 6), (3, 3), (1, 1))
        # Independent hand-derived 1D operator; the 2D operator is B tensor B.
        axis = ((F(7, 8), F(1, 8), F(0)),
                (F(1, 8), F(6, 8), F(1, 8)),
                (F(0), F(1, 8), F(7, 8)))
        expected = [[a[i] * b[j] for i in range(3) for j in range(3)]
                    for a in axis for b in axis]
        self.assertEqual(result["coefficient_matrix"], expected)
        self.assertEqual(result["rank"], 9)
        self.assertEqual(result["union_support_count"], 9)
        self.assertFalse(result["same_native_2x2_cell"])

    def test_image_and_resize_clamps_can_degenerate_to_one_native_node(self):
        result = geometry.trace_neighborhood((2, 2), (6, 6), (5, 5), (10, 10), (0, 0))
        self.assertEqual(result["supports"], [{(0, 0): F(1)}] * 9)
        self.assertEqual(result["coefficient_matrix"], [[F(1)]] * 9)
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["union_support_count"], 1)
        self.assertTrue(result["same_native_2x2_cell"])
        self.assertEqual([f["neighborhood"] for f in result["clamp_flags"]],
                         [True, True, True, True, False, False, True, False, False])
        self.assertEqual(result["clamp_flags"][4], {
            "neighborhood": False, "native_resize": True, "image_resize": True,
        })
        self.assertFalse(result["clamp_flags"][8]["image_resize"])
        self.assertTrue(result["any_clamped"])

    def test_identity_geometry_keeps_nine_nodes_without_falsely_flagging_clamp(self):
        result = geometry.trace_neighborhood((3, 3), (3, 3), (3, 3), (3, 3), (1, 1))
        expected_points = [(y, x) for y in range(3) for x in range(3)]
        self.assertEqual(result["points"], expected_points)
        self.assertEqual(result["supports"], [{p: F(1)} for p in expected_points])
        self.assertEqual(result["rank"], 9)
        self.assertFalse(result["any_clamped"])

    def test_rank_distinguishes_exact_zero_from_arbitrarily_small_rational(self):
        self.assertEqual(geometry.matrix_rank([[1, 1], [1, 1 + F(1, 10 ** 80)]]), 2)
        self.assertEqual(geometry.matrix_rank([[1, 2], [2, 4]]), 1)
        self.assertEqual(geometry.matrix_rank([[], []]), 0)
        with self.assertRaises(ValueError):
            geometry.matrix_rank([[0.0]])
        with self.assertRaises(ValueError):
            geometry.matrix_rank([[1], [1, 2]])

    def test_only_explicit_valid_contract_is_accepted(self):
        with self.assertRaises(ValueError):
            geometry.point_support((2, 2), (3, 3), (4, 3), (5, 5), (1, 1))
        with self.assertRaises(ValueError):
            geometry.trace_neighborhood((2, 2), (3, 3), (3, 3), (5, 5), (-1, 0))
        with self.assertRaises(ValueError):
            geometry.point_support((2.0, 2), (3, 3), (3, 3), (5, 5), (1, 1))


if __name__ == "__main__":
    unittest.main()
