"""Generated fixtures for topology controls, signed changes and group contrasts."""

import os
from pathlib import Path
import unittest

import numpy as np

import pilot


class TopologyPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = Path(os.environ.get("PILOT_RELEASE_CODE", pilot.CODE))
        free_path, free_digest = pilot.REFERENCES["free"]
        cls.free = pilot.module(code / pilot.E24 / free_path, free_digest)
        from importlib.util import module_from_spec, spec_from_file_location
        helper = code / "FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/simple_exchange.py"
        spec = spec_from_file_location("_generated_topology", helper)
        cls.topology = module_from_spec(spec)
        spec.loader.exec_module(cls.topology)

    @staticmethod
    def fixture():
        stock = np.zeros((11, 13), dtype=bool)
        stock[3:8, 3:8] = True
        adds = [(1, 0), (5, 2), (8, 5), (0, 12)]
        drops = [(5, 5), (3, 6), (5, 7), (4, 5)]
        coords = np.array([[y*13+x, dy*13+dx] for (y,x), (dy,dx) in zip(adds, drops)])
        p = np.array([[.98, .05], [.90, .30], [.90, .30], [.97, .10]])
        layers = np.array([2, 1, 1, 1])
        return stock, coords, p, layers

    def test_realized_count_controls_stable_scores_and_layer_priority(self):
        stock, coords, p, layers = self.fixture()
        before = [a.copy() for a in (stock, coords, p, layers)]
        natural = self.free.choose_free(p, 4)
        self.assertEqual(len(natural), 4)
        actions = pilot.choose_controls(self.free, self.topology, stock, coords, p, layers, len(natural))
        self.assertEqual(actions, {
            "simple": [(1, 1), (2, 2)],
            "free_at_simple_count": [(0, 0), (3, 3)],
            "layer1_at_simple_count": [(0, 3), (3, 1)],
        })
        top = pilot.topology_changes(stock, coords, actions)
        self.assertEqual(top["simple"], (0, 0))
        self.assertNotEqual(top["free_at_simple_count"], (0, 0))
        for original, arg in zip(before, (stock, coords, p, layers)):
            np.testing.assert_array_equal(original, arg)
        for pairs in actions.values():
            new = stock.copy().ravel()
            for a, b in pairs:
                new[coords[a, 0]] = True
                new[coords[b, 1]] = False
            self.assertEqual(new.sum(), stock.sum())

    def test_infeasible_positive_and_zero_count(self):
        stock, coords, p, layers = self.fixture()
        # The highest-score add is isolated. Dropping the interior pixel would open a hole.
        for cap in (0, 1):
            actions = pilot.choose_controls(self.free, self.topology, stock, coords[:1], p[:1], layers[:1], cap)
            self.assertEqual(actions, {name: [] for name in pilot.NEW})

    def test_invalid_removal_layer_rejected(self):
        stock, coords, p, layers = self.fixture()
        layers[0] = 3
        with self.assertRaises(ValueError):
            pilot.choose_controls(self.free, self.topology, stock, coords, p, layers, 2)

    def test_signed_global_components_holes_and_inverse(self):
        stock = np.zeros((8, 9), dtype=bool)
        stock[1:6, 1:6] = True
        center = 3 * 9 + 3
        apart, diagonal = 4 * 9 + 8, 6 * 9 + 6
        coords = np.array([[apart, center], [diagonal, center]])
        actions = {"apart": [(0, 0)], "same": [(0, 0)], "diagonal": [(1, 1)], "raw": []}
        self.assertEqual(pilot.topology_changes(stock, coords, actions),
                         {"apart": (1, 1), "same": (1, 1), "diagonal": (0, 1), "raw": (0, 0)})
        changed = stock.copy()
        changed.flat[apart], changed.flat[center] = True, False
        self.assertEqual(pilot.topology_changes(changed, np.array([[center, apart]]), {"inverse": [(0, 0)]}),
                         {"inverse": (-1, -1)})

    def test_four_connected_background_and_exterior_padding(self):
        stock = np.ones((4, 4), dtype=bool)
        stock[0, 0] = False
        stock[3, 3] = False
        # The new center background pixel touches the open image edge only at a diagonal.
        coords = np.array([[15, 5]])
        self.assertEqual(pilot.topology_changes(stock, coords, {"one": [(0, 0)]}), {"one": (0, 1)})

    def test_active_geometry_uses_group_means_and_conditional_denominator(self):
        slots = [dict(group=g, area=10, raw=(10, 5), truth=np.array([[q, 0.], [0., 0.]]))
                 for g, q in (("a", 1.), ("a", 1.), ("b", 0.), ("c", 1.))]
        acts = {arm: [[], [], [], []] for arm in pilot.MODES}
        for arm in pilot.NEW:
            acts[arm][:3] = [[(0, 0)] if arm == pilot.SIMPLE else [(1, 1)] for _ in range(3)]
        overlap, conditional = pilot.additional_geometry(slots, acts)
        self.assertEqual((conditional["source_groups"], conditional["pairs"]), (2, 3))
        self.assertEqual(overlap[pilot.SIMPLE]["net_overlap"], 2.)
        expected = (100.*6/14 - 100.*5/15) / 2
        for arm in pilot.CONTROLS:
            self.assertAlmostEqual(conditional["simple_minus"][arm]["delta_points"], expected)
            np.testing.assert_allclose(conditional["simple_minus"][arm]["interval95_points"], [0., 2*expected])


if __name__ == "__main__":
    unittest.main()
