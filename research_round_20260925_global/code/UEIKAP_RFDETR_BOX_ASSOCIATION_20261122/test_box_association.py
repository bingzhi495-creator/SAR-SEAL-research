"""Synthetic association tests. No experimental data are needed."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

spec = spec_from_file_location("test_box_association_impl", Path(__file__).with_name("box_association.py"))
box = module_from_spec(spec)
spec.loader.exec_module(box)


class BoxAssociation(unittest.TestCase):
    def test_threshold_and_tie_order_without_mask_input(self):
        labels = [(0, 0, 10, 10), (0, 0, 10, 10), (30, 30, 40, 40)]
        candidates = [(0, 0, 10, 5), (0, 0, 10, 10), (30, 30, 40, 34.999)]
        self.assertEqual(box.match_box(candidates, labels), {0:1, 1:2})

    def test_degenerate_box_and_empty_image(self):
        self.assertEqual(box.box_iou((2, 3, 2, 9), (0, 0, 10, 10)), 0)
        self.assertEqual(box.match_box([], [(0, 0, 10, 10)]), {})
        self.assertEqual(box.match_box([(0, 0, 10, 10)], []), {})

    def test_only_selected_original_candidate_defines_weak_mask(self):
        import numpy as np
        raster_iou = np.array([[.48, .51], [.68, .49]])
        self.assertEqual(box.weak_hulls({0:1, 1:2}, raster_iou), {0, 1})
        self.assertEqual(box.weak_hulls({0:2, 1:1}, raster_iou), set())


if __name__ == "__main__":
    unittest.main()
