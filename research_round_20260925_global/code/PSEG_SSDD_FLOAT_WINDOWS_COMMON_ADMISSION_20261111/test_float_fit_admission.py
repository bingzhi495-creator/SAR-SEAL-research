from copy import deepcopy
import unittest
from float_fit_admission import validate_polygon_only

class FloatOverlayTest(unittest.TestCase):
    def test_only_ordered_polygon_values_may_change(self):
        integer = {"images": [{"id": 7}], "categories": [{"id": 1}],
                   "annotations": [{"id": 8, "image_id": 7, "area": 4, "segmentation": [[0, 0, 1, 1, 1, 0]]}]}
        overlay = deepcopy(integer)
        overlay["annotations"][0]["segmentation"][0][0] = .25
        validate_polygon_only(integer, overlay)
        overlay["annotations"][0]["area"] = 5
        with self.assertRaises(ValueError):
            validate_polygon_only(integer, overlay)

if __name__ == "__main__":
    unittest.main()