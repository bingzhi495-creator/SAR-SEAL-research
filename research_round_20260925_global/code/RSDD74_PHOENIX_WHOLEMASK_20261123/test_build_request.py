"""Small synthetic checks of the label-free whole-mask request."""
import unittest

from build_request import request_rows


class WholeRequestTests(unittest.TestCase):
    def test_retains_original_order_and_empty_mask(self):
        empty = {"size": [512, 512], "counts": "PPP8"}
        other = {"size": [512, 512], "counts": "01ooo7"}
        old_image = {"image_id": 3, "image_file": "image_001.png", "decoded_rgb_sha256": "x",
                     "encoded_sha256": "y", "candidates": []}
        stock = {"image_id": 3, "native_count": 3, "unmapped_slot1_count": 1, "coco": [
            {"image_id": 3, "score": .9, "segmentation": other},
            {"image_id": 3, "score": .8, "segmentation": empty}]}
        rows = request_rows([old_image], [stock])
        self.assertEqual(rows[0]["candidates"],
                         [{"position": 1, "segmentation": other}, {"position": 2, "segmentation": empty}])
        self.assertNotIn("image_id", rows[0])
        self.assertEqual(rows[0]["image_file"], "image_001.png")


if __name__ == "__main__":
    unittest.main()
