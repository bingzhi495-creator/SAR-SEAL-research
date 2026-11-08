"""Exercise the production native FIT path on bounded synthetic PNGs."""
from io import BytesIO
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_hrsid_fit_native_aug import NativeInputAudit


class NativeInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = NativeInputAudit()
        encoded = BytesIO()
        Image.new("RGB", (800, 800), (81, 152, 24)).save(encoded, format="PNG")
        cls.encoded = encoded.getvalue()

    def test_native_resize_retains_a_label_whose_mask_became_empty(self):
        tiny = dict(id=1, image_id=10, category_id=1, iscrowd=0,
                    bbox=[3, 3, 1, 1], area=1,
                    segmentation=[[3,3,4,3,4,4,3,4]])
        result = self.audit.observe(self.encoded, 10, [tiny], 520100003)
        self.assertEqual(result["branch"], "direct_resize")
        self.assertEqual(result["before"], {"labels": 1, "empty_masks": 0})
        self.assertEqual(result["after"], {"labels": 1, "empty_masks": 1})

    def test_replays_native_jitter_and_isolates_one_empty_source_instance(self):
        valid = dict(id=2, image_id=10, category_id=1, iscrowd=0,
                     bbox=[340, 340, 42, 55], area=2310,
                     segmentation=[[340,340,382,340,382,395,340,395]])
        degenerate = dict(id=1, image_id=10, category_id=1, iscrowd=0,
                          bbox=[370, 370, 3, 1], area=3,
                          segmentation=[[370,370,371,370,372,370,370,370]])
        a = self.audit.observe(self.encoded, 10, [valid], 520100000,
                               original_annotations=[degenerate, valid])
        b = self.audit.observe(self.encoded, 10, [valid], 520100000,
                               original_annotations=[degenerate, valid])
        self.assertEqual(a, b)
        self.assertEqual(a["before"], {"labels": 1, "empty_masks": 0})
        self.assertEqual(a["after"], {"labels": 1, "empty_masks": 0})
        self.assertTrue(a["paired_source"]["image_equal"])
        self.assertTrue(a["paired_source"]["shared_rows_equal"])
        self.assertEqual(a["paired_source"]["before"], {"labels": 2, "empty_masks": 1})
        self.assertEqual(a["paired_source"]["after"], {"labels": 2, "empty_masks": 1})


if __name__ == "__main__":
    unittest.main()

