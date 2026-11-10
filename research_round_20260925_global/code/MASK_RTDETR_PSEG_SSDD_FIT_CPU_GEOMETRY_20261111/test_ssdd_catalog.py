import unittest
from ssdd_catalog import content_representatives


class SSDDScopeTests(unittest.TestCase):
    def test_duplicate_bytes_share_one_deterministic_representative(self):
        rows = [
            {"image_id":9,"original_image_ref":{"sha256":"a"}},
            {"image_id":7,"original_image_ref":{"sha256":"b"}},
            {"image_id":3,"original_image_ref":{"sha256":"a"}},
        ]
        representatives = content_representatives(rows)
        self.assertEqual(representatives["a"]["image_id"],3)
        self.assertEqual(representatives["b"]["image_id"],7)

