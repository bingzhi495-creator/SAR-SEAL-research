"""Synthetic wire-format contract for original order and strict p > .5."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as codec

sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_capture import candidate_record


class NativeCaptureTests(unittest.TestCase):
    def test_strict_native_rle_and_clipped_author_corner_extent(self):
        probabilities=np.zeros((512,512),dtype=np.float32)
        probabilities[50,100]=.5
        probabilities[51,100]=.75
        original=probabilities > .5
        box=np.array([2,6,10,14,.125,.75],dtype=np.float32)
        corners=np.array([[-7,6],[8,-2],[515,307],[3,9]],dtype=np.float32)
        record=candidate_record(1,box,corners,original,probabilities,codec)
        self.assertEqual(record["rank"],1)
        self.assertEqual(record["category_id"],1)
        self.assertEqual(record["score"],.75)
        self.assertEqual(record["rbox_xywha"],[2,6,10,14,.125])
        self.assertEqual(record["clipped_corner_xyxy"],[0,0,512,307])
        self.assertEqual(record["rle"]["size"],[512,512])
        self.assertIsInstance(record["rle"]["counts"],str)
        np.testing.assert_array_equal(codec.decode(record["rle"]).astype(bool),original)
        with self.assertRaisesRegex(ValueError,"strict native threshold"):
            candidate_record(1,box,corners,probabilities>=.5,probabilities,codec)


if __name__=="__main__":
    unittest.main()
