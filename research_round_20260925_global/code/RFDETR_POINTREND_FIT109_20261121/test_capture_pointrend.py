import unittest
from unittest.mock import Mock

import numpy as np
import torch

from capture_pointrend import eligible_arrays


class ArchiveAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.spot = Mock()
        self.spot.native_candidate_rows.return_value = [
            dict(rank=1, query_id=0, score=.9, rle=16),
            dict(rank=3, query_id=5, score=.8, rle=800),
            dict(rank=7, query_id=3, score=.4, rle=32),
        ]
        boxes = np.zeros((164, 4), dtype=np.float32)
        boxes[5] = [0., 300., 400., 800.]
        self.spot.author_boxes.return_value = boxes
        self.proof = [dict(logit_row=i, native_rank=r, query_id=q, score=s, predicted_area=a)
                      for i, (r,q,s,a) in enumerate([(3,5,.8,800), (7,3,.4,32)])]
        self.block = dict(image_id=42, raw_boxes_cxcywh=[], logit_capture=dict(candidates=self.proof))
        self.closed = np.zeros((2, 144, 144), dtype=np.float32)
        self.closed[1] = 4.
        self.observed = torch.zeros((164, 144, 144))
        self.observed[3] = 4.

    def get(self):
        return eligible_arrays(self.block, self.closed, self.observed, self.spot, int, np, torch)

    def test_rank_gaps_zero_boxes_and_exact_raw_queries_retained(self):
        arrays, mapped = self.get()
        self.assertEqual(mapped, 3)
        self.assertEqual(int(arrays["image_id"]), 42)
        np.testing.assert_array_equal(arrays["ranks"], [3,7])
        np.testing.assert_array_equal(arrays["query_ids"], [5,3])
        np.testing.assert_array_equal(arrays["areas"], [800,32])
        np.testing.assert_array_equal(arrays["scores"], [.8,.4])
        np.testing.assert_array_equal(arrays["boxes_xyxy"], [[0,.375,.5,1],[0,0,0,0]])
        np.testing.assert_array_equal(arrays["raw_logits"], self.closed)

    def test_stale_or_permuted_query_plane_rejected(self):
        self.observed[5,0,0] = 1.
        with self.assertRaisesRegex(ValueError, "logits"):
            self.get()

    def test_wrong_row_identity_rejected(self):
        self.proof[0]["query_id"] = 3
        with self.assertRaisesRegex(ValueError, "identity"):
            self.get()
