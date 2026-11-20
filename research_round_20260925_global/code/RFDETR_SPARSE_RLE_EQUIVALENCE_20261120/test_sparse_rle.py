"""Sparse compressed COCO edits must match the authoritative raster codec."""
import unittest

import numpy as np
from pycocotools import mask as codec

from sparse_rle import xor_rle


def encoded(mask):
    rle=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(size=rle["size"],counts=rle["counts"].decode("ascii"))


class SparseRLETest(unittest.TestCase):

    def test_all_tiny_masks_and_flip_subsets(self):
        for bits in range(64):
            mask=np.asarray([(bits>>i)&1 for i in range(6)],dtype=np.uint8).reshape(2,3)
            source=encoded(mask)
            for flips in range(64):
                pixels=[i for i in range(6) if (flips>>i)&1]
                expected=mask.copy()
                expected.reshape(-1)[pixels]^=1
                self.assertEqual(xor_rle(source,pixels),encoded(expected))

    def test_random_masks_and_adjacent_boundaries(self):
        rng=np.random.default_rng(20261120)
        for shape in ((1,1),(3,2),(17,19),(73,91)):
            for _ in range(40):
                mask=(rng.random(shape)<rng.uniform(.01,.99)).astype(np.uint8)
                n=min(mask.size,16)
                pixels=rng.choice(mask.size,size=n,replace=False)
                expected=mask.copy()
                expected.reshape(-1)[pixels]^=1
                self.assertEqual(xor_rle(encoded(mask),pixels),encoded(expected))
        for mask in (np.zeros((3,4),dtype=np.uint8),np.ones((3,4),dtype=np.uint8),
                     np.asarray([[1,0,1,0],[1,0,1,0],[0,0,1,0]],dtype=np.uint8)):
            for pixels in ([0],[0,1],[2,3,4],[0,11],list(range(12))):
                expected=mask.copy()
                expected.reshape(-1)[pixels]^=1
                self.assertEqual(xor_rle(encoded(mask),pixels),encoded(expected))

    def test_duplicates_and_bad_indices_rejected(self):
        source=encoded(np.zeros((2,3),dtype=np.uint8))
        for pixels in ([0,0],[-1],[6],[.5],[True]):
            with self.assertRaises(ValueError):
                xor_rle(source,pixels)

    def test_binary_counts_and_no_raster_calls(self):
        from unittest.mock import patch
        mask=np.asarray([[1,0,0],[0,1,0]],dtype=np.uint8)
        source=codec.encode(np.asfortranarray(mask))
        expected=mask.copy()
        expected.reshape(-1)[[0,2,3]]^=1
        target=codec.encode(np.asfortranarray(expected))
        with patch.object(codec,"decode",side_effect=AssertionError("No full raster decode")), \
             patch.object(codec,"encode",side_effect=AssertionError("No full raster encode")):
            self.assertEqual(xor_rle(source,[0,2,3]),target)


    def test_two_by_three_row_major_orientation(self):
        mask=np.asarray([[0,1,0],[1,0,0]],dtype=np.uint8)
        source=encoded(mask)
        pixels=[0,2,4]
        expected=mask.copy()
        expected.reshape(-1)[pixels]^=1
        self.assertEqual(xor_rle(source,pixels),encoded(expected))
        self.assertIs(xor_rle(source,[]),source)


if __name__=="__main__":
    unittest.main()
