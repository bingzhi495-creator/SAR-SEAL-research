"""Native-bounds, pixel-center sampling, and negative exclusions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as codec

spec = importlib.util.spec_from_file_location('corpus', Path(__file__).with_name('corpus.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Roi(unittest.TestCase):
    def test_integer_pixels_and_center_grid(self):
        self.assertEqual(worker.integer_crop([.1,2.,2.1,6.001], (8,8)), [1,2,3,7])
        target = np.array([[0,1],[1,0]], dtype=np.uint8)
        restored = worker.nearest(target)
        self.assertEqual(int(restored[:24,:24].sum()),0)
        self.assertEqual(int(restored[24:,:24].sum()),576)
        self.assertEqual(set(np.unique(restored)),{0,1})

    def test_bilinear_clamp_does_not_pull_outside_roi(self):
        a = np.array([[0,100]],dtype=np.uint8)
        rgb=np.repeat(a[:,:,None],3,axis=-1)
        crop=worker.bilinear_rgb(rgb)
        self.assertEqual(tuple(crop.shape),(3,48,48))
        self.assertTrue((crop[:,:,0]==0).all() and (crop[:,:,-1]==100).all())
        self.assertTrue((crop[:,:,23]<=50).all() and (crop[:,:,24]>=50).all())

    def test_training_negative_excludes_hull_swallowed_by_large_roi(self):
        occupied=np.zeros((12,12),dtype=bool)
        occupied[1,1]=True
        boxes=np.array([[0,0,10,10],[2,2,4,4],[5,5,6,6],[0,0,0,0]])
        blocked={1}
        negatives=worker.negatives(boxes,np.array([[1,1,1,1]]),occupied,blocked,4)
        self.assertEqual(negatives,[(2,0.)])

    def test_author_target_roundtrip(self):
        target=np.array([[1,0,0],[0,0,1]],dtype=np.uint8)
        rle=worker.encoded_crop(target)
        self.assertEqual(int(codec.area(rle)),2)
        np.testing.assert_array_equal(codec.decode(rle),target)


if __name__=='__main__':
    unittest.main()
