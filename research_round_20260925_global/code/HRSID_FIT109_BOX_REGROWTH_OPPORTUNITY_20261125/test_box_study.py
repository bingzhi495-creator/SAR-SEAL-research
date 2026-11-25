"""Synthetic tests for disjoint matching and exact contour threshold ties."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('box_study', Path(__file__).with_name('box_study.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ExactRoi(unittest.TestCase):
    def test_equal_logits_switch_as_one_group(self):
        z=np.array([4.,3.,3.,2.])
        gt=np.array([True,True,False,False])
        self.assertEqual(worker.threshold_ceiling(z,gt,0,0,2),2/3)

    def test_reserved_candidate_and_author_order_tie(self):
        hulls=[dict(id=8,bbox=[0,0,3,3]),dict(id=5,bbox=[0,0,3,3]),dict(id=9,bbox=[5,5,2,2])]
        boxes=np.array([[0,0,3,3],[0,0,3,3],[5,5,7,7]])
        self.assertEqual(worker.assign(hulls,boxes,{5},{0}),{8:1,9:2})

    def test_frozen_exterior_area(self):
        stock=np.array([[1,1,0],[0,1,0]],dtype=bool)
        gt=np.array([[1,1,1],[0,0,0]],dtype=bool)
        self.assertEqual(worker.box_geometry(stock,gt,[1,0,3,2])['frozen_free_area'],1.)
        self.assertEqual(worker.box_geometry(stock,gt,[1,0,3,2])['crop_only'],2/3)


if __name__=='__main__':
    unittest.main()
