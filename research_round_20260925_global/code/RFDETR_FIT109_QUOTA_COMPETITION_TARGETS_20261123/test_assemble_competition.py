"""Cross-check sparse action overlaps and cohort matching against COCO."""

import os
from pathlib import Path
from runpy import run_path
import unittest

import numpy as np
from pycocotools import mask as codec


API = run_path(str(Path(__file__).with_name("assemble_competition.py")))
COMP = API["source"](Path(os.environ["COMPETITION_SOURCE"]), API["COMPETITION"], "_test_competition")
SPARSE = API["source"](Path(os.environ["SPARSE_SOURCE"]), API["SPARSE_RLE"], "_test_sparse")


def rle(mask):
    encoded = codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(encoded,counts=encoded["counts"].decode("ascii"))


class SparseTest(unittest.TestCase):
    def test_zero_annotation_id_alias_only_changes_the_identifier(self):
        geometry = [[1.,1., 2.,1., 2.,2., 1.,2.]]
        payload = [dict(id=0,area=1.,segmentation=geometry,iscrowd=0),
                   dict(id=5,area=1.,segmentation=geometry,iscrowd=0)]
        fixed = API["rekey_zero_annotation"](payload, 11)
        self.assertEqual([row["id"] for row in fixed], [11,5])
        self.assertEqual(fixed[0], dict(payload[0],id=11))
        self.assertIs(fixed[1], payload[1])
        self.assertEqual(payload[0]["id"], 0)

    def test_all_sparse_iou_alternatives_match_coco_binary_and_crowd(self):
        stock = np.zeros((8,8),dtype=np.uint8)
        stock[:5,:4] = 1
        gt = np.stack([np.roll(stock, 1,axis=1), np.roll(stock,1,axis=0)],axis=-1)
        raw = rle(stock)
        refs = [rle(gt[:,:,j]) for j in (0,1)]
        crowd = np.array([False,True])
        areas = codec.area(refs)
        old = codec.iou([raw], refs, [0,1])
        points = np.array([[4,0],[12,1],[20,2],[28,3]],dtype=np.int64)
        order = np.arange(8).reshape(4,2)//2
        inter = API["integer_intersection"](old, np.array([20]), areas, crowd)
        moves = np.column_stack((np.arange(5), np.arange(4,-1,-1)))
        actual = API["action_overlaps"](moves,points,order,inter[0],20,areas,crowd,gt.reshape(-1,2))
        alternatives = [SPARSE.xor_rle(raw,API["action_prefix"](points,order,mov)) for mov in moves]
        np.testing.assert_array_equal(actual,codec.iou(alternatives,refs,[0,1]))

    def test_global_direct_coco_parity_includes_ties_and_crowd(self):
        stock = np.zeros((8,8),dtype=np.uint8)
        stock[:5,:4] = 1
        shifted = np.roll(stock, 2, axis=1)
        anns = []
        results = {}
        for iid in (3,1):
            anns.append(dict(id=iid,image_id=iid,category_id=1,area=20,segmentation=rle(stock),
                             iscrowd=0,bbox=[0,0,4,5]))
            anns.append(dict(id=iid+10,image_id=iid,category_id=1,area=20,segmentation=rle(shifted),
                             iscrowd=1,bbox=[2,0,4,5]))
            results[iid] = [dict(image_id=iid,category_id=1,score=.5,segmentation=rle(m))
                             for m in (shifted,stock,stock)]
        gt = dict(info={},images=[dict(id=i,width=8,height=8) for i in (3,1)],
                  annotations=anns,categories=[dict(id=1,name="ship")])
        response=API["reference_world"](gt,results,COMP)
        self.assertEqual(response.positives,2)
        self.assertEqual(response.positions[1].tolist(),[0,1,2])
        self.assertEqual(response.positions[3].tolist(),[3,4,5])
        self.assertEqual(len(response.by_threshold),10)


if __name__=="__main__":
    unittest.main()
