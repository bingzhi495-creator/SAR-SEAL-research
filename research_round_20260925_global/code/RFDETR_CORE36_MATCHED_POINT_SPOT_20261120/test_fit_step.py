"""Capacity spot checks on synthetic samples only."""
from pathlib import Path
import os
import unittest

import numpy as np

from fit_step import attach_truth, fit_heads, load_geometry, load_trainer


class FitStepTest(unittest.TestCase):
    def test_stock_rank_and_fractional_polygon_target(self):
        from pycocotools import mask as codec
        helpers=load_geometry(Path(os.environ["CODE_ROOT"]),Path(os.environ["SHAPELY_SITE"]))
        image=dict(id=1,file_name="synthetic.png",height=32,width=32)
        ann=dict(id=31,image_id=1,category_id=1,iscrowd=0,area=72.25,bbox=[8.25,8.25,8.5,8.5],
                 segmentation=[[8.25,8.25,16.75,8.25,16.75,16.75,8.25,16.75]])
        hard=np.zeros((32,32),dtype=np.uint8)
        hard[8:17,8:17]=1
        rle=codec.encode(np.asfortranarray(hard))
        rle["counts"]=rle["counts"].decode()
        candidates=[dict(rank=7,score=.9,area=81,rle=rle)]
        record=dict(image_id=1,rank=7,query_id=99,group="P1",
                    coordinates=np.array([[8*32+7,8*32+8]],dtype=np.int32),
                    features=np.zeros((1,2,14),dtype=np.float32),truth=None)
        records,audit=attach_truth(image,[ann],candidates,[record],helpers)
        self.assertEqual(audit["stock_matches"],1)
        self.assertEqual(records[0]["matched_gt_id"],31)
        self.assertEqual(records[0]["query_id"],99)
        np.testing.assert_allclose(records[0]["truth"],[[0.,.5625]],rtol=0,atol=1e-12)
        self.assertEqual(records[0]["raw_continuous"]["truth_area"],72.25)
        self.assertIsNone(record["truth"])
    def test_common_rows_and_two_real_updates(self):
        features=np.zeros((3,2,14),dtype=np.float32)
        features[:,:,0]=[.4,.6]
        features[:,:,1]=[0.,1.]
        records=[dict(image_id=i+1,rank=1,query_id=i,group="G"+str(i),coordinates=np.array([[0,1],[2,3],[4,5]]),
                      features=features.copy(),truth=np.tile([1.,0.],(3,1))) for i in range(2)]
        trainer=load_trainer(Path(os.environ["TRAIN_SOURCE"]))
        out=fit_heads(records,trainer,steps=2)
        self.assertEqual(out["audit"]["shared_pairs"],6)
        for key,parameters in (("S",1321),("L",139265)):
            audit=out["audit"]["heads"][key]
            self.assertEqual(audit["parameters"],parameters)
            self.assertEqual(audit["optimizer_steps"],[2])
            self.assertTrue(audit["finite_parameters"])
            self.assertNotEqual(audit["initial_tensor_sha256"],audit["final_tensor_sha256"])
            self.assertEqual(out["prediction"][key][0].shape,(3,2))


if __name__=="__main__":
    unittest.main()
