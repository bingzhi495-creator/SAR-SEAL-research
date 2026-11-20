"""End-to-end original mask score and synthetic native continuous cohort."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import ssdd_replay as replay
import ssdd_score as score


class ScoreTest(unittest.TestCase):
    def test_one_native_continuous_identity(self):
        from pycocotools import mask as codec
        root=Path(os.environ["CODE_ROOT"])
        args=SimpleNamespace(old_score=Path(os.environ["OLD_ROOT"])/"score_endpoint.py")
        old=replay.load(Path(os.environ["OLD_ROOT"])/"dev_endpoint.py",replay.OLD,"_original_edit")
        old_score=replay.load(args.old_score,score.OLD_SCORE,"_original_scoring")
        source=replay.load(root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                           old_score.GEOM,"_source_geometry")
        stock=replay.load(root/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",old_score.MATCH,"_source_match")
        import sys
        sys.path.append(os.environ["SHAPELY_SITE"])
        modules=source.geometry_modules(root)
        im={"id":1,"height":193,"width":214,"file_name":"000001.jpg"}
        gt={"images":[im],"categories":[{"id":1,"name":"ship"}],"annotations":[dict(id=1,image_id=1,category_id=1,
            iscrowd=0,area=64.,bbox=[60.,40.,8.,8.],segmentation=[[60.,40.,68.,40.,68.,48.,60.,48.]])]}
        hard=np.zeros((193,214),dtype=np.uint8)
        hard[40:48,60:68]=1
        rle=codec.encode(np.asfortranarray(hard))
        rle["counts"]=rle["counts"].decode()
        row=dict(image_id=1,category_id=1,score=.9,segmentation=rle)
        audit=dict(native_rank=1,score=.9,box_xyxy=[50.,30.,80.,60.],mask_area=64)
        blocks=[(dict(image_id=1,rgb_sha256="a"*64),[row],[audit],{})]
        score.geometry_adapter(args,replay,old,old_score,blocks)
        actual,witness=old_score.geometry(gt,blocks,stock,modules)
        self.assertEqual(actual["counts"]["eligible"],1)
        self.assertEqual(actual["metrics"]["free_natural"]["strict90"],1)
        self.assertEqual(actual["strict90_transitions"]["free_natural"]["clean_retained"],1)
        self.assertEqual(len(witness),1)


if __name__=="__main__":
    unittest.main()
