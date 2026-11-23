"""Synthetic-only regressions for continuous IoU replacement under original COCO matching."""

import argparse
from hashlib import sha256
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np
from pycocotools import mask


def load(path,sha=None):
    b = path.read_bytes()
    if sha and sha256(b).hexdigest()!=sha:
        raise ValueError("Wrong reference scorer")
    obj=ModuleType(path.stem)
    obj.__file__=str(path)
    exec(compile(b,str(path),"exec"),obj.__dict__)
    return obj


def rle(x0,y0,x1,y1):
    a=np.zeros((16,16),dtype=np.uint8,order="F")
    a[y0:y1,x0:x1]=1
    data=mask.encode(a)
    return dict(size=data["size"],counts=data["counts"].decode("ascii"))


def ann(image_id,aid,ring):
    return dict(id=aid,image_id=image_id,category_id=1,segmentation=[ring],area=25,bbox=[4,3,5,5],iscrowd=0)


def det(i,score,r):
    return dict(image_id=i,category_id=1,score=score,segmentation=r)


def fixture():
    truth = dict(images=[dict(id=i,height=16,width=16) for i in (2,1)],
                 categories=[dict(id=1,name="ship")],annotations=[
                     ann(1,11,[4.35,3,9.35,3,9.35,8,4.35,8]),
                     ann(2,21,[4,3,9,3,9,8,4,8])])
    blocks={i:dict(ranks=[1,2],stock=[det(i,.8,rle(4,3,9,8)),det(i,.8,rle(6,5,7,6))],
                   changes={}) for i in (2,1)}
    return truth,blocks


class ContinuousTests(unittest.TestCase):
    def test_subpixel_threshold_stable_ties_and_legacy_parity(self):
        truth,blocks=fixture()
        ctx=api.prepare(code,truth)
        expected,_=legacy.score_one(truth,blocks,"raw")
        report=api.score_arm(ctx,blocks,"raw",expected_raster=expected)
        self.assertEqual(report["original_raster"],report["supported_raster"])
        self.assertEqual(report["audit"]["new_run_areas_verified"],1)
        self.assertGreater(report["audit"]["pruned_independent_checks"],0)
        self.assertEqual(len(report["continuous"]),13)
        self.assertAlmostEqual(report["original_raster"]["AP90"],(51+50*2/3)/101,places=12)
        self.assertAlmostEqual(report["continuous"]["AP90"],51/303,places=12)
        for mode,gt in (("supported_raster",11),("continuous",0)):
            images=report["events"][mode]
            self.assertEqual(list(images),[1,2])
            self.assertEqual(images[1]["positive_gt"],1)
            self.assertEqual(images[1]["detections"],[(1,.8,False,gt),(2,.8,False,0)])
        expected_cont=23.25/(50-23.25)
        self.assertAlmostEqual(ctx.overlaps[(11,api.key_for(rle(4,3,9,8)))],expected_cont)
        again=api.score_arm(ctx,blocks,"another_unchanged_arm",expected_raster=expected)
        self.assertEqual(again["continuous"],report["continuous"])
        self.assertGreaterEqual(again["audit"]["cached_intersections"],2)

    def test_neutral_keeps_matched_detection_ignored_in_both_modes(self):
        truth,blocks=fixture()
        truth["annotations"][1]=ann(2,21,[4,3,9,8,4,8,9,3])
        ring=truth["annotations"][1]["segmentation"]
        native=mask.merge(mask.frPyObjects(ring,16,16))
        native["counts"]=native["counts"].decode("ascii")
        blocks[2]["stock"][0]=det(2,.8,native)
        ctx=api.prepare(code,truth)
        self.assertEqual(ctx.support["neutral"],1)
        report=api.score_arm(ctx,blocks,"raw")
        for mode in ("supported_raster","continuous"):
            events=report["events"][mode][2]
            self.assertEqual(events["positive_gt"],0)
            self.assertEqual(events["detections"][0],(1,.8,True,0))

    def test_pruning_threshold_and_rectangles_wrapping_columns(self):
        truth,_=fixture()
        ctx=api.prepare(code,truth)
        box=np.array([[0.,0.,5.,5.],[0,0,4.99,5.]])
        self.assertEqual(ctx.geometry.plausible(box,np.array([25.,24.95]),np.array([0.,0.,10.,5.]),50.).tolist(),
                         [True,False])
        raster=np.zeros((6,5),dtype=np.uint8,order="F")
        raster.ravel(order="F")[3:19]=1
        encoded=mask.encode(raster)
        rec=ctx.geometry.boxes_from_counts(ctx.shift.runs(encoded["counts"].decode()),6,5)
        self.assertEqual(np.prod(rec[:,2:]-rec[:,:2],axis=1).sum(),int(mask.area(encoded)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--code",type=Path,required=True)
    args,other=p.parse_known_args()
    code=args.code
    api=load(Path(__file__).with_name("continuous.py"))
    legacy=load(code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",
                "77a3fbfaa04ac08287eff4ec7bb827ce3914e61aa20dd4f1e6f9c1e3b8ccc9fa")
    unittest.main(argv=[__file__,*other])
