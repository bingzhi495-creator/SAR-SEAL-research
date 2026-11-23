"""Synthetic variable-canvas masks and decoded-group scoring checks."""
import argparse
import importlib.util
import sys
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as codec


def module():
    path=Path(__file__).with_name("score_factorial.py")
    spec=importlib.util.spec_from_file_location("pseg_factor",path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def encode(a):
    v=codec.encode(np.asfortranarray(a,dtype=np.uint8))
    return dict(size=v["size"],counts=v["counts"].decode("ascii"))


class VariableCanvas(unittest.TestCase):
    def test_tiny_float32_gap_and_true_tie_have_different_allocations(self):
        m=module()
        raw=np.zeros((9,17),bool)
        raw.flat[:34]=1
        coords=[[92,4],[120,5]]
        cand=dict(segmentation=encode(raw),coords=coords,k=1,position=1)
        two_adds=raw.copy()
        two_adds.flat[[92,120]]=True
        balanced=raw.copy()
        balanced.flat[[92,4]]=[True,False]
        threshold=two_adds.copy()
        threshold.flat[4]=False
        for margin,correct in ((2.**-54,two_adds),(2.**-149,two_adds),(0.,balanced)):
            post=dict(position=1,probabilities=[[1.,margin],[1.,1.]],half_foreground=[])
            entries={arm:{"rle":encode(answer)} for arm,answer in
                     (("nc",correct),("nt",threshold),("wc",correct),("wt",threshold))}
            checked=m.verify_candidate(cand,post,entries,raw.shape)
            self.assertEqual(checked["nc"]["adds"],2 if margin else 1)
            if margin:
                with self.assertRaises(ValueError):
                    m.verify_candidate(cand,post,dict(entries,wc={"rle":encode(balanced)}),raw.shape)

    def test_author_half_foreground_and_variable_width(self):
        m=module()
        raw=np.zeros((9,17),bool)
        raw.flat[:34]=1
        coords=np.array([[92,4],[120,5],[100,7]])
        p=np.array([[.5,.42],[.9,.5],[.2,.7]],dtype=np.float32)
        nc=raw.copy()
        nc.flat[[120,4]]=[1,0]
        nt=nc.copy()
        nt.flat[92]=1
        entry={"nc":{"rle":encode(nc)},"nt":{"rle":encode(nt)},"wc":{"rle":encode(nc)},"wt":{"rle":encode(nt)}}
        req=dict(segmentation=encode(raw),coords=coords.tolist(),k=1,position=1)
        post=dict(position=1,probabilities=p.tolist(),half_foreground=[0,3])
        out=m.verify_candidate(req,post,entry,raw.shape)
        self.assertEqual(out["nt"]["flips"],3)
        self.assertEqual(out["nt"]["narrow_zero_selected"],1)
        wrong=nt.copy()
        wrong.flat[92]=0
        with self.assertRaises(ValueError):
            m.verify_candidate(req,post,dict(entry,nt={"rle":encode(wrong)}),raw.shape)

    def test_paired_interaction_bootstrap_on_decoded_images(self):
        m=module()
        values=np.array([[[6.],[5.],[7.],[4.]], [[9.],[5.],[9.],[3.]]])
        before=np.array([[[0.],[0.],[0.],[0.]]])
        z=m.interaction_summary(values,m.ARMS,("continuous_iou_points",))
        self.assertEqual(z["continuous_iou_points"],dict(delta=2.,percentile95=[2.,2.]))
        self.assertEqual(m.interaction_summary(before,m.ARMS,("x",))["x"]["delta"],0.)

    def test_author_category_remap_and_rematch_empty_output(self):
        if CODE is None:
            self.skipTest("Pass --code to use original evaluator")
        m=module()
        c=m.load(CODE/"PSEG_TEST232_CLASSICAL_ATTRACTION_20261122/compare.py",m.COMPARE)
        scorer=m.load(CODE/"PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py",c.SCORE_SHA)
        geom=m.load(CODE/"PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py",c.GEOMETRY_SHA)
        base=geom.load(CODE/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py",geom.OLD)
        shared=geom.load(CODE/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",geom.GEOM)
        modules=shared.geometry_modules(CODE)
        a=np.zeros((9,17),bool)
        a[:5,:8]=True
        native=encode(a)
        det={"category_id":1,"image_id":"old_id","score":.8,"segmentation":native}
        new={"wt":{"rle":encode(np.zeros_like(a))}}
        ch={"1":new}
        gt=dict(info={},images=[dict(id=14,file_name="small.jpg",width=17,height=9)],
                annotations=[dict(id=5,category_id=0,image_id=14,area=40.,bbox=[0.,0.,8.,5.],
                                  segmentation=[[0.,0.,8.,0.,8.,5.,0.,5.]],iscrowd=0)],categories=[dict(id=0,name="ship")])
        scorer.ARMS=(*scorer.ARMS,*c.NEW_ARMS,*m.ARMS)
        view=scorer.score_populations(gt,{"small.jpg":([det],ch,[9,17])},[])["official"]
        records={14:dict(native=[det],edit=ch,audit=[dict(mask_area=40)],rgb="group-A")}
        refs=("raw",*c.NEW_ARMS)
        historic=dict(continuous_iou_points=100.,fp_pixels_per_gt_area_pct=0.,fn_pixels_per_gt_area_pct=0.,
                      pair_equal_continuous_iou_points=100.,strict75_count=1,strict90_count=1)
        prior=dict(original_native_matches=1,eligible_pairs=1,decoded_image_groups_with_pairs=1,arms={k:historic for k in refs})
        truth,_,_=modules.geometry.polygon_support(gt,modules.vector)
        g,e,ix,coverage,anchors=m.fixed_view(gt,[14],records,view["metrics_fraction"],prior,geom,base,modules,truth,{},refs,None)
        self.assertEqual(anchors,{5:(14,1)})
        self.assertEqual(coverage["wt"]["rematched_hulls_50"],0)
        self.assertEqual(coverage["wc"]["original_pairs_retained_50"],1)
        self.assertEqual(e["wc_minus_wt"]["different_decoded_groups"],1)
        self.assertEqual(ix["continuous_iou_points"]["delta"],100.)

    def test_pinned_inference_agrees_with_local_signed_readout(self):
        if CODE is None:
            self.skipTest("Pass --code for the original ranker")
        m=module()
        actor=m.load(Path(__file__).with_name("infer_factorial.py"),m.INFER_WORKER)
        prior=actor.load_engine(CODE/"RSDD74_PHOENIX_SUPPORT_BUDGET_20261123/infer_factorial.py")
        rng=np.random.default_rng(248)
        native=np.zeros((27,33),bool)
        native.flat[:60]=True
        q=rng.choice(np.array([.25,.5,.75],dtype=np.float32),size=native.shape)
        binary=(q>.5)|((q==.5)&(rng.random(q.shape)>.5))
        coord=np.stack((np.arange(100,116),np.arange(16)),axis=1)
        records,witness,_,remote=actor.certify(native,coord,q,binary,3,prior,codec)
        check=m.verify_candidate(dict(segmentation=encode(native),coords=coord.tolist(),k=3,position=1),
                                 dict(position=1,**witness),records,native.shape)
        for arm in m.ARMS:
            for key in ("flips","adds","drops","inside_narrow_flips","outside_narrow_flips","signed_area_delta"):
                self.assertEqual(check[arm][key],remote[arm][key])


CODE=None
if __name__=="__main__":
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--code",type=Path)
    opts,rest=p.parse_known_args()
    CODE=opts.code
    sys.argv[1:]=rest
    unittest.main()
