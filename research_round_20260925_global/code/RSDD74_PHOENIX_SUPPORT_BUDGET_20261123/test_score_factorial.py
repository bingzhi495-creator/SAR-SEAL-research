"""Native-pixel support, quota and paired geometry tests on synthetic ships."""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask as codec


def module():
    path = Path(__file__).with_name("score_factorial.py")
    spec = importlib.util.spec_from_file_location("factorial",path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def mask(sites):
    arr = np.zeros((512,512),np.uint8)
    arr.flat[list(sites)] = 1
    out = codec.encode(np.asfortranarray(arr))
    return dict(size=out["size"],counts=out["counts"].decode("ascii"))


class FactorialTests(unittest.TestCase):
    def test_rejects_wrong_whole_capped_pixel_at_same_count(self):
        m = module()
        # Known positive utilities in descending order at pixels 35,2,34 (2 is a deletion).
        native = set(range(32))
        c = dict(k=1,coords=[[34,2],[35,3]],segmentation=mask(native))
        q = [[.75,.15],[.95,.90]]
        capped = mask((native-{2})|{35})
        threshold = mask((native-{2})|{34,35})
        entries = {name:{"rle":r} for name,r in (("nc",capped),("nt",threshold),("wc",capped),("wt",threshold))}
        counts = m.check_candidate(c,q,capped,threshold,entries,lambda r:int(codec.area(r)))
        self.assertEqual(counts["wc"]["flips"],2)
        wrong = dict(entries,wc={"rle":mask(native|{34,35})})
        with self.assertRaises(ValueError):
            m.check_candidate(c,q,capped,threshold,wrong,lambda r:int(codec.area(r)))

    def test_interaction_resamples_paired_parent_differences(self):
        if CODE is None:
            self.skipTest("Pass --code to load published evaluator")
        w=module()
        _,_,_,deps=w.sources(SimpleNamespace(code=CODE))
        cfa,reserve=deps[2],deps[4]
        # Two distinct parents on date A plus one on B. Both paired count effects
        # have different means across dates but their difference is constant.
        values=np.zeros((3,4,len(cfa.FIELDS)))
        values[:,0,0]=[1.,3.,8.]
        values[:,2,0]=[3.,5.,10.]
        effect=w.interaction_summary(values,["a","a","b"],cfa,reserve)
        self.assertEqual(effect["continuous_iou_points"],dict(delta=2.,interval_95=[2.,2.]))

    def test_margin_bounds_and_whole_policy_on_original_inactive_candidate(self):
        if CODE is None:
            self.skipTest("Pass --code to load published evaluator")
        import shapely
        w=module()
        _,whole,_,deps=w.sources(SimpleNamespace(code=CODE))
        cfa,evaluator=deps[2],deps[6]
        # Historical LVIS can erase an inactive original detection. The factorial arms keep it.
        source=mask(range(64))
        polygon=[0.,0.,64.,0.,64.,1.,0.,1.]
        gt=dict(info={},images=[dict(id=1,width=512,height=512)],categories=[dict(id=1,name="ship")],
                annotations=[dict(id=1,image_id=1,category_id=1,iscrowd=0,area=64.,bbox=[0.,0.,64.,1.],segmentation=[polygon])])
        det=dict(image_id=1,category_id=1,score=.8,segmentation=source)
        blocks={1:dict(stock=[det],ranks=[1],group="A",changes={1:{"historical_lvis":{"rle":mask([])}}})}
        native,_=evaluator.score_one(gt,blocks,"raw",anchors=True)
        old,_=evaluator.score_one(gt,blocks,"historical_lvis",anchors=True)
        admissible=lambda metrics:{k:v if v>=0 else None for k,v in metrics.items()}
        names=whole.MATCHED
        known=dict(images=1,annotations=1,source_groups=1,metrics_fraction={k:admissible(old if k=="historical_lvis" else native) for k in names},
                   geometry=dict(counts=dict(native_matches=1,eligible_matches=1,invalid_polygon_matches=0,area_excluded_matches=0),
                                 group_equal_arms={k:dict(zip(cfa.FIELDS,[0,0,100,0,0] if k=="historical_lvis" else
                                                            [100,0,0,100,64])) for k in names},
                                 represented_groups=1,sensor_date_blocks=1))
        out=w.score_view({1},gt,blocks,{1:shapely.Polygon(np.array(polygon).reshape(-1,2))},{"A":("A",3)},known,whole,deps)
        self.assertEqual(out["coverage"]["wt"]["matched_hulls_50"],1)
        self.assertEqual(out["coverage"]["historical_lvis"]["matched_hulls_50"],0)
        self.assertEqual(out["interaction"]["paired_geometry"]["continuous_iou_points"]["interval_95"],[0.,0.])
        self.assertEqual(out["contrasts"]["wc_minus_wt"]["criteria"]["differing_fixed_hulls"],0)

    def test_nonpositive_bounds_allow_author_ties_at_half(self):
        w=module()
        counts={"by_arm":{arm:dict(flips=6,also_changed_by_wt=5,known_narrow_positive=0,known_narrow_nonpositive=0)
                          for arm in w.ARMS}}
        counts["by_arm"]["wt"]["also_changed_by_wt"]=6
        real={"counts":dict(threshold_zero_utility_flipped_cells=2,threshold_half_cells=3)}
        bounds=w.interval_for_nonpositive_counts(counts,real)
        self.assertEqual(bounds["wt"]["nonpositive_min"],2)
        self.assertEqual(bounds["wt"]["nonpositive_max"],2)
        self.assertEqual((bounds["wc"]["nonpositive_min"],bounds["wc"]["nonpositive_max"]),(2,3))

    def test_local_masks_and_signed_margin_checks_agree_with_frozen_writer(self):
        w=module()
        path=Path(__file__).with_name("infer_factorial.py")
        self.assertEqual(w.digest(path.read_bytes()),w.INFER_WORKER)
        spec=importlib.util.spec_from_file_location("frozen_infer",path)
        actor=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(actor)
        rng=np.random.default_rng(89)
        native=np.zeros((512,512),bool)
        native.flat[:64]=True
        coords=np.stack((np.arange(100,124),np.arange(24)),axis=1)
        prob=rng.choice(np.array([.1,.25,.5,.75,.9],dtype=np.float32),size=native.shape)
        prob.flat[coords]=rng.choice(np.array([.1,.25,.75,.9],dtype=np.float32),size=coords.shape)
        binary=(prob>.5)|((prob==.5)&native)
        results,tally=actor.policies(native,coords,prob,binary,5)
        encoded=lambda v:actor.encode(v,codec)
        entries={arm:{"rle":encoded(out)} for arm,out in results.items()}
        counts=w.check_candidate(dict(k=5,coords=coords,segmentation=encoded(native)),prob.flat[coords],
                                 entries["nc"]["rle"],entries["wt"]["rle"],entries,lambda m:int(codec.area(m)))
        source={}
        for arm,out in results.items():
            s=np.where(native,.5-prob,prob-.5)[out!=native]
            source[arm]={k+"_selected_flips":int(x) for k,x in
                         (("positive",(s>0).sum()),("zero",(s==0).sum()),("negative",(s<0).sum()),("nonpositive",(s<=0).sum()))}
        counts={"by_arm":counts}
        receipt={"counts":tally,"counts_by_arm":source}
        exact=w.verify_margin_counts(counts,receipt,w.interval_for_nonpositive_counts(counts,receipt))
        self.assertEqual(exact["wc"]["positive"]+exact["wc"]["nonpositive"],10)


CODE=None
if __name__=="__main__":
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--code",type=Path)
    opts,rest=parser.parse_known_args()
    CODE=opts.code
    sys.argv[1:]=rest
    unittest.main()
