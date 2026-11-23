"""Synthetic signed-budget and original COCO geometry checks."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from pycocotools import mask as codec

CODE = None


def load():
    path = Path(__file__).with_name("score_ablation.py")
    spec = importlib.util.spec_from_file_location("ablation", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


class SignedBudget(unittest.TestCase):
    def test_free_writes_even_negative_gains_and_relaxed_caps_imbalance(self):
        worker = load()
        p = np.array([[.45,.99],[.46,.97],[.47,.98],[.48,.96]])
        for policy, intended in (("balance",(2,2)),("relaxed",(3,1)),("free",(4,0))):
            add, drop = worker.selected_ordinals(p, 2, policy)
            self.assertEqual((len(add),len(drop)), intended)
            self.assertEqual(set(add), set(range(4-intended[0],4)))
        same = np.full((4,2), .5)
        for policy in ("balance","relaxed","free"):
            add,drop = worker.selected_ordinals(same, 2, policy)
            self.assertEqual((list(add),list(drop)), ([0,1],[0,1]))

    def test_rejects_matching_counts_at_wrong_pixel(self):
        worker = load()
        def rle(sites):
            x = np.zeros((512,512), np.uint8)
            x.flat[sites] = 1
            v = codec.encode(np.asfortranarray(x))
            return {"size":v["size"],"counts":v["counts"].decode("ascii")}
        stock = rle(list(range(32)))
        good = rle(list(range(1,32))+[33])
        bad = rle(list(range(1,32))+[32])
        candidate = {"coords":[[32,1],[33,0]],"k":1,"segmentation":stock}
        q = [[.3,.6],[.8,.2]]
        written = {arm:{"rle":good} for arm in worker.POLICIES}
        result = worker.verify_candidate(candidate, q, written, {"swaps":1,"rle":good})
        self.assertEqual(result["balance"],(1,1))
        with self.assertRaises(ValueError):
            worker.verify_candidate(candidate,q,dict(written,relaxed={"rle":bad}),{"swaps":1,"rle":good})

    def test_exposure_requires_different_masks_and_counts_shared_dates_once(self):
        import shapely
        w = load()
        def rle(n):
            x = np.zeros((512,512),np.uint8)
            x.flat[:n] = 1
            d = codec.encode(np.asfortranarray(x))
            return {"size":d["size"],"counts":d["counts"].decode("ascii")}
        stock = rle(32)
        det = {"segmentation":stock}
        blocks = {i:{"group":str(i),"changes":{1:{arm:{"rle":rle(33) if arm=="free" and i<=2 else stock}
                                                   for arm in w.POLICIES}}} for i in range(1,4)}
        anchors = {i:(i,1,det) for i in blocks}
        truth = {i:shapely.box(0,0,32,1) for i in blocks}
        e = w.exposures(anchors,blocks,truth,{"1":("A",1),"2":("A",1),"3":("A",2)})
        self.assertEqual(e["balance_minus_free"],dict(matches=2,parents=2,sensor_date_blocks=1))
        self.assertEqual(e["relaxed_minus_balance"],dict(matches=0,parents=0,sensor_date_blocks=0))

    def test_original_coco_and_geometry_count_the_same_unchanged_anchor(self):
        if CODE is None:
            self.skipTest("Pass --code for historical evaluator integration")
        import shapely
        w = load()
        whole,_,deps = w.sources(SimpleNamespace(code=CODE))
        cfa,evaluator = deps[2],deps[6]
        cells = np.zeros((512,512),np.uint8)
        cells.flat[:64]=1
        m = codec.encode(np.asfortranarray(cells))
        mask = dict(size=m["size"],counts=m["counts"].decode("ascii"))
        polygon = [0.,0.,64.,0.,64.,1.,0.,1.]
        gt = dict(info={},images=[dict(id=1,width=512,height=512)],categories=[dict(id=1,name="ship")],
                  annotations=[dict(id=1,image_id=1,category_id=1,iscrowd=0,area=64.,bbox=[0.,0.,64.,1.],
                                    segmentation=[polygon])])
        det = dict(image_id=1,category_id=1,score=.8,segmentation=mask)
        blocks = {1:dict(stock=[det],ranks=[1],group="A",changes={})}
        metrics,_ = evaluator.score_one(gt,blocks,"raw",anchors=True)
        numbers = {k:v if v>=0 else None for k,v in metrics.items()}
        fixed = dict(zip(cfa.FIELDS,[100.,0.,0.,100.,64.]))
        prior = dict(images=1,annotations=1,source_groups=1,metrics_fraction=dict(raw=numbers,phoenix=numbers),
                    geometry=dict(counts=dict(native_matches=1,eligible_matches=1,invalid_polygon_matches=0,area_excluded_matches=0),
                                  group_equal_arms=dict(raw=fixed,phoenix=fixed),represented_groups=1,sensor_date_blocks=1))
        view = w.score_view({1},gt,blocks,{1:shapely.Polygon(np.array(polygon).reshape(-1,2))},
                            {"A":("A",5)},prior,whole,deps)
        self.assertEqual(view["geometry"]["counts"]["eligible_matches"],1)
        for contrast in view["contrasts"].values():
            self.assertEqual(contrast["criteria"]["differing_fixed_hulls"],0)
            self.assertEqual(contrast["COCO_points"]["AP"],0.)
            self.assertFalse(contrast["criteria"]["exposure_supported"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--code",type=Path)
    opts,rest=parser.parse_known_args()
    CODE = opts.code
    sys.argv[1:]=rest
    unittest.main()
