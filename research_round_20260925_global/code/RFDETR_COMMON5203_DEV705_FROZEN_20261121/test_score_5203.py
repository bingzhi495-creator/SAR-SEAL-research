"""Generated masks, labels and receipts for the independent carrier adapter."""
import copy
from hashlib import sha256
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as codec
import score_5203 as s


def pred(mask,score=.8):
    r=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(image_id=10,category_id=1,score=score,segmentation=dict(size=r["size"],counts=r["counts"].decode()))


class TestReplicatedCarrier(unittest.TestCase):
    def test_capture_binding_and_fail_closed(self):
        terminal=dict(status="RFDETR_COMMON5203_DEV705_PRELABEL_TWO_STAGES_CLOSED_V1",original_exit=0,
                      native_exit=0,refinement_exit=0,DEV=True,FIT=False,TEST=False,GT_opened=False,AP=False,
                      images=705,mapped_rows=1500,fit_accepted_sha256=s.HEADS,endpoint_sha256="1"*64)
        training=dict(status="COMMON5203_SINGLE_CENTER_E24_ACCEPTED",FIT=True,DEV=False,TEST=False,AP=False,
                      arm=dict(checkpoint_sha256="1"*64))
        own=dict(status="RFDETR_COMMON5203_DEV705_NATIVE_CAPTURED_V1",images=705,mapped_rows=1500,
                 raw_hashes_sha256="4"*64,predictions_sha256="2"*64)
        refined=dict(status="RFDETR_COMMON5203_DEV705_FROZEN_CAPTURED_V1",raw_hashes_sha256="4"*64)
        s.validate_capture(terminal,training,own,refined)
        for args in ((terminal,training,dict(own,predictions_sha256=s.OLD_STOCK),refined),
                     (dict(terminal,refinement_exit=2),training,own,refined),
                     (dict(terminal,TEST=True),training,own,refined),
                     (terminal,training,own,dict(refined,raw_hashes_sha256="5"*64)),
                     (dict(terminal,endpoint_sha256=s.OLD_ENDPOINT),training,own,refined)):
            with self.assertRaises(ValueError):
                s.validate_capture(*args)

    def test_generated_development_populations_and_own_matches(self):
        if s.SHAPELY is not None:
            sys.path.append(str(s.SHAPELY))
        helper=s.load("geometry").geometry_modules(s.CODE)
        stock=np.zeros((800,800),np.uint8)
        stock[5:11,5:11]=1
        extra=np.zeros_like(stock)
        extra[300,300]=1
        rows={10:[pred(extra,.9),pred(stock,.8)]}
        image=dict(id=10,width=800,height=800,file_name="P1_synthetic.png")
        gt=dict(images=[image],categories=[dict(id=1,name="ship")],annotations=[
            dict(id=34,image_id=10,category_id=1,segmentation=[[5,5,11,5,11,11,5,11]],
                 bbox=[5,5,6,6],area=36,iscrowd=0),
            dict(id=35,image_id=10,category_id=1,segmentation=[[20,20,20,20,20,20]],
                 bbox=[20,20,0,0],area=0,iscrowd=0)])
        records,total=s.fixed_matches(gt,[image],rows,{10:[7,43]},s.load("matcher"),helper)
        self.assertEqual((total,len(records),records[0][:3]),(1,1,(10,1,34)))
        self.assertAlmostEqual(s.continuous(rows,records,helper)[0],1.)
        changed=stock.copy()
        changed[5,5]=0
        changed[4,5]=1
        newer={10:[pred(extra,.9),pred(changed,.8)]}
        self.assertAlmostEqual(s.continuous(newer,records,helper)[0],35/37)
        official=s.load("dev")
        official.OFFICIAL=s.CODE/"DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"
        measures=s.load("metrics").score_official(official,gt,rows[10],{35})
        self.assertEqual([measures[k]["annotations"] for k in ("primary","sensitivity")],[2,1])
        self.assertEqual(len(measures["sensitivity"]["all12"]),12)
        self.assertIn("AP90",measures["primary"])

    def test_paired_carrier_difference_and_group_weights(self):
        groups=[[0,1,2],[3]]
        allarms=("N",*s.ARMS)
        metric_keys=("AP","AP50","AP75","APs","AP90","AR100")
        def views(value):
            return {k:{m:value for m in metric_keys} for k in ("primary","sensitivity")}
        old={"geometry_summary":{"per_seed":{}},"coco":{}}
        now={}
        values={}
        for seed in s.SEEDS:
            old["coco"][seed]={a:views(.102 if a=="SU=F" else .1) for a in allarms}
            old["geometry_summary"]["per_seed"][seed]={"arms":{a:{"group_equal_iou_points":10.3 if a=="SU=F" else 10.}
                                                     for a in allarms}}
            now[seed]={a:views(.105 if a=="SU=F" else .1) for a in allarms}
            values[seed]={a:np.array([.2,.2,.2,.3]) if a=="SU=F" else np.array([.2]*4) for a in allarms}
        out=s.evaluate_contrasts(now,values,groups,old)["SU=F minus SF"]
        self.assertAlmostEqual(out["mean_COCO_points"]["primary"]["AP"]["5203_minus_5202"],.3)
        self.assertAlmostEqual(out["mean_fixed_iou_points"]["5203"],5.)
        self.assertAlmostEqual(out["mean_fixed_iou_points"]["5203_minus_5202"],4.7)

    def test_generated_native_signatures(self):
        p=np.zeros((800,800),np.uint8)
        p[2,3]=1
        stream,reader=s.load("stream"),s.load("reader")
        parser=s.load("parser")
        b=dict(image_id=10,raw_query_count=164,native_count=100,unmapped_slot1_count=99,native_ranks=[5],coco=[pred(p)])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"generated.jsonl"
            def run(obj,expected=None):
                x=reader.encode(obj)
                path.write_bytes(x)
                return stream.load_arm(path,dict(bytes=len(x),sha256=sha256(x).hexdigest()),[10],(800,800),1,parser,expected)
            _,_,sig=run(b)
            good=copy.deepcopy(b)
            p[2,3]=0
            p[3,3]=1
            good["coco"]=[pred(p)]
            run(good,sig)
            with self.assertRaises(ValueError):
                run(dict(good,native_ranks=[4]),sig)
            good["coco"][0]["score"]=.799
            with self.assertRaises(ValueError):
                run(good,sig)

    def test_raw_tensor_roster_requires_all_three_full_hashes(self):
        sample=dict(image_id=3,pred_logits_sha256="a"*64,pred_boxes_sha256="b"*64,pred_masks_sha256="c"*64)
        reader=s.load("reader")
        s.validate_raw_roster(reader.encode(sample),[3])
        with self.assertRaises(ValueError):
            s.validate_raw_roster(reader.encode(sample),[4])
        with self.assertRaises(ValueError):
            s.validate_raw_roster(reader.encode(dict(sample,pred_boxes_sha256="wrong")),[3])


if __name__=="__main__":
    unittest.main()
