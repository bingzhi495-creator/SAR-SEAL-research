"""Synthetic 8x8 COCO interface check; no project inputs or qualification chain."""
import contextlib
import copy
import importlib.util
import io
import unittest
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as masks
import psutil

psutil.Process().cpu_affinity([0, 1])
body_path = Path(__file__).with_name('score_body.py')
if not body_path.exists():
    body_path = Path(__file__).with_name('N25_FIXEDMASK_SCORE_BODY_UNBOUND_20261017_01.py')
sp = importlib.util.spec_from_file_location('synthetic_observer', str(body_path))
body = importlib.util.module_from_spec(sp)
sp.loader.exec_module(body)


class SyntheticControl:
    def check(self):
        pass


class OfficialInterface:
    @staticmethod
    def coco_metrics(coco, predictions, ids, factory):
        original = copy.deepcopy(coco)
        original.dataset.setdefault('info', {})
        ev = factory(original, original.loadRes(copy.deepcopy(predictions)), 'segm')
        ev.params.imgIds = list(ids)
        ev.params.catIds = [1]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return dict(stats=ev.stats.tolist(), iouType='segm', maxDets=[1, 10, 100],
                    original_GT_preserved=True, score_filter=None,
                    official_COCO_crowd_ignore_semantics=True)


def encoded(array):
    rle = masks.encode(np.asfortranarray(array, dtype=np.uint8))
    return dict(size=rle['size'], counts=rle['counts'].decode('ascii'))


class ObserverTests(unittest.TestCase):
    def test_identity_alignment_survives_rescoring_and_stable_ties(self):
        a=np.zeros((8,8),dtype=np.uint8); a[:2,:2]=1
        b=np.zeros_like(a); b[5:7,5:7]=1
        rles={0:encoded(a),99:encoded(b),50:encoded(np.ones_like(a))}
        empty=encoded(np.zeros_like(a))
        ids=list(range(1,188)); annotations=[]; values=[]; positions={}; gt_ids={}
        for image_id in ids:
            # The final image has no GT. Original GT order is B,A, while the
            # canonical check order A,B exercises explicit column reordering.
            base=2*image_id
            gt_ids[image_id]=[] if image_id==187 else [base+1,base+2]
            if image_id!=187:
                for aid,rle in [(base+2,rles[99]),(base+1,rles[0])]:
                    annotations.append(dict(id=aid,image_id=image_id,category_id=1,
                        segmentation=copy.deepcopy(rle),area=4,bbox=masks.toBbox(rle).tolist(),iscrowd=0))
            positions[image_id]=[]
            for q in [99]+list(range(99)):
                positions[image_id].append(len(values))
                values.append(dict(image_id=image_id,category_id=1,query_id=q,
                    score=.9 if q==50 else (.2 if q in (0,99) else .1),
                    segmentation=copy.deepcopy(rles.get(q,empty))))
        coco=COCO()
        coco.dataset=dict(images=[dict(id=i,height=8,width=8) for i in ids],
                          annotations=annotations,categories=[dict(id=1,name='synthetic')],info={})
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
            _,_,old=body.observe(np,SyntheticControl(),OfficialInterface,COCOeval,masks,coco,values,ids,positions,gt_ids)
            changed=[]
            for o,v in enumerate(values):
                local=positions[v['image_id']].index(o)
                matrix=old[v['image_id']]['ious']
                q=float(matrix[local].max()) if matrix.shape[1] else 0.
                changed.append(dict(v,score=float(v['score'])*q))
            _,_,new=body.observe(np,SyntheticControl(),OfficialInterface,COCOeval,masks,coco,changed,ids,positions,gt_ids)
        np.testing.assert_array_equal(old[1]['ious'],new[1]['ious'])
        np.testing.assert_array_equal(old[1]['ious'][0], [0.,1.]) # q99=B
        np.testing.assert_array_equal(old[1]['ious'][1], [1.,0.]) # q0=A
        self.assertEqual(old[1]['top10_query_ids'][:3],[50,99,0])
        self.assertEqual(new[1]['top10_query_ids'][:3],[99,0,50])
        self.assertEqual(new[187]['ious'].shape,(100,0))
        self.assertEqual(new[187]['top10_query_ids'],[99]+list(range(9)))
        self.assertEqual(len(values),len(changed))
        self.assertEqual(values[0]['score'],.2)


if __name__=='__main__':
    unittest.main()
