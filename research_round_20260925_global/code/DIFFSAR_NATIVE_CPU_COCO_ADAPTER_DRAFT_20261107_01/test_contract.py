"""Metadata/API wiring tests with stand-ins; no scientific package, data or AP run."""
import copy,importlib.util,unittest
from pathlib import Path
S=importlib.util.spec_from_file_location('adapter',str(Path(__file__).with_name('worker.py')))
w=importlib.util.module_from_spec(S);S.loader.exec_module(w)
class Contract(unittest.TestCase):
 def test_segm_copy_only_drops_bbox_never_caps_or_remaps(self):
  rows=[dict(image_id=7,category_id=1,score=i/200.0,bbox=[0,0,1,1],segmentation={'size':[2,2],'counts':'4'}) for i in range(137)]
  original=copy.deepcopy(rows);out=w.segm_copy(rows)
  self.assertEqual(rows,original);self.assertEqual(len(out),137)
  self.assertTrue(all('bbox' not in x for x in out))
  self.assertEqual([x['score'] for x in out],[x['score'] for x in rows])
  self.assertTrue(all(x['category_id']==1 for x in out))
 def test_native_population_does_not_require18700_or_nonempty_each_image(self):
  images=[dict(image_id=i,height=2,width=2) for i in range(187)]
  rows=[dict(image_id=7,category_id=1,score=i/200.,segmentation={'size':[2,2],'counts':'4'}) for i in range(137)]
  counts=w.validate_predictions(rows,images)
  self.assertEqual(counts[7],137);self.assertEqual(counts.get(6,0),0)
  self.assertEqual(sum(w.validate_predictions([],images).values()),0)
  for bad in [dict(rows[0],category_id=0),dict(rows[0],image_id=999),dict(rows[0],score=float('nan'))]:
   with self.assertRaises(ValueError):w.validate_predictions([bad],images)
 def test_one_global_official_api_pass_with_eval_only_maxdets(self):
  calls=[]
  class DT:
   def __init__(self):self.dataset={}
   def createIndex(self):calls.append('empty_createIndex')
  class GT:
   dataset={'images':[{'id':i} for i in range(187)],'categories':[{'id':1}]}
   def loadRes(self,rows):calls.append(('loadRes',len(rows),all('bbox' not in x for x in rows)));return DT()
  class Params:pass
  class Eval:
   def __init__(self,gt,dt,kind):self.params=Params();self.stats=[-1.0]*12;calls.append(('init',kind))
   def evaluate(self):calls.append(('evaluate',self.params.imgIds,self.params.catIds,self.params.maxDets))
   def accumulate(self):calls.append('accumulate')
   def summarize(self):calls.append('summarize')
  backend={'COCO':DT,'COCOeval':Eval};images=list(range(187))
  rows=[{'image_id':0,'category_id':1,'score':0.5,'bbox':[0,0,1,1],'segmentation':{'size':[2,2],'counts':'4'}} for _ in range(137)]
  result=w.evaluate_population(backend,GT(),rows,images)
  self.assertEqual(result,[-1.0]*12)
  self.assertEqual(calls.count('accumulate'),1);self.assertEqual(calls.count('summarize'),1)
  self.assertIn(('loadRes',137,True),calls);self.assertIn(('evaluate',images,[1],[1,10,100]),calls)
  calls.clear();w.evaluate_population(backend,GT(),[],images)
  self.assertIn('empty_createIndex',calls)
 def test_external_qualification_cannot_be_replaced_by_partial_or_wrong_refs(self):
  spec={k:k for k in ('predictions_ref','native_prediction_report_ref','native_prediction_qualification_ref',
   'checkpoint_ref','endpoint_qualification_ref','image_manifest_ref','evaluation_gt_ref')}
  a=dict(spec,fixed_endpoint_qualified=True,native_prediction_qualified=True,full187_verified=True,
   endpoint_updates=37050,GT_variant='ORIGINAL_FIXED_INTEGER_GT187_380',prediction_source_ref='actual_source')
  w.validate_external_admission(spec,a)
  for k,v in [('fixed_endpoint_qualified',False),('native_prediction_qualified',False),('endpoint_updates',512),
   ('predictions_ref','different_file'),('full187_verified',False)]:
   bad=dict(a);bad[k]=v
   with self.assertRaises(ValueError):w.validate_external_admission(spec,bad)
if __name__=='__main__':unittest.main()
