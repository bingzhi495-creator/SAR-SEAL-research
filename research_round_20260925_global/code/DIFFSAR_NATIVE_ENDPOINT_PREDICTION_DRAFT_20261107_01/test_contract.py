"""Metadata/IO behavior tests only; no Torch/model/checkpoint/GT is loaded."""
import copy,importlib.util,io,json,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('endpoint',str(Path(__file__).with_name('worker.py')))
w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class Contract(unittest.TestCase):
 def test_only_root_qualified_fixed37050_correct_source_endpoint(self):
  state=dict(schema='DiffSAR-native-full-state-v1',mode='FORMAL',updates=37050,
   protocol={'fixed_target':37050,'proposals':500},loader={'consumed_indices':37050},scientific_pins={'sources':{'worker':{'sha256':'abc'}}})
  admission=dict(qualified=True,selection='FIXED37050_NO_BEST',updates=37050,
   training_protocol=state['protocol'],training_scientific_pins=state['scientific_pins'])
  w.validate_endpoint_header(state,admission)
  for field,value in [('updates',36999),('mode','RESUME_VALIDATION'),('scientific_pins',{'different':'source'})]:
   bad=copy.deepcopy(state);bad[field]=value
   with self.assertRaises(ValueError):w.validate_endpoint_header(bad,admission)
  bad=copy.deepcopy(admission);bad['qualified']=False
  with self.assertRaises(ValueError):w.validate_endpoint_header(state,bad)
 def test_export_keeps_every_native_post_nms_mask_above100(self):
  rows=[{'image_id':7,'category_id':0,'score':i/200.0,'segmentation':{'size':[2,2],'counts':'4'},'bbox':[0,0,1,1]} for i in range(137)]
  out=io.StringIO();count=w.write_prediction_rows(out,rows,7,{'0':1},first=True)
  decoded=json.loads('['+out.getvalue()+']')
  self.assertEqual(count,137);self.assertEqual(len(decoded),137)
  self.assertEqual([x['score'] for x in decoded],[x['score'] for x in rows])
  self.assertTrue(all(x['category_id']==1 for x in decoded))
 def test_author1000_and_eval100_are_distinct_no_prediction_cap(self):
  facts=dict(training_proposals=500,inference_proposals=1000,sampling_steps=1,use_nms=True,
   prediction_limit=None,evaluation_max_dets=[1,10,100])
  w.validate_author_inference(facts)
  for key,value in [('inference_proposals',100),('prediction_limit',100),('sampling_steps',1000)]:
   bad=dict(facts);bad[key]=value
   with self.assertRaises(ValueError):w.validate_author_inference(bad)
if __name__=='__main__':unittest.main()
