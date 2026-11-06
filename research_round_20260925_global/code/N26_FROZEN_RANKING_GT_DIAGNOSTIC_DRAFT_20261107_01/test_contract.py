"""Stdlib/API stand-ins only; never real COCO, GT, prediction, audit or AP."""
import copy
import unittest
import tempfile
import hashlib
from pathlib import Path
import worker


class ContractTests(unittest.TestCase):
 def test_float_is_never_called_when_any_integer_reproduction_fails(self):
  calls=[]
  def evaluate(role,variant):
   calls.append((role,variant))
   ap=worker.EXPECTED_INTEGER_AP[role]
   return [ap/100+(1e-6 if role==str(worker.SEEDS[1]) else 0)]+[-1.0]*11
  report={}
  with self.assertRaisesRegex(ValueError,'Integer AP reproduction'):
   worker.run_diagnostic(evaluate,report,lambda:None)
  self.assertTrue(calls)
  self.assertTrue(all(variant=='integer' for _,variant in calls))
  self.assertFalse(report['integer_AP_reproduction_passed'])
  self.assertIsNone(report['paired_summary'])
  self.assertFalse(report['float_started'])

 def test_native_endpoint_operation_cannot_be_reused_for_this_diagnostic(self):
  with self.assertRaisesRegex(ValueError,'Independent frozen N26 diagnostic'):
   worker.validate_spec({'schema':worker.SPEC_SCHEMA,'operation':'CPU_SAR_NATIVE_FULL187_FIXED37050_COCO_SEGM_V01'})

 def test_input_identity_failure_preserves_read_state_without_parsing_or_overwriting(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'standin.json';path.write_bytes(b'{"standin":1}')
   ref={'path':str(path),'bytes':len(path.read_bytes()),'sha256':'0'*64}
   state={'bytes_read':None,'JSON_parsed':None}
   with self.assertRaisesRegex(ValueError,'Pinned input bytes'):
    worker.read_input(ref,state,lambda:None,parse=True)
   self.assertTrue(state['bytes_read']);self.assertIsNone(state['JSON_parsed'])
   self.assertEqual(path.read_bytes(),b'{"standin":1}')
   path.write_bytes(b'{invalid')
   ref.update(bytes=8,sha256=hashlib.sha256(b'{invalid').hexdigest())
   state={'bytes_read':None,'JSON_parsed':None}
   with self.assertRaises(ValueError):worker.read_input(ref,state,lambda:None,parse=True)
   self.assertTrue(state['bytes_read']);self.assertIs(state['JSON_parsed'],False)

 def test_gt_pair_allows_only_geometry_changes_for_same_annotation_identities(self):
  integer={'images':[{'id':i,'height':8,'width':9} for i in range(187)],
   'categories':[{'id':1,'name':'ship'}],
   'annotations':[{'id':i,'image_id':i%187,'category_id':1,'iscrowd':0,
    'bbox':[0,0,1,1],'area':1,'segmentation':[[0,0,1,0,1,1]]} for i in range(380)]}
  floating=copy.deepcopy(integer)
  floating['annotations'][0].update(bbox=[0.1,0.2,0.9,1.1],area=0.99)
  self.assertEqual(len(worker.validate_gt_pair(integer,floating)),187)
  floating['annotations'][0]['iscrowd']=1
  with self.assertRaisesRegex(ValueError,'non-geometric'):
   worker.validate_gt_pair(integer,floating)

 def test_frozen_predictions_preserve_all_non_score_fields_and_record_order(self):
  images=[{'id':i,'height':8,'width':9} for i in range(187)]
  base=[{'image_id':im['id'],'category_id':1,'query_id':q,'score':0.5,
   'bbox':[0,0,1,1],'segmentation':{'size':[8,9],'counts':'stand-in'}}
   for im in images for q in range(100)]
  heads={str(s):copy.deepcopy(base) for s in worker.SEEDS}
  for rows in heads.values():rows[0]['score']=0.25
  before=copy.deepcopy(heads)
  self.assertEqual(worker.validate_prediction_pair(base,heads,images)['R'],18700)
  self.assertEqual(heads,before)
  heads[str(worker.SEEDS[0])][0],heads[str(worker.SEEDS[0])][1]=heads[str(worker.SEEDS[0])][1],heads[str(worker.SEEDS[0])][0]
  with self.assertRaisesRegex(ValueError,'non-score'):
   worker.validate_prediction_pair(base,heads,images)

 def test_all_prespecified_seeds_are_paired_to_same_R_without_undefined_fabrication(self):
  calls=[]
  def evaluate(role,variant):
   calls.append((role,variant))
   ap=worker.EXPECTED_INTEGER_AP[role] if variant=='integer' else 60+worker.ROLES.index(role)
   return [ap/100]+[-1.0]*11
  report={};worker.run_diagnostic(evaluate,report,lambda:None)
  self.assertEqual(calls,[(r,v) for v in ('integer','candidate_float') for r in worker.ROLES])
  self.assertTrue(report['integer_AP_reproduction_passed'])
  summary=report['paired_summary']['candidate_float']
  self.assertEqual([x['seed'] for x in summary['paired_seeds']],list(worker.SEEDS))
  self.assertAlmostEqual(summary['mean_delta_percentage_points'],2.0)
  self.assertAlmostEqual(summary['sample_SD_delta_percentage_points'],1.0)
  self.assertIsNone(report['float_results']['R']['metrics']['percent']['AP50'])
  self.assertEqual(report['float_results']['R']['metrics']['raw_stats_fraction_with_minus1_undefined'][1],-1.0)

 def test_each_case_gets_isolated_GT_even_when_COCO_mutates_working_annotations(self):
  class COCO:
   def createIndex(self):self.indexed=True
  class Adapter:
   def __init__(self):self.observed=[]
   def evaluate_population(self,backend,gt,predictions,image_ids):
    self.observed.append(copy.deepcopy(gt.dataset))
    gt.dataset['annotations'][0].update(segmentation={'changed':'RLE'},ignore=1)
    return [-1.0]*12
  original={'images':[{'id':i} for i in range(187)],'annotations':[{'segmentation':[[0,0,1,1]]}]}
  untouched=copy.deepcopy(original);adapter=Adapter();predictions=[]
  worker.evaluate_case(adapter,{'COCO':COCO},original,predictions)
  worker.evaluate_case(adapter,{'COCO':COCO},original,predictions)
  self.assertEqual(original,untouched)
  self.assertEqual(adapter.observed,[untouched,untouched])

 def test_ledger_content_identities_are_closed_even_when_staged_paths_change(self):
  ref=lambda i:{'path':'D:/original/'+str(i),'bytes':i,'sha256':str(i%10)*64}
  ledger={'schema':'N26-independent-frozen-numeric-input-locator-ledger-v1',
   'common_R_predictions_ref':ref(1),'integer_GT_ref_from_prior_independent_mapping_metadata':ref(2),
   'local_candidate_float_GT_ref_from_prior_independent_mapping_metadata':ref(3),
   'seeds':[{'seed':s,'fixed_epoch':24,'score_rule':'S0*qhat',
    'complete_quality_predictions_ref':ref(4+i),'per_query_score_audit_ref':ref(7+i)}
    for i,s in enumerate(worker.SEEDS)]}
  originals=worker.ledger_input_refs(ledger)
  staged={k:dict(v,path='/staged/'+k) for k,v in originals.items()}
  self.assertEqual(worker.validate_input_binding(ledger,staged),originals)
  staged['quality_'+str(worker.SEEDS[1])]['sha256']='f'*64
  with self.assertRaisesRegex(ValueError,'frozen content identity'):
   worker.validate_input_binding(ledger,staged)

 def test_failed_math_keeps_attempted_false_and_unattempted_none(self):
  def failed(role,variant):raise RuntimeError('stand-in math failure')
  report={}
  with self.assertRaises(RuntimeError):worker.run_diagnostic(failed,report,lambda:None)
  self.assertIs(report['integer_results']['R']['AP_computed'],False)
  self.assertIsNone(report['integer_results'][str(worker.SEEDS[0])]['AP_computed'])
  self.assertIsNone(report['float_results']['R']['AP_computed'])
  self.assertIsNone(report['paired_summary'])

 def test_returned_math_is_complete_even_if_summary_validation_then_fails(self):
  report={}
  with self.assertRaisesRegex(ValueError,'COCO12'):
   worker.run_diagnostic(lambda role,variant:[0.5],report,lambda:None)
  self.assertIs(report['integer_results']['R']['COCO_math_completed'],True)
  self.assertIs(report['integer_results']['R']['AP_computed'],False)
  self.assertIsNone(report['float_results']['R']['AP_computed'])


if __name__=='__main__':unittest.main()
