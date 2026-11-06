"""Unexecuted frozen N26 GT sensitivity diagnostic; import is stdlib-only."""
import math
import statistics
import json
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import argparse
import datetime
import importlib
import importlib.util
import os
import sys
import time
import traceback

def reference(path):
 p=Path(path);need(p.is_absolute() and p==p.resolve(strict=True) and p.is_file(),'Canonical regular input path')
 before=p.stat();h=hashlib.sha256();total=0
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block);total+=len(block)
 after=p.stat();ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(ident(before)==ident(after),'Input changed during read')
 return {'path':str(p),'bytes':total,'sha256':h.hexdigest()}

def read_input(ref,state,save,parse):
 state['bytes_read']=False;save()
 p=Path(ref['path']);need(p.is_absolute() and p==p.resolve(strict=True) and p.is_file(),'Canonical regular input path')
 before=p.stat();raw=p.read_bytes();state['bytes_read']=True;save();after=p.stat()
 ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(ident(before)==ident(after),'Input changed during byte read')
 actual={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};state['actual_ref']=actual;save()
 need(actual==ref,'Pinned input bytes before parsing')
 if not parse:return None
 state['JSON_parsed']=False;save()
 result=json.loads(raw.decode('utf-8'))
 state['JSON_parsed']=True;save();return result

def identical(a,b):
 return json.dumps(a,sort_keys=True,allow_nan=False,separators=(',',':'))==json.dumps(b,sort_keys=True,allow_nan=False,separators=(',',':'))

def validate_gt_pair(integer,floating):
 """Identity join, never reconstruct/round geometry or claim candidate float truth."""
 need(type(integer) is dict and type(floating) is dict,'Original full COCO GT dictionaries')
 need(identical({k:v for k,v in integer.items() if k not in ('images','annotations')},
  {k:v for k,v in floating.items() if k not in ('images','annotations')}),'GT non-geometric dataset fields unchanged')
 need(len(integer['categories'])==1 and type(integer['categories'][0]['id']) is int
  and integer['categories'][0]['id']==1,'Fixed original category1')
 maps=[]
 for data in (integer,floating):
  ims=data['images'];anns=data['annotations']
  need(len(ims)==187 and len({x['id'] for x in ims})==187
   and all(type(x['id']) is int and type(x['height']) is int and type(x['width']) is int
    and min(x['height'],x['width'])>0 for x in ims),'Fixed unique187 original image identities/dimensions')
  by_id={x['id']:x for x in ims}
  need(len(anns)==380 and len({x['id'] for x in anns})==380,'Fixed distinct380 annotation identities')
  for a in anns:
   need(type(a['id']) is int and type(a['image_id']) is int and a['image_id'] in by_id
    and type(a['category_id']) is int and a['category_id']==1 and 'iscrowd' in a,'Original annotation image/category/crowd identity')
  maps.append((by_id,{x['id']:x for x in anns}))
 need(identical(maps[0][0],maps[1][0]),'GT non-geometric image fields/dimensions unchanged')
 need(set(maps[0][1])==set(maps[1][1]),'Same380 annotation identities')
 for aid,a in maps[0][1].items():
  strip=lambda x:{k:v for k,v in x.items() if k not in ('segmentation','bbox','area')}
  need(identical(strip(a),strip(maps[1][1][aid])),'GT non-geometric annotation fields unchanged')
 return integer['images']

def validate_prediction_pair(reference,heads,images):
 """Preserve frozen scores and all record positions; no inferred unique query ID."""
 need(set(heads)==set(map(str,SEEDS)),'All three fixed N26 seeds, no seed selection')
 by_id={x['id']:x for x in images};need(len(images)==len(by_id)==187,'Fixed187 GT image identities')
 counts={}
 for role,rows in [('R',reference)]+[(str(s),heads[str(s)]) for s in SEEDS]:
  need(type(rows) is list and len(rows)==18700,'Whole frozen18700 original result list')
  per_image=Counter()
  for p in rows:
   need(type(p['image_id']) is int and p['image_id'] in by_id
    and type(p['category_id']) is int and p['category_id']==1,'Original image and already-mapped category1')
   need(type(p['score']) in (int,float) and math.isfinite(p['score']),'Finite frozen score, no normalization/recomputation')
   im=by_id[p['image_id']];seg=p['segmentation'];box=p['bbox']
   need(type(seg) is dict and seg['size']==[im['height'],im['width']] and type(seg['counts']) is str,'Original compressed RLE and dimensions')
   need(type(box) is list and len(box)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in box),'Original finite bbox preserved before segm-only API copy')
   per_image[p['image_id']]+=1
  need(set(per_image)==set(by_id) and all(n==100 for n in per_image.values()),'All187 original images each100 frozen records')
  counts[role]=len(rows)
 strip=lambda x:{k:v for k,v in x.items() if k!='score'}
 for seed in SEEDS:
  for a,b in zip(reference,heads[str(seed)]):
   need(identical(strip(a),strip(b)),'Frozen per-record non-score fields/order unchanged, including explicit query fields if present')
 return counts

def evaluate_case(adapter,backend,GT_json,predictions):
 """New original COCO working object per case, since COCOeval mutates its GT."""
 gt=backend['COCO']();gt.dataset=deepcopy(GT_json);gt.createIndex()
 return adapter.evaluate_population(backend,gt,predictions,[x['id'] for x in GT_json['images']])

def ledger_input_refs(ledger):
 need(ledger['schema']=='N26-independent-frozen-numeric-input-locator-ledger-v1','Independent frozen locator ledger')
 need([x['seed'] for x in ledger['seeds']]==list(SEEDS),'Original three seeds in fixed order')
 refs={'integer_GT':ledger['integer_GT_ref_from_prior_independent_mapping_metadata'],
  'candidate_float_GT':ledger['local_candidate_float_GT_ref_from_prior_independent_mapping_metadata'],
  'R_predictions':ledger['common_R_predictions_ref']}
 for row in ledger['seeds']:
  need(row['fixed_epoch']==24 and row['score_rule']=='S0*qhat','Frozen E24 score-source role')
  refs['quality_'+str(row['seed'])]=row['complete_quality_predictions_ref']
  refs['score_audit_'+str(row['seed'])]=row['per_query_score_audit_ref']
 return refs

def validate_input_binding(ledger,execution_refs):
 originals=ledger_input_refs(ledger)
 need(set(originals)==set(execution_refs),'Exactly two GT, commonR, three quality and three audit refs')
 for role,a in originals.items():
  b=execution_refs[role]
  need(a['bytes']==b['bytes'] and a['sha256']==b['sha256'],'Exact frozen content identity '+role)
 return originals

SEEDS=(202610172601,202610172602,202610172603)
ROLES=('R',)+tuple(map(str,SEEDS))
EXPECTED_INTEGER_AP={'R':68.09938494745545,'202610172601':69.42361315215425,
 '202610172602':69.63714218423894,'202610172603':69.55134548319735}
AP_TOLERANCE_PERCENTAGE_POINTS=1e-8
STAT_NAMES=('AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large')
OPERATION='FROZEN_N26_RANKING_INTEGER_FLOAT_GT_DIAGNOSTIC'
SPEC_SCHEMA='Root-frozen-N26-integer-float-GT-diagnostic-spec-v1'
REPORT_SCHEMA='frozen-N26-integer-float-GT-diagnostic-report-v1'
LIBRARY_MODULES={'numpy':'numpy','pycocotools_coco':'pycocotools.coco',
 'pycocotools_cocoeval':'pycocotools.cocoeval','pycocotools_mask':'pycocotools.mask','pycocotools_extension':'pycocotools._mask'}
LEDGER_ID={'bytes':17191,'sha256':'3e8825926f76fe571e20ed7c905cfe6b5f3edc56e0ba3fa17992d0b70895b9c6'}
ADAPTER_ID={'bytes':17615,'sha256':'f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748'}
INPUT_ROLES=('integer_GT','candidate_float_GT','R_predictions')+tuple('quality_'+str(s) for s in SEEDS)+tuple('score_audit_'+str(s) for s in SEEDS)
PRELOCK={'population_role':'INTERNAL_DEV187_ALREADY_USED_FOR_DEVELOPMENT_NOT_OFFICIAL_TEST',
 'population_images':187,'GT_annotations':380,'category_ids':[1],'frozen_epoch':24,'seeds':list(SEEDS),
 'expected_integer_AP_percent':EXPECTED_INTEGER_AP,'absolute_AP_tolerance_percentage_points':AP_TOLERANCE_PERCENTAGE_POINTS,
 'integer_reproduction_must_pass_before_float':True,'candidate_float_is_official_or_physical_truth':False,
 'primary_integer_protocol_replaced':False,'iou_type':'segm','max_dets':[1,10,100],
 'original_predictions_per_role':18700,'original_predictions_per_image':100,
 'prediction_transform':'DEEPCOPY_DROP_BBOX_ONLY_INSIDE_CANONICAL_API',
 'no_score_change':True,'no_record_reorder':True,'no_category_remap':True,
 'full187_once_per_case':True,'no_shard_AP_average':True,'GT_geometry_difference_only':['segmentation','bbox','area'],
 'new_COCO_GT_working_deepcopy_per_case':True,'score_audit_JSON_consumed':False,
 'no_retraining_or_qhat_recomputation':True,'no_checkpoint_or_training_source_read':True,'automatic_retry':False}

def valid_ref(ref):
 need(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'Exact file reference')
 p=Path(ref['path']);need(p.is_absolute() and '..' not in p.parts and str(p)==ref['path'],'Current-platform canonical absolute path')
 need(type(ref['bytes']) is int and ref['bytes']>0 and type(ref['sha256']) is str
  and len(ref['sha256'])==64 and all(c in '0123456789abcdef' for c in ref['sha256']),'Positive bytes and SHA256')

def validate_spec(spec):
 need(spec.get('schema')==SPEC_SCHEMA and spec.get('operation')==OPERATION,'Independent frozen N26 diagnostic operation; no native37050 admission')
 need(set(spec)=={'schema','operation','run_id','prelock','source_refs','library_sources','locator_ledger_ref',
  'input_refs','output_directory','CPU_budget','external_Root_conditions'},'Thin one-spec diagnostic interface')
 need(spec['prelock']==PRELOCK and spec['run_id'].startswith('N26-frozen-ranking-GT-'),'New fixed diagnostic prelock')
 need(set(spec['source_refs'])=={'worker','canonical_coco_adapter'} and set(spec['library_sources'])==set(LIBRARY_MODULES),'Pinned actual worker and canonical API/library sources')
 need(set(spec['input_refs'])==set(INPUT_ROLES),'All frozen original input roles')
 for ref in [spec['locator_ledger_ref']]+list(spec['input_refs'].values())+list(spec['source_refs'].values())+list(spec['library_sources'].values()):valid_ref(ref)
 for key,value in LEDGER_ID.items():need(spec['locator_ledger_ref'][key]==value,'Exact independent locator metadata identity')
 for key,value in ADAPTER_ID.items():need(spec['source_refs']['canonical_coco_adapter'][key]==value,'Fixed pure evaluate_population API source')
 required=('new_diagnostic_prelock_granted','old_frozen_input_refs_newly_closed','exclusive_CPU_only_owner',
  'actual_owner_birth_verified','CPU_budget_enforced_externally','canonical_COCO_sources_explicitly_selected')
 need(set(spec['external_Root_conditions'])==set(required) and all(spec['external_Root_conditions'][k] is True for k in required),
  'Root actual new owner/resource/input admission remains an external prerequisite')
 budget=spec['CPU_budget']
 need(set(budget)=={'affinity','threads','RSS_limit_bytes','free_RAM_floor_bytes','start_RAM_bytes','deadline_seconds'},'Root future CPU resource budget')
 need(type(budget['affinity']) is list and len(set(budget['affinity']))==len(budget['affinity'])>0
  and all(type(i) is int and i>=0 for i in budget['affinity']),'Explicit CPU affinity')
 need(all(type(budget[k]) is int and budget[k]>0 for k in budget if k!='affinity')
  and budget['threads']<=len(budget['affinity']) and budget['start_RAM_bytes']>=budget['free_RAM_floor_bytes'],'Positive externally enforced CPU bounds')
 out=Path(spec['output_directory']);need(out.is_absolute() and '..' not in out.parts
  and str(out)==spec['output_directory'] and out!=Path(out.anchor),'Fresh absolute output directory')
 for ref in [spec['locator_ledger_ref']]+list(spec['input_refs'].values())+list(spec['source_refs'].values())+list(spec['library_sources'].values()):
  need(Path(ref['path'])!=out and out not in Path(ref['path']).parents,'Outputs outside frozen originals')

def need(condition,message):
 if not condition:raise ValueError(message)

def metrics(stats):
 need(len(stats)==12 and all(type(x) in (int,float) and math.isfinite(x) for x in stats),'Finite actual COCO12 summary')
 return {'raw_stats_fraction_with_minus1_undefined':list(stats),
  'percent':{k:100*v if v>=0 else None for k,v in zip(STAT_NAMES,stats)}}

def run_diagnostic(evaluate,report,save):
 """Evaluate the full population once per role/GT through supplied original COCO API."""
 initial=lambda:{role:dict(COCO_math_started=False,COCO_math_completed=None,AP_computed=None,metrics=None) for role in ROLES}
 report.update(integer_results=initial(),integer_AP_reproduction_passed=None,float_started=False,
  float_results=initial(),paired_summary=None);save()
 for role in ROLES:
  row=report['integer_results'][role];row.update(COCO_math_started=True,COCO_math_completed=False,AP_computed=False);save()
  returned=evaluate(role,'integer');row['COCO_math_completed']=True;save()
  result=metrics(returned)
  row.update(COCO_math_completed=True,AP_computed=True,metrics=result);save()
 failures=[]
 for role in ROLES:
  actual=report['integer_results'][role]['metrics']['percent']['AP']
  delta=None if actual is None else abs(actual-EXPECTED_INTEGER_AP[role])
  row=report['integer_results'][role]
  row.update(expected_AP_percent=EXPECTED_INTEGER_AP[role],absolute_difference_percentage_points=delta,
   AP_reproduction_passed=delta is not None and delta<=AP_TOLERANCE_PERCENTAGE_POINTS)
  if not row['AP_reproduction_passed']:failures.append(role)
 report['integer_AP_reproduction_passed']=not failures;save()
 need(not failures,'Integer AP reproduction failed; float evaluation prohibited: '+','.join(failures))
 report['float_started']=True;save()
 for role in ROLES:
  row=report['float_results'][role];row.update(COCO_math_started=True,COCO_math_completed=False,AP_computed=False);save()
  returned=evaluate(role,'candidate_float');row['COCO_math_completed']=True;save()
  result=metrics(returned)
  row.update(COCO_math_completed=True,AP_computed=True,metrics=result);save()
 summary={}
 for variant,key in [('integer','integer_results'),('candidate_float','float_results')]:
  ref=report[key]['R']['metrics']['percent']['AP'];deltas=[];paired=[]
  for seed in SEEDS:
   value=report[key][str(seed)]['metrics']['percent']['AP']
   delta=None if ref is None or value is None else value-ref
   paired.append({'seed':seed,'R_AP_percent':ref,'quality_AP_percent':value,'delta_percentage_points':delta})
   if delta is not None:deltas.append(delta)
  summary[variant]={'paired_seeds':paired,'n_prespecified_seeds':3,'n_defined_deltas':len(deltas),
   'mean_delta_percentage_points':statistics.mean(deltas) if len(deltas)==3 else None,
   'sample_SD_delta_percentage_points':statistics.stdev(deltas) if len(deltas)==3 else None}
 report['paired_summary']=summary;save()

def main():
 started=time.monotonic();parser=argparse.ArgumentParser()
 parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True);args=parser.parse_args()
 spec_ref=reference(str(Path(args.spec).resolve(strict=True)))
 need(spec_ref['sha256']==args.spec_sha,'Root explicit new diagnostic spec identity')
 spec=json.loads(Path(spec_ref['path']).read_bytes());validate_spec(spec)
 need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Own selected worker source')
 out=Path(spec['output_directory']);need(not out.exists(),'No overwrite/retry of any old diagnostic output')
 out.mkdir(mode=0o700)
 empty_cases=lambda:{r:dict(COCO_math_started=False,COCO_math_completed=None,AP_computed=None,metrics=None) for r in ROLES}
 report={'schema':REPORT_SCHEMA,'operation':OPERATION,'run_id':spec['run_id'],'status':'STARTED',
  'spec_ref':spec_ref,'prelock':PRELOCK,'source_refs':spec['source_refs'],'library_sources':spec['library_sources'],
  'locator_ledger_ref':spec['locator_ledger_ref'],'input_refs':spec['input_refs'],'CPU_budget':spec['CPU_budget'],
  'external_Root_conditions':spec['external_Root_conditions'],'qualified':False,'external_Root_postcheck_required':True,
  'pid':os.getpid(),'ppid':os.getppid(),'platform':sys.platform,
  'input_state':{r:{'bytes_read':None,'JSON_parsed':None} for r in INPUT_ROLES},
  'integer_results':empty_cases(),'float_results':empty_cases(),'integer_AP_reproduction_passed':None,
  'float_started':False,'paired_summary':None,'GPU_numerical_execution':False,'model_checkpoint_read':False,
  'training_source_read':False,'training_credit':0,'qhat_recomputed':False,'scores_recomputed':False,
  'score_audit_JSON_consumed':False,'qhat_multiplication_newly_verified':False,'implicit_unique_query_identity_proven':False,
  'original_encoder_source_identity_newly_verified':False,'candidate_float_is_official_or_physical_truth':False,
  'primary_integer_protocol_replaced':False,'automatic_retry':False}
 def save():
  report['elapsed_seconds']=time.monotonic()-started;temporary=out/'REPORT.next'
  with temporary.open('w',encoding='utf-8') as f:
   json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(temporary,out/'REPORT.json')
 def check():
  need(time.monotonic()-started<spec['CPU_budget']['deadline_seconds'],'Root finite diagnostic deadline')
  need(not any(m=='torch' or m.startswith('torch.') for m in sys.modules),'No Torch/model or GPU execution')
  need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only Root child environment')
  need(all(os.environ.get(k)==str(spec['CPU_budget']['threads']) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Explicit Root CPU thread limits')
 def verify(ref):need(reference(ref['path'])==ref,'Selected input/source identity unchanged')
 code=1;save()
 try:
  check()
  for ref in list(spec['source_refs'].values())+list(spec['library_sources'].values())+[spec['locator_ledger_ref']]:verify(ref)
  ledger=json.loads(Path(spec['locator_ledger_ref']['path']).read_bytes())
  originals=validate_input_binding(ledger,spec['input_refs']);report['original_ledger_input_refs']=originals;save()
  # Root explicitly selects these exact files. A version name is not a source-identity check.
  modules={name:importlib.import_module(path) for name,path in LIBRARY_MODULES.items()}
  actual={name:reference(module.__file__) for name,module in modules.items()}
  need(actual==spec['library_sources'],'Actual canonical COCO/mask/compiled-extension/NumPy sources')
  report['actual_library_refs']=actual;report['numpy_version']=modules['numpy'].__version__;save();check()
  helper_spec=importlib.util.spec_from_file_location('frozen_N26_canonical_COCO_API',spec['source_refs']['canonical_coco_adapter']['path'])
  helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
  need(callable(helper.evaluate_population),'Original frozen pure COCO API helper, never native main')
  GT={variant:read_input(spec['input_refs'][role],report['input_state'][role],save,True)
   for variant,role in [('integer','integer_GT'),('candidate_float','candidate_float_GT')]}
  images=validate_gt_pair(GT['integer'],GT['candidate_float'])
  report.update(raw_GT_non_geometry_identity_checked_before_any_COCO=True,
   actual_GT_images=187,actual_GT_annotations=380,actual_GT_category_ids=[1]);save()
  predictions={'R':read_input(spec['input_refs']['R_predictions'],report['input_state']['R_predictions'],save,True)}
  for seed in SEEDS:
   role='quality_'+str(seed);predictions[str(seed)]=read_input(spec['input_refs'][role],report['input_state'][role],save,True)
  report['prediction_counts']=validate_prediction_pair(predictions['R'],{str(s):predictions[str(s)] for s in SEEDS},images)
  report.update(all_non_score_fields_and_record_order_exact=True,all_explicit_query_fields_if_present_exact=True,
   scores_used_as_frozen=True,category_remap_applied=False);save()
  for seed in SEEDS:
   role='score_audit_'+str(seed);read_input(spec['input_refs'][role],report['input_state'][role],save,False)
  check()
  backend={'COCO':modules['pycocotools_coco'].COCO,'COCOeval':modules['pycocotools_cocoeval'].COCOeval}
  def evaluate(role,variant):
   before=time.monotonic()
   result=evaluate_case(helper,backend,GT[variant],predictions[role])
   key='integer_results' if variant=='integer' else 'float_results'
   report[key][role]['COCO_math_seconds']=time.monotonic()-before
   return result
  run_diagnostic(evaluate,report,save)
  check()
  for ref in [spec_ref,spec['locator_ledger_ref']]+list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(spec['input_refs'].values()):verify(ref)
  report.update(all_selected_sources_and_frozen_inputs_unchanged_after=True,status='DIAGNOSTIC_COMPLETE_PENDING_ROOT');code=0
 except BaseException as error:
  report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());code=1
 finally:
  report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
  print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','integer_AP_reproduction_passed','float_started')}),flush=True)
 return code

if __name__=='__main__':raise SystemExit(main())
