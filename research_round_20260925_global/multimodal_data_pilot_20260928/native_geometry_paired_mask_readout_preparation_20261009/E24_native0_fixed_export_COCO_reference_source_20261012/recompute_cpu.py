import json,hashlib,math
from pathlib import Path
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def reference(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def gate(p,phase):
 r=read(p);assert Path(p).parent.name==phase
 assert r['status']=='ACTUAL_OWNED_TERMINAL'and type(r['returncode'])is int and r['returncode']==0 and r['original_identity']
 assert r['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0)and r['root_wait_reaped']is True and r['numeric_birth_absence']['original_birth_absent']is True
 assert r.get('error')is None and r.get('abort')is None and r['wait_errors']is False and r['GPU_observation_errors']==[]
 assert math.isfinite(r['whole_GPU_observed_peak_MiB'])and 0<=r['whole_GPU_observed_peak_MiB']<=8192 and r['whole_GPU_used_limit_MiB']==8192
 assert r['whole_leaf_VRAM_certified']is False and r['GPU_scope']=='whole GPU observed, not per-process WDDM allocation'
 return r

import argparse,importlib.util,inspect,os,sys
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--spec',required=True);parser.add_argument('--acceptance',required=True);parser.add_argument('--owner-execute',action='store_true')
 args=parser.parse_args();assert args.owner_execute
 path=Path(args.spec);target=Path(args.acceptance)
 assert path.is_absolute() and target.is_absolute() and not target.exists()
 spec=read(path)
 assert spec['schema']=='ROOT-fixed-E24-native0-export-COCO-reference-recompute-spec-v1'
 assert target==Path(spec['output_path']) and len(spec['RAW_inputs'])==13
 assert len({x['receipt']['path'] for x in spec['RAW_inputs']})==13
 for row in spec['RAW_inputs']:gate(row['receipt']['path'],row['phase'])
 # Every current and ancestor RAW terminal precedes all hashes and scientific imports.
 for row in spec['sources']:assert reference(row['path'])==row
 spec_ref=reference(path);source_ref=reference(__file__)
 assert source_ref==spec['entry_source']
 admission_ref=reference(spec['CPU_admission_file']);admission=read(admission_ref['path'])
 assert admission['phase']==spec['phase'] and admission['scientific_spec']==spec_ref
 assert admission['entry']==str(Path(__file__).resolve()) and admission['entry_args']==['--spec',str(path),'--acceptance',str(target),'--owner-execute']
 assert admission['allowed_cpus']==[0,1] and admission['cuda_visible_devices']=='' and admission['automatic_retry'] is False
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'])
 import psutil
 assert psutil.Process().cpu_affinity()==[0,1] and 'torch' not in sys.modules
 legacy=read(spec['legacy_Q']['path']);old_record=legacy['mode_results']['native0']
 assert old_record['predictions']==spec['predictions'] and old_record['image_rows']==spec['image_rows']
 assert old_record['checkpoint']==spec['fixed_E24_checkpoint']
 rows=read(spec['image_rows']['path']);predictions=read(spec['predictions']['path'])
 assert len(rows)==187 and len(predictions)==18700
 import numpy as np
 from pycocotools.coco import COCO
 from pycocotools.cocoeval import COCOeval
 import pycocotools.mask as mask_module
 assert np.__version__==spec['numpy_version']
 for module_file in [inspect.getfile(COCO),inspect.getfile(COCOeval),mask_module.__file__,mask_module._mask.__file__,np.__file__,np.core._multiarray_umath.__file__]:
  assert str(Path(module_file).resolve()) in admission['bound_source_sha256']
  assert reference(module_file)['sha256']==admission['bound_source_sha256'][str(Path(module_file).resolve())]
 helper_spec=importlib.util.spec_from_file_location('_sealed_original_official_COCO_only',spec['official_COCO_helper']['path'])
 helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
 assert 'torch' not in sys.modules
 coco=COCO(spec['DEV']['path']);ids=list(coco.imgs)
 assert len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1}
 assert [r['image_id'] for r in rows]==ids
 for index,row in enumerate(rows):
  image=coco.imgs[ids[index]]
  assert row['dataset_index']==index and row['checkpoint_mode']=='native0' and row['mode']=='native0'
  assert row['original_shape']==[image['height'],image['width']]
  assert row['ann_ids']==[ann['id'] for ann in coco.imgToAnns[ids[index]]] and row['annotation_count']==len(row['ann_ids'])
  query_ids=row['prediction_query_ids'];assert len(query_ids)==100 and set(query_ids)==set(range(100))
  group=predictions[index*100:(index+1)*100]
  assert len(group)==100 and [x['query_id'] for x in group]==query_ids
  for prediction in group:
   assert set(prediction)=={'image_id','query_id','category_id','score','segmentation'}
   assert prediction['image_id']==ids[index] and prediction['category_id']==1
   assert type(prediction['score']) in (int,float) and math.isfinite(prediction['score']) and 0<=prediction['score']<=1
   assert set(prediction['segmentation'])=={'size','counts'} and prediction['segmentation']['size']==row['original_shape']
   assert isinstance(prediction['segmentation']['counts'],str)
 metric=helper.coco_metrics(coco,predictions,ids,COCOeval)
 assert len(metric['stats'])==12 and metric['maxDets']==[1,10,100] and metric['original_GT_preserved'] is True
 assert metric['score_filter'] is None and metric['official_COCO_crowd_ignore_semantics'] is True
 assert psutil.Process().cpu_affinity()==[0,1] and 'torch' not in sys.modules
 for row in spec['sources']:assert reference(row['path'])==row
 assert reference(path)==spec_ref and reference(__file__)==source_ref and reference(spec['CPU_admission_file'])==admission_ref
 result=dict(schema='ROOT-fixed-E24-native0-export-COCO-reference-recompute-CPU-qualification-v1',
  status='ROOT_FIXED_E24_NATIVE0_EXPORT_COCO_REFERENCE_RECOMPUTED_CPU',phase=spec['phase'],source=source_ref,spec=spec_ref,
  CPU_admission=admission_ref,RAW_inputs=spec['RAW_inputs'],legacy_Q=spec['legacy_Q'],fixed_E24_checkpoint=spec['fixed_E24_checkpoint'],
  DEV=spec['DEV'],predictions=spec['predictions'],image_rows=spec['image_rows'],official_COCO_helper=spec['official_COCO_helper'],
  COCO=metric,legacy_COCO12=old_record['COCO']['stats'],recomputed_minus_legacy_COCO12=[a-b for a,b in zip(metric['stats'],old_record['COCO']['stats'])],
  images=187,raw_GT=380,predictions_count=18700,all100_queries_per_image_preserved=True,empty_predictions_not_dropped=True,
  original_prediction_order_preserved=True,original_GT_crowd_ignore_flags_preserved=True,
  CPU_execution=dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',Torch_imported=False,numpy_version=np.__version__,thread_environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}),
  model_forward_replayed=False,mask_geometry_reconstructed=False,mask_quality_recomputed=False,training_performed=False,
  reference_selected_before_current_main_result=True,historically_exposed_DEV=True,no_TEST=True,no_efficacy_or_novelty_claim=True)
 with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],acceptance=reference(target))),flush=True)
if __name__=='__main__':main()
