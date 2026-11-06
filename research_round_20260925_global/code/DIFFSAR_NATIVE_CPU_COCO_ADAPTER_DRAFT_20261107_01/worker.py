"""Unexecuted thin native COCO segm adapter; module import uses stdlib only."""
from copy import deepcopy
from collections import Counter
import argparse,datetime,hashlib,importlib,json,math,os,sys,time,traceback
from pathlib import Path,PurePosixPath
def need(c,m):
 if not c:raise ValueError(m)
def segm_copy(predictions):
 out=deepcopy(predictions)
 for row in out:row.pop('bbox',None)
 return out
def validate_predictions(predictions,images):
 need(type(predictions) is list,'Whole original COCO result list')
 by_id={x['image_id']:x for x in images};counts=Counter()
 need(len(images)==len(by_id)==187,'Fixed distinct187 image identities')
 for p in predictions:
  need(type(p['image_id']) is int and p['image_id'] in by_id,'Original population image identity')
  need(type(p['category_id']) is int and p['category_id']==1,'Already mapped original category1; never remap0')
  need(type(p['score']) in (int,float) and math.isfinite(p['score']),'Finite original confidence without rescoring')
  s=p['segmentation'];im=by_id[p['image_id']]
  need(type(s) is dict and s['size']==[im['height'],im['width']] and type(s['counts']) is str,'Original-sized native compressedUTF8 RLE')
  counts[p['image_id']]+=1
 return counts
def evaluate_population(backend,coco_gt,predictions,image_ids):
 """Only official COCO API wiring, no local IoU/matching/AP mathematics."""
 results=segm_copy(predictions)
 if results:coco_dt=coco_gt.loadRes(results)
 else:
  # Official loadRes([]) assumes an element in several releases. Build its empty result dataset through original COCO APIs.
  coco_dt=backend['COCO']();coco_dt.dataset=dict(images=deepcopy(coco_gt.dataset['images']),
   categories=deepcopy(coco_gt.dataset['categories']),annotations=[]);coco_dt.createIndex()
 evaluator=backend['COCOeval'](coco_gt,coco_dt,'segm')
 evaluator.params.imgIds=list(image_ids);evaluator.params.catIds=[1];evaluator.params.maxDets=[1,10,100]
 evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
 stats=[float(x) for x in evaluator.stats]
 need(len(stats)==12 and all(math.isfinite(x) for x in stats),'Actual complete finite COCO summary')
 return stats
def validate_external_admission(spec,a):
 need(a.get('fixed_endpoint_qualified') is True and a.get('native_prediction_qualified') is True
  and a.get('full187_verified') is True and type(a.get('endpoint_updates')) is int and a['endpoint_updates']==37050,
  'Actual external Root fixed37050 and complete native prediction qualification required')
 for k in ('predictions_ref','native_prediction_report_ref','native_prediction_qualification_ref',
  'checkpoint_ref','endpoint_qualification_ref','image_manifest_ref','evaluation_gt_ref'):
  need(a.get(k)==spec[k],'Original externally qualified identity join '+k)
 need(a.get('GT_variant')=='ORIGINAL_FIXED_INTEGER_GT187_380' and 'prediction_source_ref' in a,'Fixed GT/source scope')

SCHEMA='Root-SAR-native-DEV187-COCO-segm-spec-v1'
LEASE_SCHEMA='Root-SAR-native-DEV187-COCO-segm-lease-v1'
OPERATION='CPU_SAR_NATIVE_FULL187_FIXED37050_COCO_SEGM_V01'
LIBRARY_MODULES=dict(numpy='numpy',pycocotools_coco='pycocotools.coco',pycocotools_cocoeval='pycocotools.cocoeval',
 pycocotools_mask='pycocotools.mask',pycocotools_extension='pycocotools._mask')
PRELOCK=dict(population_images=187,GT_annotations=380,GT_variant='ORIGINAL_FIXED_INTEGER_GT187_380',
 population_role='INTERNAL_DEV187_ALREADY_USED_FOR_DEVELOPMENT_NOT_OFFICIAL_TEST',category_ids=[1],
 iou_type='segm',max_dets=[1,10,100],all_population_once=True,no_shard_AP_average=True,
 prediction_pretrim=False,score_change=False,category_remap=False,prediction_transform='DEEPCOPY_DROP_BBOX_ONLY',
 endpoint_updates=37050,require_external_full_native_prediction_qualification=True,
 CPU_affinity=[0,1],threads=2,RSS_limit_bytes=16*1024**3,free_RAM_floor_bytes=16*1024**3,
 start_RAM_bytes=48*1024**3,deadline_seconds=1800,GPU_numerical_execution=False,automatic_retry=False)

def valid_ref(r):
 need(type(r) is dict and set(r)=={'path','bytes','sha256'},'Exact pinned reference')
 p=PurePosixPath(r['path']);need(p.is_absolute() and str(p)==r['path'] and '..' not in p.parts,'Canonical source path')
 need(type(r['bytes']) is int and r['bytes']>0 and type(r['sha256']) is str and len(r['sha256'])==64
  and all(c in '0123456789abcdef' for c in r['sha256']),'Positive bytes and SHA256')

def reference(path):
 p=Path(path);q=p.resolve(strict=True);need(str(q)==str(p) and q.is_file(),'Canonical regular file')
 a=q.stat();h=hashlib.sha256();n=0
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
 z=q.stat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(identity(a)==identity(z),'Source changed during verification')
 return dict(path=str(q),bytes=n,sha256=h.hexdigest())

def verify(r):need(reference(r['path'])==r,'Pinned source identity')

def metadata_refs(spec):
 # Checkpoint is provenance metadata only. This CPU adapter never opens model weights.
 return list(spec['source_refs'].values())+list(spec['library_sources'].values())+[
  spec[k] for k in ('image_manifest_ref','native_prediction_report_ref','endpoint_qualification_ref',
   'native_prediction_qualification_ref','root_lease_ref')]

def validate_spec(spec):
 need(set(spec)=={'schema','operation','run_id','prelock','source_refs','library_sources','image_manifest_ref',
  'evaluation_gt_ref','predictions_ref','native_prediction_report_ref','native_prediction_qualification_ref',
  'checkpoint_ref','endpoint_qualification_ref','root_lease_ref','output_directory'},'Exact thin COCO input interface')
 need(spec['schema']==SCHEMA and spec['operation']==OPERATION and spec['prelock']==PRELOCK,'Locked full-population standard segm math')
 need(spec['run_id'].startswith('SAR-native37050-DEV187-COCO-'),'Task identity')
 need(set(spec['source_refs'])=={'worker'} and set(spec['library_sources'])==set(LIBRARY_MODULES),'Selected adapter/library sources')
 refs=metadata_refs(spec)+[spec[k] for k in ('evaluation_gt_ref','predictions_ref','checkpoint_ref')]
 for r in refs:valid_ref(r)
 p=PurePosixPath(spec['output_directory']);need(p.is_absolute() and str(p)==spec['output_directory'] and '..' not in p.parts and str(p)!='/','Fresh output root')
 need(all(r['path']!=str(p) and not r['path'].startswith(str(p)+'/') for r in refs),'Outputs outside original inputs')

def validate_lease(spec,lease):
 need(lease.get('schema')==LEASE_SCHEMA and lease.get('status')=='ROOT_GRANTED','New future CPU lease, never mutate frozen upstream leases')
 for k in ('operation','run_id','prelock','output_directory','image_manifest_ref','evaluation_gt_ref','predictions_ref',
  'native_prediction_report_ref','native_prediction_qualification_ref','checkpoint_ref','endpoint_qualification_ref'):
  need(lease.get(k)==spec[k],'Root input/source role join '+k)
 need(lease.get('worker_ref')==spec['source_refs']['worker'] and lease.get('single_attempt') is True,'One explicit Root CPU invocation')
 need(lease.get('canonical_COCO_library_sources')==spec['library_sources'],'One explicit canonical COCO/codec source set, not version-name equivalence')
 need(lease.get('prior_project_scientific_groups_reaped') is True and lease.get('exclusive_project_scientific_leaf') is True,'Root existing scientific-owner mode')
 need(lease.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'Current-boot CPU lease')
 validate_external_admission(spec,lease['external_admission']);valid_ref(lease['external_admission']['prediction_source_ref'])
 return lease['external_admission']

def validate_population(manifest,spec):
 need(manifest['schema']=='Root-DiffSAR-DEV187-original-image-input-v1' and manifest['population_role']==PRELOCK['population_role'],'Same already-exposed internal population')
 need(manifest['evaluation_gt_ref']==spec['evaluation_gt_ref'] and manifest['GT_instances_expected']==380
  and manifest['category_id_map']=={'0':1},'Original integerGT identity and already-applied category mapping')
 images=manifest['images'];need(len(images)==187 and len({x['image_id'] for x in images})==187,'Unique187 original logical image IDs')
 for i,r in enumerate(images):need(r['dataset_index']==i and type(r['image_id']) is int,'Locked full population order')
 return images

def validate_native_report(r,spec,admission,images):
 need(r['schema']=='DiffSAR-fixed37050-native1000-prediction-report-v1'
  and r['status']=='NATIVE_ENDPOINT_PREDICTION_COMPLETE_PENDING_ROOT'
  and type(r['worker_exit_code']) is int and r['worker_exit_code']==0,'Actual completed original producer report is required but not sufficient alone')
 need(r['source_refs']['worker']==admission['prediction_source_ref'],'Externally qualified native producer source identity')
 for producer_key,spec_key in [('prediction_ref','predictions_ref'),('image_manifest_ref','image_manifest_ref'),
  ('evaluation_gt_ref','evaluation_gt_ref'),('checkpoint_ref','checkpoint_ref'),('endpoint_qualification_ref','endpoint_qualification_ref')]:
  need(r[producer_key]==spec[spec_key],'Original native product/input join '+producer_key)
 need(r['prelock']['endpoint_updates']==37050 and r['all187_completed'] is True and r['images_completed']==187
  and r['no_prediction_top100_cut'] is True and r['all_post_NMS_masks_preserved'] is True,'Qualified endpoint/all native masks, not a fixed18700 teacher role')
 need(r['category_mapping_applied_once']=={'0':1} and r['optimizer_steps']==r['training_credit']==0,'Already mapped category and inference-only source')
 need(r['model_strict_load_exact'] is True and r['selected_sources_and_image_bytes_unchanged_after'] is True,'Actual exact-load/source checks')
 rows=r['images'];expected=[x['image_id'] for x in images]
 need(len(rows)==187 and [x['image_id'] for x in rows]==expected and len({x['image_id'] for x in rows})==187,'Every original native image attempted once')
 for row in rows:
  need(type(row['native_postprocess_instances']) is int and row['native_postprocess_instances']>=0
   and row['serialized_instances']==row['native_postprocess_instances'],'Producer kept its complete per-image results')

def validate_gt(coco_gt,images):
 data=coco_gt.dataset;expected={x['image_id'] for x in images}
 need(len(data['images'])==187 and set(coco_gt.getImgIds())==expected,'Actual originalGT full187 identities')
 need(len(data['annotations'])==len(coco_gt.anns)==380
  and len({a['id'] for a in data['annotations']})==380,'Actual distinct380 annotation population')
 need(len(data['categories'])==1 and set(coco_gt.getCatIds())=={1},'Actual original ship category1')
 for a in data['annotations']:need(a['image_id'] in expected and a['category_id']==1,'GT is unchanged same population/category')
 by_id={x['image_id']:x for x in images}
 for x in data['images']:
  need([x['height'],x['width']]==[by_id[x['id']]['height'],by_id[x['id']]['width']],'Original image raster dimensions')

def available_ram():
 fields={s.split(':',1)[0]:s.split(':',1)[1] for s in Path('/proc/meminfo').read_text().splitlines()}
 return int(fields['MemAvailable'].split()[0])*1024

def main():
 started=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha',required=True);a=ap.parse_args()
 sref=reference(str(Path(a.spec).resolve(strict=True)));need(sref['sha256']==a.spec_sha,'Actual pinned CPU spec')
 spec=json.loads(Path(sref['path']).read_bytes());validate_spec(spec)
 for r in metadata_refs(spec):verify(r)
 need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Own source')
 lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());admission=validate_lease(spec,lease)
 manifest=json.loads(Path(spec['image_manifest_ref']['path']).read_bytes());images=validate_population(manifest,spec)
 native=json.loads(Path(spec['native_prediction_report_ref']['path']).read_bytes());validate_native_report(native,spec,admission,images)
 out=Path(spec['output_directory']);need(not out.exists(),'No output overwrite');out.mkdir(mode=0o700)
 report=dict(schema='SAR-native187-standardCOCO-segm-report-v1',operation=OPERATION,run_id=spec['run_id'],spec_ref=sref,
  prelock=PRELOCK,status='STARTED',qualified=False,Root_external_admission_required=True,
  externally_admitted_source_conditions_checked=True,pid=os.getpid(),pgid=os.getpgid(0),
  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_refs=spec['source_refs'],library_sources=spec['library_sources'],
  native_prediction_report_ref=spec['native_prediction_report_ref'],native_prediction_qualification_ref=spec['native_prediction_qualification_ref'],
  checkpoint_ref=spec['checkpoint_ref'],endpoint_qualification_ref=spec['endpoint_qualification_ref'],
  evaluation_gt_ref=spec['evaluation_gt_ref'],predictions_ref=spec['predictions_ref'],image_manifest_ref=spec['image_manifest_ref'],
  GT_bytes_read=None,GT_JSON_parsed=None,prediction_bytes_read=None,prediction_JSON_parsed=None,
  COCO_math_started=False,COCO_math_completed=False,AP_computed=False,metrics=None,
  GPU_numerical_execution=False,model_checkpoint_read=False,training_credit=0)
 def save():
  report['elapsed_seconds']=time.monotonic()-started;tmp=out/'REPORT.next'
  with tmp.open('w',encoding='utf-8') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,out/'REPORT.json')
 def check():
  import resource
  need(sorted(os.sched_getaffinity(0))==[0,1],'CPU01 binding')
  free=available_ram();report['minimum_available_RAM_bytes']=min(free,report.get('minimum_available_RAM_bytes',free))
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;report['worker_peak_RSS_bytes']=peak
  need(free>=PRELOCK['free_RAM_floor_bytes'] and peak<=PRELOCK['RSS_limit_bytes'] and time.monotonic()-started<1800,'Finite CPU resource bounds')
 code=1;modules={};save()
 try:
  check();need(available_ram()>=PRELOCK['start_RAM_bytes'],'48GiB startup RAM')
  need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only existing Root mode')
  modules={k:importlib.import_module(v) for k,v in LIBRARY_MODULES.items()}
  for k,m in modules.items():need(reference(m.__file__)==spec['library_sources'][k],'Actual math/codec source '+k)
  for k in ('pycocotools_mask','pycocotools_extension'):
   need(spec['library_sources'][k]==native['actual_library_refs'][k],'Same actual encoder/decoder source, not version-name inference')
  report.update(actual_library_refs={k:reference(m.__file__) for k,m in modules.items()},
   numpy_version=modules['numpy'].__version__,mask_source_identity_matches_native_encoder=True,
   canonical_COCO_library_sources_verified=True);save()
  report['GT_bytes_read']=False;save();actual=reference(spec['evaluation_gt_ref']['path']);report['GT_bytes_read']=True;save()
  need(actual==spec['evaluation_gt_ref'],'Original pinned integerGT bytes')
  report['GT_JSON_parsed']=False;save();coco_gt=modules['pycocotools_coco'].COCO(spec['evaluation_gt_ref']['path']);report['GT_JSON_parsed']=True
  validate_gt(coco_gt,images);report.update(actual_GT_images=187,actual_GT_annotations=380,actual_GT_category_ids=[1]);save();check()
  report['prediction_bytes_read']=False;save();actual=reference(spec['predictions_ref']['path']);report['prediction_bytes_read']=True;save()
  need(actual==spec['predictions_ref'],'Original complete qualified prediction bytes')
  report['prediction_JSON_parsed']=False;save()
  with Path(spec['predictions_ref']['path']).open('r',encoding='utf-8') as f:predictions=json.load(f)
  report['prediction_JSON_parsed']=True;counts=validate_predictions(predictions,images)
  need(len(predictions)==native['predictions_exported'],'All native serialized predictions, no hidden truncation')
  for row in native['images']:need(counts.get(row['image_id'],0)==row['serialized_instances'],'Exact complete producer output per image')
  report.update(prediction_records=len(predictions),images_with_no_predictions=sum(counts.get(x['image_id'],0)==0 for x in images),
   images_with_more_than100_predictions=sum(counts.get(x['image_id'],0)>100 for x in images));save();check()
  report['COCO_math_started']=True;save();before=time.monotonic()
  stats=evaluate_population({'COCO':modules['pycocotools_coco'].COCO,'COCOeval':modules['pycocotools_cocoeval'].COCOeval},
   coco_gt,predictions,[x['image_id'] for x in images])
  names=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
  report.update(COCO_math_completed=True,AP_computed=True,COCO_math_seconds=time.monotonic()-before,
   metrics=dict(raw_stats_fraction_with_minus1_undefined=stats,percent={n:100*v if v>=0 else None for n,v in zip(names,stats)}));save();check()
  for r in metadata_refs(spec)+[spec['evaluation_gt_ref'],spec['predictions_ref']]:verify(r)
  need(not any(x=='torch' or x.startswith('torch.') for x in sys.modules),'No Torch or GPU model execution in CPU evaluator')
  report.update(selected_sources_GT_predictions_unchanged_after=True,status='COCO_SEGM_COMPLETE_PENDING_ROOT');code=0
 except BaseException as e:
  report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),automatic_retry=False);code=1
 finally:
  report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
  print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','AP_computed','COCO_math_completed')}),flush=True)
 return code

if __name__=='__main__':raise SystemExit(main())
