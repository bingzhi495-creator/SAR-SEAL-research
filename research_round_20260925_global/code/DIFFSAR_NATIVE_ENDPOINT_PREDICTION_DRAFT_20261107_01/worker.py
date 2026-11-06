"""Unexecuted native final-endpoint prediction candidate; top level stdlib only."""
import argparse,datetime,gc,hashlib,importlib,json,math,os,random,time,traceback,types
from pathlib import Path,PurePosixPath
def need(value,message):
 if not value:raise ValueError(message)
def validate_endpoint_header(state,admission):
 need(admission.get('qualified') is True and admission.get('selection')=='FIXED37050_NO_BEST'
  and type(admission.get('updates')) is int and admission['updates']==37050,'External Root-qualified fixed endpoint required')
 need(state.get('schema')=='DiffSAR-native-full-state-v1' and state.get('mode')=='FORMAL'
  and type(state.get('updates')) is int and state['updates']==37050,'No partial or validation checkpoint')
 need(state.get('protocol')==admission.get('training_protocol') and state['protocol']['fixed_target']==37050
  and state['protocol']['proposals']==500,'Same qualified training500 protocol')
 need(state.get('loader',{}).get('consumed_indices')==37050,'Qualified consumed-sample endpoint')
 need(state.get('scientific_pins')==admission.get('training_scientific_pins'),'Exact qualified training source/input identity')
def write_prediction_rows(stream,rows,image_id,category_id_map,first):
 count=0
 for row in rows:
  need(row['image_id']==image_id,'Native image identity')
  key=str(row['category_id']);need(key in category_id_map,'Declared category mapping exactly once')
  need(type(row['score']) in (int,float) and math.isfinite(row['score']),'Finite original score')
  need('segmentation' in row and isinstance(row['segmentation']['counts'],str),'Native UTF8 COCO RLE required')
  mapped=dict(row);mapped['category_id']=category_id_map[key]
  if not first:stream.write(',\n')
  json.dump(mapped,stream,sort_keys=True,allow_nan=False,separators=(',',':'))
  first=False;count+=1
 return count
def validate_author_inference(facts):
 need(facts.get('training_proposals')==500 and facts.get('inference_proposals')==1000
  and facts.get('sampling_steps')==1 and facts.get('use_nms') is True,'Faithful author test1000 SAMPLE_STEP1 NMS')
 need(facts.get('prediction_limit') is None and facts.get('evaluation_max_dets')==[1,10,100],
  'No model-output cap; COCO maxDets belongs only to later evaluation')

SCHEMA='Root-DiffSAR-fixed50-native-prediction-spec-v1'
LEASE_SCHEMA='Root-DiffSAR-fixed50-native-prediction-lease-v1'
OPERATION='CUDA_DIFFSAR_FIXED37050_NATIVE1000_DEV187_PREDICTION_V01'
TEST_CONFIG_PIN=dict(bytes=7066,sha256='1220408ebd3b1dbe7c15c4ebaa389481e7447608d6770df342e248983241527b')
PRELOCK=dict(endpoint_updates=37050,training_proposals=500,inference_proposals=1000,sampling_steps=1,use_nms=True,
    prediction_limit=None,evaluation_max_dets=[1,10,100],image_population=187,GT_instances_expected=380,
    population_role='INTERNAL_DEV187_ALREADY_USED_FOR_DEVELOPMENT_NOT_OFFICIAL_TEST',
    seed=40244023,seed_timing='AFTER_STRICT_MODEL_LOAD_BEFORE_MAPPER_AND_SINGLE_ORDERED187_PASS',
    process_layout='ONE_UNSHARDED_FULL187_PASS_NO_RESUME_NO_PER_IMAGE_RESEED',batch_size=1,
    CPU_affinity=[0,1],threads=2,GPU_allocator_cap_bytes=8*1024**3,
    peak_group_RSS_cap_bytes=16*1024**3,free_RAM_floor_bytes=16*1024**3,start_RAM_bytes=48*1024**3,
    deadline_seconds=1800,AMP=False,TF32=False,optimizer_steps=0,training_credit=0,automatic_retry=False,
    metric_computation=False,one_native_forward_per_image=True,whole_device_exclusive=False)

def reference(path):
    p=Path(path);q=p.resolve(strict=True);need(str(p)==str(q) and q.is_file(),'Canonical regular source')
    a=q.stat();h=hashlib.sha256();n=0
    with q.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
    z=q.stat();key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(key(a)==key(z),'File changed during byte verification')
    return dict(path=str(q),bytes=n,sha256=h.hexdigest())

def metadata_module(ref,name):
    need(reference(ref['path'])==ref,'Pinned metadata helper')
    m=types.ModuleType(name);m.__file__=ref['path']
    exec(compile(Path(m.__file__).read_bytes(),m.__file__,'exec'),m.__dict__);return m

def library_modules(helper):
    out=dict(helper.LIBRARY_MODULES)
    out.update(d2_coco_evaluation='detectron2.evaluation.coco_evaluation',d2_data_utils='detectron2.data.detection_utils',
        d2_transforms='detectron2.data.transforms',pycocotools_mask='pycocotools.mask',pycocotools_extension='pycocotools._mask')
    return out

def selected_refs(spec):
    # GT ref is advertised to the later evaluator, never read or hashed by this prediction worker.
    return list(spec['source_refs'].values())+list(spec['library_sources'].values())+[
        spec[k] for k in ('test_config_ref','checkpoint_ref','endpoint_qualification_ref','image_manifest_ref','root_lease_ref')]

def validate_spec(spec,helper):
    need(set(spec)=={'schema','operation','run_id','source_refs','library_sources','test_config_ref','checkpoint_ref',
        'endpoint_qualification_ref','image_manifest_ref','evaluation_gt_ref','root_lease_ref','output_directory','prelock'},'Exact prediction spec')
    need(spec['schema']==SCHEMA and spec['operation']==OPERATION and spec['prelock']==PRELOCK,'Fixed native endpoint-only protocol')
    need(spec['run_id'].startswith('DiffSAR-fixed37050-native1000-'),'Run identity')
    need(set(spec['source_refs'])=={'worker','config_adapter','full_train_worker'},'Only required own source/helper refs')
    need(set(spec['library_sources'])==set(library_modules(helper))|{'tv_extension'},'Actual selected native serializer/model sources')
    for r in selected_refs(spec)+[spec['evaluation_gt_ref']]:helper.valid_ref(r)
    for k,v in TEST_CONFIG_PIN.items():need(spec['test_config_ref'][k]==v,'Fixed author SSDD test YAML')
    out=PurePosixPath(spec['output_directory']);need(out.is_absolute() and str(out)==spec['output_directory'] and '..' not in out.parts and str(out)!='/','Fresh separate output root')
    need(all(not r['path'].startswith(str(out)+'/') and r['path']!=str(out) for r in selected_refs(spec)+[spec['evaluation_gt_ref']]),'Outputs outside source/inputs')

def validate_lease(spec,lease):
    need(lease.get('schema')==LEASE_SCHEMA and lease.get('status')=='ROOT_GRANTED','New future evaluation Root lease required')
    for k in ('operation','run_id','prelock','output_directory','checkpoint_ref','endpoint_qualification_ref','image_manifest_ref','evaluation_gt_ref'):
        need(lease.get(k)==spec[k],'Root lease join '+k)
    need(lease.get('worker_ref')==spec['source_refs']['worker'] and lease.get('full_train_worker_ref')==spec['source_refs']['full_train_worker'],
        'Actual inference and qualified training helper source joins')
    need(lease.get('single_attempt') is True and lease.get('prior_project_scientific_groups_reaped') is True,'Single owned future scientific invocation')
    need(lease.get('exclusive_project_scientific_leaf') is True and lease.get('whole_device_exclusive') is False,'Root project-owned scientific leaf')
    need(lease.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'Current-boot future evaluation lease')
    a=lease['endpoint_admission']
    need(a.get('checkpoint_ref')==spec['checkpoint_ref'] and a.get('qualification_ref')==spec['endpoint_qualification_ref']
        and a.get('qualified') is True and a.get('updates')==37050 and a.get('selection')=='FIXED37050_NO_BEST',
        'Root external fixed-endpoint qualification joins; declaration cannot replace actual Root audit')
    return a

def image_plan(manifest,spec,helper):
    need(manifest['schema']=='Root-DiffSAR-DEV187-original-image-input-v1' and manifest['population_role']==PRELOCK['population_role'],'InternalDEV role, never official TEST')
    need(manifest['evaluation_gt_ref']==spec['evaluation_gt_ref'] and manifest['GT_instances_expected']==380,'Advertised later-evaluation identity, not a GT count verification')
    need(manifest['category_id_map']=={'0':1},'Fixed ship contiguous0 to originalCOCO1 mapping')
    rows=manifest['images'];need(len(rows)==187 and len({x['image_id'] for x in rows})==187,'All187 logical images')
    for i,r in enumerate(rows):
        need(set(r)=={'dataset_index','image_id','height','width','image_ref'} and r['dataset_index']==i,'Canonical ordered image metadata only')
        need(type(r['image_id']) is int and all(type(r[k]) is int and r[k]>1 for k in ('height','width')),'Original image identity/shape')
        helper.valid_ref(r['image_ref'])
    return rows

def configuration(modules,adapter,spec):
    cfg=modules['d2_config'].get_cfg();modules['config'].add_DiffSARShipInst_config(cfg)
    modules['model_ema'].add_model_ema_configs(cfg);adapter.register_inactive_bifpn_schema(cfg)
    cfg.merge_from_file(spec['test_config_ref']['path']);adapter.verify_inactive_bifpn_schema(cfg)
    d=cfg.MODEL.DiffSARShipInst
    facts=dict(training_proposals=500,inference_proposals=d.NUM_PROPOSALS,sampling_steps=d.SAMPLE_STEP,
        use_nms=d.USE_NMS,prediction_limit=None,evaluation_max_dets=[1,10,100]);validate_author_inference(facts)
    need(cfg.MODEL.META_ARCHITECTURE=='DiffSARShipInst' and cfg.MODEL.BACKBONE.NAME=='build_resnet_scje_fpn_backbone'
        and cfg.MODEL.RESNETS.DEPTH==101 and cfg.MODEL.RESNETS.DEFORM_MODULATED,'Original R101/SCJE/DCN model')
    need(d.NUM_CLASSES==1 and d.NUM_HEADS==6 and d.USE_FOCAL and not d.USE_FED_LOSS and not cfg.MODEL_EMA.ENABLED,'Original class/head/EMA settings')
    need(cfg.INPUT.FORMAT=='RGB' and cfg.INPUT.MIN_SIZE_TEST==512 and cfg.INPUT.MAX_SIZE_TEST==512
        and not cfg.TEST.AUG.ENABLED and cfg.MODEL.WEIGHTS in (None,''),'Author single-scaleRGB test with explicit own endpoint load')
    cfg.freeze();return cfg

def predict(spec,helper,modules,adapter,lease,plan,report,save,check):
    t=modules['torch'];model=None;state=None;weights=None;live=None;mapper=None;output=None;cpu=None;mapped=None;rows=None
    try:
        admission=lease['endpoint_admission'];helper.verify(spec['checkpoint_ref'])
        state=t.load(spec['checkpoint_ref']['path'],map_location='cpu',weights_only=False)
        validate_endpoint_header(state,admission)
        helper.validate_checkpoint_header(state,37050,'FORMAL',admission['training_protocol'])
        pins=state['scientific_pins'];train_ref=spec['source_refs']['full_train_worker']
        need(pins['sources']['worker']=={k:train_ref[k] for k in ('bytes','sha256')},'Actual qualified training worker identity')
        need(pins['sources']['config_adapter']=={k:spec['source_refs']['config_adapter'][k] for k in ('bytes','sha256')},'Same startup configuration compatibility source')
        for k,p in pins['libraries'].items():
            need(k in spec['library_sources'] and p=={n:spec['library_sources'][k][n] for n in ('bytes','sha256')},'Same trained model/runtime source '+k)
        weights=state['model'];expected=helper.tree_digest(t,weights)
        need(expected==admission['model_component_digest'],'Qualified endpoint model typed-byte identity')
        report.update(endpoint_checkpoint_header_verified=True,qualified_endpoint_model_component_digest=expected,
            training_protocol=state['protocol'],training_scientific_pins=pins,training_optimizer_loaded=False)
        state=None;gc.collect()  # Discard optimizer/grad/RNG state; never construct an optimizer for prediction.
        cfg=configuration(modules,adapter,spec);model=modules['d2_modeling'].build_model(cfg)
        live=model.state_dict();need(set(live)==set(weights),'All endpoint model parameters and persistent buffers')
        for name in weights:
            need(live[name].shape==weights[name].shape and live[name].dtype==weights[name].dtype,'No silent model dtype/shape cast: '+name)
        incompatible=model.load_state_dict(weights,strict=True)
        need(not incompatible.missing_keys and not incompatible.unexpected_keys,'Strict complete state load')
        actual=helper.tree_digest(t,model.state_dict());need(actual==expected,'Exact loaded endpoint model parameters/buffers')
        live=None;weights=None
        need(model.num_proposals==1000 and model.sampling_timesteps==1 and model.use_nms,'Actual native inference1000/single-step/NMS')
        model.eval();need(not model.training and all(not x.training for x in model.modules()),'Inference-only model flags')
        # Evaluation RNG is intentionally reset ONCE after construction/strict loading.
        random.seed(PRELOCK['seed']);modules['numpy'].random.seed(PRELOCK['seed']);t.manual_seed(PRELOCK['seed']);t.cuda.manual_seed_all(PRELOCK['seed'])
        mapper=modules['mapper'].DiffSARShipInstDatasetMapper(cfg,is_train=False)
        report.update(model_strict_load_exact=True,model_loaded_component_digest=actual,
            inference_proposals=1000,sampling_steps=1,native_NMS=True,all_post_NMS_masks_preserved=True,
            RNG_seed_timing=PRELOCK['seed_timing'],single_ordered_population_pass=True,model_parameter_numel=sum(p.numel() for p in model.parameters()))
        out=Path(spec['output_directory']);part=out/'predictions.coco.json.part';total=0;report['images']=[];save()
        with part.open('x',encoding='utf-8') as stream:
            stream.write('[')
            with t.no_grad():
                for item in plan:
                    check();begin=time.monotonic()
                    mapped=mapper(dict(file_name=item['image_ref']['path'],image_id=item['image_id'],height=item['height'],width=item['width']))
                    need('annotations' not in mapped and 'instances' not in mapped,'No GT supplied to native inference')
                    t.cuda.synchronize(0);fwd=time.monotonic();output=model([mapped]);t.cuda.synchronize(0);elapsed=time.monotonic()-fwd
                    need(isinstance(output,list) and len(output)==1,'Exactly one native model invocation per image')
                    cpu=output[0]['instances'].to('cpu');n=len(cpu)
                    need(n<=1000 and tuple(cpu.image_size)==(item['height'],item['width']),'Native original-sized result, no extra cap')
                    need(cpu.has('pred_masks') and tuple(cpu.pred_masks.shape)==(n,item['height'],item['width']),'All original-size instance masks')
                    # Native in-place gt_ leaves floating 0/1 masks, rather than requiring bool dtype.
                    need(bool(t.isfinite(cpu.pred_masks).all().item()) and bool(((cpu.pred_masks==0)|(cpu.pred_masks==1)).all().item()),'Original postprocessed binary values')
                    need(bool(t.isfinite(cpu.scores).all().item()) and bool(t.isfinite(cpu.pred_boxes.tensor).all().item()),'Finite original native scores/boxes')
                    rows=modules['d2_coco_evaluation'].instances_to_coco_json(cpu,item['image_id'])
                    need(len(rows)==n,'Serialize every native post-NMS mask; do not pre-trim100')
                    written=write_prediction_rows(stream,rows,item['image_id'],{'0':1},first=(total==0));need(written==n,'No dropped serialized result')
                    total+=written;stream.flush()
                    report['images'].append(dict(dataset_index=item['dataset_index'],image_id=item['image_id'],
                        native_postprocess_instances=n,serialized_instances=written,mapped_image_shape=list(mapped['image'].shape),
                        mapped_dtype=str(mapped['image'].dtype),native_mask_dtype=str(cpu.pred_masks.dtype),
                        native_forward_seconds=elapsed,IO_and_mapping_forward_export_seconds=time.monotonic()-begin))
                    report.update(images_completed=len(report['images']),predictions_exported=total,stage='NATIVE_IMAGE_%03d_EXPORTED'%len(report['images']));save()
                    mapped=None;output=None;cpu=None;rows=None;check()
            stream.write(']\n');stream.flush();os.fsync(stream.fileno())
        need(len(report['images'])==187,'Complete locked DEV187 prediction population')
        final=out/'predictions.coco.json';need(not final.exists(),'No prediction overwrite');os.rename(str(part),str(final))
        report.update(stage='NATIVE_ENDPOINT_PREDICTION187_COMPLETE',all187_completed=True,prediction_ref=reference(str(final)),
            predictions_exported=total,no_prediction_top100_cut=True,category_mapping_applied_once={'0':1},
            GT_body_read=False,GT_count_independently_verified_by_prediction=False,optimizer_steps=0,training_credit=0,AP=False)
        save()
    finally:
        model=None;state=None;weights=None;live=None;mapper=None;output=None;cpu=None;mapped=None;rows=None

def main():
    start=time.monotonic()
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha',required=True);args=ap.parse_args()
    sref=reference(str(Path(args.spec).resolve(strict=True)));need(sref['sha256']==args.spec_sha,'Actual source-bound spec')
    spec=json.loads(Path(sref['path']).read_bytes())
    helper=metadata_module(spec['source_refs']['full_train_worker'],'_DiffSAR_Ended_Training_Metadata_Helper')
    validate_spec(spec,helper)
    for ref in selected_refs(spec):
        if ref!=spec['checkpoint_ref']:helper.verify(ref)  # Check the large checkpoint immediately before trusted load.
    need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Own prediction source')
    lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());validate_lease(spec,lease)
    manifest=json.loads(Path(spec['image_manifest_ref']['path']).read_bytes());plan=image_plan(manifest,spec,helper)
    image_refs={r['image_ref']['path']:r['image_ref'] for r in plan}
    for r in image_refs.values():helper.verify(r)
    out=Path(spec['output_directory']);need(not out.exists(),'Fresh future prediction output');out.mkdir(mode=0o700)
    report=dict(schema='DiffSAR-fixed37050-native1000-prediction-report-v1',operation=OPERATION,
        run_id=spec['run_id'],spec_ref=sref,prelock=PRELOCK,status='STARTED',stage='ENTERED',qualified=False,
        Root_external_admission_required=True,source_refs=spec['source_refs'],library_sources=spec['library_sources'],
        test_config_ref=spec['test_config_ref'],checkpoint_ref=spec['checkpoint_ref'],endpoint_qualification_ref=spec['endpoint_qualification_ref'],
        image_manifest_ref=spec['image_manifest_ref'],evaluation_gt_ref=spec['evaluation_gt_ref'],
        GT_body_read=False,GT_expected_instances=380,GT_count_independently_verified_by_prediction=False,
        optimizer_steps=0,training_credit=0,AP=False,metric_computation=False,pid=os.getpid(),pgid=os.getpgid(0),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),images_completed=0)
    def save():
        report['elapsed_seconds']=time.monotonic()-start;tmp=out/'REPORT.next'
        with tmp.open('w',encoding='utf-8') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,out/'REPORT.json')
    def check():
        import resource
        need(sorted(os.sched_getaffinity(0))==[0,1],'CPU01 affinity')
        free=helper.available_ram();report['minimum_available_RAM_bytes']=min(free,report.get('minimum_available_RAM_bytes',free))
        need(free>=PRELOCK['free_RAM_floor_bytes'] and time.monotonic()-start<1800,'Resource/deadline bounds')
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;report['worker_peak_RSS_bytes']=peak
        need(peak<=PRELOCK['peak_group_RSS_cap_bytes'],'RSS16GiB')
    torch=None;code=1;modules={};save()
    try:
        check();need(helper.available_ram()>=PRELOCK['start_RAM_bytes'],'48GiB startup RAM')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
            and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Locked thread/device environment')
        modules={k:importlib.import_module(v) for k,v in library_modules(helper).items()};torch=modules['torch']
        need(torch.__version__=='2.10.0+cu130' and torch.version.cuda=='13.0' and modules['torchvision'].__version__=='0.25.0+cu130','Locked existing Torch runtime')
        for k,m in modules.items():need(reference(m.__file__)==spec['library_sources'][k],'Actual selected import '+k)
        need(spec['library_sources']['tv_extension']['path'] in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries},'Actual torchvision extension')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One real CUDA device')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,PRELOCK['GPU_allocator_cap_bytes']/props.total_memory),device=0)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        adapter=metadata_module(spec['source_refs']['config_adapter'],'_DiffSAR_Prediction_Config_Adapter')
        report.update(actual_library_refs={**{k:reference(m.__file__) for k,m in modules.items()},'tv_extension':reference(spec['library_sources']['tv_extension']['path'])},
            actual_environment=dict(torch=torch.__version__,torchvision=modules['torchvision'].__version__,cuda_build=torch.version.cuda,
                device=props.name,device_total_memory_bytes=props.total_memory,threads=torch.get_num_threads(),AMP=False,TF32=False))
        predict(spec,helper,modules,adapter,lease,plan,report,save,check);check()
        for r in selected_refs(spec)+list(image_refs.values()):helper.verify(r)
        report.update(selected_sources_and_image_bytes_unchanged_after=True,status='NATIVE_ENDPOINT_PREDICTION_COMPLETE_PENDING_ROOT');code=0
    except BaseException as e:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),
            failure_scope='RESOURCE_OR_PREDICTION_FAILURE_NOT_AP_OR_METHOD_INVALIDITY',automatic_retry=False);code=1
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                gc.collect();torch.cuda.synchronize(0);clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
                need(callable(clear),'Workspace cleanup');clear();torch.cuda.empty_cache()
                a=torch.cuda.memory_allocated(0);r=torch.cuda.memory_reserved(0);p=torch.cuda.max_memory_allocated(0)
                report.update(CUDA_allocated_after_cleanup=a,CUDA_reserved_after_cleanup=r,CUDA_peak_allocated_bytes=p,cuBLAS_workspaces_cleared=True)
                need(a==r==0 and p<=PRELOCK['GPU_allocator_cap_bytes'],'FinalCUDA0/0 peak8GiB')
            except BaseException as e:report.update(cleanup_error=str(e),status='FAILED_PRESERVED_NO_RETRY');code=1
        report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','images_completed','AP','optimizer_steps')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
