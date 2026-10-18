"""UNBOUND single external TRAIN image native capture. No numerical execution.

Only standard-library imports. A future original owned caller supplies the
admitted genuine modules, runtime, one image, loading-proof file and resources.
"""
from copy import deepcopy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

PINS=dict(restore='019e2741d739b38ec33f015a1511cf9370d7828b040cfd1b4b878fd9dcd1afaa',
    capture='7fe5faf26a189216980cb527d4f9f2b44804a27a0b1c157d5eb7b4411ef37c0b')
CAPTURE_ROLES={'Compose','PackDetInputs','pseudo_collate','observer','renderer','exporter','pref','masks'}
CAPTURE_REFS={'observer','renderer','exporter','pref','masks','Compose','pseudo_collate','formatting',
    'data_preprocessor','base_preprocessor'}


def need(ok,message):
    if not ok:raise ValueError(message)


def resolve_channel_flags(config,constructor_defaults):
    """Resolve only explicit config or real closed constructor defaults; no fallback."""
    values={};origins={}
    for key in ('bgr_to_rgb','rgb_to_bgr'):
        if key in config:value=config[key];origin='original_config'
        else:
            need(key in constructor_defaults,'Missing original config and closed constructor default: '+key)
            value=constructor_defaults[key];origin='closed_constructor_default'
        need(type(value) is bool,'Original channel flag must be boolean: '+key)
        values[key]=value;origins[key]=origin
    need(not (values['bgr_to_rgb'] and values['rgb_to_bgr']),'Original channel conversions are mutually exclusive')
    return dict(values=values,origins=origins)


def verify_loading_proof(actual_loading,fresh_restore_receipt):
    """An existing file/ref is evidence only when it equals this actual restoration."""
    need(actual_loading==fresh_restore_receipt['original_loading_proof']
         and actual_loading['teacher_checkpoint']==fresh_restore_receipt['teacher_checkpoint']
         and actual_loading['teacher_state']==fresh_restore_receipt['teacher_state_proof'],
         'Existing loading JSON must exactly match fresh original loading proof, checkpoint and state')
    return actual_loading


def _source(c,obj,reference,expected=None):
    need(type(reference) is dict and set(reference)=={'path','bytes','sha256'} and Path(reference['path']).is_absolute(),
         'Explicit original source reference')
    path=Path(c.closed(reference)).resolve(strict=True)
    actual=inspect.getsourcefile(obj) if inspect.isfunction(obj) or inspect.isclass(obj) else obj.__file__
    need(Path(actual).resolve(strict=True)==path,'Actually injected source differs')
    raw=path.read_bytes();need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Source bytes differ')
    if expected is not None:need(reference['sha256']==expected,'Original helper pin differs')


def resolve_provenance(torch,c,model,context,fresh_restore_receipt,loading_receipt_ref,*,Config,
                       capture_module,capture_modules,capture_source_refs,capture_source_ref):
    """Resolve genuine closed config plus actual loaded preprocessor, without defaults guessed here."""
    c.check();_source(c,capture_module,capture_source_ref,PINS['capture'])
    need(fresh_restore_receipt['schema']=='U47-external-teacher-CPU-restore-pending-v1'
         and fresh_restore_receipt['status']=='RESTORED_CPU_ONLY' and fresh_restore_receipt['qualified'] is False,
         'This actual original CPU teacher restoration receipt, not a qualification shortcut')
    loading=json.loads(Path(c.closed(loading_receipt_ref)).read_text(encoding='utf-8-sig'))
    verify_loading_proof(loading,fresh_restore_receipt)
    need(loading['teacher_checkpoint']==context['checkpointmap']['native0'],'Same original native0 teacher checkpoint')
    initializer=loading['original_initialization'];config_ref=initializer['official_config']
    config_sources=initializer['official_config_closure']
    need(type(config_sources) is list and config_sources and config_ref in config_sources,'Original complete config closure and root config')
    for value in config_sources:c.closed(value)
    _source(c,Config,fresh_restore_receipt['source_refs']['registration']['Config'])
    capture_module._verify_dependencies(model,capture_modules,capture_source_refs,c.closed)
    config=Config.fromfile(str(c.closed(config_ref)))
    raw=dict(config.model.data_preprocessor)
    need(raw['type']=='DetDataPreprocessor' and 'mean' in raw and 'std' in raw,'Explicit original config photometry/type required')
    pre=model.data_preprocessor
    bases=[kind for kind in type(pre).__mro__ if kind.__name__=='ImgDataPreprocessor']
    need(len(bases)==1,'Unique genuine image preprocessor base')
    _source(c,bases[0],capture_source_refs['base_preprocessor'],capture_module.PINS['base_preprocessor'])
    signature=inspect.signature(bases[0].__init__)
    defaults={name:signature.parameters[name].default for name in ('bgr_to_rgb','rgb_to_bgr')
              if name in signature.parameters and signature.parameters[name].default is not inspect.Parameter.empty}
    channels=resolve_channel_flags(raw,defaults)
    need(pre.pad_seg is False and len(pre.batch_augments)==1,'Only original non-semantic fixed padding')
    aug=pre.batch_augments[0]
    augmentation=dict(type=type(aug).__name__,size=list(aug.size),img_pad_value=aug.img_pad_value,
        pad_mask=aug.pad_mask,mask_pad_value=aug.mask_pad_value,pad_seg=aug.pad_seg)
    resolved=dict(type=raw['type'],mean=deepcopy(raw['mean']),std=deepcopy(raw['std']),**channels['values'],
        pad_size_divisor=pre.pad_size_divisor,pad_value=pre.pad_value,pad_mask=pre.pad_mask,
        pad_seg=pre.pad_seg,batch_augments=[augmentation])
    # This checks original mean/std buffers, the loaded channel flag, divisor,
    # fixed800 augment and explicit rgb_to_bgr=False against the actual model.
    capture_module._verify_loaded_preprocessing(torch,model,resolved)
    need(initializer['declared_preprocessing_overrides']==dict(resize=[800,800],keep_ratio=True,
         fixed_canvas=[800,800],augmentation=False,mask_resize='nearest'),'Original qualified preprocessing override lineage')
    proof=capture_modules['pref'].state_proof(model.state_dict())
    need(proof==fresh_restore_receipt['teacher_state_proof']==loading['teacher_state'],'Actual teacher state equals fresh and existing loading proofs')
    provenance=dict(teacher_checkpoint=deepcopy(loading['teacher_checkpoint']),loading_receipt=deepcopy(loading_receipt_ref),
        config_sources=deepcopy(config_sources),teacher_state_proof=proof,resolved_preprocessing=resolved)
    receipt=dict(schema='U48-closed-native-preprocessing-provenance-pending-v1',qualified=False,
        loading_proof_file_exact_fresh_restore=True,teacher_state_exact=True,config_sources=deepcopy(config_sources),
        original_config=deepcopy(config_ref),resolved_preprocessing=resolved,channel_flag_origins=channels['origins'],
        padding_origin='actual loaded preprocessor plus original pref.prepare_cpu fixed800 overrides',
        mean_std_origin='original closed official config; exact actual buffer check',
        base_preprocessor_source=deepcopy(capture_source_refs['base_preprocessor']),
        model_forward_executed=False,GT_used=False,new_training=False,AP=False)
    json.dumps(receipt,allow_nan=False);c.check()
    return dict(preprocessing=resolved,provenance=provenance,receipt=receipt)


def capture_one(torch,np,c,context,image_metadata,*,selection,allocator_limit_bytes,
                restore_module,restore_modules,registration,restore_source_refs,
                capture_module,capture_modules,capture_source_refs,helper_source_refs,
                loading_receipt_ref,report):
    """One preselected HRSID TRAIN image; return original natural-axis CPU packet.

    A future owned caller supplies all refs and continuous CPU/freeRAM/wholeGPU/
    deadline checks. This is not a runner, population qualification or AP path.
    Caller must discard partial outputs on failure. No automatic retry occurs.
    """
    need(type(report) is dict and not report,'Fresh caller-owned partial capture report')
    report.update(schema='U48-single-external-TRAIN-native-capture-pending-v1',status='STARTED',qualified=False,
        new_training=False,AP=False,DEV=False,TEST=False,selection=deepcopy(selection),images=1,queries=100,
        optimizer_created=False,loss_or_matching_called=False,new_image_annotations_opened=False,
        historical_lineage_source_hashing_may_include_data_bytes=True,numerical_runtime_qualified=False,
        population_qualified=False,teacher_load_receipt={},cleanup_errors=[],allocator_limit_bytes_requested=allocator_limit_bytes)
    teacher=restored=capture=packet=resolution=cuda_rng_before=None;failure=None;started=time.perf_counter()
    try:
        need(selection==dict(dataset='HRSID',split='train',dataset_index=0),'Exactly caller-preselected HRSID TRAIN ordinal0; no GT-based selection')
        need(type(image_metadata) is dict and set(image_metadata)=={'image_id','image_ref','original_shape'}
             and list(image_metadata['original_shape'])==[800,800],'Explicit image-only original HRSID800 metadata')
        need(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),
             'Original numerical versions and cold CPU teacher restoration first')
        need(type(allocator_limit_bytes) is int and 0<allocator_limit_bytes<=4*1024**3,'Explicit per-leaf allocator cap at most original4GiB')
        need(set(helper_source_refs)=={'restore','capture'} and set(capture_modules)==CAPTURE_ROLES
             and set(capture_source_refs)==CAPTURE_REFS,'Explicit original two helpers/eight capture objects/ten capture refs')
        for role,obj in (('restore',restore_module),('capture',capture_module)):
            _source(c,obj,helper_source_refs[role],PINS[role])
        need(capture_source_refs['pref']==restore_source_refs['modules']['pref'],'Same original numerical state-proof helper')
        for value in capture_source_refs.values():c.closed(value)
        c.closed(image_metadata['image_ref']);c.closed(loading_receipt_ref);c.check()
        restored=restore_module.restore_teacher_cpu(torch,c,context,modules=restore_modules,registration=registration,
            source_refs=restore_source_refs,report=report['teacher_load_receipt'])
        teacher=restored['model'];restored=None
        resolution=resolve_provenance(torch,c,teacher,context,report['teacher_load_receipt'],loading_receipt_ref,
            Config=registration['Config'],capture_module=capture_module,capture_modules=capture_modules,
            capture_source_refs=capture_source_refs,capture_source_ref=helper_source_refs['capture'])
        report['preprocessing_provenance']=resolution['receipt'];c.check()
        # First CUDA use follows complete cold-CPU restoration and config checks.
        # No model tensor is transferred before this explicit caller-chosen cap.
        total=int(torch.cuda.get_device_properties(0).total_memory)
        need(allocator_limit_bytes<=total,'Requested cap exceeds actual device memory')
        torch.cuda.set_per_process_memory_fraction(allocator_limit_bytes/total,device=0)
        torch.cuda.reset_peak_memory_stats(0)
        teacher.to(device='cuda:0',dtype=torch.float32);torch.cuda.synchronize();c.check()
        # The cold CPU restoration has completed. This snapshot intentionally
        # starts after CUDA initialization/cap/model transfer, not before it.
        cuda_rng_before=torch.cuda.get_rng_state(0).clone()
        capture=capture_module.capture_native_image(torch,np,teacher,image_metadata,modules=capture_modules,
            source_refs=capture_source_refs,preprocessing=resolution['preprocessing'],provenance=resolution['provenance'],
            closed_ref=c.closed,check=c.check)
        need(capture['natural_query_ids']==list(range(100)) and sorted(capture['export_query_ids'])==list(range(100))
             and len(capture['native_predictions'])==len(capture['S0_by_query'])==100,'Complete natural100 plus original export order')
        shapes=dict(F=(1,256,200,200),e=(1,100,256),M=(1,100,200,200),C=(1,100,2),L=(100,800,800))
        packet={key:deepcopy(value) for key,value in capture.items() if key not in shapes}
        for name,shape in shapes.items():
            need(isinstance(capture[name],torch.Tensor) and tuple(capture[name].shape)==shape
                 and capture[name].dtype==torch.float32 and bool(torch.isfinite(capture[name]).all()),'Actual original complete native field: '+name)
            packet[name]=capture[name].detach().cpu().clone()
            need(packet[name].device.type=='cpu' and not packet[name].requires_grad,'Detached CPU native field')
        report.update(native_capture_receipt=deepcopy(capture['receipt']),packet_field_shapes={k:list(v) for k,v in shapes.items()},
            natural_query_axis_unchanged=True,original_export_order_and_S0_preserved=True,CPU_packet_detached=True,
            original_teacher_forward_calls=capture['receipt']['teacher_forward_calls'],helper_source_refs=deepcopy(helper_source_refs),
            image_metadata=deepcopy(image_metadata),loading_receipt=deepcopy(loading_receipt_ref))
        capture=None;c.check()
    except BaseException as error:
        failure=error;packet=None
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
    finally:
        restored=capture=resolution=None
        if teacher is not None:
            try:teacher.cpu()
            except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        teacher=None;gc.collect()
        if torch.cuda.is_initialized():
            report['CUDA_peak_allocated_bytes']=int(torch.cuda.max_memory_allocated(0))
            for cleanup in (lambda:torch.cuda.synchronize(),lambda:torch._C._cuda_clearCublasWorkspaces(),
                            lambda:torch.cuda.empty_cache(),lambda:torch.cuda.synchronize()):
                try:cleanup()
                except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
            report['CUDA_allocated_after_release']=int(torch.cuda.memory_allocated(0))
            report['CUDA_reserved_after_release']=int(torch.cuda.memory_reserved(0))
            report['GPU_postinitialization_RNG_preserved']=cuda_rng_before is not None and bool(
                torch.equal(cuda_rng_before,torch.cuda.get_rng_state(0)))
            report['GPU_RNG_scope']='post initialization/cap/model transfer, before native capture through release'
            report['CUDA_restored_to_uninitialized']=False
        cuda_rng_before=None
        report['elapsed_seconds']=time.perf_counter()-started
        report['Root_actual_resource_qualification_claimed']=False
    try:
        need(not report['cleanup_errors'] and report.get('CUDA_allocated_after_release')==report.get('CUDA_reserved_after_release')==0
             and report['CUDA_peak_allocated_bytes']<=allocator_limit_bytes
             and report['GPU_postinitialization_RNG_preserved'] is True,'Actual pending worker cleanup/cap/post-init RNG checks')
        c.check();report['status']='CAPTURED_CPU_PENDING_ROOT_ACTUAL_AND_INTAKE';json.dumps(report,allow_nan=False)
        return dict(packet=packet,report=report)
    except BaseException as error:
        packet=None;report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
