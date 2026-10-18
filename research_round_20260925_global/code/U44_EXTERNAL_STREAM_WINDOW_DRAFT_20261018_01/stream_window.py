"""UNBOUND external-image window, not a loader, runner or qualification.

Only stdlib imports. Genuine injected numerical calls below have not been run.
The caller owns teacher loading, cold CPU arm construction/strict restoration,
input/source admission, the whole-leaf deadline and continuous resource limits.
"""
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import time
import traceback


ARMS=('PLAIN','FULL')
METHODS=('native',)+ARMS
MODEL_KEYS=('teacher',)+ARMS
SEED=202610173101
HEAD_NAMES=tuple('final_super_0_'+str(i) for i in range(1,5))
PINS=dict(native_capture='7fe5faf26a189216980cb527d4f9f2b44804a27a0b1c157d5eb7b4411ef37c0b',
    image_bridge='0cbc84217727756c25c68d7b0992328e5d1ae022ddd03da39901fc6c8901ad84',
    prediction_core='819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680',
    image_geometry='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    training_state='8f9b549a8383877aa6f0ae3b1d2a9786c44c0f486e45b447b06dea2e25f5a0c5',
    population_contract='733a208c5df49113e9dff3338b2b88ad2d0d4b26b2bbdd3676ce10417e05c9a5',
    mask_api='1281ce0b5436ffe924c96521b2ae1d589f6f0e2a7b7281b6bb284f7b2245545c')
PREF_SHA='7f7e5ce0eea8f0dae6e7fc01a4b54baafdc52fb3339b039ae50ef26b60870201'
MODULES=tuple(PINS)+('encoder',)
LIMITS=dict(deadline_seconds=1800,allowed_cpus=[0,1],threads=2,allocator_bytes=4*1024**3,
    whole_GPU_bytes=8*1024**3,maximum_RSS_bytes=16*1024**3,minimum_available_RAM_bytes=16*1024**3)


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
        and type(value['path']) is str and PureWindowsPath(value['path']).is_absolute()
        and type(value['bytes']) is int and value['bytes']>=0
        and type(value['sha256']) is str and len(value['sha256'])==64
        and set(value['sha256'])<=set('0123456789abcdef'),'Exact original strict ref')
    return value


def prepare_window_metadata(image_rows):
    """Project 1..4 consecutive image-only records, never inspect annotation data."""
    need(type(image_rows) is list and 1<=len(image_rows)<=4,'Prelocked window of one to four images')
    images=[];ids=set();paths=set()
    for offset,row in enumerate(image_rows):
        need(type(row) is dict and set(row)=={'dataset_index','image_id','image_ref','original_shape'},
             'Image identity plus outer ordinal only; no GT/parser/extra fields')
        index=row['dataset_index'];image_id=row['image_id']
        need(type(index) is int and index>=0 and type(image_id) is int and image_id>=0,'Original ordinal and image ID')
        need(index==image_rows[0]['dataset_index']+offset,'One contiguous original-order window')
        reference=strict_ref(row['image_ref']);path=str(PureWindowsPath(reference['path'])).casefold()
        need(image_id not in ids and path not in paths,'No duplicate image ID/path within this window')
        ids.add(image_id);paths.add(path)
        shape=row['original_shape']
        need(type(shape) in (list,tuple) and len(shape)==2 and all(type(v) is int and v>0 for v in shape),'Original H/W')
        images.append(dict(dataset_index=index,image_metadata=deepcopy({key:row[key] for key in ('image_id','image_ref','original_shape')})))
    return dict(dataset_range=[image_rows[0]['dataset_index'],image_rows[-1]['dataset_index']+1],images=images)


def validate_restoration_metadata(receipts,actual_state_digests):
    """Join recorded E50 provenance with independently computed current model digests."""
    need(type(receipts) is dict and set(receipts)==set(actual_state_digests)==set(ARMS),'Both restored CPU arms')
    shared=('endpoint_refs','checkpoint_refs','training_qualified','pair_commit_ref','order_ref',
            'training_sources','initialization_receipt_sha256')
    for arm in ARMS:
        row=receipts[arm]
        need(row['schema']=='U44-external-fixed50-CPU-arm-restore-pending-v1' and row['status']=='RESTORED_CPU_ONLY'
             and row['qualified'] is False and row['arm']==arm and row['seed']==SEED
             and row['fixed_epoch']==50 and row['updates_per_arm']==37050,'Recorded strict E50 restoration; never qualification')
        need(row['model_state_sha256']==actual_state_digests[arm],'Actual current CPU model differs from restored state')
        need(all(row[k] is True for k in ('all_modules_eval','all_parameters_frozen','all_parameter_grads_none',
             'CPU_checkpoint_payload_released')) and all(row[k] is False for k in ('optimizer_created','optimizer_restored','CUDA_initialized')),
             'Original CPU-only restore scope')
        need(set(row['endpoint_refs'])=={'training_qualified','pair_commit','preparation','order'}
             and set(row['checkpoint_refs'])==set(ARMS),'Original paired endpoint refs')
        for reference in list(row['endpoint_refs'].values())+list(row['checkpoint_refs'].values()):strict_ref(reference)
        need(row['checkpoint_ref']==row['checkpoint_refs'][arm] and row['training_qualified']==row['endpoint_refs']['training_qualified']
             and row['pair_commit_ref']==row['endpoint_refs']['pair_commit'] and row['order_ref']==row['endpoint_refs']['order'],'Endpoint joins')
        need(all(row[k]==receipts['PLAIN'][k] for k in shared),'Same paired E50 provenance and initialization lineage')
    return dict(seed=SEED,fixed_epoch=50,updates_per_arm=37050,
                **{k:deepcopy(receipts['PLAIN'][k]) for k in shared})


def _module_source(module,reference,c,expected=None):
    strict_ref(reference);path=Path(c.closed(reference))
    need(path.resolve()==Path(reference['path']).resolve()==Path(module.__file__).resolve(),'Actual injected module path')
    raw=path.read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Actual admitted module bytes')
    if expected is not None:need(reference['sha256']==expected,'Original numerical source pin')


def _cpu_model(torch,model):
    need(all(not value.training for value in model.modules()),'Caller supplies fully eval models')
    need(all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad and p.grad is None
             for p in model.parameters()),'Caller supplies frozen FP32 CPU parameters with no gradients')
    need(all(v.device.type=='cpu' and (not v.is_floating_point() or
             (v.dtype==torch.float32 and bool(torch.isfinite(v).all()))) for v in model.state_dict().values()),
             'Entire finite CPU state, including inactive parameters and BN buffers')


def _proof(name,model,modules,native_modules):
    if name=='teacher':return native_modules['pref'].state_proof(model.state_dict())
    return modules['prediction_core']._state_digest(modules['u36_helpers'],model.state_dict())


def _error(error):
    result=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    traceback.clear_frames(error.__traceback__);error.__traceback__=None
    return result


def _resource_check(record):
    need(record['CUDA_peak_allocated_bytes']<=LIMITS['allocator_bytes']
         and record['sampled_RSS_peak_bytes']<=LIMITS['maximum_RSS_bytes']
         and not record['RSS_sampling_errors'],'Measured window stage exceeded original resource bounds')


def _run_stage(torch,np,Image,c,name,models,window,carriers,outputs,modules,source_refs,native_modules,
               native_source_refs,preprocessing,teacher_provenance,expected_proof,record):
    """One real resident model; no retained CUDA tensor returned from this scope."""
    active=capture=bridged=item=module=None;handles=[]
    calls=dict(head=0) if name=='teacher' else dict(backbone=0,feature_exit=0,**{k:0 for k in HEAD_NAMES})
    record.update(status='STARTED',calls=calls,image_proofs=[])
    cost=dict(encode_calls=0,readout_calls=0,queries=0,encode_seconds=0.,readout_seconds=0.,
        render_seconds=0.,RLE_seconds=0.,zero_calibration_seconds=0.,changed_inside_Omega=0,changed_outside_Omega=0)
    if name!='teacher':record['cost']=cost
    def hook(key):
        def count(module,args,output):calls[key]+=1
        return count
    try:
        c.check();active=models[name]
        tick=time.perf_counter();active.to(device='cuda:0',dtype=torch.float32);torch.cuda.synchronize()
        record['CPU_to_GPU_seconds']=time.perf_counter()-tick
        need(_proof(name,active,modules,native_modules)==expected_proof,'Original state changed during transfer')
        if name=='teacher':handles.append(active.panoptic_head.register_forward_hook(hook('head')))
        else:
            handles.append(active.backbone.register_forward_hook(hook('backbone')))
            handles.append(active.backbone.legacy.CONV0_4.register_forward_hook(hook('feature_exit')))
            for key in HEAD_NAMES:
                module=getattr(active.backbone.legacy,key);handles.append(module.register_forward_hook(hook(key)))
            module=None
        with torch.no_grad():
            for position,row in enumerate(window['images']):
                c.check();tick=time.perf_counter()
                if name=='teacher':
                    capture=modules['native_capture'].capture_native_image(torch,np,active,row['image_metadata'],
                        modules=native_modules,source_refs=native_source_refs,preprocessing=preprocessing,
                        provenance=teacher_provenance,closed_ref=c.closed,check=c.check)
                    bridged=modules['image_bridge'].make_image_item(torch,np,Image,capture,row['dataset_index'],
                        row['image_metadata']['image_ref'],image_geometry=modules['image_geometry'],
                        source_refs={k:source_refs[k] for k in ('image_geometry','PIL.Image')},closed_ref=c.closed,check=c.check)
                    carriers.append(bridged);outputs['native'].extend(bridged['originals'])
                    record['image_proofs'].append(dict(dataset_index=row['dataset_index'],
                        capture=deepcopy(capture['receipt']),bridge=deepcopy(bridged['item']['rgb_receipt'])))
                    capture=bridged=None
                else:
                    item=carriers[position]
                    predictions,identity,support,proof=modules['prediction_core']._calibration_and_prediction(
                        torch,np,c,active,item['item'],item['row'],item['originals'],modules,cost,False)
                    need(identity==item['originals'] and support==[],'Zero calibration preserves complete native dictionaries')
                    outputs[name].extend(predictions);record['image_proofs'].append(proof)
                    del predictions,identity,support,proof
                    item=None
                record.setdefault('image_elapsed_seconds',[]).append(time.perf_counter()-tick)
                gc.collect();c.check()
        n=len(window['images'])
        if name=='teacher':need(calls['head']==n,'Exactly one actual teacher head forward per image')
        else:need(calls['backbone']==calls['feature_exit']==cost['encode_calls']==n
            and all(calls[k]==0 for k in HEAD_NAMES) and cost['readout_calls']==25*n and cost['queries']==100*n,
            'Exactly one backbone/CONV0_4 per image, zero old semantic heads, all100 query4')
        need(_proof(name,active,modules,native_modules)==expected_proof,'Inference changed original model/BN state')
        record['status']='COMPUTATION_COMPLETE_PENDING_CPU_RETURN'
    except BaseException as error:
        record['status']='FAILED';record['error']=_error(error)
    finally:
        for handle in handles:handle.remove()
        handles.clear();capture=bridged=item=module=None
        try:
            # Same caller-owned model object survives, with all state now on CPU.
            tick=time.perf_counter();models[name].to(device='cpu');active=None;gc.collect()
            record['GPU_to_CPU_seconds']=time.perf_counter()-tick
            _cpu_model(torch,models[name])
            need(_proof(name,models[name],modules,native_modules)==expected_proof,'Returned CPU state differs from original proof')
            record['returned_CPU_state_exact']=True
            record['cleanup']=modules['training_state']._cleanup(torch)
        except BaseException as error:
            active=None;record['status']='FAILED';record['cleanup_error']=_error(error)
    need(record['status']!='FAILED','Window stage failed; retain report, no retry or next model')
    record['status']='COMPUTATION_COMPLETE_CPU_RETURNED'


def run_window(torch,np,Image,c,models,image_rows,*,modules,source_refs,native_modules,native_source_refs,
               preprocessing,teacher_provenance,restoration_receipt,report):
    """Use genuine preloaded CPU models; return only final RLEs and scalar records.

    models is {'teacher','PLAIN','FULL'} and survives on CPU. No loading, building,
    optimizer, GT input, qualification, population merge, or output-file writing.
    Caller supplies closed/check/psutil and owns whole-leaf hard time/GPU/RAM guards.
    """
    need(type(report) is dict and not report,'Caller retains a fresh partial report')
    started=time.perf_counter();carriers=[];outputs={k:[] for k in METHODS};proofs={};rng=None;sources_verified=False
    report.update(schema='U44-external-stream-window-pending-v1',status='STARTED',stages={},
        new_training=False,AP=False,qualified=False,numerical_runtime_qualified=False,population_qualification=False,
        GT_used_for_prediction=False,teacher_loader_implemented=False,automatic_retry=False)
    try:
        window=prepare_window_metadata(image_rows)
        need(type(models) is dict and set(models)==set(MODEL_KEYS) and len({id(v) for v in models.values()})==3,'Three distinct caller-owned models')
        need(set(modules)==set(MODULES) and set(source_refs)==set(MODULES)|{'PIL.Image'},'Exact explicit numerical dependencies')
        c.check()
        for role,module in modules.items():_module_source(module,source_refs[role],c,PINS.get(role))
        _module_source(Image,source_refs['PIL.Image'],c)
        _module_source(native_modules['pref'],native_source_refs['pref'],c,PREF_SHA)
        sources_verified=True
        need(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and torch.get_default_dtype()==torch.float32
             and not torch.is_autocast_enabled(),'Original FP32 runtime; no AMP')
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2 and c.psutil.Process().cpu_affinity()==[0,1]
             and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Original CPU01/two-thread/CUDA0 scope')
        need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
             and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False,'Original deterministic/TF32 flags')
        identities=set()
        for name in MODEL_KEYS:
            _cpu_model(torch,models[name])
            ids={id(v) for v in models[name].parameters()}|{id(v) for v in models[name].buffers()}
            need(not identities.intersection(ids),'Models must not share parameter/buffer objects');identities.update(ids)
            proofs[name]=_proof(name,models[name],modules,native_modules)
        need(proofs['teacher']==teacher_provenance['teacher_state_proof'],'Actual CPU teacher must match original loading proof; never replace it')
        endpoint=validate_restoration_metadata(restoration_receipt,{a:proofs[a] for a in ARMS})
        need(set(endpoint['training_sources'])==set(modules['prediction_core'].TRAINING_ROLES),'Original seventeen training source records')
        for arm in ARMS:
            receipt=restoration_receipt[arm];model=models[arm]
            need(sorted(dict(model.named_modules()))==receipt['module_names'],'Actual restored module tree')
            effective=receipt['effective_parameter_names'];inactive=receipt['inactive_parameter_names']
            need(not set(effective)&set(inactive) and sorted(effective+inactive)==sorted(dict(model.named_parameters())),
                 'Actual complete effective/inactive parameter names')
        model=None
        report.update(endpoint,dataset_range=window['dataset_range'],images=len(image_rows),queries=100*len(image_rows),
            method_names=list(METHODS),query_batch=4,limits=deepcopy(LIMITS),source_refs=deepcopy(source_refs),
            native_source_refs=deepcopy(native_source_refs),teacher_checkpoint=deepcopy(teacher_provenance['teacher_checkpoint']))
        # The caller has already constructed/restored both arms while CUDA was cold.
        # Subsequent windows are permitted with initialized CUDA and zero allocations.
        if not torch.cuda.is_initialized():torch.cuda.init()
        torch.cuda.set_device(0)
        need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'No caller CUDA aliases may precede this window')
        total=int(torch.cuda.get_device_properties(0).total_memory)
        need(total>=LIMITS['allocator_bytes'],'Physical allocator capacity')
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/float(total),0)
        rng=native_modules['pref'].rng_snapshot(torch)
        for name in MODEL_KEYS:
            stage=report['stages'].setdefault(name,{'resources':{}})
            # Include transfer back to CPU, state checks and cleanup in RSS/time/peak sampling.
            with modules['u36_helpers']._measure(torch,c,stage['resources'],cuda=True):
                _run_stage(torch,np,Image,c,name,models,window,carriers,outputs,modules,source_refs,native_modules,
                    native_source_refs,preprocessing,teacher_provenance,proofs[name],stage)
            _resource_check(stage['resources']);stage['status']='COMPLETED'
            if name=='teacher':
                report['retained_CPU_tensor_bytes']=sum(v.numel()*v.element_size() for item in carriers
                    for v in (item['item']['e'],item['item']['M'],item['item']['L'],item['item']['rgb400']))
        need(all(len(outputs[k])==report['queries'] for k in METHODS),'All three methods retain every original query')
        after=native_modules['pref'].rng_snapshot(torch)
        need(len(rng)==len(after) and all(torch.equal(a,b) for a,b in zip(rng,after)),'Window preserves original CPU/CUDA RNG')
        del after
        report.update(status='COMPLETE_PENDING_RELEASE',models_returned_CPU_unchanged=True,
            native_S0_categories_query_order_preserved=True,explicit_zero_renderer_calibrated=True,
            teacher_capture_cost_included=True,checkpoint_loading_cost_included=False,standalone_FPS_claimed=False)
    except BaseException as error:
        report['status']='FAILED';report['error']=_error(error)
    finally:
        carriers.clear();rng=None;gc.collect()
        report['window_CPU_dense_references_released']=True
        # Do not initialize CUDA in failure cleanup or destroy caller CPU models.
        if torch.cuda.is_initialized() and sources_verified:
            try:report['cleanup']=modules['training_state']._cleanup(torch)
            except BaseException as error:report['status']='FAILED';report['cleanup_error']=_error(error)
        report['elapsed_seconds']=time.perf_counter()-started
        if report['elapsed_seconds']>LIMITS['deadline_seconds']:report['status']='FAILED';report['window_deadline_exceeded']=True
    if report['status']=='COMPLETE_PENDING_RELEASE':report['status']='COMPLETED'
    need(report['status']=='COMPLETED','External window failed; caller preserves partial report; no automatic retry')
    return dict(dataset_range=deepcopy(report['dataset_range']),predictions=outputs,
                images=report['images'],queries=report['queries'],report=report)
