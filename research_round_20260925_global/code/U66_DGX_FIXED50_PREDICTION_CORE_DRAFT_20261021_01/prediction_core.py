"""UNBOUND DGX E50 prediction wiring; injected libraries/modules, no ML imports.

No owner, launcher, output writer, metric computation or Root qualification.
The unchanged U40 per-image function owns actual encode/calibration/prediction.
"""
from copy import deepcopy
import gc
import json
from pathlib import PurePosixPath
import time
import traceback
import weakref

ARMS=('PLAIN','FULL')
RANGES=((0,47),(47,94),(94,141),(141,187))
CONSTRUCTOR_ROLES=('full_legacy','full_dsconv','full_head','full_feature','plain_legacy','plain_feature','instance_adapter')
RESTORE_ROLES=('training_state','training_entry','u36_helpers','u39_training_ops','shared_initialization')+CONSTRUCTOR_ROLES
MODULE_ROLES=RESTORE_ROLES+('restore_state','portable_reader','image_geometry','prediction_core','encoder','mask_api','population_contract')
SOURCE_ROLES=MODULE_ROLES+('dev_reader','dev_rgb_reader')
READER_ROLES=('dev_reader','dev_rgb_reader','image_geometry')
PER_IMAGE_ROLES=('encoder','mask_api','population_contract')
FIELDS=('rgb400','e','M','L')


def need(ok,message):
    if not ok:raise ValueError(message)


def _json(value):
    json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    return value


def _sources(c,modules,source_refs):
    need(type(modules) is dict and set(modules)==set(MODULE_ROLES)
        and type(source_refs) is dict and set(source_refs)==set(SOURCE_ROLES),
        'Exact19 injected modules/21 source roles; historical DEV readers are closed but not executed')
    for role in SOURCE_ROLES:
        ref=source_refs[role]
        need(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str
            and '\\' not in ref['path'] and PurePosixPath(ref['path']).anchor=='/'
            and str(PurePosixPath(ref['path']))==ref['path'] and '..' not in PurePosixPath(ref['path']).parts
            and type(ref['bytes']) is int and ref['bytes']>0 and type(ref['sha256']) is str
            and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),
            'Canonical explicit admitted DGX source ref')
        c.check();need(str(c.closed(ref))==ref['path'],'Actual admitted source bytes: '+role)
        if role in modules:
            need(getattr(modules[role],'__file__',None)==ref['path'],'Module loaded from that admitted source: '+role)


def _cost():
    return dict(encode_calls=0,readout_calls=0,queries=0,encode_seconds=0.,readout_seconds=0.,
        render_seconds=0.,RLE_seconds=0.,zero_calibration_seconds=0.,cache_RGB_IO_seconds=0.,
        restore_seconds=0.,changed_inside_Omega=0,changed_outside_Omega=0)


def _image(index,item,row,originals,entry):
    need(type(item) is dict and set(item)==set(FIELDS)|{'identity','query_ids','geometry'}
        and type(row) is dict and set(row)=={'dataset_index','image_id','qids'}
        and type(row['dataset_index']) is int and row['dataset_index']==index==entry['dataset_index']
        and row['image_id']==entry['image_id']==item['identity']['image_id'],
        'Original U59 full100 three-value ABI; no F/target associations')
    qids=item['query_ids']
    need(type(qids) is list and len(qids)==100 and all(type(q) is int for q in qids)
        and set(qids)==set(range(100)) and qids==row['qids']==entry['query_ids']
        and type(originals) is list and len(originals)==100
        and [(r['image_id'],r['query_id']) for r in originals]==[(row['image_id'],q) for q in qids],
        'Unchanged natural e/M/L axes and original export order passed directly to U40')


def _cuda_release(torch):
    torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();torch.cuda.synchronize()
    value=dict(allocated_after=int(torch.cuda.memory_allocated()),reserved_after=int(torch.cuda.memory_reserved()))
    need(value==dict(allocated_after=0,reserved_after=0),'Original hard-zero CUDA allocated/reserved release')
    return value


def _arm(torch,np,c,adapters,reader,entries,modules,span,arm,record):
    core,ops,helpers=(modules[k] for k in ('prediction_core','u39_training_ops','u36_helpers'))
    adapter=item=row=originals=loaded=module=values=identity=support=proof=None
    primary=cleanup_error=None
    handles=[];calls=dict(backbone=0,feature_exit=0,**{name:0 for name in ops.HEAD_NAMES})
    record.update(status='STARTED',cost=_cost(),resources={},predictions=[],identity_predictions=[],Omega_support=[],proofs=[])
    record['cost']['restore_seconds']=record['restore_report']['elapsed_seconds']
    def hook(name):
        def observe(module,args,output):calls[name]+=1
        return observe
    try:
        with helpers._measure(torch,c,record['resources'],cuda=True):
            adapter=adapters.pop(arm)
            record['parameters']=deepcopy(_json(record['restore_report']['parameter_report']))
            record['parameter_report_origin']='U58 verified original training effective/inactive partition before inference freeze'
            record['model_state_before_sha256']=core._state_digest(helpers,adapter.state_dict())
            adapter.eval().requires_grad_(False)
            adapter.to(device='cuda:0',dtype=torch.float32)
            need(core._state_digest(helpers,adapter.state_dict())==record['model_state_before_sha256'],
                'Entire restored trained model/BN/inactive state unchanged on CUDA transfer')
            handles.append(adapter.backbone.register_forward_hook(hook('backbone')))
            handles.append(adapter.backbone.legacy.CONV0_4.register_forward_hook(hook('feature_exit')))
            for name in ops.HEAD_NAMES:
                module=getattr(adapter.backbone.legacy,name);handles.append(module.register_forward_hook(hook(name)))
            module=None
            with torch.no_grad():
                for index in range(*span):
                    c.check();tick=time.perf_counter();loaded=reader.load(index)
                    record['cost']['cache_RGB_IO_seconds']+=time.perf_counter()-tick
                    need(type(loaded) is tuple and len(loaded)==3,'Genuine U59 returns item,row,originals')
                    item,row,originals=loaded;loaded=None
                    _image(index,item,row,originals,entries[index])
                    # Do not move/reorder item or reset any head here. The old
                    # function owns native zero calibration, one encode, query
                    # indexing, genuine trained readout and native rendering.
                    values,identity,support,proof=core._calibration_and_prediction(torch,np,c,adapter,
                        item,row,originals,{k:modules[k] for k in PER_IMAGE_ROLES},record['cost'],arm=='PLAIN')
                    need(identity==originals and type(values) is list and len(values)==100
                        and type(proof) is dict and proof['dataset_index']==index and proof['image_id']==row['image_id']
                        and proof['queries']==100 and all(proof[k] is True for k in
                            ('zero_residual_native_L_exact','native_RLE_and_prediction_dictionaries_exact',
                             'original_full_S0_class_order_preserved')),
                        'Every genuine U40 calibration/prediction must complete, including original empty native dictionaries')
                    _json([values,identity,support,proof])
                    record['predictions'].extend(values);record['identity_predictions'].extend(identity)
                    record['Omega_support'].extend(support);record['proofs'].append(proof)
                    values=identity=support=proof=None;item=row=originals=None;gc.collect();c.check()
            n=span[1]-span[0];cost=record['cost']
            need(cost['encode_calls']==calls['backbone']==calls['feature_exit']==n
                and cost['readout_calls']==25*n and cost['queries']==100*n
                and all(calls[name]==0 for name in ops.HEAD_NAMES),
                'Original one backbone/image, full100 and25 readouts, no legacy semantic heads')
            record['model_state_after_sha256']=core._state_digest(helpers,adapter.state_dict())
            need(record['model_state_after_sha256']==record['model_state_before_sha256']
                and all(not m.training for m in adapter.modules())
                and all(not p.requires_grad and p.grad is None for p in adapter.parameters()),
                'Trained parameters/BN/buffers and frozen no-grad eval unchanged')
            record['calls']=calls
        limits=core.LIMITS
        need(record['resources']['CUDA_peak_allocated_bytes']<=limits['allocator_bytes']
            and record['resources']['sampled_RSS_peak_bytes']<=limits['maximum_RSS_bytes']
            and not record['resources']['RSS_sampling_errors'],'Original measured arm resource bounds')
        record['status']='PREDICTIONS_COMPLETE_PENDING_CLEANUP'
    except BaseException as error:
        primary=error;record.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        for handle in handles:
            try:handle.remove()
            except BaseException as error:
                record.setdefault('hook_cleanup_errors',[]).append(dict(type=type(error).__name__,message=str(error)))
                traceback.clear_frames(error.__traceback__);error.__traceback__=None
                if cleanup_error is None:cleanup_error=error
        handles.clear();handle=None
        module=adapter=item=row=originals=loaded=values=identity=support=proof=None
        gc.collect()
        try:
            if torch.cuda.is_initialized():record['cleanup']=_cuda_release(torch)
        except BaseException as error:
            record.setdefault('CUDA_cleanup_errors',[]).append(dict(type=type(error).__name__,message=str(error)))
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
            if cleanup_error is None:cleanup_error=error
    if primary is not None:raise primary
    if cleanup_error is not None:
        record['status']='FAILED';raise cleanup_error


def predict_shard(torch,np,c,*,endpoint_metadata,entries,modules,source_refs,shard_index,dataset_range,report):
    """Unbound numerical leaf only. Root owns admission, enforcement and Q.

    CPU restore both genuine original models before any GPU initialization;
    then unchanged U40 per-image numerical code, one active GPU arm at a time.
    Return only JSON pending evidence; exceptions really propagate after cleanup.
    """
    need(type(report) is dict and not report,'Fresh caller-owned partial report')
    started=time.perf_counter();adapters={};reader=adapter=fresh=outside_cpu=outside_cuda=None
    model_refs=[];primary=cleanup_error=None;deadline=None
    report.update(schema='U66-DGX-fixed50-DEV-shard-pending-v1',status='STARTED',arms={},predictions={},
        qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,
        formal_training_budget_credit=0,optimizer_created=False,optimizer_restored=False,automatic_retry=False,
        GT_used_for_prediction=False,GT_rasterization_executed=False,DEV_metrics_computed=False,
        whole_population_predicted=False)
    try:
        need(type(shard_index) is int and 0<=shard_index<4 and type(dataset_range) is list
            and all(type(v) is int for v in dataset_range) and tuple(dataset_range)==RANGES[shard_index],
            'One unchanged canonical DEV shard')
        need(not torch.cuda.is_initialized() and torch.get_default_dtype()==torch.float32
            and torch.get_num_threads()==torch.get_num_interop_threads()==2,'Cold CPU FP32/two-thread entry')
        need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
            and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
            and not torch.is_autocast_enabled(),'Original deterministic FP32 inference, no AMP/TF32')
        _sources(c,modules,source_refs)
        core=modules['prediction_core'];restore=modules['restore_state'];helpers=modules['u36_helpers']
        deadline=core.LIMITS['deadline_seconds']
        endpoint=restore.validate_final_endpoint(modules['training_state'],modules['training_entry'],endpoint_metadata)
        modules['portable_reader'].validate_entries(entries)
        report.update(endpoint=deepcopy(endpoint),shard_index=shard_index,dataset_range=list(dataset_range),
            images=dataset_range[1]-dataset_range[0],queries=100*(dataset_range[1]-dataset_range[0]),
            fixed_epoch=50,updates_per_arm=37050,query_batch=4,source_refs=deepcopy(source_refs))
        outside_cpu=torch.get_rng_state().clone()
        initializer=modules['shared_initialization']
        need(set(initializer.SOURCE_SHA256)==set(CONSTRUCTOR_ROLES),'Original seven constructor roles')
        with helpers._measure(torch,c,report.setdefault('CPU_construction_resources',{}),cuda=False):
            adapters,fresh=initializer.build_pair(torch,{k:modules[k] for k in CONSTRUCTOR_ROLES},
                {k:source_refs[k] for k in CONSTRUCTOR_ROLES})
        need(type(adapters) is dict and set(adapters)==set(ARMS) and not torch.cuda.is_initialized(),
            'Exactly one cold original CPU pair')
        model_refs=[weakref.ref(value) for value in adapters.values()]
        report['fresh_constructor_receipt']=deepcopy(_json(fresh))
        for arm in ARMS:
            record=report['arms'].setdefault(arm,{})
            restore_report=record.setdefault('restore_report',{})
            result=restore.restore_arm(torch,np,c,adapters[arm],arm,endpoint_metadata,fresh,
                modules={k:modules[k] for k in RESTORE_ROLES},source_refs={k:source_refs[k] for k in RESTORE_ROLES},
                report=restore_report)
            need(result is restore_report and result['status']=='RESTORED_CPU_ONLY'
                and result['restored_model_state_exact'] is result['formal_endpoint'] is True
                and result['trained_head_reset'] is result['CUDA_initialized'] is False
                and result['CPU_checkpoint_payload_released'] is True and not torch.cuda.is_initialized(),
                'Both complete FORMAL E50 trained heads restored on CPU before first CUDA')
            _json(restore_report)
        need(torch.equal(torch.get_rng_state(),outside_cpu),'Original constructor/restores retain caller CPU RNG')
        reader=modules['portable_reader'].prepare(torch,c,entries,image_geometry=modules['image_geometry'],
            source_refs={k:source_refs[k] for k in READER_ROLES})
        need(len(reader.records)==187 and [r['dataset_index'] for r in reader.records]==list(range(187))
            and not torch.cuda.is_initialized(),'Full187 metadata reader is prepared, not prediction success')
        c.check();torch.cuda.init();torch.cuda.set_device(0)
        total=int(torch.cuda.get_device_properties(0).total_memory);budget=core.LIMITS['allocator_bytes']
        need(total>=budget,'Original allocator cap supported')
        torch.cuda.set_per_process_memory_fraction(budget/float(total),0)
        need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'Apply original cap before model transfer')
        outside_cuda=torch.cuda.get_rng_state(0).cpu().clone()
        for arm in ARMS:
            _arm(torch,np,c,adapters,reader,entries,modules,RANGES[shard_index],arm,report['arms'][arm])
            record=report['arms'][arm]
            report['predictions'][arm]=record['predictions']
            if arm=='PLAIN':
                report['identity_predictions']=record['identity_predictions'];report['Omega_support']=record['Omega_support']
            record['status']='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
            c.check()
        need(report['arms']['FULL']['identity_predictions']==report['arms']['PLAIN']['identity_predictions'],
            'Both arms retain exactly the same original full S0/class/query/empty-mask dictionaries')
        need(not adapters and torch.equal(torch.get_rng_state(),outside_cpu)
            and torch.equal(torch.cuda.get_rng_state(0).cpu(),outside_cuda),'No retained pair; CPU/CUDA inference RNG unchanged')
        _sources(c,modules,source_refs)
        report.update(model_BN_Torch_RNG_unchanged=True,original_full_S0_class_order_preserved=True,
            source_bytes_rechecked_after=True,
            cost={arm:report['arms'][arm]['cost'] for arm in ARMS},end_to_end_backbone_cost_included=True,
            standalone_FPS_claimed=False,status='PREDICTIONS_COMPLETE_PENDING_FINAL_CLEANUP')
    except BaseException as error:
        primary=error;report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        adapters.clear();reader=adapter=fresh=outside_cpu=outside_cuda=None;gc.collect()
        try:
            if torch.cuda.is_initialized():
                report['cleanup']=_cuda_release(torch)
            report['remaining_model_objects']=sum(reference() is not None for reference in model_refs)
            need(report['remaining_model_objects']==0,'All caller-created adapters released, including failure paths')
        except BaseException as error:
            cleanup_error=error;report.update(status='FAILED',cleanup_error=dict(type=type(error).__name__,message=str(error)))
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
        report['elapsed_seconds']=time.perf_counter()-started
        if deadline is not None and report['elapsed_seconds']>deadline:
            report.update(status='FAILED',deadline_exceeded=True)
            if primary is None and cleanup_error is None:cleanup_error=TimeoutError('Original prediction deadline exceeded')
    if primary is not None:raise primary
    if cleanup_error is not None:raise cleanup_error
    need(report['status']=='PREDICTIONS_COMPLETE_PENDING_FINAL_CLEANUP','Reader/restore alone cannot complete prediction')
    report['status']='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
    try:_json(report)
    except BaseException as error:
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)));raise
    return report
