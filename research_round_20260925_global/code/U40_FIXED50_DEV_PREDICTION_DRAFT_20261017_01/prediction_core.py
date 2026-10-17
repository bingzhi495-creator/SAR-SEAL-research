"""UNBOUND fixed-E50 prediction functions. No scientific imports on import.

Numerical restoration/inference/calibration have not been executed. Root must
provide an admitted, owned, supervised leaf; this is not an entry or a binder.
"""
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import time

SEED=202610173101
ARMS=('PLAIN','FULL')
RANGES=((0,47),(47,94),(94,141),(141,187))
FINAL_CURSOR=dict(global_updates=37050,completed_epochs=50,within_epoch=0)
ORDER_SHA='2afadef62eac9f797c125fa105a3a28f94cdbdd9db4b5b5a2b018d4f0d5f194e'
QUALIFIED_SCHEMA='ROOT-U40-continuous-training-segment-qualified-v1'
TRAINING_COMPONENTS=('shared_initialization','rgb_matched_reader','full_legacy','full_dsconv',
    'full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
    'matched_reader','image_geometry','u36_helpers','u39_training_ops')
TRAINING_ROLES=TRAINING_COMPONENTS+('entry_source','body_source','runtime_source','core_source')
TRAINING_PINS=dict(core_source='8f9b549a8383877aa6f0ae3b1d2a9786c44c0f486e45b447b06dea2e25f5a0c5',
    entry_source='b192e13e6ec13486700affd4fcb3cfe4df550b9bd107b60170103d22e822a68e',
    body_source='ec32a55b6316c3b48734e7278db02b5655fd8ffd8aff0d9d7de70d35949d5268',
    u39_training_ops='378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d')
SOURCE_PINS=dict(
    shared_initialization='4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    full_legacy='c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    full_dsconv='b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    full_head='6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    full_feature='f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    plain_legacy='902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    plain_feature='d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    instance_adapter='9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    u39_training_ops=TRAINING_PINS['u39_training_ops'],
    image_geometry='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    training_state=TRAINING_PINS['core_source'],
    dev_reader='daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    dev_rgb_reader='b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    population_contract='733a208c5df49113e9dff3338b2b88ad2d0d4b26b2bbdd3676ce10417e05c9a5',
    mask_api='1281ce0b5436ffe924c96521b2ae1d589f6f0e2a7b7281b6bb284f7b2245545c')
PREDICTION_ROLES=tuple(SOURCE_PINS)+('encoder',)
LIMITS=dict(deadline_seconds=1800,allowed_cpus=[0,1],threads=2,allocator_bytes=4*1024**3,
    whole_GPU_bytes=8*1024**3,maximum_RSS_bytes=16*1024**3,minimum_available_RAM_bytes=16*1024**3)


def need(ok,message):
    if not ok: raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and PureWindowsPath(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Exact absolute source/input reference')
    return value


def metadata_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate_final_endpoint(metadata):
    """Pure metadata only. Caller must first hash/read all four explicit refs."""
    refs=metadata['refs']
    need(set(refs)=={'training_qualified','pair_commit','preparation','order'},'Explicit endpoint metadata refs')
    for value in refs.values(): strict_ref(value)
    q,pair,prep,order=(metadata[k] for k in ('training_qualified','pair_commit','preparation','order'))
    need(q.get('schema')==QUALIFIED_SCHEMA and q.get('qualified') is True
         and q.get('paired_prefix_advanced') is True and type(q.get('seed')) is int and q['seed']==SEED
         and q['span']==dict(start=36864,stop=37050),'Only Root-accepted complete E50 pair')
    result=q['result']
    need(result['paired_checkpoint_prefix']==FINAL_CURSOR
         and result['pair_commit_ref']==refs['pair_commit'] and result['order_ref']==refs['order']
         and set(result['checkpoint_refs'])==set(ARMS),'Exact E50 references/cursor; no midpoint or single arm')
    for value in result['checkpoint_refs'].values(): strict_ref(value)
    sources=prep['executable_sources']
    need(prep.get('schema')=='ROOT-U40-continuous-training-preparation-v1' and prep['seed']==SEED
         and (prep['images'],prep['GT'],prep['epochs'],prep['total_updates_per_arm'])==(741,1661,50,37050)
         and set(sources)==set(TRAINING_ROLES) and set(prep['module_sources'])==set(TRAINING_COMPONENTS),
         'Same complete preparation, seed, population and seventeen-source identity')
    for value in sources.values(): strict_ref(value)
    need(all(sources[k]['sha256']==v for k,v in TRAINING_PINS.items())
         and prep['module_sources']=={k:sources[k] for k in TRAINING_COMPONENTS},'Original source pins')
    need(q['source']==sources['entry_source']
         and all(q[k]==sources[k] for k in ('body_source','runtime_source','core_source'))
         and q['module_sources']==prep['module_sources'] and q['context']==prep['context'],
         'Root endpoint has the same frozen numerical sources and FIT context')
    need(pair['schema']=='U40-paired-checkpoint-commit-v1'
         and pair['status']=='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE'
         and pair['cursor']==FINAL_CURSOR and pair['span']==q['span']
         and pair['order_ref']==refs['order'] and set(pair['checkpoints'])==set(pair['traces'])==set(ARMS),
         'Pending pair is usable only through the joined Root acceptance')
    need(pair['identity']['sources']==dict(core_source=sources['core_source'],component_sources=prep['module_sources'])
         and pair['identity']['recipe']==prep['recipe'] and order['seed']==SEED
         and pair['identity']['order_sha256']==order['order_sha256']==prep['order_sha256']==ORDER_SHA,
         'Unchanged training recipe and actual shared ORDER')
    for arm in ARMS:
        need(pair['checkpoints'][arm]['reference']==result['checkpoint_refs'][arm],'Exact arm checkpoint')
        strict_ref(pair['traces'][arm])
    return dict(seed=SEED,checkpoint_refs=deepcopy(result['checkpoint_refs']),training_sources=deepcopy(sources),
                training_qualified=deepcopy(refs['training_qualified']),pair_commit_ref=deepcopy(refs['pair_commit']),
                order_ref=deepcopy(refs['order']))


def export_query_chunks(query_ids,originals,image_id):
    need(type(query_ids) is list and len(query_ids)==100
         and all(type(q) is int for q in query_ids) and set(query_ids)==set(range(100))
         and type(originals) is list and len(originals)==100
         and [(v['image_id'],v['query_id']) for v in originals]==[(image_id,q) for q in query_ids],
         'Original export permutation must address natural e/M/L axes')
    return [query_ids[first:first+4] for first in range(0,100,4)]


def validate_population(records,native,shard_index,dataset_range):
    need(type(shard_index) is int and 0<=shard_index<4 and type(dataset_range) is list
         and all(type(v) is int for v in dataset_range) and tuple(dataset_range)==RANGES[shard_index],
         'One of the four fixed canonical shards')
    need(type(records) is list and len(records)==187 and type(native) is list and len(native)==18700
         and len({r['image_id'] for r in records})==187,'Complete original 187/18700 population')
    for index,row in enumerate(records):
        need(type(row['dataset_index']) is int and row['dataset_index']==index
             and type(row['image_id']) is int,'Canonical dataset index/image identity')
        export_query_chunks(row['qids'],native[index*100:(index+1)*100],row['image_id'])
    return RANGES[shard_index]


def _closed(c,reference):
    strict_ref(reference)
    path=Path(c.closed(reference))
    need(path.resolve()==Path(reference['path']).resolve(),'Exact admitted reference path')
    return path


def _json(c,reference):
    path=_closed(c,reference)
    data=path.read_bytes()
    need(len(data)==reference['bytes'] and hashlib.sha256(data).hexdigest()==reference['sha256'],
         'Exact admitted JSON bytes')
    return json.loads(data)


def _sources(c,modules,source_refs,core_ref,training_sources,context):
    need(set(modules)==set(source_refs)==set(PREDICTION_ROLES),'Exact seventeen prediction module roles')
    for role,module in modules.items():
        reference=source_refs[role]
        path=_closed(c,reference)
        need(Path(module.__file__).resolve()==path.resolve(),'Injected source path: '+role)
        raw=path.read_bytes()
        expected=SOURCE_PINS[role] if role!='encoder' else context['encoder_source']['sha256']
        need(reference['sha256']==expected and len(raw)==reference['bytes']
             and hashlib.sha256(raw).hexdigest()==expected,'Pinned actual prediction code: '+role)
        if role in training_sources:
            need(reference==training_sources[role],'Reuse exact frozen training component ref: '+role)
    need(source_refs['encoder']==context['encoder_source']
         and source_refs['training_state']==training_sources['core_source'],'Original encoder and training-state helpers')
    for reference in training_sources.values():
        # The final training has terminated before this future evaluation. Only
        # its code refs are closed here, never the FIT reader or its data.
        _closed(c,reference)
    own=_closed(c,core_ref)
    need(own.resolve()==Path(__file__).resolve(),'This exact prediction core must be source-closed')


def _state_digest(helpers,state):
    return metadata_sha([dict(name=name,sha256=helpers._digest(value)) for name,value in sorted(state.items())])


def _calibration_and_prediction(torch,np,c,adapter,item,row,originals,modules,cost,collect_support):
    """Own all one-image CUDA aliases; return only JSON-compatible CPU values."""
    encoder,masks,contract=(modules[k] for k in ('encoder','mask_api','population_contract'))
    started=time.perf_counter()
    qids=item['query_ids']; image_id=row['image_id']
    chunks=export_query_chunks(qids,originals,image_id)
    geometry=item['geometry']
    c.check(); torch.cuda.synchronize(); tick=time.perf_counter()
    rgb=item['rgb400'].to(device='cuda:0',dtype=torch.float32)
    encoded=adapter.encode(rgb)
    torch.cuda.synchronize(); cost['encode_seconds']+=time.perf_counter()-tick
    need(tuple(encoded.shape)==(1,32,400,400) and encoded.dtype==torch.float32
         and not encoded.requires_grad and bool(torch.isfinite(encoded).all()),'One finite shared encode per image')
    cost['encode_calls']+=1
    del rgb
    rles=[]; identity_rles=[]; support=[]
    for group_index,queries in enumerate(chunks):
        c.check(); begin=group_index*4
        selected=torch.tensor(queries,dtype=torch.int64,device='cpu')
        # All three inputs are natural query-axis tensors. Scores are never input.
        e=item['e'].index_select(0,selected).to('cuda:0')
        M=item['M'].index_select(0,selected).to('cuda:0')
        L=item['L'].index_select(0,selected).to('cuda:0')
        native_binary=L>0; omega=L.abs()<=2
        torch.cuda.synchronize(); tick=time.perf_counter()
        zero=torch.zeros((len(queries),1,400,400),dtype=torch.float32,device='cuda:0')
        rendered=geometry.render_delta(zero)[:,0]
        need(tuple(rendered.shape)==tuple(L.shape) and torch.equal(rendered,torch.zeros_like(L))
             and torch.equal(L+rendered,L),'All native logits exactly unchanged by zero renderer')
        zero_binary=(L+rendered)>0
        for offset,qid in enumerate(queries):
            rle=encoder.segmentation(np,masks,zero_binary[offset].cpu().numpy().astype(np.uint8))
            need(rle==originals[begin+offset]['segmentation'],'Every zero native RLE exactly matches, including empty')
            identity_rles.append(rle)
            if collect_support:
                support.append(dict(dataset_index=row['dataset_index'],image_id=image_id,query_id=qid,
                    Omega=encoder.segmentation(np,masks,omega[offset].cpu().numpy().astype(np.uint8))))
        torch.cuda.synchronize(); cost['zero_calibration_seconds']+=time.perf_counter()-tick
        del zero,rendered,zero_binary
        tick=time.perf_counter()
        delta=adapter.predict(encoded,e,M)
        torch.cuda.synchronize(); cost['readout_seconds']+=time.perf_counter()-tick
        need(tuple(delta.shape)==(len(queries),1,400,400) and delta.dtype==torch.float32
             and not delta.requires_grad and bool(torch.isfinite(delta).all()),'Trained finite FP32 residual, head never reset')
        tick=time.perf_counter()
        corrected=L+geometry.render_delta(delta)[:,0]
        need(corrected.shape==L.shape and corrected.dtype==torch.float32
             and bool(torch.isfinite(corrected).all()),'Original-grid native L plus rendered delta')
        binary=corrected>0; changed=binary!=native_binary
        cost['changed_inside_Omega']+=int((changed&omega).sum().item())
        cost['changed_outside_Omega']+=int((changed&~omega).sum().item())
        torch.cuda.synchronize(); cost['render_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        for offset in range(len(queries)):
            rles.append(encoder.segmentation(np,masks,binary[offset].cpu().numpy().astype(np.uint8)))
        cost['RLE_seconds']+=time.perf_counter()-tick
        cost['readout_calls']+=1; cost['queries']+=len(queries)
        del selected,e,M,L,native_binary,omega,delta,corrected,binary,changed
    identity=contract.replace_segmentations(originals,identity_rles,image_id,qids)
    need(identity==originals,'Whole original S0/class/query/non-segmentation dictionaries exactly calibrated')
    predictions=contract.replace_segmentations(originals,rles,image_id,qids)
    del encoded,geometry
    return predictions,identity,support,dict(dataset_index=row['dataset_index'],image_id=image_id,queries=100,
        zero_residual_native_L_exact=True,native_RLE_and_prediction_dictionaries_exact=True,
        original_full_S0_class_order_preserved=True,elapsed_seconds=time.perf_counter()-started)


def _arm(torch,np,c,adapters,reader,native,modules,endpoint,metadata,fresh_receipt,span,arm,record):
    """Pop the active model so the caller never retains a CUDA model alias."""
    ops,helpers,training=(modules[k] for k in ('u39_training_ops','u36_helpers','training_state'))
    adapter=saved=item=params=module=None
    handles=[]; predictions=[]; identities=[]; supports=[]; proofs=[]
    calls=dict(backbone=0,feature_exit=0,**{name:0 for name in ops.HEAD_NAMES})
    cost=dict(encode_calls=0,readout_calls=0,queries=0,encode_seconds=0.,readout_seconds=0.,
        render_seconds=0.,RLE_seconds=0.,zero_calibration_seconds=0.,cache_RGB_IO_seconds=0.,
        restore_seconds=0.,changed_inside_Omega=0,changed_outside_Omega=0)
    record.update(status='STARTED',cost=cost,resources={})
    def hook(name):
        def observe(module,args,output): calls[name]+=1
        return observe
    try:
        with helpers._measure(torch,c,record['resources'],cuda=True):
            tick=time.perf_counter()
            adapter=adapters.pop(arm)
            effective,inactive,record['parameters']=ops._partition(adapter)
            saved=torch.load(str(_closed(c,endpoint['checkpoint_refs'][arm])),map_location='cpu')
            pair=metadata['pair_commit']
            need(saved['schema']=='U40-continuous-arm-state-v1' and saved['arm']==arm
                 and saved['cursor']==FINAL_CURSOR and saved['identity']==pair['identity']
                 and saved['effective_parameter_names']==sorted(effective)
                 and saved['inactive_parameter_names']==sorted(inactive)
                 and saved['order_ref']==endpoint['order_ref']
                 and saved['actual_trace_sha256']==pair['actual_trace_sha256']
                 and saved['requested_trace_sha256']==training.expected_trace(metadata['order'],37050),
                 'Exact final arm, source/recipe/cursor/name/actual/requested-order identity')
            need(saved['initialization_receipt']==fresh_receipt,'Original common104/adapter initialization lineage exact')
            need(all(value.device.type=='cpu' and (not value.is_floating_point() or
                (value.dtype==torch.float32 and bool(torch.isfinite(value).all()))) for value in saved['model'].values()),
                'Entire finite CPU FP32 model state, including BN and inactive states')
            adapter.load_state_dict(saved['model'],strict=True)
            training._equal(torch,adapter.state_dict(),saved['model'],'E50 CPU model/BN/inactive')
            need(set(dict(adapter.named_modules()))==set(saved['module_training_modes']), 'Same full module tree')
            record['model_state_before_sha256']=_state_digest(helpers,adapter.state_dict())
            adapter.eval().requires_grad_(False)
            adapter.to(device='cuda:0',dtype=torch.float32)
            training._equal(torch,adapter.state_dict(),saved['model'],'E50 GPU model/BN/inactive')
            saved=None; gc.collect()
            # torch.load included the original CPU optimizer dictionary; it is
            # now released. No optimizer object or optimizer load was created.
            record['restored_model_state_exact']=True
            record['initialization_lineage_exact']=True
            record['optimizer_created']=record['optimizer_restored']=False
            record['cpu_checkpoint_may_include_optimizer_payload']=True
            cost['restore_seconds']=time.perf_counter()-tick
            handles.append(adapter.backbone.register_forward_hook(hook('backbone')))
            handles.append(adapter.backbone.legacy.CONV0_4.register_forward_hook(hook('feature_exit')))
            for name in ops.HEAD_NAMES:
                module=getattr(adapter.backbone.legacy,name)
                handles.append(module.register_forward_hook(hook(name)))
            module=None
            with torch.no_grad():
                for index in range(*span):
                    c.check(); tick=time.perf_counter(); item=reader.load(index)
                    cost['cache_RGB_IO_seconds']+=time.perf_counter()-tick
                    values,identity,support,proof=_calibration_and_prediction(torch,np,c,adapter,item,
                        reader.records[index],native[index*100:(index+1)*100],modules,cost,arm=='PLAIN')
                    predictions.extend(values); identities.extend(identity); supports.extend(support); proofs.append(proof)
                    del values,identity,support,proof
                    item=None; gc.collect(); c.check()
            record['calls']=calls
            n=span[1]-span[0]
            need(cost['encode_calls']==calls['backbone']==calls['feature_exit']==n
                 and cost['readout_calls']==25*n and cost['queries']==100*n
                 and all(calls[name]==0 for name in ops.HEAD_NAMES),'Exactly one backbone/image; all100; no old semantic heads')
            record['model_state_after_sha256']=_state_digest(helpers,adapter.state_dict())
            need(record['model_state_after_sha256']==record['model_state_before_sha256']
                 and all(not m.training for m in adapter.modules())
                 and all(not p.requires_grad and p.grad is None for p in adapter.parameters()),
                 'Every trained parameter/BN/buffer unchanged; eval and no gradients')
            record['status']='PREDICTIONS_COMPLETE_PENDING_CLEANUP'
    except BaseException as error:
        record['status']='FAILED'; record['error']=ops._failure(error)
    finally:
        for handle in handles: handle.remove()
        handles.clear()
        module=adapter=params=saved=item=None
        gc.collect()
        try: record['cleanup']=training._cleanup(torch)
        except BaseException as error:
            record['status']='FAILED'; record['cleanup_error']=ops._failure(error)
    if record['status']!='FAILED':
        need(record['resources']['CUDA_peak_allocated_bytes']<=LIMITS['allocator_bytes']
             and record['resources']['sampled_RSS_peak_bytes']<=LIMITS['maximum_RSS_bytes']
             and not record['resources']['RSS_sampling_errors'],'Arm measured resource limits')
        record['status']='COMPLETED'
    return predictions,identities,supports,proofs


def predict_shard(torch,np,c,*,endpoint_refs,dev_context_ref,original_image_refs,modules,source_refs,
                  prediction_core_ref,shard_index,dataset_range,output_directory,limits,report):
    """Future admitted GPU leaf body only; result is pending Root acceptance.

    c must supply closed/check/write_json/psutil. Root's wrapper owns actual
    hard1800s, whole-GPU8GiB and continuous RSS/free-RAM enforcement. This core
    enforces allocator4GiB and observes resources; it does not launch/reap.
    """
    need(type(report) is dict and not report,'Caller retains a fresh partial report')
    started=time.perf_counter(); directory=None; adapters={}; reader=None; ops=None
    report.update(schema='U40-fixed50-full-method-DEV-shard-pending-v1',status='STARTED',
        arms={},predictions={},new_training=False,formal_main_training_started=False,
        AP=False,DEV=True,TEST=False,qualified=False,automatic_retry=False)
    try:
        need(not torch.cuda.is_initialized() and torch.__version__=='2.0.1+cu118'
             and np.__version__=='1.23.5' and platform.python_version()=='3.8.17'
             and torch.get_default_dtype()==torch.float32,'Pinned cold CPU runtime')
        need(limits==LIMITS and torch.get_num_threads()==torch.get_num_interop_threads()==2
             and c.psutil.Process().cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0',
             'Explicit unchanged resource/CPU/GPU contract')
        need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
             and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
             and not torch.is_autocast_enabled(),'Fixed FP32 deterministic inference; no AMP')
        c.check()
        metadata=dict(refs=deepcopy(endpoint_refs),**{k:_json(c,r) for k,r in endpoint_refs.items()})
        endpoint=validate_final_endpoint(metadata)
        context=_json(c,dev_context_ref)
        _sources(c,modules,source_refs,prediction_core_ref,endpoint['training_sources'],context)
        ops=modules['u39_training_ops']; training=modules['training_state']; helpers=modules['u36_helpers']
        training.validate_order(metadata['order'])
        need(metadata['pair_commit']['identity']['recipe']==training.RECIPE
             and metadata['pair_commit']['identity']['runtime']==dict(python=platform.python_version(),
                 torch=str(torch.__version__),numpy=str(np.__version__)),'Original exact training recipe/runtime')
        directory=Path(output_directory)
        need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists()
             and not directory.is_symlink(),'Exclusive future prediction output')
        directory.mkdir()
        report.update(endpoint,dev_context=deepcopy(dev_context_ref),prediction_sources=deepcopy(source_refs),
            prediction_core_source=deepcopy(prediction_core_ref),shard_index=shard_index,dataset_range=list(dataset_range),
            fixed_epoch=50,updates_per_arm=37050,query_batch=4,limits=deepcopy(limits),
            historical_target_fields_may_be_in_bundle=True,GT_used_for_prediction=False,
            DEV_COCO_annotation_opened=False,original_teacher_forward=False,
            inference_GT_scope='No new DEV COCO annotation input; historical bundle association fields are not model inputs')
        io_tick=time.perf_counter()
        reader=modules['dev_rgb_reader'].prepare(torch,np,c,context,original_image_refs,
            dev_reader=modules['dev_reader'],image_geometry=modules['image_geometry'],
            source_refs={k:source_refs[k] for k in ('dev_reader','image_geometry')})
        native=_json(c,context['native_predictions'])
        span=validate_population(reader.records,native,shard_index,dataset_range)
        report['images']=span[1]-span[0]; report['queries']=100*report['images']
        report['metadata_preparation_seconds']=time.perf_counter()-io_tick
        report['original_image_references_metadata_sha256']=metadata_sha(reader.image_references)
        outside_cpu=torch.get_rng_state().clone()
        tick=time.perf_counter()
        initializer=modules['shared_initialization']
        with helpers._measure(torch,c,report.setdefault('CPU_construction_resources',{}),cuda=False):
            adapters,receipt=initializer.build_pair(torch,{k:modules[k] for k in initializer.SOURCE_SHA256},
                {k:source_refs[k] for k in initializer.SOURCE_SHA256})
        report['fresh_constructor_receipt']=receipt
        report['CPU_construction_seconds']=time.perf_counter()-tick
        need(not torch.cuda.is_initialized() and torch.equal(torch.get_rng_state(),outside_cpu),
             'Exactly one cold CPU pair; external CPU RNG retained')
        torch.cuda.init(); torch.cuda.set_device(0)
        total=int(torch.cuda.get_device_properties(0).total_memory)
        need(total>=LIMITS['allocator_bytes'],'Physical allocator budget supported')
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/float(total),0)
        need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'Apply fixed cap before first model transfer')
        outside_cuda=torch.cuda.get_rng_state(0).cpu().clone()
        for arm in ARMS:
            values,identities,supports,proofs=_arm(torch,np,c,adapters,reader,native,modules,endpoint,
                metadata,receipt,span,arm,report['arms'].setdefault(arm,{}))
            need(report['arms'][arm]['status']=='COMPLETED','Incomplete arm; preserve partial, never retry')
            report['arms'][arm]['proofs']=proofs
            report['predictions'][arm]=c.write_json(directory/(arm+'_PREDICTIONS.json'),values)
            if arm=='PLAIN':
                report['identity_predictions']=c.write_json(directory/'IDENTITY_PREDICTIONS.json',identities)
                report['Omega_support']=c.write_json(directory/'OMEGA_SUPPORT.json',supports)
            need(identities==native[span[0]*100:span[1]*100],'Complete exact native identity population per arm')
            del values,identities,supports,proofs
            gc.collect(); c.check()
        need(not adapters and torch.equal(torch.get_rng_state(),outside_cpu)
             and torch.equal(torch.cuda.get_rng_state(0).cpu(),outside_cuda),'All models released; inference RNG unchanged')
        report.update(original_full_S0_class_order_preserved=True,
            native_all_RLE_and_identity_prediction_dictionaries_exact=True,zero_residual_native_L_exact=True,
            model_BN_RNG_unchanged=True,cost={a:report['arms'][a]['cost'] for a in ARMS},
            end_to_end_backbone_cost_included=True,teacher_precompute_cost_included=False,standalone_FPS_claimed=False,
            CPU_checkpoint_optimizer_payload_loaded_but_not_restored=True,optimizer_created=False,
            global_paired_endpoint_modified=False,status='PREDICTIONS_COMPLETE_PENDING_FINAL_CLEANUP')
        report['torch_peak_allocated_bytes']=max(report['arms'][a]['resources']['CUDA_peak_allocated_bytes'] for a in ARMS)
        report['sampled_RSS_peak_bytes']=max([report['CPU_construction_resources']['sampled_RSS_peak_bytes']]+
            [report['arms'][a]['resources']['sampled_RSS_peak_bytes'] for a in ARMS])
        reader=None; del native,metadata,receipt,outside_cpu,outside_cuda
    except BaseException as error:
        report['status']='FAILED'
        if ops is not None: report['error']=ops._failure(error)
        else:
            import traceback
            report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            traceback.clear_frames(error.__traceback__); error.__traceback__=None
    finally:
        adapters.clear(); reader=None; gc.collect()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache(); torch.cuda.synchronize()
                report['cleanup']=dict(allocated_after=int(torch.cuda.memory_allocated()),
                    reserved_after=int(torch.cuda.memory_reserved()))
                need(report['cleanup']==dict(allocated_after=0,reserved_after=0),'Final hard-zero CUDA release')
            except BaseException as error:
                report['status']='FAILED'; report['cleanup_error']=dict(type=type(error).__name__,message=str(error))
        report['elapsed_seconds']=time.perf_counter()-started
        if report['elapsed_seconds']>LIMITS['deadline_seconds']:
            report['status']='FAILED'; report['deadline_exceeded']=True
    if report['status']=='PREDICTIONS_COMPLETE_PENDING_FINAL_CLEANUP': report['status']='COMPLETED'
    if directory is not None and directory.is_dir():
        report_ref=c.write_json(directory/'DEV_REPORT.json',report)
    else: report_ref=None  # caller must preserve supplied partial report on pre-output failure
    need(report['status']=='COMPLETED','Fixed-E50 prediction failed; partial report preserved; no retry')
    result={k:deepcopy(report[k]) for k in ('predictions','identity_predictions','Omega_support','dataset_range',
        'shard_index','images','queries','seed','fixed_epoch','updates_per_arm','training_qualified',
        'pair_commit_ref','checkpoint_refs','order_ref','dev_context','training_sources','prediction_sources')}
    result.update(report_ref=report_ref,new_training=False,AP=False,DEV=True,TEST=False,qualified=False)
    return result
