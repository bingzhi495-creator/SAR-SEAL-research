"""PLANNED, UNBOUND independent DGX original-update body. No ML imports at import.

Only Root may admit inputs and call run_training in a future owned DGX leaf.
Metadata tests are not model, optimizer, checkpoint or resource validation.
"""
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import random
import re
import sys
import time

SEED = 202610173101
IMAGES, EPOCHS, TOTAL = 741, 50, 37050
ARMS = ('PLAIN', 'FULL')
U39_SHA256 = '378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d'
ORDER_RULE = 'python38-private-Random-full-seed-new-range741-shuffle-each-epoch-v1'
TRACE_START = hashlib.sha256(b'U57-actual-matched-input-trace-v1').hexdigest()
LIMITS = dict(deadline_seconds=1800, allowed_cpus=[0, 1], threads=2,
              allocator_bytes=4*1024**3,
              maximum_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3)
RECIPE = dict(seed=SEED, images=IMAGES, epochs=EPOCHS, updates=TOTAL,
              dtype='float32', image_batch=1, branch_size=400, canvas_size=800, query_chunk=1,
              optimizer='AdamW', lr=3e-4, weight_decay=1e-4, betas=[.9, .999], eps=1e-8,
              foreach=False, fused=False, AMP=False, VJP=False, checkpoint_recompute=False,
              encoder_saves='canonical_save_on_cpu_pin_memory_true', scheduler=None,
              loss='query_mean_original_grid_BCEWithLogits_plus_softDice_smooth1',
              order_rule=ORDER_RULE, pretrained_UNet_weights=False)
ENVIRONMENT=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64')
PRODUCER_SHA256='8f9b549a8383877aa6f0ae3b1d2a9786c44c0f486e45b447b06dea2e25f5a0c5'
SOURCE_PINS={
    'shared_initialization':'4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    'full_legacy':'c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    'full_dsconv':'b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    'full_head':'6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    'full_feature':'f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    'plain_legacy':'902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    'plain_feature':'d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    'instance_adapter':'9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    'image_geometry':'65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    'u36_helpers':'1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    'u39_training_ops':U39_SHA256,
    'portable_reader':'b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def backend_policy():
    return dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic_algorithms_enabled=False,
        deterministic_warn_only=False,cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
        cudnn_benchmark=False,cudnn_deterministic=True)


def backend_settings(torch):
    return dict(CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)


def training_contract(mode,run_id):
    need(mode in ('FORMAL','SMOKE') and type(run_id) is str and 1<=len(run_id)<=128
         and '/' not in run_id and '\\' not in run_id and run_id not in ('.','..'),'Explicit separate DGX mode/run identity')
    return dict(schema='U57-DGX-original-update-training-prelock-v1',mode=mode,run_id=run_id,
        environment=deepcopy(ENVIRONMENT),limits=deepcopy(LIMITS),backend_policy=backend_policy(),
        recipe=deepcopy(RECIPE),smoke_indices=[54,710] if mode=='SMOKE' else None)


def validate_prelock(prelock):
    need(type(prelock) is dict and metadata_sha256(prelock)==metadata_sha256(training_contract(prelock['mode'],prelock['run_id'])),
         'Exact independent DGX original-update recipe, nonstrict backend and separate smoke/formal profile')


def validate_span(segment,prelock):
    validate_prelock(prelock)
    need(type(segment) is dict and set(segment)=={'start','stop'},'Exact same two-arm segment span')
    start,stop=segment['start'],segment['stop']
    need(type(start) is int and type(stop) is int and 0<=start<stop<= (2 if prelock['mode']=='SMOKE' else TOTAL),
         'Nonempty common cursor span within the separately prelocked budget')
    if prelock['mode']=='SMOKE':need((start,stop) in ((0,1),(1,2)),'One smoke comprises exactly two adjacent one-update segments')
    return start,stop


def metadata_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _json_state(value):
    return json.loads(json.dumps(value, allow_nan=False))


def validate_order_producer(value):
    """Producer is CPython3.8; the independent DGX consumer may be3.12."""
    need(type(value) is dict and value.get('python_implementation')=='CPython'
         and value.get('python_major_minor')==[3,8] and type(value.get('python_version')) is str
         and re.fullmatch(r'3\.8\.[0-9]+',value['python_version']) is not None,
         'A newly generated original CPython3.8 order, not a consumer-version substitution')


def validate_order(plan):
    need(type(plan) is dict and plan.get('seed') == SEED, 'Exact order metadata')
    validate_order_producer(plan)
    fixed = dict(schema='U40-private-epoch-order-v1', seed=SEED, images=IMAGES,
        epochs=EPOCHS, updates=TOTAL, rule=ORDER_RULE,
        python_implementation='CPython', python_version=plan['python_version'], python_major_minor=[3,8])
    need(set(plan) == set(fixed) | {'generator_initial_state','generator_final_state','order','order_sha256'}
         and all(type(plan[k]) is type(v) and plan[k] == v for k,v in fixed.items()), 'Exact order recipe/version')
    need(type(plan['order']) is list and len(plan['order']) == TOTAL
         and all(type(i) is int for i in plan['order']), 'Complete integer order')
    for epoch in range(EPOCHS):
        need(sorted(plan['order'][epoch*IMAGES:(epoch+1)*IMAGES]) == list(range(IMAGES)), 'Every epoch is the full population')
    for label in ('generator_initial_state','generator_final_state'):
        state=plan[label]
        need(type(state) is list and len(state)==3 and state[0]==3 and type(state[1]) is list
             and len(state[1])==625 and all(type(v) is int and 0<=v<2**32 for v in state[1][:-1])
             and type(state[1][-1]) is int and 0<=state[1][-1]<=624 and state[2] is None,
             'Original recorded Python MT state structure; consumer never reshuffles or invents producer RNG')
    need(plan['order_sha256'] == metadata_sha256({k:v for k,v in plan.items() if k != 'order_sha256'}),
         'Stored order and initial/final generator-state digest')
    # Resume consumes the stored list/cursor. It never calls shuffle again.
    return plan


def cursor(completed):
    need(type(completed) is int and 0 <= completed <= TOTAL, 'Valid global completed-update cursor')
    return dict(global_updates=completed, completed_epochs=completed//IMAGES, within_epoch=completed % IMAGES)


def validate_parent_accepted(value,parent,prelock,start,order_ref):
    """Root summary metadata only; pending worker pair or local U40 cannot stand in."""
    need(type(value) is dict and value.get('schema')=='Root-U57-DGX-original-update-training-accepted-v1'
         and value.get('root_accepted') is True and value.get('paired_prefix_advanced') is True
         and value.get('mode')==prelock['mode'] and value.get('run_id')==prelock['run_id'],
         'Actual Root-accepted same DGX run/profile parent, never local/smoke/formal substitution')
    result=value['result']
    need(result['pair_commit_ref']==parent['pair_commit_ref'] and result['checkpoint_refs']==parent['checkpoint_refs']
         and result['trace_refs']==parent['trace_refs'] and result['order_ref']==order_ref
         and result['paired_checkpoint_prefix']==cursor(start),'Root accepted exact paired parent cursor and all refs')


def append_trace(previous, ordinal, index):
    return metadata_sha256(dict(previous=previous, global_ordinal=ordinal, dataset_index=index))


def expected_trace(plan, stop):
    cursor(stop)
    value = TRACE_START
    for ordinal, index in enumerate(plan['order'][:stop]):
        value = append_trace(value, ordinal, index)
    return value


def _ref(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def _closed(c, reference):
    need(type(reference) is dict and set(reference) == {'path', 'bytes', 'sha256'}
         and Path(reference['path']).is_absolute(), 'Strict admitted input reference')
    path = Path(c.closed(reference))
    need(_ref(path) == reference, 'Exact input bytes and path')
    return path


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return _ref(path)


def _verify_sources(c, modules, source_refs, core_ref):
    need(set(modules)==set(source_refs)==set(SOURCE_PINS),'Exactly eleven original numeric components plus U54 reader')
    for role,pin in SOURCE_PINS.items():
        need(source_refs[role]['sha256']==pin
             and Path(modules[role].__file__).resolve()==_closed(c,source_refs[role]).resolve(),
             'Exact injected original component source/object: '+role)
    need(_closed(c, core_ref).resolve() == Path(__file__).resolve(), 'Caller admitted this exact new core')
    return modules['u39_training_ops']


def _cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: _cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(torch, v) for v in value)
    need(value is None or type(value) in (str, int, float, bool), 'Only tensor/scalar checkpoint trees')
    return value


def _signature(torch, helpers, value):
    if isinstance(value, torch.Tensor):
        need(value.device.type == 'cpu', 'Checkpoint tensor must be CPU')
        return dict(tensor_sha256=helpers._digest(value), shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return dict(mapping=[dict(key_type=type(k).__name__, key=k, value=_signature(torch, helpers, v))
                             for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return dict(sequence_type=type(value).__name__, values=[_signature(torch, helpers, v) for v in value])
    need(value is None or type(value) in (str, int, float, bool), 'Scalar signature')
    return value


def _equal(torch, actual, expected, label):
    if isinstance(expected, torch.Tensor):
        need(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
             and tuple(actual.shape) == tuple(expected.shape)
             and torch.equal(actual.detach().cpu(), expected), 'Exact tensor restore: '+label)
    elif isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), 'Exact state keys: '+label)
        for key in expected:
            _equal(torch, actual[key], expected[key], label+'/'+str(key))
    elif isinstance(expected, (list, tuple)):
        need(type(actual) is type(expected) and len(actual) == len(expected), 'Exact state sequence: '+label)
        for i, value in enumerate(expected):
            _equal(torch, actual[i], value, label+'/'+str(i))
    else:
        need(type(actual) is type(expected) and actual == expected, 'Exact state scalar: '+label)


def _rng(torch, np):
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=n[0], keys=n[1].tolist(), position=int(n[2]),
        has_gauss=int(n[3]), cached_gaussian=float(n[4])), torch_CPU=torch.get_rng_state().clone(),
        torch_CUDA=torch.cuda.get_rng_state(0).cpu().clone())


def _restore_rng(torch, np, value):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(value['torch_CPU'])
    torch.cuda.set_rng_state(value['torch_CUDA'], 0)
    _equal(torch, _rng(torch, np), value, 'all arm RNG')


def _seed_arm(torch, np):
    random.seed(SEED)
    np.random.seed(SEED % 2**32)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _modes(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


def _optimizer_devices(optimizer, params, names):
    return {name: {key: str(value.device) for key, value in optimizer.state[params[name]].items()
                   if hasattr(value, 'device')} for name in names}


def _save_checkpoint(torch, helpers, c, path, payload):
    """One arm CPU snapshot at a time; exact serialized roundtrip, no replay."""
    expected = metadata_sha256(_signature(torch, helpers, payload))
    partial = path.with_suffix(path.suffix+'.partial')
    need(not path.exists() and not partial.exists(), 'Never overwrite checkpoint or failed partial')
    with partial.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    payload.clear()  # No second complete CPU snapshot retained for roundtrip.
    gc.collect()
    loaded = None
    try:
        c.check()
        loaded = torch.load(str(partial), map_location='cpu', weights_only=True)  # independent Torch2.10 portable tensor/primitive state
        need(metadata_sha256(_signature(torch, helpers, loaded)) == expected, 'All saved CPU state roundtrips exactly')
    finally:
        loaded = None
        gc.collect()
    need(not path.exists(), 'Exclusive final checkpoint name')
    os.rename(str(partial), str(path))
    return dict(reference=_ref(path), CPU_state_signature_sha256=expected, serialization_roundtrip_verified=True)


def _actual_event(stage, ordinal):
    return dict(global_ordinal=ordinal, dataset_index=stage['dataset_index'], image_id=stage['image_id'],
                G=stage['G'], query_ids=stage['query_ids'], ann_ids=stage['ann_ids'],
                original_shape=stage['original_shape'], actual_img_shape=stage['actual_img_shape'],
                original_image=stage['original_image'])


def _cleanup(torch):
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    value = dict(allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()))
    need(value == dict(allocated=0, reserved=0), 'Hard-zero complete arm CUDA release')
    return value


def _train_arm(torch, np, c, reader, adapters, modules, plan, start, stop,
               identity, parent_ref, parent_pair, directory, arm, record, report):
    ops, helpers = modules['u39_training_ops'], modules['u36_helpers']
    adapter = optimizer = params = saved = payload = legacy = head_buffers = None
    handles = []
    record.update(status='RUNNING', updates_completed=0, updates=[], checkpoint=None, active_update=None)
    calls = dict(backbone=0, feature_exit=0, legacy_forward=0, **{k: 0 for k in ops.HEAD_NAMES})
    def hook(name):
        def count(module, args, output):
            calls[name] += 1
        return count
    try:
        adapter = adapters.pop(arm)  # Caller never holds the active CUDA model.
        effective, inactive, record['parameters'] = ops._partition(adapter)
        need(all(value.device.type == 'cpu' and value.dtype == torch.float32 for value in adapter.parameters())
             and not torch.is_autocast_enabled(), 'Original fresh CPU FP32 skeleton; no autocast')
        names = sorted(effective)
        initial_receipt = report['fresh_constructor_receipt']
        actual_trace = TRACE_START
        requested_trace = expected_trace(plan, start)
        if start:
            saved = torch.load(str(_closed(c, parent_ref)), map_location='cpu', weights_only=True)
            need(saved['schema'] == 'U57-DGX-original-update-arm-state-v1' and saved['arm'] == arm
                 and saved['identity'] == identity and saved['cursor'] == cursor(start)
                 and saved['effective_parameter_names'] == names and saved['inactive_parameter_names'] == sorted(inactive)
                 and saved['requested_trace_sha256'] == requested_trace
                 and saved['order_ref'] == report['order_ref'], 'Exact arm/recipe/source/cursor/order parent checkpoint')
            need(metadata_sha256(_signature(torch, helpers, saved)) == parent_pair['checkpoints'][arm]['CPU_state_signature_sha256'],
                 'Parent complete CPU state signature')
            need(saved['actual_trace_sha256'] == parent_pair['actual_trace_sha256'], 'Same accepted paired actual prefix')
            adapter.load_state_dict(saved['model'], strict=True)
            _equal(torch, adapter.state_dict(), saved['model'], 'model including BN/inactive')
            need(set(_modes(adapter)) == set(saved['module_training_modes']), 'Exact restored module tree')
            for name, module in adapter.named_modules():
                module.training = saved['module_training_modes'][name]
            initial_receipt = saved['initialization_receipt']
            actual_trace = saved['actual_trace_sha256']
        else:
            adapter.train()
        adapter.to(device='cuda:0', dtype=torch.float32)
        params = dict(adapter.named_parameters())
        need(set(params) == effective | inactive and all(v.requires_grad for v in params.values()), 'Original parameter/flag set')
        optimizer = torch.optim.AdamW([params[name] for name in names], lr=3e-4, weight_decay=1e-4,
                                     betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
        if start:
            optimizer.load_state_dict(saved['optimizer'])
            _equal(torch, adapter.state_dict(), saved['model'], 'GPU model restore')
            _equal(torch, optimizer.state_dict(), saved['optimizer'], 'optimizer mapping/moments/groups')
            need(_optimizer_devices(optimizer, params, names) == saved['optimizer_state_devices'], 'Original AdamW state devices')
            need(_modes(adapter) == saved['module_training_modes'], 'Training modes restored exactly')
            helpers._optimizer_report(torch, optimizer, params, effective, start)
            _restore_rng(torch, np, saved['rng'])  # after construction/load, before next input
            record['restored_all_state_exact'] = True
            saved = None
            gc.collect()
        else:
            _seed_arm(torch, np)
            record['restored_all_state_exact'] = None
        legacy = adapter.backbone.legacy
        head_buffers = helpers._head_buffers(torch, legacy)
        need(len(head_buffers) == (12 if arm == 'PLAIN' else 18), 'Original inactive semantic-head BN inventory')
        record['semantic_head_BN_buffer_count'] = len(head_buffers)
        for module, label in ((adapter.backbone,'backbone'), (legacy.CONV0_4,'feature_exit'), (legacy,'legacy_forward')):
            handles.append(module.register_forward_hook(hook(label)))
        for label in ops.HEAD_NAMES:
            handles.append(getattr(legacy, label).register_forward_hook(hook(label)))
        parameter_ids = {name: id(value) for name, value in params.items()}
        bn_before = ops._bn_signature(torch, legacy, helpers)
        record['start_cursor'] = cursor(start)
        with (directory/(arm+'.trace.jsonl')).open('x', encoding='utf-8', newline='\n') as trace:
            for ordinal in range(start, stop):
                stage = dict(status='RUNNING', global_step=ordinal+1, optimizer_step_returned=False)
                record['active_update'] = stage
                before = dict(calls)
                need(ops._bn_signature(torch, legacy, helpers) == bn_before, 'BN continuous across images')
                ops._update(torch, c, reader, plan['order'][ordinal], adapter, optimizer, params, effective, inactive,
                            helpers, ordinal+1, stage, record, report)
                need({name: id(value) for name, value in adapter.named_parameters()} == parameter_ids, 'Same parameters across segment')
                delta_calls = {k: calls[k]-before[k] for k in calls}
                need(delta_calls['backbone'] == delta_calls['feature_exit'] == 1
                     and delta_calls['legacy_forward'] == 0 and all(delta_calls[k] == 0 for k in ops.HEAD_NAMES),
                     'One encode per update, old semantic heads bypassed')
                helpers._verify_buffers(torch, legacy, head_buffers)
                bn_before = ops._bn_signature(torch, legacy, helpers)
                gc.collect()
                resources = ops._sample(torch, c, 'after_image_release_with_Adam_BN_resident', True)
                need(resources['CUDA_peak_allocated_bytes'] <= LIMITS['allocator_bytes']
                     and resources['RSS_bytes'] <= LIMITS['maximum_RSS_bytes']
                     and resources['available_RAM_bytes'] >= LIMITS['minimum_available_RAM_bytes'], 'Unchanged resource limits')
                event = _actual_event(stage, ordinal)
                actual_trace = metadata_sha256(dict(previous=actual_trace, actual_input=event))
                requested_trace = append_trace(requested_trace, ordinal, plan['order'][ordinal])
                row = dict(actual_input=event, actual_trace_sha256=actual_trace, requested_trace_sha256=requested_trace,
                    cursor=cursor(ordinal+1), loss=stage['loss'], BCE=stage['BCE'], softDice=stage['softDice'],
                    effective_gradient_tensors=len(effective), inactive_gradient_tensors=len(inactive),
                    optimizer_state_entries=len(optimizer.state), optimizer_step_returned=True,
                    calls=delta_calls, elapsed_seconds=stage['training_with_checks_elapsed_seconds'],
                    CPU_saved_tensor_events=stage['encoder_saved_on_cpu'], resources=resources)
                trace.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n'); trace.flush()
                record['updates'].append(dict(global_step=ordinal+1, dataset_index=event['dataset_index'],
                    G=event['G'], loss=stage['loss'], actual_trace_sha256=actual_trace))
                record['active_update'] = None
                stage = None
        need(record['updates_completed'] == stop-start and requested_trace == expected_trace(plan, stop), 'Exact actual update budget')
        c.check()
        payload = dict(schema='U57-DGX-original-update-arm-state-v1', arm=arm, identity=deepcopy(identity),
            cursor=cursor(stop), model=_cpu_tree(torch, adapter.state_dict()),
            optimizer=_cpu_tree(torch, optimizer.state_dict()), effective_parameter_names=names,
            inactive_parameter_names=sorted(inactive), optimizer_state_devices=_optimizer_devices(optimizer, params, names),
            module_training_modes=_modes(adapter), rng=_rng(torch, np), order_ref=report['order_ref'],
            requested_trace_sha256=requested_trace, actual_trace_sha256=actual_trace,
            initialization_receipt=initial_receipt, previous_checkpoint=deepcopy(parent_ref),
            previous_pair=deepcopy(report['parent_pair_ref']))
        record['checkpoint'] = _save_checkpoint(torch, helpers, c, directory/(arm+'.pth'), payload)
        payload = None
        record['trace_ref'] = _ref(directory/(arm+'.trace.jsonl'))
        record['end_cursor'] = cursor(stop)
        record['actual_trace_sha256'] = actual_trace
        record['requested_trace_sha256'] = requested_trace
        record['status'] = 'CHECKPOINT_WRITTEN'
    except BaseException as error:
        record['status'] = 'FAILED'
        if record['active_update'] is not None:
            record['active_update']['status'] = 'FAILED'
        record['error'] = ops._failure(error)
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        # The hook setup loop otherwise leaves its final module in this frame.
        module = adapter = params = optimizer = saved = payload = legacy = head_buffers = None
        gc.collect()
        try:
            record['GPU_cleanup'] = _cleanup(torch)
        except BaseException as error:
            record['status'] = 'FAILED'
            record['cleanup_error'] = ops._failure(error)


def _input_validation(c,context,manifest):
    summary=json.loads(_closed(c,context['U54_validation_ref']).read_bytes())
    refs=context['U54_validation_sources']
    need(type(refs) is dict and set(refs)=={'worker_report_ref','owner_receipt_ref'}
         and summary['schema']=='Root-U54-DGX-full-FIT-accepted-v1'
         and summary['status']=='ROOT_ACCEPTED_ALL741_CPU_VALIDATION' and summary['root_accepted'] is True
         and summary['images']==741 and summary['GT']==1661 and summary['failed_loads']==0
         and summary['dataset_indices']==list(range(741)) and summary['manifest_ref']==context['manifest_ref']
         and summary['package_refs_sha256']==hashlib.sha256(json.dumps(
             [row['package_ref'] for row in manifest['rows']],sort_keys=True,separators=(',',':'),
             ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()
         and summary['no_training'] is True and summary['CUDA_uninitialized'] is True,
         'Actual complete Root-accepted U54 inputs required before any model construction; not a planned CPU check')
    terminal=summary['original_terminal']
    need(all(type(terminal[k]) is int and terminal[k]==0 for k in
             ('root_observed_tool_exit_code','SSH_returncode','worker_returncode'))
         and all(terminal[k] is True for k in ('reaped','owner_original_birth_absent','child_original_birth_absent',
                                             'all_owned_processes_absent','source_and_logs_byte_verified')),
         'Actual accepted input-validation parent terminal and complete source/log evidence')
    for key,reference in refs.items():
        need(summary[key]==reference,'Explicit accepted U54 associated source reference: '+key)
        _closed(c,reference)
    return summary


def _order_generation(c,reference,plan,order_ref,prelock):
    value=json.loads(_closed(c,reference).read_bytes())
    producer={key:plan[key] for key in ('python_implementation','python_version','python_major_minor')}
    need(type(value) is dict and set(value)=={'schema','root_created_new','mode','run_id','producer',
         'generator_source_sha256','order_ref','order_metadata_sha256','no_old_training_state_read'}
         and value['schema']=='U57-independent-Python38-order-generation-v1'
         and value['root_created_new'] is True and value['mode']=='FORMAL'
         and type(value['run_id']) is str and value['run_id'] and value['producer']==producer
         and value['generator_source_sha256']==PRODUCER_SHA256 and value['order_ref']==order_ref
         and value['order_metadata_sha256']==plan['order_sha256'] and value['no_old_training_state_read'] is True,
         'New independent original Python3.8 production receipt; no bound local ORDER or old training state')
    if prelock['mode']=='FORMAL':need(value['run_id']==prelock['run_id'],'Order generation belongs to this formal experiment')
    return value


def completed_training(report,segment,prelock):
    """Actual worker metadata gate; does not advance a Root frontier or replay NN."""
    start,stop=validate_span(segment,prelock)
    need(report['schema']=='U57-DGX-original-update-training-segment-report-v1'
         and report['status']=='SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE'
         and report['mode']==prelock['mode'] and report['run_id']==prelock['run_id'] and report['seed']==SEED
         and report['span']==dict(start=start,stop=stop) and report['paired_checkpoint_prefix']==cursor(stop)
         and report['successful_optimizer_updates']==report['optimizer_steps_entered']==2*(stop-start)
         and report['new_training'] is True and report['qualified'] is report['paired_prefix_advanced'] is False
         and all(report[key] is False for key in ('AP','DEV','TEST','whole_trajectory_bitexact_claimed','weights_published','automatic_retry'))
         and report['pilot_only'] is (prelock['mode']=='SMOKE')
         and report['formal_main_training_started'] is (prelock['mode']=='FORMAL')
         and report['formal_training_budget_credit']==0
         and set(report['checkpoint_refs'])==set(report['trace_refs'])==set(ARMS)
         and report['final_GPU_cleanup']==dict(allocated=0,reserved=0)
         and report['backend_settings_before']==report['backend_settings_after']==prelock['backend_policy']
         and all(report[key] is True for key in ('caller_CPU_RNG_restored','caller_NumPy_RNG_restored',
              'caller_Python_RNG_restored','caller_postinitialization_CUDA_RNG_restored','source_unchanged_after')),
         'Both exact spans completed, safely paired, source-closed, RNG-restored and GPU empty; no self-qualification')
    for arm in ARMS:
        row=report['arms'][arm]
        need(row['status']=='CHECKPOINT_WRITTEN' and row['updates_completed']==stop-start
             and row['start_cursor']==cursor(start) and row['end_cursor']==cursor(stop)
             and row['checkpoint']['reference']==report['checkpoint_refs'][arm]
             and row['trace_ref']==report['trace_refs'][arm]
             and row['checkpoint']['serialization_roundtrip_verified'] is True
             and row['GPU_cleanup']==dict(allocated=0,reserved=0)
             and row['restored_all_state_exact'] is (True if start else None)
             and row['actual_trace_sha256']==report['actual_trace_sha256'],
             'Both full-state checkpoints/actual input chains/strict state restoration and hard GPU cleanup')


def run_training(torch,np,c,*,modules,source_refs,context,segment,prelock,body_source_ref,output_directory,report):
    """Future Root-owned DGX invocation only; neither old U40 nor candidate code runs here.

    No implicit parent, automatic retry, model selection or partial-arm reuse.
    Two state files and actual input traces must finish before a pending pair
    commit exists. Actual terminal and Root acceptance remain external.
    """
    need(type(report) is dict and not report,'Fresh caller-owned partial training report')
    started=time.perf_counter();adapters=reader=ops=None
    outside_cpu=outside_numpy=outside_python=outside_cuda=None
    report.update(schema='U57-DGX-original-update-training-segment-report-v1',status='STARTED',arms={},
        seed=SEED,successful_optimizer_updates=0,optimizer_steps_entered=0,new_training=False,
        formal_training_attempted=False,formal_main_training_started=False,pilot_only=prelock.get('mode')=='SMOKE',
        formal_training_budget_credit=0,paired_prefix_advanced=False,qualified=False,AP=False,DEV=False,TEST=False,
        whole_trajectory_bitexact_claimed=False,weights_published=False,automatic_retry=False,
        whole_GPU_observed_peak_MiB=None,whole_GPU_WDDM8GiB_contract_certified=False,cleanup_errors=[])
    try:
        validate_prelock(prelock);start,stop=validate_span(segment,prelock)
        need(type(context) is dict and set(context)=={'manifest_ref','order_ref','order_generation_ref',
             'U54_validation_ref','U54_validation_sources','parent_pair'},'Explicit new DGX data/order/qualification/model-parent refs')
        actual=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),
                    system=platform.system(),machine=platform.machine())
        need(actual==ENVIRONMENT and not torch.cuda.is_initialized() and torch.get_default_dtype()==torch.float32
             and torch.is_grad_enabled() and not torch.is_autocast_enabled()
             and torch.get_num_threads()==torch.get_num_interop_threads()==2
             and c.psutil.Process().cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0',
             'Independent cold DGX3.12/Torch2.10 runtime, original FP32 and admitted CPU01/GPU0')
        report['backend_settings_before']=backend_settings(torch)
        need(report['backend_settings_before']==prelock['backend_policy'],
             'New explicitly nonstrict algorithm policy; unsupported strict FULL experiment is not silently mixed')
        outside_cpu=torch.get_rng_state().clone();outside_numpy=np.random.get_state();outside_python=random.getstate()
        ops=_verify_sources(c,modules,source_refs,body_source_ref)
        manifest=json.loads(_closed(c,context['manifest_ref']).read_bytes())
        modules['portable_reader'].validate_manifest(manifest)
        _input_validation(c,context,manifest)
        plan=json.loads(_closed(c,context['order_ref']).read_bytes());validate_order(plan)
        _order_generation(c,context['order_generation_ref'],plan,context['order_ref'],prelock)
        active_plan=dict(plan,order=prelock['smoke_indices'] if prelock['mode']=='SMOKE' else plan['order'])
        report.update(mode=prelock['mode'],run_id=prelock['run_id'],prelock=deepcopy(prelock),span=dict(start=start,stop=stop),
            order_ref=deepcopy(context['order_ref']),parent_pair_ref=None,
            output_directory=str(output_directory),input_validation_ref=deepcopy(context['U54_validation_ref']))
        stable_context={key:context[key] for key in context if key!='parent_pair'}
        identity=dict(mode=prelock['mode'],run_id=prelock['run_id'],recipe=deepcopy(RECIPE),environment=actual,
            backend_policy=deepcopy(prelock['backend_policy']),sources=dict(body_source=body_source_ref,component_sources=source_refs),
            context=stable_context,order_sha256=plan['order_sha256'],execution_order_sha256=metadata_sha256(active_plan['order']))
        report['identity']=deepcopy(identity)
        parent_pair=None;parent_refs={arm:None for arm in ARMS};parent=context['parent_pair']
        if start==0:
            need(parent is None,'Fresh FORMAL or one-time SMOKE starts0 without any local/old/hidden state')
        else:
            need(type(parent) is dict and set(parent)=={'qualified_ref','pair_commit_ref','checkpoint_refs','trace_refs'}
                 and set(parent['checkpoint_refs'])==set(parent['trace_refs'])==set(ARMS),'Explicit complete Root-accepted DGX paired parent')
            accepted=json.loads(_closed(c,parent['qualified_ref']).read_bytes())
            validate_parent_accepted(accepted,parent,prelock,start,context['order_ref'])
            parent_pair=json.loads(_closed(c,parent['pair_commit_ref']).read_bytes());parent_refs=parent['checkpoint_refs']
            need(parent_pair['schema']=='U57-DGX-paired-checkpoint-commit-v1'
                 and parent_pair['status']=='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE'
                 and parent_pair['identity']==identity and parent_pair['cursor']==cursor(start)
                 and parent_pair['order_ref']==context['order_ref'] and parent_pair['checkpoint_refs']==parent_refs
                 and parent_pair['trace_refs']==parent['trace_refs'],
                 'Same new DGX profile/sources/data/order and paired frontier; local2816 cannot enter')
            for arm in ARMS:
                need(parent_pair['checkpoints'][arm]['reference']==parent_refs[arm]
                     and parent_pair['traces'][arm]==parent['trace_refs'][arm],'Parent direct and complete state/trace references match')
                _closed(c,parent_refs[arm]);_closed(c,parent['trace_refs'][arm])
            report['parent_pair_ref']=deepcopy(parent['pair_commit_ref'])
        report['parent_checkpoint_refs']=deepcopy(parent_refs)
        directory=Path(output_directory)
        need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists() and not directory.is_symlink(),
             'Exclusive explicitly provided new segment directory')
        directory.mkdir()
        reader=modules['portable_reader'].prepare(torch,c,manifest,image_geometry=modules['image_geometry'],
            source_refs={key:source_refs[key] for key in ('portable_reader','image_geometry')})
        need(len(reader.records)==741 and sum(row['G'] for row in reader.records)==1661,'Complete original matched FIT metadata')
        initializer=modules['shared_initialization']
        adapters,receipt=initializer.build_pair(torch,{key:modules[key] for key in initializer.SOURCE_SHA256},
                                               {key:source_refs[key] for key in initializer.SOURCE_SHA256})
        need(receipt['seed']==receipt['adapter_seed']==SEED and len(receipt['common_state'])==104
             and receipt['common_parameter_tensors']==68 and receipt['common_BN_buffers']==36,
             'Original common104 and shared zero-readout initialization; full/plain capacities are not identical')
        report['fresh_constructor_receipt']=receipt
        report['constructor_role']='initial_common_pair' if not start else 'CPU_skeleton_then_all_trained_state_restored'
        need(not torch.cuda.is_initialized(),'Single actual CPU construction before first CUDA')
        torch.cuda.init();torch.cuda.set_device(0)
        total=int(torch.cuda.get_device_properties(0).total_memory)
        need(total>=LIMITS['allocator_bytes'],'Device supports unchanged4GiB allocator')
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/float(total),0)
        need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'Allocator cap before model/input transfer')
        outside_cuda=torch.cuda.get_rng_state(0).cpu().clone()
        report['formal_training_attempted']=prelock['mode']=='FORMAL'
        for arm in ARMS:
            record=report['arms'][arm]={}
            _train_arm(torch,np,c,reader,adapters,modules,active_plan,start,stop,identity,parent_refs[arm],parent_pair,directory,arm,record,report)
            need(record['status']=='CHECKPOINT_WRITTEN','Actual arm/state/cleanup failure; no paired advance or automatic retry: '+arm)
            c.check()
        need(report['successful_optimizer_updates']==report['optimizer_steps_entered']==2*(stop-start)
             and report['arms']['PLAIN']['actual_trace_sha256']==report['arms']['FULL']['actual_trace_sha256'],
             'Same complete two-arm actual input prefix and exact independent update budgets')
        _verify_sources(c,modules,source_refs,body_source_ref)
        for key in ('manifest_ref','order_ref','order_generation_ref','U54_validation_ref'):_closed(c,context[key])
        report['source_unchanged_after']=True
        report['backend_settings_after']=backend_settings(torch)
        need(report['backend_settings_after']==prelock['backend_policy'],'Nonstrict original-update backend stayed fixed')
        report['reader_visits']=list(reader.visits)
        need(reader.visits==active_plan['order'][start:stop]*2 and reader.failed_loads==0,
             'Only the original same ordered full-G inputs actually visited by both arms')
        report['status']='READY_FOR_PAIRED_COMMIT'
    except BaseException as error:
        report['status']='FAILED'
        report['error']=ops._failure(error) if ops is not None else dict(type=type(error).__name__,message=str(error))
    finally:
        if reader is not None:
            report['reader_visits']=list(reader.visits)
            report['reader_failed_loads']=int(reader.failed_loads)
        adapters=reader=None;gc.collect()
        if torch.cuda.is_initialized():
            try:
                if outside_cuda is not None:
                    torch.cuda.set_rng_state(outside_cuda,0)
                    report['caller_postinitialization_CUDA_RNG_restored']=bool(torch.equal(outside_cuda,torch.cuda.get_rng_state(0).cpu()))
                report['final_GPU_cleanup']=_cleanup(torch)
            except BaseException as error:
                report['status']='FAILED';report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error)))
        if outside_cpu is not None:
            try:
                torch.set_rng_state(outside_cpu);np.random.set_state(outside_numpy);random.setstate(outside_python)
                report['caller_CPU_RNG_restored']=bool(torch.equal(outside_cpu,torch.get_rng_state()))
                now=np.random.get_state()
                report['caller_NumPy_RNG_restored']=now[0]==outside_numpy[0] and bool(np.array_equal(now[1],outside_numpy[1])) and now[2:]==outside_numpy[2:]
                report['caller_Python_RNG_restored']=random.getstate()==outside_python
            except BaseException as error:
                report['status']='FAILED';report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error)))
        outside_cpu=outside_numpy=outside_python=outside_cuda=None
        report['formal_main_training_started']=prelock.get('mode')=='FORMAL' and report['optimizer_steps_entered']>0
        report['elapsed_seconds']=time.perf_counter()-started
    if report['status']=='READY_FOR_PAIRED_COMMIT':
        try:
            c.check()
            checkpoints={arm:report['arms'][arm]['checkpoint'] for arm in ARMS}
            traces={arm:report['arms'][arm]['trace_ref'] for arm in ARMS}
            cp_refs={arm:checkpoints[arm]['reference'] for arm in ARMS}
            for value in list(cp_refs.values())+list(traces.values()):need(_ref(value['path'])==value,'Complete new state/trace bytes unchanged before pair commit')
            actual_trace=report['arms']['PLAIN']['actual_trace_sha256']
            pair=dict(schema='U57-DGX-paired-checkpoint-commit-v1',status='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE',
                identity=identity,cursor=cursor(stop),span=dict(start=start,stop=stop),order_ref=report['order_ref'],
                previous_pair=report['parent_pair_ref'],checkpoints=checkpoints,traces=traces,
                checkpoint_refs=cp_refs,trace_refs=traces,actual_trace_sha256=actual_trace)
            report.update(checkpoint_refs=cp_refs,trace_refs=traces,
                actual_trace_sha256=actual_trace,paired_checkpoint_prefix=cursor(stop),status='SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE')
            completed_training(report,segment,prelock)
            report['pair_commit_ref']=_write_json(directory/'PAIR_COMMIT.json',pair)
        except BaseException as error:
            report.update(status='FAILED',commit_error=dict(type=type(error).__name__,message=str(error)))
    report['elapsed_seconds']=time.perf_counter()-started
    json.dumps(report,allow_nan=False)
    return report

