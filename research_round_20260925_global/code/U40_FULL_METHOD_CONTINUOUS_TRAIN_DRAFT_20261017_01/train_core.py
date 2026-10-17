"""PLANNED, UNBOUND continuous training core. No scientific imports at import.

Only Root may admit inputs and call train_segment in a future owned leaf.
Metadata tests are not model, optimizer, checkpoint or resource validation.
"""
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

SEED = 202610173101
IMAGES, EPOCHS, TOTAL = 741, 50, 37050
ARMS = ('PLAIN', 'FULL')
U39_SHA256 = '378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d'
ORDER_RULE = 'python38-private-Random-full-seed-new-range741-shuffle-each-epoch-v1'
TRACE_START = hashlib.sha256(b'U40-actual-matched-input-trace-v1').hexdigest()
LIMITS = dict(deadline_seconds=1800, allowed_cpus=[0, 1], threads=2,
              allocator_bytes=4*1024**3, whole_GPU_bytes=8*1024**3,
              maximum_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3)
RECIPE = dict(seed=SEED, images=IMAGES, epochs=EPOCHS, updates=TOTAL,
              dtype='float32', image_batch=1, branch_size=400, canvas_size=800, query_chunk=1,
              optimizer='AdamW', lr=3e-4, weight_decay=1e-4, betas=[.9, .999], eps=1e-8,
              foreach=False, fused=False, AMP=False, VJP=False, checkpoint_recompute=False,
              encoder_saves='canonical_save_on_cpu_pin_memory_true', scheduler=None,
              loss='query_mean_original_grid_BCEWithLogits_plus_softDice_smooth1',
              order_rule=ORDER_RULE, pretrained_UNet_weights=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def metadata_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _json_state(value):
    return json.loads(json.dumps(value, allow_nan=False))


def generate_order(seed=SEED):
    """Pure metadata; production also requires the recorded/current Python 3.8."""
    need(type(seed) is int and seed == SEED, 'Only the first full prelocked seed is implemented')
    rng = random.Random(seed)
    initial = _json_state(rng.getstate())
    order = []
    for _ in range(EPOCHS):
        epoch = list(range(IMAGES))
        rng.shuffle(epoch)
        order.extend(epoch)
    plan = dict(schema='U40-private-epoch-order-v1', seed=seed, images=IMAGES,
        epochs=EPOCHS, updates=TOTAL, rule=ORDER_RULE,
        python_implementation=platform.python_implementation(), python_version=platform.python_version(),
        python_major_minor=list(sys.version_info[:2]), generator_initial_state=initial,
        generator_final_state=_json_state(rng.getstate()), order=order)
    plan['order_sha256'] = metadata_sha256(plan)
    return plan


def validate_order(plan, require_python38=True):
    need(type(plan) is dict and plan.get('seed') == SEED, 'Exact order metadata')
    if require_python38:
        need(tuple(sys.version_info[:2]) == (3, 8)
             and plan.get('python_major_minor') == [3, 8], 'Production order requires Python 3.8')
    fixed = dict(schema='U40-private-epoch-order-v1', seed=SEED, images=IMAGES,
        epochs=EPOCHS, updates=TOTAL, rule=ORDER_RULE,
        python_implementation=platform.python_implementation(), python_version=platform.python_version(),
        python_major_minor=list(sys.version_info[:2]))
    need(set(plan) == set(fixed) | {'generator_initial_state','generator_final_state','order','order_sha256'}
         and all(type(plan[k]) is type(v) and plan[k] == v for k,v in fixed.items()), 'Exact order recipe/version')
    need(type(plan['order']) is list and len(plan['order']) == TOTAL
         and all(type(i) is int for i in plan['order']), 'Complete integer order')
    for epoch in range(EPOCHS):
        need(sorted(plan['order'][epoch*IMAGES:(epoch+1)*IMAGES]) == list(range(IMAGES)), 'Every epoch is the full population')
    need(plan['generator_initial_state'] == _json_state(random.Random(SEED).getstate()), 'Original private generator initial state')
    need(plan['order_sha256'] == metadata_sha256({k:v for k,v in plan.items() if k != 'order_sha256'}),
         'Stored order and initial/final generator-state digest')
    # Resume consumes the stored list/cursor. It never calls shuffle again.
    return plan


def cursor(completed):
    need(type(completed) is int and 0 <= completed <= TOTAL, 'Valid global completed-update cursor')
    return dict(global_updates=completed, completed_epochs=completed//IMAGES, within_epoch=completed % IMAGES)


def validate_segment(segment, plan):
    need(type(segment) is dict and type(segment.get('seed')) is int and segment['seed'] == SEED,
         'First full seed only')
    start, stop = segment.get('start'), segment.get('stop')
    cursor(start); cursor(stop)
    need(start < stop and segment.get('order_sha256') == plan['order_sha256'], 'Nonempty fixed common span/order')
    return start, stop


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
    need(set(modules) == set(source_refs) and 'u39_training_ops' in modules, 'Explicit U39 plus twelve components')
    ops = modules['u39_training_ops']
    need(source_refs['u39_training_ops']['sha256'] == U39_SHA256
         and Path(ops.__file__).resolve() == _closed(c, source_refs['u39_training_ops']).resolve(), 'Pinned original U39')
    need(set(modules) == set(ops.SOURCE_SHA256) | {'u39_training_ops'}, 'Exactly thirteen module roles')
    ops._closed_sources(c, {k: modules[k] for k in ops.SOURCE_SHA256},
                        {k: source_refs[k] for k in ops.SOURCE_SHA256})
    need(_closed(c, core_ref).resolve() == Path(__file__).resolve(), 'Caller admitted this exact new core')
    return ops


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
        loaded = torch.load(str(partial), map_location='cpu')  # trusted admitted self-produced ABI, Torch2.0.1
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
            saved = torch.load(str(_closed(c, parent_ref)), map_location='cpu')
            need(saved['schema'] == 'U40-continuous-arm-state-v1' and saved['arm'] == arm
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
        payload = dict(schema='U40-continuous-arm-state-v1', arm=arm, identity=deepcopy(identity),
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


def train_segment(torch, np, c, context, modules, source_refs, segment,
                  parent_checkpoint_refs, output_directory, report=None, image_control=None):
    """Future owned formal segment; FAILED is never success, no automatic retry.

    Caller owns source/library admission, accepted parent refs, CPU affinity,
    continuous RSS/RAM/whole-GPU limits and hard 1800s process supervision.
    Caller must preserve this report and reject non-completed status.
    """
    if report is None:
        report = {}
    need(type(report) is dict and not report, 'Fresh partial report')
    report.update(schema='U40-continuous-segment-report-v1', status='STARTED', arms={},
        successful_optimizer_updates=0, optimizer_steps_entered=0, new_training=False,
        formal_main_training_started=False, formal_training_attempted=False, paired_prefix_advanced=False, AP=False, DEV=False, TEST=False,
        qualified=False, whole_trajectory_bitexact_claimed=False, weights_published=False, automatic_retry=False)
    started = time.perf_counter()
    adapters = reader = ops = None
    try:
        need(tuple(sys.version_info[:2]) == (3,8) and platform.python_implementation() == 'CPython', 'Pinned production Python 3.8')
        need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
             and not torch.cuda.is_initialized() and torch.get_default_dtype() == torch.float32, 'Pinned cold CPU runtime')
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2
             and c.psutil.Process().cpu_affinity() == [0,1] and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Admitted CPU01/2threads/GPU0')
        need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
             and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False,
             'Fixed FP32 deterministic settings')
        need(segment.get('limits') == LIMITS and segment.get('recipe') == RECIPE, 'Explicit unchanged prelocked limits/recipe')
        ops = _verify_sources(c, modules, source_refs, segment['core_source'])
        directory = Path(output_directory)
        need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists() and not directory.is_symlink(), 'Exclusive scientific segment output')
        directory.mkdir()
        parent_pair = None
        if segment['start'] == 0:
            need(parent_checkpoint_refs == {'PLAIN':None,'FULL':None} and segment.get('parent_pair_ref') is None
                 and segment.get('order_ref') is None, 'Fresh first segment has no hidden parent fallback')
            plan = generate_order()
            validate_order(plan)
            report['order_ref'] = _write_json(directory/'ORDER.json', plan)
        else:
            plan = json.loads(_closed(c, segment['order_ref']).read_bytes())
            validate_order(plan)
            report['order_ref'] = deepcopy(segment['order_ref'])
            parent_pair = json.loads(_closed(c, segment['parent_pair_ref']).read_bytes())
        start, stop = validate_segment(segment, plan)
        need(type(parent_checkpoint_refs) is dict and set(parent_checkpoint_refs) == set(ARMS), 'Both exact arm parents')
        report.update(span=dict(start=start,stop=stop), parent_pair_ref=deepcopy(segment.get('parent_pair_ref')),
                      parent_checkpoint_refs=deepcopy(parent_checkpoint_refs), output_directory=str(directory))
        source_identity = dict(core_source=segment['core_source'], component_sources=source_refs)
        identity = dict(recipe=RECIPE, sources=source_identity, context=context,
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__)),
            order_sha256=plan['order_sha256'])
        report['identity'] = deepcopy(identity)
        if start:
            need(parent_pair['schema'] == 'U40-paired-checkpoint-commit-v1'
                 and parent_pair['status'] == 'PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE'
                 and parent_pair['identity'] == identity and parent_pair['cursor'] == cursor(start)
                 and parent_pair['order_ref'] == report['order_ref'] and set(parent_pair['checkpoints']) == set(ARMS)
                 and set(parent_pair['traces']) == set(ARMS), 'Exact paired accepted parent frontier')
            for arm in ARMS:
                need(parent_pair['checkpoints'][arm]['reference'] == parent_checkpoint_refs[arm], 'No cross-arm/seed checkpoint substitution')
                _closed(c, parent_checkpoint_refs[arm])
                _closed(c, parent_pair['traces'][arm])
        rgb = modules['rgb_matched_reader']
        reader = rgb.prepare(torch,np,c,context,matched_reader=modules['matched_reader'],
            image_geometry=modules['image_geometry'],source_refs={k:source_refs[k] for k in rgb.SOURCE_SHA256},image_control=image_control)
        need(len(reader.records) == IMAGES and sum(r['G'] for r in reader.records) == 1661, 'Entire original FIT metadata population')
        initializer = modules['shared_initialization']
        adapters, receipt = initializer.build_pair(torch,{k:modules[k] for k in initializer.SOURCE_SHA256},
                                                   {k:source_refs[k] for k in initializer.SOURCE_SHA256})
        report['fresh_constructor_receipt'] = receipt
        report['constructor_role'] = 'initial_common_pair' if not start else 'temporary_CPU_skeleton_before_strict_parent_restore'
        need(not torch.cuda.is_initialized(), 'Only one cold CPU pair construction per leaf')
        torch.cuda.init(); torch.cuda.set_device(0)
        total = int(torch.cuda.get_device_properties(0).total_memory)
        need(total >= LIMITS['allocator_bytes'], 'Fixed budget supported by physical device')
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/float(total),0)
        need(torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0, 'Cap before model/input transfer')
        report['formal_training_attempted'] = True
        for arm in ARMS:
            record = report['arms'][arm] = {}
            _train_arm(torch,np,c,reader,adapters,modules,plan,start,stop,identity,parent_checkpoint_refs[arm],
                       parent_pair,directory,arm,record,report)
            if record['status'] != 'CHECKPOINT_WRITTEN':
                raise RuntimeError('Arm failed; no paired frontier advance or retry: '+arm)
            c.check()
        need(report['successful_optimizer_updates'] == 2*(stop-start)
             and report['optimizer_steps_entered'] == 2*(stop-start), 'Both exact update budgets')
        need(report['arms']['PLAIN']['actual_trace_sha256'] == report['arms']['FULL']['actual_trace_sha256'], 'Same actual matched input prefix')
        _verify_sources(c,modules,source_refs,segment['core_source'])
        report['status'] = 'READY_FOR_PAIRED_COMMIT'
    except BaseException as error:
        report['status'] = 'FAILED'
        if ops is not None:
            report['error'] = ops._failure(error)
        else:
            report['error'] = dict(type=type(error).__name__,message=str(error))
    finally:
        adapters = reader = None
        gc.collect()
        if torch.cuda.is_initialized():
            try:
                report['final_GPU_cleanup'] = _cleanup(torch)
            except BaseException as error:
                report['status'] = 'FAILED'
                report['cleanup_error'] = dict(type=type(error).__name__,message=str(error))
        report['formal_main_training_started'] = report['optimizer_steps_entered'] > 0
        report['elapsed_seconds'] = time.perf_counter()-started
    if report['status'] == 'READY_FOR_PAIRED_COMMIT':
        try:
            c.check()
            for arm in ARMS:
                for value in (report['arms'][arm]['checkpoint']['reference'], report['arms'][arm]['trace_ref']):
                    need(_ref(value['path']) == value, 'New complete checkpoint/actual trace unchanged before paired commit')
            pair = dict(schema='U40-paired-checkpoint-commit-v1',
                status='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE',identity=identity,cursor=cursor(stop),
                span=dict(start=start,stop=stop),order_ref=report['order_ref'],previous_pair=report['parent_pair_ref'],
                checkpoints={arm:report['arms'][arm]['checkpoint'] for arm in ARMS},
                traces={arm:report['arms'][arm]['trace_ref'] for arm in ARMS},
                actual_trace_sha256=report['arms']['PLAIN']['actual_trace_sha256'])
            report['pair_commit_ref'] = _write_json(directory/'PAIR_COMMIT.json',pair)
            report['status'] = 'SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE'
            report['paired_checkpoint_prefix'] = cursor(stop)
            # Actual terminal/Root acceptance is outside this function.
        except BaseException as error:
            report['status'] = 'FAILED'
            report['commit_error'] = dict(type=type(error).__name__,message=str(error))
    report['elapsed_seconds'] = time.perf_counter()-started  # includes checkpoint/trace rehash and paired commit
    json.dumps(report,allow_nan=False)
    return report
