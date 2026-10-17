"""UNBOUND U39 resource probe; no scientific imports or work at module import."""
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


SOURCE_SHA256 = {
    'shared_initialization': '4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    'rgb_matched_reader': '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025',
    'full_legacy': 'c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    'full_dsconv': 'b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    'full_head': '6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    'full_feature': 'f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    'plain_legacy': '902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    'plain_feature': 'd93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    'instance_adapter': '9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    'matched_reader': '82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870',
    'image_geometry': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    'u36_helpers': '1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
}
HEAD_NAMES = tuple('final_super_0_'+str(index) for index in range(1, 5))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def select_image_order(records):
    need(type(records) in (list, tuple) and bool(records), 'Nonempty original metadata records')
    seen = set()
    for row in records:
        need(type(row) is dict and type(row.get('dataset_index')) is int
             and 0 <= row['dataset_index'] < 741 and row['dataset_index'] not in seen
             and type(row.get('G')) is int and 1 <= row['G'] <= 100
             and type(row.get('identity')) is dict, 'Unique original ordinal and matched G')
        shape = row['identity'].get('original_shape')
        need(type(shape) in (list, tuple) and len(shape) == 2
             and all(type(value) is int and value > 0 for value in shape), 'Positive original H,W')
        seen.add(row['dataset_index'])
    small = min(records, key=lambda row: (row['G'], row['identity']['original_shape'][0]
                *row['identity']['original_shape'][1], row['dataset_index']))
    heavy = min(records, key=lambda row: (-row['G']*row['identity']['original_shape'][0]
                *row['identity']['original_shape'][1], row['dataset_index']))
    return [small['dataset_index'], heavy['dataset_index'], small['dataset_index']]


def validate_prelock(pre, records):
    need(type(pre) is dict and pre.get('schema') == 'U39-full-method-cross-image-resource-prelock-v1',
         'Explicit U39 prelock required')
    order = select_image_order(records)
    need(type(pre.get('image_order')) is list and pre['image_order'] == order
         and all(type(value) is int for value in pre['image_order']), 'Prelocked small/maxGarea/small order')
    fixed = dict(updates_per_arm=3, total_pilot_updates=6, image_batch_size=1,
                 query_chunk=1, branch_size=400, canvas_size=800, initialization_seed=202610173101)
    need(all(type(pre.get(key)) is int and pre[key] == value for key, value in fixed.items())
         and pre.get('arms') == ['PLAIN', 'FULL'] and pre.get('dtype') == 'float32'
         and pre.get('training_encode_saved_tensor_mode') == 'save_on_cpu_pin_memory_true',
         'Fixed FP32/B1/400/query1/three-update contract')
    need(pre.get('optimizer') == dict(name='AdamW', lr=3e-4, weight_decay=1e-4,
         betas=[.9, .999], eps=1e-8, foreach=False, fused=False)
         and pre['optimizer']['foreach'] is False and pre['optimizer']['fused'] is False,
         'Fixed AdamW recipe, no foreach/fused')
    need(all(pre.get(key) is False for key in
             ('checkpointing', 'AMP', 'VJP', 'automatic_retry', 'main_training'))
         and all(pre.get(key) is True for key in ('no_AP', 'no_DEV', 'no_TEST')),
         'No checkpoint/VJP/AMP/retry/main-training or evaluation')
    return order


def _closed_sources(c, modules, references):
    need(type(modules) is dict and type(references) is dict
         and set(modules) == set(references) == set(SOURCE_SHA256), 'Exact twelve injected source-closed modules')
    for role, digest in SOURCE_SHA256.items():
        ref = references[role]
        need(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}
             and type(ref['path']) is str and Path(ref['path']).is_absolute()
             and type(ref['bytes']) is int and ref['bytes'] >= 0 and ref['sha256'] == digest,
             'Exact reviewed source reference: '+role)
        path = Path(c.closed(ref)).resolve(strict=True)
        need(Path(modules[role].__file__).resolve(strict=True) == path,
             'Injected module file must equal closed source: '+role)
        data = path.read_bytes()
        need(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == digest,
             'Reviewed source bytes: '+role)


def _sample(torch, c, label, cuda):
    c.check()
    result = dict(label=label, RSS_bytes=int(c.psutil.Process().memory_info().rss),
                  available_RAM_bytes=int(c.psutil.virtual_memory().available))
    if cuda:
        torch.cuda.synchronize()
        result.update(CUDA_allocated_bytes=int(torch.cuda.memory_allocated()),
                      CUDA_reserved_bytes=int(torch.cuda.memory_reserved()),
                      CUDA_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                      CUDA_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    return result


def _elapsed(start):
    value = time.perf_counter()-start
    need(math.isfinite(value) and value >= 0, 'Finite nonnegative worker time')
    return value


def _failure(error):
    # Do not retain exception tracebacks that own failed CUDA-frame tensors.
    result = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        following = current.__cause__ or current.__context__
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current = following
    return result


def _partition(adapter):
    params = dict(adapter.named_parameters())
    inactive_names = adapter.backbone.inactive_parameter_names()
    need(type(inactive_names) is tuple and len(inactive_names) == len(set(inactive_names)),
         'Original declared inactive names, before any execution')
    inactive = {'backbone.'+name for name in inactive_names}
    need(inactive and inactive <= set(params) and all(value.requires_grad for value in params.values()),
         'Preserve all original parameters/requires_grad; allocate every declared inactive parameter')
    effective = set(params)-inactive
    need(effective, 'Nonempty effective parameter partition')
    record = dict(allocated=sum(value.numel() for value in params.values()),
        effective=sum(params[name].numel() for name in effective),
        inactive=sum(params[name].numel() for name in inactive),
        effective_names=sorted(effective), inactive_names=sorted(inactive),
        manifest=[dict(name=name, shape=list(value.shape), numel=value.numel(),
                       dtype=str(value.dtype), requires_grad=value.requires_grad, inactive=name in inactive)
                  for name, value in params.items()])
    return effective, inactive, record


def _bn_signature(torch, legacy, helpers):
    return {name+'.'+key: helpers._digest(value)
            for name, module in legacy.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def _update(torch, c, reader, index, adapter, optimizer, params, effective, inactive,
            helpers, step, stage, arm_record, report):
    """All per-image/query CUDA locals die before the caller samples residency."""
    item = x = encoded = e = m = native = gt = delta = rendered = logits = probability = bce = dice = loss = None
    started = time.perf_counter()
    try:
        c.check()
        item = reader.load(index)  # Unmodified source/GT/matcher/RGB validations.
        row = reader.records[index]
        G = item['G']
        need(item['identity'] == row['identity'] and item['query_ids'] == row['query_ids']
             and G == row['G'] and type(G) is int and 1 <= G <= 100,
             'Entire original matched GT/query order; never rematch/filter')
        stage.update(dataset_index=index, image_id=item['identity']['image_id'], G=G,
            query_ids=list(item['query_ids']), ann_ids=list(item['identity']['ann_ids']),
            original_shape=list(item['identity']['original_shape']),
            actual_img_shape=list(item['identity']['actual_img_shape']),
            original_image=item['identity']['image'], data_read_elapsed_seconds=_elapsed(started))
        need(tuple(item['e'].shape) == (G, 256) and tuple(item['M'].shape) == (G, 200, 200)
             and tuple(item['L'].shape) == tuple(item['targets'].shape)
             == (G,)+tuple(item['identity']['original_shape']), 'Original matched reader tensor ABI')
        # Reader-validated GT/F stay CPU; only this query's target/native-L move
        # to CUDA. Model inputs contain RGB, native e and native coarse M only.
        torch.cuda.reset_peak_memory_stats()
        x = item['rgb400'].to(device='cuda:0', dtype=torch.float32)
        need(tuple(x.shape) == (1, 3, 400, 400), 'Single RGB400 image')
        optimizer.zero_grad(set_to_none=True)
        stage['optimizer_state_entries_before'] = len(optimizer.state)
        need(len(optimizer.state) == (0 if step == 1 else len(effective)),
             'AdamW moments remain resident across successive image updates')
        torch.cuda.synchronize()
        train_started = time.perf_counter()
        saved = stage['encoder_saved_on_cpu'] = {}
        with helpers._observed_save_on_cpu(torch, saved):
            encoded = adapter.encode(x)
        need(tuple(encoded.shape) == (1, 32, 400, 400) and encoded.dtype == torch.float32
             and encoded.requires_grad and encoded.grad_fn is not None
             and bool(torch.isfinite(encoded).all()), 'One finite intact shared encoding per image update')
        chunks = []
        weighted_loss = weighted_bce = weighted_dice = 0.
        zero_identity = True if step == 1 else None
        for ordinal in range(G):
            e = item['e'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            m = item['M'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            native = item['L'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            gt = item['targets'][ordinal:ordinal+1].to(device='cuda:0', dtype=torch.float32)
            delta = adapter.predict(encoded, e, m)
            rendered = item['geometry'].render_delta(delta)[:, 0]
            logits = native+rendered
            if step == 1:
                identical = bool((delta == 0).all() and (rendered == 0).all() and torch.equal(logits, native))
                zero_identity = zero_identity and identical
                need(identical, 'Pristine zero readout preserves matched native logits exactly')
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
            bce = bce.flatten(1).mean(1)
            probability = logits.sigmoid()
            dice = 1-(2*(probability*gt).flatten(1).sum(1)+1)/(probability.flatten(1).sum(1)+gt.flatten(1).sum(1)+1)
            loss = (bce+dice).mean()/G
            need(bool(torch.isfinite(loss)), 'Finite original-grid BCE+softDice')
            weighted_loss += float(loss.detach().item())
            weighted_bce += float(bce.detach().item())/G
            weighted_dice += float(dice.detach().item())/G
            chunks.append(dict(raw_GT_ordinal=ordinal, query_id=item['query_ids'][ordinal],
                               weight=1./G, retain_graph=ordinal+1 < G))
            loss.backward(retain_graph=ordinal+1 < G)
            e = m = native = gt = delta = rendered = logits = probability = bce = dice = loss = None
            c.check()
        encoded = x = None
        helpers._require_actual_cpu_saves(saved)
        stage.update(loss=weighted_loss, BCE=weighted_bce, softDice=weighted_dice,
            chunks=chunks, image_encodes=1, all_matched_GT=G, initial_matched_zero_identity=zero_identity)
        helpers._gradient_report(torch, params, effective, inactive, stage)
        stage['optimizer_step_entered'] = True
        report['new_training'] = True
        report['optimizer_steps_entered'] += 1
        optimizer.step()
        # Count a returned real optimizer step immediately, even if a later
        # metadata/resource check fails; partial output must preserve this fact.
        arm_record['updates_completed'] += 1
        report['successful_optimizer_updates'] += 1
        stage['optimizer_step_returned'] = True
        stage['optimizer_state'] = helpers._optimizer_report(torch, optimizer, params, effective, step)
        need(all(params[name].grad is None for name in inactive), 'Inactive gradients remain None')
        torch.cuda.synchronize()
        stage['training_with_checks_elapsed_seconds'] = _elapsed(train_started)
    finally:
        # Adam/BN persist; gradients, each image, query and graph do not.
        optimizer.zero_grad(set_to_none=True)
        item = x = encoded = e = m = native = gt = delta = rendered = logits = probability = bce = dice = loss = None
        stage['image_scope_elapsed_seconds'] = _elapsed(started)


def _run_arm(torch, c, reader, adapters, order, helpers, arm, record, report):
    handles = []
    adapter = params = optimizer = legacy = before_buffers = None
    counters = dict(backbone=0, feature_exit=0, legacy_forward=0, **{name: 0 for name in HEAD_NAMES})
    record.update(status='RUNNING', updates_completed=0, updates=[], after_arm_release=None)

    def count(name):
        def hook(module, inputs, output):
            counters[name] += 1
        return hook

    try:
        # Pop INSIDE this frame: passing the active model as a Python call
        # argument can retain it on the caller's evaluation stack until return,
        # preventing a truthful in-function CUDA-zero cleanup check.
        adapter = adapters.pop(arm)
        effective, inactive, record['parameters'] = _partition(adapter)
        need(all(value.device.type == 'cpu' and value.dtype == torch.float32
                 for value in adapter.parameters()), 'Fresh initialized CPU arm before device transfer')
        adapter.to(device='cuda:0', dtype=torch.float32)
        adapter.train()
        params = dict(adapter.named_parameters())
        parameter_ids = {name: id(value) for name, value in params.items()}
        need(set(params) == effective|inactive and not torch.is_autocast_enabled(), 'Unchanged FP32 parameter partition')
        legacy = adapter.backbone.legacy
        before_buffers = helpers._head_buffers(torch, legacy)
        need(len(before_buffers) == (12 if arm == 'PLAIN' else 18), 'Original unused semantic-head BN inventory')
        record['semantic_head_BN_buffer_count'] = len(before_buffers)
        handles.extend((adapter.backbone.register_forward_hook(count('backbone')),
            legacy.CONV0_4.register_forward_hook(count('feature_exit')),
            legacy.register_forward_hook(count('legacy_forward'))))
        for name in HEAD_NAMES:
            handles.append(getattr(legacy, name).register_forward_hook(count(name)))
        optimizer = torch.optim.AdamW([params[name] for name in sorted(effective)],
            lr=3e-4, weight_decay=1e-4, betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
        record['optimizer'] = dict(name='AdamW', lr=3e-4, weight_decay=1e-4,
            betas=[.9, .999], eps=1e-8, foreach=False, fused=False,
            state_reused_across_images=True, parameter_partition='predeclared effective only')
        previous_bn = _bn_signature(torch, legacy, helpers)
        record['before_first_image'] = _sample(torch, c, 'model_resident_before_first_image', True)
        for step, index in enumerate(order, 1):
            stage = dict(status='RUNNING', step=step, dataset_index=index, optimizer_step_returned=False)
            record['updates'].append(stage)
            before_calls = dict(counters)
            current_bn = _bn_signature(torch, legacy, helpers)
            need(current_bn == previous_bn, 'BN state is carried unchanged between updates; no per-image reset')
            stage['BN_before'] = current_bn
            stage['BN_carried_from_previous_image'] = True
            _update(torch, c, reader, index, adapter, optimizer, params, effective, inactive,
                    helpers, step, stage, record, report)
            need({name: id(value) for name, value in adapter.named_parameters()} == parameter_ids,
                 'Same model parameters across all image updates')
            stage['calls'] = {name: counters[name]-before_calls[name] for name in counters}
            need(stage['calls']['backbone'] == stage['calls']['feature_exit'] == 1
                 and stage['calls']['legacy_forward'] == 0
                 and all(stage['calls'][name] == 0 for name in HEAD_NAMES),
                 'One feature encode; no legacy full-forward or old semantic heads')
            stage['semantic_head_BN_buffers'] = helpers._verify_buffers(torch, legacy, before_buffers)
            previous_bn = _bn_signature(torch, legacy, helpers)
            stage['BN_after'] = previous_bn
            gc.collect()
            # No empty_cache between images: record real persistent optimizer,
            # BN and allocator residency after per-image locals have gone away.
            stage['after_image_release'] = _sample(torch, c, 'post_update_image_graph_and_gradients_released', True)
            need(stage['after_image_release']['CUDA_peak_allocated_bytes'] <= 4096*1024**2,
                 'Actual per-stage allocated peak remains within 4GiB')
            stage['status'] = 'PASS'
        record['total_calls'] = dict(counters)
        need(record['updates_completed'] == 3 and counters['backbone'] == counters['feature_exit'] == 3,
             'Three actual sequential image updates on this original arm')
        record['status'] = 'PASS'
    except BaseException as error:
        record['status'] = 'FAILED'
        record['error'] = _failure(error)
        if record['updates'] and record['updates'][-1]['status'] == 'RUNNING':
            record['updates'][-1]['status'] = 'FAILED'
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        # Do not leave the last Parameter/model/optimizer in loop aliases.
        params = optimizer = legacy = before_buffers = adapter = None
        gc.collect()
        try:
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            record['after_arm_release'] = dict(CUDA_allocated_bytes=int(torch.cuda.memory_allocated()),
                                               CUDA_reserved_bytes=int(torch.cuda.memory_reserved()))
            need(all(value == 0 for value in record['after_arm_release'].values()), 'Complete arm CUDA release')
        except BaseException as error:
            record['status'] = 'FAILED'
            record['cleanup_error'] = _failure(error)


def probe(torch, np, c, context, modules, source_refs, pre, report=None, image_control=None):
    """Return completed/FAILED JSON metadata; never return tensors or save weights.

    Future Root caller must reject status != COMPLETED_PENDING_ROOT_REVIEW and
    preserve partial metadata. The caller owns imports, source/admission/runtime
    library gates, held terminal/FIRST, resource watcher and output serialization.
    It MUST enter with cold CUDA and defer GPU initialization to this function.
    """
    if report is None:
        report = {}
    need(type(report) is dict and not report, 'Fresh caller-owned partial report dictionary')
    started = time.perf_counter()
    report.update(schema='U39-full-method-cross-image-resource-report-pending-v1', status='STARTED',
        source_refs=deepcopy(source_refs), prelock=deepcopy(pre), arms={},
        successful_optimizer_updates=0, optimizer_steps_entered=0, new_training=False,
        pilot_only=True, formal_main_training_started=False, formal_training_budget_credit=0,
        AP=False, DEV=False, TEST=False, qualified=False, full_method_training_eligible=False,
        innovation_established=False, resource_speedup_established=False, weights_loaded=False, weights_written=False,
        no_fullFIT_native100_M_or_S0_required=True, expected_reader_population_images=741,
        expected_reader_population_GT=1661,
        resource_scope='Per-stage peaks and post-image residency; wrapper owns continuous/hard limits')
    reader = adapters = initialization = None
    try:
        need(not torch.cuda.is_initialized() and torch.get_default_dtype() == torch.float32,
             'Cold CUDA and unchanged FP32 default before actual CPU initialization')
        need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
             and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
             'Pinned runtime/two CPU threads admitted by future Root wrapper')
        need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False,
             'Future Root wrapper must explicitly set deterministic=True/benchmark=False')
        c.check()
        _closed_sources(c, modules, source_refs)
        initializer, rgb_reader, helpers = (modules[key] for key in
            ('shared_initialization', 'rgb_matched_reader', 'u36_helpers'))
        reader = rgb_reader.prepare(torch, np, c, context,
            matched_reader=modules['matched_reader'], image_geometry=modules['image_geometry'],
            source_refs={key: source_refs[key] for key in rgb_reader.SOURCE_SHA256}, image_control=image_control)
        need(len(reader.records) == 741 and sum(row['G'] for row in reader.records) == 1661,
             'Full original metadata population; only resource-proposal images are loaded')
        report['actual_reader_metadata_images'] = len(reader.records)
        report['actual_reader_metadata_GT'] = sum(row['G'] for row in reader.records)
        order = validate_prelock(pre, reader.records)
        report['image_order'] = list(order)
        report['distinct_probe_images'] = len(set(order))
        report['selected_metadata'] = [deepcopy(reader.records[index]) for index in order]
        report['input_reference_count'] = len(reader.input_references)
        adapters, initialization = initializer.build_pair(torch,
            {key: modules[key] for key in initializer.SOURCE_SHA256},
            {key: source_refs[key] for key in initializer.SOURCE_SHA256})
        need(initialization['schema'] == 'U34-common-operation-initialization-v1'
             and initialization['seed'] == initialization['adapter_seed'] == pre['initialization_seed']
             and len(initialization['common_state']) == 104
             and initialization['common_parameter_tensors'] == 68 and initialization['common_BN_buffers'] == 36,
             'Actual original common104 plus identical adapter initialization')
        report['initialization'] = initialization  # U34 returns JSON digests, no tensor snapshots.
        initialization = None
        gc.collect()
        report['CPU_pair_before_CUDA'] = _sample(torch, c, 'actual_shared_initialization_CPU_only', False)
        need(not torch.cuda.is_initialized(), 'Reader preparation and paired construction remain CPU only')
        # This is the first CUDA initialization in the future owned process.
        # Set the real 4GiB allocator cap BEFORE moving any model/input to GPU.
        torch.cuda.init()
        torch.cuda.set_device(0)
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        need(total_memory >= 4096*1024**2, 'Device must support the fixed allocator budget')
        fraction = (4096*1024**2)/float(total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        report['CUDA_allocator_limit'] = dict(device=0, limit_bytes=4096*1024**2,
            device_total_bytes=total_memory, fraction=fraction, set_before_model_or_input_transfer=True)
        need(torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0,
             'No model/input CUDA allocation before allocator-limit registration')
        for arm in ('PLAIN', 'FULL'):
            record = report['arms'][arm] = {}
            # The callee pops its own arm; this caller never obtains an active
            # model reference. The other pristine arm stays CPU, no snapshots.
            _run_arm(torch, c, reader, adapters, order, helpers, arm, record, report)
            if record['status'] != 'PASS':
                report['status'] = 'FAILED'
                report['error'] = dict(type='ArmResourceProbeFailed', arm=arm,
                    message='Partial original update evidence preserved; no retry or method-effectiveness conclusion')
                break
            c.check()
        else:
            need(report['successful_optimizer_updates'] == 6
                 and all(report['arms'][arm]['updates_completed'] == 3 for arm in ('PLAIN', 'FULL')),
                 'Exactly three real updates per arm, six disposable updates total')
            _closed_sources(c, modules, source_refs)
            report['status'] = 'COMPLETED_PENDING_ROOT_REVIEW'
    except BaseException as error:
        report['status'] = 'FAILED'
        report['error'] = _failure(error)
    finally:
        reader = adapters = initialization = None
        gc.collect()
        report['elapsed_seconds'] = _elapsed(started)
        report['CUDA_initialized'] = bool(torch.cuda.is_initialized())
        if report['CUDA_initialized']:
            try:
                torch.cuda.synchronize()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                report['final_CUDA_cleanup'] = dict(CUDA_allocated_bytes=int(torch.cuda.memory_allocated()),
                                                     CUDA_reserved_bytes=int(torch.cuda.memory_reserved()))
                need(all(value == 0 for value in report['final_CUDA_cleanup'].values()), 'Final CUDA allocation/cache zero')
            except BaseException as error:
                report['status'] = 'FAILED'
                report['final_cleanup_error'] = _failure(error)
    json.dumps(report, allow_nan=False)
    return report
