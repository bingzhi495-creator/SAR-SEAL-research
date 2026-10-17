"""UNBOUND real-image400 disposable two-update probe body; NOT EXECUTED.

Only Root's future source-closed, resource-admitted leaf may call run(). This
file does not import Torch, create an entry/binder, read a packet at import, or
write a qualification. The two original CPU contracts run before CUDA. The
PLAIN/FULL arms are sequential and start from new random backbone parameters.
This manual resource revision changes training query chunks only, from four to one.
"""

import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import types
import unittest


SOURCE_ROLES = (
    'full_legacy', 'full_dsconv', 'full_head', 'full_feature',
    'plain_legacy', 'plain_feature', 'instance_adapter',
    'image_geometry', 'geometry_test', 'plain_test',
)
HEAD_NAMES = tuple('final_super_0_'+str(i) for i in range(1, 5))
ORIGINAL_HW = (283, 503)
CONTENT_HW = (450, 800)
ADAMW = dict(lr=3e-4, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
ATOL, RTOL = 2e-5, 1e-5
RSS_SAMPLE_SECONDS = .05


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _load(c, reference, name):
    path = Path(c.closed(reference))
    raw = path.read_bytes()
    need(len(raw) == reference['bytes']
         and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Exact source closure: '+name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def _digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)),
                             sort_keys=True, separators=(',', ':')).encode())
    digest.update(value.numpy().tobytes(order='C'))
    return digest.hexdigest()


def _tensor(torch, value, shape, label, dtype=None):
    need(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
         and value.device.type == 'cpu' and not value.requires_grad,
         'Fixed CPU packet tensor identity: '+label)
    if dtype is not None:
        need(value.dtype == dtype, 'Packet dtype: '+label)
    need(bool(torch.isfinite(value).all()), 'Finite packet tensor: '+label)
    return value


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


@contextlib.contextmanager
def _measure(torch, c, record, cuda):
    """CUDA allocator high-water mark plus explicitly sampled process RSS."""
    process = c.psutil.Process()
    samples = [int(process.memory_info().rss)]
    stop = threading.Event()
    errors = []

    def poll():
        try:
            while not stop.wait(RSS_SAMPLE_SECONDS):
                samples.append(int(process.memory_info().rss))
        except BaseException as exc:
            errors.append(type(exc).__name__+': '+str(exc))

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record['CUDA_allocated_before_bytes'] = int(torch.cuda.memory_allocated())
    started = time.monotonic()
    watcher = threading.Thread(target=poll, name='U35-stage-RSS-observer', daemon=True)
    watcher.start()
    try:
        c.check()
        yield
        if cuda:
            torch.cuda.synchronize()
        c.check()
    finally:
        stop.set()
        watcher.join(timeout=1.)
        samples.append(int(process.memory_info().rss))
        record['elapsed_seconds'] = time.monotonic()-started
        record['sampled_RSS_peak_bytes'] = max(samples)
        record['RSS_samples'] = len(samples)
        record['RSS_sampling_interval_seconds'] = RSS_SAMPLE_SECONDS
        record['RSS_sampling_errors'] = errors
        record['RSS_is_sampled_not_OS_absolute_peak'] = True
        if cuda:
            record['CUDA_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            record['CUDA_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
            record['CUDA_allocated_after_bytes'] = int(torch.cuda.memory_allocated())


def _public_contracts(torch, c, modules, report):
    need(not torch.cuda.is_initialized(), 'CPU public contracts require cold CUDA')
    original_init = torch.cuda._lazy_init
    attempts = []
    started_ids, successful_ids = [], []

    def forbidden(*args, **kwargs):
        attempts.append('torch.cuda._lazy_init')
        raise RuntimeError('CUDA is forbidden during the two original public CPU contracts')

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            c.check()
            need(not torch.cuda.is_initialized(), 'Public test must remain on CPU')
            started_ids.append(test.id())
            super().startTest(test)

        def addSuccess(self, test):
            successful_ids.append(test.id())
            super().addSuccess(test)

        def stopTest(self, test):
            need(not torch.cuda.is_initialized(), 'Public test initialized CUDA')
            c.check()
            super().stopTest(test)

    torch.cuda._lazy_init = forbidden
    try:
        geometry = unittest.defaultTestLoader.loadTestsFromTestCase(
            modules['geometry_test'].ImageGeometryContract)
        plain = modules['plain_test'].make_real_plain_suite(
            torch, modules['plain_legacy'].UnetPlusPlus)
        need(geometry.countTestCases() == plain.countTestCases() == 1,
             'Exactly one original geometry4/8 and one original plain64 test')
        for name, suite in (('geometry4_8', geometry), ('plain64', plain)):
            record = report[name] = dict(status='RUNNING')
            with _measure(torch, c, record, cuda=False):
                result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2,
                                                resultclass=Result).run(suite)
            record.update(tests_run=result.testsRun, failures=len(result.failures),
                          errors=len(result.errors), skipped=len(result.skipped),
                          expected_failures=len(result.expectedFailures),
                          unexpected_successes=len(result.unexpectedSuccesses),
                          status='PASS' if result.wasSuccessful() else 'FAIL')
            need(result.wasSuccessful() and result.testsRun == 1 and not result.skipped
                 and not result.expectedFailures and not result.unexpectedSuccesses,
                 'Actual public contract failed: '+name)
        report.update(started_test_ids=started_ids, successful_test_ids=successful_ids,
                      CUDA_initialization_attempts=attempts, CUDA_initialized=False)
        need(len(started_ids) == len(successful_ids) == 2 and not attempts,
             'Both original tests must actually pass once')
    finally:
        torch.cuda._lazy_init = original_init
    gc.collect()


def _packet(torch, spec, c):
    # The body reads only the newly admitted packet. It does not follow its
    # historical input_refs, specifications, weights or qualification chains.
    # Preserve the installed Torch2.0.1 reader ABI for this exact source-closed,
    # self-produced U32 cache. No compatibility retry or alternative loader.
    packet = torch.load(str(c.closed(spec['native_fields_packet'])), map_location='cpu')
    need(type(packet) is dict and packet['schema'] == 'U32-actual-native100-FIT-fields-v1'
         and packet['dataset_index'] == 691
         and packet['actual_M_from_teacher_forward'] is True
         and packet['M_reconstructed'] is False, 'Actual U32 native100 packet required')
    identity = packet['identity']
    need(identity['image_id'] == 871 and identity['image'] == spec['image']
         and tuple(identity['original_shape']) == ORIGINAL_HW
         and tuple(identity['actual_img_shape']) == CONTENT_HW
         and tuple(identity['batch_input_shape']) == (800, 800),
         'Exact real JPEG identity, native orientation and original/content/canvas shapes')
    ids = packet['raw_GT_ids']
    need(type(ids) is list and len(ids) == 9 and len(set(ids)) == 9
         and all(type(value) is int for value in ids) and ids == list(identity['ann_ids']),
         'All nine raw GT IDs in native order')
    shapes = dict(native_F256=(1, 256, 200, 200), native_M200=(1, 100, 200, 200),
                  native_L=(100, *ORIGINAL_HW), native_e0=(1, 100, 256),
                  native_class_logits=(1, 100, 2), native_scores_by_query=(100,),
                  native_top_scores_by_export_order=(100,),
                  native_quality_by_export_order=(100,))
    for name, shape in shapes.items():
        _tensor(torch, packet[name], shape, name, torch.float32)
    gt = _tensor(torch, packet['raw_GT'], (9, *ORIGINAL_HW), 'raw_GT')
    need(bool(((gt == 0) | (gt == 1)).all())
         and bool((gt.reshape(9, -1).sum(1) > 0).all()), 'Nine nonempty binary native GT masks')
    need(packet['natural_query_ids'] == list(range(100)), 'All 100 natural query IDs retained')
    order = packet['native_export_query_ids']
    need(type(order) is list and len(order) == 100
         and all(type(q) is int for q in order) and sorted(order) == list(range(100)),
         'Original complete export query permutation')
    # native_top_scores are original class probabilities, not final S0. The
    # latter already includes native mask quality. Never replace/recompute S0.
    need(packet['category_id'] == 1, 'Single original category retained')
    return packet


def _assigned_queries(torch, packet):
    assignment = packet['assignment']
    indices = _tensor(torch, assignment['raw_GT_index_by_query'], (100,),
                      'assignment.raw_GT_index_by_query', torch.int64)
    need(bool(((indices >= -2) & (indices < 9)).all()), 'Original assignment range [-2,8]')
    annotations = assignment['raw_GT_annotation_id_by_query']
    need(type(annotations) is list and len(annotations) == 100,
         'Original query-to-raw-annotation identity list')
    queries = []
    for g, annotation in enumerate(packet['raw_GT_ids']):
        selected = (indices == g).nonzero(as_tuple=False).flatten().tolist()
        need(len(selected) == 1, 'Each of all nine GTs has exactly one original assigned query')
        q = int(selected[0])
        need(annotations[q] == annotation, 'Assigned query must retain the exact raw GT ID')
        queries.append(q)
    need(len(set(queries)) == 9, 'Distinct matched queries for all nine original GTs')
    return queries


def _head_buffers(torch, legacy):
    values = {}
    for head_name in HEAD_NAMES:
        for subname, module in getattr(legacy, head_name).named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                for name, value in module.named_buffers(recurse=False):
                    values[head_name+'.'+subname+'.'+name] = value.detach().cpu().clone()
    return values


def _verify_buffers(torch, legacy, before):
    after = _head_buffers(torch, legacy)
    need(set(after) == set(before), 'Semantic-head BN buffer names changed')
    rows = [dict(name=name, unchanged=bool(torch.equal(before[name], after[name])))
            for name in sorted(before)]
    need(all(row['unchanged'] for row in rows), 'Unused semantic-head BN buffer changed')
    return rows


def _gradient_report(torch, params, effective, inactive, record):
    rows = []
    for name, parameter in params.items():
        grad = parameter.grad
        row = dict(name=name, inactive=name in inactive, exists=grad is not None)
        if grad is not None:
            finite = bool(torch.isfinite(grad).all())
            row.update(finite=finite, exact_zero=bool((grad == 0).all()),
                       max_abs=float(grad.detach().abs().max().item()) if finite else None)
        rows.append(row)
    record['gradients'] = rows
    need(all(row['exists'] and row['finite'] for row in rows if row['name'] in effective),
         'Every predeclared effective parameter must have a finite gradient (zero allowed)')
    need(all(not row['exists'] for row in rows if row['name'] in inactive),
         'Predeclared inactive parameters must retain grad=None')
    return rows


def _optimizer_report(torch, optimizer, params, effective, expected_step):
    rows = []
    for name in sorted(effective):
        parameter = params[name]
        state = optimizer.state.get(parameter)
        need(state is not None and set(state) == {'step', 'exp_avg', 'exp_avg_sq'},
             'Actual AdamW state for every effective parameter: '+name)
        step = float(state['step'].item())
        finite = all(bool(torch.isfinite(state[key]).all()) for key in ('exp_avg', 'exp_avg_sq'))
        need(step == expected_step and finite, 'Actual finite AdamW moments/step: '+name)
        need(tuple(state['exp_avg'].shape) == tuple(parameter.shape)
             and tuple(state['exp_avg_sq'].shape) == tuple(parameter.shape), 'AdamW state shape')
        need(bool(torch.isfinite(parameter).all()), 'Finite parameter after optimizer step')
        rows.append(dict(name=name, step=step, moments_finite=finite,
                         moment_bytes=sum(state[k].numel()*state[k].element_size()
                                          for k in ('exp_avg', 'exp_avg_sq'))))
    need(len(optimizer.state) == len(effective), 'Optimizer contains only effective parameters')
    return rows


def _compare(torch, first, second, label):
    difference = (first-second).abs()
    finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
    passed = finite and bool(torch.allclose(first, second, atol=ATOL, rtol=RTOL))
    result = dict(label=label, passed=passed, atol=ATOL, rtol=RTOL,
                  max_abs_error=float(difference.max().item()),
                  max_allowed_element_error=float((ATOL+RTOL*second.abs()).max().item()))
    need(passed, 'Prelocked query grouping/permutation tolerance: '+label)
    return result


def _verify_pre(pre):
    need(pre['schema'] == 'U35-real-image400-chunk1-two-update-prelock-v1'
         and pre['arms'] == ['PLAIN', 'FULL']
         and pre['dataset_index'] == 691 and pre['image_id'] == 871
         and pre['queries'] == 100 and pre['GT'] == 9 and pre['query_chunk'] == 1
         and pre['input_size'] == 400 and pre['original_shape'] == list(ORIGINAL_HW)
         and pre['content_shape'] == list(CONTENT_HW)
         and pre['geometry_tests'] == pre['plain64_tests'] == 1
         and pre['stage1_atol'] == ATOL and pre['stage1_rtol'] == RTOL
         and pre['optimizer_updates_per_arm'] == 2
         and pre['optimizer'] == dict(name='AdamW', lr=3e-4, weight_decay=1e-4,
                                      betas=[.9, .999], eps=1e-8)
         and pre['stage1_restore_zero_before_training'] is True
         and pre['pilot_only'] is True and pre['new_training'] is True
         and pre['formal_main_training_started'] is False
         and all(pre[key] is False for key in ('AP', 'DEV', 'TEST'))
         and all(type(pre[key]) is int and pre[key] >= 0
                 for key in ('backbone_initialization_seed', 'adapter_initialization_seed')),
         'Fixed single-image400, original100, two-update disposable pilot contract')


def _arm(torch, modules, geometry, image400, packet, query_ids, arm, pre, c,
         record, report, previous_projection_identity):
    """All arm-owned CUDA tensors go out of scope before the next arm exists."""
    handles = []
    counters = dict(backbone=0, feature_exit=0, legacy_forward=0,
                    **{name:0 for name in HEAD_NAMES})
    legacy = wrapper = adapter = optimizer = None

    def count(name):
        def hook(module, inputs, value):
            counters[name] += 1
            if name == 'feature_exit':
                need(tuple(value.shape) == (1, 64, 400, 400)
                     and value.dtype == torch.float32, 'Actual CONV0_4 is U64x400x400 FP32')
        return hook

    def finish_stage(stage, before, buffers):
        delta = {key:counters[key]-before[key] for key in counters}
        stage['calls'] = delta
        need(delta['backbone'] == delta['feature_exit'] == 1 and delta['legacy_forward'] == 0
             and all(delta[name] == 0 for name in HEAD_NAMES),
             'Exactly one feature-only backbone/exit; zero original legacy-forward/semantic-head calls')
        stage['semantic_head_BN_buffers'] = _verify_buffers(torch, legacy, buffers)

    try:
        initialization = record['initialization'] = {}
        with _measure(torch, c, initialization, cuda=False):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(pre['backbone_initialization_seed'])
                if arm == 'PLAIN':
                    legacy = modules['plain_legacy'].UnetPlusPlus(num_classes=1, deep_supervision=False)
                    wrapper = modules['plain_feature'].make_plain_feature_backbone(torch, legacy, input_size=400)
                else:
                    legacy = modules['full_legacy'].UnetPlusPlus(num_classes=1, deep_supervision=False,
                        use_x2_transformer=False, input_size=400,
                        instantiate_x2_transformer=False, include_legacy_x4_transformer=False)
                    need(legacy.encoder_x2 is None and not hasattr(legacy, 'encoder_x4'),
                         'Full transformers explicitly disabled AND uninstantiated')
                    wrapper = modules['full_feature'].make_feature_backbone(torch, legacy)
                torch.random.default_generator.manual_seed(pre['adapter_initialization_seed'])
                adapter = modules['instance_adapter'].make_adapter(torch, wrapper,
                    'legacy.CONV0_4', 64, query_channels=256, hidden=32)
            initialization.update(backbone_seed=pre['backbone_initialization_seed'],
                adapter_seed=pre['adapter_initialization_seed'], pretrained_UNet_loaded=False,
                backbone_weights_cross_arm_matched=False, input_size=400,
                transformers_instantiated=False, feature_channels=64, adapter_hidden=32)
            projection_identity = {name:_digest(value) for name, value in adapter.state_dict().items()
                                   if not name.startswith('backbone.')}
            if previous_projection_identity is not None:
                need(projection_identity == previous_projection_identity,
                     'Shared adapter projections/readout have identical actual initial tensors')
            initialization['adapter_initial_tensor_sha256'] = projection_identity
            params = dict(adapter.named_parameters())
            inactive_names = wrapper.inactive_parameter_names()
            need(type(inactive_names) is tuple and len(inactive_names) == len(set(inactive_names)),
                 'Source-declared inactive names, before any execution')
            inactive = {'backbone.'+name for name in inactive_names}
            need(inactive and inactive <= set(params), 'Inactive parameter names exist exactly in adapter')
            need(all(value.requires_grad for value in params.values()),
                 'No silent freezing or initially disabled parameter outside the declared contract')
            effective = set(params)-inactive
            need(effective, 'Nonempty effective parameter set')
            record['parameters'] = dict(
                allocated=sum(value.numel() for value in params.values()),
                effective=sum(params[name].numel() for name in effective),
                inactive=sum(params[name].numel() for name in inactive),
                effective_names=sorted(effective), inactive_names=sorted(inactive),
                manifest=[dict(name=name, shape=list(value.shape), numel=value.numel(),
                    dtype=str(value.dtype), requires_grad=value.requires_grad, inactive=name in inactive)
                    for name, value in params.items()])
            need(record['parameters']['allocated'] == record['parameters']['effective']
                 + record['parameters']['inactive'], 'Complete parameter partition')
            before_buffers = _head_buffers(torch, legacy)
            need(len(before_buffers) == (12 if arm == 'PLAIN' else 18),
                 'Exactly the original semantic-head BN buffers')
            record['semantic_head_BN_buffer_count'] = len(before_buffers)
            handles.extend((wrapper.register_forward_hook(count('backbone')),
                            legacy.CONV0_4.register_forward_hook(count('feature_exit')),
                            legacy.register_forward_hook(count('legacy_forward'))))
            for name in HEAD_NAMES:
                handles.append(getattr(legacy, name).register_forward_hook(count(name)))

        setup = record['device_setup'] = {}
        with _measure(torch, c, setup, cuda=True):
            adapter.to(device='cuda:0', dtype=torch.float32)
            # Module.to normally preserves Parameter objects; assert rather
            # than accidentally optimizing CPU/stale objects on another API.
            current = dict(adapter.named_parameters())
            need(set(current) == set(params) and all(current[name] is value for name, value in params.items()),
                 'Parameter objects preserved during explicit device transfer')
            x = image400.to(device='cuda:0', dtype=torch.float32)
            e = packet['native_e0'][0].to(device='cuda:0')
            m = packet['native_M200'][0].to(device='cuda:0')
            native = packet['native_L'].to(device='cuda:0')
            targets = packet['raw_GT'].to(device='cuda:0', dtype=torch.float32)
            need(all(value.dtype == torch.float32 for value in adapter.parameters())
                 and not torch.is_autocast_enabled(), 'FP32 model and no autocast')

        adapter.eval()
        stage0 = record['stages']['stage0_zero'] = dict(status='RUNNING', mode='eval', autograd=False)
        before = dict(counters)
        with _measure(torch, c, stage0, cuda=True), torch.no_grad():
            need(bool((adapter.head[-1].weight == 0).all())
                 and bool((adapter.head[-1].bias == 0).all()), 'Actual zero final head')
            encoded = adapter.encode(x)
            outputs = []
            for start in range(0, 100, 4):
                stop = min(start+4, 100)
                delta = adapter.predict(encoded, e[start:stop], m[start:stop])
                rendered = geometry.render_delta(delta)[:, 0]
                need(bool((delta == 0).all()) and bool((rendered == 0).all()), 'Zero residual remains exactly zero')
                outputs.append((native[start:stop]+rendered).cpu())
                del delta, rendered
                c.check()
            reproduced = torch.cat(outputs, dim=0)
            need(torch.equal(reproduced, packet['native_L'])
                 and _digest(reproduced) == _digest(packet['native_L']), 'Original native L reproduced bit-exactly')
            binary = reproduced > 0
            need(torch.equal(binary, packet['native_L'] > 0), 'Original native binary masks unchanged')
            order = packet['native_export_query_ids']
            stage0.update(native_L_bitexact=True, binary_exact=True,
                natural_query_ids=list(range(100)), export_query_ids=list(order), category_id=1,
                original_S0_by_query_sha256=_digest(packet['native_scores_by_query']),
                export_S0_sha256=_digest(packet['native_scores_by_query'][order]),
                original_class_logits_sha256=_digest(packet['native_class_logits']),
                exported_binary_sha256=_digest(binary[order]), recomputed_scores=False,
                native_L_sha256=_digest(reproduced), query_chunks=25, all_queries=100)
            del outputs, reproduced, binary, encoded
            finish_stage(stage0, before, before_buffers)
            stage0['status'] = 'PASS'

        stage1 = record['stages']['stage1_nonzero_query_equivalence'] = dict(
            status='RUNNING', query_ids=query_ids[:2], mode='eval', autograd=False)
        before = dict(counters)
        saved_weight = adapter.head[-1].weight.detach().clone()
        saved_bias = adapter.head[-1].bias.detach().clone()
        try:
            with _measure(torch, c, stage1, cuda=True), torch.no_grad():
                adapter.head[-1].weight.fill_(.05)
                adapter.head[-1].bias.fill_(.01)
                encoded = adapter.encode(x)
                selected = query_ids[:2]
                grouped = adapter.predict(encoded, e[selected], m[selected])
                individual = torch.cat([adapter.predict(encoded, e[q:q+1], m[q:q+1]) for q in selected], 0)
                reverse = list(reversed(selected))
                permuted = adapter.predict(encoded, e[reverse], m[reverse]).flip(0)
                stage1['comparisons'] = [
                    _compare(torch, grouped, individual, 'delta400_group_vs_individual'),
                    _compare(torch, grouped, permuted, 'delta400_group_vs_inverse_permutation'),
                    _compare(torch, geometry.render_delta(grouped), geometry.render_delta(individual),
                             'original_render_group_vs_individual'),
                    _compare(torch, geometry.render_delta(grouped), geometry.render_delta(permuted),
                             'original_render_group_vs_inverse_permutation'),
                ]
                stage1['nonzero_delta_count'] = int((grouped != 0).sum().item())
                need(stage1['nonzero_delta_count'] > 0, 'Actual nonzero readout exercised')
                del encoded, grouped, individual, permuted
                finish_stage(stage1, before, before_buffers)
                stage1['status'] = 'PASS'
        finally:
            with torch.no_grad():
                adapter.head[-1].weight.copy_(saved_weight)
                adapter.head[-1].bias.copy_(saved_bias)
            del saved_weight, saved_bias
            stage1['zero_head_restored'] = bool((adapter.head[-1].weight == 0).all()
                                               and (adapter.head[-1].bias == 0).all())
        need(stage1['zero_head_restored'], 'Training starts from the original zero readout')

        optimizer = torch.optim.AdamW([params[name] for name in sorted(effective)], **ADAMW,
                                     foreach=False)
        need(len(optimizer.state) == 0, 'Fresh disposable AdamW, no inherited moments')
        record['optimizer'] = dict(name='AdamW', lr=3e-4, weight_decay=1e-4,
            betas=[.9, .999], eps=1e-8, foreach=False, fresh_state=True,
            loss='mean_G(BCEWithLogits_mean_pixels + 1-(2*sum(sigmoid(logit)*GT)+1)/(sum(sigmoid(logit))+sum(GT)+1))',
            original_GT_as_model_input=False, data_images=1, main_training=False)
        adapter.train()
        for step in (1, 2):
            stage = record['stages']['stage2_update'+str(step)] = dict(
                status='RUNNING', step=step, mode='train', autograd=True)
            before = dict(counters)
            discarded_bn = {name:value.detach().cpu().clone() for name, value in legacy.named_buffers()
                            if name.startswith('cbam_stage_up_') and '.bn1.' in name}
            with _measure(torch, c, stage, cuda=True):
                stage['optimizer_state_entries_before'] = len(optimizer.state)
                need(len(optimizer.state) == (0 if step == 1 else len(effective)),
                     'Second forward must include existing AdamW states in actual memory footprint')
                optimizer.zero_grad(set_to_none=True)
                encoded = adapter.encode(x)
                need(tuple(encoded.shape) == (1, 32, 400, 400)
                     and encoded.requires_grad and encoded.grad_fn is not None
                     and bool(torch.isfinite(encoded).all()), 'Actual finite shared image encoding and intact graph')
                weighted_loss = weighted_bce = weighted_dice = 0.
                chunks = []
                for first in range(0, 9, pre['query_chunk']):
                    end = min(first+pre['query_chunk'], 9)
                    selected = query_ids[first:end]
                    delta = adapter.predict(encoded, e[selected], m[selected])
                    logits = native[selected]+geometry.render_delta(delta)[:, 0]
                    gt = targets[first:end]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
                    bce = bce.flatten(1).mean(1)
                    probability = logits.sigmoid()
                    dice = 1-(2*(probability*gt).flatten(1).sum(1)+1)/(probability.flatten(1).sum(1)+gt.flatten(1).sum(1)+1)
                    fraction = (end-first)/9.
                    loss = (bce+dice).mean()*fraction
                    need(bool(torch.isfinite(loss)), 'Finite original-resolution real-assignment loss')
                    weighted_loss += float(loss.detach().item())
                    weighted_bce += float(bce.detach().mean().item())*fraction
                    weighted_dice += float(dice.detach().mean().item())*fraction
                    chunks.append(dict(raw_GT_indices=list(range(first, end)), query_ids=selected,
                                       weight=fraction, retain_graph=end < 9))
                    loss.backward(retain_graph=end < 9)
                    del delta, logits, gt, probability, bce, dice, loss
                    c.check()
                del encoded
                stage.update(loss=weighted_loss, BCE=weighted_bce, softDice=weighted_dice,
                             chunks=chunks, all_GT=9, image_encodes=1)
                _gradient_report(torch, params, effective, inactive, stage)
                stage['optimizer_step_entered'] = True
                report['new_training'] = True
                optimizer.step()
                record['updates_completed'] += 1
                report['successful_optimizer_updates'] += 1
                stage['optimizer_state'] = _optimizer_report(torch, optimizer, params, effective, step)
                stage['inactive_grads_all_None_after_step'] = all(params[name].grad is None for name in inactive)
                need(stage['inactive_grads_all_None_after_step'], 'Optimizer must not create inactive gradients')
                finish_stage(stage, before, before_buffers)
                current_buffers = dict(legacy.named_buffers())
                stage['original_discarded_bn1_buffers'] = [
                    dict(name=name, changed=not bool(torch.equal(value, current_buffers[name].detach().cpu())))
                    for name, value in sorted(discarded_bn.items())]
                stage['all_model_BN_buffers_claimed_frozen'] = False
                stage['status'] = 'PASS'
        record['total_calls'] = dict(counters)
        need(counters['backbone'] == counters['feature_exit'] == 4 and counters['legacy_forward'] == 0
             and all(counters[name] == 0 for name in HEAD_NAMES), 'Four actual single-image encodes per arm')
        return projection_identity
    finally:
        for handle in handles:
            handle.remove()
        # The caller performs GC/cache release only after this function's local
        # model, optimizer and input tensors have gone out of scope.


def run(torch, np, spec, c, pre):
    """Execute only inside a future Root-owned admitted leaf; never self-qualify."""
    _verify_pre(pre)
    c.check()
    need(not torch.cuda.is_initialized()
         and torch.get_num_threads() == torch.get_num_interop_threads() == 2
         and torch.get_default_dtype() == torch.float32,
         'Start cold with two CPU threads and FP32 default: run the two CPU contracts first')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         'Fresh absolute caller-owned output directory; no retry/overwrite')
    output.mkdir()
    log_path, report_path = output/'probe.log', output/'REPORT.json'
    report = dict(schema='U35-real-image400-chunk1-two-update-report-v1', status='RUNNING',
                  source_refs={key:spec[key] for key in SOURCE_ROLES},
                  native_fields_packet=spec['native_fields_packet'], image=spec['image'],
                  prelock=spec['prelock'], spec=c.execution_spec_ref,
                  admission=c.execution_admission_ref, public_contracts={}, arms={},
                  new_training=False, pilot_only=True, formal_main_training_started=False,
                  AP=False, DEV=False, TEST=False, qualified=False,
                  all741_qualification=False, convergence_or_effectiveness_claim=False,
                  old_UNet_weights_loaded=False, successful_optimizer_updates=0)
    aliases = ('dsconv', 'head', 'model_full', 'feature_backbone', 'U35_plain_legacy',
               'plain_feature', 'full_instance_adapter', 'image_geometry',
               'U35_GEOMETRY_PUBLIC', 'U35_PLAIN_PUBLIC')
    absent = object()
    previous = {name:sys.modules.get(name, absent) for name in aliases}
    cpu_rng = torch.get_rng_state().clone()
    failure = None
    started = time.monotonic()
    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                modules = {}
                mapping = (
                    ('full_dsconv', 'dsconv'), ('full_head', 'head'),
                    ('full_legacy', 'model_full'), ('full_feature', 'feature_backbone'),
                    ('plain_legacy', 'U35_plain_legacy'), ('plain_feature', 'plain_feature'),
                    ('instance_adapter', 'full_instance_adapter'), ('image_geometry', 'image_geometry'),
                    ('geometry_test', 'U35_GEOMETRY_PUBLIC'), ('plain_test', 'U35_PLAIN_PUBLIC'),
                )
                for role, alias in mapping:
                    modules[role] = _load(c, spec[role], alias)
                _public_contracts(torch, c, modules, report['public_contracts'])
                packet = _packet(torch, spec, c)
                report['identity'] = packet['identity']
                tensor_names = ('native_F256', 'native_M200', 'native_L', 'native_e0',
                                'native_class_logits', 'native_scores_by_query',
                                'native_top_scores_by_export_order', 'native_quality_by_export_order', 'raw_GT')
                input_digests = {key:_digest(packet[key]) for key in tensor_names}
                report['packet_tensor_sha256'] = input_digests
                query_ids = _assigned_queries(torch, packet)
                report['training_query_ids_in_raw_GT_order'] = query_ids
                report['training_raw_GT_ids'] = list(packet['raw_GT_ids'])
                from PIL import Image
                with Image.open(c.closed(spec['image'])) as image:
                    need(image.size == (503, 283) and image.format == 'JPEG', 'Exact original JPEG dimensions/type')
                    orientation = image.getexif().get(274)
                    need(orientation in (None, 1), 'No EXIF rotation is allowed or silently applied')
                    array = np.array(image.convert('RGB'), dtype=np.uint8, copy=True)
                need(array.shape == (283, 503, 3) and array.dtype == np.uint8, 'Original RGB uint8 layout')
                geometry = modules['image_geometry'].make_geometry(torch, ORIGINAL_HW, CONTENT_HW, 800, 400)
                rgb = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
                image400 = geometry.prepare_rgb(rgb)
                need(tuple(image400.shape) == (1, 3, 400, 400) and image400.dtype == torch.float32
                     and bool(torch.isfinite(image400).all()), 'Actual 800-pad to 400 RGB branch')
                report['image_preparation'] = dict(EXIF_orientation=orientation, auto_rotated=False,
                    original_shape=list(ORIGINAL_HW), content_shape=list(CONTENT_HW), canvas=[800, 800],
                    branch_shape=list(image400.shape), convention='RGB_uint8_to_float32_div255',
                    teacher_normalization_reconstructed=False, sha256=_digest(image400))
                need(spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
                     'Existing 4GiB allocator / 8GiB whole-GPU limits remain unchanged')
                torch.cuda.set_device(0)
                total_memory = int(torch.cuda.get_device_properties(0).total_memory)
                fraction = 4096*1024**2/total_memory
                need(0 < fraction <= 1, 'Actual device supports the unchanged allocator allowance')
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                report['FP32'] = dict(autocast=False, TF32=False, allocator_limit_MiB=4096,
                                      whole_GPU_limit_MiB=8192, sequential_arms=True)
                projection_identity = None
                for arm in ('PLAIN', 'FULL'):
                    c.check()
                    arm_report = report['arms'][arm] = dict(status='RUNNING', stages={}, updates_completed=0)
                    projection_identity = _arm(torch, modules, geometry, image400, packet, query_ids,
                                               arm, pre, c, arm_report, report, projection_identity)
                    arm_report['status'] = 'PASS'
                    gc.collect()
                    torch.cuda.synchronize()
                    torch._C._cuda_clearCublasWorkspaces()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    arm_report['after_arm_release'] = dict(
                        CUDA_allocated_bytes=int(torch.cuda.memory_allocated()),
                        CUDA_reserved_bytes=int(torch.cuda.memory_reserved()))
                    need(all(value == 0 for value in arm_report['after_arm_release'].values()),
                         'All arm-owned model, graph and optimizer allocations released before next arm')
                    c.check()
                need(report['successful_optimizer_updates'] == 4
                     and all(report['arms'][arm]['updates_completed'] == 2 for arm in ('PLAIN', 'FULL')),
                     'Two actual updates on each of both sequential arms')
                need(all(_digest(packet[key]) == value for key, value in input_digests.items()),
                     'All original fields, GT, query embeddings, class logits and S0 remain immutable')
                for role in SOURCE_ROLES+('native_fields_packet', 'image', 'prelock'):
                    need(c.ref(spec[role]['path']) == spec[role], 'Admitted input/source unchanged: '+role)
                report['status'] = 'COMPLETED_PENDING_ROOT_REVIEW'
                report['new_training'] = True
                print('Both actual disposable two-update arms completed; no main training/AP qualification.', flush=True)
            except BaseException as exc:
                failure = exc
                report['status'] = 'FAILED'
                report['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
            finally:
                for name, value in previous.items():
                    if value is absent:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
                torch.set_rng_state(cpu_rng)
                report['elapsed_seconds'] = time.monotonic()-started
                report['source_aliases_restored'] = True
                report['retry_attempted'] = False
                # Preserve truthful failure even if the resource controller has
                # already rejected a check; do not ask it to authorize success.
                with report_path.open('x', encoding='utf-8', newline='\n') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write('\n')
    if failure is not None:
        raise RuntimeError('U35 pilot failed; preserved '+str(report_path)+' and '+str(log_path)) from failure
    return dict(report_ref=c.ref(report_path), log_ref=c.ref(log_path),
                pilot_optimizer_updates={'PLAIN':2, 'FULL':2}, new_training=True, pilot_only=True,
                formal_main_training_started=False, AP=False, DEV=False, TEST=False,
                qualified=False, full_method_training_eligible=False)
