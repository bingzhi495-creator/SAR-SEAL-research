"""Independent bounded CPU qualification of two fixed posterior objectives.

Only four prelocked micros replay floating losses/gradients/clip/AdamW.
DEV scores and masks are independently reconstructed from common raw fields.
A practical near-zero consistency rule is not a certified GPU error bound.
This inert SOURCE must be admitted by ROOT before any numerical execution.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types

SPEC_SCHEMA = 'ROOT-query-pixel-posterior-two-objective-fullFIT24-spec-v1'
PREFIX = 'query-pixel-posterior-two-objective-fullFIT24'
QUAL_SCHEMA = 'ROOT-query-pixel-posterior-two-objective-fullFIT24-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_PIXEL_POSTERIOR_TWO_OBJECTIVE_FULLFIT24_CPU_QUALIFIED'
ARMS = ('posterior_BCE', 'bounded_BCE')
DETAILS = (0, 1, 17782, 17783)
CONTROLS = ('native_official', 'native_zero_budget',
    'P_native_posterior_budget', 'P_corrected_official', 'P_corrected_posterior_budget',
    'B_native_posterior_budget', 'B_corrected_official', 'B_corrected_posterior_budget')
PRIOR = (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
    ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
    ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase'))
NAMES = ('feature_projection.weight', 'feature_projection.bias', 'query_projection.weight', 'query_projection.bias',
    'hidden.weight', 'hidden.bias', 'final.weight', 'final.bias')
SHAPES = ((16, 256, 1, 1), (16,), (16, 256), (16,), (32, 36), (32,), (1, 32), (1,))
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
NUMERIC = dict(point_rtol=1e-4, point_atol=1e-6, gradient_rtol=1e-3, gradient_atol=1e-7,
    moment_rtol=1e-4, moment_atol=1e-8, class_rtol=1e-6, class_atol=1e-6,
    quality_rtol=1e-4, quality_atol=1e-4, CPU_logit_nearzero_envelope='1e-4+1e-4*abs(CPU_logit)',
    certified_GPU_error_bound=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect(row, values, label):
    require(isinstance(row, dict), label + ': dictionary required')
    for key, value in values.items():
        require(key in row and row[key] == value, label + ': differs at ' + key)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
            count += len(chunk)
    return dict(path=str(path), bytes=count, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and reference(row['path']) == row, 'Exact absolute pinned reference required')
    return Path(row['path']).resolve(strict=True)


def read_ref(row):
    return read(bound(row))


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Pinned Python SOURCE required')
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == row['sha256'] and len(source) == row['bytes'], 'SOURCE changed before execution')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def finite(value, label, nonnegative=False):
    require(type(value) in (float, int) and math.isfinite(value) and (not nonnegative or value >= 0),
        label + ': finite scalar required')


def close(actual, expected, label, rtol=1e-4, atol=1e-6):
    finite(actual, label)
    finite(expected, label)
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label + ': numeric tolerance exceeded')


def terminal(row, phase):
    require(row['status'] == 'ACTUAL_OWNED_TERMINAL' and row.get('phase', phase) == phase and
        row['original_identity'] is not None and type(row['returncode']) is int and row['returncode'] == 0 and
        row['root_wait_reaped'] is True and row['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        row['numeric_birth_absence']['original_birth_absent'] is True and row['abort'] is None and row['error'] is None and
        row['wait_errors'] is False and row['GPU_observation_errors'] == [], 'STOP before hashes/imports: terminal held0/reap/birth absence required')
    finite(row['whole_GPU_observed_peak_MiB'], 'Actual observed GPU peak', True)
    require(row['whole_GPU_observed_peak_MiB'] <= 8192 and row['whole_GPU_used_limit_MiB'] == 8192 and
        row['whole_leaf_VRAM_certified'] is False and row['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
        'Whole-GPU observed scope differs')


def source_gate(spec_path, spec, raw):
    source, reader, spec_ref = reference(spec['entry_source']['path']), reference(__file__), reference(spec_path)
    require(source == spec['entry_source'] and reader == spec['CPU_reader'] and
        Path(source['path']).parent == Path(reader['path']).parent and Path(source['path']).name == 'train.py',
        'Current producer/actual CPU reader bindings differ')
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read_ref(admission_ref)
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], driver_sha256=OWNER_SHA, allowed_cpus=[0, 1], cuda_visible_devices='0',
        whole_GPU_used_limit_MiB=8192, qualified_leaf_deadline_seconds=spec['deadline_seconds'],
        entry_args=['--spec', str(spec_path), '--owner-execute']), 'Current original GPU admission')
    require(Path(admission['entry']).resolve() == Path(source['path']) and raw['admission_sha256'] == admission_ref['sha256'] and
        admission.get('scientific_spec', spec_ref) == spec_ref and admission['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'],
        'Current raw/SPEC/admission join differs')
    require(len({r['path'] for r in spec['sources']}) == len(spec['sources']), 'Unique complete current closure required')
    for row in spec['sources']:
        bound(row)
        require(admission['bound_source_sha256'].get(row['path']) == row['sha256'], 'Current owner must bind every original/new closure reference')
    for role in ('entry_source', 'CPU_reader', 'context', 'SOURCE_proposal', 'posterior_readout_source',
                 'native_adapter_source', 'native_preflight_CPU_qualification', 'native_preflight_spec',
                 'native_preflight_report', 'native_preflight_entry_source', 'native_preflight_CPU_reader_source',
                 'native_preflight_receipt', 'preflight_helper_source'):
        require(spec[role] in spec['sources'], 'Required current owner role absent: ' + role)
    proposal = read_ref(spec['SOURCE_proposal'])
    expect(proposal, dict(schema=PREFIX + '-SOURCE-proposal-v1', producer_SOURCE=source, arms=list(ARMS), controls=list(CONTROLS),
        epochs=24, microsteps=17784, optimizer_updates=8892, raw_GT_visits=39864), 'Current inert SOURCE proposal')
    require(proposal['CPU_numeric_protocol'] == NUMERIC, 'Prelocked independent CPU numeric protocol differs')
    writer = load_source(source, 'sealed_two_objective_provenance_only')
    # Reuse provenance gates only. Never call producer numerical helpers/execute.
    ctx, prior_raw, repeated_spec, repeated_admission = writer.gate(spec_path, spec)
    require(repeated_spec == spec_ref and repeated_admission == admission_ref and
        ctx['FIT_annotation'] in spec['sources'] and ctx['DEV_annotation'] in spec['sources'] and
        ctx['CPU_reader'] in spec['sources'], 'Context FIT/DEV/CPU SOURCE must be current owner-bound')
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(owner['sha256'] == OWNER_SHA, 'Actual ROOT owner SOURCE differs')
    return dict(source=source, reader=reader, spec=spec_ref, GPU_admission=admission_ref, owner_driver=owner), ctx, prior_raw


def tensor_record(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def cpu_tensor(torch, value, shape, label, dtype=None):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided and
        value.dtype == (torch.float32 if dtype is None else dtype) and tuple(value.shape) == tuple(shape) and
        not value.requires_grad and bool(torch.isfinite(value).all()), label + ': finite detached CPU tensor required')


def tensor_close(torch, actual, expected, label, rtol=1e-4, atol=1e-6):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32 and
        bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()) and
        bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)), label + ': independent FP32 tolerance exceeded')
    return dict(max_absolute_error=float((actual - expected).abs().max()) if actual.numel() else 0.,
        rtol=rtol, atol=atol, bitwise_equality_claimed=False)


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(tree_equal(torch, left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def check_weights(torch, weights, label):
    require(isinstance(weights, dict) and tuple(weights) == NAMES and sum(v.numel() for v in weights.values()) == 9441,
        label + ': exact eight ordered tensors/9441 scalars required')
    for name, shape in zip(NAMES, SHAPES):
        cpu_tensor(torch, weights[name], shape, label + ' ' + name)


def cold_state(torch, seed):
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        torch.random.set_rng_state(generator.get_state())
        layers = (torch.nn.Conv2d(256, 16, 1, bias=True, device='cpu', dtype=torch.float32),
            torch.nn.Linear(256, 16, bias=True, device='cpu', dtype=torch.float32),
            torch.nn.Linear(36, 32, bias=True, device='cpu', dtype=torch.float32),
            torch.nn.Linear(32, 1, bias=True, device='cpu', dtype=torch.float32))
        state = {}
        for prefix, layer in zip(('feature_projection', 'query_projection', 'hidden', 'final'), layers):
            for name, value in layer.state_dict().items():
                state[prefix + '.' + name] = value.clone()
        state['final.weight'].zero_()
        state['final.bias'].zero_()
    check_weights(torch, state, 'Independent cold initialization')
    return state


def render_original(torch, field, identity):
    resized = torch.nn.functional.interpolate(field, size=(800, 800), mode='bilinear', align_corners=False)
    h, w = identity['actual_img_shape']
    return torch.nn.functional.interpolate(resized[:, :, :h, :w], size=tuple(identity['original_shape']),
        mode='bilinear', align_corners=False)[0]


def projected_fields(torch, packet, weights):
    pixel200 = torch.nn.functional.conv2d(packet['F'], weights['feature_projection.weight'], weights['feature_projection.bias'])
    pixel = render_original(torch, pixel200, packet['identity'])
    query = torch.nn.functional.linear(packet['e0'][0], weights['query_projection.weight'], weights['query_projection.bias'])
    native = render_original(torch, packet['M200'], packet['identity']).detach()
    probability = native.sigmoid()
    local = torch.nn.functional.avg_pool2d(probability[:, None], 3, stride=1, padding=1, count_include_pad=False)[:, 0]
    return pixel, query, native, (probability - local).detach()


def point_loss(torch, packet, weights, arm, query_ids, indices, targets):
    if not len(query_ids):
        return sum((v.reshape(-1)[:0].sum() for v in weights.values()), targets.new_zeros(())), None
    pixel, query, native, contrast = projected_fields(torch, packet, weights)
    G, K = indices.shape
    pixel_points = pixel.flatten(1).t()[indices.reshape(-1)]
    query_points = query[query_ids][:, None].expand(G, K, 16).reshape(-1, 16)
    L = native[query_ids].flatten(1).gather(1, indices).reshape(-1)
    C = contrast[query_ids].flatten(1).gather(1, indices).reshape(-1)
    direct36 = torch.cat((pixel_points, query_points, L[:, None], L.sigmoid()[:, None],
        (L > 0).to(L.dtype)[:, None], C[:, None]), 1)
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(direct36, weights['hidden.weight'], weights['hidden.bias']))
    u = torch.nn.functional.linear(hidden, weights['final.weight'], weights['final.bias']).flatten()
    logits = (L + (u if arm == ARMS[0] else u.tanh())).reshape(G, K)
    sampled_target = targets.flatten(1).gather(1, indices).to(torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, sampled_target, reduction='none').mean(1).mean()
    return loss, logits


def private_indices(torch, count, shape, micro):
    generator = torch.Generator(device='cpu').manual_seed(3800000 + micro)
    return torch.randint(shape[0] * shape[1], (count, 12544), generator=generator, dtype=torch.int64)


def digest_contract(row, shape, dtype='float32'):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == list(shape) and
        row['dtype'] == dtype and isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Exact SOURCE-observed digest required')


def gradient_record(torch, tensors, recorded, label):
    require(set(tensors) == set(recorded) == set(NAMES), label + ': all eight gradients required')
    for name in NAMES:
        cpu_tensor(torch, tensors[name], SHAPES[NAMES.index(name)], label + ' ' + name)
        require(tensor_record(tensors[name]) == recorded[name]['digest'], label + ': actual gradient digest differs')
        close(float(torch.linalg.vector_norm(tensors[name])), recorded[name]['norm'], label + ': norm', 1e-6, 1e-7)


def trace_check(torch, trace, orders, cache, state):
    require(len(trace) == 17784 and len(orders) == 24, 'Complete17784 trace/24orders required')
    visits, previous, boundaries, activity = 0, {a: None for a in ARMS}, {}, {a: {n: 0 for n in NAMES} for a in ARMS}
    generator = torch.Generator(device='cpu').manual_seed(2026101021)
    source_ids = list(cache)
    for epoch, row in enumerate(orders):
        require(row['epoch'] == epoch and row['RNG_before'] == tensor_record(generator.get_state()), 'Private order before-state differs')
        permuted = [source_ids[i] for i in torch.randperm(741, generator=generator).tolist()]
        require(row['image_ids'] == permuted and row['RNG_after'] == tensor_record(generator.get_state()), 'Private original FIT order differs')
    for g, row in enumerate(trace):
        image_id = orders[g // 741]['image_ids'][g % 741]
        target = cache[image_id]
        expect(row, dict(global_microstep=g, epoch=g // 741, offset=g % 741, image_id=image_id,
            raw_GT=len(target['ann_ids']), integer_pixel_seed=3800000 + g, matching_reused=g >= 741,
            pending_accumulation_before=g % 2, pending_accumulation_after=(g + 1) % 2), 'Complete micro trace')
        require(row['identity']['ann_ids'] == target['ann_ids'] and set(row['arms']) == set(ARMS), 'Raw GT/order/two arms differ')
        expect(row['forward'], dict(head_calls=1, backbone_calls=1, pixel_decoder_calls=1, mask_embed_calls=10,
            seed=100000 + image_id, external_RNG_restored=True, hooks_removed=True, all4_GT_fields_absent=True,
            classes_all10_detached=True, first9_masks_detached=True, native_all10_tensor_versions_unchanged=True), 'Single shared frozen GT-free forward')
        require(len(row['forward']['private_RNG_before']) == len(row['forward']['private_RNG_after']) == 2, 'Native private CPU/CUDA RNG proofs required')
        for role, shape in (('classes10', (1, 100, 2)), ('native_masks10', (1, 100, 200, 200))):
            if g in DETAILS:
                require(len(row['forward'][role]) == 10, 'All ten detailed stage digests required')
                for digest in row['forward'][role]:
                    digest_contract(digest, shape)
            else:
                require(row['forward'][role] is None, 'All-stage field hash scope is four details only')
        indices = private_indices(torch, len(target['ann_ids']), target['shape'], g)
        require(tensor_record(indices) == row['indices_digest'], 'Actual private replacement integer draws differ')
        visits += len(target['ann_ids'])
        for arm in ARMS:
            data = row['arms'][arm]
            step = (g + 1) % 2 == 0
            expect(data, dict(optimizer_update=step, updates_completed=(g + 1) // 2, learning_rate=.0001), 'Global accumulation/update/LR')
            finite(data['loss'], 'Actual image objective', True)
            require(set(data['parameter_before']) == set(data['parameter_after']) == set(data['accumulated_gradients_before_clip']) == set(NAMES),
                'Only eight standalone active tensors may occur')
            for name, shape in zip(NAMES, SHAPES):
                digest_contract(data['parameter_before'][name], shape)
                digest_contract(data['parameter_after'][name], shape)
                digest_contract(data['accumulated_gradients_before_clip'][name]['digest'], shape)
                norm = data['accumulated_gradients_before_clip'][name]['norm']
                finite(norm, 'Actual active accumulated norm', True)
                activity[arm][name] += int(norm > 0)
            require(previous[arm] is None or previous[arm] == data['parameter_before'], 'Parameter trajectory continuity differs')
            previous[arm] = data['parameter_after']
            if step:
                finite(data['clip_norm_before'], 'Actual preclip norm', True)
                require(set(data['clipped_gradients']) == set(NAMES), 'All eight clipped gradients required')
                for name, shape in zip(NAMES, SHAPES):
                    digest_contract(data['clipped_gradients'][name]['digest'], shape)
                    finite(data['clipped_gradients'][name]['norm'], 'Actual clipped norm', True)
                    require(data['clipped_gradients'][name]['norm'] <= .01001, 'Clip.01 norm exceeded')
            else:
                require(data['clip_norm_before'] is None and data['clipped_gradients'] is None and
                    data['parameter_before'] == data['parameter_after'], 'Pending micro cannot update/clip')
            before, after = data['epoch_boundary_gradient_before'], data['epoch_boundary_gradient_after']
            require((before is not None) == (g > 0 and g % 741 == 0 and g % 2 == 1) and
                (after is not None) == ((g + 1) % 741 == 0 and (g + 1) % 2 == 1), 'Odd741 pending boundary scope differs')
            if before is not None:
                require(before == boundaries[arm], 'Actual pending gradient digest changed across epoch boundary')
            if after is not None:
                boundaries[arm] = after
    require(visits == 39864 and len(state['epoch_summaries']) == 24, 'Complete GT visits/24summaries required')
    for epoch, row in enumerate(state['epoch_summaries']):
        expect(row, dict(epoch=epoch, microsteps=741,
            optimizer_updates=((epoch + 1) * 741) // 2 - (epoch * 741) // 2,
            cumulative_optimizer_updates=((epoch + 1) * 741) // 2, pending_accumulation=((epoch + 1) * 741) % 2),
            'Odd741 epoch summary')
        require(set(row['image_loss_mean']) == set(ARMS), 'Both observed epoch objective means required')
        for arm in ARMS:
            close(row['image_loss_mean'][arm], math.fsum(r['arms'][arm]['loss'] for r in trace[epoch * 741:(epoch + 1) * 741]) / 741,
                'Observed epoch mean reaggregation', 1e-12, 1e-9)
    return dict(microsteps=17784, updates_per_arm=8892, raw_GT_visits_per_arm=visits,
        independently_regenerated_integer_draws=17784, independently_regenerated_epoch_orders=24,
        pending_odd741_gradient_digest_joins=12, epoch_flush=False, observed_gradient_activity=activity,
        all17784_floating_losses_or_gradients_replayed=False)


def AdamW_audit(torch, g, weights, saved):
    before, after = saved['optimizer_state_before'], saved['optimizer_state_after']
    require(len(before['param_groups']) == len(after['param_groups']) == 8 and before['param_groups'] == after['param_groups'],
        'Eight unchanged singleton AdamW groups required')
    if g % 2 == 0:
        require(tree_equal(torch, before, after) and all(torch.equal(weights[n].detach(), saved['parameters_after'][n]) for n in NAMES),
            'Pending micro changed weights/moments')
        return dict(optimizer_update=False, moments_and_weights_exactly_preserved=True)
    proofs, keys = {}, []
    for name, group in zip(NAMES, before['param_groups']):
        expect(group, dict(lr=.0001, weight_decay=.05, betas=(.9, .999), eps=1e-8, amsgrad=False,
            maximize=False, capturable=False, differentiable=False), 'Actual singleton AdamW recipe')
        require(len(group['params']) == 1 and group['params'][0] not in keys, 'Unique singleton AdamW identities required')
        key = group['params'][0]
        keys.append(key)
        original = weights[name].detach()
        if key in before['state']:
            previous = before['state'][key]
            m, v, step = previous['exp_avg'], previous['exp_avg_sq'], float(previous['step'])
        else:
            require(g == 1, 'Only first update may have fresh empty moments')
            m, v, step = torch.zeros_like(original), torch.zeros_like(original), 0.
        require(step == (g + 1) // 2 - 1, 'Bounded before-update moment step differs')
        gradient = saved['clipped_gradients'][name]
        mean, variance = m * .9 + gradient * .1, v * .999 + gradient.square() * .001
        new_step = step + 1
        denominator = variance.sqrt() / math.sqrt(1 - .999 ** new_step) + 1e-8
        expected = original * (1 - .0001 * .05) - (.0001 / (1 - .9 ** new_step)) * mean / denominator
        actual = after['state'][key]
        require(set(actual) == {'step', 'exp_avg', 'exp_avg_sq'} and float(actual['step']) == new_step,
            'Actual AdamW moment/step population differs')
        proofs[name] = dict(first_moment=tensor_close(torch, actual['exp_avg'], mean, 'AdamW first moment', 1e-4, 1e-8),
            second_moment=tensor_close(torch, actual['exp_avg_sq'], variance, 'AdamW second moment', 1e-4, 1e-8),
            parameter_delta=tensor_close(torch, saved['parameters_after'][name] - original, expected - original, 'AdamW delta', 1e-3, 1e-7))
    require(set(after['state']) == set(keys), 'Foreign/native optimizer state forbidden')
    return dict(optimizer_update=True, independently_recomputed_actual_clipped_gradient_update=proofs,
        full_optimizer_trajectory_replayed=False)


def audit_check(torch, packet, trace_row, target, matching, cold):
    g = trace_row['global_microstep']
    expect(packet, dict(schema=PREFIX + '-audit-v1', global_microstep=g, identity=trace_row['identity'], matching=matching), 'Four-step audit identity')
    require(set(packet['arms']) == set(ARMS), 'Both detailed arm packets required')
    teacher = dict(packet['teacher'], identity=packet['identity'])
    for name, shape in (('F', (1, 256, 200, 200)), ('e0', (1, 100, 256)), ('h', (1, 100, 256)),
                        ('classes', (1, 100, 2)), ('M200', (1, 100, 200, 200))):
        cpu_tensor(torch, teacher[name], shape, 'Audit native ' + name)
    require(tensor_record(teacher['classes']) == trace_row['forward']['classes10'][-1] and
        tensor_record(teacher['M200']) == trace_row['forward']['native_masks10'][-1], 'Native final packet/ten-stage digests differ')
    cpu_tensor(torch, packet['raw_GT'], tuple(target['targets'].shape), 'Audit raw GT', torch.bool)
    require(torch.equal(packet['raw_GT'], target['targets'].to(torch.bool)), 'Audit GT differs from independent original annotations')
    expected_indices = private_indices(torch, len(target['ann_ids']), target['shape'], g)
    require(torch.equal(packet['indices'], expected_indices), 'Audit private integer positions differ')
    native_projection = tensor_close(torch, teacher['M200'], torch.einsum('bqc,bchw->bqhw', teacher['e0'], teacher['F']),
        'Audit native e0/F projection')
    results = {}
    for arm in ARMS:
        saved, row = packet['arms'][arm], trace_row['arms'][arm]
        check_weights(torch, saved['parameters_before'], 'Audit before')
        check_weights(torch, saved['parameters_after'], 'Audit after')
        require({n: tensor_record(v) for n, v in saved['parameters_before'].items()} == row['parameter_before'] and
            {n: tensor_record(v) for n, v in saved['parameters_after'].items()} == row['parameter_after'], 'Audit/trace active tensor join differs')
        if g < 2:
            require(all(torch.equal(saved['parameters_before'][n], cold[n]) for n in NAMES), 'Cold first pair weights differ')
        weights = {n: v.clone().requires_grad_(True) for n, v in saved['parameters_before'].items()}
        loss, logits = point_loss(torch, teacher, weights, arm, matching['query_by_GT'], expected_indices, packet['raw_GT'])
        close(float(loss.detach()), row['loss'], 'Independent ' + arm + ' objective')
        logits_proof = tensor_close(torch, saved['point_logits'], logits.detach(), 'Independent original-raster point logits')
        gradients = dict(zip(NAMES, torch.autograd.grad(loss / 2, tuple(weights.values()), allow_unused=False)))
        gradient_proofs = {}
        for n in NAMES:
            cpu_tensor(torch, saved['micro_gradients'][n], SHAPES[NAMES.index(n)], 'Actual micro gradient')
            gradient_proofs[n] = tensor_close(torch, gradients[n], saved['micro_gradients'][n], 'Independent micro gradient ' + n, 1e-3, 1e-7)
            pending = saved['pending_gradients_before'][n]
            if g % 2 == 0:
                require(pending is None, 'First micro of each accumulation pair must have no pending gradient')
            else:
                cpu_tensor(torch, pending, SHAPES[NAMES.index(n)], 'Actual pending gradient')
            expected_accumulated = gradients[n] if pending is None else gradients[n] + pending
            tensor_close(torch, saved['accumulated_gradients_before_clip'][n], expected_accumulated, 'Independent accumulated gradient ' + n, 1e-3, 1e-7)
            if g < 2 and not n.startswith('final.'):
                require(bool((saved['micro_gradients'][n] == 0).all()), 'Zero final layer must yield present cold zero upstream gradients')
        gradient_record(torch, saved['accumulated_gradients_before_clip'], row['accumulated_gradients_before_clip'], 'Actual accumulated')
        if g % 2:
            actual_accumulated = saved['accumulated_gradients_before_clip']
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(actual_accumulated[n]) for n in NAMES]))
            close(float(norm), row['clip_norm_before'], 'Actual independent preclip norm', 1e-6, 1e-7)
            coefficient = min(1., .01 / (float(norm) + 1e-6))
            gradient_record(torch, saved['clipped_gradients'], row['clipped_gradients'], 'Actual clipped')
            for n in NAMES:
                tensor_close(torch, saved['clipped_gradients'][n], actual_accumulated[n] * coefficient, 'Independent clip.01 ' + n, 1e-4, 1e-8)
        else:
            require(saved['clipped_gradients'] is None, 'Pending micro cannot have clipped gradients')
        results[arm] = dict(loss=float(loss.detach()), logits=logits_proof, micro_gradient=gradient_proofs,
            optimizer=AdamW_audit(torch, g, weights, saved))
        del weights, gradients, loss, logits
    return dict(global_microstep=g, native_projection=native_projection, arms=results)


def checkpoint_check(torch, module_source, spec, ctx, refs, record, trace, matching, state, final_optimizers, cold):
    checkpoints, proofs, modules = {}, {}, {}
    generator = torch.Generator(device='cpu').manual_seed(2026101021)
    for _ in range(24):
        torch.randperm(741, generator=generator)
    for arm in ARMS:
        cp_ref = record['checkpoints'][arm]
        require(Path(cp_ref['path']).name == arm + '_E24.pth', 'Only fixed readoutE24 checkpoint allowed')
        cp = torch.load(str(bound(cp_ref)), map_location='cpu')
        require(set(cp) == {'schema', 'meta', 'readout_state_dict', 'optimizer_state_dict', 'order_generator_state', 'fixed_assignment_rows'} and
            cp['schema'] == PREFIX + '-checkpoint-v1', 'Exact readout-only checkpoint schema required')
        expect(cp['meta'], dict(arm=arm, phase=spec['phase'], spec=refs['spec'], source=refs['source'],
            posterior_readout_source=spec['posterior_readout_source'], native_adapter_source=spec['native_adapter_source'],
            readout_seed=2026100911, order_seed=2026101021, point_seed_base=3800000, kappa=1.,
            epochs=24, microsteps=17784, optimizer_updates=8892, raw_GT_visits=39864, teacher_epoch=24,
            teacher_checkpoint=ctx['teacher_checkpoint'], frozen_native610=True, parameter_tensors=8, parameter_scalars=9441,
            optimizer_param_names=[[n] for n in NAMES], learning_rate=.0001, no_previous_optimizer_or_order_restoration=True,
            formal_resumable=False, fixed_endpoint_epoch=24, DEV_selection=False), 'Fixed final checkpoint metadata')
        weights, optimizer = cp['readout_state_dict'], cp['optimizer_state_dict']
        check_weights(torch, weights, 'Final checkpoint')
        require({n: tensor_record(v) for n, v in weights.items()} == trace[-1]['arms'][arm]['parameter_after'] and
            {n: tensor_record(v) for n, v in weights.items()} == state['final_readout_states'][arm] and
            {n: tensor_record(v) for n, v in cold.items()} == trace[0]['arms'][arm]['parameter_before'] and
            {n: tensor_record(v) for n, v in cold.items()} == state['initial_readout_states'][arm], 'CP/start/endpoint/full trace identity differs')
        require(tree_equal(torch, optimizer, final_optimizers[arm]) and
            len(optimizer['state']) == len(optimizer['param_groups']) == 8, 'Final actual audited optimizer must equal endpoint CP')
        keys = []
        for name, shape, group in zip(NAMES, SHAPES, optimizer['param_groups']):
            expect(group, dict(lr=.0001, weight_decay=.05, betas=(.9, .999), eps=1e-8, amsgrad=False,
                maximize=False, capturable=False, differentiable=False), 'Final singleton AdamW recipe')
            require(len(group['params']) == 1 and group['params'][0] not in keys, 'Unique singleton AdamW groups required')
            key = group['params'][0]
            keys.append(key)
            moment = optimizer['state'][key]
            require(set(moment) == {'step', 'exp_avg', 'exp_avg_sq'} and float(moment['step']) == 8892,
                'All eight actual states must reach step8892')
            for role in ('exp_avg', 'exp_avg_sq'):
                cpu_tensor(torch, moment[role], shape, 'Final ' + role)
            require(bool((moment['exp_avg_sq'] >= 0).all()), 'Finite nonnegative second moment required')
        require(set(optimizer['state']) == set(keys), 'Foreign/native optimizer state forbidden')
        cache = cp['fixed_assignment_rows']
        require(isinstance(cache, dict) and all(type(k) is str and k.isdecimal() and str(int(k)) == k and int(k) > 0 for k in cache) and
            cache == matching and torch.equal(generator.get_state(), cp['order_generator_state']),
            'Canonical string assignment map/all nested values/private final order state differ')
        module = module_source.QueryPixelPosteriorReadout(seed=2026100911, kappa=1.)
        require(tuple(dict(module.named_parameters())) == NAMES and len(module.state_dict()) == 8 and
            sum(p.numel() for p in module.parameters()) == 9441 and not list(module.buffers()), 'Strict independent CPU module population differs')
        module.load_state_dict(weights, strict=True)
        require(all(torch.equal(module.state_dict()[n], v) for n, v in weights.items()), 'Strict final readout CPU tensor identity differs')
        modules[arm] = module.eval()
        checkpoints[arm] = weights
        proofs[arm] = dict(checkpoint=cp_ref, parameter_tensors=8, parameter_scalars=9441, AdamW_states=8, AdamW_step=8892,
            changed_parameter_names=[n for n in NAMES if not torch.equal(weights[n], cold[n])],
            strict_CPU_loaded=True, teacher610_unchanged=True, eligible_for_formal_resume=False)
        del cp, module
    del modules
    return checkpoints, proofs


def decode_binary(np, masks, prediction, shape):
    require(set(prediction) == {'image_id', 'query_id', 'category_id', 'score', 'segmentation'} and
        prediction['category_id'] == 1 and type(prediction['query_id']) is int and 0 <= prediction['query_id'] < 100,
        'Exact original all100 prediction fields required')
    finite(prediction['score'], 'Actual exported score', True)
    require(prediction['score'] <= 1 and prediction['segmentation']['size'] == list(shape) and
        set(prediction['segmentation']) == {'size', 'counts'} and isinstance(prediction['segmentation']['counts'], str),
        'Original RLE/finite bounded score differs')
    binary = masks.decode(dict(size=list(shape), counts=prediction['segmentation']['counts'].encode('ascii')))
    require(binary.dtype == np.uint8 and list(binary.shape) == list(shape) and bool(np.isin(binary, [0, 1]).all()),
        'Original binary RLE decoding differs')
    return binary


def encode_prediction(np, masks, image_id, query_id, binary, score):
    rle = masks.encode(np.asfortranarray(binary.cpu().numpy().astype(np.uint8)))
    counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
    return dict(image_id=image_id, query_id=query_id, category_id=1, score=float(score),
        segmentation=dict(size=list(rle['size']), counts=counts))


def dense_block(torch, pixel_hidden, queries, native, weights, start, stop):
    L = native[start:stop]
    p = L.sigmoid()
    C = p - torch.nn.functional.avg_pool2d(p[:, None], 3, stride=1, padding=1, count_include_pad=False)[:, 0]
    W = weights['hidden.weight']
    q = torch.nn.functional.linear(queries[start:stop], W[:, 16:32])
    hidden = pixel_hidden[None] + q[:, None, None, :]
    for j, value in enumerate((L, p, (L > 0).to(L.dtype), C)):
        hidden = hidden + value[..., None] * W[:, 32 + j]
    u = torch.nn.functional.linear(torch.nn.functional.silu(hidden), weights['final.weight'], weights['final.bias'])[..., 0]
    posterior, corrected = (L + u).sigmoid(), L + u.tanh()
    require(bool(torch.isfinite(posterior).all()) and bool(torch.isfinite(corrected).all()), 'Independent dense outputs must be finite')
    return posterior, corrected


def quality(torch, logits, posterior, support, budget):
    binary = support.to(dtype=torch.float32)
    area = binary.sum()
    if budget:
        return (binary * posterior).sum() / (area + ((1 - binary) * posterior).sum() + 1e-6)
    return (binary * logits.sigmoid()).sum() / (area + 1e-6)


def best_geometry(torch, binary, target, query, current):
    for index, gt in enumerate(target):
        area, predicted = int(gt.sum()), int(binary.sum())
        intersection = int((binary & gt).sum())
        union = predicted + area - intersection
        iou = intersection / union if union else 1.
        candidate = dict(best_query_id=query, max_mask_IoU=iou, TP=intersection, FP=predicted - intersection,
            FN=area - intersection, GT_pixels=area, prediction_pixels=predicted, empty_GT_raster=area == 0)
        if current[index] is None or iou > current[index]['max_mask_IoU'] or (
                iou == current[index]['max_mask_IoU'] and query < current[index]['best_query_id']):
            current[index] = candidate


def DEV_replay(torch, np, masks, coco, COCOeval, official, ctx_reader, ctx, spec, record, weights, DEV_cache, started):
    ids, data = list(DEV_cache), {}
    require(len(record['DEV_fields']) == 187 and len({r['path'] for r in record['DEV_fields']}) == 187 and
        set(record['mode_rows']) == set(CONTROLS), 'Exactly187 shared packets/eight controls required')
    for control in CONTROLS:
        mode = record['mode_rows'][control]
        require(set(mode) == {'predictions', 'score_factors', 'image_rows', 'checkpoint'}, 'Exact inference control artifact fields required')
        expected_CP = record['checkpoints'][ARMS[0] if control.startswith('P_') else ARMS[1]] if control[:2] in ('P_', 'B_') else ctx['teacher_checkpoint']
        require(mode['checkpoint'] == expected_CP, 'Inference control must bind its own final weights')
        predictions, factors, rows = [read_ref(mode[k]) for k in ('predictions', 'score_factors', 'image_rows')]
        require(len(predictions) == len(factors) == 18700 and len(rows) == 187 and [r['image_id'] for r in rows] == ids,
            'Complete original DEV order/all100 fields required')
        factor_map = {(r['image_id'], r['query_id']): r for r in factors}
        prediction_map = {(r['image_id'], r['query_id']): r for r in predictions}
        require(len(factor_map) == len(prediction_map) == 18700, 'Each image/query exactly once required')
        for prediction, factor in zip(predictions, factors):
            require(set(factor) == {'image_id', 'query_id', 'class_probability', 'mask_quality', 'official_score'} and
                (prediction['image_id'], prediction['query_id']) == (factor['image_id'], factor['query_id']),
                'Prediction/factor actual native row order differs')
            for key in ('class_probability', 'mask_quality', 'official_score'):
                finite(factor[key], 'Exported factor', True)
                require(factor[key] <= 1, 'Exported factor outside0..1')
            product = float(torch.tensor(factor['class_probability'], dtype=torch.float32) *
                torch.tensor(factor['mask_quality'], dtype=torch.float32))
            require(product == factor['official_score'] == prediction['score'], 'Exact exported FP32 factor-product arithmetic differs')
        data[control] = dict(predictions=predictions, factors=factor_map, by_query=prediction_map, rows=rows,
            CPU_predictions=[], binary_flips=0, changed_queries=0, max_flipped_logit=0., max_quality_error=0., max_score_error=0.)
    geometry, packet_proofs = {key: [] for key in ('native', 'P_bounded', 'B_bounded')}, []
    with torch.no_grad():
        for index, packet_ref in enumerate(record['DEV_fields']):
            require(time.monotonic() - started < spec['deadline_seconds'], 'CPU numerical replay deadline exceeded')
            packet = torch.load(str(bound(packet_ref)), map_location='cpu')
            image_id, target = ids[index], DEV_cache[ids[index]]
            require(set(packet) == {'schema', 'image_id', 'identity', 'F', 'e0', 'M200', 'classes'}, 'Exact shared native-field packet keys required')
            expect(packet, dict(schema=PREFIX + '-DEV-fields-v1', image_id=image_id, identity=target['identity']), 'Common native-field packet identity')
            for name, shape in (('F', (1, 256, 200, 200)), ('e0', (1, 100, 256)),
                                ('M200', (1, 100, 200, 200)), ('classes', (1, 100, 2))):
                cpu_tensor(torch, packet[name], shape, 'DEV common raw ' + name)
            projection = tensor_close(torch, packet['M200'], torch.einsum('bqc,bchw->bqhw', packet['e0'], packet['F']),
                'DEV native e0/F projection')
            classes, class_packet = packet['classes'][0], data['native_official']['rows'][index]['native_class_packet']
            require(torch.equal(torch.tensor(class_packet['values'], dtype=torch.float32), classes) and
                {k: class_packet[k] for k in ('shape', 'dtype', 'sha256')} == tensor_record(classes),
                'Actual full100 class values/digest differ')
            class_probability = classes.softmax(-1)[:, 0]
            query_ids = data['native_official']['rows'][index]['query_ids']
            require(len(query_ids) == 100 and all(type(q) is int for q in query_ids) and sorted(query_ids) == list(range(100)),
                'All100 original query ordinals required')
            expected_forward = data['native_official']['rows'][index]['forward']
            expect(expected_forward, dict(head_calls=1, backbone_calls=1, pixel_decoder_calls=1, mask_embed_calls=10,
                seed=100000 + image_id, external_RNG_restored=True, hooks_removed=True, all4_GT_fields_absent=True,
                classes_all10_detached=True, first9_masks_detached=True, native_all10_tensor_versions_unchanged=True), 'DEV GT-free single frozen forward')
            require(expected_forward['classes10'] is None and expected_forward['native_masks10'] is None,
                'DEV full-stage hash proof is not exported')
            for control in CONTROLS:
                row = data[control]['rows'][index]
                geometry_kind = 'native' if control in ('native_official', 'native_zero_budget') or '_native_' in control else 'bounded'
                budget_control = control not in ('native_official', 'P_corrected_official', 'B_corrected_official')
                expect(row, dict(control=control, image_id=image_id, identity=packet['identity'], query_ids=query_ids,
                    teacher_fields=packet_ref, native_class_packet=class_packet, forward=expected_forward, geometry_kind=geometry_kind,
                    quality_kind='budget' if budget_control else 'official', base_official_mask_query_score_TorchEqual=True,
                    exported_score_replaced_for_budget=budget_control, factor_official_score_key_is_exported_score=True), 'Eight-control common native identities')
                require([p['query_id'] for p in data[control]['predictions'][index * 100:(index + 1) * 100]] == query_ids,
                    'Actual export query row order differs')
            native = render_original(torch, packet['M200'], packet['identity'])
            model_fields = {}
            for arm in ARMS:
                pixel200 = torch.nn.functional.conv2d(packet['F'], weights[arm]['feature_projection.weight'], weights[arm]['feature_projection.bias'])
                pixel = render_original(torch, pixel200, packet['identity']).permute(1, 2, 0)
                query = torch.nn.functional.linear(packet['e0'][0], weights[arm]['query_projection.weight'], weights[arm]['query_projection.bias'])
                pixel_hidden = torch.nn.functional.linear(pixel, weights[arm]['hidden.weight'][:, :16], weights[arm]['hidden.bias'])
                model_fields[arm] = (pixel_hidden, query)
                del pixel200, pixel, query, pixel_hidden
            CPU_by_query = {control: {} for control in CONTROLS}
            best = {key: [None] * len(target['ann_ids']) for key in geometry}
            gt_binary = target['targets'].to(torch.bool)
            for start in range(0, 100, 4):
                stop = min(start + 4, 100)
                L, p = native[start:stop], native[start:stop].sigmoid()
                cases = {'native_official': (L, p, False), 'native_zero_budget': (L, p, True)}
                for arm, label in zip(ARMS, ('P', 'B')):
                    learned, corrected = dense_block(torch, *model_fields[arm], native, weights[arm], start, stop)
                    cases[label + '_native_posterior_budget'] = (L, learned, True)
                    cases[label + '_corrected_official'] = (corrected, learned, False)
                    cases[label + '_corrected_posterior_budget'] = (corrected, learned, True)
                for local, query_id in enumerate(range(start, stop)):
                    native_export = data['native_official']['by_query'][(image_id, query_id)]
                    native_binary_np = decode_binary(np, masks, native_export, target['shape'])
                    for control, (logits, posterior, budget_control) in cases.items():
                        actual = data[control]['by_query'][(image_id, query_id)]
                        GPU_binary_np = decode_binary(np, masks, actual, target['shape'])
                        GPU_binary = torch.from_numpy(np.ascontiguousarray(GPU_binary_np)).to(torch.bool)
                        CPU_logit, CPU_posterior = logits[local], posterior[local]
                        CPU_binary = CPU_logit > 0
                        flip = CPU_binary != GPU_binary
                        count = int(flip.sum())
                        if count:
                            envelope = 1e-4 + 1e-4 * CPU_logit.abs()
                            require(bool((CPU_logit.abs()[flip] <= envelope[flip]).all()),
                                'GPU RLE/CPU binary disagreement outside practical near-zero envelope: ' + control)
                            data[control]['max_flipped_logit'] = max(data[control]['max_flipped_logit'], float(CPU_logit.abs()[flip].max()))
                        data[control]['binary_flips'] += count
                        data[control]['changed_queries'] += int(count > 0)
                        if control in ('native_zero_budget', 'P_native_posterior_budget', 'B_native_posterior_budget'):
                            require(np.array_equal(GPU_binary_np, native_binary_np), 'Native-geometry controls must exactly retain exported native masks')
                        if control.endswith('corrected_posterior_budget'):
                            same_geometry = control.replace('_posterior_budget', '_official')
                            require(actual['segmentation'] == data[same_geometry]['by_query'][(image_id, query_id)]['segmentation'],
                                'Corrected official/budget controls must share exact exported RLE')
                        factor = data[control]['factors'][(image_id, query_id)]
                        close(float(class_probability[query_id]), factor['class_probability'], 'Independent native class softmax', 1e-6, 1e-6)
                        # Compare source scalar using exported support, then separately
                        # compute CPU score on independently thresholded CPU support.
                        on_exported_support = quality(torch, CPU_logit, CPU_posterior, GPU_binary, budget_control)
                        close(float(on_exported_support), factor['mask_quality'], 'Independently computed exported-support quality/budget', 1e-4, 1e-4)
                        CPU_quality = quality(torch, CPU_logit, CPU_posterior, CPU_binary, budget_control)
                        CPU_score = class_probability[query_id] * CPU_quality
                        require(bool(torch.isfinite(CPU_score)) and 0 <= float(CPU_score) <= 1, 'Independent finite CPU score required')
                        data[control]['max_quality_error'] = max(data[control]['max_quality_error'], abs(float(on_exported_support) - factor['mask_quality']))
                        data[control]['max_score_error'] = max(data[control]['max_score_error'], abs(float(CPU_score) - actual['score']))
                        CPU_by_query[control][query_id] = encode_prediction(np, masks, image_id, query_id, CPU_binary, CPU_score)
                        geometry_key = 'native' if control == 'native_official' else 'P_bounded' if control == 'P_corrected_official' else 'B_bounded' if control == 'B_corrected_official' else None
                        if geometry_key is not None:
                            best_geometry(torch, CPU_binary, gt_binary, query_id, best[geometry_key])
                        del GPU_binary_np, GPU_binary, CPU_logit, CPU_posterior, CPU_binary, flip, on_exported_support, CPU_quality, CPU_score
                    del native_binary_np
                del cases, L, p, learned, corrected, logits, posterior
            for control in CONTROLS:
                data[control]['CPU_predictions'].extend(CPU_by_query[control][q] for q in query_ids)
            for key in geometry:
                for annotation_id, row in zip(target['ann_ids'], best[key]):
                    ann = coco.anns[annotation_id]
                    geometry[key].append(dict(row, image_id=image_id, annotation_id=annotation_id,
                        iscrowd=int(ann.get('iscrowd', 0)), ignore=int(ann.get('ignore', 0))))
            packet_proofs.append(dict(image_id=image_id, packet=packet_ref, native_projection=projection, original_query_ids=query_ids))
            del packet, classes, class_packet, class_probability, native, model_fields, CPU_by_query, best, gt_binary
            if (index + 1) % 25 == 0 or index == 186:
                print(json.dumps(dict(CPU_reconstructed_images=index + 1, controls=8,
                    binary_disagreements={c: data[c]['binary_flips'] for c in CONTROLS})), flush=True)
            gc.collect()
    results = {}
    for control in CONTROLS:
        item = data[control]
        require(len(item['CPU_predictions']) == len(item['predictions']) == 18700, 'Exactly all18700 independently recomputed predictions required')
        GPU_metrics = official.coco_metrics(coco, item['predictions'], ids, COCOeval)
        CPU_metrics = official.coco_metrics(coco, item['CPU_predictions'], ids, COCOeval)
        results[control] = dict(checkpoint=record['mode_rows'][control]['checkpoint'], predictions=record['mode_rows'][control]['predictions'],
            score_factors=record['mode_rows'][control]['score_factors'], image_rows=record['mode_rows'][control]['image_rows'],
            queries=18700, GPU_export_COCO12=GPU_metrics, CPU_reconstructed_COCO12=CPU_metrics,
            COCO12_exact_dictionary_agreement=GPU_metrics == CPU_metrics,
            CPU_prediction_digest=hashlib.sha256(json.dumps(item['CPU_predictions'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest(),
            threshold_disagreements=item['binary_flips'], queries_with_disagreement=item['changed_queries'],
            maximum_abs_CPU_logit_at_disagreement=item['max_flipped_logit'], all_disagreements_within_practical_CPU_nearzero_envelope=True,
            max_abs_exported_support_quality_error=item['max_quality_error'], max_abs_CPU_vs_GPU_score_error=item['max_score_error'],
            CPU_masks_and_scores_independently_recomputed=True, GPU_original_logits_saved=False,
            measured_CPU_GPU_logit_allclose=False, certified_GPU_numeric_error_bound=False)
        del item['CPU_predictions'], item['predictions'], item['by_query'], item['factors']
    summaries = {}
    for key, rows in geometry.items():
        require(len(rows) == 380 and {r['annotation_id'] for r in rows} == set(coco.anns), 'All380 original raw GT retained in geometry diagnostics')
        summaries[key] = dict(raw_GT=380, GT_best100_macro_IoU=math.fsum(r['max_mask_IoU'] for r in rows) / 380,
            summed_TP=sum(r['TP'] for r in rows), summed_FP=sum(r['FP'] for r in rows), summed_FN=sum(r['FN'] for r in rows),
            best100_ties='lowest_actual_query_ordinal', one_to_one_assignment=False, rows=rows)
    return dict(controls=results, geometry=summaries, common_native_packets=packet_proofs, numeric_protocol=NUMERIC,
        all187_DEV380_raw_GT_and_all100_queries_retained=True, CPU_GT_used_only_for_metrics=True,
        calibrated_IoU_claimed=False, GPU_forward_replayed=False)


def run(spec_file, acceptance_file):
    started = time.monotonic()
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and target.parent.is_dir() and not target.exists(),
        'Absolute original SPEC/fresh CPU acceptance in existing parent required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)  # Execution metadata only; all raw terminals precede SHA/import.
    expect(spec, dict(schema=SPEC_SCHEMA, arms=list(ARMS), controls=list(CONTROLS), epochs=24, microsteps=17784,
        optimizer_updates=8892, raw_GT_visits=39864, accumulation=2, readout_seed=2026100911, order_seed=2026101021,
        point_seed_base=3800000, kappa=1., CPU_affinity=[0, 1], allocator_limit_MiB=4096,
        whole_GPU_owner_limit_MiB=8192, no_feature_cache=True), 'Fixed two-objective SPEC')
    require(type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and Path(spec['phase']).name == spec['phase'],
        'Safe current phase/positive deadline required')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt_path)
    terminal(raw, spec['phase'])
    ctx_metadata = read(spec['context']['path'])
    for receipt, phase in PRIOR:
        terminal(read(ctx_metadata[receipt]['path']), ctx_metadata[phase])
    terminal(read(spec['native_preflight_receipt']['path']), spec['native_preflight_phase'])
    # CURRENT + six original + native preflight: all eight actual terminals
    # precede hashes, imports, scientific qualification/CP/GT/export access.
    refs, ctx, prior_raw = source_gate(spec_path, spec, raw)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and output.is_dir(), 'Actual original GPU output directory required')
    report_ref, start_ref, aggregate_ref = [reference(output / n) for n in ('REPORT.json', 'START.json', 'QUALIFIED.json')]
    report, start, aggregate = read_ref(report_ref), read_ref(start_ref), read_ref(aggregate_ref)
    expect(report, dict(schema=PREFIX + '-report-v1', status='QUERY_PIXEL_POSTERIOR_TWO_OBJECTIVE_FULLFIT24_GPU_COMPLETE_PENDING_CPU',
        phase=spec['phase'], source=refs['source'], reader=refs['reader'], spec=refs['spec'], context=spec['context'],
        GPU_admission=refs['GPU_admission'], native_preflight_CPU_qualification=spec['native_preflight_CPU_qualification'],
        arms=list(ARMS), controls=list(CONTROLS), epochs=24, microsteps=17784, optimizer_updates=8892,
        raw_GT_visits=39864, accumulation=2, epoch_flush=False, readout_seed=2026100911, order_seed=2026101021,
        point_seed_base=3800000, kappa=1., learning_rate=.0001, LR_schedule=False,
        all_native610_frozen=True, shared_teacher_forward=True, feature_cache_used=False, fixed_endpoint_epoch=24,
        DEV_historically_exposed=True, DEV_selection=False, no_TEST_access=True, AP_computed=False,
        GPU_optimizer_bitwise_determinism_claimed=False, full_training_floating_CPU_replay_claimed=False,
        calibrated_IoU_claimed=False, efficacy_claimed=False, novelty_claimed=False, convergence_claimed=False,
        compute_or_native_objective_matched_claimed=False, error=None), 'Completed original GPU report')
    expect(start, dict(schema=PREFIX + '-start-v1', phase=spec['phase'], spec=refs['spec'], context=spec['context'],
        source=refs['source'], reader=refs['reader'], arms=list(ARMS), controls=list(CONTROLS),
        readout_seed=2026100911, order_seed=2026101021, point_seed_base=3800000, epochs=24, microsteps=17784,
        updates=8892, accumulation=2, epoch_flush=False, learning_rate=.0001, LR_schedule=False, kappa=1.,
        original_seven_raw_gates_before_any_hash_or_import=True, fresh_optimizer_state_entries={a: 0 for a in ARMS},
        previous_readout_checkpoint_loaded=False, fixed_endpoint_epoch=24, DEV_selection=False), 'Actual START')
    record = report['result']
    expect(aggregate, dict(schema=PREFIX + '-aggregate-v1', status='QUERY_PIXEL_POSTERIOR_TWO_OBJECTIVE_FULLFIT24_PENDING_CPU_READER',
        phase=spec['phase'], source=refs['source'], spec=refs['spec'], report=report_ref, result=record, independently_CPU_qualified=False),
        'Pending aggregate/report joins')
    require(set(record) == {'mode_rows', 'checkpoints', 'trace', 'order', 'matching', 'state_proof', 'audits', 'DEV_fields'} and
        set(record['checkpoints']) == set(ARMS) and set(record['audits']) == {str(g) for g in DETAILS}, 'Exact eight result artifacts/four details required')
    finite(report['elapsed_seconds'], 'Actual GPU elapsed', True)
    require(report['elapsed_seconds'] < spec['deadline_seconds'] and report['cleanup']['allocated_after'] == 0 and
        type(report['torch_peak_allocated_bytes']) is int and 0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2,
        'Actual allocator/deadline/cleanup proof differs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import torch
    import numpy as np
    import cv2
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from scipy.optimize import linear_sum_assignment
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized() and psutil.Process().cpu_affinity() == [0, 1], 'CPU01/CUDAempty/noCUDA required')
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    rng_before = torch.get_rng_state().clone()
    ctx_reader = load_source(ctx['CPU_reader'], 'sealed_context_CPU_helpers')
    pre = load_source(ctx['preflight_CPU_reader'], 'sealed_teacher_lineage_CPU_helpers')
    old = load_source(ctx['reused_CPU_reader'], 'sealed_original_state_CPU_helpers')
    teacher_reader = load_source(ctx['training_CPU_reader'], 'sealed_nativeE24_CPU_helpers')
    initializer = load_source(ctx['initializer_source'], 'sealed_original_native_initializer')
    official = load_source(ctx['official_evaluation_CPU_reader'], 'sealed_official_COCO_helpers')
    diag = load_source(ctx['prior_diag_CPU_reader_source'], 'sealed_library_SOURCE_helpers')
    readout_module = load_source(spec['posterior_readout_source'], 'sealed_readout_CPU_shape_only')
    training, teacher_q = pre.teacher_lineage(ctx, prior_raw['teacher_receipt'])
    model, teacher = pre.load_teacher(torch, old, teacher_reader, initializer, MODELS, revert_sync_batchnorm, Config, ctx, training, teacher_q)
    model.requires_grad_(False).eval()
    ctx_reader.imported_libraries(torch, model, ctx)
    diag.library_paths(torch, model, ctx)
    require(torch.__version__ == '2.0.1+cu118', 'Qualified Torch version required')
    for obj, role in ((torch.version, 'torch_version_source'), (torch.nn.functional.interpolate, 'interpolation_SOURCE'),
                      (type(model.panoptic_head), 'native_mask_head_SOURCE'), (type(model.panoptic_fusion_head), 'native_fusion_SOURCE')):
        require(Path(inspect.getsourcefile(obj)).resolve() == bound(ctx[role]).resolve(), 'Actual imported SOURCE differs: ' + role)
    state = read_ref(record['state_proof'])
    native, BN_modules, BN_affine, BN_buffers = pre.typed_native(torch, model)
    require(len(teacher) == len(model.state_dict()) == 610 and len(native) == 451 and len(BN_modules) == 53 and
        len(BN_affine) == 106 and len(BN_buffers) == 159 and state['native_parameter_names'] == native and
        state['BN_module_names'] == BN_modules and state['BN_parameter_names'] == BN_affine and state['BN_buffer_names'] == BN_buffers,
        'Independent full native610/451/typed BN population differs')
    expect(state, dict(schema=PREFIX + '-state-proof-v1', full_native610_CPU_exact=True, all_native451_frozen=True,
        native_parameter_grads_None=True, all_modules_eval=True, frozen_BN_all_exact=True, hook_cache_clear=True,
        external_RNG_restored=True, GT_free_forward=True, fresh_optimizer_state_entries={a: 0 for a in ARMS},
        completed_epochs=24, microsteps=17784, optimizer_updates={a: 8892 for a in ARMS},
        training_raw_GT_visits=39864, teacher_forward_calls=17971, audit_microsteps=list(DETAILS),
        audit_refs=record['audits'], no_pending_accumulation_gradients=True, previous_readout_checkpoint_loaded=False,
        no_teacher_optimizer_or_order_restoration=True, fixed_endpoint_epoch=24, DEV_selection=False,
        LR_schedule=False, feature_cache_used=False), 'Frozen native/fresh readout endpoint proof')
    for key in ('before', 'after', 'native_teacher610'):
        old.check_state(torch, teacher, state[key], 'Actual immutable native teacher ' + key)
    old.check_state(torch, teacher, state['teacher_loading']['trained_fullstate'], 'Actual full610 strict teacher loading')
    expect(state['teacher_loading'], dict(mode='native0', checkpoint=ctx['teacher_checkpoint'], qualification=ctx['teacher_qualification'],
        CP_qualified=True, strict_fullstate_loaded=True, native_state_tensors=610, adapter_state_tensors=0,
        registered_parameters=451, original_frozen_BN_exact=True, inference_backbone_with_cp=False, no_random_missing_keys=True),
        'Actual qualified complete teacher loader')
    cold = cold_state(torch, 2026100911)
    for arm in ARMS:
        old.check_state(torch, cold, state['initial_readout_states'][arm], 'Actual fresh seed9441 ' + arm)
    contract = read_ref(ctx['SOURCE_contract'])
    protocol = dict(contract, order_seed=2026101021)
    FIT_coco, DEV_coco = COCO(str(bound(ctx['FIT_annotation']))), COCO(str(bound(ctx['DEV_annotation'])))
    FIT_cache, _ = ctx_reader.raw_population(torch, np, FIT_coco, read_ref(ctx['FIT_annotation']), protocol, 'train')
    DEV_cache, _ = ctx_reader.raw_population(torch, np, DEV_coco, read_ref(ctx['DEV_annotation']), protocol, 'evaluation')
    require(not set(FIT_cache).intersection(DEV_cache) and len(FIT_cache) == 741 and len(DEV_cache) == 187 and
        Path(ctx['image_root']).resolve() == Path('E:/PSeg_SSDD/voc_style/JPEGImages_train').resolve(), 'Canonical disjoint original FIT/DEV populations required')
    datasets = {partition: CocoDataset(ann_file=str(bound(ctx[role])), data_prefix=dict(img=ctx['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
        for partition, role in (('FIT', 'FIT_annotation'), ('DEV', 'DEV_annotation'))}
    trace, orders, matching = read_ref(record['trace']), read_ref(record['order']), read_ref(record['matching'])
    require(set(matching) == {str(i) for i in FIT_cache}, 'All741 canonical string matching keys required')
    matches = ctx_reader.matching_check(torch, np, old, matching, FIT_cache, linear_sum_assignment)
    for row in trace:
        ctx_reader.identity_check(torch, np, cv2, FIT_coco, datasets['FIT'], ctx, FIT_cache[row['image_id']], row['identity'])
    for row in read_ref(record['mode_rows']['native_official']['image_rows']):
        ctx_reader.identity_check(torch, np, cv2, DEV_coco, datasets['DEV'], ctx, DEV_cache[row['image_id']], row['identity'])
    trace_proof = trace_check(torch, trace, orders, FIT_cache, state)
    audits, previous, final_optimizers = [], None, {}
    for g in DETAILS:
        packet = torch.load(str(bound(record['audits'][str(g)])), map_location='cpu')
        if g % 2:
            require(previous is not None and previous['global_microstep'] == g - 1, 'Audited pair must be consecutive')
            for arm in ARMS:
                saved, earlier = packet['arms'][arm], previous['arms'][arm]
                require(tree_equal(torch, saved['pending_gradients_before'], earlier['accumulated_gradients_before_clip']) and
                    tree_equal(torch, saved['parameters_before'], earlier['parameters_after']) and
                    tree_equal(torch, saved['optimizer_state_before'], earlier['optimizer_state_after']), 'Actual audited accumulation pair continuity differs')
        audits.append(audit_check(torch, packet, trace[g], FIT_cache[trace[g]['image_id']], matching[str(trace[g]['image_id'])], cold))
        if g == DETAILS[-1]:
            final_optimizers = {a: packet['arms'][a]['optimizer_state_after'] for a in ARMS}
        previous = packet if g % 2 == 0 else None
        del packet
        gc.collect()
    weights, checkpoint_proofs = checkpoint_check(torch, readout_module, spec, ctx, refs, record, trace, matching, state, final_optimizers, cold)
    endpoint = DEV_replay(torch, np, masks, DEV_coco, COCOeval, official, ctx_reader, ctx, spec, record, weights, DEV_cache, started)
    require(all(torch.equal(model.state_dict()[n], value) for n, value in teacher.items()) and
        torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES'] == '' and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1] and
        time.monotonic() - started < spec['deadline_seconds'], 'CPU native state/external RNG/resources/deadline changed')
    for row in spec['sources']:
        bound(row)
    for row in (refs['spec'], report_ref, start_ref, aggregate_ref, reference(receipt_path)):
        bound(row)
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], source=refs['source'], reader_source=refs['reader'],
        spec=refs['spec'], report=report_ref, start=start_ref, GPU_qualification=aggregate_ref, parent_receipt=reference(receipt_path),
        GPU_admission=refs['GPU_admission'], owner_driver=refs['owner_driver'], context=spec['context'],
        SOURCE_proposal=spec['SOURCE_proposal'], native_preflight_CPU_qualification=spec['native_preflight_CPU_qualification'],
        posterior_readout_source=spec['posterior_readout_source'], native_adapter_source=spec['native_adapter_source'],
        teacher_checkpoint=ctx['teacher_checkpoint'], teacher_qualification=ctx['teacher_qualification'], arms=list(ARMS), controls=list(CONTROLS),
        completed_epochs=24, microsteps=17784, optimizer_updates=8892, raw_GT_visits=39864, fixed_endpoint_epoch=24,
        readout_seed=2026100911, order_seed=2026101021, point_seed_base=3800000, kappa=1., fixed_learning_rate=.0001,
        previous_readout_checkpoint_loaded=False, previous_optimizer_or_order_RNG_loaded=False, formal_resume_qualified=False,
        native610_and_typed_BN106_affine159_buffers_CPU_exact=True, zero_native_state_mutants=True,
        full_state_proof=record['state_proof'], artifacts=record, matching=matches, training=trace_proof, audits=audits,
        checkpoint_results=checkpoint_proofs, endpoint_results=endpoint, numeric_protocol=NUMERIC,
        current_and_seven_prior_raw_terminals_before_hashes_imports=True, complete_current_SOURCE_closure_GPU_owner_bound=True,
        independent_floating_audit_microsteps=list(DETAILS), all17784_floating_training_losses_gradients_replayed=False,
        independent_CPU_dense_masks_and_scores_and_COCO12=True, official_COCO12_maxDets=[1, 10, 100],
        all187_DEV380_raw_GT_and_all100_queries_retained=True, no_TEST_access=True, AP_computed=True,
        DEV_historically_exposed=True, DEV_selection=False, calibrated_IoU_claimed=False,
        efficacy_claimed=False, novelty_claimed=False, convergence_claimed=False, SOTA_claimed=False,
        compute_or_native_objective_matched_claimed=False, CPU_affinity=[0, 1], torch_threads=2, torch_interop_threads=2,
        CUDA_initialized=False, elapsed_seconds=time.monotonic() - started,
        evidence_limits=[
            'One frozen original nativeE24 teacher and one readout seed; exposed DEV is evaluated only at the prelocked final readoutE24 endpoint.',
            'Four actual microsteps support independent loss/gradient/clip/AdamW replay. Remaining trace scalars/orders/counts/state joins are not a complete CUDA optimizer trajectory replay.',
            'The exported native100xGT matching matrix minimum is CPU checked; sampled native matching cost derivation is not independently rerun.',
            'Independent dense inference preserves source-observed native query export order, all100 queries and original class identity. CPU masks/scores and all12 COCO statistics are recomputed; GPU exports are separately evaluated and compared.',
            'Permitted binary disagreement lies near zero in CPU logit space. Without saved original GPU logits this is a practical consistency rule, not measured logit allclose or a certified GPU error bound.',
            'A learned posterior is supervised only on matched queries. Its pixel budget and the zero-learning native-posterior budget are new scores, with no calibrated IoU or unmatched/duplicate-query calibration guarantee.',
            'P trains the unbounded posterior logit L+u; B trains L+tanh(u). Shared seeds/data/budget do not establish equal realized trajectories or convergence.',
            'GT-best100 geometry is an explicitly oracle, non-one-to-one metric retaining crowd/ignore/empty rasters; it does not deploy a GT selector.',
            'No blind TEST, novelty, SOTA, full-network additional-training or compute/objective matching claim.'])
    with target.open('x', encoding='utf-8') as stream:
        json.dump(acceptance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=QUAL_STATUS, acceptance=reference(target)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out-acceptance', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Only ROOT-admitted CPU qualification may execute')
    run(args.spec, args.out_acceptance)


if __name__ == '__main__':
    main()
