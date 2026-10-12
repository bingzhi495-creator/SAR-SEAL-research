"""SOURCE-only own qualified E24 readout weights plus48 fresh training epochs.

ROOT alone runs this single-attempt entry. Eight prior raw terminals gate all
hashes, qualifications, data and scientific imports. The P arm optimizes L+u;
the B arm optimizes its deployed L+tanh(u). Both retain the same eight tensors.
The scalar head is Mask-Scoring-inspired: equal9441 scalar count only, six rather
than eight tensors and different computation. All100 native-query max-GT IoU
targets retain duplicates. Only the fixed phase48/effective-readout72 DEV endpoint
is evaluated, with two additional P L+u geometry controls and the original nine.
Teacher24+readout72 data exposure96 is not native96 compute matching.
"""
import argparse
from copy import deepcopy
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types

SCHEMA = 'ROOT-query-pixel-posterior-scalar-control-E24warm-additional48-Puncap-eval-spec-v1'
PREFIX = 'query-pixel-posterior-scalar-control-E24warm-additional48-Puncap-eval'
FIRST24_PREFIX = 'query-pixel-posterior-scalar-control-fullFIT24'
ARMS = ('posterior_BCE', 'bounded_BCE', 'query_scalar_IoU')
PIXEL_ARMS = ARMS[:2]
DETAILS = (0, 1, 35566, 35567)
FIRST24_CONTROLS = ('native_official', 'native_zero_budget',
    'P_native_posterior_budget', 'P_corrected_official', 'P_corrected_posterior_budget',
    'B_native_posterior_budget', 'B_corrected_official', 'B_corrected_posterior_budget', 'scalar_native_IoU')
CONTROLS = FIRST24_CONTROLS + ('P_uncapped_official', 'P_uncapped_posterior_budget')
CPU_NUMERIC_PROTOCOL = dict(point_rtol=1e-4, point_atol=1e-6, gradient_rtol=1e-3,
    gradient_atol=1e-7, moment_rtol=1e-4, moment_atol=1e-8, class_rtol=1e-6,
    class_atol=1e-6, quality_rtol=1e-4, quality_atol=1e-4,
    native_projection_rtol=1e-4, native_projection_atol=1e-4,
    CPU_logit_nearzero_envelope='1e-4+1e-4*abs(CPU_logit)', certified_GPU_error_bound=False)
MODULES = ('preflight_source', 'micro_preflight_source', 'initializer_source', 'native_helpers',
    'native_preflight_helpers', 'official_evaluation_source', 'previous_control_entry_source', 'prior_diag_entry_source')
NEW_ROLES = ('entry_source', 'CPU_reader', 'SOURCE_proposal', 'posterior_readout_source', 'native_adapter_source',
    'preflight_helper_source', 'native_preflight_entry_source', 'native_preflight_CPU_reader_source',
    'native_preflight_spec', 'native_preflight_CPU_qualification', 'native_preflight_report', 'native_preflight_receipt',
    'query_iou_control_source', 'scalar_CPU_fixture', 'scalar_fixture_source',
    'first24_training_spec', 'first24_CPU_qualification', 'first24_report', 'first24_receipt',
    'first24_entry_source', 'first24_CPU_reader_source', 'first24_SOURCE_proposal')
PRIOR = (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
    ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
    ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase'))
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'


def phase_LR_factor(completed_updates):
    require(type(completed_updates) is int and 0 <= completed_updates <= 17784, 'Valid completed phase update count required')
    return .1 ** sum(completed_updates >= milestone for milestone in (15808, 17126))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
    require(isinstance(row, dict) and Path(row['path']).is_absolute() and reference(row['path']) == row,
        'Exact absolute pinned reference required')
    return Path(row['path']).resolve(strict=True)


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Only explicitly pinned Python SOURCE may be loaded')
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == row['sha256'], 'SOURCE changed before compilation')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    require(not path.exists() and not temp.exists(), 'Fresh atomic JSON required')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def terminal(row, phase):
    require(row['status'] == 'ACTUAL_OWNED_TERMINAL' and row['original_identity'] is not None and
        type(row['returncode']) is int and row['returncode'] == 0 and row['root_wait_reaped'] is True and
        row['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        row['numeric_birth_absence']['original_birth_absent'] is True and row['abort'] is None and
        row['error'] is None and row['wait_errors'] is False and row['GPU_observation_errors'] == [],
        'STOP before hashes/imports: actual held0/reap/original-birth absence/noerror required')
    peak = row['whole_GPU_observed_peak_MiB']
    require(row.get('phase', phase) == phase and row['whole_GPU_used_limit_MiB'] == 8192 and
        row['whole_leaf_VRAM_certified'] is False and row['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation' and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192, 'Observed whole-GPU scope differs')


def gate(spec_path, spec):
    # Read construction metadata only; do not hash the context until ALL raw gates.
    require(Path(spec['context']['path']).is_absolute(), 'Absolute read-only context required')
    ctx = read(spec['context']['path'])
    raw = {}
    for receipt, phase in PRIOR:
        require(Path(ctx[receipt]['path']).is_absolute(), 'Absolute original raw receipt required')
        raw[receipt] = read(ctx[receipt]['path'])
        terminal(raw[receipt], ctx[phase])
    raw['native_preflight_receipt'] = read(spec['native_preflight_receipt']['path'])
    terminal(raw['native_preflight_receipt'], spec['native_preflight_phase'])
    raw['first24_receipt'] = read(spec['first24_receipt']['path'])
    terminal(raw['first24_receipt'], spec['first24_phase'])
    require(spec['schema'] == SCHEMA and spec['arms'] == list(ARMS) and spec['controls'] == list(CONTROLS) and
        spec['epochs'] == 48 and spec['microsteps'] == 35568 and spec['optimizer_updates'] == 17784 and
        spec['raw_GT_visits'] == 79728 and spec['accumulation'] == 2 and spec['readout_seed'] == 2026100912 and
        spec['order_seed'] == 2026101022 and spec['point_seed_base'] == 3900000 and spec['kappa'] == 1 and
        spec['phase_epochs'] == 48 and spec['effective_readout_epochs'] == 72 and spec['teacher_epoch'] == 24 and
        spec['total_data_exposure_epochs'] == 96 and spec['LR_schedule'] is True and
        spec['schedule'] == dict(milestones=[15808, 17126], gamma=.1, warmup_updates=0, warmup_start_factor=1) and
        spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
        type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and spec['no_feature_cache'] is True,
        'Prelocked two-objective numerical/resource scope differs')
    require(ctx['schema'] == 'ROOT-shared-readout-objective-control-fullFIT24-spec-v1' and
        ctx['expected_epoch'] == 24 and ctx['readout_seed']['seed_id'] == 0, 'Original nativeE24 seed0 construction context required')
    source, spec_ref = reference(__file__), reference(spec_path)
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read(bound(admission_ref))
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['driver_sha256'] == OWNER_SHA and admission['allowed_cpus'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and admission['whole_GPU_used_limit_MiB'] == 8192 and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        Path(admission['entry']).resolve() == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'], 'Current ROOT single-attempt owner differs')
    require(source == spec['entry_source'] and source in spec['sources'] and spec['context'] in spec['sources'] and
        len({r['path'] for r in spec['sources']}) == len(spec['sources']), 'Current unique closure/self/context binding absent')
    require(admission['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'] and
        admission.get('scientific_spec', spec_ref) == spec_ref, 'Current SPEC is not owner-bound')
    for role in NEW_ROLES:
        require(spec[role] in spec['sources'], 'New role absent from current owner closure: ' + role)
    require(set(spec['warm_checkpoints']) == set(ARMS) and all(r in spec['sources'] for r in spec['warm_checkpoints'].values()),
        'All three own qualified E24 checkpoints must be current-owner bound')
    for row in ctx['sources']:
        require(row in spec['sources'], 'Original context closure must remain exactly retained')
    for row in spec['sources']:
        require(admission['bound_source_sha256'].get(row['path']) == row['sha256'], 'Full current closure must be owner-bound')
        bound(row)
    proposal = read(bound(spec['SOURCE_proposal']))
    require(proposal['schema'] == PREFIX + '-SOURCE-proposal-v1' and proposal['producer_SOURCE'] == source and
        proposal['CPU_numeric_protocol'] == CPU_NUMERIC_PROTOCOL, 'Current authoring proposal/protocol differs')
    fixture = read(bound(spec['scalar_CPU_fixture']))
    require(fixture['status'] == 'CAPACITY_MATCHED_QUERY_IOU_SYNTHETIC_CPU_PASS' and
        fixture['sources'] == dict(fixture=spec['scalar_fixture_source'], control=spec['query_iou_control_source']) and
        fixture['synthetic_execution_performed'] is True and fixture['parameter_count'] == 9441 and fixture['parameter_tensors'] == 6 and
        fixture['execution'] == dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2,
            cuda_visible_devices='', CUDA_initialized=False) and fixture['scientific_data_accessed'] is False and
        fixture['native_integration_qualified'] is False and fixture['target_generation_qualified'] is False,
        'Actual source-bound scalar synthetic CPU PASS required')
    for flag in ('capacity_9441_six_tensors_no_buffers', 'private_CPU_RNG_preserved',
        'cold_final_only_and_warm_all6_gradients', 'independent_scalar_MSE_and_gradients',
        'all100_query_order_and_permutations', 'target_bounds_and_empty_descriptors', 'teacher_inputs_detached_unchanged'):
        require(fixture['checks'][flag] is True, 'Missing actual scalar fixture check: ' + flag)
    pre = read(bound(spec['native_preflight_spec']))
    q = read(bound(spec['native_preflight_CPU_qualification']))
    report = read(bound(spec['native_preflight_report']))
    require(q['schema'] == 'ROOT-query-pixel-posterior-native-preflight-CPU-qualification-v1' and
        q['status'] == 'ROOT_QUERY_PIXEL_POSTERIOR_NATIVE_PREFLIGHT_CPU_QUALIFIED' and
        q['source'] == spec['native_preflight_entry_source'] and q['reader_source'] == spec['native_preflight_CPU_reader_source'] and
        q['spec'] == spec['native_preflight_spec'] and q['report'] == spec['native_preflight_report'] and
        q['parent_receipt'] == spec['native_preflight_receipt'] and q['context'] == spec['context'] and
        q['optimizer_steps'] == 0, 'Actual native integration CPU qualification is required')
    for role in ('posterior_readout_source', 'native_adapter_source', 'preflight_helper_source'):
        require(pre[role] == q[role] == spec[role], 'Qualified integration component changed: ' + role)
    require(pre['entry_source'] == spec['native_preflight_entry_source'] and pre['context'] == spec['context'] and
        pre['phase'] == spec['native_preflight_phase'] and report['source'] == pre['entry_source'] and
        report['spec'] == spec['native_preflight_spec'] and report['context'] == spec['context'] and
        report['schema'] == 'query-pixel-posterior-native-preflight-entry-report-v1' and
        report['status'] == 'QUERY_PIXEL_POSTERIOR_NATIVE_PREFLIGHT_ENTRY_BACKWARD_PASS' and report['optimizer_steps'] == 0,
        'Qualified native integration report/SPEC/source joins differ')
    require(q['GPU_admission'] == report['GPU_admission'] and q['GPU_admission'] in spec['sources'] and
        raw['native_preflight_receipt']['admission_sha256'] == q['GPU_admission']['sha256'] and
        Path(pre['GPU_admission_file']).resolve() == Path(q['GPU_admission']['path']).resolve(), 'Integration raw/admission join differs')
    actual = report['result']
    require(actual['native610_exact'] is True and actual['native451_frozen_gradNone'] is True and
        actual['BN106_affine159_buffers_exact'] is True and actual['all_native_modules_eval'] is True and
        actual['native_all10_tensor_versions_unchanged'] is True and actual['external_RNG_restored'] is True and
        actual['hooks_removed'] is True and actual['no_optimizer_step'] is True and
        actual['native_forward_calls'] == 1 and actual['backward_calls'] == 2, 'Actual native cold/warm integration proof differs')
    first24_gate(spec, raw['first24_receipt'])
    return ctx, raw, spec_ref, admission_ref


def first24_gate(spec, raw):
    """Actual qualified own E24 weights, not optimizer/order/RNG restoration."""
    previous = read(bound(spec['first24_training_spec']))
    q = read(bound(spec['first24_CPU_qualification']))
    report = read(bound(spec['first24_report']))
    require(previous['schema'] == 'ROOT-' + FIRST24_PREFIX + '-spec-v1' and previous['phase'] == spec['first24_phase'] and
        previous['entry_source'] == spec['first24_entry_source'] and previous['CPU_reader'] == spec['first24_CPU_reader_source'] and
        previous['SOURCE_proposal'] == spec['first24_SOURCE_proposal'] and previous['context'] == spec['context'] and
        previous['epochs'] == 24 and previous['microsteps'] == 17784 and previous['optimizer_updates'] == 8892 and
        previous['raw_GT_visits'] == 39864 and previous['arms'] == list(ARMS) and previous['controls'] == list(FIRST24_CONTROLS),
        'Qualified first24 SPEC/source/context/phase scope differs')
    for role in ('posterior_readout_source', 'native_adapter_source', 'query_iou_control_source', 'preflight_helper_source',
        'native_preflight_entry_source', 'native_preflight_CPU_reader_source', 'native_preflight_spec',
        'native_preflight_CPU_qualification', 'native_preflight_report', 'native_preflight_receipt',
        'scalar_CPU_fixture', 'scalar_fixture_source', 'readout_seed', 'order_seed', 'point_seed_base', 'kappa'):
        require(previous[role] == spec[role], 'First24 component/protocol changed: ' + role)
    require(q['schema'] == 'ROOT-' + FIRST24_PREFIX + '-CPU-qualification-v1' and
        q['status'] == 'ROOT_QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_FULLFIT24_CPU_QUALIFIED' and
        q['source'] == spec['first24_entry_source'] and q['reader_source'] == spec['first24_CPU_reader_source'] and
        q['spec'] == spec['first24_training_spec'] and q['report'] == spec['first24_report'] and
        q['parent_receipt'] == spec['first24_receipt'] and q['SOURCE_proposal'] == spec['first24_SOURCE_proposal'] and
        q['context'] == spec['context'] and q['phase'] == spec['first24_phase'] and q['arms'] == list(ARMS) and
        q['completed_epochs'] == 24 and q['microsteps'] == 17784 and q['optimizer_updates'] == 8892 and
        q['raw_GT_visits'] == 39864 and q['fixed_endpoint_epoch'] == 24 and q['DEV_selection'] is False and
        q['native610_and_typed_BN106_affine159_buffers_CPU_exact'] is True and q['zero_native_state_mutants'] is True and
        q['artifacts']['checkpoints'] == spec['warm_checkpoints'] and report['result'] == q['artifacts'] and
        report['source'] == spec['first24_entry_source'] and report['spec'] == spec['first24_training_spec'] and
        set(q['checkpoint_results']) == set(ARMS), 'Actual first24 CPU qualified source/endpoint/own checkpoints differ')
    for role in ('GPU_admission', 'start', 'GPU_qualification'):
        require(q[role] in spec['sources'], 'Qualified first24 primary evidence must be current owner-bound: ' + role)
        bound(q[role])
    admission = read(bound(q['GPU_admission']))
    require(raw['admission_sha256'] == q['GPU_admission']['sha256'] and
        Path(previous['GPU_admission_file']).resolve() == Path(q['GPU_admission']['path']).resolve() and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['automatic_retry'] is False and
        admission['phase'] == spec['first24_phase'] and admission['driver_sha256'] == OWNER_SHA and
        admission['entry_args'] == ['--spec', spec['first24_training_spec']['path'], '--owner-execute'] and
        Path(admission['entry']).resolve() == Path(spec['first24_entry_source']['path']).resolve() and
        admission['bound_source_sha256'].get(spec['first24_training_spec']['path']) == spec['first24_training_spec']['sha256'],
        'Original first24 raw/owner/SPEC joins differ')
    for row in previous['sources']:
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Entire actually admitted first24 closure must be preserved')
        bound(row)
    for role in ('state_proof', 'matching'):
        require(q['artifacts'][role] in spec['sources'], 'Qualified first24 proof/matching must be current owner-bound')
        bound(q['artifacts'][role])
    state = read(bound(q['artifacts']['state_proof']))
    require(state['completed_epochs'] == 24 and state['full_native610_CPU_exact'] is True and
        state['optimizer_updates'] == {a: 8892 for a in ARMS}, 'Actual first24 endpoint state scope differs')
    for arm in ARMS:
        qualified = q['checkpoint_results'][arm]
        require(qualified['checkpoint'] == spec['warm_checkpoints'][arm] and qualified['strict_CPU_loaded'] is True and
            qualified['teacher610_unchanged'] is True and qualified['AdamW_step'] == 8892,
            'Each arm must warm only from its own qualified E24 weights')
    return q, state, read(bound(q['artifacts']['matching']))


def warm_readout_states(torch, pref, shared, spec, readouts, previous_state):
    loaded = {}
    for arm, module in readouts.items():
        cp = torch.load(str(bound(spec['warm_checkpoints'][arm])), map_location='cpu')
        require(cp['schema'] == FIRST24_PREFIX + '-checkpoint-v1' and cp['meta']['arm'] == arm and
            cp['meta']['spec'] == spec['first24_training_spec'] and cp['meta']['source'] == spec['first24_entry_source'] and
            cp['meta']['phase'] == spec['first24_phase'] and cp['meta']['epochs'] == 24 and cp['meta']['optimizer_updates'] == 8892,
            'Only actual own first24 endpoint weights may warm this phase')
        weights = cp['readout_state_dict']
        require(set(weights) == set(module.state_dict()) and all(v.device.type == 'cpu' and v.dtype == torch.float32 and
            not v.requires_grad and tuple(v.shape) == tuple(module.state_dict()[n].shape) and bool(torch.isfinite(v).all())
            for n, v in weights.items()) and shared.records(pref, weights) == previous_state['final_readout_states'][arm],
            'Warm tensors must be exact complete qualified FP32 own-arm states')
        module.load_state_dict(weights, strict=True)
        require(all(torch.equal(module.state_dict()[n], v) for n, v in weights.items()), 'Strict warm CPU state readback differs')
        loaded[arm] = dict(checkpoint=spec['warm_checkpoints'][arm], initial_readout_state=shared.records(pref, module.state_dict()),
        strict_CPU_loaded=True, previous_optimizer_loaded=False, previous_order_loaded=False, previous_external_RNG_loaded=False)
        del cp, weights
    return loaded


def point_packet(readout, fields, query_ids, targets, indices):
    """Same original integer positions for pixel, query, L, contrast and raw GT."""
    count, samples = indices.shape
    H, W = fields['native_logits'].shape[-2:]
    pixel = fields['projected_pixel'][0].flatten(1).t()[indices.reshape(-1)]
    query = fields['projected_query'][0, query_ids][:, None].expand(count, samples, 16).reshape(-1, 16)
    native = fields['native_logits'][0, query_ids].reshape(count, H * W).gather(1, indices)
    contrast = fields['contrast'][0, query_ids].reshape(count, H * W).gather(1, indices)
    result = readout.point_outputs(pixel, query, native.reshape(-1), contrast.reshape(-1))
    target = targets.reshape(count, H * W).gather(1, indices).to(pixel.dtype)
    return result, target


def objective(torch, adapter, readout, fields, query_ids, targets, indices, arm, detailed):
    if arm == ARMS[0]:
        loss = adapter.matched_point_loss(readout, fields, query_ids, targets, indices)
        if not detailed:
            return loss, None
    if len(targets) == 0:
        return sum((p.reshape(-1)[:0].sum() for p in readout.parameters()), fields['projected_pixel'].new_zeros(())), None
    result, target = point_packet(readout, fields, query_ids, targets, indices)
    logits = result['posterior_logits' if arm == ARMS[0] else 'corrected_logits'].reshape_as(target)
    derived = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1).mean()
    if arm == ARMS[0]:
        require(torch.equal(loss.detach(), derived.detach()), 'Detailed P objective differs from immutable adapter')
    else:
        loss = derived
    return loss, logits


def budget(torch, support, posterior):
    B = support.to(posterior.dtype)
    return (B * posterior).flatten(1).sum(1) / (B.flatten(1).sum(1) + ((1 - B) * posterior).flatten(1).sum(1) + 1e-6)


def score_rows(torch, shared, np, masks_api, fusion, classes, logits, posterior, image_id, pref, use_budget):
    predictions, factors, qids, class_packet = shared.export_original(torch, np, masks_api, fusion, classes, logits, image_id, pref)
    if use_budget:
        quality = budget(torch, logits > 0, posterior)
        class_probability = classes[0].softmax(-1)[:, 0]
        scores = class_probability * quality
        require(bool(torch.isfinite(scores).all()) and bool(((scores >= 0) & (scores <= 1)).all()), 'Finite bounded budget scores required')
        for prediction, factor in zip(predictions, factors):
            q = prediction['query_id']
            prediction['score'] = float(scores[q].item())
            factor['mask_quality'] = float(quality[q].item())
            factor['official_score'] = prediction['score']
    return predictions, factors, qids, class_packet


def scalar_supervision(torch, pref, scalar, native_original, classes, targets, identity, micro, cache, ctx, np, masks_api):
    """GT-free four descriptors; cache only all100 geometric target scalars."""
    with torch.no_grad():
        H, W = native_original.shape[-2:]
        require(tuple(native_original.shape) == (100, H, W) and tuple(targets.shape[-2:]) == (H, W) and
            H * W < 2 ** 24 and not torch.backends.cuda.matmul.allow_tf32 and targets.dtype == torch.bool,
            'FP32 exact-integer overlap count scope requires HW<2**24/TF32off/original bool GT')
        binary, posterior = native_original > 0, native_original.sigmoid()
        flat = binary.flatten(1).to(torch.float32)
        area = flat.sum(1)
        official_quality = (flat * posterior.flatten(1)).sum(1) / (area + 1e-6)
        zero_budget = budget(torch, binary, posterior)
        class_probability = classes[0].softmax(-1)[:, 0]
        descriptors = scalar.assemble_native_descriptors(class_probability[None], binary[None],
            official_quality[None], zero_budget[None])
        key, reused = str(identity['image_id']), str(identity['image_id']) in cache
        if not reused or micro in DETAILS:
            raw = targets.flatten(1).to(torch.float32)
            intersection = flat @ raw.t()
            union = area[:, None] + raw.sum(1)[None] - intersection
            require(bool(torch.isfinite(intersection).all()) and bool((intersection == intersection.round()).all()) and
                bool((union == union.round()).all()) and bool(((intersection >= 0) & (union >= 0) & (union <= H * W)).all()),
                'Native/GT intersections and unions must be finite exact integer counts')
            values = (torch.where(union > 0, intersection / union.clamp_min(1), torch.zeros_like(union)).max(1).values
                if len(targets) else flat.new_zeros(100))
            require(bool(torch.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()), 'Finite all100 max-GT IoU required')
            binary_digest = pref.digest_tensor(binary)
            if reused:
                require(torch.equal(values.cpu(), torch.tensor(cache[key]['targets'], dtype=torch.float32)) and
                    binary_digest == cache[key]['source']['native_binary_masks_digest'], 'Audited native support/cached targets changed')
            else:
                cache[key] = dict(targets=values.cpu().tolist(), source=dict(image_id=identity['image_id'], first_microstep=micro,
                    raw_ann_ids=identity['ann_ids'], original_shape=identity['original_shape'], teacher_epoch=24,
                    teacher_seed=100000 + identity['image_id'], teacher_checkpoint=ctx['teacher_checkpoint'],
                    native_binary_masks_digest=binary_digest, raw_GT_digest=pref.digest_tensor(targets),
                    query_ordinals=list(range(100)), empty_union_target=0., no_GT_target=0.,
                    target_kind='maximum_native_binary_IoU_over_all_raw_GT', integer_count_HW_bound=2 ** 24))
        rles = None
        if micro in DETAILS:
            actual = binary.cpu().numpy().astype(np.uint8)
            rles = []
            for query_id, mask in enumerate(actual):
                encoded = masks_api.encode(np.asfortranarray(mask))
                counts = encoded['counts'].decode('ascii') if isinstance(encoded['counts'], bytes) else encoded['counts']
                rles.append(dict(query_id=query_id, segmentation=dict(size=list(encoded['size']), counts=counts)))
        return descriptors, torch.tensor(cache[key]['targets'], device=native_original.device, dtype=torch.float32)[None], cache[key]['source'], reused, rles


def dense_posterior_logits(torch, readout, fields):
    """DEV-only direct L+u; immutable dense API omits this fourth field.

    Repeat its factorized first-linear arithmetic in query chunks4, with no
    inverse sigmoid or new model parameters. The original nine exports still
    use the immutable dense_outputs call unchanged.
    """
    require(not torch.is_grad_enabled(), 'Uncapped dense geometry is inference-only')
    pixel, query, native = fields['projected_pixel'], fields['projected_query'], fields['native_logits']
    batch, _, height, width = pixel.shape
    W = readout.hidden.weight
    pixel_hidden = torch.nn.functional.linear(pixel.permute(0, 2, 3, 1), W[:, :16],
        readout.hidden.bias).permute(0, 3, 1, 2)
    query_hidden = torch.nn.functional.linear(query, W[:, 16:32])
    posterior_logits = torch.empty_like(native)
    for start in range(0, native.shape[1], 4):
        stop = min(start + 4, native.shape[1])
        L = native[:, start:stop]
        p = L.sigmoid()
        local_mean = torch.nn.functional.avg_pool2d(p.reshape(batch * (stop - start), 1, height, width),
            kernel_size=3, stride=1, padding=1, count_include_pad=False).reshape(batch, stop - start, height, width)
        hidden = pixel_hidden[:, None] + query_hidden[:, start:stop, :, None, None]
        for j, value in enumerate((L, p, (L > 0).to(L.dtype), p - local_mean)):
            hidden.addcmul_(value[:, :, None], W[:, 32 + j].reshape(1, 1, 32, 1, 1))
        u = torch.nn.functional.linear(torch.nn.functional.silu(hidden).permute(0, 1, 3, 4, 2),
            readout.final.weight, readout.final.bias).squeeze(-1)
        block = L + u
        require(bool(torch.isfinite(u).all()) and bool(torch.isfinite(block).all()), 'Finite direct uncapped L+u required')
        posterior_logits[:, start:stop] = block
    return posterior_logits


def execute(torch, np, masks_api, ctx, spec, spec_ref, split, helpers, model, anchor, loading,
            readouts, optimizers, datasets, cocos, output, started, previous_matching):
    core, pref, initializer, old, native, official, shared, diag = helpers
    adapter = load_source(spec['native_adapter_source'], 'two_objective_adapter')
    scalar = load_source(spec['query_iou_control_source'], 'scalar_IoU_control')
    renderer = partial(diag.render_original, torch)
    deadline = lambda: require(time.monotonic() - started < spec['deadline_seconds'], 'Deadline exceeded; no automatic retry')
    before_hooks = core.hook_inventory(model)
    outside_rng = pref.rng_snapshot(torch)
    order_gen = torch.Generator(device='cpu').manual_seed(spec['order_seed'])
    initial = {a: shared.cpu_copy(torch, m.state_dict()) for a, m in readouts.items()}
    trace, orders, matches, audits, summaries, scalar_cache = [], [], {}, {}, [], {}
    expected = {r['image_id']: r for r in split['train_rows']}
    boundary = {}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    for epoch in range(48):
        before = order_gen.get_state().clone()
        order = [split['train_ids'][i] for i in torch.randperm(741, generator=order_gen).tolist()]
        orders.append(dict(epoch=epoch, image_ids=order, RNG_before=pref.digest_tensor(before), RNG_after=pref.digest_tensor(order_gen.get_state())))
        for offset, image_id in enumerate(order):
            deadline()
            micro, detailed = epoch * 741 + offset, epoch * 741 + offset in DETAILS
            boundary_before = None
            if micro > 0 and micro % 741 == 0 and micro % 2 == 1:
                boundary_before = {a: {n: pref.digest_tensor(p.grad) for n, p in m.named_parameters()} for a, m in readouts.items()}
                require(boundary_before == boundary, 'Pending gradients changed across odd741 epoch boundary')
            data, samples, identity = core.image_data(native, old, model, datasets['train'], cocos['train'], image_id)
            captured, forward, protected = shared.teacher_forward(torch, pref, core, model, data, samples, image_id, detailed)
            identity['metainfo'] = dict(img_shape=list(identity['actual_img_shape']), batch_input_shape=[800, 800], ori_shape=list(identity['original_shape']))
            gt, processed = shared.original_targets(torch, np, cocos['train'], identity, expected[image_id], data, model.panoptic_head)
            key = str(image_id)
            if key not in matches:
                matches[key] = shared.fixed_matching(torch, pref, core, old, model.panoptic_head, captured, processed, identity)
                require(matches[key]['query_by_GT'] == previous_matching[key]['query_by_GT'] and
                    matches[key]['raw_ann_ids'] == previous_matching[key]['raw_ann_ids'], 'Fresh native matching differs from qualified first24 assignment')
            match = matches[key]
            require(match['raw_ann_ids'] == identity['ann_ids'] and len(match['query_by_GT']) == len(gt), 'Raw/processed GT matching row changed')
            query_ids = torch.tensor(match['query_by_GT'], dtype=torch.int64, device='cuda:0')
            targets = gt.to(torch.bool)
            generator = torch.Generator(device='cpu').manual_seed(spec['point_seed_base'] + micro)
            indices_CPU = torch.randint(targets.shape[1] * targets.shape[2], (len(targets), 12544), generator=generator, dtype=torch.int64)
            indices = indices_CPU.to('cuda:0')
            row = dict(global_microstep=micro, epoch=epoch, offset=offset, image_id=image_id, identity=identity, raw_GT=len(gt),
                forward=forward, input_digest=pref.digest_tensor(data['inputs']), indices_digest=pref.digest_tensor(indices_CPU),
                integer_pixel_seed=spec['point_seed_base'] + micro, matching_reused=epoch > 0,
                pending_accumulation_before=micro % 2, pending_accumulation_after=(micro + 1) % 2, arms={})
            audit = dict(schema=PREFIX + '-audit-v1', global_microstep=micro, identity=identity,
                teacher=captured, raw_GT=targets, matching=match, indices=indices_CPU, arms={}) if detailed else None
            for arm in ARMS:
                module, optimizer = readouts[arm], optimizers[arm]
                lr = .0001 * phase_LR_factor(micro // 2)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                parameters_before = shared.cpu_copy(torch, module.state_dict()) if detailed else None
                optimizer_before = shared.cpu_copy(torch, optimizer.state_dict()) if detailed else None
                pending = {n: None if p.grad is None else shared.cpu_copy(torch, p.grad) for n, p in module.named_parameters()} if detailed else None
                before_record = shared.records(pref, module.state_dict())
                fields, quality_predictions = None, None
                if arm in PIXEL_ARMS:
                    fields = adapter.project_native_fields(module, captured['F'], captured['e0'], captured['M200'],
                        identity['actual_img_shape'], identity['original_shape'], renderer)
                    if arm == PIXEL_ARMS[0]:
                        scalar_descriptors, scalar_targets, scalar_source, scalar_reused, scalar_RLEs = scalar_supervision(torch, pref, scalar,
                            fields['native_logits'][0], captured['classes'], targets, identity, micro, scalar_cache, ctx, np, masks_api)
                        if detailed:
                            audit.update(scalar_targets=scalar_targets, scalar_descriptors=scalar_descriptors,
                                scalar_cache_source=scalar_source, native_binary_RLE100=scalar_RLEs)
                    loss, point_logits = objective(torch, adapter, module, fields, query_ids, targets, indices, arm, detailed)
                else:
                    quality_predictions = module(captured['e0'], scalar_descriptors)
                    loss, point_logits = scalar.all_query_mse(quality_predictions, scalar_targets), None
                require(loss.requires_grad and bool(torch.isfinite(loss)), 'Finite live proper matched point BCE required')
                micro_grads = torch.autograd.grad(loss / 2, tuple(module.parameters()), retain_graph=True) if detailed else None
                (loss / 2).backward()
                gradients = shared.gradient_records(torch, pref, module)
                accumulated = {n: shared.cpu_copy(torch, p.grad) for n, p in module.named_parameters()} if detailed else None
                step = (micro + 1) % 2 == 0
                norm, clipped = None, None
                if step:
                    norm = float(torch.nn.utils.clip_grad_norm_(tuple(module.parameters()), .01, error_if_nonfinite=True))
                    require(math.isfinite(norm) and norm >= 0, 'Finite preclip norm required')
                    clipped = shared.gradient_records(torch, pref, module)
                    optimizer.step()
                after_boundary = {n: pref.digest_tensor(p.grad) for n, p in module.named_parameters()} if (micro + 1) % 741 == 0 and (micro + 1) % 2 == 1 else None
                row['arms'][arm] = dict(loss=float(loss.detach()), parameter_before=before_record,
                    parameter_after=shared.records(pref, module.state_dict()), accumulated_gradients_before_clip=gradients,
                    clipped_gradients=clipped, clip_norm_before=norm, optimizer_update=step, updates_completed=(micro + 1) // 2,
                    learning_rate=lr, epoch_boundary_gradient_before=None if boundary_before is None else boundary_before[arm],
                    epoch_boundary_gradient_after=after_boundary)
                if arm == ARMS[2]:
                    row['arms'][arm].update(scalar_target_digest=pref.digest_tensor(scalar_targets),
                        scalar_descriptor_digest=pref.digest_tensor(scalar_descriptors), scalar_target_cache_reused=scalar_reused)
                if detailed:
                    audit['arms'][arm] = shared.cpu_copy(torch, dict(parameters_before=parameters_before,
                        optimizer_state_before=optimizer_before, pending_gradients_before=pending,
                        micro_gradients={n: g for (n, p), g in zip(module.named_parameters(), micro_grads)},
                        accumulated_gradients_before_clip=accumulated,
                        clipped_gradients=None if not step else {n: p.grad for n, p in module.named_parameters()},
                        parameters_after=module.state_dict(), optimizer_state_after=optimizer.state_dict(), point_logits=point_logits,
                        **({} if arm in PIXEL_ARMS else dict(quality_predictions=quality_predictions))))
                if step:
                    optimizer.zero_grad(set_to_none=True)
                del fields, loss, point_logits, quality_predictions, micro_grads, accumulated, parameters_before, optimizer_before, pending
            require(all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m.training for m in model.modules()) and
                [v._version for v in protected['tensors']] == protected['versions'], 'Native teacher or earlier stage tensors changed')
            forward['native_all10_tensor_versions_unchanged'] = True
            if (micro + 1) % 741 == 0 and (micro + 1) % 2 == 1:
                boundary = {a: row['arms'][a]['epoch_boundary_gradient_after'] for a in ARMS}
            if detailed:
                audits[str(micro)] = shared.packet(torch, output / ('AUDIT_%05d.pth' % micro), audit)
            trace.append(row)
            del captured, protected, processed, gt, targets, data, samples, indices, indices_CPU, audit, scalar_descriptors, scalar_targets, scalar_RLEs
            if (micro + 1) % 64 == 0:
                print(json.dumps(dict(completed_microsteps=micro + 1, updates_per_arm=(micro + 1) // 2)), flush=True)
        cumulative = ((epoch + 1) * 741) // 2
        summary = dict(epoch=epoch, microsteps=741, optimizer_updates=cumulative - epoch * 741 // 2,
            cumulative_optimizer_updates=cumulative, pending_accumulation=((epoch + 1) * 741) % 2,
            image_loss_mean={a: math.fsum(r['arms'][a]['loss'] for r in trace[epoch * 741:(epoch + 1) * 741]) / 741 for a in ARMS})
        require(all((p.grad is not None) == bool(summary['pending_accumulation']) for m in readouts.values() for p in m.parameters()), 'Epoch flush changed pending gradient state')
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    require(len(trace) == 35568 and len(matches) == 741 and all(p.grad is None for m in readouts.values() for p in m.parameters()), 'Exact24 complete FIT epochs/no pending gradients required')
    checkpoints = {}
    for arm in ARMS:
        tensor_count = 8 if arm in PIXEL_ARMS else 6
        require(len(optimizers[arm].state) == tensor_count and all(float(s['step']) == 17784 for s in optimizers[arm].state.values()), 'All active AdamW moments must reach17784')
        checkpoints[arm] = shared.packet(torch, output / (arm + '_PHASE_E48_EFFECTIVE_E72.pth'), dict(schema=PREFIX + '-checkpoint-v1',
            meta=dict(arm=arm, phase=spec['phase'], spec=spec_ref, source=spec['entry_source'],
                posterior_readout_source=spec['posterior_readout_source'], native_adapter_source=spec['native_adapter_source'],
                query_iou_control_source=spec['query_iou_control_source'],
                readout_seed=spec['readout_seed'], order_seed=spec['order_seed'], point_seed_base=spec['point_seed_base'], kappa=1.,
                epochs=48, phase_epochs=48, effective_readout_epochs=72, total_data_exposure_epochs=96,
                microsteps=35568, optimizer_updates=17784, raw_GT_visits=79728, teacher_epoch=24,
                teacher_checkpoint=ctx['teacher_checkpoint'], frozen_native610=True, parameter_tensors=tensor_count, parameter_scalars=9441,
                objective='matched_posterior_point_BCE' if arm == ARMS[0] else 'matched_bounded_point_BCE' if arm == ARMS[1] else 'all100_native_max_GT_IoU_MSE',
                optimizer_param_names=[[n] for n, p in readouts[arm].named_parameters()], learning_rate=lr, base_learning_rate=.0001,
                LR_schedule=True, schedule=dict(milestones=[15808, 17126], gamma=.1, warmup_updates=0, warmup_start_factor=1),
                warm_checkpoint=spec['warm_checkpoints'][arm], first24_CPU_qualification=spec['first24_CPU_qualification'],
                weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
                no_previous_optimizer_or_order_restoration=True, formal_resumable=False, fixed_endpoint_epoch=48, DEV_selection=False),
            readout_state_dict=readouts[arm].state_dict(), optimizer_state_dict=optimizers[arm].state_dict(),
            order_generator_state=order_gen.get_state(), fixed_assignment_rows=matches))
    for name, value in (('TRACE', trace), ('ORDER', orders), ('MATCHING', matches)):
        write(output / (name + '.json'), value)
    require(len(scalar_cache) == 741, 'All741 scalar target cache entries required')
    write(output / 'SCALAR_TARGETS.json', scalar_cache)
    exports = {c: dict(predictions=[], score_factors=[], image_rows=[]) for c in CONTROLS}
    DEV_fields = []
    for expected_DEV in split['evaluation_rows']:
        deadline()
        image_id = expected_DEV['image_id']
        data, samples, identity = core.image_data(native, old, model, datasets['DEV'], cocos['DEV'], image_id)
        captured, forward, protected = shared.teacher_forward(torch, pref, core, model, data, samples, image_id)
        require(identity['ann_ids'] == expected_DEV['annotation_ids'], 'DEV all raw annotation identities changed')
        packet_ref = shared.packet(torch, output / ('DEV_FIELDS_%06d.pth' % image_id), dict(schema=PREFIX + '-DEV-fields-v1',
            image_id=image_id, identity=identity, F=captured['F'], e0=captured['e0'], M200=captured['M200'], classes=captured['classes']))
        DEV_fields.append(packet_ref)
        with torch.no_grad():
            native_original = diag.render_original(torch, captured['M200'], identity)
            nativep0 = native_original.sigmoid()
            cases = [('native_official', native_original, nativep0, False),
                     ('native_zero_budget', native_original, nativep0, True)]
            for control, logits, posterior, use_budget in cases:
                values = score_rows(torch, shared, np, masks_api, model.panoptic_fusion_head, captured['classes'], logits, posterior, image_id, pref, use_budget)
                append_export(exports[control], values, control, identity, packet_ref, forward, 'native', 'budget' if use_budget else 'official')
                if control == 'native_official':
                    native_base = deepcopy(values)
            B0 = native_original > 0
            class_probability = captured['classes'][0].softmax(-1)[:, 0]
            official_quality = (B0 * nativep0).flatten(1).sum(1) / (B0.flatten(1).sum(1) + 1e-6)
            scalar_descriptors = scalar.assemble_native_descriptors(class_probability[None], B0[None],
                official_quality[None], budget(torch, B0, nativep0)[None])
            scalar_quality = readouts[ARMS[2]](captured['e0'], scalar_descriptors)[0]
            scalar_quality = torch.where(B0.flatten(1).any(1), scalar_quality, torch.zeros_like(scalar_quality))
            for prediction, factor in zip(native_base[0], native_base[1]):
                query = prediction['query_id']
                prediction['score'] = float((class_probability[query] * scalar_quality[query]).item())
                factor['mask_quality'], factor['official_score'] = float(scalar_quality[query].item()), prediction['score']
            append_export(exports['scalar_native_IoU'], native_base, 'scalar_native_IoU', identity, packet_ref, forward, 'native', 'scalar')
            del native_base, B0, class_probability, official_quality, scalar_descriptors, scalar_quality
            del cases, nativep0, logits, posterior, values
            for arm, label in zip(PIXEL_ARMS, ('P', 'B')):
                fields = adapter.project_native_fields(readouts[arm], captured['F'], captured['e0'], captured['M200'],
                    identity['actual_img_shape'], identity['original_shape'], renderer)
                dense = readouts[arm].dense_outputs(fields['projected_pixel'], fields['projected_query'], fields['native_logits'], query_chunk_size=4)
                for suffix, logits, posterior, use_budget in (
                    ('native_posterior_budget', native_original, dense['posterior'][0], True),
                    ('corrected_official', dense['corrected_logits'][0], dense['posterior'][0], False),
                    ('corrected_posterior_budget', dense['corrected_logits'][0], dense['posterior'][0], True)):
                    control = label + '_' + suffix
                    values = score_rows(torch, shared, np, masks_api, model.panoptic_fusion_head, captured['classes'], logits, posterior, image_id, pref, use_budget)
                    append_export(exports[control], values, control, identity, packet_ref, forward,
                        'native' if suffix.startswith('native') else 'bounded', 'budget' if use_budget else 'official')
                if arm == ARMS[0]:
                    dense['posterior_logits'] = dense_posterior_logits(torch, readouts[arm], fields)
                    for suffix, use_budget in (('uncapped_official', False), ('uncapped_posterior_budget', True)):
                        control = label + '_' + suffix
                        values = score_rows(torch, shared, np, masks_api, model.panoptic_fusion_head, captured['classes'],
                            dense['posterior_logits'][0], dense['posterior'][0], image_id, pref, use_budget)
                        append_export(exports[control], values, control, identity, packet_ref, forward,
                            'unbounded', 'budget' if use_budget else 'official')
                del dense, fields, logits, posterior, values
        require([v._version for v in protected['tensors']] == protected['versions'], 'Native DEV stage versions changed')
        forward['native_all10_tensor_versions_unchanged'] = True
        del data, samples, captured, protected, native_original
    mode_rows = {}
    for control, values in exports.items():
        require(len(values['image_rows']) == 187 and len(values['predictions']) == len(values['score_factors']) == 18700, 'Full187 DEV/all100 queries required')
        mode_rows[control] = {}
        for name, value in values.items():
            path = output / (control + '_' + name.upper() + '.json')
            write(path, value)
            mode_rows[control][name] = reference(path)
        arm = ARMS[0] if control.startswith('P_') else ARMS[1] if control.startswith('B_') else ARMS[2] if control == 'scalar_native_IoU' else None
        mode_rows[control]['checkpoint'] = ctx['teacher_checkpoint'] if arm is None else checkpoints[arm]
    model.cpu()
    require(all(torch.equal(model.state_dict()[n], v) for n, v in anchor.items()) and len(model.state_dict()) == 610 and
        core.hook_inventory(model) == before_hooks and core.equal_rng(torch, outside_rng, pref.rng_snapshot(torch)), 'Native610/hook/external RNG endpoint invariant changed')
    BN = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    proof = dict(schema=PREFIX + '-state-proof-v1', native_teacher610=shared.records(pref, anchor), before=shared.records(pref, anchor),
        after=shared.records(pref, model.state_dict()), teacher_loading=loading, native_parameter_names=list(dict(model.named_parameters())),
        BN_module_names=[n for n, m in BN], BN_parameter_names=[n + '.' + k for n, m in BN for k, p in m.named_parameters(recurse=False)],
        BN_buffer_names=[n + '.' + k for n, m in BN for k, v in m.named_buffers(recurse=False)],
        full_native610_CPU_exact=True, all_native451_frozen=True, native_parameter_grads_None=True, all_modules_eval=True,
        frozen_BN_all_exact=True, hook_cache_clear=True, external_RNG_restored=True, GT_free_forward=True,
        initial_readout_states={a: shared.records(pref, initial[a]) for a in ARMS},
        final_readout_states={a: shared.records(pref, m.state_dict()) for a, m in readouts.items()},
        fresh_optimizer_state_entries={a: 0 for a in ARMS}, completed_epochs=48, microsteps=35568,
        optimizer_updates={a: 17784 for a in ARMS}, training_raw_GT_visits=79728, teacher_forward_calls=35755,
        epoch_summaries=summaries, audit_microsteps=list(DETAILS), audit_refs=audits,
        no_pending_accumulation_gradients=True, previous_readout_checkpoint_loaded=True, no_teacher_optimizer_or_order_restoration=True,
        phase_epochs=48, effective_readout_epochs=72, total_data_exposure_epochs=96, teacher_epoch=24,
        first24_CPU_qualification=spec['first24_CPU_qualification'], weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
        fresh_matching_matches_qualified_first24=True, fixed_endpoint_epoch=48, DEV_selection=False, LR_schedule=True,
        schedule=dict(milestones=[15808, 17126], gamma=.1, warmup_updates=0, warmup_start_factor=1), feature_cache_used=False,
        scalar_target_cache_only=True, scalar_target_cache_images=741,
        scalar_parameter_count_matched=True, scalar_tensor_count_matched=False, scalar_compute_matched=False)
    require(len(proof['native_parameter_names']) == 451 and len(BN) == 53 and len(proof['BN_parameter_names']) == 106 and len(proof['BN_buffer_names']) == 159, 'Typed native BN population differs')
    write(output / 'STATE_PROOF.json', proof)
    return dict(mode_rows=mode_rows, checkpoints=checkpoints, trace=reference(output / 'TRACE.json'), order=reference(output / 'ORDER.json'),
        matching=reference(output / 'MATCHING.json'), state_proof=reference(output / 'STATE_PROOF.json'), audits=audits, DEV_fields=DEV_fields,
        scalar_target_cache=reference(output / 'SCALAR_TARGETS.json'))


def append_export(target, values, control, identity, packet_ref, forward, geometry_kind, quality_kind):
    predictions, factors, qids, class_packet = values
    target['predictions'].extend(predictions)
    target['score_factors'].extend(factors)
    target['image_rows'].append(dict(control=control, image_id=identity['image_id'], identity=identity, query_ids=qids,
        native_class_packet=class_packet, teacher_fields=packet_ref, forward=forward, geometry_kind=geometry_kind,
        quality_kind=quality_kind, base_official_mask_query_score_TorchEqual=True,
        exported_score_replaced_for_budget=quality_kind == 'budget', factor_official_score_key_is_exported_score=True,
        exported_score_replaced_for_scalar=quality_kind == 'scalar'))
    if quality_kind == 'scalar':
        target['image_rows'][-1]['empty_native_mask_quality_forced_zero'] = True


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)
    ctx, raw, spec_ref, admission_ref = gate(spec_path, spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir() and Path(spec['phase']).name == spec['phase'], 'Fresh absolute output/safe phase required')
    helpers = [load_source(ctx[r], 'two_objective_' + r) for r in MODULES]
    core, pref, initializer, old, native, official, shared, diag = helpers
    previous_q, previous_state, previous_matching = first24_gate(spec, raw['first24_receipt'])
    teacher = core.qualified_teacher(ctx, raw['teacher_receipt'])
    import psutil
    import numpy as np
    import torch
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and not torch.cuda.is_initialized(), 'CPU01/coldCUDA0 required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    old.torch = torch
    model, state, meta, loading = official.prepare_model(torch, pref, initializer, None, None, None,
        MODELS, revert_sync_batchnorm, Config, tuple(CocoDataset.METAINFO['classes']), ctx, teacher, 'native0')
    model.requires_grad_(False).eval()
    anchor = shared.cpu_copy(torch, model.state_dict())
    require(len(anchor) == 610 and len(list(model.parameters())) == 451 and all(not getattr(m, 'with_cp', False) for m in model.modules()), 'Strict native610/noncheckpoint teacher required')
    del state, meta
    for obj, role in ((torch.version, 'torch_version_source'), (torch.nn.functional.interpolate, 'interpolation_SOURCE'),
        (torch.optim.AdamW, 'adamw_SOURCE'), (torch.nn.utils.clip_grad_norm_, 'clip_grad_SOURCE'),
        (type(model.panoptic_head), 'native_mask_head_SOURCE'), (type(model.panoptic_fusion_head), 'native_fusion_SOURCE')):
        shared.source_file(obj, ctx[role])
    require(torch.__version__ == '2.0.1+cu118' and model.panoptic_head.num_points == 12544 and
        [type(c).__name__ for c in model.panoptic_head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [float(c.weight) for c in model.panoptic_head.assigner.match_costs] == [2., 5., 5.], 'Qualified Torch/native matching recipe differs')
    readout = load_source(spec['posterior_readout_source'], 'two_objective_readout')
    scalar = load_source(spec['query_iou_control_source'], 'scalar_control_module')
    readouts = {a: readout.QueryPixelPosteriorReadout(seed=spec['readout_seed'], kappa=1.) for a in PIXEL_ARMS}
    require(all(len(m.state_dict()) == 8 and sum(p.numel() for p in m.parameters()) == 9441 and not list(m.buffers()) for m in readouts.values()) and
        all(torch.equal(readouts[ARMS[0]].state_dict()[n], v) for n, v in readouts[ARMS[1]].state_dict().items()), 'Two fresh identical9441/8tensor CPU initializations required')
    readouts[ARMS[2]] = scalar.CapacityMatchedQueryIoUReadout(seed=spec['readout_seed'])
    require(len(readouts[ARMS[2]].state_dict()) == 6 and sum(p.numel() for p in readouts[ARMS[2]].parameters()) == 9441 and
        not list(readouts[ARMS[2]].buffers()), 'Fresh scalar9441/6tensor/no-buffer control required')
    warm_loading = warm_readout_states(torch, pref, shared, spec, readouts, previous_state)
    split = read(bound(ctx['SOURCE_contract']))['split']
    datasets, cocos = {}, {}
    for name, role, images, instances in (('train', 'FIT_annotation', 741, 1661), ('DEV', 'DEV_annotation', 187, 380)):
        datasets[name] = CocoDataset(ann_file=str(bound(ctx[role])), data_prefix=dict(img=ctx['image_root']), metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
        cocos[name] = COCO(str(bound(ctx[role])))
        require(len(datasets[name]) == len(cocos[name].imgs) == images and len(cocos[name].anns) == instances, 'Whole fixed original population required')
    require(split['empty_GT_train_images'] == 0 and split['empty_GT_evaluation_images'] == 0 and not torch.cuda.is_initialized(), 'This real matcher path requires original positive-GT population/allCPU beforeCUDA')
    output.mkdir()
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    model.to('cuda:0')
    for module in readouts.values():
        module.to('cuda:0').train()
    del module
    optimizers = {a: torch.optim.AdamW([dict(params=[p]) for p in m.parameters()], lr=1e-4, weight_decay=.05, betas=(.9, .999), eps=1e-8) for a, m in readouts.items()}
    require(all(len(o.state) == 0 for o in optimizers.values()), 'Fresh independent8/8/6 tensor AdamW moments required')
    write(output / 'START.json', dict(schema=PREFIX + '-start-v1', phase=spec['phase'], spec=spec_ref, context=spec['context'],
        source=spec['entry_source'], reader=spec['CPU_reader'], arms=list(ARMS), controls=list(CONTROLS),
        readout_seed=2026100912, order_seed=2026101022, point_seed_base=3900000, epochs=48, phase_epochs=48,
        effective_readout_epochs=72, teacher_epoch=24, total_data_exposure_epochs=96, microsteps=35568,
        updates=17784, accumulation=2, epoch_flush=False, learning_rate=.0001, LR_schedule=True, kappa=1.,
        schedule=dict(milestones=[15808, 17126], gamma=.1, warmup_updates=0, warmup_start_factor=1),
        original_eight_raw_gates_before_any_hash_or_import=True, fresh_optimizer_state_entries={a: 0 for a in ARMS},
        previous_readout_checkpoint_loaded=True, weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
        first24_CPU_qualification=spec['first24_CPU_qualification'], warm_checkpoints=spec['warm_checkpoints'],
        warm_loading=warm_loading, initial_readout_states={a: shared.records(pref, m.state_dict()) for a, m in readouts.items()},
        fixed_endpoint_epoch=48, DEV_selection=False))
    result = execute(torch, np, masks_api, ctx, spec, spec_ref, split, helpers, model, anchor, loading, readouts,
        optimizers, datasets, cocos, output, started, previous_matching)
    del model, readouts, optimizers, anchor
    gc.collect()
    cleanup = pref.clear_gpu(torch)
    require(cleanup['allocated_after'] == 0 and time.monotonic() - started < spec['deadline_seconds'], 'Actual cleanup/deadline differs')
    for row in spec['sources']:
        bound(row)
    report = dict(schema=PREFIX + '-report-v1', status='QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_GPU_COMPLETE_PENDING_CPU',
        phase=spec['phase'], source=spec['entry_source'], reader=spec['CPU_reader'], spec=spec_ref, context=spec['context'],
        GPU_admission=admission_ref, native_preflight_CPU_qualification=spec['native_preflight_CPU_qualification'],
        arms=list(ARMS), controls=list(CONTROLS), result=result, epochs=48, microsteps=35568, optimizer_updates=17784,
        raw_GT_visits=79728, accumulation=2, epoch_flush=False, readout_seed=2026100912, order_seed=2026101022,
        phase_epochs=48, effective_readout_epochs=72, teacher_epoch=24, total_data_exposure_epochs=96,
        point_seed_base=3900000, kappa=1., learning_rate=.0001, LR_schedule=True,
        schedule=dict(milestones=[15808, 17126], gamma=.1, warmup_updates=0, warmup_start_factor=1),
        first24_CPU_qualification=spec['first24_CPU_qualification'], warm_checkpoints=spec['warm_checkpoints'],
        weights_warm_start=True, previous_optimizer_or_order_RNG_loaded=False,
        all_native610_frozen=True, shared_teacher_forward=True, feature_cache_used=False, fixed_endpoint_epoch=48,
        DEV_historically_exposed=True, DEV_selection=False, no_TEST_access=True, AP_computed=False,
        GPU_optimizer_bitwise_determinism_claimed=False, full_training_floating_CPU_replay_claimed=False,
        calibrated_IoU_claimed=False, efficacy_claimed=False, novelty_claimed=False, convergence_claimed=False,
        compute_or_native_objective_matched_claimed=False, cleanup=cleanup, error=None,
        scalar_parameter_count_matched=True, scalar_tensor_count_matched=False, scalar_compute_matched=False,
        scalar_complete_prior_method_reproduction_claimed=False, scalar_target_cache_only=True,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        evidence_limits=['Only first/last two micros permit independent training floating audit.',
            'P trains L+u and exports both bounded L+tanh(u) and uncapped L+u controls; B trains its bounded logit.',
            'Matched-only BCE does not calibrate all100 queries; duplicate queries are not IoU-zero labels.',
            'Own qualified readoutE24 weights warm this fresh48 phase; previous optimizer/order/RNG are not restored.',
            'Scalar head is Mask-Scoring-inspired;9441 scalar count matches, six vs eight tensors/compute do not.',
            'Scalar targets are each native mask maximum IoU over all raw GT; duplicates may have high targets.',
            'CPU dense factorization changes operation order; tolerances/near-zero checks are not certified GPU error bounds.',
            'One frozen teacher/readout seed and exposed DEV; only phase48/effective-readout72 is evaluated.',
            'Teacher24 plus readout72 data exposure96 does not match native96 computation, optimizer trajectory or objective.',
            'Prelocked48 budget and milestone schedule are not convergence or novelty evidence.'])
    write(output / 'REPORT.json', report)
    write(output / 'QUALIFIED.json', dict(schema=PREFIX + '-aggregate-v1', status='QUERY_PIXEL_POSTERIOR_SCALAR_CONTROL_E24WARM_ADDITIONAL48_PUNCAP_EVAL_PENDING_CPU_READER',
        phase=spec['phase'], source=spec['entry_source'], spec=spec_ref, report=reference(output / 'REPORT.json'), result=result, independently_CPU_qualified=False))
    print(json.dumps(dict(status=report['status'], output_directory=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Only ROOT-owned admitted single attempt may execute')
    run(args.spec)


if __name__ == '__main__':
    main()
