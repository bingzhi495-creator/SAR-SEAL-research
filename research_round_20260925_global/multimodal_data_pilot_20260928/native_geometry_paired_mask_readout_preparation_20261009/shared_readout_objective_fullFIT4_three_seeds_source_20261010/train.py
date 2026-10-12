"""Inert SOURCE: one prelocked readout seed, full FIT E4, exposed DEV.

ROOT alone executes. Six prior actual raw terminals precede every hash,
qualification, checkpoint, GT read or scientific import. One frozen teacher
forward feeds five readouts per visit; only native assignment is cached.
Global accumulation crosses odd741 epoch boundaries without a flush.
Pooling controls remain prior-inspired controls, not full K-Net reproductions.
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

SCHEMA = 'ROOT-shared-readout-objective-control-fullFIT4-spec-v1'
ARMS = ('raw_uniform_BCE', 'raw_full_BCE', 'pair_pooled_full_BCE', 'mask_pooled_full_BCE', 'positive_global_gain')
ENDPOINTS = ('native0',) + ARMS
DETAIL = (0, 1, 2962, 2963)
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
            size += len(chunk)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and Path(row['path']).is_absolute() and reference(row['path']) == row,
        'Exact absolute pinned reference required')
    return Path(row['path']).resolve(strict=True)


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    require(not path.exists() and not temp.exists(), 'Fresh atomic JSON required')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Explicit Python SOURCE required')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def per_image_BCE(torch, original_logits, targets, indices=None):
    """Pure per-GT pixel mean then image GT mean; duplicates stay duplicated."""
    require(torch.is_tensor(original_logits) and original_logits.ndim == 3 and original_logits.dtype == torch.float32 and
        min(original_logits.shape[1:]) > 0 and bool(torch.isfinite(original_logits).all()), 'Finite FP32 GxHxW logits required')
    require(torch.is_tensor(targets) and targets.shape == original_logits.shape and targets.device == original_logits.device and
        bool(torch.isfinite(targets).all()) and bool(((targets == 0) | (targets == 1)).all()), 'Same-device binary targets required')
    G, H, W = original_logits.shape
    if indices is not None:
        require(torch.is_tensor(indices) and indices.dtype == torch.int64 and indices.device == original_logits.device and
            indices.ndim == 2 and indices.shape[0] == G and indices.shape[1] > 0 and
            bool(((indices >= 0) & (indices < H * W)).all()), 'Actual flattened integer GxK indices required')
    if G == 0:
        return original_logits.sum() * 0.0
    logits, target = original_logits.reshape(G, H * W), targets.to(torch.float32).reshape(G, H * W)
    if indices is not None:
        logits, target = logits.gather(1, indices), target.gather(1, indices)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1).mean()


def fixed_assignment_rows(raw_ann_ids, pos_inds, assigned_gt_indices):
    """Pure map in actual preprocessed GT row order, never sorted ann-ID order."""
    require(all(isinstance(v, (list, tuple)) for v in (raw_ann_ids, pos_inds, assigned_gt_indices)) and
        all(type(v) is int for v in list(raw_ann_ids) + list(pos_inds) + list(assigned_gt_indices)) and
        len(set(raw_ann_ids)) == len(raw_ann_ids) and len(pos_inds) == len(assigned_gt_indices) == len(raw_ann_ids) and
        len(set(pos_inds)) == len(pos_inds) and all(0 <= q < 100 for q in pos_inds) and
        sorted(assigned_gt_indices) == list(range(len(raw_ann_ids))), 'Complete unique native/raw GT assignment required')
    by_gt = dict(zip(assigned_gt_indices, pos_inds))
    return [dict(annotation_id=a, GT_index=j, query_index=by_gt[j]) for j, a in enumerate(raw_ann_ids)]


def score_gain(torch, native_original, log_gain):
    """Positive scalar AFTER original rendering; empty G retains connectivity."""
    require(torch.is_tensor(native_original) and native_original.dtype == torch.float32 and
        (native_original.numel() > 0 or (native_original.ndim == 3 and min(native_original.shape[1:]) > 0)) and
        bool(torch.isfinite(native_original).all()) and torch.is_tensor(log_gain) and log_gain.shape == torch.Size([]) and
        log_gain.dtype == torch.float32 and log_gain.device == native_original.device and bool(torch.isfinite(log_gain)),
        'Finite original logits and same-device scalar log_gain required')
    gain = log_gain.exp()
    require(bool(torch.isfinite(gain)) and bool(gain > 0), 'Actual gain must be finite and positive')
    result = native_original * gain
    require(bool(torch.isfinite(result).all()), 'Scaled original logits must remain finite')
    return result


def raw_terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
        type(receipt['returncode']) is int and receipt['returncode'] == 0 and receipt['root_wait_reaped'] is True and
        receipt['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt['numeric_birth_absence']['original_birth_absent'] is True and receipt['abort'] is None and
        receipt['error'] is None and receipt['wait_errors'] is False and receipt['GPU_observation_errors'] == [],
        'STOP before hashes/imports: actual held0/reap/birth absence/noerror required')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(receipt.get('phase', phase) == phase and receipt['whole_GPU_used_limit_MiB'] == 8192 and
        receipt['whole_leaf_VRAM_certified'] is False and
        receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation' and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192, 'Prior observed whole-GPU scope differs')


def owner_gate(spec_path, spec):
    source, spec_ref = reference(__file__), reference(spec_path)
    admission = read(spec['GPU_admission_file'])
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and admission['driver_sha256'] == OWNER_SHA and
        admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] == 5400 and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'], 'Actual current owner/entry/resources differ')
    hashes = admission['bound_source_sha256']
    require(source == spec['entry_source'] and source in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
        len({row['path'] for row in spec['sources']}) == len(spec['sources']) and
        hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current entry/SPEC/reader unique closure absent')
    for row in spec['sources']:
        require(hashes.get(row['path']) == row['sha256'], 'Entire SOURCE/input closure must be owner-bound')
        bound(row)
    require(admission.get('scientific_spec', spec_ref) == spec_ref, 'Owner scientific SPEC differs')
    return spec_ref


def previous_control_gate(spec, contract, raw):
    roles = ('previous_control_spec', 'previous_control_receipt', 'previous_control_CPU_qualification',
        'previous_control_CPU_consumer_manifest', 'previous_control_CPU_failure', 'previous_control_CPU_reader_source',
        'previous_control_admitted_CPU_reader_source', 'previous_control_contract', 'previous_control_entry_source', 'previous_control_phase')
    for role in roles:
        require(spec[role] == contract[role], 'Canonical prior control lineage differs: ' + role)
    previous = read(bound(spec['previous_control_spec']))
    q = read(bound(spec['previous_control_CPU_qualification']))
    require(q['schema'] == 'ROOT-shared-readout-objective-control-FIT128-CPU-qualification-v1' and
        q['status'] == 'ROOT_SHARED_READOUT_OBJECTIVE_CONTROL_FIT128_CPU_QUALIFIED' and
        q['source'] == previous['entry_source'] == spec['previous_control_entry_source'] and
        q['reader_source'] == spec['previous_control_CPU_reader_source'] and
        q['admitted_reader_source'] == previous['CPU_reader'] == spec['previous_control_admitted_CPU_reader_source'] and
        q['spec'] == spec['previous_control_spec'] and q['parent_receipt'] == spec['previous_control_receipt'] and
        q['CPU_consumer_manifest'] == spec['previous_control_CPU_consumer_manifest'] and
        q['original_CPU_failure'] == spec['previous_control_CPU_failure'] and
        q['SOURCE_contract'] == previous['SOURCE_contract'] == spec['previous_control_contract'] and
        q['phase'] == previous['phase'] == spec['previous_control_phase'] and
        q['successor_reader_separately_consumer_bound_before_imports'] is True and
        q['original_GPU_admitted_SOURCE_closure_entries'] == 101 and q['successor_reader_in_original_GPU_admission'] is False and
        q['native610_and_typed_BN106_affine159_buffers_CPU_exact'] is True and q['zero_native_state_mutants'] is True and
        q['all_selected_raw_GT_including_empty_crowd_ignore_preserved'] is True and q['AP_computed'] is True,
        'Actual v2 CPU qualification with truthful original v1 lineage required')
    consumer = read(bound(q['CPU_consumer_manifest']))
    require(set(consumer) == {'schema', 'inputs', 'original_spec', 'admitted_CPU_reader', 'successor_CPU_reader',
        'original_CPU_failure', 'execution', 'allowed_change'} and
        consumer['inputs'] == 'ORIGINAL_FINITE_GPU_ARTIFACTS_READ_ONLY' and consumer['original_spec'] == q['spec'] and
        consumer['admitted_CPU_reader'] == q['admitted_reader_source'] and consumer['successor_CPU_reader'] == q['reader_source'] and
        consumer['original_CPU_failure'] == q['original_CPU_failure'] and
        consumer['execution'] == dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices=''),
        'Separately bound original-artifact CPU successor consumer differs')
    admission = read(bound(q['GPU_admission']))
    require(raw['admission_sha256'] == q['GPU_admission']['sha256'] and admission['driver_sha256'] == OWNER_SHA and
        admission['phase'] == q['phase'] and admission['automatic_retry'] is False and
        Path(previous['GPU_admission_file']).resolve(strict=True) == Path(q['GPU_admission']['path']).resolve(strict=True) and
        admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'] and
        len(previous['sources']) == contract['previous_control_original_admission_SOURCE_count'] == 101 and
        q['admitted_reader_source'] in previous['sources'] and q['reader_source'] not in previous['sources'] and
        q['reader_source']['path'] not in admission['bound_source_sha256'], 'Original101 admission cannot retroactively admit v2')
    for row in previous['sources']:
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Every original101 owner binding must be retained')
        bound(row)
    for role, row in contract['base_role_SOURCE_refs'].items():
        if role not in ('entry_source', 'CPU_reader'):
            require(previous[role] == spec[role] == row, 'Preserved SOURCE role differs: ' + role)
    for role in ('checkpointmap', 'qualificationmap', 'teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec',
        'teacher_receipt', 'teacher_phase', 'preflight_CPU_qualification', 'preflight_training_spec', 'preflight_receipt',
        'preflight_phase', 'short2_CPU_qualification', 'short2_training_spec', 'short2_receipt', 'short2_phase',
        'prior_DEV_spec', 'prior_DEV_CPU_qualification', 'prior_DEV_receipt', 'prior_DEV_phase', 'FIT_annotation', 'DEV_annotation', 'image_root'):
        require(previous[role] == spec[role], 'Original qualified ancestry/input differs: ' + role)
    for role in ('source', 'reader_source', 'admitted_reader_source', 'report', 'start', 'GPU_qualification', 'owner_driver', 'original_CPU_failure'):
        bound(q[role])
    require(q['owner_driver']['sha256'] == OWNER_SHA, 'Original qualified owner differs')
    return load_source(spec['previous_control_entry_source'], 'sealed_shared_control_helpers'), previous


def fixture_gate(spec, contract):
    sources = dict(producer=spec['entry_source'], fixture=spec['fullFIT_CPU_fixture_source'],
        contract=spec['SOURCE_contract'], pool_source=spec['pool_source'])
    execution = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices='')
    for role in ('fullFIT_CPU_fixture', 'fullFIT_CPU_fixture_spec', 'fullFIT_CPU_fixture_source', 'pool_source'):
        require(spec[role] in spec['sources'], 'Current actual fixture/pool SOURCE absent from owner closure')
    manifest, fixture = read(bound(spec['fullFIT_CPU_fixture_spec'])), read(bound(spec['fullFIT_CPU_fixture']))
    require(manifest == dict(schema=contract['fixture_schema'], inputs='SOURCE_ONLY', sources=sources, execution=execution) and
        fixture['schema'] == contract['fixture_schema'] and fixture['status'] == contract['fixture_status'] and
        fixture['sources'] == sources and fixture['execution'] == execution and
        Path(fixture['spec_path']).resolve(strict=True) == Path(spec['fullFIT_CPU_fixture_spec']['path']).resolve(strict=True) and
        fixture['synthetic_execution_performed'] is True and fixture['torch_version'] == '2.0.1+cu118' and
        fixture['helper_SOURCE_refs'] == contract['helper_SOURCE_refs'] and fixture['library_SOURCE_refs'] == contract['library_SOURCE_refs'],
        'Actual CPU fixture PASS/manifest/source/helper/library joins differ')
    for flag in ('independent_scalar_BCE_and_coefficient_gradients', 'duplicate_12544_indices_preserved',
        'same_expected_full_BCE_enumeration', 'per_GT_then_image_then_accum2_normalization', 'private_CPU_sampling_RNG_restored',
        'connected_empty_GT', 'connected_empty_gain', 'fixed_assignment_raw_ann_mapping_and_rejections',
        'raw_cold_warm_two_weight_gradients', 'frozen_F_e0_unchanged_and_no_grad', 'gain_after_original_renderer_gradient',
        'positive_gain_binary_examples', 'quality_score_ranking_change', 'empty_prediction_quality_zero', 'external_CPU_RNG_unchanged',
        'pooled_same_RAW_initialization', 'pooled_cold_warm_two_weight_gradients', 'pair_mean_before_MLP_nonlinear_control',
        'independent_scalar_FG_BG_pooling', 'pool_valid_padding_and_denominator_clamp', 'pooled_connected_zero_support',
        'pooled_native100_query_axis', 'pooled_no_extra_parameters_buffers_or_head_bias'):
        require(fixture[flag] is True, 'Actual synthetic fixture check absent: ' + flag)
    require(fixture['CUDA_initialized'] is False and fixture['native_model_integration_qualified'] is False and
        fixture['Hungarian_optimality_qualified'] is False and fixture['AP_computed'] is False, 'Synthetic scope differs')
    return fixture


def library_paths(shared, torch, model, spec):
    for obj, role in ((torch.version, 'torch_version_source'), (torch.nn.functional.interpolate, 'interpolation_SOURCE'),
        (torch.optim.AdamW, 'adamw_SOURCE'), (torch.nn.utils.clip_grad_norm_, 'clip_grad_SOURCE'),
        (type(model.panoptic_head), 'native_mask_head_SOURCE'), (type(model.panoptic_fusion_head), 'native_fusion_SOURCE')):
        shared.source_file(obj, spec[role])
    head = model.panoptic_head
    for obj, role in ((head.preprocess_gt, 'native_parent_head_SOURCE'), (head.get_targets, 'native_parent_head_SOURCE'),
        (head._get_targets_single, 'native_mask_head_SOURCE'), (type(head.assigner), 'hungarian_SOURCE'),
        (type(head.sampler), 'mask_sampler_SOURCE'),
        (inspect.getmodule(type(head.sampler)).MaskSamplingResult, 'mask_sampling_result_SOURCE'),
        (inspect.getmodule(type(head)).point_sample, 'point_sample_SOURCE'),
        (inspect.getmodule(head.preprocess_gt).preprocess_panoptic_gt, 'panoptic_GT_SOURCE')):
        shared.source_file(obj, spec[role])
    require(head.num_points == 12544 and
        [type(c).__name__ for c in head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [float(c.weight) for c in head.assigner.match_costs] == [2.0, 5.0, 5.0], 'Actual teacher matcher recipe differs')
    for cost in head.assigner.match_costs:
        shared.source_file(type(cost), spec['match_cost_SOURCE'])


def tensor_diagnostics(torch, pref, result):
    return None if result is None else {key: pref.digest_tensor(v) if torch.is_tensor(v) else v
        for key, v in result['diagnostics'].items()}


def raw_forward(torch, shared, module, fields, valid, observe):
    if not observe:
        return module(fields['F'], fields['e0'], valid, native_logits=fields['M200']), None
    with shared.observed_RAW_inputs(module) as chunks:
        result = module(fields['F'], fields['e0'], valid, native_logits=fields['M200'])
    h, E = torch.cat(chunks, dim=1), result['diagnostics']['eligible'].detach()
    require(tuple(h.shape) == (1, 100, 16, 512) and not h.requires_grad and bool(torch.isfinite(h).all()) and
        tuple(E.shape) == (1, 100, 16) and E.dtype == torch.bool, 'Actual RAW input/context dimensions differ')
    return result, dict(h=h, eligible=E)


def render_selected(torch, diag, masks, positive, identity):
    return (diag.render_original(torch, masks, identity).index_select(0, positive) if positive.numel() == 0
        else diag.render_original(torch, masks.index_select(1, positive), identity))


def gradient_digests(torch, pref, module):
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()), 'Present finite pending gradient required')
    return {name: pref.digest_tensor(p.grad) for name, p in module.named_parameters()}


def execute(torch, np, masks_api, shared, core, pref, old, native, diag, base, hooks, model, anchor, loading,
        modules, optimizers, spec, spec_ref, contract, datasets, cocos, output, started):
    deadline = lambda: require(time.monotonic() - started < 5400, 'Fixed deadline exceeded; no retry')
    seed = spec['readout_seed']
    native_names = list(dict(model.named_parameters()))
    BN = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    BN_affine = [name + '.' + k for name, m in BN for k, p in m.named_parameters(recurse=False)]
    BN_buffers = [name + '.' + k for name, m in BN for k, p in m.named_buffers(recurse=False)]
    require(len(native_names) == 451 and len(BN) == 53 and len(BN_affine) == 106 and len(BN_buffers) == 159, 'Typed frozen teacher population differs')
    initial = shared.arm_states(torch, modules)
    model.to('cuda:0').eval()
    order_generator = torch.Generator(device='cpu').manual_seed(seed['order_seed'])
    trace, orders, matches, audits, epoch_summaries = [], [], {}, {}, []
    updates = {arm: 0 for arm in ARMS}
    step_counts = {arm: {name: 0 for name, p in m.named_parameters()} for arm, m in modules.items()}
    boundary = {}
    expected_train = {r['image_id']: r for r in contract['split']['train_rows']}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    for epoch in range(4):
        prior_updates = dict(updates)
        order_before = order_generator.get_state().clone()
        order = [contract['split']['train_ids'][i] for i in torch.randperm(741, generator=order_generator).tolist()]
        orders.append(dict(epoch=epoch, readout_seed=seed, image_ids=order,
            RNG_before=pref.digest_tensor(order_before), RNG_after=pref.digest_tensor(order_generator.get_state())))
        for offset, image_id in enumerate(order):
            deadline()
            micro, detailed = epoch * 741 + offset, epoch * 741 + offset in DETAIL
            boundary_before = None
            if micro in (741, 2223):
                boundary_before = {arm: gradient_digests(torch, pref, module) for arm, module in modules.items()}
                require(boundary_before == boundary, 'Odd epoch boundary discarded/changed actual pending gradients')
            data, samples, identity = core.image_data(native, old, model, datasets['train'], cocos['train'], image_id)
            require(all(not any(k in sample for k in core.GT_FIELDS) for sample in samples), 'GT entered native neural forward')
            fields, forward, protected = shared.teacher_forward(torch, pref, core, model, data, samples, image_id, detailed)
            identity['metainfo'] = dict(img_shape=list(identity['actual_img_shape']), batch_input_shape=[800, 800], ori_shape=list(identity['original_shape']))
            targets, processed = shared.original_targets(torch, np, cocos['train'], identity, expected_train[image_id], data, model.panoptic_head)
            key = str(image_id)
            if key not in matches:
                matches[key] = shared.fixed_matching(torch, pref, core, old, model.panoptic_head, fields, processed, identity)
            match = matches[key]
            require(match['rows'] == fixed_assignment_rows(identity['ann_ids'], match['positive_query_indices'], match['assigned_GT_indices']),
                'Fixed matching cache lost raw GT row identity')
            positive = torch.tensor(match['query_by_GT'], dtype=torch.long, device='cuda:0')
            valid = hooks.metadata_valid_fraction(torch, [samples[0].metainfo], fields['F'])
            ambient = torch.get_rng_state().clone()
            generator = torch.Generator(device='cpu').manual_seed(seed['integer_pixel_seed_base'] + micro)
            indices_CPU = torch.randint(targets.shape[1] * targets.shape[2], (len(targets), 12544), generator=generator, dtype=torch.int64)
            indices = indices_CPU.to('cuda:0')
            require(torch.equal(ambient, torch.get_rng_state()), 'Private integer sampling altered ambient CPU RNG')
            audit = dict(schema='shared-readout-objective-control-fullFIT4-audit-v1', global_microstep=micro, epoch=epoch,
                readout_seed=seed, image_id=image_id, identity=identity, matching=match, teacher=fields, valid_fraction=valid,
                original_targets=targets, indices=indices_CPU, RAWstates_before=shared.arm_states(torch, modules),
                optimizer_states_before={a: shared.cpu_copy(torch, o.state_dict()) for a, o in optimizers.items()}, arms={}) if detailed else None
            row = dict(global_microstep=micro, epoch=epoch, offset=offset, readout_seed=seed, image_id=image_id, identity=identity,
                raw_GT=len(targets), input_digest=pref.digest_tensor(data['inputs']), forward=forward, matching_image_id=image_id,
                matching_reused=epoch > 0, indices_digest=pref.digest_tensor(indices_CPU),
                integer_pixel_seed=seed['integer_pixel_seed_base'] + micro, integer_pixel_draws=len(targets) * 12544,
                sampled_with_replacement=True, shared_native_fields=shared.records(pref, fields) if detailed else None,
                teacher_all451_grad_None=True, all_teacher_modules_eval=True, pending_accumulation_before=micro % 2,
                pending_accumulation_after=(micro + 1) % 2, arms={})
            with torch.no_grad():
                native_original = render_selected(torch, diag, fields['M200'], positive, identity)
            context = None
            for arm in ARMS:
                module, optimizer = modules[arm], optimizers[arm]
                before = shared.records(pref, module.state_dict())
                pending = {n: None if p.grad is None else shared.cpu_copy(torch, p.grad) for n, p in module.named_parameters()} if detailed else None
                result = None
                if arm == ARMS[-1]:
                    original = score_gain(torch, native_original, module.log_gain)
                else:
                    if arm in ARMS[:2]:
                        result, observed = raw_forward(torch, shared, module, fields, valid, arm == ARMS[0] or detailed)
                        if arm == ARMS[0]:
                            context = observed
                            if detailed:
                                audit['RAW_context'] = shared.cpu_copy(torch, dict(context, activation='silu', actual_hidden_pre_hook_observed=True))
                        elif detailed:
                            require(torch.equal(observed['h'], context['h']) and torch.equal(observed['eligible'], context['eligible']),
                                'RAW full/uniform actual contexts differ')
                        del observed
                    else:
                        result = module(fields['F'], fields['e0'], valid, native_logits=fields['M200'],
                            context=context if arm == 'pair_pooled_full_BCE' else None)
                    original = render_selected(torch, diag, result['mask_logits'], positive, identity)
                used = indices if arm == ARMS[0] else None
                loss = per_image_BCE(torch, original, targets, used)
                require(loss.requires_grad and bool(torch.isfinite(loss)), 'Finite live final-only readout BCE required')
                micro_grad = torch.autograd.grad(loss / 2.0, list(module.parameters()), retain_graph=True) if detailed else None
                (loss / 2.0).backward()
                gradients = shared.gradient_records(torch, pref, module)
                accumulated = {n: shared.cpu_copy(torch, p.grad) for n, p in module.named_parameters()} if detailed else None
                step, norm, clipped = (micro + 1) % 2 == 0, None, None
                if step:
                    norm = float(torch.nn.utils.clip_grad_norm_(list(module.parameters()), .01, error_if_nonfinite=True).item())
                    require(math.isfinite(norm) and norm >= 0, 'Finite clipping norm required')
                    clipped = shared.gradient_records(torch, pref, module)
                    for name, p in module.named_parameters():
                        step_counts[arm][name] += int(p.grad is not None)
                    optimizer.step()
                    updates[arm] += 1
                per_GT = targets.new_empty((0,)) if len(targets) == 0 else torch.nn.functional.binary_cross_entropy_with_logits(
                    original.reshape(len(targets), -1).gather(1, indices) if used is not None else original.reshape(len(targets), -1),
                    targets.reshape(len(targets), -1).gather(1, indices) if used is not None else targets.reshape(len(targets), -1), reduction='none').mean(1)
                after_boundary = gradient_digests(torch, pref, module) if micro in (740, 2222) else None
                row['arms'][arm] = dict(loss=float(loss.item()), per_GT_loss=per_GT.detach().cpu().tolist(),
                    supervision='uniform_original_integer_pixels' if used is not None else 'full_original_raster',
                    pixels_per_GT=12544 if used is not None else int(targets.shape[1] * targets.shape[2]),
                    draw_count=len(targets) * (12544 if used is not None else int(targets.shape[1] * targets.shape[2])),
                    parameter_before=before, accumulated_gradients_before_clip=gradients, clipped_gradients=clipped,
                    clip_norm_before=norm, optimizer_update=step, updates_completed=updates[arm],
                    parameter_after=shared.records(pref, module.state_dict()), readout_diagnostics=tensor_diagnostics(torch, pref, result),
                    cold_hidden_gradient_exact_zero=bool(torch.count_nonzero(module.hidden.weight.grad) == 0) if micro < 2 and arm != ARMS[-1] else None,
                    epoch_boundary_gradient_before=None if boundary_before is None else boundary_before[arm],
                    epoch_boundary_gradient_after=after_boundary)
                if detailed:
                    audit['arms'][arm] = shared.cpu_copy(torch, dict(original_logits=original, pending_gradients_before=pending,
                        micro_gradients={n: g for (n, p), g in zip(module.named_parameters(), micro_grad)},
                        accumulated_gradients_before_clip=accumulated,
                        clipped_gradients=None if not step else {n: p.grad for n, p in module.named_parameters()},
                        readout_result=result, parameters_after=module.state_dict(), optimizer_state_after=optimizer.state_dict()))
                if step:
                    optimizer.zero_grad(set_to_none=True)
                del original, loss, per_GT, result, micro_grad, pending, accumulated
            require(all(p.grad is None and not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules()) and
                [v._version for v in protected['tensors']] == protected['versions'], 'Frozen native parameters/eval/stage versions changed')
            forward['native_all10_tensor_versions_unchanged'] = True
            if micro in (740, 2222):
                boundary = {arm: row['arms'][arm]['epoch_boundary_gradient_after'] for arm in ARMS}
            if detailed:
                audits[str(micro)] = shared.packet(torch, output / ('AUDIT_%04d.pth' % micro), audit)
            trace.append(row)
            del data, samples, targets, processed, fields, protected, valid, native_original, indices, indices_CPU, audit, context
            if (micro + 1) % 64 == 0:
                print(json.dumps(dict(seed_id=seed['seed_id'], completed_microsteps=micro + 1, updates_per_arm=updates)), flush=True)
        expected_updates, cumulative, pending_count = (370, 371, 370, 371)[epoch], (370, 741, 1111, 1482)[epoch], (1, 0, 1, 0)[epoch]
        require(all(updates[a] - prior_updates[a] == expected_updates and updates[a] == cumulative for a in ARMS) and
            all((p.grad is not None) == bool(pending_count) for m in modules.values() for p in m.parameters()), 'Global accumulation epoch accounting differs')
        epoch_summaries.append(dict(epoch=epoch, microsteps=741, optimizer_updates=expected_updates,
            cumulative_optimizer_updates=cumulative, pending_accumulation=pending_count))
    require(len(trace) == 2964 and len(matches) == 741 and updates == {a: 1482 for a in ARMS} and
        all(p.grad is None for m in modules.values() for p in m.parameters()), 'Exact global E4 counters/no pending gradients required')
    checkpoints = {}
    for arm, module in modules.items():
        optimizer = optimizers[arm]
        names = [[n] for n, p in module.named_parameters()]
        require(len(optimizer.state) == len(names) and all(float(v['step'].item()) == 1482 for v in optimizer.state.values()), 'Actual readout AdamW step1482 required')
        checkpoints[arm] = shared.packet(torch, output / (arm + '_E4.pth'), dict(schema='shared-readout-objective-control-fullFIT4-checkpoint-v1',
            readout_state_dict=module.state_dict(), optimizer_state_dict=optimizer.state_dict(), meta=dict(arm=arm, phase=spec['phase'],
                spec=spec_ref, source=spec['entry_source'], SOURCE_contract=spec['SOURCE_contract'], readout_seed=seed,
                completed_epochs=4, microsteps=2964, optimizer_updates=1482, raw_GT_visits=6644, optimizer_param_names=names,
                teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'], teacher_mode='native0',
                teacher_epoch=24, frozen_native=True, final_only_BCE=True, formal_resumable=False, teacher_optimizer_loaded=False,
                teacher_order_RNG_loaded=False, preflight_or_SHORT2_CP_loaded=False,
                gain_after_original_renderer=arm == ARMS[-1], capacity_or_optimizer_matched=arm != ARMS[-1]),
            order_generator_state=order_generator.get_state(), fixed_assignment_rows=matches))
    for name, value in (('TRACE', trace), ('ORDER', orders), ('MATCHING', matches)):
        write(output / (name + '.json'), value)
    endpoint = {mode: dict(predictions=[], score_factors=[], image_rows=[]) for mode in ENDPOINTS}
    evaluation_matches = {}
    for expected in contract['split']['evaluation_rows']:
        deadline()
        image_id = expected['image_id']
        data, samples, identity = core.image_data(native, old, model, datasets['evaluation'], cocos['evaluation'], image_id)
        fields, forward, protected = shared.teacher_forward(torch, pref, core, model, data, samples, image_id)
        identity['metainfo'] = dict(img_shape=list(identity['actual_img_shape']), batch_input_shape=[800, 800], ori_shape=list(identity['original_shape']))
        targets, processed = shared.original_targets(torch, np, cocos['evaluation'], identity, expected, data, model.panoptic_head)
        evaluation_matches[str(image_id)] = shared.fixed_matching(torch, pref, core, old, model.panoptic_head, fields, processed, identity)
        valid = hooks.metadata_valid_fraction(torch, [samples[0].metainfo], fields['F'])
        with torch.no_grad():
            native_original = diag.render_original(torch, fields['M200'], identity)
            native_predictions, native_factors, native_qids, _, _ = base.official_predictions(torch, np, masks_api, model,
                fields['classes'], fields['M200'], samples, identity, pref)
            context = None
            for mode in ENDPOINTS:
                result = None
                if mode == 'native0':
                    original = native_original
                elif mode == ARMS[-1]:
                    original = score_gain(torch, native_original, modules[mode].log_gain)
                    require(torch.equal(original > 0, native_original > 0), 'Actual gain binary agreement required; no universal FP32 claim')
                else:
                    if mode in ARMS[:2]:
                        result, observed = raw_forward(torch, shared, modules[mode], fields, valid, mode == ARMS[0])
                        if mode == ARMS[0]:
                            context = observed
                        del observed
                    else:
                        result = modules[mode](fields['F'], fields['e0'], valid, native_logits=fields['M200'],
                            context=context if mode == 'pair_pooled_full_BCE' else None)
                    original = diag.render_original(torch, result['mask_logits'], identity)
                predictions, factors, qids, class_packet = shared.export_original(torch, np, masks_api, model.panoptic_fusion_head,
                    fields['classes'], original, image_id, pref)
                if mode == 'native0':
                    require(predictions == native_predictions and factors == native_factors and qids == native_qids, 'Actual official native all100 export differs')
                endpoint[mode]['predictions'].extend(predictions)
                endpoint[mode]['score_factors'].extend(factors)
                endpoint[mode]['image_rows'].append(dict(mode=mode, readout_seed=seed, image_id=image_id, identity=identity,
                    forward=forward, GT=pref.digest_tensor(processed.masks), input_digest=pref.digest_tensor(data['inputs']),
                    native_fields=shared.records(pref, fields), native_class_packet=class_packet, query_ids=qids,
                    original_logits=pref.digest_tensor(original), masks200=pref.digest_tensor(fields['M200'] if result is None else result['mask_logits']),
                    official_fusion_mask_score_query_TorchEqual=True, all100_queries_retained=True, GT_free_forward=True,
                    gain=float(modules[mode].log_gain.exp().item()) if mode == ARMS[-1] else None,
                    gain_binary_mask_agreement=True if mode == ARMS[-1] else None, readout_diagnostics=tensor_diagnostics(torch, pref, result)))
                del original, result
        require([v._version for v in protected['tensors']] == protected['versions'], 'DEV native stage tensor versions changed')
        forward['native_all10_tensor_versions_unchanged'] = True
        del data, samples, targets, processed, fields, protected, valid, native_original, context
    write(output / 'HELDOUT_MATCHING.json', evaluation_matches)
    mode_rows = {}
    for mode, values in endpoint.items():
        require(len(values['image_rows']) == 187 and len(values['predictions']) == len(values['score_factors']) == 18700, 'All187 DEV/all100 population differs')
        row = {}
        for key, value in values.items():
            path = output / (mode + '_' + key.upper() + '.json')
            write(path, value)
            row[key] = reference(path)
        row['checkpoint'] = spec['teacher_checkpoint'] if mode == 'native0' else checkpoints[mode]
        mode_rows[mode] = row
    model.cpu()
    after = model.state_dict()
    require(set(after) == set(anchor) and all(torch.equal(after[k], anchor[k]) for k in anchor), 'All native610 must remain CPU exact')
    proof = dict(schema='shared-readout-objective-control-fullFIT4-state-proof-v1', readout_seed=seed,
        native_teacher610=shared.records(pref, anchor), before=shared.records(pref, anchor), after=shared.records(pref, after),
        teacher_loading=loading, native_parameter_names=native_names, BN_module_names=[n for n, m in BN],
        BN_parameter_names=BN_affine, BN_buffer_names=BN_buffers, full_native610_CPU_exact=True, frozen_BN_all_exact=True,
        all_native451_frozen=True, all_modules_eval=True, native_parameter_grads_None=True, hook_cache_clear=True, GT_free_forward=True,
        teacher_forward_calls=3151, teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, feature_cache_used=False,
        initial_readout_states=shared.records(pref, initial[ARMS[0]]), initial_gain_state=shared.records(pref, initial[ARMS[-1]]),
        all_four_matrix_initializations_CPU_exact=True, final_readout_states={a: shared.records(pref, m.state_dict()) for a, m in modules.items()},
        fresh_optimizer_state_entries={a: 0 for a in ARMS}, optimizer_step_nonNone_gradient_counts=step_counts,
        completed_epochs=4, microsteps=2964, optimizer_updates=updates, no_pending_accumulation_gradients=True,
        epoch_summaries=epoch_summaries, odd_epoch_boundaries_actual_pending_gradients_preserved=True,
        audit_microsteps=list(DETAIL), audit_refs=audits, training_raw_GT=1661, training_raw_GT_visits_per_arm=6644,
        evaluation_raw_GT=380, DEV_historically_exposed=True, blind_holdout_claimed=False)
    write(output / 'STATE_PROOF.json', proof)
    for module in modules.values():
        module.cpu()
    return dict(mode_rows=mode_rows, checkpoints=checkpoints, trace=reference(output / 'TRACE.json'), order=reference(output / 'ORDER.json'),
        matching=reference(output / 'MATCHING.json'), heldout_matching=reference(output / 'HELDOUT_MATCHING.json'),
        state_proof=reference(output / 'STATE_PROOF.json'), audits=audits)


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)  # Metadata only until all six actual prior terminals.
    raw = {}
    for receipt_role, phase_role in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
        ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'), ('prior_diag_receipt', 'prior_diag_phase'),
        ('previous_control_receipt', 'previous_control_phase')):
        require(Path(spec[receipt_role]['path']).is_absolute(), 'Absolute original raw receipt required')
        raw[receipt_role] = read(spec[receipt_role]['path'])
        raw_terminal(raw[receipt_role], spec[phase_role])
    require(spec['schema'] == SCHEMA and spec['arms'] == list(ARMS) and spec['current_endpoints'] == list(ENDPOINTS) and
        spec['epochs'] == 4 and spec['microsteps'] == 2964 and spec['optimizer_updates'] == 1482 and spec['accumulation'] == 2 and
        spec['raw_GT_visits_per_arm'] == 6644 and spec['no_feature_cache'] is True and spec['expected_epoch'] == 24 and
        spec['CPU_affinity'] == [0, 1] and spec['deadline_seconds'] == 5400 and spec['allocator_limit_MiB'] == 4096 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Current numerical/resource scope differs')
    for key, value in (('total_microsteps', 2964), ('total_optimizer_updates', 1482)):
        if key in spec:
            require(spec[key] == value, 'Legacy current counter differs: ' + key)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['GPU_admission_file']).is_absolute(), 'Fresh absolute output/admission required')
    spec_ref = owner_gate(spec_path, spec)
    contract = read(bound(spec['SOURCE_contract']))
    require(contract['kind'] == 'ROOT_INERT_PRELOCKED_FULLFIT4_SOURCE_CONTRACT_NOT_SCIENCE_ADMISSION' and
        contract['schema'] == SCHEMA and contract['arms'] == list(ARMS) and contract['current_endpoints'] == list(ENDPOINTS) and
        contract['detail_microsteps'] == list(DETAIL) and spec['readout_seed'] in contract['readout_seeds'] and
        spec['phase'].startswith(contract['phase_prefix'] + str(spec['readout_seed']['seed_id']) + '_') and Path(spec['phase']).name == spec['phase'] and
        spec['FIT_annotation'] == contract['split']['train_annotation_reference'] and spec['DEV_annotation'] == contract['split']['evaluation_annotation_reference'],
        'Canonical full FIT/exposed DEV/one prelocked seed contract differs')
    for key in ('epochs', 'microsteps', 'optimizer_updates', 'raw_GT_visits_per_arm', 'accumulation'):
        require(spec[key] == contract[key], 'Authoritative protocol counter differs')
    for role, row in contract['library_SOURCE_refs'].items():
        require(spec[role] == row and row in spec['sources'], 'Actual library SOURCE closure differs')
    fixture_gate(spec, contract)
    shared, previous = previous_control_gate(spec, contract, raw['previous_control_receipt'])
    shared.fixture_gate(previous, read(bound(spec['previous_control_contract'])))
    diag, prior_diag = shared.prior_gate(spec, contract, raw['prior_diag_receipt'])
    base, previous_DEV_q = diag.prior_DEV_gate(spec, raw['prior_DEV_receipt'], read(bound(spec['prior_diag_contract'])))
    short = load_source(spec['short2_source'], 'sealed_short2_ancestry')
    short.preflight_gate(spec, raw['preflight_receipt'])
    base.qualified_short2(spec, raw['short2_receipt'])
    core = load_source(spec['preflight_source'], 'sealed_native_teacher_ancestry')
    teacher_row = core.qualified_teacher(spec, raw['teacher_receipt'])
    core.fixture_gate(spec)
    output.mkdir(parents=True)
    joins = {role: spec[role] for role in ('entry_source', 'CPU_reader', 'SOURCE_contract', 'teacher_checkpoint', 'teacher_qualification',
        'previous_control_spec', 'previous_control_CPU_qualification', 'previous_control_receipt', 'previous_control_CPU_consumer_manifest',
        'previous_control_CPU_reader_source', 'previous_control_admitted_CPU_reader_source', 'previous_control_CPU_failure',
        'fullFIT_CPU_fixture', 'fullFIT_CPU_fixture_spec', 'fullFIT_CPU_fixture_source', 'pool_source')}
    write(output / 'START.json', dict(schema='shared-readout-objective-control-fullFIT4-start-v1', phase=spec['phase'], spec=spec_ref,
        readout_seed=spec['readout_seed'], arms=list(ARMS), current_endpoints=list(ENDPOINTS), joins=joins,
        original_six_raw_gates_before_any_hash_or_import=True, shared_teacher_forward=True, feature_cache_used=False,
        fixed_matching_cache_only=True, epochs=4, microsteps=2964, updates=1482, accumulation=2, epoch_flush=False,
        DEV_historically_exposed=True, blind_holdout_claimed=False, no_TEST_access=True, no_automatic_retry=True))
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
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'Actual CPU01/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.__version__ == '2.0.1+cu118' and torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Qualified Torch/threads differ')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'fullFIT_tensor_helpers')
    initializer = load_source(spec['initializer_source'], 'fullFIT_original_initializer')
    old = load_source(spec['native_helpers'], 'fullFIT_actual_target_observers'); old.torch = torch
    native = load_source(spec['native_preflight_helpers'], 'fullFIT_GT_free_data_loader')
    official = load_source(spec['official_evaluation_source'], 'fullFIT_native610_teacher_loader')
    readout = load_source(spec['readout_source'], 'fullFIT_exact_RAW_operator')
    geometry = load_source(spec['geometry_source'], 'fullFIT_exact_RAW_geometry')
    hooks = load_source(spec['hooks_source'], 'fullFIT_metadata_valid_fraction')
    pool = load_source(spec['pool_source'], 'fullFIT_exact_pooling_controls')
    model, state, meta, loading = official.prepare_model(torch, pref, initializer, None, None, None, MODELS,
        revert_sync_batchnorm, Config, tuple(CocoDataset.METAINFO['classes']), spec, teacher_row, 'native0')
    model.requires_grad_(False).eval()
    anchor = shared.cpu_copy(torch, model.state_dict())
    library_paths(shared, torch, model, spec)
    seed = spec['readout_seed']
    modules = {a: readout.build_adapter(torch, geometry, mode='raw', hidden_seed=seed['hidden_seed'],
        permutation_seed=seed['permutation_seed']) for a in ARMS[:2]}
    modules['pair_pooled_full_BCE'] = pool.build_adapter(torch, kind='pair_pooled', hidden_seed=seed['hidden_seed'])
    modules['mask_pooled_full_BCE'] = pool.build_adapter(torch, kind='mask_pooled', hidden_seed=seed['hidden_seed'])
    gain = torch.nn.Module()
    gain.register_parameter('log_gain', torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device='cpu')))
    modules[ARMS[-1]] = gain
    cold = shared.arm_states(torch, modules)
    require(len(anchor) == 610 and len(list(model.parameters())) == 451 and not torch.cuda.is_initialized() and
        all(len(modules[a].state_dict()) == 2 and not list(modules[a].buffers()) and sum(p.numel() for p in modules[a].parameters()) == 393216 and
            all(torch.equal(cold[a][k], cold[ARMS[0]][k]) for k in cold[ARMS[0]]) for a in ARMS[:4]),
        'All four CPU matrix arms must be fresh bitexact common initialization before CUDA')
    datasets, cocos = {}, {}
    for name, role, images, instances in (('train', 'FIT_annotation', 741, 1661), ('evaluation', 'DEV_annotation', 187, 380)):
        datasets[name] = CocoDataset(ann_file=str(bound(spec[role])), data_prefix=dict(img=spec['image_root']),
            metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
        cocos[name] = COCO(str(bound(spec[role])))
        require(len(datasets[name]) == len(cocos[name].imgs) == images and len(cocos[name].anns) == instances, 'Whole fixed population differs')
        for expected in contract['split'][name + '_rows']:
            require(int(datasets[name].get_data_info(expected['dataset_index'])['img_id']) == expected['image_id'] and
                cocos[name].get_ann_ids(img_ids=[expected['image_id']]) == expected['annotation_ids'], 'Original raw ID/order population differs')
    require(not torch.cuda.is_initialized(), 'Complete teacher/readout/data CPU preparation must precede CUDA')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096 * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory, 0)
    for module in modules.values():
        module.to('cuda:0')
    optimizers = {a: torch.optim.AdamW([dict(params=[p]) for n, p in m.named_parameters()],
        lr=.01 if a == ARMS[-1] else .0001, weight_decay=.05, betas=(.9, .999), eps=1e-8) for a, m in modules.items()}
    require(all(len(o.state) == 0 for o in optimizers.values()), 'Fresh independent readout-only optimizers required')
    result = execute(torch, np, masks_api, shared, core, pref, old, native, diag, base, hooks, model, anchor, loading,
        modules, optimizers, spec, spec_ref, contract, datasets, cocos, output, started)
    del model, modules, optimizers, state, anchor, cold
    gc.collect()
    cleanup = pref.clear_gpu(torch)
    require(cleanup['allocated_after'] == 0 and time.monotonic() - started < 5400, 'Actual final cleanup/deadline differs')
    report = dict(schema=contract['report_schema'], status=contract['report_status'], phase=spec['phase'], source=spec['entry_source'],
        spec=spec_ref, SOURCE_contract=spec['SOURCE_contract'], readout_seed=seed, joins=joins, arms=list(ARMS), current_endpoints=list(ENDPOINTS),
        result=result, error=None, epochs=4, microsteps=2964, optimizer_updates_per_arm=1482, accumulation=2, epoch_flush=False,
        train_images=741, train_raw_GT=1661, raw_GT_visits_per_arm=6644, evaluation_images=187, evaluation_raw_GT=380,
        teacher_forward_calls=3151, shared_teacher_forward=True, full_native610_and_BN_exact=True, all_native451_frozen=True,
        GT_free_forward=True, feature_cache_used=False, fixed_assignment_cache_only=True, final_only_BCE=True,
        no_Dice_training_loss=True, no_unmatched_mask_loss=True, no_TEST_access=True, DEV_historically_exposed=True,
        blind_holdout_claimed=False, official_all100_endpoint_exports=True, AP_computed_by_GPU_producer=False,
        AP_pending_independent_CPU_reader=True, gain_location='after_original_renderer_before_instance_postprocess',
        gain_capacity_or_optimizer_matched=False, GPU_optimizer_bitwise_determinism_claimed=False,
        native_additional_training_matched=False, complete_KNet_reproduction_claimed=False, compute_matched_claimed=False,
        full_empty_image_model_integration_qualified=False, efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False,
        SOTA_claimed=False, previous_SHORT2_causal_explanation_claimed=False, cleanup=cleanup,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        required_SPEC_numerical_scope=dict(epochs=4, microsteps=2964, optimizer_updates=1482, raw_GT_visits_per_arm=6644,
            accumulation=2, no_feature_cache=True, CPU_affinity=[0, 1], allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192, deadline_seconds=5400),
        evidence_limits=contract['nonclaims'] + ['Only four saved microsteps and updates1/1482 permit independent floating audit.',
            'RAW geometry recomputes e0.F in chunks8; no bitexact native100 geometry claim.',
            'Each job repeats readout initialization/order/integer sampling, not the already fixed teacher training.',
            'Only fixed E4 and historically exposed DEV; no convergence, TEST or scientific method verdict.'])
    write(output / 'REPORT.json', report)
    write(output / 'QUALIFIED.json', dict(schema='shared-readout-objective-control-fullFIT4-aggregate-v1',
        status='ROOT_SHARED_READOUT_OBJECTIVE_CONTROL_FULLFIT4_PENDING_CPU_READER', phase=spec['phase'], source=spec['entry_source'],
        spec=spec_ref, readout_seed=seed, report=reference(output / 'REPORT.json'), result=result, independently_CPU_qualified=False, AP_computed=False))
    print(json.dumps(dict(status=contract['report_status'], seed_id=seed['seed_id'], CPU_qualification_pending=True)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Only ROOT-owned admitted execution is permitted')
    run(args.spec)


if __name__ == '__main__':
    main()
