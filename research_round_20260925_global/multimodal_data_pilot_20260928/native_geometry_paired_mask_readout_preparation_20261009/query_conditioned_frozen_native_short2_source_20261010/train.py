"""Inert ROOT-owned four-arm, frozen E24 teacher, full-FIT SHORT2 SOURCE.

All four modes start from the same native0 E24 full610 state plus fresh two
weights. Every native parameter is frozen and every module stays in eval.
Two complete741/1661 FIT epochs use one fresh private CPU order generator;
global accumulation-two crosses the odd741 epoch boundary without a flush.
All thirty original native loss atoms and ten actual assignments remain.
There is no feature cache, final-only objective, scheduler, teacher optimizer
or RNG restoration, preflight-checkpoint resume, DEV, TEST or AP selection.

Compact input/GT/loss/matching/RNG/gradient/hook evidence is retained per micro.
Dense native field digests and full E/RMS packets occur ONLY at first4/last4
global microsteps. A CPU consumer must independently verify the final612
state, native610 preservation, full FIT coverage and actual step741 moments.
SHORT2 is a short-budget feature-hypothesis probe, not an efficacy, novelty,
convergence, SOTA, equal-compute or publication qualification by itself.
SOURCE authoring performed no scientific imports, execution or payload reads.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import types


SCHEMA = 'ROOT-query-conditioned-frozen-native-short2-spec-v1'
REPORT_SCHEMA = 'query-conditioned-frozen-native-short2-report-v1'
CP_SCHEMA = 'query-conditioned-frozen-native-short2-checkpoint-v1'
PROOF_SCHEMA = 'query-conditioned-frozen-native-short2-state-proof-v1'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
FIT_IMAGES, FIT_GT, EPOCHS, MICROSTEPS, UPDATES = 741, 1661, 2, 1482, 741
DETAIL_MICROS = (0, 1, 2, 3, 1478, 1479, 1480, 1481)
PREFIX = 'panoptic_head.query_pair_readout.'
ACTIVE = [PREFIX + name for name in ('hidden.weight', 'final.weight')]
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
PREFLIGHT_SHA = 'e9abe88fae01a4d7f80d374ad2e0bafccb37cd655e3e15dc39e7a64535a1a428'
PREFLIGHT_READER_SHA = '38ad577aa674c7cf4e5fa740b3345b744bf7d920efabbbde9a8ff144ea8b1ddd'
PREFLIGHT_SPEC_SHA = '6ca7bf3202afa8ce34de7627aac26558a7d5b84611d6da1521046fe9a0f76a3c'
PREFLIGHT_Q_SHA = '4471892c213ff097f2e6a55a33dba47333c29debabd3adb854fe0598ba507fc9'
PREFLIGHT_RECEIPT_SHA = '8311ac8017b4fb4c01cd46065eaa4e3b7787480815801dd0769d701ba607a185'
PREFLIGHT_PHASE = 'query_conditioned_frozen_native_preflight_20261010_01'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda: stream.read(1048576), b''):
            digest.update(value)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
            Path(row['path']).is_absolute() and reference(row['path']) == row, 'Absolute pinned input differs')
    return Path(row['path'])


def load_source(row, name):
    path = bound(row)
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'],
            'SOURCE changed before compilation')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def raw_terminal(receipt, phase):
    """Bootstrap RAW gate only: no SOURCE/SPEC/Q/GT/CP hashes or imports."""
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
            type(receipt['returncode']) is int and receipt['returncode'] == 0 and
            receipt['root_wait_reaped'] is True and receipt['held_handle_exit'] ==
            dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            receipt['numeric_birth_absence']['original_birth_absent'] is True and
            receipt['abort'] is None and receipt['error'] is None and receipt['wait_errors'] is False and
            receipt['GPU_observation_errors'] == [], 'STOP before scientific access: actual raw held0/reap/birth/noerror required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Actual prior terminal phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'Actual prior whole-GPU scope differs')


def owner_gate(spec_path, spec):
    admission = read(Path(spec['GPU_admission_file']).resolve(strict=True))
    entry, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] == 10800 and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(entry['path']) and
            admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
            admission['driver_sha256'] == OWNER_SHA, 'Actual current SHORT2 owner/resources/entry differ')
    hashes = admission['bound_source_sha256']
    require(spec['entry_source'] == entry and entry in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
            hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SHORT2 entry/reader/SPEC not owner-bound')
    for source in spec['sources']:
        require(hashes.get(source['path']) == source['sha256'], 'Complete SHORT2 SOURCE closure not owner-bound')
        bound(source)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current scientific SPEC differs')
    return spec_ref


def preflight_gate(spec, receipt):
    """Read qualified metadata, never a technical preflight checkpoint."""
    for role, size, sha in (('preflight_source', 44920, PREFLIGHT_SHA),
                           ('preflight_CPU_reader', 65719, PREFLIGHT_READER_SHA),
                           ('preflight_training_spec', 24062, PREFLIGHT_SPEC_SHA),
                           ('preflight_CPU_qualification', 338510, PREFLIGHT_Q_SHA),
                           ('preflight_receipt', 1073, PREFLIGHT_RECEIPT_SHA)):
        require(spec[role]['bytes'] == size and spec[role]['sha256'] == sha, 'Canonical actual preflight pin differs: ' + role)
        bound(spec[role])
    require(spec['preflight_phase'] == PREFLIGHT_PHASE, 'Canonical preflight owner phase required')
    previous = read(bound(spec['preflight_training_spec']))
    q = read(bound(spec['preflight_CPU_qualification']))
    require(previous['schema'] == 'ROOT-query-conditioned-frozen-native-preflight-spec-v1' and
            previous['phase'] == q['phase'] == PREFLIGHT_PHASE and
            previous['entry_source'] == q['source'] == spec['preflight_source'] and
            previous['CPU_reader'] == q['reader_source'] == spec['preflight_CPU_reader'] and
            q['spec'] == spec['preflight_training_spec'] and q['parent_receipt'] == spec['preflight_receipt'] and
            q['schema'] == 'ROOT-query-conditioned-frozen-native-preflight-CPU-qualification-v1' and
            q['status'] == 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_PREFLIGHT_TECHNICALLY_QUALIFIED' and
            q['modes'] == list(MODES) and q['image_ids'] == previous['image_ids'] == [1, 2, 3, 5] and
            q['actual_trace_records'] == 16 and q['AP_computed'] is False and q['efficacy_claimed'] is False,
            'Actual full four-arm technical CPU qualification/source/SPEC/receipt join differs')
    for key in ('teacher_full610_exact_before_after_every_arm', 'native451_frozen_and_grads_None',
                'typed_BN106_affine159_buffers_independently_exact', 'only_two_weights_changed',
                'every_AdamW_moment_and_actual_step2_verified', 'all_raw_GT_independently_reconstructed',
                'ten_native_assignments_cover_all_raw_GT', 'native30_loss_total_consistent',
                'safe_whole_head_final_only_hooks_source_observed', 'actual_native_runtime_technically_qualified'):
        require(q[key] is True, 'Actual technical preflight qualification assertion missing: ' + key)
    for key in ('teacher_training_spec', 'teacher_qualification', 'teacher_checkpoint', 'teacher_receipt',
                'operator_CPU_fixture', 'FIT_annotation'):
        require(q[key] == previous[key] == spec[key], 'Actual technical preflight input lineage differs: ' + key)
    for key in ('initializer_source', 'micro_preflight_source', 'native_helpers', 'native_preflight_helpers',
                'reused_CPU_reader', 'evaluation_source', 'training_source', 'training_CPU_reader',
                'readout_source', 'geometry_source', 'hooks_source', 'operator_CPU_fixture_source', 'image_root'):
        require(previous[key] == spec[key], 'Qualified reused SOURCE/input role differs: ' + key)
    require(len(previous['sources']) == 52 and all(row in spec['sources'] for row in previous['sources']) and
            spec['preflight_source'] in spec['sources'] and spec['preflight_CPU_reader'] in spec['sources'],
            'Actual 52-source technical closure must remain bound in SHORT2')
    admission_path = bound(q['GPU_admission'])
    admission = read(admission_path)
    require(receipt['admission_sha256'] == q['GPU_admission']['sha256'] and
            Path(previous['GPU_admission_file']).resolve(strict=True) == admission_path.resolve(strict=True) and
            admission['phase'] == PREFLIGHT_PHASE and admission['driver_sha256'] == OWNER_SHA and
            admission['entry_args'] == ['--spec', spec['preflight_training_spec']['path'], '--owner-execute'],
            'Actual technical preflight receipt/admission join differs')
    for key in ('report', 'start', 'GPU_qualification', 'owner_driver'):
        bound(q[key])


def readout_checks(observation, mode):
    counts, canonical = observation['eligible_counts'][0], observation['canonical_eligible_counts'][0]
    eligible, supported = sum(count > 0 for count in counts), sum(observation['support_nonempty'][0])
    require(counts == canonical and len(counts) == 100 and all(type(value) is int and 0 <= value <= 16 for value in counts),
            'Canonical per-query E counts differ')
    require(all(sum(row) == count for row, count in zip(observation['eligible'][0], counts)) and
            all(count == 0 or support for count, support in zip(counts, observation['support_nonempty'][0])),
            'Actual E/count/support compact consistency differs')
    for side in ('inside', 'outside'):
        before = observation['input_rms'][side + '_before'][0]
        after = observation['input_rms'][side + '_after'][0]
        for rows in (before, after):
            require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        for row in rows for value in row), 'Reported input RMS nonfinite/negative')
        if mode == 'raw':
            require(before == after, 'Raw input RMS changed')
        else:
            require(all(value == 0 or abs(value - 1) <= 3e-5 for row in after for value in row),
                    'Normalized reported RMS must be zero or near unit')
    return dict(eligible_pairs_total=sum(counts), eligible_queries=eligible, supported_queries=supported,
        canonical_count_equal=True, E_counts_checked=True, global_coefficient_fields_equal=True,
        zero_count_basis='reported_FP32_input_RMS_equals_zero', auxiliary_query_detached=True,
        all_queries_retained=True, input_RMS_normalized=mode != 'raw',
        input_RMS_finite_nonnegative_checked=True,
        normalized_nonzero_unit_or_zero_checked=(True if mode != 'raw' else None),
        raw_input_RMS_before_after_exact_checked=(True if mode == 'raw' else None),
        hidden_seed=2026100911, permutation_seed=2026101013,
        permutation_kind='seeded_256_cycle_not_all_derangements_uniform', parameters=393216, trainable_tensors=2,
        global_pool=True, dense_readout='same_unnormalized_F', empty_E_connected_zero=True,
        equal_compute_claimed=False, RMS_zero_is_exact_zero_vector_claimed=False)


def train_arm(torch, core, pref, old, native, hooks, teacher_reader, reused_reader,
              spec, spec_ref, teacher_train, prepared, teacher, dataset, coco, output, plan, plan_ref,
              mode, shared_detailed, started):
    model, loading, cold = prepared
    directory = output / (mode + '_training')
    directory.mkdir()
    initial = dict(teacher, **{PREFIX + key: value for key, value in cold.items()})
    before = pref.state_proof(initial)
    model.to('cuda:0')
    policy = core.frozen_policy(torch, model)
    entry = teacher_train['initializers']['mask2former']
    optimizer, audit = pref.make_optimizer(torch, model, entry, policy)
    require(len(optimizer.state) == 0 and len(optimizer.param_groups) == 2 and
            audit['clip_grad'] == dict(max_norm=0.01, norm_type=2) and
            all(group['lr'] == 0.0001 and group['weight_decay'] == 0.05 and
                tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8 for group in optimizer.param_groups),
            'Fresh native AdamW must contain only the two declared query weights')
    head, adapter = model.panoptic_head, model.panoptic_head.query_pair_readout
    named = dict(model.named_parameters())
    hook_baseline = core.hook_inventory(model)
    activity, epochs, updates, GT_visits, warm_both = {name: 0 for name in ACTIVE}, [], 0, 0, False
    step_activity = {name: 0 for name in ACTIVE}
    boundary = None
    trace_path = directory / 'TRACE.jsonl'
    pending_path = directory / 'TRACE.jsonl.pending'
    optimizer.zero_grad(set_to_none=True)
    try:
        with pending_path.open('x', encoding='utf-8', newline='\n') as trace_stream:
            for epoch, order in enumerate(plan['orders'], 1):
                seen, epoch_GT, epoch_updates_before = [], 0, updates
                for within, index in enumerate(order):
                    global_micro = (epoch - 1) * FIT_IMAGES + within
                    detailed = global_micro in DETAIL_MICROS
                    pref.check_deadline(started, spec['deadline_seconds'])
                    boundary_before = None
                    if global_micro == FIT_IMAGES:
                        require(all(named[name].grad is not None for name in ACTIVE), 'Odd epoch boundary lost pending gradients')
                        boundary_before = [pref.digest_tensor(named[name].grad) for name in ACTIVE]
                        require(boundary_before == boundary, 'Accumulated epoch1 gradients changed before epoch2 forward')
                    image_id = plan['dataset_image_ids'][index]
                    data, samples, identity = core.image_data(native, old, model, dataset, coco, image_id)
                    require(identity['dataset_index'] == index, 'Actual FIT image/order index differs')
                    target = pref.check_targets(torch, data, identity)
                    external = pref.rng_snapshot(torch)
                    original_outputs, field_rows, bridge_rows = [], [], []

                    def original_observer(module, inputs, output):
                        require(len(output) == 2 and len(output[0]) == len(output[1]) == 10 and
                                all(not value.requires_grad for values in output for value in values),
                                'Every original native class/mask field must remain frozen')
                        original_outputs.append(output)
                        if detailed:
                            field_rows.append(dict(classes=[pref.digest_tensor(value) for value in output[0]],
                                native_masks=[pref.digest_tensor(value) for value in output[1]],
                                native_classes_require_grad=[False] * 10, native_masks_require_grad=[False] * 10))

                    def adapted_observer(module, inputs, output):
                        original = original_outputs[0]
                        require(output[0] is original[0] and all(output[1][i] is original[1][i] for i in range(9)) and
                                output[1][-1].requires_grad, 'Only final native mask may be replaced by the live two-weight path')
                        if detailed:
                            field_rows[0].update(adapted_final_mask=pref.digest_tensor(output[1][-1]),
                                classes10_objects_preserved=True, first9_mask_objects_preserved=True,
                                adapted_final_requires_grad=True)

                    def input_observer(module, inputs, output):
                        require(len(inputs) == 3 and not inputs[0].requires_grad and not inputs[1].requires_grad and
                                output['mask_logits'].requires_grad and output['logit_residual'].requires_grad,
                                'Frozen F/e0 and ambient new-weight autograd required')
                        if detailed:
                            bridge_rows.append(dict(feature=pref.digest_tensor(inputs[0]),
                                coefficient=pref.digest_tensor(inputs[1]), valid_fraction=pref.digest_tensor(inputs[2]),
                                feature_requires_grad=False, coefficient_requires_grad=False, residual_requires_grad=True))

                    original_handle = head.register_forward_hook(original_observer)
                    input_handle = adapter.register_forward_hook(input_observer)
                    adapted_handle = None
                    seed = 1000000 + global_micro
                    try:
                        with torch.random.fork_rng(devices=[0]):
                            torch.random.default_generator.manual_seed(seed)
                            torch.cuda.manual_seed(seed)
                            private_before = pref.rng_snapshot(torch)
                            require(torch.is_grad_enabled() and not data['inputs'].requires_grad, 'Ambient forward autograd required')
                            with hooks.query_pair_readout_context(torch, head, [sample.metainfo for sample in samples]) as record:
                                adapted_handle = head.register_forward_hook(adapted_observer)
                                features = model.extract_feat(data['inputs'])
                                classes, masks = head(features, samples)
                            gt = old.posterior_gt(head, data, identity)
                            raw_masks = data['data_samples'][0].gt_instances.masks.to_tensor(dtype=torch.uint8, device='cpu')
                            require(gt.masks.dtype == torch.int64 and
                                    torch.equal(gt.masks.detach().to(device='cpu', dtype=torch.uint8), raw_masks) and
                                    torch.equal(gt.labels, data['data_samples'][0].gt_instances.labels),
                                    'Native posterior complete padded int64 GT differs from raw targets')
                            posterior = dict(mask=identity['native_GT'], mask_uint8=target['masks'], labels=target['labels'],
                                GT_count=len(gt), complete_all_raw_annotations=True, preprocess_after_head_forward=True)
                            with old.observe_targets(head) as matches:
                                losses = head.loss_by_feat(classes, masks, [gt], [data['data_samples'][0].metainfo])
                            require(set(losses) == old.ATOMIC_KEYS and len(losses) == 30 and len(matches) == 10,
                                    'Original native30/ten actual native target records required')
                            summed, scalars = pref.loss_terms(torch, losses)
                            total, parsed = model.parse_losses(losses)
                            require(total.requires_grad and bool(torch.isfinite(total)) and torch.equal(total, summed),
                                    'Original native parse_losses total disconnected/nonfinite/different')
                            for match in matches:
                                require(len(set(match['positive_query_indices'])) == len(identity['ann_ids']) and
                                        sorted(match['assigned_GT_indices']) == list(range(len(identity['ann_ids']))) and
                                        all(0 <= q < 100 for q in match['positive_query_indices']), 'Native matching dropped raw GT')
                            require(len(original_outputs) == 1 and record['grad_mode_enabled_at_start'] is True and
                                    record['native_graph_properties']['feature_requires_grad'] is False and
                                    record['native_graph_properties']['coefficient_requires_grad'] is False and
                                    record['native_graph_properties']['native_final_mask_requires_grad'] is False,
                                    'Native graph must stay frozen')
                            if detailed:
                                require(len(field_rows) == len(bridge_rows) == 1, 'Detailed observer count differs')
                                native_snapshot = dict(classes=field_rows[0]['classes'], masks=field_rows[0]['native_masks'],
                                    feature=bridge_rows[0]['feature'], coefficient=bridge_rows[0]['coefficient'],
                                    valid_fraction=bridge_rows[0]['valid_fraction'],
                                    support={key: record['readout_observation'][key] for key in
                                             ('eligible', 'eligible_counts', 'canonical_eligible_counts', 'rho_mass', 'support_nonempty')})
                                if mode == MODES[0]:
                                    shared_detailed[global_micro] = native_snapshot
                                require(native_snapshot == shared_detailed[global_micro],
                                        'Observed frozen native fields/E/support differ at declared snapshots')
                            (total / 2).backward()
                            private_after = pref.rng_snapshot(torch)
                    finally:
                        original_handle.remove()
                        input_handle.remove()
                        if adapted_handle is not None:
                            adapted_handle.remove()
                    external_after = pref.rng_snapshot(torch)
                    require(core.equal_rng(torch, external, external_after) and
                            core.hook_inventory(model) == hook_baseline and hooks._contexts_inactive(head) and
                            record['hooks_removed'] is True and record['cache_cleared_after_forward'] is True,
                            'External RNG or scoped hook/cache cleanup differs')
                    norms = []
                    for name in ACTIVE:
                        gradient = named[name].grad
                        require(gradient is not None and bool(torch.isfinite(gradient).all()), 'Actual accumulated two-weight gradient absent/nonfinite')
                        norms.append(float(gradient.detach().float().norm().cpu()))
                        activity[name] += int(norms[-1] > 0)
                    cold_hidden_zero = None
                    if global_micro < 2:
                        cold_hidden_zero = bool(torch.count_nonzero(named[ACTIVE[0]].grad).item() == 0)
                        require(cold_hidden_zero and record['cold_zero_final_weight'] is True and
                                record['cold_native_value_equal'] is True, 'Fresh cold hidden-zero/native final identity differs')
                    else:
                        warm_both = warm_both or all(value > 0 for value in norms)
                    require(all(named[name].grad is None for name in policy['native_names']) and
                            all(not module.training for module in model.modules()), 'Frozen native451 grad/eval boundary broken')
                    stepped = (global_micro + 1) % 2 == 0
                    preclip = None
                    if stepped:
                        for name in ACTIVE:
                            require(named[name].grad is not None, 'Actual optimizer step has a missing two-weight gradient')
                            step_activity[name] += 1
                        global_norm = torch.nn.utils.clip_grad_norm_([named[name] for name in ACTIVE], max_norm=0.01, norm_type=2)
                        require(bool(torch.isfinite(global_norm)), 'Actual global clip norm nonfinite')
                        preclip = float(global_norm.detach().cpu())
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        updates += 1
                    boundary_after = None
                    if global_micro == FIT_IMAGES - 1:
                        boundary = [pref.digest_tensor(named[name].grad) for name in ACTIVE]
                        boundary_after = boundary
                    observation = record['readout_observation']
                    checks = readout_checks(observation, mode)
                    compact_hook = dict(record)
                    if not detailed:
                        compact_hook['readout_observation'] = None
                    trace = dict(mode=mode, microstep=global_micro, global_microstep=global_micro, epoch=epoch,
                        epoch_microstep=within, dataset_index=index, image_id=image_id, identity=identity, GT=target,
                        posterior_GT=posterior, input=pref.digest_tensor(data['inputs']), loss_fields=scalars,
                        native_total=float(total.detach().cpu()), native_loss_only=True, Sobel_added=False,
                        auxiliary_loss_added=False, matches=list(matches), detailed_observation=detailed,
                        native_hook=compact_hook, readout_checks=checks, readout_observation=(observation if detailed else None),
                        head_fields=(field_rows[0] if detailed else None), bridge_inputs=(bridge_rows[0] if detailed else None),
                        accumulation_divisor=2, pending_accumulation_before=global_micro % 2,
                        pending_accumulation_after=(global_micro + 1) % 2, accumulation_gradient_before=boundary_before,
                        accumulation_gradient_after=boundary_after, optimizer_step=stepped, optimizer_updates=updates,
                        preclip_global_norm=preclip, gradient_clipping_applied=stepped,
                        gradient_norms_aligned_active_names=norms, gradients_are_accumulated_preclip=True,
                        cold_hidden_gradient_exact_zero=cold_hidden_zero, native_parameter_grads_None=True,
                        all_native_modules_eval=True, hook_cache_clear=True, GT_removed_from_forward=True,
                        private_seed=seed, private_RNG_before=[pref.digest_tensor(value) for value in private_before],
                        private_RNG_after=[pref.digest_tensor(value) for value in private_after],
                        external_RNG_before=[pref.digest_tensor(value) for value in external],
                        external_RNG_after=[pref.digest_tensor(value) for value in external_after], external_RNG_restored=True,
                        common_realized_matching_or_points_claimed=False)
                    trace_stream.write(json.dumps(trace, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n')
                    seen.append(index)
                    epoch_GT += len(identity['ann_ids'])
                    GT_visits += len(identity['ann_ids'])
                    if (global_micro + 1) % 32 == 0 or within == FIT_IMAGES - 1:
                        trace_stream.flush()
                        print(json.dumps(dict(status='SHORT2_PROGRESS', mode=mode, epoch=epoch,
                            microsteps=global_micro + 1, updates=updates, GT_visits=GT_visits)), flush=True)
                    del data, samples, features, classes, masks, gt, raw_masks, losses, total, summed, parsed, matches
                    del original_outputs, field_rows, bridge_rows, record, compact_hook, observation, trace
                require(seen == order and sorted(seen) == list(range(FIT_IMAGES)) and epoch_GT == FIT_GT and
                        updates == epoch * FIT_IMAGES // 2, 'Actual complete741/1661/global accumulation epoch coverage differs')
                epochs.append(dict(epoch=epoch, microsteps=FIT_IMAGES, updates_this_epoch=updates - epoch_updates_before,
                    cumulative_updates=updates, GT_visits=epoch_GT, seen_dataset_indices=seen,
                    pending_accumulation=epoch * FIT_IMAGES % 2))
                # Deliberately no epoch-boundary zero_grad, optimizer step or scheduler.
            trace_stream.flush()
            os.fsync(trace_stream.fileno())
        require(not trace_path.exists(), 'Refuse to overwrite trace artifact')
        pending_path.rename(trace_path)
        require(updates == UPDATES and len(epochs) == EPOCHS and GT_visits == EPOCHS * FIT_GT and warm_both and
                all(activity[name] > 0 for name in ACTIVE) and all(value.grad is None for value in model.parameters()),
                'Full SHORT2 counters/warm activity/no pending gradients differ')
        final = pref.cpu_tree(torch, model.state_dict())
        require(final.keys() == initial.keys() and all(bool(torch.isfinite(value).all()) for value in final.values()) and
                all(torch.equal(value.detach().cpu(), final[name]) for name, value in model.state_dict().items()) and
                all(torch.equal(final[name], value) for name, value in teacher.items()),
                'Actual finite full612 CPU transfer or full610 frozen endpoint differs')
        changed = {name: not torch.equal(initial[name], value) for name, value in final.items()}
        require({name for name, value in changed.items() if value} == set(ACTIVE), 'Only two fresh weights may change')
        final_optimizer = pref.cpu_tree(torch, optimizer.state_dict())
        require(len(final_optimizer['state']) == 2 and all(float(value['step']) == UPDATES
            for value in final_optimizer['state'].values()), 'Actual two AdamW moment states must be at step741')
        group_names = [[name] for name in ACTIVE]
        meta = dict(schema=CP_SCHEMA, kind='mask2former', mode=mode, phase=spec['phase'], source=spec['entry_source'],
            spec=spec_ref, plan=plan_ref, completed_epoch=2, completed_epochs=2, initial_epoch=0, epochs_this_run=2, total_epochs=2,
            global_microsteps=MICROSTEPS, global_updates=UPDATES, consumed_raw_GT=GT_visits,
            FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, batch_size=1, accumulation=2, policy=policy, optimizer=audit,
            optimizer_param_names=group_names, teacher_checkpoint=spec['teacher_checkpoint'],
            teacher_qualification=spec['teacher_qualification'], teacher_training_spec=spec['teacher_training_spec'],
            teacher_receipt=spec['teacher_receipt'], preflight_CPU_qualification=spec['preflight_CPU_qualification'],
            teacher_mode='native0', teacher_epoch=24,
            preflight_training_spec=spec['preflight_training_spec'], readout_source=spec['readout_source'],
            geometry_source=spec['geometry_source'], hooks_source=spec['hooks_source'],
            native610_exact=True, frozen_native=True, native_loss_only=True, native_loss_atoms=30,
            no_Sobel_or_auxiliary_loss=True, no_AP=True, LR_decay=False,
            fresh_optimizer_state_entries=0, teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False,
            teacher_external_RNG_loaded=False, preflight_checkpoint_loaded=False, resume_checkpoint=None,
            resume_qualification=None, no_pending_accumulation_gradients=True, no_scheduler=True, no_warmup=True,
            no_feature_cache=True, short_budget_hypothesis_probe=True, eligible_for_formal_resume=False, AP_computed=False)
        checkpoint = dict(state_dict=final, adapter_state_dict={name: final[PREFIX + name] for name in cold},
            optimizer_state_dict=final_optimizer, external_rng_state=pref.cpu_tree(torch, pref.rng_snapshot(torch)), meta=meta)
        checkpoint_path = directory / 'NODE_E2.pth'
        with checkpoint_path.open('xb') as stream:
            torch.save(checkpoint, stream)
        require(pref.tree_equal(torch, checkpoint, torch.load(str(checkpoint_path), map_location='cpu')), 'Endpoint CPU readback differs')
        proof = dict(schema=PROOF_SCHEMA, mode=mode, native_teacher610=pref.state_proof(teacher), before=before,
            after=pref.state_proof(final), changed=changed, policy=policy, optimizer=audit,
            adapter_names=ACTIVE, positive_gradient_microcounts=activity, actual_final_optimizer=
            teacher_reader.optimizer_snapshot(torch, reused_reader, final_optimizer, group_names),
            fresh_optimizer_state_entries=0, teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False,
            teacher_external_RNG_loaded=False, preflight_checkpoint_loaded=False, native610_exact=True,
            native_grad_None=True, native_grad_None_microchecks=MICROSTEPS, all_native_modules_eval=True,
            hook_cache_clear=True, frozen_BN_all_exact=True, warm_both_weights_positive=True,
            CPU_checkpoint_readback_exact=True, GPU_to_CPU_readback_state_exact=True, teacher_loading=loading,
            cold_adapter=pref.state_proof(cold), cold_constructor_repeated_exact=True, cold_constructor_RNG_restored=True,
            all_adapter_weights_positive_and_changed=True, optimizer_step_nonNone_gradient_counts=step_activity,
            no_pending_accumulation_gradients=True, completed_epochs=2,
            fresh_identical_adapter_initialization_across_modes=True, epoch_boundary_accumulation_preserved=True,
            boundary_gradient_digest=boundary, global_microsteps=MICROSTEPS, global_updates=UPDATES,
            consumed_raw_GT=GT_visits, epoch_summaries=epochs, detailed_microsteps=list(DETAIL_MICROS),
            common_realized_matching_or_points_claimed=False)
        proof_path = directory / 'TRAIN_STATE_PROOF.json'
        core.write(proof_path, proof)
        return dict(mode=mode, checkpoint=reference(checkpoint_path), state_proof=reference(proof_path),
            traces=reference(trace_path), trace_records=MICROSTEPS, plan=plan_ref, epochs=2, completed_epoch=2, completed_epochs=2,
            global_microsteps=MICROSTEPS, global_updates=UPDATES, consumed_raw_GT=GT_visits,
            FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, detailed_microsteps=list(DETAIL_MICROS), epoch_summaries=epochs,
            native610_exact=True, frozen_native=True, native_loss_only=True, AP_computed=False,
            eligible_for_formal_resume=False, no_pending_accumulation_gradients=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.to('cpu')


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)  # Metadata only; deliberately not hashed yet.
    require(Path(spec['teacher_receipt']['path']).is_absolute(), 'Absolute raw teacher receipt required')
    teacher_receipt = read(spec['teacher_receipt']['path'])
    raw_terminal(teacher_receipt, spec['teacher_phase'])  # FIRST before any SOURCE/Q/CP/GT/SPEC hash/import.
    require(Path(spec['preflight_receipt']['path']).is_absolute(), 'Absolute raw preflight receipt required')
    preflight_receipt = read(spec['preflight_receipt']['path'])
    raw_terminal(preflight_receipt, spec['preflight_phase'])
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('query_conditioned_frozen_native_short2_') and
            spec['modes'] == list(MODES) and spec['training_all_FIT'] is True and
            spec['preflight_image_ids'] == [1, 2, 3, 5] and spec['epochs'] == EPOCHS and
            spec['FIT_images'] == FIT_IMAGES and spec['FIT_GT'] == FIT_GT and
            spec['total_microsteps'] == MICROSTEPS and spec['total_updates'] == UPDATES and
            spec['order_seed'] == 2026100917 and spec['loss_seed_base'] == 1000000 and
            spec['expected_epoch'] == 24 and spec['hidden_seed'] == 2026100911 and spec['permutation_seed'] == 2026101013 and
            spec['accumulation'] == 2 and spec['learning_rate'] == 0.0001 and spec['weight_decay'] == 0.05 and
            spec['clip_grad'] == dict(max_norm=0.01, norm_type=2) and spec['checkpoint_epoch'] == 2 and
            spec['detail_microsteps'] == list(DETAIL_MICROS) and spec['CPU_affinity'] == [0, 1] and
            spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
            spec['deadline_seconds'] == 10800 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
            'Exact four-arm full FIT SHORT2 matched recipe/resource/snapshot scope required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['image_root']).is_absolute() and
            Path(spec['GPU_admission_file']).is_absolute(), 'Fresh absolute output/current admission/image root required')
    spec_ref = owner_gate(spec_path, spec)
    preflight_gate(spec, preflight_receipt)
    core = load_source(spec['preflight_source'], 'sealed_actual_query_frozen_preflight_helpers')
    teacher_row = core.qualified_teacher(spec, teacher_receipt)
    core.fixture_gate(spec)
    output.mkdir(parents=True)
    core.write(output / 'START.json', dict(schema='query-conditioned-frozen-native-short2-start-v1',
        phase=spec['phase'], source=spec['entry_source'], spec=spec_ref, modes=list(MODES),
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_receipt=spec['teacher_receipt'], preflight_CPU_qualification=spec['preflight_CPU_qualification'],
        preflight_training_spec=spec['preflight_training_spec'], training_all_FIT=True, epochs=2,
        FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, total_microsteps=MICROSTEPS, total_updates=UPDATES,
        accumulation=2, order_seed=2026100917, loss_seed_base=1000000, detail_microsteps=list(DETAIL_MICROS),
        teacher_terminal_gate_before_scientific_access=True, preflight_terminal_gate_before_scientific_access=True,
        all_four_CPU_models_before_CUDA_required=True, fresh_optimizer_state_entries=0,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        preflight_checkpoint_loaded=False, no_scheduler=True, no_warmup=True, no_feature_cache=True,
        no_formal_resume=True, no_DEV=True, no_TEST=True, no_AP=True, no_retry=True))
    import psutil
    import torch
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
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual torch2/2 threads required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'short2_native_pref')
    initializer = load_source(spec['initializer_source'], 'short2_native_initializer')
    old = load_source(spec['native_helpers'], 'short2_native_images'); old.torch = torch
    native = load_source(spec['native_preflight_helpers'], 'short2_native_image_helpers')
    evaluation = load_source(spec['evaluation_source'], 'short2_native_E24_loader')
    teacher_reader = load_source(spec['training_CPU_reader'], 'short2_teacher_CPU_helpers')
    reused_reader = load_source(spec['reused_CPU_reader'], 'short2_original_CPU_helpers')
    readout = load_source(spec['readout_source'], 'short2_query_readout')
    geometry = load_source(spec['geometry_source'], 'short2_query_geometry')
    hooks = load_source(spec['hooks_source'], 'short2_query_whole_head_bridge')
    require(tuple(readout.MODES) == tuple(hooks.MODES) == MODES, 'Actual four query modes required')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Actual original COCO80 ordering required')
    helper_spec = dict(initializer_source=spec['initializer_source'], training_source=spec['training_source'],
        checkpointmap={'native0': spec['teacher_checkpoint']}, qualificationmap={'native0': spec['teacher_qualification']}, expected_epoch=24)
    prepared, teacher, shared_cold = {}, None, None
    for mode in MODES:
        pref.check_deadline(started, spec['deadline_seconds'])
        model, state, meta, loading = evaluation.prepare_model(torch, pref, initializer, None, None, None,
            MODELS, revert_sync_batchnorm, Config, classes, helper_spec, teacher_row, 'native0')
        require(not torch.cuda.is_initialized(), 'All four fresh CPU teachers must precede CUDA')
        if teacher is None:
            teacher = state
        require(pref.tree_equal(torch, teacher, state), 'All arms must load exactly the same full610 teacher')
        rng_before = torch.get_rng_state().clone()
        adapter = hooks.register_query_pair_readout(torch, readout, geometry, model, mode,
            spec['readout_source'], spec['geometry_source'], hidden_seed=2026100911, permutation_seed=2026101013)
        cold = pref.cpu_tree(torch, adapter.state_dict())
        repeated = readout.build_adapter(torch, geometry, mode=mode, hidden_seed=2026100911, permutation_seed=2026101013)
        require(torch.equal(rng_before, torch.get_rng_state()) and
                pref.tree_equal(torch, cold, pref.cpu_tree(torch, repeated.state_dict())) and bool((cold['final.weight'] == 0).all()),
                'Fresh repeated private constructor/zero final differs')
        if shared_cold is None:
            shared_cold = cold
        require(pref.tree_equal(torch, shared_cold, cold), 'Fresh two-weight initialization differs across four modes')
        model.requires_grad_(False).eval()
        adapter.requires_grad_(True)
        core.frozen_policy(torch, model)
        prepared[mode] = (model, loading, cold)
        del state, meta, adapter, repeated
    dataset = CocoDataset(ann_file=str(bound(spec['FIT_annotation'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT_annotation'])))
    ids = [int(dataset.get_data_info(index)['img_id']) for index in range(len(dataset))]
    require(len(dataset) == len(coco.imgs) == FIT_IMAGES and len(coco.anns) == FIT_GT and
            len(set(ids)) == FIT_IMAGES and set(ids) == set(map(int, coco.imgs)) and not torch.cuda.is_initialized(),
            'All four CPU models and complete unique FIT741/1661 population required before CUDA')
    rng_before_order = torch.get_rng_state().clone()
    generator = torch.Generator(device='cpu')
    generator.manual_seed(2026100917)
    orders = [torch.randperm(FIT_IMAGES, generator=generator, device='cpu').tolist() for epoch in range(EPOCHS)]
    require(torch.equal(rng_before_order, torch.get_rng_state()) and
            all(sorted(order) == list(range(FIT_IMAGES)) for order in orders), 'Fresh private common order/RNG differs')
    plan = dict(schema='query-conditioned-frozen-native-short2-order-plan-v1', seed=2026100917, epochs=2,
        FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, orders=orders, dataset_image_ids=ids, private_CPU_generator=True,
        teacher_order_RNG_loaded=False, external_CPU_RNG_restored=True)
    plan_path = output / 'PLAN.json'
    core.write(plan_path, plan)
    plan_ref = reference(plan_path)
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096 * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory, 0)
    results, shared_detailed, cleanup = {}, {}, {}
    for mode in MODES:
        current = prepared.pop(mode)
        results[mode] = train_arm(torch, core, pref, old, native, hooks, teacher_reader, reused_reader,
            spec, spec_ref, teacher_row['training_spec'], current, teacher, dataset, coco, output, plan, plan_ref,
            mode, shared_detailed, started)
        del current
        cleanup[mode] = pref.clear_gpu(torch)
    report = dict(schema=REPORT_SCHEMA, phase=spec['phase'], source=spec['entry_source'], spec=spec_ref,
        status='SHORT2_QUERY_CONDITIONED_FROZEN_NATIVE_COMPLETE_NO_METHOD_RESULT', error=None,
        modes=list(MODES), mode_rows=results, plan=plan_ref, teacher_checkpoint=spec['teacher_checkpoint'],
        teacher_qualification=spec['teacher_qualification'], teacher_receipt=spec['teacher_receipt'],
        preflight_CPU_qualification=spec['preflight_CPU_qualification'], preflight_training_spec=spec['preflight_training_spec'],
        training_all_FIT=True, epochs=2, FIT_images=FIT_IMAGES, FIT_GT=FIT_GT,
        global_microsteps_per_arm=MICROSTEPS, global_updates_per_arm=UPDATES, consumed_raw_GT_per_arm=EPOCHS * FIT_GT,
        detailed_microsteps=list(DETAIL_MICROS), native610_exact=True, frozen_native=True,
        native_loss_only=True, native_loss_atoms=30, all_four_CPU_models_before_CUDA=True,
        only_one_model_on_CUDA_at_a_time=True, teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False,
        teacher_external_RNG_loaded=False, preflight_checkpoint_loaded=False, no_scheduler=True, no_warmup=True,
        no_feature_cache=True, epoch_boundary_accumulation_preserved=True, no_AP=True, no_DEV=True, no_TEST=True,
        AP_computed=False, efficacy_claimed=False, formal_resume_eligible=False, scientific_method_decision_computed=False,
        common_realized_matching_or_points_claimed=False, native_field_equality_checked_only_at_detailed_microsteps=True,
        cleanup=cleanup, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt')
    report_path = output / 'REPORT.json'
    core.write(report_path, report)
    core.write(output / 'QUALIFIED.json', dict(schema='query-conditioned-frozen-native-short2-aggregate-v1',
        status='ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_PENDING_CPU_READER', source=spec['entry_source'],
        spec=spec_ref, report=reference(report_path), plan=plan_ref, mode_rows=results,
        teacher_checkpoint=spec['teacher_checkpoint'], preflight_CPU_qualification=spec['preflight_CPU_qualification'],
        independently_CPU_qualified=False, efficacy_claimed=False, AP_computed=False, formal_resume_eligible=False))
    print(json.dumps(dict(status='SHORT2_COMPLETE_CPU_QUALIFICATION_PENDING', modes=list(MODES))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Inert SHORT2 SOURCE requires explicit ROOT --owner-execute')
    try:
        run(args.spec)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
