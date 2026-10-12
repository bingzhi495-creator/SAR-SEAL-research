"""Own qualified joint first24 weights plus fresh additional48; inert SOURCE.

A native30 + lambda1 native-L point BCE; B native30 + lambda1 L+u point BCE.
Each arm restores its own qualified nativeE48 and B's complete eight head tensors.
Only weights are warm; native345 and B8 AdamW/order/RNG restart at phase zero.
Original teacherE24 constructs the architecture and typedBN106/159 reference.
Native schedule is phase15808/17126 gamma.1; B head retains fixed1e-4 LR.
Fixed final phase48 is effective native96 and B head72, with no DEV/TEST/AP here.
This is not a restored optimizer trajectory, compute match or convergence claim.
"""
import argparse
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import types

PREFIX = 'query-pixel-joint-native-E24warm-additional48'
SCHEMA = 'ROOT-' + PREFIX + '-spec-v1'
ARMS = ('A', 'B')
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
MODULES = ('native_loader_source', 'evaluation_source', 'micro_preflight_source', 'initializer_source',
    'training_source', 'native_helpers', 'renderer_source', 'posterior_readout_source',
    'joint_capture_source', 'joint_losses_source', 'joint_adapter_source', 'native_objective_source')
LIBRARIES = ('adamw_SOURCE', 'clip_grad_SOURCE', 'hungarian_SOURCE', 'interpolation_SOURCE',
    'mask_sampler_SOURCE', 'mask_sampling_result_SOURCE', 'match_cost_SOURCE', 'native_fusion_SOURCE',
    'native_mask_head_SOURCE', 'native_parent_head_SOURCE', 'panoptic_GT_SOURCE', 'point_sample_SOURCE',
    'solver_source', 'torch_version_source')
PREFLIGHT_ROLES = ('joint_preflight_entry_source', 'joint_preflight_CPU_reader', 'joint_preflight_spec',
    'joint_preflight_CPU_qualification', 'joint_preflight_report', 'joint_preflight_receipt')
NUMERIC = dict(forward_rtol=1e-4, forward_atol=1e-6, gradient_rtol=1e-3, gradient_atol=1e-7,
    native_projection_rtol=1e-4, native_projection_atol=1e-4)
DETAILS = (0, 1, 35566, 35567)
SCHEDULE = dict(gamma=.1, milestones=[15808, 17126], warmup_start_factor=1, warmup_updates=0)
HEAD_RECIPE = dict(type='AdamW', lr=1e-4, weight_decay=.05, betas=[.9, .999], eps=1e-8)
COUNTERS = dict(epochs=48, microsteps=35568, optimizer_updates=17784, raw_GT_visits_per_arm=79728)
FIRST24_PREFIX = 'query-pixel-joint-native-fullFIT24'
FIRST24_ROLES = ('first24_training_spec', 'first24_CPU_qualification', 'first24_report', 'first24_receipt',
    'first24_entry_source', 'first24_CPU_reader_source')
FIRST24_COUNTERS = dict(epochs=24, microsteps=17784, optimizer_updates=8892, raw_GT_visits_per_arm=39864)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] > 0 and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref, 'Pinned input differs')
    return Path(ref['path'])


def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and
        hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before compilation')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def terminal(raw, phase):
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(k in raw and raw[k] is None for k in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: strict owned terminal0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw scope differs')


def source_gate(path, spec):
    # Prior locator JSON only; all original raw terminals precede any hash/import.
    first = read(spec['first24_training_spec']['path']); tiny_spec = read(first['joint_preflight_spec']['path'])
    require(set(tiny_spec['training_receiptmap']) == set(tiny_spec['training_phasemap']) == set(MODES), 'Original four raws required')
    for mode in MODES:
        terminal(read(tiny_spec['training_receiptmap'][mode]['path']), tiny_spec['training_phasemap'][mode])
    for row in tiny_spec['prior_receipts']:
        require(set(row) == {'receipt', 'phase'}, 'Explicit prior raw required')
        terminal(read(row['receipt']['path']), row['phase'])
    terminal(read(first['joint_preflight_receipt']['path']), first['joint_preflight_phase'])
    raw = read(spec['first24_receipt']['path']); terminal(raw, spec['first24_phase'])
    owner = read(spec['GPU_admission_file']); source, spec_ref = reference(__file__), reference(path)
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['automatic_retry'] is False and
        owner['phase'] == spec['phase'] and owner['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
        owner['cuda_visible_devices'] == '0' and owner['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        owner['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and owner['driver_sha256'] == OWNER_SHA and
        Path(owner['entry']).resolve() == Path(source['path']) and owner['entry_args'] == ['--spec', str(path), '--owner-execute'],
        'Current owner/entry/resources differ')
    hashes = owner['bound_source_sha256']
    require(hashes.get(str(path)) == spec_ref['sha256'] and spec['entry_source'] == source and
        len({r['path'] for r in spec['sources']}) == len(spec['sources']), 'Current SPEC/unique closure differs')
    for ref in spec['sources']:
        bound(ref); require(hashes.get(ref['path']) == ref['sha256'], 'Whole closure must be owner-bound')
    for role in MODULES + LIBRARIES + PREFLIGHT_ROLES + FIRST24_ROLES + ('entry_source', 'CPU_reader', 'training_CPU_reader',
            'reused_CPU_reader', 'readout_source', 'geometry_source', 'hooks_source', 'FIT'):
        require(spec[role] in spec['sources'], 'Required role absent from closure: ' + role)
    require(set(spec['warm_checkpoints']) == set(ARMS) and all(r in spec['sources'] for r in spec['warm_checkpoints'].values()) and
        first == read(bound(spec['first24_training_spec'])) and all(r in spec['sources'] for r in first['sources']), 'Own first24 closure/checkpoints absent')
    for role in MODULES + LIBRARIES + PREFLIGHT_ROLES + ('FIT', 'image_root', 'checkpointmap', 'qualificationmap',
            'training_receiptmap', 'training_phasemap', 'prior_receipts', 'joint_preflight_phase', 'initializers',
            'readout_seed', 'point_seed_base', 'loss_seed_base', 'order_seed', 'kappa', 'eps', 'auxiliary_weight', 'backbone_with_cp'):
        require(spec[role] == first[role], 'Own qualified construction/recipe differs: ' + role)
    prior = load(spec['first24_entry_source'], 'qualified_own_joint_first24_lineage_ONLY')
    first_source, first_spec, first_owner, tiny = prior.source_gate(Path(spec['first24_training_spec']['path']), first)
    q = read(bound(spec['first24_CPU_qualification']))
    require(first['schema'] == 'ROOT-'+FIRST24_PREFIX+'-spec-v1' and q['schema'] == 'ROOT-'+FIRST24_PREFIX+'-CPU-qualification-v1' and
        q['status'] == 'ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_CPU_QUALIFIED' and
        q['source'] == first_source == spec['first24_entry_source'] == first['entry_source'] and
        q['reader_source'] == spec['first24_CPU_reader_source'] == first['CPU_reader'] and
        q['spec'] == first_spec == spec['first24_training_spec'] and q['GPU_admission'] == first_owner and
        q['parent_receipt'] == spec['first24_receipt'] and q['phase'] == spec['first24_phase'] == first['phase'] and
        q['report'] == spec['first24_report'] and q['counters_per_arm'] == FIRST24_COUNTERS and
        q['completed_phase_epochs'] == 24 and q['effective_native_epoch'] == 48 and
        q['phase_microsteps_per_arm'] == 17784 and q['native_updates_per_arm'] == q['B_readout_updates'] == 8892 and
        q['phase_raw_GT_visits_per_arm'] == 39864 and q['CPU_numeric_protocol'] == NUMERIC and
        q['teacher_checkpoint'] == spec['checkpointmap']['native0'] and q['teacher_qualification'] == spec['qualificationmap']['native0'] and
        q['no_AP'] is q['no_DEV'] is q['no_TEST'] is True and q['technical_weights_loaded'] is q['formal_resume_eligible'] is False and
        q['CPU_execution'] == dict(device='cpu', affinity=[0, 1], CUDA_VISIBLE_DEVICES='', torch_num_threads=2,
            torch_num_interop_threads=2, CUDA_initialized=False), 'Actual own first24 CPU qualification differs')
    require(raw['admission_sha256'] == first_owner['sha256'] and set(q['result']) == set(q['arms']) == set(ARMS) and
        all(q[k] in spec['sources'] for k in ('GPU_admission', 'start', 'GPU_qualification')), 'Actual first24 primary refs absent')
    report = read(bound(spec['first24_report']))
    require(report['schema'] == FIRST24_PREFIX+'-report-v1' and report['status'] == 'QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_GPU_COMPLETE_PENDING_CPU' and
        report['source'] == first_source and report['spec'] == first_spec and report['GPU_admission'] == first_owner and
        report['phase'] == first['phase'] and report['result'] == q['result'] and report['error'] is None and
        report['cleanup']['allocated_after'] == 0 and report['counters_per_arm'] == FIRST24_COUNTERS and report['actual_total_backward_calls'] == 35568,
        'Actual completed own first24 report differs')
    for arm in ARMS:
        row, summary = q['result'][arm], q['arms'][arm]
        require(row['checkpoint'] == summary['checkpoint'] == spec['warm_checkpoints'][arm] and row['completed_phase_epochs'] == 24 and
            row['effective_native_epoch'] == 48 and row['phase_microsteps'] == 17784 and row['phase_updates'] == 8892 and row['phase_raw_GT_visits'] == 39864 and
            summary['strict_native610_CPU_population'] is summary['initial_native610_CPU_exact'] is summary['typed_BN106_affine159_buffers_CPU_exact'] is True and
            summary['native345_moments_match_actual_trace_counts'] is True and summary['readout_moment_entries'] == (8 if arm == 'B' else 0), 'Own arm qualified endpoint differs')
        require(all(row[k] in spec['sources'] for k in ('checkpoint', 'endpoint_proof', 'trace', 'order', 'epochs', 'start')) and
            len(row['audits']) == 4 and all(r in spec['sources'] for r in row['audits']), 'Own arm full primary artifact closure absent')
    if 'scientific_spec' in owner:
        require(owner['scientific_spec'] == spec_ref, 'Current scientific SPEC differs')
    return source, spec_ref, reference(spec['GPU_admission_file']), tiny


def warm_weights(torch, pref, spec, prepared, heads):
    """Strict CPU own endpoints: deserialize old containers, restore weights only."""
    shapes = {'feature_projection.weight': (16,256,1,1), 'feature_projection.bias': (16,),
        'query_projection.weight': (16,256), 'query_projection.bias': (16,), 'hidden.weight': (32,36),
        'hidden.bias': (32,), 'final.weight': (1,32), 'final.bias': (1,)}
    for arm in ARMS:
        model, original, policy, teacher_loading = prepared[arm]
        cp = torch.load(str(bound(spec['warm_checkpoints'][arm])), map_location='cpu'); meta = cp['meta']
        require(cp['schema'] == FIRST24_PREFIX+'-checkpoint-v1' and meta['source'] == spec['first24_entry_source'] and
            meta['spec'] == spec['first24_training_spec'] and meta['phase'] == spec['first24_phase'] and meta['arm'] == arm and
            meta['teacher_checkpoint'] == spec['checkpointmap']['native0'] and meta['teacher_qualification'] == spec['qualificationmap']['native0'] and
            meta['initial_native_epoch'] == 24 and meta['completed_phase_epochs'] == 24 and meta['effective_native_epoch'] == 48 and
            meta['phase_microsteps'] == 17784 and meta['phase_updates'] == 8892 and meta['phase_raw_GT_visits'] == 39864 and
            meta['technical_weights_loaded'] is meta['formal_resume_eligible'] is False and meta['no_pending_accumulation_gradients'] is True,
            'Own complete first24 checkpoint metadata differs')
        state, head = cp['native_state_dict'], cp['readout_state_dict']
        require(set(state) == set(original) and len(state) == 610 and all(v.device.type == 'cpu' and v.dtype == original[n].dtype and
            v.shape == original[n].shape and bool(torch.isfinite(v).all()) for n,v in state.items()) and
            all(torch.equal(state[n],original[n]) for n in policy['frozen_names']+policy['BN_buffer_names']), 'Own native610/typed originalBN differs')
        result = model.load_state_dict(state, strict=True)
        require(not result.missing_keys and not result.unexpected_keys and pref.tree_equal(torch,model.state_dict(),state), 'Strict native warm load differs')
        if arm == 'B':
            require(set(head) == set(shapes) and all(v.device.type == 'cpu' and v.dtype == torch.float32 and tuple(v.shape) == shapes[n] and
                bool(torch.isfinite(v).all()) for n,v in head.items()), 'Own warm B8 complete finite state differs')
            result = heads[arm].load_state_dict(head, strict=True)
            require(not result.missing_keys and not result.unexpected_keys and pref.tree_equal(torch,heads[arm].state_dict(),head), 'Strict warm B8 load differs')
        else:
            require(head == {} and heads[arm] is None, 'A must have no external readout state')
        warm = pref.cpu_tree(torch, model.state_dict())
        loading = dict(original_teacher_loading=teacher_loading, warm_checkpoint=spec['warm_checkpoints'][arm],
            first24_CPU_qualification=spec['first24_CPU_qualification'], strict_native610_warm_load_CPU_exact=True,
            strict_B8_warm_load_CPU_exact=True if arm=='B' else None, weights_warm_start=True,
            previous_optimizer_order_or_external_RNG_restored=False, constructor_seed_is_not_loaded_weight_identity=True)
        prepared[arm] = (model, warm, policy, loading)
        del cp, meta, state, head, warm




def gradient_norms(torch, named):
    live = [(n, p.grad.detach().float().norm()) for n, p in named.items() if p.grad is not None]
    values = torch.stack([v for _, v in live]).cpu().tolist() if live else []
    require(all(math.isfinite(v) and v >= 0 for v in values), 'Nonfinite accumulated gradients')
    present = dict(zip([n for n, _ in live], values))
    return {n: present.get(n) for n in named}


def pending_gradients(pref, native_named, readout_named):
    return {kind: {n: None if p.grad is None else pref.digest_tensor(p.grad) for n, p in named.items()}
        for kind, named in (('native', native_named), ('readout', readout_named))}


def save_packet(torch, pref, path, value):
    value = pref.cpu_tree(torch, value)
    with path.open('xb') as stream:
        torch.save(value, stream)
    return reference(path)


def optimizer_steps(torch, opt, named, counts):
    ids = [g['params'][0] for g in opt.param_groups]
    require(len(ids) == len(named) and set(ids) == set(named.values()), 'Optimizer singleton membership differs')
    for name, parameter in named.items():
        state = opt.state.get(parameter)
        if counts[name] == 0:
            require(state is None, 'Unexpected optimizer moment state')
            continue
        require(state is not None and set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and
            float(state['step']) == counts[name] and bool(torch.isfinite(state['exp_avg']).all()) and
            bool(torch.isfinite(state['exp_avg_sq']).all()) and bool((state['exp_avg_sq'] >= 0).all()), 'Actual AdamW moment count/finite state differs')


def arm_run(torch, np, cv2, m, tiny, prepared, readout, construction, spec, source, spec_ref, output, dataset, coco, started, arm):
    native, legacy, pref, initializer, common, old, diag, posterior, capture, joint, adapter, objective = m
    output.mkdir(); model, anchor, policy, loading = prepared; model.to('cuda:0')
    require(dict(pref.trainability(torch, model), kind='mask2former') == policy, 'Native CUDA policy differs')
    native_opt, native_audit = pref.make_optimizer(torch, model, construction['initializers']['mask2former'], policy)
    named = dict(model.named_parameters()); native_named = {n: named[n] for n in policy['active_names']}
    heads = {} if readout is None else dict(readout.to('cuda:0').named_parameters())
    head_opt = None if not heads else torch.optim.AdamW([dict(params=[p]) for p in heads.values()], lr=1e-4, weight_decay=.05, betas=(.9,.999), eps=1e-8)
    require(len(native_opt.state) == 0 and len(native_opt.param_groups) == len(native_named) == 345 and
        (head_opt is None or len(head_opt.state) == 0 and len(heads) == 8 and sum(p.numel() for p in heads.values()) == 9441), 'Fresh optimizer membership differs')
    fresh_native_entries = len(native_opt.state)
    fresh_head_entries = 0 if head_opt is None else len(head_opt.state)
    names = [[row['name']] for row in native_audit['parameter_groups']]
    require(set(n[0] for n in names) == set(native_named), 'Original native paramwise groups differ')
    base_lrs = [g['lr'] for g in native_opt.param_groups]
    before_head = pref.cpu_tree(torch, {} if readout is None else readout.state_dict())
    require(not heads or len(before_head) == 8 and len(list(readout.buffers())) == 0 and
        all(bool(torch.isfinite(v).all()) for v in before_head.values()), 'Own warm B8 complete finite no-buffer head required')
    frozen = {n: model.state_dict()[n].detach().clone() for n in policy['frozen_names'] + policy['BN_buffer_names']}
    hooks_before = {n: (tuple(v._forward_pre_hooks), tuple(v._forward_hooks)) for n, v in model.named_modules()}
    activities = {kind: {n: 0 for n in ns} for kind, ns in (('native', native_named), ('readout', heads))}
    counts = {kind: {n: 0 for n in ns} for kind, ns in (('native', native_named), ('readout', heads))}
    external_initial = pref.rng_snapshot(torch)
    write(output/'START_STATE_PROOF.json', dict(schema=PREFIX+'-start-state-v1', source=source, spec=spec_ref, arm=arm,
        teacher_loading=loading, actual_native610=pref.state_proof(anchor), policy=policy, native_optimizer=native_audit,
        actual_native_optimizer=common.optimizer_summary(torch, pref, native_opt, names), base_lrs=base_lrs,
        initial_readout=pref.state_proof(before_head), readout_optimizer=HEAD_RECIPE if heads else None,
        fresh_native_optimizer_state_entries=fresh_native_entries, fresh_readout_optimizer_state_entries=fresh_head_entries, old_optimizer_restored=False,
        old_RNG_restored=False, technical_weights_loaded=False, phase_microsteps=0, phase_updates=0,
        initial_native_epoch=48, phase_initial_epoch=0, auxiliary_weight=spec['auxiliary_weight'],
        weights_warm_start=True, warm_checkpoint=spec['warm_checkpoints'][arm], first24_CPU_qualification=spec['first24_CPU_qualification'],
        effective_native_epoch_at_phase_end=96, effective_B_readout_epoch_at_phase_end=72))
    order_rng = torch.Generator(device='cpu').manual_seed(spec['order_seed'])
    orders, epochs, audits = [], [], []
    updates, micro_count, GT_visits, boundary_count, pending_boundary = 0, 0, 0, 0, None
    native_opt.zero_grad(set_to_none=True)
    if head_opt is not None: head_opt.zero_grad(set_to_none=True)
    try:
        with (output/'TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for epoch in range(48):
                before_rng = order_rng.get_state().clone(); order = torch.randperm(741, generator=order_rng).tolist()
                require(sorted(order) == list(range(741)), 'Complete private741 permutation required')
                orders.append(dict(phase_epoch=epoch, order_seed=spec['order_seed'], dataset_indices=order,
                    image_ids=[int(dataset.get_data_info(i)['img_id']) for i in order],
                    private_CPU_RNG_before=pref.digest_tensor(before_rng), private_CPU_RNG_after=pref.digest_tensor(order_rng.get_state())))
                epoch_GT, initial_updates = 0, updates
                epoch_losses = {k: [] for k in ('native', 'auxiliary', 'total')}
                for within, index in enumerate(order):
                    pref.check_deadline(started, spec['deadline_seconds']); g = epoch*741+within
                    require(g == micro_count, 'Global micro counter differs')
                    boundary_join = None
                    if within == 0 and g % 2 == 1:
                        observed = pending_gradients(pref, native_named, heads)
                        require(pending_boundary is not None and pending_boundary['global_micro'] == g-1 and
                            observed == pending_boundary['gradients'], 'Odd741 pending gradients changed before next forward')
                        boundary_join = dict(previous_global_micro=g-1, after_previous_odd_epoch=pending_boundary['gradients'],
                            before_next_forward=observed, exact=True, SOURCE_observed_not_CPU_gradient_replay=True)
                        boundary_count += 1; pending_boundary = None
                    factor = common.factor(SCHEDULE, updates)
                    for group, base in zip(native_opt.param_groups, base_lrs): group['lr'] = base*factor
                    data, stripped, gray, raw_gray, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
                    del stripped, gray, raw_gray
                    identity['dataset_index'] = index
                    target = pref.check_targets(torch, data, identity)
                    raw_GT, mapping, validator = tiny.raw_GT_validator(torch, np, cv2, coco, data, identity)
                    point_rng = torch.Generator(device='cpu').manual_seed(spec['point_seed_base']+g)
                    indices = torch.randint(raw_GT.shape[1]*raw_GT.shape[2], (len(raw_GT), 12544), generator=point_rng, dtype=torch.int64).to('cuda:0')
                    detail = g in DETAILS; packet = None
                    external = pref.rng_snapshot(torch)
                    with torch.random.fork_rng(devices=[0]):
                        seed = spec['loss_seed_base']+g
                        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
                        private_before = pref.rng_snapshot(torch)
                        r = joint.joint_native_losses(torch, pref, common, old, capture, adapter, objective, model, data, readout,
                            partial(diag.render_original, torch), identity, identity['ann_ids'], raw_GT, mapping, validator, indices, arm, g)
                        live = r['live_native_fields']; L = r['auxiliary_fields']['native_logits']
                        loss_values = dict(native=float(r['native_total'].detach()), auxiliary=float(r['auxiliary'].detach()), total=float(r['total'].detach()))
                        require(r['metadata']['auxiliary_weight'] == spec['auxiliary_weight'] == 1., 'Lambda1 native+auxiliary recipe differs')
                        if detail:
                            L.retain_grad()
                            for n in ('F256', 'e0', 'M200'): live[n].retain_grad()
                            packet = pref.cpu_tree(torch, dict(schema=PREFIX+'-audit-v1', arm=arm, global_micro=g, identity=identity,
                                raw_GT=raw_GT, query_ids=r['query_ids'], point_indices=indices, F256=live['F256'], e0=live['e0'], M200=live['M200'],
                                readout_before={} if readout is None else readout.state_dict(),
                                readout_pending_before={n: None if p.grad is None else p.grad for n, p in heads.items()},
                                readout_optimizer_before=None if head_opt is None else head_opt.state_dict(),
                                auxiliary_loss=loss_values['auxiliary'], native_total=loss_values['native'], metadata=r['metadata'],
                                auxiliary_gradient_scale=.5, native_gradient_replay_claimed=False))
                        (r['total']/2).backward()  # One total backward; no extra autograd.grad.
                        private_after = pref.rng_snapshot(torch)
                    external_after = pref.rng_snapshot(torch)
                    require(pref.tree_equal(torch, external, external_after), 'External private-loss RNG not restored')
                    norms, head_norms = gradient_norms(torch, native_named), gradient_norms(torch, heads)
                    require(any(v is not None for v in norms.values()) and all(v is not None for v in head_norms.values()),
                        'Native gradients/B8 connected finite gradients required; zero is permitted')
                    require(all(named[n].grad is None for n in policy['frozen_names']), 'Frozen BN affine received gradients')
                    BN_modes = {n: v.training for n, v in model.named_modules() if n in policy['BN_modules']}
                    require(not any(BN_modes.values()) and all(torch.equal(model.state_dict()[n], v) for n, v in frozen.items()), 'Typed BN state/mode changed')
                    for kind, values in (('native', norms), ('readout', head_norms)):
                        for n, value in values.items(): activities[kind][n] += int(value is not None and value > 0)
                    if detail:
                        require(L.grad is not None and bool(torch.isfinite(L.grad).all()) and
                            all(live[n].grad is not None and bool(torch.isfinite(live[n].grad).all()) for n in ('F256','e0','M200')), 'Audit live/local gradients absent/nonfinite')
                        packet.update(auxiliary_L_gradient=L.grad[0,r['query_ids']].detach().cpu().clone(),
                            readout_accumulated_before_clip={n:p.grad.detach().cpu().clone() for n,p in heads.items()},
                            native_live_total_gradient_norms={n:float(live[n].grad.norm()) for n in ('F256','e0','M200')})
                    stepped, global_norm = (g+1)%2 == 0, None
                    if stepped:
                        global_norm = float(torch.nn.utils.clip_grad_norm_(list(native_named.values())+list(heads.values()), .01, norm_type=2))
                        require(math.isfinite(global_norm) and global_norm >= 0, 'Combined global clip nonfinite')
                        if detail: packet['readout_clipped_gradients'] = {n:p.grad.detach().cpu().clone() for n,p in heads.items()}
                        for kind, values in (('native', norms), ('readout', head_norms)):
                            for n, value in values.items(): counts[kind][n] += int(value is not None)
                        native_opt.step()
                        if head_opt is not None: head_opt.step()
                        native_opt.zero_grad(set_to_none=True)
                        if head_opt is not None: head_opt.zero_grad(set_to_none=True)
                        updates += 1
                    if detail:
                        packet.update(combined_preclip_norm=global_norm, readout_clipped_gradients=packet.get('readout_clipped_gradients'),
                            readout_after=pref.cpu_tree(torch, {} if readout is None else readout.state_dict()),
                            readout_optimizer_after=None if head_opt is None else pref.cpu_tree(torch, head_opt.state_dict()),
                            optimizer_update=stepped, native_updates=updates, readout_updates=updates if heads else 0)
                        audits.append(save_packet(torch, pref, output/('MICRO_%d.pth'%g), packet))
                    epoch_GT += len(identity['ann_ids']); GT_visits += len(identity['ann_ids']); micro_count += 1
                    odd_epoch_end = None
                    if within == 740 and micro_count%2 == 1:
                        require(not stepped and pending_boundary is None, 'Odd741 must retain its pending micro')
                        odd_epoch_end = pending_gradients(pref, native_named, heads)
                        pending_boundary = dict(global_micro=g, gradients=odd_epoch_end)
                    for key in epoch_losses: epoch_losses[key].append(loss_values[key])
                    row = dict(arm=arm, global_micro=g, phase_epoch=epoch, effective_native_epoch_index=48+epoch, within_epoch=within,
                        identity=identity, GT=target, input=pref.digest_tensor(data['inputs']), point_indices=pref.digest_tensor(indices),
                        point_seed=spec['point_seed_base']+g, point_count_per_GT=12544, sampling_with_replacement=True,
                        final_query_ids=r['metadata']['final_query_by_raw_GT'], metadata=r['metadata'], losses=loss_values,
                        native_gradient_norms=norms, readout_gradient_norms=head_norms, gradients_accumulated_before_clip=True,
                        auxiliary_weight=1., accumulation_divisor=2, optimizer_update=stepped, phase_updates=updates,
                        readout_updates=updates if heads else 0, scheduler_u_before_step=updates-int(stepped), native_lr_factor=factor,
                        B_readout_lr=1e-4 if heads else None, combined_preclip_norm=global_norm, actual_backward_calls=1,
                        frozen_BN_parameter_grads_None=True, BN_module_training=BN_modes, detailed_observation=detail,
                        odd_epoch_end_pending_gradients=odd_epoch_end, pending_boundary_gradient_join=boundary_join,
                        private_seed=seed, private_RNG_before=[pref.digest_tensor(v) for v in private_before],
                        private_RNG_after=[pref.digest_tensor(v) for v in private_after],
                        external_RNG_before=[pref.digest_tensor(v) for v in external], external_RNG_after=[pref.digest_tensor(v) for v in external_after],
                        external_RNG_restored=True, dynamic_current_matching=True)
                    stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
                    if micro_count%100 == 0 or micro_count == 35568:
                        stream.flush(); print(json.dumps(dict(phase=spec['phase'],arm=arm,microsteps=micro_count,updates=updates,
                            raw_GT_visits=GT_visits,elapsed_seconds=time.monotonic()-started),sort_keys=True),flush=True)
                    del data, raw_GT, validator, indices, r, live, L, packet, external, external_after, private_before, private_after
                require(epoch_GT == 1661 and updates == ((epoch+1)*741)//2, 'Complete741/1661 globalacc2 epoch required')
                epochs.append(dict(phase_epoch=epoch, images=741, raw_GT=epoch_GT, phase_microsteps=micro_count,
                    epoch_updates=updates-initial_updates, phase_updates=updates, pending_accumulation_microsteps=micro_count%2,
                    training_loss_summary={k:dict(image_loss_sum=math.fsum(v),image_loss_mean=math.fsum(v)/741,
                        mean_denominator='741 image-averaged losses') for k,v in epoch_losses.items()}))
        require(micro_count == 35568 and updates == 17784 and GT_visits == 79728 and len(epochs) == 48 and
            boundary_count == 24 and pending_boundary is None and all(p.grad is None for p in list(named.values())+list(heads.values())),
            'Complete phase counters/24 odd741 joins/no pending gradients required')
        require(hooks_before == {n:(tuple(v._forward_pre_hooks),tuple(v._forward_hooks)) for n,v in model.named_modules()} and
            pref.tree_equal(torch, external_initial, pref.rng_snapshot(torch)), 'Hooks/external RNG leaked')
        optimizer_steps(torch, native_opt, native_named, counts['native'])
        if head_opt is not None: optimizer_steps(torch, head_opt, heads, counts['readout'])
        final = pref.cpu_tree(torch, model.state_dict()); final_head = pref.cpu_tree(torch, {} if readout is None else readout.state_dict())
        require(set(final) == set(anchor) and len(final) == 610 and all(bool(torch.isfinite(v).all()) for v in final.values()) and
            all(torch.equal(final[n],anchor[n]) for n in frozen) and all(bool(torch.isfinite(v).all()) for v in final_head.values()), 'Finite full610/typedBN/head endpoint differs')
        write(output/'ORDER.json', orders); write(output/'EPOCH_ROWS.json', epochs)
        refs = dict(trace=reference(output/'TRACE.jsonl'), order=reference(output/'ORDER.json'), epochs=reference(output/'EPOCH_ROWS.json'),
            start=reference(output/'START_STATE_PROOF.json'), audits=audits)
        checkpoint = dict(schema=PREFIX+'-checkpoint-v1', native_state_dict=final, readout_state_dict=final_head,
            native_optimizer_state_dict=pref.cpu_tree(torch,native_opt.state_dict()),
            readout_optimizer_state_dict=None if head_opt is None else pref.cpu_tree(torch,head_opt.state_dict()),
            private_order_state=dict(seed=spec['order_seed'],generated_phase_epochs=48,torch_CPU_generator_state=order_rng.get_state().clone()),
            external_rng_state=pref.cpu_tree(torch,pref.rng_snapshot(torch)),
            meta=dict(source=source,spec=spec_ref,phase=spec['phase'],arm=arm,teacher_checkpoint=spec['checkpointmap']['native0'],
                teacher_qualification=spec['qualificationmap']['native0'],initial_native_epoch=48,completed_phase_epochs=48,effective_native_epoch=96,
                phase_microsteps=micro_count,phase_updates=updates,phase_raw_GT_visits=GT_visits,policy=policy,native_optimizer=native_audit,
                native_optimizer_param_names=names,readout_optimizer_param_names=[[n] for n in heads],base_lrs=base_lrs,
                stored_native_lr_factor=common.factor(SCHEDULE,updates-1),schedule=SCHEDULE,readout_optimizer=HEAD_RECIPE if heads else None,
                order_seed=spec['order_seed'],loss_seed_base=spec['loss_seed_base'],point_seed_base=spec['point_seed_base'],readout_seed=spec['readout_seed'],
                auxiliary_weight=1.,kappa=1.,eps=1e-6,accumulation=2,combined_clip_max_norm=.01,fresh_native_optimizer_state_entries=fresh_native_entries,
                original_teacher_epoch=24, completed_joint_training_epochs=72, effective_readout_epoch=72 if heads else 0,
                weights_warm_start=True,warm_checkpoint=spec['warm_checkpoints'][arm],first24_CPU_qualification=spec['first24_CPU_qualification'],
                fresh_readout_optimizer_state_entries=fresh_head_entries,old_optimizer_restored=False,old_RNG_restored=False,technical_weights_loaded=False,
                no_pending_accumulation_gradients=True,formal_resume_eligible=False,AP_computed=False,artifacts=refs))
        cp_path = output/'FINAL_PHASE_E48_EFFECTIVE_E96.pth'
        with cp_path.open('xb') as saved:
            torch.save(checkpoint,saved)
        cp_ref = reference(cp_path)
        require(pref.tree_equal(torch,checkpoint,torch.load(str(cp_path),map_location='cpu')), 'Full CPU checkpoint/optimizer/RNG readback differs')
        proof = dict(schema=PREFIX+'-endpoint-proof-v1',source=source,spec=spec_ref,arm=arm,checkpoint=cp_ref,artifacts=refs,
            before_native610=pref.state_proof(anchor),after_native610=pref.state_proof(final),
            changed_native={n:not torch.equal(v,anchor[n]) for n,v in final.items()},before_readout=pref.state_proof(before_head),
            after_readout=pref.state_proof(final_head),policy=policy,teacher_loading=loading,native_optimizer=native_audit,
            actual_native_optimizer=common.optimizer_summary(torch,pref,native_opt,names),
            actual_readout_optimizer=None if head_opt is None else common.optimizer_summary(torch,pref,head_opt,[[n] for n in heads]),
            positive_accumulated_gradient_microcounts=activities,optimizer_step_nonNone_gradient_counts=counts,
            completed_phase_epochs=48,effective_native_epoch=96,phase_microsteps=micro_count,phase_updates=updates,phase_raw_GT_visits=GT_visits,
            original_teacher_epoch=24,completed_joint_training_epochs=72,effective_readout_epoch=72 if heads else 0,
            weights_warm_start=True,warm_checkpoint=spec['warm_checkpoints'][arm],first24_CPU_qualification=spec['first24_CPU_qualification'],
            frozen_BN106_affine159_buffers_exact=True,no_pending_accumulation_gradients=True,odd741_gradient_joins=boundary_count,
            fresh_native_optimizer_state_entries=fresh_native_entries,fresh_readout_optimizer_state_entries=fresh_head_entries,
            native_loss_atoms=30,ten_current_assignments_per_micro=True,one_total_backward_per_micro=True,
            four_audit_microsteps=list(DETAILS),CPU_checkpoint_readback_exact=True,technical_weights_loaded=False,
            old_optimizer_restored=False,old_RNG_restored=False,formal_resume_eligible=False,native30_CPU_replay_claimed=False,
            full_optimizer_trajectory_CPU_replay_claimed=False,conditioning_isolated_claimed=False,compute_matched_claimed=False)
        write(output/'ENDPOINT_PROOF.json',proof)
        return dict(arm=arm,checkpoint=cp_ref,endpoint_proof=reference(output/'ENDPOINT_PROOF.json'),**refs,
            completed_phase_epochs=48,effective_native_epoch=96,phase_microsteps=micro_count,phase_updates=updates,phase_raw_GT_visits=GT_visits,
            native_parameter_tensors=451,native_state_tensors=610,native_trainable_tensors=345,readout_parameter_tensors=len(heads),
            readout_parameters=sum(p.numel() for p in heads.values()),fresh_native_optimizer_state_entries=fresh_native_entries,
            weights_warm_start=True,warm_checkpoint=spec['warm_checkpoints'][arm],effective_readout_epoch=72 if heads else 0,
            fresh_readout_optimizer_state_entries=fresh_head_entries,odd741_gradient_joins=24,no_pending_accumulation_gradients=True)
    finally:
        native_opt.zero_grad(set_to_none=True)
        if head_opt is not None: head_opt.zero_grad(set_to_none=True)
        model.cpu()
        if readout is not None: readout.cpu()


def run(spec_file):
    started=time.monotonic(); path=Path(spec_file)
    require(path.is_absolute(), 'Absolute --spec required'); path=path.resolve(strict=True); spec=read(path)
    require(spec['schema']==SCHEMA and spec['kind']=='mask2former' and spec['teacher_epoch']==24 and spec['arms']==list(ARMS) and
        all(type(spec[k])is int and spec[k]==v for k,v in COUNTERS.items()) and spec['accumulation']==2 and spec['auxiliary_weight']==1. and
        spec['schedule']==SCHEDULE and spec['B_readout_optimizer']==HEAD_RECIPE and spec['B_readout_LR_schedule'] is False and
        spec['order_seed']==2026101021 and spec['loss_seed_base']==1000000 and spec['point_seed_base']==3800000 and spec['readout_seed']==2026100911 and
        all(type(spec[k])is int for k in ('order_seed','loss_seed_base','point_seed_base','readout_seed')) and
        spec['CPU_affinity']==[0,1] and spec['allocator_limit_MiB']==4096 and spec['whole_GPU_owner_limit_MiB']==8192 and
        type(spec['deadline_seconds'])is int and 0<spec['deadline_seconds']<=43200 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and
        spec['CPU_numeric_protocol']==NUMERIC and spec['kappa']==1. and spec['eps']==1e-6 and spec['backbone_with_cp'] is True and
        spec['no_feature_cache'] is True and spec['no_DEV_selection'] is True, 'Prelocked formal joint SPEC differs')
    source,spec_ref,admission,tiny=source_gate(path,spec)
    legacy=load(spec['evaluation_source'],'qualified_original_E24_lineage_ONLY'); teachers=legacy.qualified_training(spec)
    for row in teachers.values():
        q=row['qualification']
        for ref in row['training_spec']['sources']+[q[k] for k in ('source','reader_source','spec','report','start','GPU_admission','owner_driver')]+list(q['model_proof']['proof_refs'].values()):
            require(ref in spec['sources'],'Complete original teacher/proof lineage absent from current owner closure')
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(),'Fresh absolute output required'); output.mkdir(parents=True)
    write(output/'START.json',dict(schema=PREFIX+'-start-v1',source=source,spec=spec_ref,GPU_admission=admission,phase=spec['phase'],
        arms=list(ARMS),teacher_checkpoint=spec['checkpointmap']['native0'],teacher_qualification=spec['qualificationmap']['native0'],
        joint_preflight_CPU_qualification=spec['joint_preflight_CPU_qualification'],technical_weights_loaded=False,
        first24_CPU_qualification=spec['first24_CPU_qualification'],warm_checkpoints=spec['warm_checkpoints'],weights_warm_start=True,
        initial_native_epoch=48,completed_phase_epochs_at_endpoint=48,effective_native_epoch_at_endpoint=96,effective_B_readout_epoch_at_endpoint=72,
        planned_fresh_native_optimizer_state_entries=0,planned_fresh_readout_optimizer_state_entries=0,old_optimizer_restored=False,old_RNG_restored=False,
        counters_per_arm=COUNTERS,schedule=SCHEDULE,B_readout_optimizer=HEAD_RECIPE,B_readout_LR_schedule=False,
        order_seed=spec['order_seed'],loss_seed_base=spec['loss_seed_base'],point_seed_base=spec['point_seed_base'],readout_seed=spec['readout_seed'],
        auxiliary_weight=1.,accumulation=2,four_audit_microsteps=list(DETAILS),no_DEV=True,no_TEST=True,no_AP=True))
    import psutil
    import torch
    import numpy as np
    import cv2
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity()==[0,1] and not torch.cuda.is_initialized(),'CPU01/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes'])==80 and CocoDataset.METAINFO['classes'][8]=='boat','boat8/null80 required')
    m=[load(spec[k],'_formal_joint_'+k) for k in MODULES]
    native,legacy,pref,initializer,common,old,diag,posterior,capture,joint,adapter,objective=m; old.torch=torch
    construction=dict(teachers['native0']['training_spec'],training_source=spec['training_source'],checkpointmap=spec['checkpointmap'],
        qualificationmap=spec['qualificationmap'],order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    require(construction['FIT']==spec['FIT'] and construction['image_root']==spec['image_root'] and
        construction['backbone_with_cp'] is spec['backbone_with_cp'] is True and
        spec['initializers']==construction['initializers'] and construction['schedule']==native.SCHEDULE,
        'Original FIT/native paramwise initializer/checkpoint/schedule policy differs')
    prepared={arm:native.prepare_training_model(torch,pref,initializer,common,MODELS,revert_sync_batchnorm,Config,
        tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0']) for arm in ARMS}
    require(pref.tree_equal(torch,prepared['A'][1],prepared['B'][1]) and prepared['A'][2:]==prepared['B'][2:],
        'Independent A/B originalE24 initialization differs')
    heads={'A':None,'B':posterior.QueryPixelPosteriorReadout(spec['readout_seed'],spec['kappa'],spec['eps'])}
    warm_weights(torch,pref,spec,prepared,heads)  # CPU weights only; no prior optimizer/order/RNG restored.
    for arm in ARMS: native.imported_library_gate(torch,prepared[arm][0],spec)
    dataset=CocoDataset(ann_file=str(bound(spec['FIT'])),data_prefix=dict(img=spec['image_root']),metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(bound(spec['FIT']))); require(len(dataset)==len(coco.imgs)==741 and len(coco.anns)==1661,'Original fullFIT741/1661 required')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0); torch.cuda.reset_peak_memory_stats(0)
    result,error,cleanup={},None,None
    try:
        for arm in ARMS:
            result[arm]=arm_run(torch,np,cv2,m,tiny,prepared[arm],heads[arm],construction,spec,source,spec_ref,output/arm,dataset,coco,started,arm)
            del prepared[arm],heads[arm]
            arm_cleanup=pref.clear_gpu(torch); require(arm_cleanup['allocated_after']==0,'Serial arm CUDA cleanup failed')
            result[arm]['cleanup']=arm_cleanup
        del prepared,heads; cleanup=pref.clear_gpu(torch)
        require(result['A']['order']['sha256']==result['B']['order']['sha256'],'Paired private order stream differs')
        for ref in spec['sources']: bound(ref)
        require(reference(path)==spec_ref and time.monotonic()-started<spec['deadline_seconds'],'Final bindings/deadline differ')
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),no_retry=True); traceback.clear_frames(exc.__traceback__)
        if 'prepared' in locals():
            for row in prepared.values(): row[0].cpu()
            del prepared
        if 'heads' in locals():
            for head in heads.values():
                if head is not None: head.cpu()
            del heads
        gc.collect(); cleanup=pref.clear_gpu(torch)
    report=dict(schema=PREFIX+'-report-v1',status='QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_GPU_COMPLETE_PENDING_CPU' if error is None else 'QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_STOP',
        source=source,spec=spec_ref,GPU_admission=admission,phase=spec['phase'],result=result,error=error,cleanup=cleanup,counters_per_arm=COUNTERS,
        CPU_numeric_protocol=NUMERIC,schedule=SCHEDULE,B_readout_optimizer=HEAD_RECIPE,B_readout_LR_schedule=False,auxiliary_weight=1.,
        actual_total_backward_calls=71136 if error is None else None,extra_backward_or_autograd_grad=False,serial_arms=True,
        native_AdamW_groups=345,B_readout_AdamW_groups=8,separate_AdamW_optimizers=True,combined_gradient_clip=True,
        technical_weights_loaded=False,old_optimizer_restored=False,old_RNG_restored=False,no_DEV=True,no_TEST=True,no_AP=True,
        first24_CPU_qualification=spec['first24_CPU_qualification'],warm_checkpoints=spec['warm_checkpoints'],weights_warm_start=True,
        initial_native_epoch=48,completed_phase_epochs=48,effective_native_epoch=96,effective_B_readout_epoch=72,
        DEV_evaluation_separate=True,formal_resume_eligible=False,native30_CPU_replay_claimed=False,full_optimizer_trajectory_CPU_replay_claimed=False,
        efficacy_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False,
        elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    write(output/'REPORT.json',report)
    if error is None:
        write(output/'QUALIFIED.json',dict(schema=PREFIX+'-aggregate-v1',status='ROOT_QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_PENDING_CPU_READER',
            source=source,spec=spec_ref,report=reference(output/'REPORT.json'),result=result,phase=spec['phase'],
            independently_CPU_qualified=False,AP_computed=False,formal_resume_eligible=False))
    require(error is None,'FORMAL_STOP: '+str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned execution required'); run(args.spec)


if __name__ == '__main__':
    main()
