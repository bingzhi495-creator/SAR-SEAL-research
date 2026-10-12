"""Inert ROOT-owned fresh single-arm, complete24 spatial-mask training source.

Actual modes native0/global/spatial_halves/spatial_reversed share fresh COCO
initialization, all741 FIT images per pass, order2026100917, native30/AdamW,
batch1/acc2, full native convolution training and frozen typed BN. No Sobel,
auxiliary loss, GT geometry, inference, DEV, AP, retry or resume. All actual
17784 microsteps/8892 updates/39864 raw GT visits must finish in one owned
attempt. Six fixed even nodes snapshot full CPU model/optimizer/RNG and
readback without rebuilding, zero_grad, reseeding or interrupting training.

Native0 is bare610; spatial modes register actual readout and original
geometry modules, adding only two weights. Whole-head final mask replacement
keeps class list and earlier nine mask objects/gates. Realized native features,
geometry, eligibility, scores and assignments may diverge during joint
training. Global uses one projection; spatial/reversed use two, so equal
parameters/data/updates do not assert equal whole computation or efficacy.
Explicit owner-execute/current admission and ROOT-qualified native technical
seam are required; technical CPs and all old arm CPs are refused. SOURCE only.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback
from contextlib import contextmanager, nullcontext

SPEC_SCHEMA = 'ROOT-native-spatial-position-preserving-common24-spec-v1'
REPORT_SCHEMA = 'native-spatial-position-preserving-common24-report-v1'
CP_SCHEMA = 'native-spatial-position-preserving-common24-checkpoint-v1'
KINDS = ('mask2former',)
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
ADAPTER_PREFIX = 'panoptic_head.spatial_mask_readout.'
FIT_IMAGES, FIT_GT, TOTAL_EPOCHS = 741, 1661, 24
TOTAL_MICROSTEPS, TOTAL_UPDATES, TOTAL_GT_VISITS = 17784, 8892, 39864
CHECKPOINT_NODES = (2, 4, 8, 12, 18, 24)
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
READOUT_SHA = '7da9561ed30675fd0d692ce2ad883e0c8cccfddfd9f7913761da0fe4fff0a09d'
GEOMETRY_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
HOOKS_SHA = '23f36f3fab9b4a006bd0db65427b68223dc99c4929da458504ed5a3aa9185806'



def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(Path(row['path']).is_absolute(), 'Absolute bound path required')
    actual = reference(row['path'])
    require(actual == {key: row[key] for key in ('path', 'bytes', 'sha256')}, 'Pinned input differs')
    return Path(actual['path'])


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load_source(row, name):
    path = bound(row)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def owner_source_gate(spec_path, spec):
    """No scientific hash/import before explicit current owned-admission metadata."""
    spec_path = Path(spec_path).resolve(strict=True)
    require(spec['phase'].startswith('native_spatial_position_preserving_common24_') and
            Path(spec['GPU_admission_file']).is_absolute(), 'Fresh owned phase/admission required')
    admission_path = Path(spec['GPU_admission_file']).resolve(strict=True)
    admission = json.loads(admission_path.read_text(encoding='utf-8-sig'))
    entry, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
            Path(admission['entry']).resolve(strict=True) == Path(entry['path']) and
            admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
            admission['driver_sha256'] == OWNER_SHA,
            'STOP before SCI: fresh explicit owner admission/entry/resources differ')
    require(spec['entry_source'] == entry and entry in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
            admission['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'],
            'Current entry/consumer/SPEC are not owner-bound')
    for source in spec['sources']:
        require(admission['bound_source_sha256'].get(source['path']) == source['sha256'],
                'Current preflight source closure absent from owner admission')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Owner scientific SPEC differs')


def factor(schedule, completed_updates):
    """Before next update u=completed_updates: linear warmup times milestone decay."""
    u, warmup = completed_updates, schedule['warmup_updates']
    multiplier = (schedule['warmup_start_factor'] +
        (1. - schedule['warmup_start_factor']) * u / warmup) if u < warmup else 1.
    return multiplier * schedule['gamma'] ** sum(m <= u for m in schedule['milestones'])


def order_plan(np, seed):
    rng = np.random.RandomState(seed)
    orders, states = [], []
    for epoch in range(TOTAL_EPOCHS + 1):
        state = rng.get_state()
        states.append([state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])])
        if epoch < TOTAL_EPOCHS:
            orders.append(rng.permutation(FIT_IMAGES).tolist())
    require(all(sorted(order) == list(range(FIT_IMAGES)) for order in orders), 'Every epoch must cover741 exactly')
    return dict(seed=seed, orders=orders, RNG_states_at_epoch_boundaries=states,
        order_sha256=json_sha(orders), total_epochs=TOTAL_EPOCHS, private_numpy_rng=True)


def training_protocol(spec, order_sha):
    return dict(kind=spec['kind'], mode=spec['mode'],
        checkpoint=spec['initializers']['mask2former']['checkpoint'],
        config=spec['initializers']['mask2former']['config'], FIT=spec['FIT'], image_root=spec['image_root'],
        initializers_policy=spec['initializers']['mask2former'], native_helpers=spec['native_helpers'],
        initializer_source=spec['initializer_source'], micro_preflight_source=spec['micro_preflight_source'],
        training_source=reference(__file__), common_schedule_source=spec['common_schedule_source'],
        readout_source=spec['readout_source'], geometry_source=spec['geometry_source'],
        hooks_source=spec['hooks_source'], native_preflight_source=spec['native_preflight_source'],
        native_preflight_qualification=spec['native_preflight_qualification'],
        adapter_factory_hidden_seed=2026100911 if spec['mode'] != 'native0' else None,
        backbone_with_cp=spec['backbone_with_cp'], schedule=spec['schedule'],
        order_seed=spec['order_seed'], order_sha256=order_sha, total_epochs=24,
        total_microsteps=17784, total_updates=8892, total_raw_GT_visits=39864,
        checkpoint_nodes=list(CHECKPOINT_NODES), batch_size=1, accumulation=2, loss_seed_base=1000000,
        no_augmentation=True, full_R50_conv=True, typed_BN_frozen=True,
        native_loss_only=True, native_loss_atoms=30, Sobel=False,
        final_mask_logit_readout=spec['mode'] != 'native0', adapter_geometry_has_no_GT_input=True,
        realized_cross_mode_points_or_eligibility_equal_claimed=False,
        equal_whole_compute_claimed=False, scheduler_index='completed_updates_before_next_step',
        fresh_single_owned_24=True, resume_allowed=False)



def optimizer_summary(torch, pref, optimizer, names):
    state = pref.cpu_tree(torch, optimizer.state_dict())
    require(len(state['param_groups']) == len(names), 'Optimizer name/group length mismatch')
    groups, named_state = [], {}
    for group, group_names in zip(state['param_groups'], names):
        require(len(group['params']) == len(group_names) == 1, 'Qualified one-parameter groups required')
        groups.append(dict(names=group_names, settings={key: value for key, value in group.items() if key != 'params'}))
        for identifier, name in zip(group['params'], group_names):
            named_state[name] = {key: dict(tensor=pref.digest_tensor(value), scalar=float(value) if value.ndim == 0 else None)
                if isinstance(value, torch.Tensor) else dict(value=value)
                for key, value in state['state'].get(identifier, {}).items()}
    return dict(parameter_groups=groups, named_state=named_state,
        state_entries=len(state['state']), optimizer_state_sha256=json_sha(dict(groups=groups, state=named_state)))


@contextmanager
def observe_native_loss(torch, pref, old, readout, hooks, head, data, mode, global_micro):
    """Passive final-mask observations; all GT use remains original native loss."""
    record = dict(head_fields=None, spatial_hook=None, adapter_diagnostics=None, matches=None)
    handles = []
    def head_fields(module, inputs, output):
        require(record['head_fields'] is None and len(output) == 2 and
                len(output[0]) == len(output[1]) == 10 and
                all(tuple(x.shape) == (1,100,2) and x.requires_grad for x in output[0]) and
                all(tuple(x.shape) == (1,100,200,200) and x.requires_grad for x in output[1]),
                'Expected one live ten-readout head including actual spatial final mask')
        record['head_fields'] = dict(readouts=10, classes_shape=[1,100,2], masks_shape=[1,100,200,200],
            all_fields_require_grad=True,
            cold_micro_digests=(dict(classes=[pref.digest_tensor(x) for x in output[0]],
                masks=[pref.digest_tensor(x) for x in output[1]]) if global_micro == 0 else None))
        return None
    def adapter_fields(module, inputs, output):
        require(record['adapter_diagnostics'] is None and inputs[0].requires_grad and
                inputs[1].requires_grad and output['logit_residual'].requires_grad and
                output['mask_logits'].requires_grad, 'Exactly one live final spatial readout required')
        d = output['diagnostics']; counts = d['eligible_counts']
        require(tuple(counts.shape) == (1,100) and tuple(d['half_eligible_counts'].shape) == (1,100,2),
                'Actual all100 geometry/halfcounts required')
        record['adapter_diagnostics'] = dict(mode=d['mode'], eligibility_frame=d['eligibility_frame'],
            all_queries=100, queries_with_support=int((d['rho_mass'] > 1e-12).sum().cpu()),
            queries_with_eligible_pairs=int((counts > 0).sum().cpu()), eligible_pairs_total=int(counts.sum().cpu()),
            eligible_counts_by_query=counts[0].detach().cpu().tolist(),
            canonical_eligible_pairs_total=int(d['canonical_eligible_counts'].sum().cpu()),
            half_eligible_counts=d['half_eligible_counts'][0].detach().cpu().tolist(),
            global_fallback=d['global_fallback'][0].detach().cpu().tolist(),
            input_F_live=True, input_e0_live=True, residual_live=True, geometry_stop_gradient=True,
            GT_used_for_geometry=False, no_query_dropped=True)
        return None
    try:
        if mode != 'native0':
            handles.append(head.spatial_mask_readout.register_forward_hook(adapter_fields))
        scope = nullcontext(None) if mode == 'native0' else hooks.spatial_readout_context(
            torch,head,[sample.metainfo for sample in data['data_samples']])
        with scope as hook_record, old.observe_targets(head) as targets:
            # Register after the whole-head spatial finish hook, not before it.
            handles.append(head.register_forward_hook(head_fields))
            yield record
        record['matches'] = list(targets)
        record['spatial_hook'] = dict(hook_record) if hook_record is not None else None
        require(record['head_fields'] is not None and len(record['matches']) == 10 and
                ((record['adapter_diagnostics'] is None) == (mode == 'native0')),
                'Original ten native assignments/final observer incomplete')
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()



def save_node(torch, pref, model, optimizer, spec, spec_ref, output, epoch, native_cold, cold,
              policy, audit, names, base_lrs, protocol, plan, activity, update_nonNone,
              micro_count, GT_visits, completed_updates, initialization):
    """CPU snapshot only; never move/rebuild the live model or clear its optimizer."""
    require(epoch in CHECKPOINT_NODES and micro_count == epoch*FIT_IMAGES and
            GT_visits == epoch*FIT_GT and completed_updates == epoch*FIT_IMAGES//2 and
            all(value.grad is None for value in model.parameters()), 'Node counters/pending gradients differ')
    final = pref.cpu_tree(torch,model.state_dict())
    require(final.keys() == cold.keys() and all(bool(torch.isfinite(v).all()) for v in final.values()),
            'Node full state schema/finite differs')
    frozen = policy['frozen_names']+policy['BN_buffer_names']
    require(all(torch.equal(final[n],cold[n]) for n in frozen),'Original typed BN changed at node')
    changed = {n:not torch.equal(v,cold[n]) for n,v in final.items()}
    require(any(changed[n] for n in policy['active_names']),'No actual active tensor changed at node')
    extra = [ADAPTER_PREFIX+s for s in ('hidden.weight','final.weight')] if spec['mode'] != 'native0' else []
    require(all(activity[n] > 0 and changed[n] for n in extra),'Both spatial weights must actually learn by first node')
    node_lr = factor(spec['schedule'],completed_updates-1)
    require(all(g['lr'] == base*node_lr for g,base in zip(optimizer.param_groups,base_lrs)),
            'Node stored LR must be multiplier(U-1)')
    actual_optimizer=optimizer_summary(torch,pref,optimizer,names)
    proof=dict(schema='native-spatial-position-preserving-common24-node-state-proof-v1',
        kind=spec['kind'],mode=spec['mode'],completed_epoch=epoch,
        original_native_cold=pref.state_proof(native_cold),cold=pref.state_proof(cold),
        before=pref.state_proof(cold),after=pref.state_proof(final),changed=changed,
        policy=policy,positive_gradient_microcounts=dict(activity),
        optimizer_step_nonNone_gradient_counts=dict(update_nonNone),gradient_activity_is_accumulated=True,
        optimizer=audit,actual_final_optimizer=actual_optimizer,frozen_BN_all_exact=True,
        all_final_state_finite=True,original_checkpoint=spec['initializers']['mask2former']['checkpoint'],
        before_is_original_fresh_cold=True,initialization_proof=initialization,
        native_microsteps=micro_count,optimizer_updates=completed_updates,raw_GT_visits=GT_visits,
        no_pending_accumulation_gradients=True,node_does_not_restart_training=True)
    meta=dict(schema=CP_SCHEMA,kind=spec['kind'],mode=spec['mode'],phase=spec['phase'],
        spec=spec_ref,source=reference(__file__),training_protocol=protocol,initial_epoch=0,
        completed_epoch=epoch,global_microsteps=micro_count,global_updates=completed_updates,
        consumed_raw_GT=GT_visits,optimizer_param_names=names,optimizer=audit,base_lrs=base_lrs,
        stored_lr_factor=node_lr,resume_checkpoint=None,resume_qualification=None,
        no_pending_accumulation_gradients=True,not_technical_preflight_checkpoint=True,
        fresh_single_owned_24=True,eligible_for_automatic_resume=False)
    checkpoint=dict(state_dict=final,optimizer_state_dict=pref.cpu_tree(torch,optimizer.state_dict()),
        external_rng_state=pref.cpu_tree(torch,pref.rng_snapshot(torch)),private_order_state=dict(
            seed=plan['seed'],order_sha256=plan['order_sha256'],next_epoch=epoch,
            numpy_state=plan['RNG_states_at_epoch_boundaries'][epoch]),meta=meta)
    cp_path=output/('NODE_E%02d.pth'%epoch)
    with cp_path.open('xb') as stream: torch.save(checkpoint,stream)
    readback=torch.load(cp_path,map_location='cpu')
    require(pref.tree_equal(torch,checkpoint,readback),'Full node model/optimizer/RNG CPU readback differs')
    proof['CPU_checkpoint_readback_exact']=True
    proof_path=output/('NODE_E%02d_STATE_PROOF.json'%epoch)
    write(proof_path,proof)
    result=dict(completed_epoch=epoch,checkpoint=reference(cp_path),state_proof=reference(proof_path),
        global_microsteps=micro_count,global_updates=completed_updates,consumed_raw_GT=GT_visits,
        CPU_checkpoint_readback_exact=True,frozen_BN_all_exact=True,no_pending_accumulation_gradients=True,
        stored_lr_factor=node_lr,node_does_not_restart_training=True)
    del final,checkpoint,readback,proof,actual_optimizer
    return result



def train_full24(torch, pref, old, readout, geometry, hooks, spec, spec_ref, output, prepared, dataset, coco, plan, started):
    model, native_cold, initialization = prepared
    require(len(model.state_dict()) == 610 and len(list(model.parameters())) == 451,
            'Original strict COCO-to-boat native610/451 required')
    if spec['mode'] != 'native0':
        hooks.register_spatial_mask_readout(torch,readout,geometry,model,spec['mode'],
            spec['readout_source'],spec['geometry_source'])
    cold = pref.cpu_tree(torch, model.state_dict())
    require(all(torch.equal(cold[name], value) for name, value in native_cold.items()),
            'Registration changed the original native610')
    initialization.update(mode=spec['mode'], cold_adapter={name[len(ADAPTER_PREFIX):]: pref.digest_tensor(value)
        for name, value in cold.items() if name.startswith(ADAPTER_PREFIX)},
        factory_hidden_seed=readout.INITIALIZATION_SEED if spec['mode'] != 'native0' else None,
        registered_native_boundary_exact=True)
    write(output / 'INITIALIZATION_PROOF.json', initialization)
    model.to('cuda:0')
    policy = pref.trainability(torch, model); policy['kind'] = spec['kind']
    extra = int(spec['mode'] != 'native0') * 2
    require(policy['trainable_tensors'] == 345 + extra and policy['parameter_tensors'] == 451 + extra and
        policy['state_entries'] == 610 + extra and policy['frozen_tensors'] == 106 and
        len(policy['BN_buffer_names']) == 159 and len(policy['backbone_conv_names']) == 53,
        'Expected full native345 plus optional adapter2, frozenBN106+159')
    require(all(value.dtype == torch.float32 for value in model.parameters()), 'FP32 native parameters required')
    optimizer, audit = pref.make_optimizer(torch, model, spec['initializers'][spec['kind']], policy)
    names = [[row['name']] for row in audit['parameter_groups']]
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    protocol = training_protocol(spec, plan['order_sha256'])
    require(spec['initial_epoch']==0 and spec['resume_checkpoint'] is None and spec['resume_qualification'] is None,
            'Fresh full24 cannot resume technical or any earlier arm checkpoint')
    resumed=dict(resumed=False,checkpoint=None,qualification=None,global_microsteps=0,
        global_updates=0,consumed_raw_GT=0,cold_ship_initialization=True)
    segment_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    start_rng = pref.rng_snapshot(torch)
    write(output / 'START_STATE_PROOF.json', dict(kind=spec['kind'], mode=spec['mode'], resume=resumed,
        actual_loaded_model_state=pref.state_proof(segment_before), policy=policy,
        actual_optimizer=optimizer_summary(torch, pref, optimizer, names),
        external_RNG=[pref.digest_tensor(value) for value in start_rng],
        base_lrs=base_lrs, training_protocol=protocol))
    del start_rng
    named = dict(model.named_parameters())
    activity = {name: 0 for name in policy['active_names']}
    update_nonNone = {name: 0 for name in policy['active_names']}
    epoch_rows, micro_count, segment_GT, nodes = [], 0, 0, {}
    completed_updates = resumed['global_updates']
    optimizer.zero_grad(set_to_none=True)
    try:
        with (output / 'TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for epoch in range(TOTAL_EPOCHS):
                seen_indices, epoch_GT = [], 0
                # No zero_grad or accumulation reset at this odd741 boundary.
                for within, index in enumerate(plan['orders'][epoch]):
                    pref.check_deadline(started, spec['deadline_seconds'])
                    global_micro = epoch * FIT_IMAGES + within
                    multiplier = factor(spec['schedule'], completed_updates)
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group['lr'] = base_lr * multiplier
                    data, stripped, gray800, raw_gray, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
                    del stripped, gray800, raw_gray
                    identity['dataset_index'] = index
                    targets = pref.check_targets(torch, data, identity)
                    external = pref.rng_snapshot(torch)
                    seed = 1000000 + global_micro
                    with torch.random.fork_rng(devices=[0]):
                        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
                        private_before = pref.rng_snapshot(torch)
                        with observe_native_loss(torch, pref, old, readout, hooks, model.panoptic_head,
                                                 data, spec['mode'], global_micro) as observation:
                            losses = model.loss(data['inputs'], data['data_samples'])
                        total, atoms = pref.loss_terms(torch, losses)
                        require(len(atoms) == 30, 'Exactly native ten-readout30 loss atoms required')
                        for match in observation['matches']:
                            require(len(set(match['positive_query_indices'])) == len(identity['ann_ids']) and
                                sorted(match['assigned_GT_indices']) == list(range(len(identity['ann_ids']))) and
                                all(0 <= q < 100 for q in match['positive_query_indices']),
                                'All original GT must remain in each native assignment')
                        final_match = observation['matches'][-1]
                        supports = observation['adapter_diagnostics']
                        positive_support = ([dict(annotation_id=identity['ann_ids'][gi], assigned_GT_index=gi,
                            query_index=q, eligible_pairs=supports['eligible_counts_by_query'][q])
                            for q, gi in zip(final_match['positive_query_indices'], final_match['assigned_GT_indices'])]
                            if supports is not None else None)
                        (total / 2).backward()
                        private_after = pref.rng_snapshot(torch)
                    external_after = pref.rng_snapshot(torch)
                    require(all(torch.equal(a, b) for a, b in zip(external, external_after)), 'External private-loss RNG was not restored')
                    gradients, grad_names = [], []
                    for name in policy['active_names']:
                        grad = named[name].grad
                        if grad is not None:
                            gradients.append(grad.detach().float().norm()); grad_names.append(name)
                    values = torch.stack(gradients).cpu().tolist() if gradients else []
                    require(all(math.isfinite(value) for value in values), 'Nonfinite accumulated native gradient norm')
                    norm_map = dict(zip(grad_names, values))
                    aligned_norms = [norm_map.get(name) for name in policy['active_names']]
                    for name, value in norm_map.items():
                        if value > 0: activity[name] += 1
                    require(all(named[name].grad is None for name in policy['frozen_names']), 'Frozen BN affine received gradients')
                    bn_modes = {name: module.training for name, module in model.named_modules() if name in policy['BN_modules']}
                    require(not any(bn_modes.values()), 'Typed BN training mode changed')
                    stepped, clipping, global_norm = False, False, None
                    if (global_micro + 1) % 2 == 0:
                        settings = spec['initializers'][spec['kind']]['clip_grad']
                        if settings is None:
                            require(spec['kind'] != 'mask2former', 'Native M2F positive clipping policy required')
                            norm = torch.stack(gradients).norm() if gradients else total.detach().new_zeros(())
                        else:
                            require(set(settings) == {'max_norm', 'norm_type'} and math.isfinite(settings['max_norm']) and
                                settings['max_norm'] > 0 and settings['norm_type'] == 2, 'Positive finite explicit clipping policy required')
                            norm = torch.nn.utils.clip_grad_norm_([named[name] for name in policy['active_names']], **settings)
                            clipping = True
                        require(bool(torch.isfinite(norm)), 'Nonfinite native global gradient norm')
                        global_norm = float(norm.detach().cpu())
                        for name in grad_names: update_nonNone[name] += 1
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                        completed_updates += 1; stepped = True
                    seen_indices.append(index); epoch_GT += len(identity['ann_ids'])
                    segment_GT += len(identity['ann_ids']); micro_count += 1
                    row = dict(kind=spec['kind'], mode=spec['mode'], global_microstep=global_micro, segment_microstep=micro_count - 1,
                        epoch=epoch, within_epoch=within, identity=identity, GT=targets,
                        input=pref.digest_tensor(data['inputs']), loss_fields=atoms, native_total=float(total.detach().cpu()),
                        native_model_loss_only=True, accumulation_divisor=2, optimizer_step=stepped,
                        original_native_loss_atoms=30, native_matches=observation['matches'],
                        native_head_fields=observation['head_fields'], spatial_hook=observation['spatial_hook'],
                        adapter_diagnostics=supports, final_positive_GT_eligibility=positive_support,
                        support_statistics_are_posterior_only=True, support_used_for_loss_or_selection=False,
                        global_updates=completed_updates, lr_factor=multiplier, scheduler_u_before_step=completed_updates - int(stepped),
                        gradient_clipping_applied=clipping, preclip_global_norm=global_norm,
                        gradient_norms_aligned_active_names=aligned_norms, gradients_are_accumulated_preclip=True,
                        frozen_BN_parameter_grads_None=True, BN_module_training=bn_modes,
                        private_seed=seed, private_RNG_before=[pref.digest_tensor(v) for v in private_before],
                        private_RNG_after=[pref.digest_tensor(v) for v in private_after],
                        external_RNG_before=[pref.digest_tensor(v) for v in external],
                        external_RNG_after=[pref.digest_tensor(v) for v in external_after], external_RNG_restored=True)
                    stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
                    if micro_count % 100 == 0 or micro_count == TOTAL_MICROSTEPS:
                        stream.flush()
                        progress = dict(kind=spec['kind'], mode=spec['mode'], segment_microsteps=micro_count,
                            global_microsteps=global_micro + 1, global_updates=completed_updates,
                            segment_raw_GT=segment_GT, elapsed_seconds=time.monotonic() - started)
                        with (output / 'TRAIN_PROGRESS.json').open('w', encoding='utf-8') as handle:
                            json.dump(progress, handle, sort_keys=True, allow_nan=False)
                        print(json.dumps(progress, sort_keys=True), flush=True)
                    del data, losses, total, gradients, grad, private_before, private_after, external, external_after
                    del observation, supports, positive_support
                    if stepped: del norm
                require(seen_indices == plan['orders'][epoch] and sorted(seen_indices) == list(range(FIT_IMAGES)) and
                    epoch_GT == FIT_GT, 'Actual complete741/1661 epoch coverage failed')
                epoch_rows.append(dict(epoch=epoch, dataset_indices_sha256=json_sha(seen_indices),
                    images=FIT_IMAGES, raw_GT=epoch_GT, global_microsteps=(epoch + 1) * FIT_IMAGES,
                    global_updates=completed_updates, pending_accumulation_microsteps=((epoch + 1) * FIT_IMAGES) % 2))
                completed_epoch=epoch+1
                if completed_epoch in CHECKPOINT_NODES:
                    stream.flush()
                    nodes[str(completed_epoch)]=save_node(torch,pref,model,optimizer,spec,spec_ref,output,
                        completed_epoch,native_cold,cold,policy,audit,names,base_lrs,protocol,plan,
                        activity,update_nonNone,micro_count,segment_GT,completed_updates,
                        reference(output/'INITIALIZATION_PROOF.json'))
        require(micro_count==TOTAL_MICROSTEPS and segment_GT==TOTAL_GT_VISITS and
            completed_updates==TOTAL_UPDATES and set(nodes)=={str(e) for e in CHECKPOINT_NODES} and
            all(value.grad is None for value in model.parameters()),'Complete24/no pending gradients/all nodes required')
        write(output/'EPOCH_ROWS.json',epoch_rows)
        final_node=nodes['24']
        return dict(kind=spec['kind'],mode=spec['mode'],nodes=nodes,
            checkpoint=final_node['checkpoint'],state_proof=final_node['state_proof'],
            start_state_proof=reference(output/'START_STATE_PROOF.json'),
            initialization_proof=reference(output/'INITIALIZATION_PROOF.json'),
            trace=reference(output/'TRACE.jsonl'),epoch_rows=reference(output/'EPOCH_ROWS.json'),
            completed_epoch=24,global_microsteps=micro_count,global_updates=completed_updates,
            consumed_raw_GT=segment_GT,CPU_checkpoint_readback_exact=True,frozen_BN_all_exact=True,
            single_owned_full24=True,no_pending_accumulation_gradients=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.to('cpu')



def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    require(spec['schema'] == SPEC_SCHEMA and spec['kind'] in KINDS and spec['mode'] in MODES and
        isinstance(spec['admitted_modes'], list) and tuple(spec['admitted_modes']) == MODES and
        spec['total_epochs'] == TOTAL_EPOCHS and
        type(spec['initial_epoch']) is int and spec['initial_epoch'] == 0 and spec['epochs_this_run'] == 24 and
        spec['resume_checkpoint'] is None and spec['resume_qualification'] is None and
        spec['checkpoint_nodes'] == list(CHECKPOINT_NODES) and
        spec['order_seed'] == 2026100917 and type(spec['backbone_with_cp']) is bool,
        'Four actual modes/fresh single-owned24/fixed six nodes/no resume required')
    owner_source_gate(spec_path,spec)
    require(spec['readout_source']['bytes']==9802 and spec['readout_source']['sha256']==READOUT_SHA and
        spec['geometry_source']['bytes']==17500 and spec['geometry_source']['sha256']==GEOMETRY_SHA and
        spec['hooks_source']['bytes']==21688 and spec['hooks_source']['sha256']==HOOKS_SHA and
        all(spec[key] in spec['sources'] for key in ('readout_source','geometry_source','hooks_source',
            'native_preflight_source','native_preflight_CPU_reader','common_schedule_source')),
        'Exact separately sealed spatial/geometry/hook/native preflight roles required')
    schedule = spec['schedule']
    require(set(schedule) == {'warmup_updates', 'warmup_start_factor', 'milestones', 'gamma'} and
        type(schedule['warmup_updates']) is int and 0 <= schedule['warmup_updates'] < TOTAL_UPDATES and
        math.isfinite(schedule['warmup_start_factor']) and 0 < schedule['warmup_start_factor'] <= 1 and
        (schedule['warmup_updates'] > 0 or schedule['warmup_start_factor'] == 1) and
        schedule['milestones'] == [7904,8563] and all(type(m) is int and 0 < m < TOTAL_UPDATES for m in schedule['milestones']) and
        math.isfinite(schedule['gamma']) and 0 < schedule['gamma'] <= 1, 'Explicit fixed global-update scheduler required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['image_root']).is_absolute(), 'Fresh absolute output/image root required')
    require(spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
        spec['whole_GPU_owner_limit_MiB'] == 8192 and type(spec['deadline_seconds']) is int and 0 < spec['deadline_seconds'] <= 10800 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'ROOT CPU01/CUDA0/4GiB allocator/8GiB owner required')
    for row in spec['sources']: bound(row)
    initializer_qual = json.loads(bound(spec['initializer_qualification']).read_text(encoding='utf-8-sig'))
    micro_qual = json.loads(bound(spec['micro_preflight_qualification']).read_text(encoding='utf-8-sig'))
    paired_qual = json.loads(bound(spec['native_preflight_qualification']).read_text(encoding='utf-8-sig'))
    require(initializer_qual['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
        initializer_qual['all_initializers_strictly_compatible'] is True and initializer_qual['source'] == spec['initializer_source'] and
        micro_qual['schema'] == 'ROOT-native-coco-comparator-micro-preflight-CPU-qualification-v1' and
        micro_qual['status'] == 'ROOT_NATIVE_COCO_COMPARATOR_MICRO_PREFLIGHT_TECHNICALLY_QUALIFIED' and
        micro_qual['source'] == spec['micro_preflight_source'], 'ROOT exact independent initializer/native-loss technical qualification required')
    require(paired_qual['schema'] == 'ROOT-native-spatial-position-preserving-native-bridge-preflight-CPU-qualification-v1' and
        paired_qual['status'] == 'ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_NATIVE_BRIDGE_PREFLIGHT_TECHNICALLY_QUALIFIED' and
        paired_qual['source'] == spec['native_preflight_source'] and
        paired_qual['reader_source'] == spec['native_preflight_CPU_reader'] and
        paired_qual['native_runtime_qualified'] is True and
        paired_qual['eligible_for_formal_resume'] is False and paired_qual['formal_training_qualified'] is False and
        paired_qual['AP_computed'] is False,
        'ROOT exact spatial native technical qualification required, never its technical initializer CP')
    technical_spec = json.loads(bound(paired_qual['spec']).read_text(encoding='utf-8-sig'))
    require(spec['initializers']['mask2former']['optimizer']['type'] == 'AdamW' and
        spec['initializers']['mask2former']['optimizer']['lr'] == 0.0001 and
        spec['initializers']['mask2former']['clip_grad'] == dict(max_norm=0.01,norm_type=2),
        'Common native AdamW1e-4/clip.01 policy is locked')
    require(technical_spec['readout_source'] == spec['readout_source'] and
        technical_spec['geometry_source'] == spec['geometry_source'] and
        technical_spec['hooks_source'] == spec['hooks_source'] and
        technical_spec['native_helpers'] == spec['native_helpers'] and
        technical_spec['initializer_source'] == spec['initializer_source'] and
        technical_spec['initializers']['mask2former'] == spec['initializers']['mask2former'] and
        technical_spec['initializers']['mask2former']['checkpoint'] == spec['initializers']['mask2former']['checkpoint'] and
        technical_spec['initializers']['mask2former']['config'] == spec['initializers']['mask2former']['config'] and
        technical_spec['training_backbone_with_cp'] is spec['backbone_with_cp'],
        'Formal native architecture/source/init/checkpointing must match qualified technical seam')
    output.mkdir(parents=True)
    write(output / 'START.json', dict(schema='native-spatial-position-preserving-common24-start-v1', phase=spec['phase'],
        source=reference(__file__), spec=reference(spec_path), kind=spec['kind'], mode=spec['mode'], initial_epoch=spec['initial_epoch'],
        epochs_this_run=24, total_epochs=24, checkpoint_nodes=list(CHECKPOINT_NODES), no_resume=True, no_AP=True, no_DEV=True, no_retry=True))
    import psutil
    import numpy as np
    import torch
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU01/coldCUDA mismatch')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'qualified_comparator_micro_source')
    initializer = load_source(spec['initializer_source'], 'qualified_comparator_initializer_source')
    old = load_source(spec['native_helpers'], 'qualified_comparator_native_helpers'); old.torch = torch
    readout=load_source(spec['readout_source'],'qualified_spatial_readout')
    geometry=load_source(spec['geometry_source'],'sealed_original_geometry')
    hooks=load_source(spec['hooks_source'],'qualified_spatial_whole_head_hooks')
    bound(spec['common_schedule_source'])
    require(readout.INITIALIZATION_SEED == geometry.INITIALIZATION_SEED == 2026100911 and
        readout.TRAINABLE_TENSORS == 2 and readout.TRAINABLE_SCALARS == 393216 and
        tuple(readout.MODES) == tuple(hooks.MODES) == MODES[1:],
        'Exact sealed factory modes/capacity/default hidden seed required')
    prepared = pref.prepare_cpu(torch, initializer, MODELS, revert_sync_batchnorm, Config,
        tuple(CocoDataset.METAINFO['classes']), spec, spec['kind'], initializer_qual['initializers'][spec['kind']])
    dataset = CocoDataset(ann_file=str(bound(spec['FIT'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT'])))
    require(len(dataset) == len(coco.imgs) == FIT_IMAGES and len(coco.anns) == FIT_GT, 'Complete741/1661 FIT required')
    plan = order_plan(np, spec['order_seed']); write(output / 'ORDER.json', plan)
    torch.cuda.init(); capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    torch.cuda.reset_peak_memory_stats(0)
    error_record, result, cleanup = None, None, None
    try:
        result = train_full24(torch, pref, old, readout, geometry, hooks, spec, reference(spec_path), output, prepared, dataset, coco, plan, started)
        del prepared
        cleanup = pref.clear_gpu(torch)
        for row in spec['sources']: bound(row)
        bound(spec['initializers'][spec['kind']]['checkpoint'])
        pref.check_deadline(started, spec['deadline_seconds'])
    except Exception as error:
        error_record = dict(type=type(error).__name__, message=str(error), no_retry=True, no_fallback=True,
            OOM_technical_stop=isinstance(error, torch.cuda.OutOfMemoryError))
        traceback.clear_frames(error.__traceback__)
        if 'prepared' in locals(): prepared[0].to('cpu'); del prepared
        gc.collect()
        try: cleanup = pref.clear_gpu(torch)
        except Exception as cleanup_error: error_record['cleanup_error'] = str(cleanup_error)
    report = dict(schema=REPORT_SCHEMA, status=('NATIVE_SPATIAL_COMMON24_OWNED_COMPLETE_NO_AP' if error_record is None else
        'NATIVE_SPATIAL_COMMON24_STOP_NO_METHOD_RESULT'), phase=spec['phase'], source=reference(__file__),
        spec=reference(spec_path), kind=spec['kind'], mode=spec['mode'], result=result, error=error_record, cleanup=cleanup,
        initializer_qualification=spec['initializer_qualification'], micro_preflight_qualification=spec['micro_preflight_qualification'],
        native_preflight_qualification=spec['native_preflight_qualification'],
        readout_source=spec['readout_source'],geometry_source=spec['geometry_source'],hooks_source=spec['hooks_source'],
        order=reference(output / 'ORDER.json'), order_sha256=plan['order_sha256'],
        initial_epoch=0,epochs_this_run=24,total_epochs=24,checkpoint_nodes=list(CHECKPOINT_NODES),
        checkpoint=result['checkpoint'] if result else None,nodes=result['nodes'] if result else None,
        completed_epoch=result['completed_epoch'] if result else None,
        global_microsteps=result['global_microsteps'] if result else None,
        global_updates=result['global_updates'] if result else None,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        full_R50_conv_trainable=True, typed_BN_frozen=True, native_loss_only=True,
        original_native_loss_atoms=30, adapter_geometry_has_no_GT_input=True,
        realized_cross_mode_points_or_eligibility_equal_claimed=False,
        unchanged_class_scores_after_training_claimed=False,
        no_inference=True, no_DEV=True, no_AP=True, GT_free_inference_runtime_verified=False,single_owned_fresh_full24=True,no_resume=True,equal_whole_compute_claimed=False,
        scientific_method_decision_computed=False, published_recipe_reproduction_claimed=False)
    write(output / 'REPORT.json', report)
    if error_record: raise RuntimeError('TECHNICAL_STOP: ' + error_record['type'] + ': ' + error_record['message'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args()
    require(args.owner_execute,'SOURCE_ONLY: explicit ROOT owned full24 execution/admission required')
    run(args.spec)



if __name__ == '__main__':
    main()
