"""UNBOUND actual single-FIT native100 field capture; stdlib-only at import.

Only a future Root-owned admitted leaf may call run. This captures the original
teacher's M directly; it does not reconstruct M, train, or run PR point actions.
"""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

PACKET_SCHEMA = 'U32-actual-native100-FIT-fields-v1'
REPORT_SCHEMA = 'U32-native100-FIT-capture-report-pending-v1'
LEGACY_PACKET_SCHEMA = 'N22-FIT32-natural100-fullC-actual-GPU-action-packet-v1'
SOURCE_ROLES = ('entry_source', 'body_source', 'runtime_source', 'legacy_capture_source')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _json(c, ref):
    return json.loads(Path(c.closed(ref)).read_text(encoding='utf-8-sig'))


def _load(c, ref):
    path = Path(c.closed(ref))
    name = 'U32_SOURCE_CLOSED_LEGACY_N22_CAPTURE'
    loader = importlib.util.spec_from_file_location(name, path)
    require(loader is not None and loader.loader is not None, 'Legacy Source loader missing')
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    loader.loader.exec_module(module)
    return module


def _tensor_record(torch, pref, value):
    require(value.device.type == 'cpu' and not value.requires_grad,
            'Only detached CPU tensors enter the capture report')
    return dict(shape=list(value.shape), dtype=str(value.dtype), device='cpu',
                finite=bool(torch.isfinite(value).all()) if value.is_floating_point() else True,
                digest=pref.digest_tensor(value))


def _capture(torch, np, spec, base, c, legacy, boot, item, anchor):
    """N22's original frozen-forward prefix, followed only by field/score checks."""
    _call, _tensor = legacy._call, legacy._tensor
    _same_bits, _frozen = legacy._same_bits, legacy._frozen
    modules, model, pr = boot['modules'], boot['model'], boot['head']
    pref, old, diag, joint, preflight, cache7, targets = (
        modules[k] for k in ('pref', 'old', 'diag', 'joint', 'preflight', 'cache7', 'targets'))
    index, row = item['dataset_index'], item['old_row']
    packet = torch.load(str(_call(c, 'closed', item['packet_ref'])), map_location='cpu')
    carrier = torch.load(str(_call(c, 'closed', item['native_L_ref'])), map_location='cpu')
    require(packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1'
            and carrier['schema'] == 'N16-actual-GPU-natural100-L-carrier-v1'
            and packet['dataset_index'] == carrier['dataset_index'] == index
            and packet['identity'] == carrier['identity'] == row['identity']
            and packet['natural_native_L_carrier'] == item['native_L_ref']
            and row['packet'] == item['packet_ref'] and row['natural_native_L_carrier'] == item['native_L_ref'],
            'Qualified N16 packet/native-L/row binding drift')
    for key in ('source', 'spec', 'GPU_admission', 'phase'):
        require(packet[key] == carrier[key], 'N16 carrier parent mismatch: ' + key)
    require(packet['teacher_checkpoint'] == base['model_refs']['native0_checkpoint']
            and packet['fixed_PR_checkpoint'] == base['model_refs']['fixed_LOG_PR_checkpoint']
            and packet['natural_query_ids'] == list(range(100)), 'Frozen teacher/PR/all100 drift')
    identity = packet['identity']; H, W = identity['original_shape']
    old_L = carrier['native_L']
    _tensor(torch, old_L, torch.float32, (100, H, W), 'Qualified native-L carrier')
    seed = packet['matching_loss_seed']
    require(type(seed) is int and seed == row['matching_loss_seed'], 'Original matcher seed drift')
    require(index == 691 and identity['image_id'] == 871 and len(identity['ann_ids']) == 9
            and H != W, 'Preselected nonsquare FIT691/image871/G9 identity required')
    require(anchor['schema'] == LEGACY_PACKET_SCHEMA and anchor['dataset_index'] == index
            and anchor['identity'] == identity and anchor['natural_query_ids'] == list(range(100))
            and anchor['category_id'] == 1 and anchor['matching_loss_seed'] == seed,
            'Original N22 anchor identity/order/seed drift')
    for key, value in (('packet', item['packet_ref']), ('native_L', item['native_L_ref']),
                       ('teacher', base['model_refs']['native0_checkpoint']),
                       ('fixed_PR', base['model_refs']['fixed_LOG_PR_checkpoint'])):
        require(anchor['input_refs'][key] == value, 'N22 anchor input binding drift: ' + key)
    _tensor(torch, anchor['native_L'], torch.float32, (100, H, W), 'N22 anchor native-L')
    _tensor(torch, anchor['native_e0'], torch.float32, (1, 100, 256), 'N22 anchor all100 e')
    _tensor(torch, anchor['native_class_logits'], torch.float32, (1, 100, 2), 'N22 anchor class')
    for key in ('native_scores_by_query', 'native_top_scores_by_export_order',
                'native_quality_by_export_order'):
        _tensor(torch, anchor[key], torch.float32, (100,), 'N22 anchor ' + key)
    _frozen(model, 'Teacher'); _frozen(pr, 'PR')
    outside = pref.rng_snapshot(torch)
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        _call(c, 'check')
        data, stripped, gray800, gray, observed = old.load_image(
            model, boot['dataset'], boot['coco'], boot['dataset'].get_data_info(index))
        del stripped, gray800, gray
        observed['dataset_index'] = index
        require(observed == identity, 'Original image/parser/resize identity drift')
        processed = pref.check_targets(torch, data, observed)
        require(pref.tree_equal(torch, processed, packet['processed_target_proof']), 'Processed targets drift')
        raw_GT, mapping, validator = preflight.raw_GT_validator(
            torch, np, boot['cv2'], boot['coco'], data, observed)
        require(mapping == packet['raw_target_mapping'] and identity['ann_ids'] == packet['raw_GT_ids']
                and pref.digest_tensor(raw_GT) == packet['raw_GT_digest'], 'Original real-GT/mapping drift')
        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
        native_head = model.panoptic_head
        with joint._observe_processed_GT(torch, native_head, identity['ann_ids'], raw_GT,
                                         mapping, identity, validator) as gt_observation:
            with targets.observe_all100_targets(torch, native_head, old.digest_tensor) as (matches, observations):
                with cache7.observed_fields(torch, native_head) as fields:
                    losses = model.loss(data['inputs'], data['data_samples'])
        require(len(losses) == 30 and set(losses) == set(old.ATOMIC_KEYS) == set(packet['native_loss_atoms']),
                'Thirty original loss atoms required')
        for value in losses.values():
            values = value if isinstance(value, (tuple, list)) else [value]
            require(values and all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in values),
                    'Frozen loss atoms must be finite')
        require(matches == packet['native_stage_matches'] and len(matches) == 10
                and pref.tree_equal(torch, observations, packet['all_ten_target_observations'])
                and pref.tree_equal(torch, gt_observation, packet['actual_raw_processed_GT_correspondence']),
                'Original all10 matcher/GT correspondence drift')
        F, e = fields['F'][0], fields['e'][-1]
        classes, mask_stages = fields['output'][0]
        native_class, M = classes[-1], mask_stages[-1]
        _same_bits(torch, e, packet['native_e0'], 'All100 e')
        _same_bits(torch, native_class, packet['native_class_logits'], 'All100 class')
        assignment = targets.final_assignment(torch, observations[-1], mapping,
                                              identity['ann_ids'], native_head.num_classes)
        require(pref.tree_equal(torch, assignment, packet['assignment']), 'Original full100 assignment drift')
        native_L = diag.render_original(torch, M, identity)
        _tensor(torch, native_L, torch.float32, (100, H, W), 'Actual original100 L', 'cuda')
        actual_L = native_L.detach().cpu().clone().contiguous()
        error = float((actual_L - old_L).abs().max())
        require(error <= 1.e-4, 'Native-L numeric drift exceeds absolute1e-4/rtol0')
        # No domains_from_native, PR points, full-C feature rendering or actions.
        require(torch.equal(actual_L > 0, old_L > 0), 'N16 original all100 sign drift')
        _tensor(torch, F, torch.float32, (1, 256, 200, 200), 'Actual native dense F', 'cuda')
        _tensor(torch, M, torch.float32, (1, 100, 200, 200), 'Actual native all100 M', 'cuda')
        _tensor(torch, e, torch.float32, (1, 100, 256), 'Actual native all100 e', 'cuda')
        _tensor(torch, native_class, torch.float32, (1, 100, 2), 'Actual native class', 'cuda')
        require(isinstance(raw_GT, torch.Tensor) and tuple(raw_GT.shape) == (9, H, W)
                and not raw_GT.requires_grad and bool(((raw_GT == 0) | (raw_GT == 1)).all()),
                'Complete original binary GT9 required')
        _same_bits(torch, actual_L, anchor['native_L'], 'Original N22 all100 native-L anchor')
        _same_bits(torch, e, anchor['native_e0'], 'Original N22 all100 e anchor')
        _same_bits(torch, native_class, anchor['native_class_logits'], 'Original N22 class anchor')
        _same_bits(torch, raw_GT, anchor['raw_GT'], 'Original N22 raw GT anchor')
        require(list(identity['ann_ids']) == anchor['raw_GT_ids']
                and tuple(mapping) == tuple(anchor['raw_target_mapping'])
                and pref.tree_equal(torch, assignment, anchor['assignment'])
                and matches == anchor['native_stage_matches']
                and pref.tree_equal(torch, observations, anchor['all_ten_target_observations'])
                and pref.tree_equal(torch, processed, anchor['processed_target_proof'])
                and pref.tree_equal(torch, gt_observation, anchor['actual_raw_processed_GT_correspondence']),
                'Original N22 matcher/GT/processed-target anchor drift')
        top, order = modules['kernels'].order(torch, native_class, types.SimpleNamespace(num_classes=1))
        query_order = order.detach().cpu().tolist()
        require(len(query_order) == 100 and sorted(query_order) == list(range(100))
                and query_order == anchor['native_export_query_ids'], 'Original N22 export query order drift')
        binary, quality, scores = modules['kernels'].quality(torch, native_L[order], top)
        by_query = scores.new_empty(100); by_query[order] = scores
        _same_bits(torch, top, anchor['native_top_scores_by_export_order'], 'Original N22 top-score anchor')
        _same_bits(torch, quality, anchor['native_quality_by_export_order'], 'Original N22 mask-quality anchor')
        _same_bits(torch, by_query, anchor['native_scores_by_query'], 'Original N22 S0-by-query anchor')
        require(torch.equal(binary, native_L[order] > 0), 'Original export uses native sign masks')
        payload = dict(schema=PACKET_SCHEMA, source=spec['entry_source'], body_source=spec['body_source'],
            spec=c.execution_spec_ref, GPU_admission=c.execution_admission_ref, phase=spec['phase'],
            prelock=spec['prelock'], dataset_index=index, identity=identity, natural_query_ids=list(range(100)),
            teacher_checkpoint=base['model_refs']['native0_checkpoint'],
            fixed_PR_checkpoint=base['model_refs']['fixed_LOG_PR_checkpoint'],
            native_F256=F.detach().cpu().clone().contiguous(),
            native_M200=M.detach().cpu().clone().contiguous(), native_L=actual_L,
            native_e0=e.detach().cpu().clone().contiguous(),
            native_class_logits=native_class.detach().cpu().clone().contiguous(),
            raw_GT=raw_GT.detach().cpu().clone().contiguous(), raw_GT_ids=list(identity['ann_ids']),
            raw_GT_digest=pref.digest_tensor(raw_GT), raw_target_mapping=tuple(mapping),
            assignment=legacy._cpu_tree(torch, assignment), native_stage_matches=legacy._cpu_tree(torch, matches),
            all_ten_target_observations=legacy._cpu_tree(torch, observations),
            processed_target_proof=legacy._cpu_tree(torch, processed),
            actual_raw_processed_GT_correspondence=legacy._cpu_tree(torch, gt_observation),
            native_loss_atoms=legacy._cpu_tree(torch, losses), matching_loss_seed=seed,
            native_export_query_ids=query_order, native_scores_by_query=by_query.detach().cpu().clone().contiguous(),
            native_top_scores_by_export_order=top.detach().cpu().clone().contiguous(),
            native_quality_by_export_order=quality.detach().cpu().clone().contiguous(), category_id=1,
            input_refs=dict(legacy_N22_spec=spec['legacy_N22_spec'], anchor_N22_packet=spec['anchor_N22_packet'],
                N16_packet=item['packet_ref'], N16_native_L=item['native_L_ref'], image=identity['image'],
                FIT=base['model_refs']['FIT'], teacher=base['model_refs']['native0_checkpoint'],
                fixed_PR=base['model_refs']['fixed_LOG_PR_checkpoint']),
            actual_M_from_teacher_forward=True, M_reconstructed=False, optimizer_created=False,
            training_updates=0, PR_points_executed=False, qualified=False, new_training=False,
            AP=False, DEV=False, TEST=False)
        checks = dict(original_image_identity_exact=True, original_matching_seed_exact=True,
            original_processed_targets_exact=True, original_raw_GT_digest_exact=True,
            original_raw_mapping_exact=True, original_native30_loss_atom_keys_exact=True,
            native30_loss_atoms_finite=True, native30_loss_atom_count=len(losses),
            original_all10_matches_exact=True, original_all10_target_observations_exact=True,
            original_all10_raw_processed_GT_correspondence_exact=True,
            original_N16_e_bitexact=True, original_N16_class_bitexact=True,
            original_N16_assignment_exact=True, N16_native_L_max_abs_error=error,
            N16_native_L_atol=1.e-4, N16_native_L_rtol=0., N16_native_sign_exact=True,
            N22_anchor_native_L_bitexact=True, N22_anchor_native_e0_bitexact=True,
            N22_anchor_class_bitexact=True, N22_anchor_raw_GT_bitexact=True,
            N22_anchor_raw_GT_ids_exact=True, N22_anchor_raw_mapping_exact=True,
            N22_anchor_assignment_exact=True, N22_anchor_all10_matches_exact=True,
            N22_anchor_all10_target_observations_exact=True, N22_anchor_processed_targets_exact=True,
            N22_anchor_all10_raw_processed_GT_correspondence_exact=True,
            N22_anchor_export_query_order_exact=True, N22_anchor_top_scores_bitexact=True,
            N22_anchor_mask_quality_bitexact=True, N22_anchor_S0_by_query_bitexact=True,
            actual_native_F_shape_checked=True, actual_native_M100_shape_checked=True,
            all_native_fields_FP32_finite_detached=True, complete_raw_GT9_binary=True,
            actual_M_from_teacher_forward=True, M_reconstructed=False,
            domains_computed=False, PR_points_executed=False, fullC_actions_executed=False)
        # No GPU tensor, observation closure or loss escapes this function.
        del binary, quality, scores, by_query, top, order, native_L, F, e, M, native_class
        del fields, classes, mask_stages, raw_GT, data, losses, packet, carrier, old_L, values, value
    require(pref.tree_equal(torch, outside, pref.rng_snapshot(torch)), 'External CPU/CUDA RNG not restored')
    _frozen(model, 'Teacher'); _frozen(pr, 'PR')
    checks['external_CPU_CUDA_RNG_restored'] = True
    return payload, checks


def run(torch, np, spec, c, pre):
    """One new admitted task; historical spec is only a closed model dependency."""
    started = time.monotonic()
    require(pre['schema'] == 'U32-native100-M-one-FIT-prelock-v1'
            and pre['dataset_index'] == spec['dataset_index'] == 691
            and pre['images'] == 1 and pre['queries'] == 100
            and pre['native_L_anchor_bitexact'] is True
            and all(pre[k] is True for k in ('no_new_training', 'no_AP', 'no_DEV', 'no_TEST')),
            'Exact single-FIT prelock required')
    require(not torch.cuda.is_initialized(), 'Original CPU construction requires cold CUDA')
    source_refs = {key: spec[key] for key in SOURCE_ROLES}
    for ref in source_refs.values():
        c.closed(ref)
    legacy = _load(c, spec['legacy_capture_source'])
    base = _json(c, spec['legacy_N22_spec'])
    selected = legacy.validate_selected_rows(base)
    items = [item for item in selected if item['dataset_index'] == spec['dataset_index']]
    require(len(items) == 1, 'Selected FIT691 must occur exactly once in original32 metadata')
    item = items[0]
    anchor = torch.load(str(c.closed(spec['anchor_N22_packet'])), map_location='cpu')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and output.parent.is_dir() and not output.exists()
            and not output.is_symlink(), 'Fresh absolute owned output required')
    output.mkdir()
    boot = legacy.bootstrap(torch, np, base, c)
    pref = boot['modules']['pref']
    pr_forward_calls = [0]
    def observed_pr_forward(_module, _args):
        pr_forward_calls[0] += 1
    handle = boot['head'].register_forward_pre_hook(observed_pr_forward)
    try:
        c.check()
        payload, checks = _capture(torch, np, spec, base, c, legacy, boot, item, anchor)
        require(pr_forward_calls[0] == 0, 'No PR head forward belongs to native-M capture')
        require(pref.state_proof(boot['model'].state_dict()) == boot['teacher_state_before']
                and pref.state_proof(boot['head'].state_dict()) == boot['PR_state_before'],
                'Frozen teacher/PR state changed')
        teacher_state_proof, pr_state_proof = boot['teacher_state_before'], boot['PR_state_before']
        loading = boot['loading']
        checks.update(teacher_state_unchanged=True, PR_state_unchanged=True,
                      teacher_frozen_eval_no_grad=True, PR_frozen_eval_no_grad=True, PR_forward_calls=0)
    finally:
        handle.remove()
        # Release all GPU model aliases before the owning entry's allocator gate.
        boot['model'].cpu(); boot['head'].cpu()
        boot.clear()
        gc.collect(); torch.cuda.synchronize(0); torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache(); torch.cuda.synchronize(0)
    del anchor, base, legacy, boot
    c.check()
    tensor_records = {key: _tensor_record(torch, pref, payload[key]) for key in (
        'native_F256', 'native_M200', 'native_e0', 'native_class_logits', 'native_L', 'raw_GT',
        'native_scores_by_query', 'native_top_scores_by_export_order', 'native_quality_by_export_order')}
    atom_records = {}
    for name, value in payload['native_loss_atoms'].items():
        values = value if isinstance(value, (tuple, list)) else [value]
        atom_records[name] = [_tensor_record(torch, pref, part) for part in values]
    packet_path = output / 'NATIVE100_FIELDS_0691.pth'
    with packet_path.open('xb') as stream:
        torch.save(payload, stream)
    packet_ref = c.ref(packet_path)
    report = dict(schema=REPORT_SCHEMA, status='ACTUAL_SINGLE_FIT_NATIVE100_FIELDS_PENDING_ORIGINAL_ACTUAL0_AND_ROOT',
        source=spec['entry_source'], body_source=spec['body_source'], source_refs=source_refs,
        spec=c.execution_spec_ref, GPU_admission=c.execution_admission_ref, phase=spec['phase'],
        prelock=spec['prelock'], parent_ROOT_qualified=spec['parent_ROOT_qualified'],
        input_refs=payload['input_refs'], packet_ref=packet_ref,
        dataset_index=691, image_id=871, identity=payload['identity'], images=1, queries=100, raw_GT=9,
        natural_query_ids=payload['natural_query_ids'], native_export_query_ids=payload['native_export_query_ids'],
        matching_loss_seed=payload['matching_loss_seed'], tensor_records=tensor_records,
        raw_GT_digest=payload['raw_GT_digest'], native_loss_atom_records=atom_records, checks=checks,
        loading=loading, teacher_state_proof=teacher_state_proof, PR_state_proof=pr_state_proof,
        legacy_bootstrap_validated_original_selected_metadata_rows=32,
        legacy_bootstrap_checked_original32_image_source_references=True,
        actual_decoded_images=1, actual_teacher_loss_forwards=1,
        PR_head_constructed_and_frozen=True, PR_head_forward_calls=0,
        optimizer_created=False, training_updates=0, domains_computed=False, fullC_actions_executed=False,
        qualified=False, original_actual0_self_certified=False, new_training=False,
        FIT=True, AP=False, DEV=False, TEST=False, utility_evaluated=False, innovation_established=False,
        scope=('Original N22 bootstrap validates all32 metadata/image refs; only FIT691 is decoded and '
               'forwarded. Captured M/F are original dense teacher outputs. Exact anchor agreement '
               'does not establish U31 size400 backbone behavior, training utility or AP. No domains '
               'or PR points are evaluated; the unused historical PR head is constructed and frozen.'),
        elapsed_seconds=time.monotonic()-started)
    report_ref = c.write_json(output / 'CAPTURE_REPORT.json', report)
    del payload, values, value
    gc.collect(); c.check()
    return dict(packet_ref=packet_ref, report_ref=report_ref, dataset_index=691, image_id=871,
                images=1, queries=100, qualified=False, new_training=False, AP=False, DEV=False, TEST=False)
