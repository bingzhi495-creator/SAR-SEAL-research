"""UNBOUND_DRAFT cache numerical body. No source/RAW admission, IO, CLI or run.

ROOT supplies already qualified modules/model/data/old_packet and frozen E24 PR.
modules: mapping with pref, old, diag, joint, preflight, cache7 (ordinary N07
module), common; or ten-slot (native,legacy,pref,initializer,common,old,
diag,joint,preflight,cache7). No module objects are put in JSON spec.
helper API is NumPy utility_labels(P,G,A), error_labels(P,G), INT64 classes.
Returns (row, packet); CPU-only tensors, no fabricated negative native targets.
All background targets are an explicit NEW empty-set supervision extension.
Full natural GPU L is saved CPU FP32; ROOT must lock its 400*H*W bytes/image.
Only synthetic sampling-helper tests have run, never this scientific body.
"""
from contextlib import contextmanager
import types


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def stratum_plan(matched, background):
    """Target: per-image uniform active query, then uniform C point.

    Proposal: equally likely nonempty assignment strata, uniform active query
    within stratum, then uniform C point. No class/outcome rebalancing.
    """
    require(len(set(matched)) == len(matched) and len(set(background)) == len(background)
            and not set(matched).intersection(background), 'Distinct active strata required')
    total = len(matched) + len(background)
    if total == 0:
        return {}
    groups = [('matched', list(matched)), ('background', list(background))]
    present = sum(bool(q) for _, q in groups)
    p_s = 1.0 / present
    return {name: dict(query_ids=q, n_stratum=len(q), n_active=total, p_s=p_s,
                       draws=2048 if present == 2 else 4096,
                       weight=(len(q) / total) / p_s)
            for name, q in groups if q}


def sample_stratified(torch, domain_indices, matched, background, seed):
    plan = stratum_plan(matched, background)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    before = rng.get_state().clone()
    queries, flats, weights, strata = [], [], [], []
    for group_id, name in enumerate(('matched', 'background')):
        if name not in plan:
            continue
        group = plan[name]
        query_draws = torch.randint(group['n_stratum'], (group['draws'],), generator=rng).tolist()
        for ordinal in query_draws:
            q = group['query_ids'][ordinal]
            candidates = domain_indices[q]
            require(candidates.dtype == torch.int64 and candidates.device.type == 'cpu' and len(candidates) > 0,
                    'Sample only nonempty active physical C queries')
            offset = int(torch.randint(len(candidates), (1,), generator=rng).item())
            queries.append(q)
            flats.append(int(candidates[offset].item()))
            weights.append(group['weight'])
            strata.append(group_id)
    return dict(query_ids=torch.tensor(queries, dtype=torch.int64),
                flat_indices=torch.tensor(flats, dtype=torch.int64),
                importance_weights=torch.tensor(weights, dtype=torch.float32),
                stratum_ids=torch.tensor(strata, dtype=torch.uint8),
                sampler_state_before=before, sampler_state_after=rng.get_state().clone(),
                plan=plan, seed=seed, replacement=True, duplicates_preserved=True,
                draw_order='matched then background; vector query draw then sequential point draws')


@contextmanager
def observe_all100_targets(torch, head, digest_tensor):
    """Observe the existing seven-tuple once; never invoke assignment twice."""
    original = head._get_targets_single
    had_instance = '_get_targets_single' in head.__dict__
    previous = head.__dict__.get('_get_targets_single')
    legacy, expanded = [], []
    def observed(current, *args, **kwargs):
        result = original(*args, **kwargs)
        require(len(result) == 7, 'Original native seven-tuple required')
        labels, label_weights, mask_targets, mask_weights, pos, neg, sampling = result
        require(tuple(labels.shape) == tuple(label_weights.shape) == tuple(mask_weights.shape) == (100,),
                'Natural100 target vectors required')
        require(torch.equal(pos, sampling.pos_inds) and torch.equal(neg, sampling.neg_inds),
                'Returned positive/negative indices must equal actual sampler indices')
        assigned = sampling.pos_assigned_gt_inds.detach().cpu().tolist()
        positive, negative = pos.detach().cpu().tolist(), neg.detach().cpu().tolist()
        require(len(positive) == len(assigned) and len(set(positive + negative)) == len(positive) + len(negative),
                'Disjoint original sampler assignment required')
        legacy.append(dict(positive_query_indices=positive, assigned_GT_indices=assigned,
                           positives=len(positive), target_labels=digest_tensor(labels)))
        expanded.append(dict(labels=labels.detach().cpu().clone().contiguous(),
            label_weights=label_weights.detach().cpu().clone().contiguous(),
            mask_weights=mask_weights.detach().cpu().clone().contiguous(),
            positive_query_indices=positive, negative_query_indices=negative,
            assigned_GT_indices=assigned, mask_targets_digest=digest_tensor(mask_targets),
            sampler_positive_query_indices=sampling.pos_inds.detach().cpu().tolist(),
            sampler_negative_query_indices=sampling.neg_inds.detach().cpu().tolist()))
        return result
    head._get_targets_single = types.MethodType(observed, head)
    try:
        yield legacy, expanded
        require(len(legacy) == len(expanded) == 10, 'All original ten assignment stages required')
    finally:
        if had_instance:
            head._get_targets_single = previous
        else:
            delattr(head, '_get_targets_single')


def final_assignment(torch, observation, mapping, ann_ids, num_classes):
    labels, lw, mw = observation['labels'], observation['label_weights'], observation['mask_weights']
    require(labels.dtype == lw.dtype == torch.int64 and mw.dtype == torch.float32
            and bool(torch.isfinite(lw).all()) and bool(torch.isfinite(mw).all()), 'Exact finite native target dtype required')
    positive, negative = observation['positive_query_indices'], observation['negative_query_indices']
    require(len(ann_ids) == len(mapping) and sorted(mapping) == list(range(len(ann_ids)))
            and len(set(ann_ids)) == len(ann_ids), 'Complete distinct raw GT IDs/mapping required')
    require(len(set(positive)) == len(positive) and len(set(negative)) == len(negative)
            and not set(positive).intersection(negative) and all(0 <= q < 100 for q in positive + negative),
            'Exact sampler positive/negative query sets required')
    ignored = sorted(set(range(100)) - set(positive) - set(negative))
    require(set(positive) | set(negative) | set(ignored) == set(range(100))
            and len(positive) + len(negative) + len(ignored) == 100, 'Native pos/neg/ignore partition must cover100')
    raw_gt = torch.full((100,), -2, dtype=torch.int64)  # -2 ignored; -1 explicitly NEW background empty set
    kinds = ['ignored'] * 100
    for q, assigned in zip(positive, observation['assigned_GT_indices']):
        require(0 <= q < 100 and 0 <= assigned < len(mapping) and labels[q] != num_classes
                and lw[q] > 0 and mw[q] > 0, 'Positive native mask target must be valid foreground')
        raw_gt[q] = mapping[assigned]
        kinds[q] = 'matched'
    for q in negative:
        require(0 <= q < 100 and labels[q] == num_classes and mw[q] == 0,
                'Background extension requires actual sampler neg AND background label AND mask weight0')
        if lw[q] > 0:
            raw_gt[q] = -1
            kinds[q] = 'background'
    require(sorted(raw_gt[torch.tensor(positive, dtype=torch.int64)].tolist()) == list(range(len(ann_ids)))
            and all(int(raw_gt[q]) == -2 for q in ignored), 'Exact complete positive raw-GT mapping; native ignores unchanged')
    valid = raw_gt != -2
    return dict(natural_query_ids=list(range(100)), raw_GT_index_by_query=raw_gt,
                raw_GT_annotation_id_by_query=[ann_ids[int(g)] if g >= 0 else None for g in raw_gt.tolist()],
                supervision_kind_by_query=kinds, valid_targets=valid,
                native_positive_query_indices=list(positive), native_negative_query_indices=list(negative),
                native_ignored_query_indices=ignored,
                final_native_labels=labels, final_native_label_weights=lw, final_native_mask_weights=mw,
                background_empty_set_is_new_supervision_extension=True,
                original_native_negative_mask_supervised=False, ignore_never_zero_filled=True)


def sampled_features_unique(torch, renderer, F, identity, flat_indices):
    """Original two-stage renderer, 16 channels at a time; one F row per unique flat."""
    require(flat_indices.dtype == torch.int64 and flat_indices.device.type == 'cpu' and flat_indices.ndim == 1,
            'Unique physical flat indices must be CPU INT64')
    result = torch.empty((len(flat_indices), 256), dtype=torch.float32, device='cpu')
    if len(flat_indices) == 0:
        return result
    device_indices = flat_indices.to(F.device)
    for first in range(0, 256, 16):
        original = renderer(F[:, first:first + 16], identity)
        require(original.dtype == torch.float32 and tuple(original.shape) == (16,) + tuple(identity['original_shape'])
                and bool(torch.isfinite(original).all()), 'Original GPU F renderer differs')
        result[:, first:first + 16].copy_(original.flatten(1)[:, device_indices].T.detach().cpu())
        del original
    return result.contiguous()


def tensor_payload_bytes(torch, value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu' and not value.requires_grad and bool(torch.isfinite(value).all()),
                'Body return payload must be finite CPU-only tensors')
        return value.numel() * value.element_size()
    if type(value) is dict:
        return sum(tensor_payload_bytes(torch, item) for item in value.values())
    if type(value) in (tuple, list):
        return sum(tensor_payload_bytes(torch, item) for item in value)
    return 0


def cache_image(torch, np, cv2, ndimage, modules, model, dataset, coco, spec, context,
                old_packet, old_row, index, pr_head, diag15, helper):
    if type(modules) is dict:
        pref, old, diag, joint, preflight, cache7 = (modules[k] for k in ('pref', 'old', 'diag', 'joint', 'preflight', 'cache7'))
    else:
        require(len(modules) == 10, 'ROOT must supply ten-module tuple or named mapping')
        _, _, pref, _, _, old, diag, joint, preflight, cache7 = modules
    require(index == old_packet['dataset_index'] == old_row['dataset_index'] and old_packet['identity'] == old_row['identity'],
            'Already qualified old packet/row/index required')
    require(old_packet['teacher_checkpoint'] == context['checkpointmap']['native0'], 'Same frozen teacher checkpoint required')
    info = dataset.get_data_info(index)
    data, stripped, gray800, gray, identity = old.load_image(model, dataset, coco, info)
    del stripped, gray800, gray
    identity['dataset_index'] = index
    require(identity == old_packet['identity'], 'Original FIT parser/image/GT identity differs')
    processed_proof = pref.check_targets(torch, data, identity)
    raw_GT, mapping, validator = preflight.raw_GT_validator(torch, np, cv2, coco, data, identity)
    raw_cpu = raw_GT.detach().cpu().contiguous()
    head = model.panoptic_head
    matching_seed = spec['matching_loss_seed_base'] + index
    require(matching_seed == old_packet['matching_loss_seed'] == old_row['matching_loss_seed'], 'Original matching seed must be replayed')
    sample_seed = spec['point_seed_base'] + index
    outside = pref.rng_snapshot(torch)
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(matching_seed)
        torch.cuda.manual_seed(matching_seed)
        with joint._observe_processed_GT(torch, head, identity['ann_ids'], raw_GT, mapping, identity, validator) as gt_observation:
            with observe_all100_targets(torch, head, old.digest_tensor) as (matches, all_targets):
                with cache7.observed_fields(torch, head) as fields:
                    losses = model.loss(data['inputs'], data['data_samples'])
        require(set(losses) == set(old.ATOMIC_KEYS) and len(losses) == 30, 'Exactly thirty original native loss atoms required')
        loss_values = {}
        for name, value in losses.items():
            atoms = list(value) if isinstance(value, (tuple, list)) else [value]
            require(atoms and all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in atoms), 'Frozen native losses must be finite')
            loss_values[name] = float(sum(v.mean() for v in atoms).cpu())
        queries = cache7.matched_queries(matches, mapping, len(raw_GT))
        require(matches == old_packet['native_stage_matches'] and queries == old_packet['query_ids'].tolist() == old_row['query_ids'],
                'Exact old ten-stage matching and raw-GT/query map must be reproduced')
        F, e = fields['F'][0], fields['e'][-1]
        classes, mask_stages = fields['output'][0]
        native_class, M = classes[-1], mask_stages[-1]
        matched_q = torch.tensor(queries, dtype=torch.int64, device=e.device)
        require(torch.equal(e[0, matched_q].cpu(), old_packet['query_embeddings'])
                and torch.equal(native_class.cpu(), old_packet['native_class_logits']), 'Exact old final e/C required')
        assignment = final_assignment(torch, all_targets[-1], mapping, identity['ann_ids'], head.num_classes)
        require(all(int(assignment['raw_GT_index_by_query'][q]) == g and
                    assignment['raw_GT_annotation_id_by_query'][q] == identity['ann_ids'][g]
                    for g, q in enumerate(queries)), 'Final assignment must equal old matched query/raw annotation IDs exactly')
        render = lambda value, identity: diag.render_original(torch, value, identity)
        natural_L = render(M, identity)
        require(natural_L.dtype == torch.float32 and natural_L.device.type == 'cuda'
                and tuple(natural_L.shape) == (100,) + tuple(identity['original_shape']) and bool(torch.isfinite(natural_L).all()),
                'Natural all100 original GPU L required')
        natural_cpu = natural_L.detach().cpu().clone().contiguous()
        # FIT-only diagnostic, never used by assignment, sampling, labels or deployment.
        native_binary_float = (natural_L > 0).flatten(1).float()
        raw_GT_float = raw_GT.to(natural_L.device).flatten(1).float()
        intersections = native_binary_float @ raw_GT_float.T
        unions = native_binary_float.sum(1, keepdim=True) + raw_GT_float.sum(1)[None, :] - intersections
        require(bool((unions > 0).all()), 'Original positive raw GT yields nonempty diagnostic unions')
        max_IoU = (intersections / unions).max(1).values.detach().cpu().contiguous()
        del native_binary_float, raw_GT_float, intersections, unions
        old_replay = natural_cpu[torch.tensor(queries, dtype=torch.int64)].flatten(1).gather(1, old_packet['point_indices'])
        # Legacy matched-only native rendering was numerically qualified, not bit-identical.
        require(torch.allclose(old_replay, old_packet['native_logits'], rtol=spec['old_native_L_rtol'], atol=spec['old_native_L_atol']),
                'ROOT must carry the OLD qualified practical L tolerance, never silently replace it')
        domain_indices, C_counts, K_counts = {}, [], []
        for q in range(100):
            P, Omega, C, idx, positions = diag15.domains_from_native(np, ndimage, natural_cpu[q].numpy())
            C_counts.append(len(idx)); K_counts.append(len(positions))
            domain_indices[q] = torch.from_numpy(idx)
        active = [q for q in range(100) if assignment['valid_targets'][q] and C_counts[q] > 0 and K_counts[q] > 0]
        positive = [q for q in active if assignment['supervision_kind_by_query'][q] == 'matched']
        background = [q for q in active if assignment['supervision_kind_by_query'][q] == 'background']
        sample = sample_stratified(torch, domain_indices, positive, background, sample_seed)
        sq, flat = sample['query_ids'], sample['flat_indices']
        unique_flat, sample_to_unique = torch.unique(flat, sorted=True, return_inverse=True)
        features = sampled_features_unique(torch, render, F, identity, unique_flat)
        native = natural_cpu.flatten(1)[sq, flat].contiguous()
        targets = torch.zeros(len(flat), dtype=torch.bool)
        for j, q in enumerate(sq.tolist()):
            g = int(assignment['raw_GT_index_by_query'][q].item())
            require(g >= -1, 'Ignored query cannot receive empty-set target')
            if g >= 0:
                targets[j] = raw_cpu[g].flatten()[flat[j]]
        residual, combined = torch.empty_like(native), torch.empty_like(native)
        require(not pr_head.training and all(not p.requires_grad and p.grad is None for p in pr_head.parameters()),
                'ROOT supplies frozen qualified strong E24 head, never a trainable/N14 head')
        PR_calls = 0
        for first in range(0, len(flat), 2048):
            stop = min(first + 2048, len(flat))
            f = features[sample_to_unique[first:stop]].to(e.device)
            query_e = e[0, sq[first:stop].to(e.device)]
            logits = native[first:stop].to(e.device)
            result = pr_head(f, query_e, logits, logits.new_zeros((stop-first, 49)), arm='PR')
            u = result['residual']
            require(u.dtype == torch.float32 and u.shape == logits.shape and bool(torch.isfinite(u).all()), 'Actual ungated fixed PR action must be finite FP32')
            after = logits + u
            require(bool(torch.isfinite(after).all()), 'Actual combined action must be finite FP32')
            residual[first:stop].copy_(u.cpu()); combined[first:stop].copy_(after.cpu())
            PR_calls += 1
            del f, query_e, logits, result, u, after
        P_sample, A_sample = native.numpy() > 0, combined.numpy() > 0
        G_sample = targets.numpy()
        labels = dict(utility_labels=torch.from_numpy(helper.utility_labels(P_sample, G_sample, A_sample)),
                      error_labels=torch.from_numpy(helper.error_labels(P_sample, G_sample)),
                      delta=torch.from_numpy(P_sample != A_sample), error=torch.from_numpy(P_sample != G_sample))
        require(labels['utility_labels'].dtype == labels['error_labels'].dtype == torch.int64, 'Fixed labelclass dtype/order required')
        packet = dict(schema='N16-all100-FIT-action-sampled-cache-body-v1', dataset_index=index, identity=identity,
            natural_query_ids=list(range(100)), native_e0=e.detach().cpu().clone().contiguous(),
            native_class_logits=native_class.detach().cpu().clone().contiguous(), natural_native_L=natural_cpu,
            max_native_mask_IoU_to_any_GT=max_IoU, max_IoU_is_FIT_diagnostic_only=True,
            assignment=assignment, all_ten_target_observations=all_targets, native_stage_matches=matches,
            C_counts_all100=C_counts, K_counts_all100=K_counts, active_queries=active,
            active_matched_queries=positive, active_background_queries=background,
            unique_flat_indices=unique_flat, sampled_F_unique=features, sample_to_unique=sample_to_unique,
            samples=dict(**sample, native_L=native, actual_PR_residual=residual, actual_combined=combined,
                         native_P=torch.from_numpy(P_sample), actual_after_A=torch.from_numpy(A_sample), targets=targets, **labels),
            matching_loss_seed=matching_seed, native_loss_atoms=loss_values,
            raw_target_mapping=mapping, raw_GT_ids=identity['ann_ids'], raw_GT_digest=pref.digest_tensor(raw_GT),
            original_cache_packet=old_row['packet'],
            actual_raw_processed_GT_correspondence=gt_observation, processed_target_proof=processed_proof,
            new_background_empty_set_supervision=True, candidate_GT_independent=True,
            helper_classes=dict(utility=['help','harm','noop'], error=['FP','FN','correct']),
            PR_label_generation_chunk_calls=PR_calls, actual_label_generation_points=len(flat),
            fixed_refiner_has_seen_FIT=True, paired_scout_same_samples=True, optimizer_created=False, weights_updated=False,
            numeric_CPU_qualified=False, actual_source_terminal_qualified=False,
            numeric_rule='ROOT independent same-GPU-F/e/L PR formula replay: absolute1e-4,rtol0,strictsign; pending',
            original_matcher_negative_mask_supervision=False, original_global_RNG_restored=True)
    require(pref.tree_equal(torch, outside, pref.rng_snapshot(torch)), 'Private teacher matching RNG must restore external state')
    require(not model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'Teacher remains frozen/eval')
    raw_bytes = tensor_payload_bytes(torch, packet)
    N = identity['original_shape'][0] * identity['original_shape'][1]
    weighted_counts = {}
    for name, stratum_id in (('matched', 0), ('background', 1)):
        in_group = sample['stratum_ids'] == stratum_id
        weighted_counts[name] = dict(points=int(in_group.sum()),
            utility=[float(sample['importance_weights'][in_group & (labels['utility_labels'] == k)].double().sum()) for k in range(3)],
            error=[float(sample['importance_weights'][in_group & (labels['error_labels'] == k)].double().sum()) for k in range(3)])
    bg_queries = [q for q in range(100) if assignment['supervision_kind_by_query'][q] == 'background']
    row = dict(dataset_index=index, image_id=identity['image_id'], identity=identity, raw_GT=len(raw_GT),
        natural_queries=100, points=len(flat), unique_sampled_F_points=len(unique_flat),
        active_queries=len(active), active_matched_queries=len(positive), active_background_queries=len(background),
        matching_loss_seed=matching_seed, point_seed=sample_seed, C_counts_all100=C_counts, K_counts_all100=K_counts,
        stratum_plan=sample['plan'], utility_class_counts=torch.bincount(labels['utility_labels'], minlength=3).tolist(),
        error_class_counts=torch.bincount(labels['error_labels'], minlength=3).tolist(),
        weighted_class_counts_by_stratum=weighted_counts,
        importance_weight_min=float(sample['importance_weights'].min()) if len(flat) else None,
        importance_weight_max=float(sample['importance_weights'].max()) if len(flat) else None,
        importance_weight_sum=float(sample['importance_weights'].double().sum()),
        native_background_queries=len(bg_queries),
        native_background_max_IoU_ge_point5=sum(float(max_IoU[q]) >= .5 for q in bg_queries),
        background_IoU_is_duplicate_or_potential_TP_diagnostic_not_label=True,
        raw_tensor_payload_bytes=raw_bytes, natural_L_raw_bytes=400*N,
        unique_F_and_index_raw_bytes=1032*len(unique_flat), maximum_unique_F_and_index_raw_bytes=1032*4096,
        label_generation_PR_points=len(flat), label_generation_PR_chunk_calls=PR_calls,
        one_teacher_loss_forward=True, all_ten_matching_stages=10, old_matching_e_C_exact=True,
        old_L_practical_replay=True, new_background_empty_set_supervision=True, no_AP=True, no_training=True,
        CPU_qualification_pending=True, resource_scope='ROOT separately budgets raw/readback/metadata/serialized/RAM/VRAM')
    return row, packet
