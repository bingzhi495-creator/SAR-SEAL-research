"""UNBOUND / NOT_EXECUTED N20 fused DEV scoring body; no IO/teacher/renderer/gates/COCO/CLI.

Run after the per-image extra-F capture body, in the same owned GPU leaf. ROOT
keeps the original physical shared carrier and new extra/map member, frozen
E8/E24 heads/PR and all source/terminal/native/R/export qualification. This body
does not independently qualify a newly captured input. No GT is an input.
"""

from time import perf_counter

ARMS = ('LOCAL_ERROR', 'CENTER_ERROR')
EPOCHS = (24, 8)
MODES = ('atmost', 'exact')
POINT_CHUNK = 2048
ABSOLUTE_TOLERANCE = 1e-4
STATE_SHAPES = {'spatial_weight': (9, 256), 'hidden.weight': (32, 513),
                'hidden.bias': (32,), 'output.weight': (3, 32), 'output.bias': (3,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _union_numpy(torch, np, value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu' and value.layout == torch.strided
                and value.dtype == torch.int64 and value.ndim == 1, 'CPU INT64 flat indices')
        value = value.numpy()
    require(isinstance(value, np.ndarray) and value.dtype == np.int64 and value.ndim == 1
            and (len(value) == 0 or bool(np.all(value[1:] > value[:-1]))),
            'Strict unique ascending INT64 physical indices')
    return value


def _positions(np, union, flat):
    positions = np.searchsorted(union, flat)
    require(len(flat) == 0 or (bool(np.all(positions < len(union)))
            and np.array_equal(union[positions], flat)), 'Every point must be covered by actual old union F')
    return positions.astype(np.int64, copy=False)


def _neighbor_inputs(torch, np, legacy, union, old_F, value, H, W, device):
    require(type(value) is dict and set(value) == {
        'extra_F', 'extra_flat_indices', 'neighbor_to_combined'}, 'Exact three-key extra/map input')
    extra = value['extra_F']
    flats = _union_numpy(torch, np, value['extra_flat_indices'])
    mapping = value['neighbor_to_combined']
    U, E = len(union), len(flats)
    legacy._finite_tensor(torch, extra, (E, 256), device, 'Actual GPU extra F transferred once by caller')
    require(E <= min(8 * U, H * W - U) and bool(((flats >= 0) & (flats < H * W)).all())
            and len(np.intersect1d(union, flats)) == 0, 'Extra union-minus-centers bounds/disjointness')
    require(isinstance(mapping, torch.Tensor) and mapping.device.type == 'cpu'
            and mapping.layout == torch.strided and mapping.dtype == torch.int64
            and tuple(mapping.shape) == (U, 9) and not mapping.requires_grad
            and bool(((mapping >= 0) & (mapping < U + E)).all())
            and torch.equal(mapping[:, 4], torch.arange(U, dtype=torch.int64)), 'Exact CPU mapping and center column4')
    y, x = union // W, union % W
    neighbors = np.stack([np.clip(y + dy, 0, H - 1) * W + np.clip(x + dx, 0, W - 1)
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
    combined_flat = np.concatenate((union, flats))
    require(np.array_equal(combined_flat[mapping.numpy()], neighbors)
            and np.array_equal(np.setdiff1d(np.unique(neighbors), union), flats),
            'Exact original-pixel radius1 replicate mapping and extra difference')
    # One image-level copy of unique raw values, no shared model projection.
    return torch.cat((old_F, extra), dim=0), mapping


def _cost(legacy):
    value = legacy._cost()
    value.update(feature_neighbor_gather_calls=0, feature_neighbor_gather_points=0,
                 feature_neighbor_gather_seconds=0., captured_common_blocks=0,
                 captured_common_reference_seconds=0.)
    return value


def _captured_common(torch, np, legacy, blocks, L, e, union, F_unique,
                     q, active, cost, callback, check):
    """Reuse ROOT-decoded original GPU common-C blocks, never call the PR head."""
    tick = perf_counter()
    n, device = len(active), L.device
    require(type(blocks) in (list, tuple) and len(blocks) == (n + POINT_CHUNK - 1) // POINT_CHUNK,
            'Complete captured common-C blocks, empty only when original active support is empty')
    output = dict(native_L=np.empty(n, dtype=np.float32), residual=np.empty(n, dtype=np.float32),
        combined=np.empty(n, dtype=np.float32),
        actual_GPU_residual=torch.empty(n, dtype=torch.float32, device=device),
        actual_GPU_combined=torch.empty(n, dtype=torch.float32, device=device),
        actual_GPU_values_reloaded_from_qualified_capture=True, fresh_PR_forward_executed=False)
    for block_index, block in enumerate(blocks):
        check()
        require(type(block) is dict and set(block) == {
            'flat_indices', 'sampled_F', 'e0', 'native_L', 'GPU_raw_residual', 'GPU_combined'},
            'Original decoder six-field common PR block ABI')
        first = block_index * POINT_CHUNK
        stop = min(first + POINT_CHUNK, n)
        flat = block['flat_indices']
        require(isinstance(flat, torch.Tensor) and flat.device.type == 'cpu' and flat.dtype == torch.int64
                and tuple(flat.shape) == (stop - first,) and np.array_equal(flat.numpy(), active[first:stop]),
                'Actual common-C source chunks must cover complete canonical C in original2048 blocks')
        for key, shape in (('sampled_F', (stop - first, 256)), ('e0', (256,)),
                           ('native_L', (stop - first,)), ('GPU_raw_residual', (stop - first,)),
                           ('GPU_combined', (stop - first,))):
            value = block[key]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    and value.dtype == torch.float32 and tuple(value.shape) == shape
                    and not value.requires_grad and bool(torch.isfinite(value).all()),
                    'Exact finite saved actual common PR CPU tensor: ' + key)
        positions = torch.from_numpy(_positions(np, union, flat.numpy()).copy()).to(device)
        expected_F = F_unique.index_select(0, positions).detach().cpu().contiguous()
        index = flat.to(device)
        expected_L = L[q, index // L.shape[2], index % L.shape[2]].detach().cpu().contiguous()
        for actual, expected in ((block['sampled_F'], expected_F),
                                 (block['e0'], e[q].detach().cpu()), (block['native_L'], expected_L)):
            require(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)),
                    'Prior common PR must use exactly the same actual F/e/L bytes')
        output['native_L'][first:stop] = block['native_L'].numpy()
        output['residual'][first:stop] = block['GPU_raw_residual'].numpy()
        output['combined'][first:stop] = block['GPU_combined'].numpy()
        output['actual_GPU_residual'][first:stop].copy_(block['GPU_raw_residual'])
        output['actual_GPU_combined'][first:stop].copy_(block['GPU_combined'])
        cost['captured_common_blocks'] += 1
        if callback is not None:
            capture_tick = perf_counter()
            event = dict(query_id=q, control='COMMON_C_REFERENCE', scout_epoch=None, evalmode='shared',
                stage='full_C_PR', point_chunk=POINT_CHUNK, chunk_first=first, flat_indices=flat.clone(),
                sampled_F=F_unique.index_select(0, positions).detach(), e0=e[q].detach(),
                native_L=L[q, index // L.shape[2], index % L.shape[2]].detach(),
                GPU_raw_residual=output['actual_GPU_residual'][first:stop].detach(),
                GPU_combined=output['actual_GPU_combined'][first:stop].detach(),
                actual_device_type=device.type, synthetic_only=(device.type == 'cpu'),
                actual_input_scope='actual_GPU_old_union_plus_new_extra_feature_neighbors',
                actual_Source_REF_qualified_here=False, reused_qualified_common_PR=True,
                fresh_PR_forward_executed=False, result_origin='qualified_prior_GPU_common_C_capture')
            callback(event)
            cost['callback_calls'] += 1
            cost['callback_seconds'] += perf_counter() - capture_tick
        del index, positions, expected_F, expected_L
    cost['captured_common_reference_seconds'] += perf_counter() - tick
    return output


def _run_points(torch, np, legacy, L, e, union, F_unique, combined_F, mapping,
                q, indices, model, kind, cost, callback, check, control, epoch, mode, stage):
    n, device = len(indices), L.device
    require(indices.dtype == np.int64 and indices.ndim == 1
            and (n == 0 or bool(np.all(indices[1:] > indices[:-1]))), 'Canonical ascending distinct supports')
    output = dict(native_L=np.empty(n, dtype=np.float32))
    if kind == 'scout':
        actual = torch.empty((n, 3), dtype=torch.float32, device=device)
        output.update(logits=np.empty((n, 3), dtype=np.float32), actual_GPU_logits=actual)
    else:
        output.update(residual=np.empty(n, dtype=np.float32), combined=np.empty(n, dtype=np.float32),
            actual_GPU_residual=torch.empty(n, dtype=torch.float32, device=device),
            actual_GPU_combined=torch.empty(n, dtype=torch.float32, device=device))
    for first in range(0, n, POINT_CHUNK):
        check()
        stop = min(first + POINT_CHUNK, n)
        flat = indices[first:stop]
        index = torch.from_numpy(flat.copy()).to(device)
        legacy._sync(torch, device)
        tick = perf_counter()
        positions = torch.from_numpy(_positions(np, union, flat).copy())
        F = F_unique.index_select(0, positions.to(device))
        embedding = e[q].unsqueeze(0).expand(stop - first, 256)
        native = L[q, index // L.shape[2], index % L.shape[2]]
        legacy._sync(torch, device)
        cost['gather_calls'] += 1
        cost['gather_points'] += stop - first
        cost['gather_seconds'] += perf_counter() - tick
        if kind == 'scout':
            tick = perf_counter()
            if control == 'LOCAL_ERROR':
                rows = mapping.index_select(0, positions).reshape(-1).to(device)
                neighbors = combined_F.index_select(0, rows).reshape(stop - first, 9, 256)
                del rows
            else:
                neighbors = F[:, None, :].repeat(1, 9, 1)
            legacy._finite_tensor(torch, neighbors, (stop - first, 9, 256), device, 'Actual nine F inputs')
            legacy._sync(torch, device)
            cost['feature_neighbor_gather_calls'] += 1
            cost['feature_neighbor_gather_points'] += stop - first
            cost['feature_neighbor_gather_seconds'] += perf_counter() - tick
            tick = perf_counter()
            result = model(F, embedding, native, neighbors)
            legacy._finite_tensor(torch, result, (stop - first, 3), device, 'Actual five-tensor scout logits')
            legacy._sync(torch, device)
            cost['scout_calls'] += 1
            cost['scout_points'] += stop - first
            cost['scout_seconds'] += perf_counter() - tick
            actual[first:stop].copy_(result)
        else:
            tick = perf_counter()
            result = model(F, embedding, native, native.new_zeros((stop - first, 49)), arm='PR')
            require(isinstance(result, dict) and 'residual' in result, 'Original raw PR residual ABI')
            residual = result['residual']
            combined = native + residual
            legacy._finite_tensor(torch, residual, (stop - first,), device, 'Actual original residual')
            legacy._finite_tensor(torch, combined, (stop - first,), device, 'Actual FP32 L+u')
            legacy._sync(torch, device)
            cost['PR_calls'] += 1
            cost['PR_points'] += stop - first
            cost['PR_seconds'] += perf_counter() - tick
            output['actual_GPU_residual'][first:stop].copy_(residual)
            output['actual_GPU_combined'][first:stop].copy_(combined)
        tick = perf_counter()
        output['native_L'][first:stop] = native.detach().cpu().numpy()
        if kind == 'scout':
            output['logits'][first:stop] = result.detach().cpu().numpy()
        else:
            output['residual'][first:stop] = residual.detach().cpu().numpy()
            output['combined'][first:stop] = combined.detach().cpu().numpy()
        cost['readback_seconds'] += perf_counter() - tick
        if callback is not None:
            tick = perf_counter()
            event = dict(query_id=q, control=control, scout_epoch=epoch, evalmode=mode, stage=stage,
                point_chunk=POINT_CHUNK, chunk_first=first, flat_indices=torch.from_numpy(flat.copy()),
                sampled_F=F.detach(), e0=e[q].detach(), native_L=native.detach(),
                actual_device_type=device.type, synthetic_only=(device.type == 'cpu'),
                actual_input_scope='actual_GPU_old_union_plus_new_extra_feature_neighbors',
                actual_Source_REF_qualified_here=False)
            if kind == 'scout':
                event.update(neighbors_F=neighbors.detach(), neighbor_union_positions=positions.clone(),
                    neighbor_center_column=4, actual_GPU_logits=result.detach(), scout_logits=result.detach())
            else:
                event.update(GPU_raw_residual=residual.detach(), GPU_combined=combined.detach())
            callback(event)
            cost['callback_calls'] += 1
            cost['callback_seconds'] += perf_counter() - tick
            del event
        if kind == 'scout':
            del neighbors
        else:
            del residual, combined
        del index, positions, F, embedding, native, result
    return output


def evaluate_all100(torch, np, ndimage, helper, legacy_evalbody, native_L, e0,
                    union_flat_indices, F_unique, neighbor_inputs, models, frozen_PR,
                    *, point_callback=None, check_callback=None, common_PR_provider=None,
                    synthetic_only=False):
    """Yield eight policy profiles/query, preserving N17 selection/action rules.

    neighbor_inputs is exactly {extra_F,extra_flat_indices,neighbor_to_combined}:
    extra_F is already FP32 on the same GPU as old F_unique (one caller transfer),
    indices/map are CPU INT64. No full-C [C,9,256] array is built: the temporary
    nine-F inputs are at most2048 rows/18MiB. Ranking is once over ALL C scores,
    never chunk-local. Only saved actual GPU logits define FP64 ERROR ranking.

    Callback must copy/check actual chunk tensors before retaining them. A fused
    bundle may deduplicate neighbors_F only after byteexact comparison against
    its persisted old-F+extra/map inputs, keeping a member ID and actual flats.
    Old N17 extra8/native-L decoder is incompatible with this new scout input.

    Optional common_PR_provider(q,active_C_indices,K) returns the original closed
    N17 decoder's six-field CPU common_PR_chunks. ROOT authenticates its physical
    bundle/members/carrier/protocol/head/query/K. Reuse avoids a new full-C PR
    forward. Callbacks mark those values as reloaded old actual outputs, not new
    PR execution. Every new selected support still receives fresh original PR
    and the unchanged absolute1e-4 plus strict-sign check against that reference.
    If no provider is supplied, the original fresh full-C reference path remains.
    """
    require(type(synthetic_only) is bool and (point_callback is None or callable(point_callback))
            and (check_callback is None or callable(check_callback))
            and (common_PR_provider is None or callable(common_PR_provider)), 'Explicit mode and callable hooks')
    check = (lambda: None) if check_callback is None else check_callback
    require(isinstance(native_L, torch.Tensor) and native_L.ndim == 3
            and min(native_L.shape) > 0, 'Original natural-query L grid')
    device = native_L.device
    Q, H, W = native_L.shape
    require((device.type == 'cuda' and Q == 100 and not synthetic_only)
            or (device.type == 'cpu' and synthetic_only), 'Actual CUDA natural100 only; CPU is synthetic')
    require(not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled(), 'FP32, no AMP')
    legacy = legacy_evalbody
    require(legacy.POINT_CHUNK == POINT_CHUNK and legacy.ABSOLUTE_TOLERANCE == 1e-4,
            'Unchanged original2048 chunk and absolute tolerance')
    legacy._finite_tensor(torch, native_L, (Q, H, W), device, 'Original L')
    legacy._finite_tensor(torch, e0, (Q, 256), device, 'Original embeddings')
    union = _union_numpy(torch, np, union_flat_indices)
    require(bool(((union >= 0) & (union < H * W)).all()), 'Original union bounds')
    legacy._finite_tensor(torch, F_unique, (len(union), 256), device, 'Old actual unique center F')
    combined_F, mapping = _neighbor_inputs(torch, np, legacy, union, F_unique, neighbor_inputs, H, W, device)
    require(type(models) is dict and set(models) == {(ep, arm) for ep in EPOCHS for arm in ARMS},
            'All fixed E24/E8 two-arm models required, no winner selection')
    for key, model in list(models.items()) + [('fixed_PR', frozen_PR)]:
        require(not model.training, 'Frozen eval models only; state/mode never changed here')
        for parameter in model.parameters():
            require(parameter.device == device and parameter.dtype == torch.float32
                    and not parameter.requires_grad and parameter.grad is None
                    and bool(torch.isfinite(parameter).all()), 'Frozen finite FP32 parameters on input device')
        if key != 'fixed_PR':
            state = model.state_dict()
            require(set(state) == set(STATE_SHAPES) and not tuple(model.buffers())
                    and len(tuple(model.parameters())) == 5 and sum(v.numel() for v in state.values()) == 18851
                    and all(tuple(state[name].shape) == shape for name, shape in STATE_SHAPES.items()),
                    'Five trained18851 N20 scout tensors; spatial weights may be zero or nonzero')
    merged, domains = np.zeros((H, W), dtype=np.bool_), []
    for q in range(Q):
        check()
        L_cpu = native_L[q].detach().cpu().clone().contiguous().numpy()
        domain = legacy.domains_from_native(np, ndimage, L_cpu)
        merged |= domain[2]
        domains.append((L_cpu, domain))
    require(np.array_equal(union, np.flatnonzero(merged).astype(np.int64)), 'Exact complete natural-query C union')
    for q in range(Q):
        check()
        start = perf_counter()
        L_cpu, (P, Omega, C, C_indices, K) = domains[q]
        active = C_indices if K else np.empty(0, dtype=np.int64)
        common_cost, profiles, model_costs, policy_costs = _cost(legacy), {}, {}, {}
        with torch.no_grad():
            if common_PR_provider is None:
                common = _run_points(torch, np, legacy, native_L, e0, union, F_unique, combined_F, mapping,
                    q, active, frozen_PR, 'PR', common_cost, point_callback, check,
                    'COMMON_C_REFERENCE', None, 'shared', 'full_C_PR')
            else:
                provider_tick = perf_counter()
                blocks = common_PR_provider(q, active.copy(), K)
                common_cost['captured_common_reference_seconds'] += perf_counter() - provider_tick
                common = _captured_common(torch, np, legacy, blocks, native_L, e0, union, F_unique,
                    q, active, common_cost, point_callback, check)
                del blocks
            for epoch in EPOCHS:
                for arm in ARMS:
                    model_cost = _cost(legacy)
                    points = _run_points(torch, np, legacy, native_L, e0, union, F_unique, combined_F, mapping,
                        q, active, models[(epoch, arm)], 'scout', model_cost, point_callback, check,
                        arm, epoch, 'shared-atmost-exact', 'scout_C_logits')
                    if K:
                        tick = perf_counter()
                        scores = helper.error_score(points['logits'])
                        require(scores.dtype == np.float64 and scores.shape == (len(C_indices),)
                                and bool(np.isfinite(scores).all()), 'Canonical FP64 ERROR scores from actual GPU logits')
                        model_cost['score_calls'] += 1
                        model_cost['score_seconds'] += perf_counter() - tick
                    else:
                        scores = np.empty(0, dtype=np.float64)
                    grid = np.zeros((H, W), dtype=np.float64)
                    if K:
                        grid.ravel()[C_indices] = scores
                    model_costs['E%d_%s' % (epoch, arm)] = model_cost
                    for mode in MODES:
                        check()
                        cost = _cost(legacy)
                        tick = perf_counter()
                        selected = helper.choose_candidates(L_cpu, C, grid, K, mode=mode)
                        cost['sorting_calls'] += 1
                        cost['sorting_seconds'] += perf_counter() - tick
                        selected_flat = np.flatnonzero(selected).astype(np.int64)
                        require(len(selected_flat) <= K and not bool((selected & ~C).any())
                                and (mode != 'exact' or len(selected_flat) == K), 'Original true K support')
                        subset = _run_points(torch, np, legacy, native_L, e0, union, F_unique, combined_F, mapping,
                            q, selected_flat, frozen_PR, 'PR', cost, point_callback, check, arm, epoch, mode, 'selected_PR')
                        compatibility = legacy._qualify_subset(np, q, arm, selected_flat, subset, common, active)
                        tick = perf_counter()
                        mask = P.copy()
                        mask.ravel()[selected_flat] = subset['combined'] > 0
                        require(np.array_equal(mask[~selected], P[~selected]), 'Outside selected support stays native')
                        cost['mask_seconds'] += perf_counter() - tick
                        name = 'E%d_%s_%s' % (epoch, arm, mode)
                        policy_costs[name] = cost
                        profiles[name] = dict(epoch=epoch, arm=arm, mode=mode,
                            main_result=(epoch == 24 and mode == 'atmost'), mask=mask,
                            selected_flat_indices=selected_flat, actual_GPU_C_logits=points['actual_GPU_logits'],
                            canonical_CPU_FP64_C_scores=scores, native_selected_L=subset['native_L'],
                            actual_raw_residual=subset['residual'], actual_combined=subset['combined'],
                            actual_GPU_selected_residual=subset['actual_GPU_residual'],
                            actual_GPU_selected_combined=subset['actual_GPU_combined'], PR_compatibility=compatibility,
                            model_cost_key='E%d_%s' % (epoch, arm), policy_cost=cost)
                        del subset
                    del points, grid
        legacy._sync(torch, device)
        yield dict(query_id=q, natural_queries=Q, all100_actual_path=(not synthetic_only), synthetic_only=synthetic_only,
            native_L=L_cpu, native_mask=P, Omega=Omega, C=C, C_flat_indices=C_indices, K=K, profiles=profiles,
            common_C=common, common_C_PR_cost=common_cost, per_model_costs=model_costs, per_policy_costs=policy_costs,
            shared_diagnostic_seconds=perf_counter() - start, actual_GPU_logits_shared_atmost_exact_same_model=True,
            cross_model_forward_or_projection_shared=False, native_R_refs_loaded=False, reference_reuse_verified=False,
            numerical_qualification_conferred=False, ROOT_Source_or_terminal_qualified=False, AP_evaluated=False,
            original_scores_classes_S0_export_order_not_consumed_or_modified=True, main_epoch=24, curve_epoch=8,
            actual_GPU_neighbor_inputs_require_capture_byte_proof=True, neighbor_input_schema='N20-feature-neighbors9-v1',
            common_C_reference_reused=(common_PR_provider is not None), selected_PR_fresh_on_new_supports=True,
            standalone_total_time_measured=False, inclusive_costs_nonadditive=True)
