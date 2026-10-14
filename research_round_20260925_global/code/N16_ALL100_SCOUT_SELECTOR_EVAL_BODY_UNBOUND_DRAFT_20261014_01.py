"""UNBOUND: GT-independent N16 deployment algorithm body, without a runner.

ROOT owns real gates, trained states, source closure, CUDA/resource admission,
native F rendering, S0/classes/query order, export, COCO and all file access.
This module accepts already-rendered actual FP32 F/e/L, never GT or targets.
Only synthetic CPU execution is supported by synthetic_only=True.

UTILITY/ERROR score all C then run original PR only on selected ascending points.
Common full-C PR is a separately measured numerical reference and the necessary
input of dense-flip/MagNet controls. Every cheap subset must match its common-C
residual/combined within absolute1e-4, rtol0, AND exact strict>0 signs. Failure
raises; no fallback, tolerance adjustment or fake noop is permitted.

This shared diagnostic execution is NOT a standalone speed measurement. Full-C
strong-control PR costs cannot be taken as zero or inferred by subtraction.
Training's mixed-query/replacement batches versus deployment C batches still
need a FUTURE ACTUAL same-input absolute1e-4/strict-sign/label compatibility gate;
neither this body nor synthetic tests establish that scientific qualification.
"""
from time import perf_counter

POINT_CHUNK = 2048
ABSOLUTE_TOLERANCE = 1.e-4
MAIN_CONTROLS = ('UTILITY', 'ERROR', 'DENSE_ERROR_FLIP', 'MAGNET')


class NumericQualificationFailure(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('N16_SELECTED_PR_BATCH_NUMERICAL_QUALIFICATION_FAILURE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sync(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _finite_tensor(torch, value, shape, device, name):
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided
            and value.dtype == torch.float32 and value.device == device
            and tuple(value.shape) == tuple(shape), name + ': exact dense FP32 shape/device required')
    require(not value.requires_grad and bool(torch.isfinite(value).all()), name + ': finite detached input/output required')


def domains_from_native(np, ndimage, native_L):
    """Identical physical domain expression to the ordinary N15 body excerpt."""
    require(isinstance(native_L, np.ndarray) and not np.ma.isMaskedArray(native_L)
            and native_L.dtype == np.float32 and native_L.ndim == 2
            and bool(np.isfinite(native_L).all()), 'Finite FP32 native L[H,W] required')
    P, Omega = native_L > 0, np.abs(native_L) <= 2
    dilation = ndimage.binary_dilation(P, structure=np.ones((17, 17), dtype=np.bool_),
                                     iterations=1, border_value=0)
    C = Omega | P | dilation
    indices = np.flatnonzero(C).astype(np.int64)
    K = int(np.count_nonzero(Omega))
    require(K <= len(indices) and not np.any(Omega & ~C) and not np.any(P & ~C), 'Physical C must contain Omega/native P')
    return P, Omega, C, indices, K


def _cost():
    return dict(gather_calls=0, gather_points=0, gather_seconds=0.,
                scout_calls=0, scout_points=0, scout_seconds=0.,
                PR_calls=0, PR_points=0, PR_seconds=0.,
                readback_seconds=0., score_calls=0, score_seconds=0.,
                sorting_calls=0, sorting_seconds=0., mask_seconds=0.,
                callback_calls=0, callback_seconds=0.)


def _run_points(torch, np, F, e0, Lq, q, indices, model, kind, cost,
                callback, control, evalmode, stage):
    """No shared projection: each scout invocation gathers and forwards anew."""
    n, device = len(indices), F.device
    require(indices.dtype == np.int64 and indices.ndim == 1
            and (n == 0 or bool(np.all(indices[1:] > indices[:-1]))), 'Real point indices must be distinct ascending INT64')
    native_out = np.empty(n, dtype=np.float32)
    if kind == 'scout':
        output = {'native_L': native_out, 'logits': np.empty((n, 3), dtype=np.float32)}
    else:
        output = {'native_L': native_out, 'residual': np.empty(n, dtype=np.float32),
                  'combined': np.empty(n, dtype=np.float32)}
    for first in range(0, n, POINT_CHUNK):
        stop = min(first + POINT_CHUNK, n)
        _sync(torch, device); start = perf_counter()
        index_cpu = torch.from_numpy(indices[first:stop].copy())
        index = index_cpu.to(device)
        row_index, col_index = index // Lq.shape[1], index % Lq.shape[1]
        # Gather actual spatial coordinates without flattening/copying a whole
        # non-contiguous 256-channel original tensor on every point chunk.
        f = F[:, row_index, col_index].t().contiguous()
        e = e0[q].unsqueeze(0).expand(stop - first, 256)
        native = Lq[row_index, col_index]
        _sync(torch, device)
        cost['gather_calls'] += 1; cost['gather_points'] += stop-first
        cost['gather_seconds'] += perf_counter() - start
        start = perf_counter()
        if kind == 'scout':
            result = model(f, e, native)
            _finite_tensor(torch, result, (stop-first, 3), device, 'Scout logits')
            _sync(torch, device)
            cost['scout_calls'] += 1; cost['scout_points'] += stop-first
            cost['scout_seconds'] += perf_counter() - start
        else:
            result = model(f, e, native, native.new_zeros((stop-first, 49)), arm='PR')
            require(isinstance(result, dict) and 'residual' in result, 'Original PR residual ABI required')
            residual = result['residual']
            _finite_tensor(torch, residual, (stop-first,), device, 'Original PR residual')
            combined = native + residual  # Actual original-device FP32 addition.
            _finite_tensor(torch, combined, (stop-first,), device, 'Actual PR combined logits')
            _sync(torch, device)
            cost['PR_calls'] += 1; cost['PR_points'] += stop-first
            cost['PR_seconds'] += perf_counter() - start
        start = perf_counter()
        native_cpu = native.detach().cpu().clone().contiguous()
        output['native_L'][first:stop] = native_cpu.numpy()
        if kind == 'scout':
            logits_cpu = result.detach().cpu().clone().contiguous()
            output['logits'][first:stop] = logits_cpu.numpy()
        else:
            residual_cpu = residual.detach().cpu().clone().contiguous()
            combined_cpu = combined.detach().cpu().clone().contiguous()
            output['residual'][first:stop] = residual_cpu.numpy()
            output['combined'][first:stop] = combined_cpu.numpy()
        cost['readback_seconds'] += perf_counter() - start
        if callback is not None:
            start = perf_counter()
            event = dict(query_id=q, control=control, evalmode=evalmode, stage=stage,
                         point_chunk=POINT_CHUNK, chunk_first=first,
                         flat_indices=index_cpu.clone(), sampled_F=f.detach().cpu().clone().contiguous(),
                         e0=e0[q].detach().cpu().clone().contiguous(), native_L=native_cpu.clone(),
                         actual_input_scope='caller_actual_original_device_FP32_F_e_L',
                         synthetic_only=(device.type == 'cpu'))
            if kind == 'scout':
                event['scout_logits'] = logits_cpu.clone()
            else:
                event['GPU_raw_residual'] = residual_cpu.clone()
                event['GPU_combined'] = combined_cpu.clone()
            callback(event)
            cost['callback_calls'] += 1
            cost['callback_seconds'] += perf_counter() - start
            del event
        del index_cpu, index, row_index, col_index, f, e, native, result, native_cpu
        if kind == 'scout':
            del logits_cpu
        else:
            del residual, combined, residual_cpu, combined_cpu
    return output


def _qualify_subset(np, q, control, selected, subset, common, C_indices):
    positions = np.searchsorted(C_indices, selected)
    require(bool(np.array_equal(C_indices[positions], selected)), 'Selected points must be physical C points')
    require(bool(np.array_equal(subset['native_L'], common['native_L'][positions])), 'Same actual native point inputs required')
    residual_error = np.abs(subset['residual'].astype(np.float64) - common['residual'][positions].astype(np.float64))
    combined_error = np.abs(subset['combined'].astype(np.float64) - common['combined'][positions].astype(np.float64))
    details = dict(query_id=q, control=control, compared_points=len(selected),
                   empty_selection=(len(selected) == 0), atol=ABSOLUTE_TOLERANCE, rtol=0.,
                   residual_max_absolute_error=float(residual_error.max()) if len(selected) else 0.,
                   combined_max_absolute_error=float(combined_error.max()) if len(selected) else 0.,
                   residual_absolute_pass=bool(np.all(residual_error <= ABSOLUTE_TOLERANCE)),
                   combined_absolute_pass=bool(np.all(combined_error <= ABSOLUTE_TOLERANCE)),
                   strict_sign_exact=bool(np.array_equal(subset['combined'] > 0, common['combined'][positions] > 0)),
                   reference='same_query_full_C_ascending_2048_actual_PR', fallback_used=False)
    if not (details['residual_absolute_pass'] and details['combined_absolute_pass'] and details['strict_sign_exact']):
        raise NumericQualificationFailure(details)
    return details


def evaluate_all100_scout_selectors(torch, np, ndimage, helper,
        F_original, e0, native_L, utility_scout, error_scout, frozen_PR,
        *, evalmode='atmost', point_callback=None, synthetic_only=False):
    """Yield one private CPU query result at a time, in natural order0..99.

    F_original is already rendered [256,H,W], e0 is [100,256], native_L is
    [100,H,W]. Actual inputs must be on the same CUDA device and models frozen
    in eval mode. No teacher rendering, native scores, classes, export order,
    target/GT, file I/O, projection-sharing or optimizer enters this interface.

    atmost returns four MAIN controls; exact returns ONLY two supplemental
    scouts. An external consumer/callback must release/store each query result
    as required; exhausting this iterator into a list defeats its memory scope.
    Point callbacks receive private actual CPU point inputs/outputs, not refs;
    ROOT must bind real capture files and provenance if used for qualification.
    """
    require(type(synthetic_only) is bool and evalmode in ('atmost', 'exact'), 'Explicit synthetic flag and budget mode required')
    require(point_callback is None or callable(point_callback), 'Point callback must be callable or None')
    require(isinstance(F_original, torch.Tensor) and F_original.ndim == 3
            and F_original.shape[0] == 256 and min(F_original.shape[1:]) > 0, 'Already-rendered F[256,H,W] required')
    device = F_original.device
    require(device.type == 'cuda' or (device.type == 'cpu' and synthetic_only), 'Actual path requires CUDA; CPU is SYNTHETIC_ONLY')
    require(not synthetic_only or device.type == 'cpu', 'This body only permits CPU synthetic execution')
    h, w = F_original.shape[1:]
    _finite_tensor(torch, F_original, (256,h,w), device, 'Original F')
    _finite_tensor(torch, e0, (100,256), device, 'Original e0')
    _finite_tensor(torch, native_L, (100,h,w), device, 'Original native L')
    for name, model in (('utility',utility_scout), ('error',error_scout), ('PR',frozen_PR)):
        require(not model.training, name + ' must be in eval mode; body does not change model mode')
        for parameter in model.parameters():
            require(parameter.device == device and parameter.dtype == torch.float32
                    and not parameter.requires_grad and parameter.grad is None
                    and bool(torch.isfinite(parameter).all()), name + ' must have frozen finite FP32 parameters on the input device')
    controls = MAIN_CONTROLS if evalmode == 'atmost' else ('UTILITY', 'ERROR')
    for q in range(100):
        _sync(torch, device); query_start = perf_counter()
        start = perf_counter()
        L_cpu = native_L[q].detach().cpu().clone().contiguous().numpy()
        P, Omega, C, C_indices, K = domains_from_native(np, ndimage, L_cpu)
        domain_seconds = perf_counter() - start
        common_cost, results = _cost(), {}
        common = dict(native_L=np.empty(0,dtype=np.float32), residual=np.empty(0,dtype=np.float32), combined=np.empty(0,dtype=np.float32))
        with torch.no_grad():
            if K > 0:
                common = _run_points(torch,np,F_original,e0,native_L[q],q,C_indices,frozen_PR,
                                     'PR',common_cost,point_callback,'COMMON_C_REFERENCE',evalmode,'full_C_PR')
            for control in controls:
                control_start, cost = perf_counter(), _cost()
                C_scores = np.empty(0,dtype=np.float64)
                if K == 0:
                    selected = np.zeros((h,w),dtype=np.bool_)
                else:
                    if control in ('UTILITY','ERROR','DENSE_ERROR_FLIP'):
                        model = utility_scout if control == 'UTILITY' else error_scout
                        points = _run_points(torch,np,F_original,e0,native_L[q],q,C_indices,model,
                                             'scout',cost,point_callback,control,evalmode,'scout_C_logits')
                    start = perf_counter()
                    if control == 'UTILITY':
                        C_scores = helper.utility_score(points['logits'])
                    elif control == 'ERROR':
                        C_scores = helper.error_score(points['logits'])
                    elif control == 'DENSE_ERROR_FLIP':
                        actual_flip = (common['native_L'] > 0) != (common['combined'] > 0)
                        C_scores = helper.dense_error_flip_score(actual_flip, points['logits'])
                    else:
                        C_scores = helper.magnet_score(common['native_L'],common['residual'])
                    require(C_scores.dtype == np.float64 and C_scores.shape == (len(C_indices),)
                            and bool(np.isfinite(C_scores).all()), 'Canonical CPU FP64 score required')
                    cost['score_calls'] += 1; cost['score_seconds'] += perf_counter()-start
                    score = np.zeros((h,w),dtype=np.float64)
                    score.ravel()[C_indices] = C_scores
                    start = perf_counter()
                    selected = helper.choose_candidates(L_cpu,C,score,K,mode=evalmode)
                    cost['sorting_calls'] += 1; cost['sorting_seconds'] += perf_counter()-start
                    del score
                    if control != 'MAGNET':
                        del points
                selected_flat = np.flatnonzero(selected).astype(np.int64)
                require(len(selected_flat) <= K and not np.any(selected & ~C)
                        and (evalmode != 'exact' or len(selected_flat) == K), 'Physical selected budget/support differs')
                if control in ('UTILITY','ERROR'):
                    subset = _run_points(torch,np,F_original,e0,native_L[q],q,selected_flat,frozen_PR,
                                         'PR',cost,point_callback,control,evalmode,'selected_PR')
                    compatibility = _qualify_subset(np,q,control,selected_flat,subset,common,C_indices)
                    raw_u, after = subset['residual'], subset['combined']
                    native_selected = subset['native_L']
                else:
                    positions = np.searchsorted(C_indices,selected_flat)
                    raw_u, after = common['residual'][positions].copy(), common['combined'][positions].copy()
                    native_selected = common['native_L'][positions].copy()
                    compatibility = None  # These controls execute their necessary full-C action.
                start = perf_counter()
                mask = P.copy()
                mask.ravel()[selected_flat] = after > 0  # Copy actual GPU addition/sign; do not recompute CPU L+u.
                require(mask.dtype == np.bool_ and bool(np.array_equal(mask[~selected],P[~selected])), 'Outside selected support must remain native exactly')
                cost['mask_seconds'] += perf_counter()-start
                results[control] = dict(evalmode=evalmode,mask=mask,selected_flat_indices=selected_flat,
                    native_selected_L=native_selected,actual_raw_residual=raw_u,actual_combined=after,
                    C_scores=C_scores,selected_only_vs_full_C_compatibility=compatibility,cost=cost,
                    control_measured_seconds=perf_counter()-control_start,
                    PR_role='selected_only' if control in ('UTILITY','ERROR') else 'shared_necessary_full_C',
                    strong_control_standalone_full_C_PR_calls_required=common_cost['PR_calls'] if control in ('DENSE_ERROR_FLIP','MAGNET') else None,
                    strong_control_full_C_PR_measured_separately=False,
                    score_is_not_a_new_residual=True,projection_shared_with_other_controls=False)
                if control in ('UTILITY','ERROR'):
                    del subset
        _sync(torch,device)
        yield dict(query_id=q,natural_query_ids=list(range(100)),evalmode=evalmode,
            supplemental_exact=(evalmode=='exact'),native_L=L_cpu,native_mask=P,
            Omega=Omega,C=C,C_flat_indices=C_indices,C_points=len(C_indices),K=K,
            controls=results,common_C=dict(**common,cost=common_cost,
                scope='qualification_reference_and_strong_control_required_input',
                numerical_forward_skipped_K0=(K==0)),domain_seconds=domain_seconds,
            shared_diagnostic_seconds=perf_counter()-query_start,
            shared_diagnostic_runtime_only=True,standalone_total_time_measured=False,
            costs_must_not_be_subtracted_to_claim_cheap_deployment_speed=True,
            requires_future_actual_training_to_C_batch_compatibility=True,
            independent_CPU_scout_and_exact_selected_indices_qualification='ROOT_external_pending',
            ROOT_scientific_qualification_conferred=False,synthetic_only=synthetic_only,
            original_scores_classes_and_export_order_touched=False,
            memory_scope='one_query_results_and_2048_point_chunks; consumer_must_release')
