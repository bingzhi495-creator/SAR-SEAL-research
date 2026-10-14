"""UNBOUND_DRAFT: independent CPU PR with captured exact GPU point inputs.

NOT RUN; not a complete runner, admission, Q, or efficacy/innovation conclusion.
ROOT must integrate this helper in a NEW bound CPU runner, retaining the old v01
failure unchanged. ROOT handles current own_gate(cpu=True), complete SOURCE
closure, legacy94/failed8/pins102, independent failed9, probe95, capture96 actual
GPU0 joins, CPU01/two threads/runtime/resources/deadline, locked374+6 provenance,
exclusive output and actual terminal/reap. See the accompanying plan/capture
ancestry helper. GPU owner_ref_join uses '0'; CPU uses ''. No source/data/weight
loader, scientific execution entry point or publishing is implemented here.

Torch/NumPy/OpenCV/SciPy, eight frozen CPU weights, exact GPU selector L and
actual candidate packets/GT arrays are explicit ROOT-qualified arguments. No CPU
feature rendering occurs. Independently copied functional LN/linear/ReLU consumes
captured actual GPU F_points in unchanged separate ascending2048 partitions.
Formula tolerances remain exactly1e-4 atol/rtol. Original near-zero support rule
is a practical allowance, not a certified bound; actual GPU C remains unchanged.
Integer width2, OpenCV-vs-SciPy distances, original LOG binary, fixed pair pixel
counts, per-pair and population JSON remain required. The pinned oracle/statistics
implementation is reused; this is not independent oracle sorting-code validation.
"""

import hashlib
import json
import math
import sys


SCHEMA = 'N13-fixed-PR-exact-GPU-point-inputs-v1'
SPEC_SCHEMA = 'ROOT-N13-fixed-PR-exact-GPU-input-CPU-spec-v1'
PARTITIONS = ('originalLOG', 'addedHalo')
WEIGHT_SHAPES = ((91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,))
LOCKED = {'matched_count': 374, 'unmatched_gt_count': 6, 'total_gt_count': 380}
DISTANCE_NAMES = ('D_le_1', 'D_gt_1_le_2', 'D_gt_2_le_4', 'D_gt_4_or_infinite')
DISTANCE_SOURCE = 'native_in_domain_opposite_chessboard_D;empty_full_infinite;fixed_bands1_2_4'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def cpu_tensor(t, value, name, shape, dtype):
    require(isinstance(value, t.Tensor) and value.device.type == 'cpu' and value.dtype == dtype and
            value.layout == t.strided and tuple(value.shape) == tuple(shape), 'CPU tensor descriptor differs: ' + name)
    require(bool(t.isfinite(value).all().item()), 'Nonfinite CPU tensor: ' + name)
    return value


def tensor_digest(t, value):
    require(isinstance(value, t.Tensor) and value.device.type == 'cpu' and value.layout == t.strided,
            'Captured tensor digest requires CPU strided storage')
    descriptor = dict(shape=list(value.shape), dtype=str(value.dtype), byte_order=sys.byteorder)
    digest = hashlib.sha256()
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode('ascii'))
    digest.update(b'\n')
    digest.update(value.detach().contiguous().view(t.uint8).numpy().tobytes(order='C'))
    return dict(descriptor, sha256=digest.hexdigest())


def independent_pr(t, weights, features, native):
    coarse = native.clamp(-8.0, 8.0).unsqueeze(1) / 8.0
    hidden = t.nn.functional.layer_norm(features, (256,), weight=None, bias=None, eps=1e-5)
    for offset in (0, 2, 4):
        hidden = t.nn.functional.relu(t.nn.functional.linear(t.cat((hidden, coarse), dim=1),
                                                            weights[offset], weights[offset + 1]))
    return t.nn.functional.linear(t.cat((hidden, coarse), dim=1), weights[6], weights[7]).squeeze(1)


def check_exact_GPU_point_inputs(t, np, *, candidates, exact_GPU_L, old_LOG, new_halo,
                                 weights, expected_provenance, legacy_support, formula_comparison,
                                 check_deadline):
    require(not t.cuda.is_initialized(), 'Independent CPU leaf must leave CUDA uninitialized')
    require(formula_comparison == {'atol': 1e-4, 'rtol': 1e-4}, 'Original1e-4 formula tolerances must not change')
    require(type(legacy_support) is dict and set(legacy_support) == {'refined_support_atol', 'refined_support_rtol'} and
            all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in legacy_support.values()),
            'ROOT must pass exact values from the already bound legacy numeric_protocol')
    require(callable(check_deadline), 'ROOT must pass the prelocked1800-second deadline check')
    require(isinstance(exact_GPU_L, t.Tensor) and exact_GPU_L.ndim == 2, 'Exact frozen GPU L required')
    shape = tuple(exact_GPU_L.shape)
    cpu_tensor(t, exact_GPU_L, 'exact frozen GPU L', shape, t.float32)
    for name, mask in (('old_LOG', old_LOG), ('new_halo', new_halo)):
        cpu_tensor(t, mask, name, shape, t.bool)
    require(t.equal(old_LOG, exact_GPU_L.abs() <= 2) and not bool((old_LOG & new_halo).any().item()),
            'Only exact frozen GPU L may determine the original domains')
    require(isinstance(weights, (tuple, list)) and len(weights) == 8, 'Ordered frozen eight weights required')
    for index, (value, expected_shape) in enumerate(zip(weights, WEIGHT_SHAPES)):
        cpu_tensor(t, value, 'weight' + str(index), expected_shape, t.float32)
    require(sum(value.numel() for value in weights) == 40497, 'Frozen head scalar count differs')
    frozen_weights = tuple(value.clone() for value in weights)
    expected_candidate_keys = {'status', 'refined_binary_rule', 'point_chunk_size', 'originalLOG', 'addedHalo',
                               'native_support_L_gt_zero', 'merged_C', 'exact_GPU_point_inputs'}
    require(type(candidates) is dict and set(candidates) == expected_candidate_keys and
            candidates['refined_binary_rule'] == 'strict_gt_zero' and candidates['point_chunk_size'] == 2048,
            'Augmented actual GPU candidate schema differs')
    packet = candidates['exact_GPU_point_inputs']
    require(type(packet) is dict and set(packet) == {'schema', 'provenance', 'point_chunk_size', 'refined_binary_rule',
                'capture_contract', 'originalLOG', 'addedHalo'} and packet['schema'] == SCHEMA and
            packet['point_chunk_size'] == 2048 and packet['refined_binary_rule'] == 'strict_gt_zero' and
            packet['provenance'] == expected_provenance, 'Actual captured GPU input provenance/schema differs')
    contract = packet['capture_contract']
    require(contract == dict(actual_GPU_raw_PR_calls_observed=True,
                actual_sampled_inputs_match_original_GPU_F_and_L_exact=True, CPU_copy_has_no_dtype_conversion=True,
                original_PR_arithmetic_called_unchanged=True, original_adapter_binding_restored=True,
                GT_or_selection_or_K_passed_to_PR=False, qualification_self_certified=False), 'Actual capture contract differs')
    Lflat = exact_GPU_L.reshape(-1)
    checks = {}
    first_call = 0
    with t.no_grad():
        for label, domain in zip(PARTITIONS, (old_LOG, new_halo)):
            check_deadline()
            part, captured = candidates[label], packet[label]
            indices = t.from_numpy(np.flatnonzero(domain.numpy()).astype(np.int64, copy=False))
            count = int(indices.numel())
            require(type(captured) is dict and set(captured) == {'flat_indices', 'F_points', 'native_logit_points',
                    'chunk_count', 'chunks', 'tensor_digests'}, 'Captured partition schema differs: ' + label)
            cpu_tensor(t, captured['flat_indices'], label + '.capture.indices', (count,), t.int64)
            cpu_tensor(t, captured['F_points'], label + '.actual_GPU_F_points', (count, 256), t.float32)
            cpu_tensor(t, captured['native_logit_points'], label + '.exact_GPU_L_points', (count,), t.float32)
            require(type(captured['tensor_digests']) is dict and set(captured['tensor_digests']) ==
                    {'flat_indices', 'F_points', 'native_logit_points'} and all(
                    tensor_digest(t, captured[name]) == captured['tensor_digests'][name] for name in captured['tensor_digests']),
                    'Captured CPU FP32/int64 bytes or descriptor changed')
            require(t.equal(captured['flat_indices'], indices) and t.equal(captured['native_logit_points'], Lflat[indices]),
                    'Captured index/L differs from exact frozen GPU sampling')
            require(type(part) is dict and set(part) == {'flat_indices', 'raw_residual', 'refined_logit', 'binary_candidate', 'chunk_count'},
                    'Actual GPU numerical partition schema differs')
            cpu_tensor(t, part['flat_indices'], label + '.GPU.indices', (count,), t.int64)
            cpu_tensor(t, part['raw_residual'], label + '.GPU.u', (count,), t.float32)
            cpu_tensor(t, part['refined_logit'], label + '.GPU.logit', (count,), t.float32)
            cpu_tensor(t, part['binary_candidate'], label + '.GPU.binary', (count,), t.bool)
            chunk_count = (count + 2047) // 2048
            expected_chunks = [dict(actual_call_ordinal=first_call + ordinal, chunk_index=ordinal,
                start_ordinal=start, end_ordinal=min(count, start + 2048))
                for ordinal, start in enumerate(range(0, count, 2048))]
            require(captured['chunks'] == expected_chunks and type(captured['chunk_count']) is int and
                    captured['chunk_count'] == part['chunk_count'] == chunk_count and t.equal(part['flat_indices'], indices) and
                    t.equal(part['refined_logit'], captured['native_logit_points'] + part['raw_residual']) and
                    t.equal(part['binary_candidate'], part['refined_logit'] > 0), 'Actual original GPU chunk/logit/binary consistency differs')
            first_call += chunk_count
            max_u_error = max_logit_error = 0.0
            sign_differences = 0
            per_chunk = []
            for start in range(0, count, 2048):
                check_deadline()
                end = min(count, start + 2048)
                native = captured['native_logit_points'][start:end]
                residual = independent_pr(t, weights, captured['F_points'][start:end], native)
                refined = native + residual
                require(bool(t.isfinite(residual).all().item()) and bool(t.isfinite(refined).all().item()),
                        'Same-input independent CPU formula produced nonfinite values')
                GPU_u, GPU_logit = part['raw_residual'][start:end], part['refined_logit'][start:end]
                u_error, logit_error = (residual - GPU_u).abs(), (refined - GPU_logit).abs()
                u_over = int((u_error > 1e-4 + 1e-4 * GPU_u.abs()).sum().item())
                logit_over = int((logit_error > 1e-4 + 1e-4 * GPU_logit.abs()).sum().item())
                require(u_over == 0 and logit_over == 0 and t.allclose(residual, GPU_u, atol=1e-4, rtol=1e-4) and
                        t.allclose(refined, GPU_logit, atol=1e-4, rtol=1e-4),
                        'Same-input CPU residual/logit exceeds unchanged1e-4 comparison: ' + label)
                changed = (refined > 0) != part['binary_candidate'][start:end]
                tau = legacy_support['refined_support_atol'] + legacy_support['refined_support_rtol'] * refined.abs()
                outside = int((changed & (refined.abs() > tau)).sum().item())
                require(outside == 0, 'Same-input CPU/GPU binary differs outside original practical allowance: ' + label)
                differences = int(changed.sum().item())
                sign_differences += differences
                max_u_error = max(max_u_error, float(u_error.max().item()))
                max_logit_error = max(max_logit_error, float(logit_error.max().item()))
                per_chunk.append(dict(start_ordinal=start, end_ordinal=end, max_raw_residual_abs_error=float(u_error.max().item()),
                    max_refined_logit_abs_error=float(logit_error.max().item()), near_zero_sign_differences=differences,
                    sign_differences_outside_original_practical_allowance=outside))
            checks[label] = dict(points=count, chunk_count=chunk_count, max_raw_residual_abs_error=max_u_error,
                max_refined_logit_abs_error=max_logit_error, near_zero_sign_differences=sign_differences,
                sign_differences_outside_original_practical_allowance=0, chunks=per_chunk, same_actual_GPU_inputs=True,
                formula_comparison={'atol': 1e-4, 'rtol': 1e-4}, certified_error_bound=False)
    A = exact_GPU_L > 0
    cpu_tensor(t, candidates['native_support_L_gt_zero'], 'stored GPU native A', shape, t.bool)
    cpu_tensor(t, candidates['merged_C'], 'stored frozen GPU C', shape, t.bool)
    merged = A.contiguous().clone()
    for label in PARTITIONS:
        merged.reshape(-1)[candidates[label]['flat_indices']] = candidates[label]['binary_candidate']
    require(t.equal(candidates['native_support_L_gt_zero'], A) and t.equal(candidates['merged_C'], merged),
            'Actual frozen GPU C/native support fails exact assembly')
    require(all(t.equal(before, after) for before, after in zip(frozen_weights, weights)) and not t.cuda.is_initialized(),
            'Same-input CPU check changed frozen weights or initialized CUDA')
    return checks


def independent_domains(np, P, L):
    padded = np.pad(P.astype(np.int64), ((2, 2), (2, 2)), mode='constant', constant_values=0)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode='constant').cumsum(0).cumsum(1)
    counts = integral[5:, 5:] - integral[:-5, 5:] - integral[5:, :-5] + integral[:-5, :-5]
    require(counts.shape == P.shape and bool(((counts >= 0) & (counts <= 25)).all()), 'Independent5x5 zero-pad count differs')
    old = np.abs(L) <= 2
    width = (P & (counts <= 22)) | (~P & (counts >= 3))
    halo = width & ~old
    return dict(native_P=P, old_LOG=old, native_width2=width, new_halo=halo, Omega=old | halo)


def independent_distance_bands(np, cv2, ndimage, P):
    if not P.any() or P.all():
        distance = np.full(P.shape, np.inf, dtype=np.float64)
        scipy_distance = distance.copy()
    else:
        distance = np.where(P, cv2.distanceTransform(P.astype(np.uint8), cv2.DIST_C, 3),
                               cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_C, 3)).astype(np.float64)
        scipy_distance = np.where(P, ndimage.distance_transform_cdt(P, metric='chessboard'),
                                  ndimage.distance_transform_cdt(~P, metric='chessboard')).astype(np.float64)
        require(bool((distance >= 1).all()) and np.array_equal(distance, np.floor(distance)), 'Positive integer native distance required')
    require(np.array_equal(distance, scipy_distance), 'Independent OpenCV DIST_C differs from original SciPy distance')
    bands = (distance <= 1, (distance > 1) & (distance <= 2), (distance > 2) & (distance <= 4), distance > 4)
    require(np.array_equal(sum(mask.astype(np.int8) for mask in bands), np.ones(P.shape, dtype=np.int8)), 'Native bands must partition the raster')
    return {name: dict(mask=mask, source=DISTANCE_SOURCE) for name, mask in zip(DISTANCE_NAMES, bands)}


def pixels(np, A, G):
    return dict(pixel_TP=int(np.count_nonzero(A & G)), pixel_FP=int(np.count_nonzero(A & ~G)),
                pixel_FN=int(np.count_nonzero(~A & G)))


def diagnose_exact_input_pair(t, np, cv2, ndimage, statistics, oracle, *, candidate, exact_GPU_L,
                              weights, expected_capture_provenance, legacy_support, formula_comparison,
                              original_LOG_binary, G, locked_pair, metadata, check_deadline):
    """Pure supplied-array path; ROOT must first establish every actual reference.

    Candidate/selector/weights/GT/pair/metadata identities require the full new
    runner's bound SOURCE checks, not caller declarations alone. This function
    verifies numerical/Boolean/statistical values and returns no qualification.
    """
    identity = {key: locked_pair[key] for key in ('image_id', 'gt_id', 'query_id')}
    require(candidate['identity'] == identity and expected_capture_provenance['identity'] == identity and
            candidate['source'] == expected_capture_provenance['source'] and
            candidate['spec'] == expected_capture_provenance['spec'] and
            candidate['GPU_admission'] == expected_capture_provenance['GPU_admission'] and
            candidate['phase'] == expected_capture_provenance['phase'] and
            candidate['native_packet'] == expected_capture_provenance['native_packet'] and
            candidate['selector'] == expected_capture_provenance['selector'] and
            candidate['head_checkpoint'] == expected_capture_provenance['head_checkpoint'], 'Bound captured pair provenance differs')
    A = (exact_GPU_L > 0).numpy().copy()
    domains = independent_domains(np, A, exact_GPU_L.numpy())
    require(set(candidate['domains']) == set(domains), 'Stored GPU domain schema differs')
    for name, domain in domains.items():
        cpu_tensor(t, candidate['domains'][name], 'actual GPU domain.' + name, A.shape, t.bool)
        require(np.array_equal(domain, candidate['domains'][name].numpy()), 'Independent exact-L/integer-width2 domain differs: ' + name)
    checks = check_exact_GPU_point_inputs(t, np, candidates=candidate['candidates'], exact_GPU_L=exact_GPU_L,
        old_LOG=candidate['domains']['old_LOG'], new_halo=candidate['domains']['new_halo'], weights=weights,
        expected_provenance=expected_capture_provenance, legacy_support=legacy_support,
        formula_comparison=formula_comparison, check_deadline=check_deadline)
    C = candidate['candidates']['merged_C'].numpy().copy()  # Always the actual frozen GPU candidate.
    require(isinstance(G, np.ndarray) and G.dtype == np.dtype(np.bool_) and G.shape == A.shape and G.any() and
            isinstance(original_LOG_binary, np.ndarray) and original_LOG_binary.dtype == np.dtype(np.bool_) and
            original_LOG_binary.shape == A.shape, 'Actual explicit GT/original LOG Boolean rasters required')
    reconstructed = A.copy()
    reconstructed[domains['old_LOG']] = C[domains['old_LOG']]
    require(np.array_equal(reconstructed, original_LOG_binary) and pixels(np, A, G) ==
            {key: locked_pair['native'][key] for key in ('pixel_TP', 'pixel_FP', 'pixel_FN')} and
            pixels(np, reconstructed, G) == {key: locked_pair['refined'][key] for key in ('pixel_TP', 'pixel_FP', 'pixel_FN')},
            'Original LOG full binary or locked native/LOG pair pixel counts differ')
    bands = independent_distance_bands(np, cv2, ndimage, A)
    require(candidate['diagnostic']['metadata'] == metadata, 'Actual captured GPU pair metadata differs')
    diagnostic = statistics.diagnose_instance(oracle, A=A, C=C, G=G, old_LOG=domains['old_LOG'],
        new_halo=domains['new_halo'], S=domains['old_LOG'], K=int(np.count_nonzero(domains['old_LOG'])),
        image_id=identity['image_id'], gt_id=identity['gt_id'], query_id=identity['query_id'],
        candidate_independent_of_selection=True, metadata=metadata, distance_bands=bands)
    require(diagnostic == candidate['diagnostic'], 'Same-input CPU pair statistics must reproduce complete actual GPU JSON exactly')
    return dict(identity=identity, diagnostic=diagnostic, numerical_checks=checks,
        exact_GPU_input_digests_checked=True, independent_integer_width2_exact=True,
        independent_OpenCV_chessboard_exact=True, original_LOG_full_binary_exact=True,
        locked_native_LOG_pixel_counts_exact=True, actual_frozen_GPU_C_statistics_exact=True,
        CPU_feature_rerender_used=False, CPU_binary_substituted_into_GPU_C=False,
        qualification_generated=False, independent_oracle_sorting_implementation=False)


def diagnose_complete_population(statistics, oracle, *, checked_pairs, unmatched_gt,
                                 metadata, expected_GPU_population):
    require(isinstance(checked_pairs, (tuple, list)) and len(checked_pairs) == 374 and
            isinstance(unmatched_gt, (tuple, list)) and len(unmatched_gt) == 6, 'Complete locked374+6 population required')
    population = statistics.diagnose_population(oracle,
        matched_diagnostics=[row['diagnostic'] for row in checked_pairs], unmatched_gt=unmatched_gt,
        metadata=metadata, locked_population=LOCKED)
    require(population == expected_GPU_population, 'Complete fixed374+6 population must reproduce actual GPU JSON exactly')
    return population


if __name__ == '__main__':
    raise SystemExit('UNBOUND_DRAFT helper only: ROOT must integrate/review/bind a NEW complete CPU runner; no attempt or Q is launched.')
