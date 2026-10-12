"""Inert ROOT CPU reader for the full passive paired population.

Runtime gates the current owned GPU terminal, all3 trained CP terminals and the
same-CP formal DEV evaluation terminal BEFORE science imports/payload hashes.
No model forward, AP, training, GT-selected routing, RNG or method decision.

SPEC refs: entry_source/CPU_reader/geometry_source/helper_source,
evaluation_source/evaluation_CPU_source/training_source/initializer_source,
micro_preflight_source/native_helpers/adapter_source/hooks_source/DEV,
formal_evaluation_qualification; checkpointmap/qualificationmap for native0,
paired, fg_fine; image_root/expected_epoch/sources/GPU_admission_file,
output_directory/phase/fixed resource-population fields. CLI --spec --out.
Refs are {path:absolute str, bytes:int, sha256:lowercase64hex}.
GPU_admission_file is an absolute string, avoiding SPEC/admission hash cycles.

Own GT means posterior stock-assigned annotation under paired-on final costs,
NOT independently validated semantic identity or detection recall. Other GT is
union of all other annotations; overlaps are retained. Original-coordinate
occupancy and native nearest800 occupancy are different position measures.
No feature-purity/actual-sea-background/causality/method-gain claim.
All100 queries, all16 slots, E0 and all380 raw GT remain in their denominators.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

SCHEMA = 'ROOT-native-geometry-paired-passive-population-spec-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
HELPER_SHA = '54d3b9fb0866fe6e88cea7272b7667fc70997644c2c8732762529968a32f5cb4'
GEOMETRY_SHA = '83517a440eff645c3f032cc8cd68209df43afa6bd7b8f611b3a497eb529352ad'
PRODUCER_BYTES = 36710
PRODUCER_SHA = '0124bdd73c91b9ebcb635ca686d0cf948413eab3012db8971da5173ef4a586da'
MODES = ('native0', 'paired', 'fg_fine')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and Path(row['path']).is_absolute() and
            reference(row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')},
            'Pinned reference differs')
    return Path(row['path'])


def load_source(row, name):
    loader = importlib.util.spec_from_file_location(name, bound(row))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
            type(receipt['returncode']) is int and receipt['returncode'] == 0 and
            receipt['root_wait_reaped'] is True and receipt['held_handle_exit'] ==
            dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            receipt['numeric_birth_absence']['original_birth_absent'] is True and
            receipt['abort'] is None and receipt['error'] is None and receipt['wait_errors'] is False and
            receipt['GPU_observation_errors'] == [], 'STOP before scientific access: owned actual0/held reap required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Owned phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'Owned resource/scope differs')


def digest_array(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and
            row['shape'] == list(shape) and row['dtype'] == dtype and
            isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
            set(row['sha256']) <= set('0123456789abcdef'), 'Tensor digest interface differs')


def array_packet(np, row, shape, dtype):
    require(set(row) == {'shape', 'dtype', 'sha256', 'values'}, 'Actual small tensor packet required')
    digest_contract({key: row[key] for key in ('shape', 'dtype', 'sha256')}, shape, dtype)
    array = np.asarray(row['values'], dtype=np.dtype(dtype))
    require(tuple(array.shape) == tuple(shape) and bool(np.isfinite(array).all()) and
            digest_array(array) == {key: row[key] for key in ('shape', 'dtype', 'sha256')},
            'Independent small-array shape/finite/actual bytes differ')
    return array


def audit_cost(np, torch, scipy_assignment, components, ann_ids, posq, assigned_GT, q_by_GT):
    """Audit exported weighted costs, never recompute the12544 point integrals.

    actual total is the original CPU argument supplied to native SciPy. The
    CPU stack.sum is observed separately, not asserted bit-identical to CUDA.
    Two rounded additions give gamma2*sum|cost|; three tiny terms cover FTZ.
    A float64 gamma2 allowance includes rounding of the reference reduction.
    Hungarian is independently solved on the unchanged actual total matrix.
    """
    G = len(ann_ids)
    require(0 <= G <= 100 and len(set(ann_ids)) == G, 'Raw GT/query capacity differs')
    keys = ('classification_weight2', 'BCE_weight5', 'Dice_weight5', 'total')
    require(set(components) == set(keys), 'Native weighted component roles differ')
    arrays = {key: array_packet(np, components[key], (100, G), 'float32') for key in keys}
    require(len(posq) == len(assigned_GT) == len(q_by_GT) == G and
            all(type(q) is int and 0 <= q < 100 for q in posq + q_by_GT) and
            all(type(g) is int and 0 <= g < G for g in assigned_GT) and
            len(set(posq)) == G and len(set(q_by_GT)) == G and sorted(assigned_GT) == list(range(G)) and
            all(q_by_GT[g] == q for q, g in zip(posq, assigned_GT)), 'Actual assignment/annotation order differs')
    if G == 0:
        return dict(GT=0, matrix_shape=[100, 0], semantic_empty_not_SciPy_observed=True,
                    total_digest=components['total'], original_assignment_objective=0.,
                    independent_assignment_objective=0., cost_reconstruction_integrals_recomputed=False)
    weighted = [arrays[key] for key in keys[:3]]
    reconstructed = torch.stack([torch.from_numpy(a.copy()) for a in weighted]).sum(dim=0).numpy()
    stacked64 = np.stack([a.astype(np.float64) for a in weighted], axis=0)
    reference64 = stacked64.sum(axis=0, dtype=np.float64)
    sumabs = np.abs(stacked64).sum(axis=0, dtype=np.float64)
    u32, u64 = 2. ** -24, 2. ** -53
    gamma32, gamma64 = 2 * u32 / (1 - 2 * u32), 2 * u64 / (1 - 2 * u64)
    error_bound = (gamma32 + gamma64) * sumabs + 3 * float(np.finfo(np.float32).tiny)
    actual_error = np.abs(arrays['total'].astype(np.float64) - reference64)
    CPU_error = np.abs(reconstructed.astype(np.float64) - reference64)
    require(bool((actual_error <= error_bound).all()) and bool((CPU_error <= error_bound).all()),
            'Weighted component sum exceeds the predeclared rounding/FTZ envelope')
    rows, columns = scipy_assignment(arrays['total'])
    require(len(rows) == G and sorted(columns.tolist()) == list(range(G)), 'Independent assignment discarded GT')
    original_values = [float(arrays['total'][q_by_GT[g], g]) for g in range(G)]
    independent_values = [float(arrays['total'][int(q), int(g)]) for q, g in zip(rows, columns)]
    original_objective, independent_objective = math.fsum(original_values), math.fsum(independent_values)
    # fsum is correctly rounded; compare objective values, allowing tie qids.
    objective_bound = u64 / (1 - u64) * (math.fsum(abs(v) for v in original_values) +
                                       math.fsum(abs(v) for v in independent_values))
    require(abs(original_objective - independent_objective) <= objective_bound,
            'Recorded native assignment is not optimal for its actual solver total')
    return dict(GT=G, matrix_shape=[100, G], total_digest=components['total'],
                actual_total_matrix_used_unchanged_for_CPU_assignment=True,
                CPU_stack_sum_bitwise_equal_actual_total=bool(np.array_equal(reconstructed.view(np.uint32),
                                                                            arrays['total'].view(np.uint32))),
                maxabs_actual_minus_reference64=float(actual_error.max()),
                maxabs_CPU_stack_minus_actual=float(np.abs(reconstructed.astype(np.float64) - arrays['total']).max()),
                max_rounding_bound=float(error_bound.max()), all_component_sums_within_rounding_bound=True,
                rounding_rule='gamma2_FP32_plus_gamma2_FP64_times_sumabs_plus_3_FP32_tiny',
                original_assignment_objective=original_objective, independent_assignment_objective=independent_objective,
                objective_bound=objective_bound, ties_allow_different_query_ids=True,
                cost_reconstruction_integrals_recomputed=False, independent_query_semantics_proven=False)


def capture_arrays(np, capture, metadata):
    require(capture['schema'] == 'passive-paired-adapter-capture-v1' and
            capture['calls'] == capture['expected_calls'] == 1 and capture['actual_capture_available'] is True and
            capture['hooks_removed'] is True and capture['forward_hook_returned_None'] is True and
            capture['F_saved'] is capture['full_rho_observed'] is capture['M0_observed'] is
            capture['raw_covariance_observed'] is capture['geometry_recomputed'] is False and
            capture['all_queries_retained'] is capture['GT_position_labels_are_not_feature_purity'] is True and
            capture['metadata'] == metadata, 'Passive adapter scope/metadata differs')
    packet = capture['capture']; source = packet['geometry']
    require(packet['native_query_ids'] == list(range(100)) and packet['F_shape'] == [1, 256, 200, 200] and
            packet['e0_shape'] == [1, 100, 256] and source['mode'] == 'paired' and
            source['eligibility_frame'] == 'canonical' and 'e0' not in packet and 'new_coeff' not in packet,
            'Actual all100 paired geometry-only scope differs')
    shapes = dict(inside=((1, 100, 16, 2), 'float32'), outside=((1, 100, 16, 2), 'float32'),
                  eligible=((1, 100, 16), 'bool'), outside_donor_slots=((1, 100, 16), 'int64'),
                  centre_xy=((1, 100, 2), 'float32'))
    for key in ('rho_mass', 'canonical_theta', 'used_theta', 'sigma_major', 'sigma_minor'):
        shapes[key] = ((1, 100), 'float32')
    for key in ('canonical_eligible_counts', 'eligible_counts'):
        shapes[key] = ((1, 100), 'int64')
    for key in ('near_isotropic', 'degenerate', 'shuffle_can_be_nontrivial'):
        shapes[key] = ((1, 100), 'bool')
    parsed = {key: array_packet(np, source[key], shape, dtype)[0] for key, (shape, dtype) in shapes.items()}
    E = parsed['eligible']; counts = E.sum(axis=-1, dtype=np.int64)
    require(np.array_equal(counts, parsed['eligible_counts']) and np.array_equal(counts, parsed['canonical_eligible_counts']) and
            np.array_equal(counts > 1, parsed['shuffle_can_be_nontrivial']) and
            np.array_equal(parsed['canonical_theta'], parsed['used_theta']) and bool((parsed['rho_mass'] >= 0).all()) and
            bool((parsed['sigma_major'] >= 1).all()) and bool((parsed['sigma_minor'] >= 1).all()) and
            bool(((parsed['outside_donor_slots'] >= 0) & (parsed['outside_donor_slots'] < 16)).all()),
            'Geometry/count/frame consistency differs')
    for key in ('inside', 'outside'):
        require(bool(((parsed[key][E] >= 0) & (parsed[key][E] <= 199)).all()), 'Eligible point outside paired0..199 domain')
    digest_contract(packet['coefficient_residual'], (1, 100, 256), 'float32')
    require(capture['empty_query_residual_exact_zero'] is True and capture['M0_all_nonpositive_supported'] is False and
            capture['outside_donor_slots_applied'] is False, 'Actual E0 assertion/omitted M0/no-donor scope differs')
    has_support = array_packet(np, capture['module_has_support'], (1, 100), 'bool')[0]
    require(source['configuration']['mass_eps'] == 1e-12 and
            np.array_equal(has_support, parsed['rho_mass'] > np.float32(source['configuration']['mass_eps'])),
            'Returned module support/rho-mass consistency differs')
    parsed['module_has_support'] = has_support
    ih, iw = metadata['img_shape'][:2]
    index = np.arange(200, dtype=np.float32)
    yf = np.clip(np.float32(ih) - np.float32(4) * index, 0, 4) / np.float32(4)
    xf = np.clip(np.float32(iw) - np.float32(4) * index, 0, 4) / np.float32(4)
    expected = (yf[:, None] * xf[None, :])[None, None].astype(np.float32)
    digest_contract(packet['actual_valid_fraction'], (1, 1, 200, 200), 'float32')
    require(digest_array(expected) == packet['actual_valid_fraction'], 'Metadata-only F200 cell-area map digest differs')
    return parsed


def point_statistics(values, E):
    require(len(values) == len(E) == 16, 'All16 actual original slots required')
    chosen = [v for v, eligible in zip(values, E) if eligible]
    return dict(all16_mean=math.fsum(values) / 16, all16_sum=math.fsum(values),
                eligible_mean=None if not chosen else math.fsum(chosen) / len(chosen),
                eligible_sum=math.fsum(chosen), eligible_points=len(chosen))


def sample_batch(np, geometry, mask, points, metadata, space):
    """Validate the complete binary mask once, then reuse sealed scalar geometry.

    This is a NumPy validation optimization, not a different occupancy measure.
    Only fields consumed by occupancy_image are returned; point order, four
    neighbour order, math.fsum and zero padding match measure_occupancy.
    """
    meta = geometry._metadata(metadata['img_shape'], metadata['ori_shape'], (800, 800), (200, 200))
    require(space in ('nearest_800', 'original_proxy'), 'Unknown GT occupancy measure')
    shape = meta['batch_hw'] if space == 'nearest_800' else meta['ori_hw']
    array = np.asarray(mask)
    require(array.ndim == 2 and array.shape == shape, 'Complete GT mask shape differs from its measure')
    require(array.dtype.kind in ('b', 'i', 'u', 'f'), 'GT mask must contain numeric Boolean/exact0/1, not text/object/complex')
    require(bool(np.isfinite(array).all()) and bool(((array == 0) | (array == 1)).all()),
            'Every GT mask entry must be finite exact0/1')
    if space == 'nearest_800':
        ih, iw = meta['img_hw']
        require(not bool(np.any(array[ih:, :])) and not bool(np.any(array[:ih, iw:])),
                'nearest_800 GT must be zero in every padded pixel')
    result = []
    for xy in points:
        require(len(xy) == 2, 'Each sample must contain x_feature,y_feature')
        mapped = geometry.map_feature_point(xy[0], xy[1], metadata['img_shape'], metadata['ori_shape'])
        feature_stencil = geometry.bilinear_stencil(*mapped['feature_xy'], shape=meta['feature_hw'])
        canvas_stencil = geometry.bilinear_stencil(*mapped['canvas800_xy'], shape=meta['batch_hw'])
        GT_stencil = canvas_stencil if space == 'nearest_800' else geometry.bilinear_stencil(
            *mapped['original_xy'], shape=meta['ori_hw'])
        result.append(dict(
            GT_bilinear_occupancy=math.fsum(item['weight'] * float(array[item['row'], item['column']])
                                           for item in GT_stencil if item['in_bounds']),
            direct800_domain=geometry._canvas_fractions(canvas_stencil, meta['img_hw']),
            sampled_feature_cell_valid_area_fraction=geometry._metadata_fraction(feature_stencil, meta['img_hw'])))
    return result


def occupancy_image(np, geometry, raw_masks, native_masks, ann_ids, q_by_GT, parsed, metadata, image_id):
    """Each mask validation runs once per relevant batch, not once per point.

    Union maps: per-pixel annotation count, then count>0; other union excludes
    own Boolean BEFORE threshold/interpolation, preserving real overlap.
    One GT owns at most one stock-matched query. Unmatched own is undefined.
    Slot order is sixteen inside, then sixteen outside, per query. Do NOT use
    the paired diagnostic outside_donor_slots: it is unused weak precomputation.
    """
    G = len(ann_ids)
    ih, iw = metadata['img_shape'][:2]; oh, ow = metadata['ori_shape'][:2]
    require(native_masks.shape == (G, 800, 800) and raw_masks.shape == (G, oh, ow), 'Complete GT mask shapes differ')
    query_to_gt = {q: g for g, q in enumerate(q_by_GT)}
    per_query_points = [parsed['inside'][q].tolist() + parsed['outside'][q].tolist() for q in range(100)]
    all_points = [xy for points in per_query_points for xy in points]
    counts_native = native_masks.sum(axis=0, dtype=np.int64)
    counts_raw = raw_masks.sum(axis=0, dtype=np.int64)
    union = {}
    for space, counts in (('nearest_800', counts_native), ('original_proxy', counts_raw)):
        union[space] = sample_batch(np, geometry, counts > 0, all_points, metadata, space)
    own_other = {}
    for g, q in enumerate(q_by_GT):
        own_other[q] = {}
        for space, masks, counts in (('nearest_800', native_masks, counts_native), ('original_proxy', raw_masks, counts_raw)):
            own = sample_batch(np, geometry, masks[g], per_query_points[q], metadata, space)
            other = sample_batch(np, geometry, (counts - masks[g].astype(np.int64)) > 0, per_query_points[q], metadata, space)
            own_other[q][space] = (own, other)
    rows = []
    for q in range(100):
        E = parsed['eligible'][q].tolist(); g = query_to_gt.get(q)
        row = dict(image_id=image_id, query_id=q, stock_assigned_annotation_id=None if g is None else ann_ids[g],
                   unmatched_query=g is None, eligible_pairs=int(sum(E)), E0=not any(E), all16_slots_preserved=True,
                   rho_mass=float(parsed['rho_mass'][q]), theta=float(parsed['used_theta'][q]),
                   module_has_support=bool(parsed['module_has_support'][q]),
                   degenerate=bool(parsed['degenerate'][q]), near_isotropic=bool(parsed['near_isotropic'][q]),
                   outside_donor_slots_not_applied=True, own_GT_is_posterior_stock_assignment=True, measures={})
        for space in ('nearest_800', 'original_proxy'):
            allGT = union[space][q * 32:(q + 1) * 32]
            own, other = (None, allGT) if g is None else own_other[q][space]
            measure = {}
            for label, point_rows in (('own_GT', own), ('other_GT_union', other), ('all_GT_union', allGT)):
                measure[label] = None if point_rows is None else {
                    side: point_statistics([p['GT_bilinear_occupancy'] for p in point_rows[offset:offset + 16]], E)
                    for side, offset in (('inside', 0), ('outside', 16))}
            row['measures'][space] = measure
        # Domain fractions are mask-independent, read once from the800 union proxy.
        canvas_points = union['nearest_800'][q * 32:(q + 1) * 32]
        row['domain'] = {}
        for side, offset in (('inside', 0), ('outside', 16)):
            part = canvas_points[offset:offset + 16]
            row['domain'][side] = {key: point_statistics([p['direct800_domain'][key] for p in part], E)
                for key in ('valid_image_pixel_support_fraction', 'right_bottom_padding_pixel_support_fraction',
                            'outside_canvas_zero_padding_fraction')}
            row['domain'][side]['sampled_feature_cell_valid_area_fraction'] = point_statistics(
                [p['sampled_feature_cell_valid_area_fraction'] for p in part], E)
        rows.append(row)
    require(len(rows) == 100 and sum(not r['unmatched_query'] for r in rows) == G, 'Query/raw GT population changed')
    return rows


def aggregate_occupancy(rows):
    require(len(rows) == 18700 and len({(r['image_id'], r['query_id']) for r in rows}) == 18700 and
            sum(not r['unmatched_query'] for r in rows) == 380, 'Full18700 query/380 assigned raw GT denominators required')
    summary = dict(images=187, queries=18700, raw_GT=380, matched_queries=380, unmatched_queries=18320,
                   E0_queries=sum(r['E0'] for r in rows), E0_matched_GT=sum(r['E0'] and not r['unmatched_query'] for r in rows),
                   module_support_false_queries=sum(not r['module_has_support'] for r in rows),
                   module_support_false_matched_GT=sum(not r['module_has_support'] and not r['unmatched_query'] for r in rows),
                   eligible_pairs=sum(r['eligible_pairs'] for r in rows), query_filter=None, score_filter=None,
                   IoU_filter=None, GT_empty_filter=None, measures={})
    for space in ('nearest_800', 'original_proxy'):
        summary['measures'][space] = {}
        for label in ('own_GT', 'other_GT_union', 'all_GT_union'):
            population = [r for r in rows if r['measures'][space][label] is not None]
            expected = 380 if label == 'own_GT' else 18700
            require(len(population) == expected, 'Occupancy population denominator differs')
            result = dict(all_slot_query_denominator=expected, E0_retained_count=sum(r['E0'] for r in population), sides={})
            for side in ('inside', 'outside'):
                stats = [r['measures'][space][label][side] for r in population]
                points = sum(s['eligible_points'] for s in stats)
                result['sides'][side] = dict(all16_mean_over_full_population=math.fsum(s['all16_mean'] for s in stats) / expected,
                    eligible_point_denominator=points, eligible_pair_weighted_mean=None if points == 0 else
                    math.fsum(s['eligible_sum'] for s in stats) / points,
                    eligible_mean_available_queries=sum(s['eligible_mean'] is not None for s in stats),
                    eligible_mean_missing_queries=sum(s['eligible_mean'] is None for s in stats))
            summary['measures'][space][label] = result
    return summary


def prior_terminal_gates(spec):
    """Raw-read only lineage needed to gate ALL terminals before any payload hash."""
    for mode in MODES:
        q = read(spec['qualificationmap'][mode]['path'])
        require(q['schema'] == 'ROOT-native-geometry-paired-common-schedule-CPU-qualification-v1' and
                q['status'] == 'ROOT_NATIVE_GEOMETRY_PAIRED_COMMON_SEGMENT_CPU_QUALIFIED' and q['mode'] == mode,
                'Actual mode-specific training qualification metadata required')
        terminal(read(q['parent_receipt']['path']), q['phase'])
    q = read(spec['formal_evaluation_qualification']['path'])
    require(q['schema'] == 'ROOT-native-geometry-paired-common-evaluation-CPU-qualification-v1' and
            q['status'] == 'ROOT_NATIVE_GEOMETRY_PAIRED_COMMON_DEV_CPU_EVALUATED',
            'Actual formal evaluation CPU qualification metadata required')
    terminal(read(q['parent_receipt']['path']), q['phase'])


def source_gate(spec_path, spec, receipt):
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('native_geometry_paired_passive_population_') and
            spec['modes'] == list(MODES) and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(MODES) and
            type(spec['expected_epoch']) is int and spec['expected_epoch'] in range(12, 25, 2) and
            spec['evaluation_images'] == 187 and spec['raw_GT'] == 380 and spec['CPU_affinity'] == [0, 1] and
            spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
            spec['deadline_seconds'] == 900 and spec['automatic_retry'] is False,
            'Locked full population/even epoch/resource contract differs')
    require(spec['head_seed_base'] == 100000 and spec['matching_seed_base'] == 1000000 and
            spec['matching_num_points'] == 12544 and spec['occupancy_definitions'] ==
            ['direct_nearest800_point', 'original_position_proxy'], 'Sampling seeds/native matching/measure definitions differ')
    require(isinstance(spec['GPU_admission_file'], str) and Path(spec['GPU_admission_file']).is_absolute(),
            'Absolute admission path required')
    adm_ref = reference(spec['GPU_admission_file']); adm = read(spec['GPU_admission_file'])
    entry = reference(Path(__file__).with_name('producer.py')); reader = reference(__file__); spec_ref = reference(spec_path)
    require(entry['bytes'] == PRODUCER_BYTES and entry['sha256'] == PRODUCER_SHA, 'Reviewed producer source differs')
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(receipt['admission_sha256'] == adm_ref['sha256'] and owner['sha256'] == OWNER_SHA and
            adm['driver_sha256'] == OWNER_SHA and adm['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            adm['automatic_retry'] is False and adm['phase'] == spec['phase'] and adm['allowed_cpus'] == [0, 1] and
            adm['cuda_visible_devices'] == '0' and adm['qualified_leaf_deadline_seconds'] == 900 and
            adm['whole_GPU_used_limit_MiB'] == 8192 and Path(adm['entry']).resolve(strict=True) == Path(entry['path']) and
            adm['entry_args'] == ['--spec', str(spec_path)] and spec['entry_source'] == entry and spec['CPU_reader'] == reader,
            'Actual receipt/admission/producer/reader/SPEC linkage differs')
    require(adm['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC is not admission-bound')
    for row in spec['sources']:
        bound(row)
        require(adm['bound_source_sha256'].get(row['path']) == row['sha256'], 'Current source not admission-bound')
    for key in ('entry_source', 'CPU_reader', 'helper_source', 'geometry_source', 'evaluation_source', 'evaluation_CPU_source',
                'training_source', 'initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source', 'hooks_source'):
        require(spec[key] in spec['sources'], 'Executing source role missing: ' + key)
    require(spec['helper_source']['bytes'] == 16788 and spec['helper_source']['sha256'] == HELPER_SHA and
            spec['geometry_source']['bytes'] == 10315 and spec['geometry_source']['sha256'] == GEOMETRY_SHA and
            spec['evaluation_source']['bytes'] == 31808 and
            spec['evaluation_source']['sha256'] == '87342c38452b717a4c9c7f817580645d8247ac59de85d77479520f36e3f69d23' and
            spec['evaluation_CPU_source']['bytes'] == 40576 and
            spec['evaluation_CPU_source']['sha256'] == '66290f823afaec477f700e2c16f98aafb5f81f5924d8c921273c4e8348f492b4',
            'Known unchanged helper/geometry/formal evaluation source differs')
    if 'scientific_spec' in adm:
        require(adm['scientific_spec'] == spec_ref, 'Admission scientific SPEC differs')
    return entry, reader, spec_ref, adm_ref, owner


def formal_evaluation_gate(spec, evaluator_CPU):
    q = read(bound(spec['formal_evaluation_qualification']))
    formal_spec = read(bound(q['spec'])); receipt = read(bound(q['parent_receipt']))
    terminal(receipt, q['phase'])
    entry, reader, spec_ref, admission, owner = evaluator_CPU.admission_gate(Path(q['spec']['path']), formal_spec, receipt)
    require(q['source'] == entry == spec['evaluation_source'] and q['reader_source'] == reader == spec['evaluation_CPU_source'] and
            q['spec'] == spec_ref and q['GPU_admission'] == admission and q['owner_driver'] == owner and
            q['expected_epoch'] == formal_spec['expected_epoch'] == spec['expected_epoch'] and
            q['checkpointmap'] == formal_spec['checkpointmap'] == spec['checkpointmap'] and
            q['qualificationmap'] == formal_spec['qualificationmap'] == spec['qualificationmap'] and
            formal_spec['DEV'] == spec['DEV'] and formal_spec['image_root'] == spec['image_root'] and
            q['full_native_state_CPU_checkpoint_verified'] is q['all187_images_all380_original_GT_preserved'] is
            q['all100_predictions_per_image_preserved'] is q['no_added_NMS_or_score_cutoff'] is True,
            'Formal qualified same-epoch/same-CP/full DEV population/source linkage differs')
    report = read(bound(q['report'])); bound(q['start'])
    require(report['source'] == entry and report['spec'] == spec_ref and report['phase'] == q['phase'] and
            report['checkpointmap'] == spec['checkpointmap'] and report['qualificationmap'] == spec['qualificationmap'] and
            report['error'] is None, 'Formal accepted evaluation report linkage differs')
    record = report['mode_reports']['paired']; result = q['mode_results']['paired']
    require(record['checkpoint'] == result['checkpoint'] == spec['checkpointmap']['paired'] and
            record['qualification'] == result['qualification'] == spec['qualificationmap']['paired'] and
            record['predictions'] == result['predictions'] and record['image_rows'] == result['image_rows'] and
            record['query_score_factors'] == result['score_factor_proof']['source'] and
            record['state_proof'] == result['state_proof']['source'] and record['CP_qualified'] is True and
            record['images'] == 187 and record['raw_GT'] == 380 and record['predictions_count'] == 18700,
            'Formal paired actual endpoint references/population differ')
    return q, record


def verify_RNG_summary(row):
    require(isinstance(row, dict) and set(row) == {'CPU', 'CUDA'}, 'CPU/CUDA0 RNG summaries required')
    for value in row.values():
        require(isinstance(value['shape'], list) and len(value['shape']) == 1 and
                type(value['shape'][0]) is int and value['shape'][0] > 0, 'RNG byte vector shape differs')
        digest_contract(value, value['shape'], 'uint8')


def reconstruct_GT(np, mmcv, coco, image, ann_ids, metadata, GT_digest, label_digest):
    """Use native BitmapMasks rescale delegates, not an independent resize guess."""
    anns = coco.imgToAnns[image['id']]
    require([a['id'] for a in anns] == ann_ids, 'All original annotation order must equal native loader order')
    G = len(anns); oh, ow = image['height'], image['width']; ih, iw = metadata['img_shape'][:2]
    raw = []
    for ann in anns:
        raster = coco.annToMask(ann)
        require(raster.shape == (oh, ow) and bool(((raster == 0) | (raster == 1)).all()), 'Original binary GT raster differs')
        raw.append(raster.astype(np.uint8))
    raw_masks = np.stack(raw) if G else np.empty((0, oh, ow), dtype=np.uint8)
    native = []
    for raster in raw_masks:
        resized = mmcv.imrescale(raster, (800, 800), interpolation='nearest')
        require(resized.shape == (ih, iw) and bool(((resized == 0) | (resized == 1)).all()),
                'Official nearest BitmapMasks rescale differs from actual metadata')
        native.append(mmcv.impad(resized, shape=(800, 800), pad_val=0))
    native_masks = np.stack(native) if G else np.empty((0, 800, 800), dtype=np.uint8)
    require(not bool(native_masks[:, ih:, :].any()) and not bool(native_masks[:, :, iw:].any()), 'Native GT pad not zero')
    digest_contract(GT_digest, (G, 800, 800), 'int64'); digest_contract(label_digest, (G,), 'int64')
    require(digest_array(native_masks.astype(np.int64)) == GT_digest and
            digest_array(np.zeros(G, dtype=np.int64)) == label_digest,
            'Independent all-GT native nearest800/labels digests disagree with actual posterior GT')
    return raw_masks, native_masks


def state_gate(proof, report, qualified, spec):
    require(proof['schema'] == 'native-geometry-paired-passive-population-state-RNG-proof-v1' and
            proof['before'] == proof['after'] and proof['full612_state_CPU_CUDA_RNG_flags_hooks_byte_exact'] is True and
            proof['CUDA_RNG_observed'] is True and proof['unobserved_all10_population_identity_claimed'] is False and
            proof['formal_final_and_official_all100_regression_exact'] is True, 'Population frozen/RNG proof differs')
    names = set(qualified['fullstate']) - set(qualified['BN_buffer_names'])
    require(len(names) == 453 and len(qualified['fullstate']) == 612, 'Independent paired native/parameter boundary differs')
    for snapshot in (proof['before'], proof['after']):
        require(snapshot['schema'] == 'passive-replay-execution-snapshot-v1' and snapshot['cuda_device'] == 0 and
                snapshot['state'] == qualified['fullstate'] and
                set(snapshot['parameter_requires_grad']) == set(snapshot['parameter_grad_is_None']) == names and
                all(flag is False for flag in snapshot['parameter_requires_grad'].values()) and
                all(flag is True for flag in snapshot['parameter_grad_is_None'].values()) and
                snapshot['module_training'] and all(flag is False for flag in snapshot['module_training'].values()),
                'Actual612 frozen state/parameter/module summaries differ from independent CPU CP')
        verify_RNG_summary(snapshot['RNG'])
    loading = proof['loading_proof']
    require(loading['checkpoint'] == spec['checkpointmap']['paired'] and loading['qualification'] == spec['qualificationmap']['paired'] and
            loading['trained_fullstate'] == qualified['fullstate'] and loading['initial_native_state'] == qualified['initial_native_state'] and
            loading['native_state_tensors'] == 610 and loading['adapter_state_tensors'] == 2 and
            loading['registered_parameters'] == 453 and loading['strict_fullstate_loaded'] is loading['CP_qualified'] is True,
            'Actual loaded checkpoint proof differs')
    require(report['cleanup_allocated_bytes'] == 0 and report['error'] is None and
            type(report['torch_peak_allocated_bytes']) is int and 0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2,
            'Actual producer cleanup/allocator failed')


def official_regression(np, mask_utils, rows, factors, images, image_rows, prior_rows, prior_factors):
    """Decode current RLEs and check exact same-CP per-qid factors, no AP."""
    require(len(rows) == len(factors) == len(prior_rows) == len(prior_factors) == 18700, 'All18700 official endpoints required')
    old_pred = {(p['image_id'], p['query_id']): p for p in prior_rows}
    old_factor = {(p['image_id'], p['query_id']): p for p in prior_factors}
    require(len(old_pred) == len(old_factor) == 18700, 'Formal per-query duplicates')
    seen, query_order, empty = set(), {}, 0
    ids = [image['id'] for image in images]
    for i, (p, factor) in enumerate(zip(rows, factors)):
        require(set(p) == {'image_id', 'query_id', 'category_id', 'score', 'segmentation'} and
                type(p['query_id']) is int and 0 <= p['query_id'] < 100 and p['category_id'] == 1 and
                p['image_id'] == ids[i // 100] and set(factor) ==
                {'image_id', 'query_id', 'class_probability', 'mask_quality', 'official_score'} and
                (factor['image_id'], factor['query_id']) == (p['image_id'], p['query_id']), 'Native image/qid/factor layout differs')
        key = (p['image_id'], p['query_id']); require(key not in seen, 'Duplicate official query')
        seen.add(key); query_order.setdefault(p['image_id'], []).append(p['query_id'])
        image = images[i // 100]; rle = p['segmentation']
        require(set(rle) == {'size', 'counts'} and rle['size'] == [image['height'], image['width']] and
                isinstance(rle['counts'], str), 'Actual original-size RLE required')
        encoded = dict(size=rle['size'], counts=rle['counts'].encode('ascii'))
        binary = mask_utils.decode(encoded)
        require(binary.shape == (image['height'], image['width']) and binary.dtype == np.uint8 and
                bool(((binary == 0) | (binary == 1)).all()) and int(mask_utils.area(encoded)) == int(binary.sum(dtype=np.int64)),
                'Independent official RLE binary/area validation failed')
        empty += int(not bool(binary.any()))
        original, previous_factor = old_pred[key], old_factor[key]
        require(set(original) == set(p) and original['segmentation'] == rle and original['category_id'] == 1 and
                set(previous_factor) == set(factor), 'Formal per-qid RLE/schema differs')
        for name in ('class_probability', 'mask_quality', 'official_score'):
            value, previous = factor[name], previous_factor[name]
            require(type(value) in (int, float) and type(previous) in (int, float) and math.isfinite(value) and
                    math.isfinite(previous) and 0 <= value <= 1 and float(np.float32(value)) == value and
                    float(np.float32(previous)) == previous and
                    np.asarray(value, dtype=np.float32).tobytes() == np.asarray(previous, dtype=np.float32).tobytes(),
                    'Factor is not exact bounded FP32 or differs from same-CP formal endpoint')
        score = factor['official_score']
        require(type(p['score']) in (int, float) and float(np.float32(p['score'])) == p['score'] and
                np.asarray(p['score'], dtype=np.float32).tobytes() == np.asarray(score, dtype=np.float32).tobytes() ==
                np.asarray(original['score'], dtype=np.float32).tobytes(), 'Actual/formal official FP32 score differs')
        product = factor['class_probability'] * factor['mask_quality']
        bound_error = 2. ** -24 / (1 - 2. ** -24) * abs(product) + 2 * float(np.finfo(np.float32).tiny)
        require(abs(score - product) <= bound_error, 'Official product outside FP32 rounding/FTZ range')
        del binary
    require(seen == set(old_pred) == set(old_factor) == {(i, q) for i in ids for q in range(100)} and
            all(query_order[row['image_id']] == row['prediction_query_ids'] for row in image_rows),
            'Original all100 query ID population/order changed')
    return dict(entries=18700, empty_predictions_retained=empty, exact_same_CP_RLE_factor_score_regression=True,
                independent_CPU_RLE_decode_and_factor_arithmetic=True, omitted_logits_quality_independently_recomputed=False)


def image_gate(row, cost, capture_row, image, index, spec, prior_row, execution):
    ann_ids = row['ann_ids']; identity = row['identity']; oh, ow = image['height'], image['width']
    scale = 800. / max(oh, ow); actual = [int(oh * scale + .5), int(ow * scale + .5)]
    require(row['image_id'] == identity['image_id'] == image['id'] and row['dataset_index'] == identity['dataset_index'] == index and
            row['annotation_count'] == len(ann_ids) and identity['ann_ids'] == ann_ids and
            identity['original_shape'] == [oh, ow] and identity['actual_img_shape'] == actual and
            identity['batch_input_shape'] == [800, 800] and identity['scale_factor'] == [actual[1] / ow, actual[0] / oh] and
            Path(identity['image']['path']).resolve() == (Path(spec['image_root']) / image['file_name']).resolve() and
            row['checkpoint'] == spec['checkpointmap']['paired'] and row['mode'] == 'paired' and row['apply'] is True,
            'Actual complete image/annotation/native geometry/CP binding differs')
    metadata = row['metadata']
    require(metadata['img_shape'][:2] == actual and metadata['ori_shape'][:2] == [oh, ow] and
            metadata['batch_input_shape'] == [800, 800] and metadata['scale_factor'] == identity['scale_factor'], 'Capture metadata differs')
    require(prior_row['image'] == identity['image'] and prior_row['ann_ids'] == ann_ids and
            prior_row['original_shape'] == [oh, ow] and prior_row['actual_img_shape'] == actual and
            prior_row['batch_input_shape'] == [800, 800] and prior_row['scale_factor'] == identity['scale_factor'] and
            prior_row['dataset_index'] == index and prior_row['final_fields'] == row['final_fields'] and
            prior_row['mode'] == prior_row['checkpoint_mode'] == 'paired', 'Exact same-CP formal input/final fields regression differs')
    require(row['all10_is_observed_digest_only'] is True and set(row['all10_fields']) == {'classes', 'masks'} and
            len(row['all10_fields']['classes']) == len(row['all10_fields']['masks']) == 10, 'Actual ten readout digests required')
    for field, shape in (('classes', (1, 100, 2)), ('masks', (1, 100, 200, 200))):
        for packet in row['all10_fields'][field]:
            digest_contract(packet, shape, 'float32')
        require(row['all10_fields'][field][-1] == row['final_fields'][field], 'Final digest not joined to actual last readout')
    require(row['formal_regression'] == dict(final_dense_digest_exact=True, official_all100_RLE_and_FP32_factors_exact=True,
            joined_by_image_and_query_ID=True, unobserved_all10_identity_claimed=False), 'Formal regression scope differs')
    require(row['head_seed'] == 100000 + image['id'] and row['matching_seed'] == cost['seed'] == 1000000 + image['id'] and
            row['external_RNG_before'] == row['external_RNG_after'] == cost['external_RNG_before'] == cost['external_RNG_after'] ==
            execution['before']['RNG'] and row['external_RNG_restored'] is cost['external_RNG_restored'] is True,
            'Private seed/external fork restoration summaries differ')
    for private in (row['head_private_RNG'], cost['private_RNG']):
        require(set(private) == {'before', 'after'}, 'Private RNG summary scope differs')
        for snapshot in private.values():
            verify_RNG_summary(snapshot)
    require(row['hook_signature_before'] == row['hook_signature_after'] and
            isinstance(row['hook_signature_before'], str) and len(row['hook_signature_before']) == 64 and
            set(row['hook_signature_before']) <= set('0123456789abcdef'), 'Per-image hook summaries differ')
    for key in ('native_passive_matching_observers_removed', 'no_GT_in_forward', 'GT_posthoc_matching', 'all_raw_GT_retained'):
        require(row[key] is True, 'Actual forward/posterior scope differs: ' + key)
    require(row['feature_extraction_calls'] == row['head_forward_calls'] == row['stock_final_matching_calls'] == 1 and
            row['loss_executed'] is row['optimizer_executed'] is row['query_filter'] is False and row['predictions'] == 100,
            'Actual native call counts/query population differ')
    hook = row['paired_hook']
    for key in ('native_class_path_untouched', 'earlier_gate_paths_untouched', 'all_queries_retained',
                'cache_cleared_after_forward', 'hooks_removed'):
        require(hook[key] is True, 'Native adapter hook scope differs: ' + key)
    require(hook['apply'] is True and hook['mode'] == 'paired' and hook['paired_source'] == spec['adapter_source'] and
            hook['forward_calls'] == hook['completed_forwards'] == hook['pixel_decoder_calls'] == 1 and
            hook['coefficient_calls'] == 10 and hook['unchanged_early_readouts'] == 9 and hook['final_coefficient_replacements'] == 1,
            'Actual one final coefficient replacement differs')
    require(capture_row['image_id'] == cost['image_id'] == row['image_id'] and capture_row['dataset_index'] == index and
            capture_row['ann_ids'] == cost['ann_ids'] == ann_ids and cost['GT800'] == row['GT800'] and cost['labels'] == row['labels'],
            'Capture/cost/native GT digest annotation joins differ')
    require(cost['classification_already_weight2'] is cost['BCE_already_weight5'] is cost['Dice_already_weight5'] is True and
            cost['native_cost_weights'] == [2., 5., 5.] and cost['Dice_eps'] == 1. and cost['num_points'] == 12544 and
            cost['get_targets_single_calls'] == 1 and cost['cost_calls'] == (3 if ann_ids else 0) and
            cost['SciPy_calls'] == (1 if ann_ids else 0) and cost['total_is_actual_SciPy_CPU_argument'] is bool(ann_ids) and
            cost['semantic_empty_not_SciPy_observed'] is (not bool(ann_ids)) and cost['all_raw_GT_retained'] is True and
            cost['cost_integrals_independently_recomputed'] is cost['loss_executed'] is False and
            cost['GT_order'] == 'raw annotation_mapping/loader order', 'Stock matching weighted/empty semantics differ')
    expected_ann = [-1] * 100
    for g, q in enumerate(cost['q_by_GT']):
        require(type(q) is int and 0 <= q < 100, 'Invalid stock matched qid')
        expected_ann[q] = ann_ids[g]
    require(cost['ann_by_query'] == expected_ann, 'Stock native query/annotation join differs')
    return metadata


def qualify(spec_file, out_file):
    require(Path(spec_file).is_absolute() and Path(out_file).is_absolute() and not Path(out_file).exists(),
            'Absolute SPEC/fresh output required')
    spec_path = Path(spec_file).resolve(strict=True); spec = read(spec_path)
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    receipt = read(receipt_path); terminal(receipt, spec['phase'])
    prior_terminal_gates(spec)
    # All current/prior actual0 terminals passed BEFORE scientific hashes/imports.
    entry, reader, spec_ref, admission, owner = source_gate(spec_path, spec, receipt)
    evaluator = load_source(spec['evaluation_source'], 'population_CPU_qualified_evaluator')
    evaluator.qualified_training(spec)  # complete source/admission/CP linkage of all3
    evaluator_CPU = load_source(spec['evaluation_CPU_source'], 'population_CPU_qualified_state_reader')
    qualifications = evaluator_CPU.qualified_training(spec)
    formal_q, formal_record = formal_evaluation_gate(spec, evaluator_CPU)
    report_path = Path(spec['output_directory']) / 'REPORT.json'; report = read(report_path)
    require(report['schema'] == 'native-geometry-paired-passive-population-report-v1' and
            report['status'] == 'COMPLETE_PASSIVE_PAIRED_POPULATION_DIAGNOSTIC_PENDING_ROOT_TERMINAL' and
            report['phase'] == spec['phase'] and report['source'] == entry and report['CPU_reader'] == reader and
            report['spec'] == spec_ref and report['GPU_admission'] == admission and report['owner_driver'] == owner and
            report['expected_epoch'] == spec['expected_epoch'] and report['checkpointmap'] == spec['checkpointmap'] and
            report['qualificationmap'] == spec['qualificationmap'] and report['mode'] == 'paired' and
            report['formal_evaluation_qualification'] == spec['formal_evaluation_qualification'] and
            report['formal_evaluation_report'] == formal_q['report'] and
            report['formal_image_rows'] == formal_record['image_rows'] and report['formal_predictions'] == formal_record['predictions'] and
            report['formal_factors'] == formal_record['query_score_factors'] and
            report['images'] == report['feature_extraction_calls'] == report['head_forward_calls'] == report['stock_final_matching_calls'] == 187 and
            report['raw_GT'] == 380 and report['predictions_count'] == report['query_score_factors_count'] == 18700 and
            type(report['runtime_seconds']) in (int, float) and math.isfinite(report['runtime_seconds']) and
            0 < report['runtime_seconds'] <= 900 and report['error'] is None, 'Actual terminal/report/population/source linkage differs')
    for key in ('no_GT_in_forward', 'GT_used_only_in_loader_and_posterior_matching', 'no_optimizer', 'no_loss', 'no_AP',
                'all_raw_GT_retained', 'all10_is_observed_digest_only', 'formal_final_and_official_all100_regression_exact',
                'full_rho_M0_unknown', 'GT_position_is_not_feature_purity', 'actual_total_is_native_SciPy_argument',
                'empty_GT_total_semantic_only'):
        require(report[key] is True, 'Actual diagnostic scope differs: ' + key)
    for key in ('query_filter', 'outside_donor_slots_applied', 'unobserved_all10_population_identity_claimed',
                'M0_all_nonpositive_supported', 'occupancy_computed', 'component_sum_CPU_GPU_bitidentity_claimed',
                'cost_integrals_independently_recomputed'):
        require(report[key] is False, 'Actual omitted/false scope differs: ' + key)
    require(report['occupancy_definitions'] == spec['occupancy_definitions'], 'Locked occupancy measures differ')
    names = dict(image_rows='IMAGE_ROWS.json', capture_rows='CAPTURE_ROWS.json', native_cost_rows='NATIVE_COST_ROWS.json',
                 official_predictions='ALL_PREDICTIONS.json', query_score_factors='QUERY_SCORE_FACTORS.json',
                 state_RNG_proof='STATE_RNG_PROOF.json')
    require(set(report['outputs']) == set(names) and all(Path(report['outputs'][key]['path']).resolve(strict=True) ==
            (Path(spec['output_directory']) / name).resolve(strict=True) for key, name in names.items()),
            'Current evidence paths/roles differ')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import torch
    import mmcv
    from scipy.optimize import linear_sum_assignment
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/no CUDA initialization required')
    geometry = load_source(spec['geometry_source'], 'population_CPU_pinned_geometry')
    qualified = evaluator_CPU.checkpoint_state(torch, spec, qualifications['paired'], 'paired')
    outputs = {key: read(bound(row)) for key, row in report['outputs'].items()}
    execution = outputs['state_RNG_proof']; state_gate(execution, report, qualified, spec)
    coco = COCO(str(bound(spec['DEV']))); images = list(coco.imgs.values())
    require(len(images) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1} and
            len({image['id'] for image in images}) == 187, 'Complete original DEV187/380/category1 required')
    image_rows, captures, cost_rows = (outputs[key] for key in ('image_rows', 'capture_rows', 'native_cost_rows'))
    require(len(image_rows) == len(captures) == len(cost_rows) == 187 and
            [row['image_id'] for row in image_rows] == [image['id'] for image in images] and
            sum(row['annotation_count'] for row in image_rows) == 380, 'Full ordered image/capture/cost/allGT population required')
    prior_image_rows = read(bound(formal_record['image_rows']))
    require(len(prior_image_rows) == 187 and [row['image_id'] for row in prior_image_rows] == [image['id'] for image in images],
            'Formal full ordered image population differs')
    official = official_regression(np, mask_utils, outputs['official_predictions'], outputs['query_score_factors'], images,
                                  image_rows, read(bound(formal_record['predictions'])),
                                  read(bound(formal_record['query_score_factors'])))
    occupancy_rows, cost_proofs, GT_rows = [], [], []
    for index, (image, row, captured, cost, prior) in enumerate(zip(images, image_rows, captures, cost_rows, prior_image_rows)):
        metadata = image_gate(row, cost, captured, image, index, spec, prior, execution)
        parsed = capture_arrays(np, captured['capture'], metadata)
        cost_proof = audit_cost(np, torch, linear_sum_assignment, cost['components'], row['ann_ids'],
                               cost['positive_query_indices'], cost['assigned_GT_indices'], cost['q_by_GT'])
        cost_proofs.append(dict(image_id=image['id'], **cost_proof))
        raw, native = reconstruct_GT(np, mmcv, coco, image, row['ann_ids'], metadata, row['GT800'], row['labels'])
        current = occupancy_image(np, geometry, raw, native, row['ann_ids'], cost['q_by_GT'], parsed, metadata, image['id'])
        for g, ann in enumerate(coco.imgToAnns[image['id']]):
            qid = cost['q_by_GT'][g]; gt_original_pixels = int(raw[g].sum(dtype=np.int64)); gt_native_pixels = int(native[g].sum(dtype=np.int64))
            current[qid].update(original_GT_pixels=gt_original_pixels, nearest800_GT_pixels=gt_native_pixels,
                                empty_original_GT=gt_original_pixels == 0, empty_nearest800_GT=gt_native_pixels == 0,
                                iscrowd=int(ann.get('iscrowd', 0)), ignore=int(ann.get('ignore', 0)))
            GT_rows.append(dict(image_id=image['id'], annotation_id=ann['id'], query_id=qid, GT_index=g,
                                original_pixels=gt_original_pixels, nearest800_pixels=gt_native_pixels,
                                iscrowd=int(ann.get('iscrowd', 0)), ignore=int(ann.get('ignore', 0)),
                                E0=current[qid]['E0'], eligible_pairs=current[qid]['eligible_pairs']))
        occupancy_rows.extend(current)
        del parsed, raw, native, current
    require(len(GT_rows) == 380 and {(r['image_id'], r['annotation_id']) for r in GT_rows} ==
            {(ann['image_id'], ann['id']) for ann in coco.anns.values()}, 'A raw annotation was omitted from diagnosis')
    summary = aggregate_occupancy(occupancy_rows)
    summary.update(empty_original_GT=sum(r['original_pixels'] == 0 for r in GT_rows),
                   empty_nearest800_GT=sum(r['nearest800_pixels'] == 0 for r in GT_rows),
                   crowd_raw_GT=sum(r['iscrowd'] != 0 for r in GT_rows), ignore_raw_GT=sum(r['ignore'] != 0 for r in GT_rows))
    require(not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES'] == '' and
            psutil.Process().cpu_affinity() == [0, 1], 'CPU consumer initialized CUDA or changed CPU policy')
    acceptance = dict(schema='ROOT-native-geometry-paired-passive-population-CPU-diagnostic-v1',
        status='ROOT_PASSIVE_PAIRED_POPULATION_SAMPLING_CPU_DIAGNOSED', phase=spec['phase'], source=entry, reader_source=reader,
        spec=spec_ref, GPU_admission=admission, owner_driver=owner, parent_receipt=reference(receipt_path),
        report=reference(report_path), start=reference(Path(spec['output_directory']) / 'START.json'),
        checkpoint=spec['checkpointmap']['paired'], qualificationmap=spec['qualificationmap'],
        expected_epoch=spec['expected_epoch'], formal_evaluation_qualification=spec['formal_evaluation_qualification'],
        source_outputs=report['outputs'], geometry_source=spec['geometry_source'], occupancy_summary=summary,
        query_rows=occupancy_rows, raw_GT_assignment_rows=GT_rows, cost_reconstruction_and_assignment=cost_proofs,
        official_endpoint_regression=official, independent_CPU_native_nearest800_allGT_digest_exact=True,
        independent_CPU_full_paired612_CP_digest_matches_before_after=True,
        CPU_independently_replayed_ten_outputs=False, CPU_independently_verified_M0_sign_or_E=False,
        E0_residual_zero_source_observed_but_omitted_values_not_CPU_verified=True,
        outside_donor_slots_applied=False, RNG_summary_verified_but_generator_trajectory_not_replayed=True,
        AP_computed=False, scientific_method_verdict=None, feature_purity_claimed=False, causal_claimed=False,
        acquisition_independence_claimed=False, CPU_affinity=[0, 1], torch_threads=2, CUDA_initialized=False,
        limits=[
            'Own means stock final-cost assigned annotation, which depends on paired-on masks; it is not semantic identity or detection recall.',
            'Native nearest800 and original-coordinate bilinear occupancy are separate measures; the original proxy does not reproduce native two-resize rendering.',
            'Own and other union can overlap; pixel location is not feature purity, sea background, sampling utility or causal evidence.',
            'All16 statistics include unused/ineligible/fallback positions; E0 eligible means are None, never imputed as successful observations.',
            'All380 raw annotations and18700 queries retain denominators; eligible pair-weighted means explicitly use their separate point denominator.',
            'Actual E/coordinates are retained, not independently reconstructed from omitted F/M0. Paired unused weak donor precomputation is never applied.',
            'Weighted-cost sums are checked with a predeclared floating rounding bound, not bitwise GPU/CPU identity or independent sampling integral reconstruction.',
            'CPU SciPy independently solves the actual exported total; optimal objective agreement permits different qids on ties.',
            'Full612 CP tensors are independently CPU checked; all10/RNG/hook evidence is source-bound digest summary, not forward or CUDA-generator replay.',
            'Same-CP qualified formal masks/factors/final digests agree exactly; this finite observation is not a method-gain/AP/causal verdict.'])
    with Path(out_file).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=acceptance['status'], output=reference(out_file)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); qualify(args.spec, args.out)


if __name__ == '__main__':
    main()
