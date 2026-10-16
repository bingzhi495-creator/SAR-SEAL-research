"""UNBOUND independent CPU reconstruction of N22 saved full-C GPU actions.

Production entry: run(torch, np, spec, control). No teacher, PR, F or matcher
forward is loaded/executed. Torch/NumPy are injected by the owned CPU wrapper.
Synthetic tests use fake Torch tensors and NumPy, never actual GPU outputs.
The R comparator is the fixed Omega restriction of this same full-C forward,
not a bit-exact reproduction of an earlier S-only forward or an AP result.
"""
import importlib.util
import json
from pathlib import Path


PACKET_SCHEMA = 'N22-FIT32-natural100-fullC-actual-GPU-action-packet-v1'
REPORT_SCHEMA = 'N22-INDEPENDENT-CPU-fullC-count-verification-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(type(value) is int and value >= 0, name + ' must be a nonnegative integer')
    return value


def _tensor(torch, np, value, dtype, shape, name):
    _require(torch.is_tensor(value), name + ' must be a saved tensor')
    _require(value.device.type == 'cpu' and value.dtype == getattr(torch, dtype),
             name + ' must have its exact saved CPU dtype')
    _require(not value.requires_grad, name + ' must be detached')
    result = value.detach().numpy()
    _require(tuple(result.shape) == tuple(shape), name + ' shape mismatch')
    if dtype == 'float32':
        _require(bool(np.isfinite(result).all()), name + ' contains nonfinite values')
    return result


def _indices(torch, np, value, pixel_count, name):
    _require(torch.is_tensor(value), name + ' must be an int64 tensor')
    result = _tensor(torch, np, value, 'int64', (int(value.numpy().size),), name)
    _require(bool(((result >= 0) & (result < pixel_count)).all()), name + ' out of bounds')
    _require(len(result) < 2 or bool((result[1:] > result[:-1]).all()),
             name + ' must be strictly increasing and unique')
    return result


def dilate17(np, positive):
    """Exactly one 17x17 all-ones binary dilation with outside-image zero.

    Two Boolean radius-eight one-dimensional max filters are equivalent to
    the square structuring element. Integer prefix sums avoid new packages.
    """
    _require(positive.ndim == 2 and positive.dtype == np.bool_, 'Expected 2D bool mask')
    result = positive
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (8, 8)
        padded = np.pad(result, padding, mode='constant', constant_values=False)
        prefix = np.cumsum(padded, axis=axis, dtype=np.int32)
        zero_shape = list(prefix.shape)
        zero_shape[axis] = 1
        prefix = np.concatenate((np.zeros(zero_shape, dtype=np.int32), prefix), axis=axis)
        left, right = [slice(None), slice(None)], [slice(None), slice(None)]
        left[axis], right[axis] = slice(None, -17), slice(17, None)
        result = (prefix[tuple(right)] - prefix[tuple(left)]) > 0
    return result


def _population(torch, np, packet, gt_count):
    assignment = packet['assignment']
    natural = list(range(100))
    _require(packet['natural_query_ids'] == natural and assignment['natural_query_ids'] == natural,
             'Natural query identities must be exactly 0..99')
    raw = _tensor(torch, np, assignment['raw_GT_index_by_query'], 'int64', (100,), 'raw GT assignment')
    _require(bool(((raw >= -2) & (raw < gt_count)).all()), 'Invalid original raw-GT assignment')
    positive, negative, ignored = (np.flatnonzero(raw >= 0).tolist(),
                                  np.flatnonzero(raw == -1).tolist(),
                                  np.flatnonzero(raw == -2).tolist())
    _require(len(set(raw[raw >= 0].tolist())) == len(positive), 'Original matcher reuses a GT')
    _require(assignment['native_positive_query_indices'] == positive
             and assignment['native_negative_query_indices'] == negative
             and assignment['native_ignored_query_indices'] == ignored,
             'Original matcher query partitions disagree')
    kinds = ['matched' if x >= 0 else 'background' if x == -1 else 'ignored' for x in raw]
    ids = [packet['raw_GT_ids'][int(x)] if x >= 0 else None for x in raw]
    _require(assignment['supervision_kind_by_query'] == kinds
             and assignment['raw_GT_annotation_id_by_query'] == ids,
             'Original matcher kind or annotation identity mismatch')
    valid = _tensor(torch, np, assignment['valid_targets'], 'bool', (100,), 'valid targets')
    _require(bool(np.array_equal(valid, raw != -2)), 'Ignored queries must remain excluded')
    _require(assignment['background_empty_set_is_new_supervision_extension'] is True
             and assignment['original_native_negative_mask_supervised'] is False
             and assignment['ignore_never_zero_filled'] is True,
             'Background/ignored proxy boundary changed')
    for key in ('final_native_labels', 'final_native_label_weights'):
        _tensor(torch, np, assignment[key], 'int64', (100,), key)
    _tensor(torch, np, assignment['final_native_mask_weights'], 'float32', (100,), 'native mask weights')
    return raw, dict(image_count=1, collected_queries=100, unassigned_queries=len(negative),
                     ignored_queries=len(ignored), unmatched_gt=gt_count-len(positive),
                     total_real_gt=gt_count)


def _region(np, region, c_region, native_p, gt, union_gt, on_c_gt, before, after):
    native_fp = native_p & ~gt & region
    fp_c = before & ~on_c_gt & c_region
    repair_fp_c = fp_c & ~after
    result = dict(C_count=int(np.count_nonzero(c_region)),
                  native_fn=int(np.count_nonzero(gt & ~native_p & region)),
                  native_fp=int(np.count_nonzero(native_fp)),
                  pr_beneficial_fn=int(np.count_nonzero(~before & on_c_gt & after & c_region)),
                  pr_beneficial_fp=int(np.count_nonzero(repair_fp_c)),
                  pr_harmful_tp=int(np.count_nonzero(before & on_c_gt & ~after & c_region)),
                  pr_harmful_tn=int(np.count_nonzero(~before & ~on_c_gt & after & c_region)),
                  native_fp_other_real_GT=int(np.count_nonzero(native_fp & union_gt)),
                  native_fp_pure_background=int(np.count_nonzero(native_fp & ~union_gt)))
    return result, repair_fp_c


def verify_packet(torch, np, packet, *, check):
    """Rebuild all 100 queries' C/actions, then count only real original pairs."""
    check()
    _require(packet['schema'] == PACKET_SCHEMA, 'Unexpected packet schema')
    dataset_index = _integer(packet['dataset_index'], 'dataset_index')
    _require(type(packet['identity']) is dict and 'image_id' in packet['identity'], 'Missing image identity')
    native = packet['native_L']
    _require(torch.is_tensor(native) and native.numpy().ndim == 3, 'native_L must be 100xHxW')
    shape = tuple(native.numpy().shape)
    _require(shape[0] == 100 and min(shape[1:]) > 0, 'native_L population or spatial shape mismatch')
    native = _tensor(torch, np, native, 'float32', shape, 'native_L')
    h, w = shape[1:]
    gt_tensor = packet['raw_GT']
    _require(torch.is_tensor(gt_tensor) and gt_tensor.numpy().ndim == 3, 'Missing real raw GT masks')
    gt_count = int(gt_tensor.numpy().shape[0])
    gt = _tensor(torch, np, gt_tensor, 'bool', (gt_count, h, w), 'raw_GT')
    _require(gt_count == 0 or bool(gt.reshape(gt_count, -1).any(axis=1).all()), 'Real GT mask is empty')
    ann_ids = packet['raw_GT_ids']
    _require(type(ann_ids) is list and len(ann_ids) == gt_count
             and all(type(x) is int for x in ann_ids) and len(set(ann_ids)) == gt_count,
             'Real annotation IDs are not unique integer IDs')
    _require(packet['identity']['ann_ids'] == ann_ids, 'GT order differs from identity.ann_ids')
    mapping = packet['raw_target_mapping']
    _require(type(mapping) in (tuple, list) and all(type(x) is int for x in mapping)
             and sorted(mapping) == list(range(gt_count)), 'Processed-to-raw GT mapping is not bijective')
    _require(len(packet['native_stage_matches']) == 10
             and len(packet['all_ten_target_observations']) == 10,
             'Original ten-stage observation metadata is missing')
    _require('processed_target_proof' in packet and 'actual_raw_processed_GT_correspondence' in packet,
             'Original processed/raw correspondence evidence is missing')
    _tensor(torch, np, packet['native_e0'], 'float32', (1, 100, 256), 'native_e0')
    _tensor(torch, np, packet['native_class_logits'], 'float32', (1, 100, 2), 'native_class_logits')
    _require(sorted(packet['native_export_query_ids']) == list(range(100)), 'Export query IDs lost a natural query')
    for key in ('native_scores_by_query', 'native_top_scores_by_export_order', 'native_quality_by_export_order'):
        _tensor(torch, np, packet[key], 'float32', (100,), key)
    _require(type(packet['category_id']) is int and packet['category_id'] == 1, 'Unexpected category')
    _require(type(packet['queries']) is list and len(packet['queries']) == 100, 'Full all-100 query capture required')
    assignments, audit = _population(torch, np, packet, gt_count)
    union_gt = np.any(gt, axis=0)
    records, regional_counts = [], []
    max_difference, total_c, total_selected = 0.0, 0, 0
    for q, saved in enumerate(packet['queries']):
        check()
        _require(type(saved['query_id']) is int and saved['query_id'] == q, 'Query order or identity mismatch')
        l = native[q]
        p = l > np.float32(0)
        omega = np.abs(l) <= np.float32(2)
        c = omega | p | dilate17(np, p)
        expected_c = np.flatnonzero(c.reshape(-1)).astype(np.int64, copy=False)
        expected_s = np.flatnonzero(omega.reshape(-1)).astype(np.int64, copy=False)
        c_indices = _indices(torch, np, saved['C_indices'], h*w, 'C_indices')
        selected = _indices(torch, np, saved['StrongR_S_indices'], h*w, 'StrongR_S_indices')
        _require(bool(np.array_equal(c_indices, expected_c)), 'C differs from native Omega/P/17 dilation')
        _require(bool(np.array_equal(selected, expected_s)), 'StrongR S must equal native Omega')
        _require(type(saved['C_count']) is int and saved['C_count'] == len(c_indices)
                 and type(saved['K']) is int and saved['K'] == len(selected), 'Saved C/K counts disagree')
        n = len(c_indices)
        on_c_l = _tensor(torch, np, saved['native_L_on_C'], 'float32', (n,), 'native_L_on_C')
        _require(bool(np.array_equal(on_c_l, l.reshape(-1)[c_indices])), 'Saved native L on C mismatch')
        u = _tensor(torch, np, saved['PR_u'], 'float32', (n,), 'PR_u')
        actual = _tensor(torch, np, saved['PR_combined'], 'float32', (n,), 'PR_combined')
        with np.errstate(over='raise', invalid='raise'):
            cpu_combined = np.add(on_c_l, u, dtype=np.float32)
        _require(bool(np.isfinite(cpu_combined).all()), 'CPU FP32 combined contains nonfinite values')
        _require(bool(np.array_equal(cpu_combined > np.float32(0), actual > np.float32(0))),
                 'CPU FP32 L+u and actual GPU combined signs disagree')
        if n:
            max_difference = max(max_difference, float(np.max(np.abs(
                cpu_combined.astype(np.float64) - actual.astype(np.float64)))))
        before = _tensor(torch, np, saved['native_positive'], 'bool', (n,), 'native_positive')
        after = _tensor(torch, np, saved['actual_after_positive'], 'bool', (n,), 'actual_after_positive')
        flips = _tensor(torch, np, saved['actual_flip'], 'bool', (n,), 'actual_flip')
        _require(bool(np.array_equal(before, on_c_l > 0))
                 and bool(np.array_equal(after, actual > 0))
                 and bool(np.array_equal(flips, before ^ after)), 'Saved GPU action signs disagree')
        in_s = omega.reshape(-1)[c_indices]
        _require(int(np.count_nonzero(in_s)) == len(selected), 'S must be a distinct subset of C')
        total_c += n
        total_selected += len(selected)
        raw_index = int(assignments[q])
        if raw_index < 0:
            continue
        g = gt[raw_index]
        on_c_gt = g.reshape(-1)[c_indices]
        fn, fp, tp, tn = ~before & on_c_gt, before & ~on_c_gt, before & on_c_gt, ~before & ~on_c_gt
        bf, br = fn & after, fp & ~after
        ht, hn = tp & ~after, tn & after
        row = dict(pair_id=str(dataset_index)+':'+str(q)+':'+str(ann_ids[raw_index]),
                   image_id=packet['identity']['image_id'], query_index=q, gt_index=raw_index,
                   I=int(np.count_nonzero(p & g)), U=int(np.count_nonzero(p | g)),
                   native_fn=int(np.count_nonzero(~p & g)), native_fp=int(np.count_nonzero(p & ~g)),
                   candidate_fn=int(np.count_nonzero(fn)), candidate_fp=int(np.count_nonzero(fp)),
                   candidate_tp=int(np.count_nonzero(tp)), candidate_tn=int(np.count_nonzero(tn)),
                   pr_beneficial_fn=int(np.count_nonzero(bf)), pr_beneficial_fp=int(np.count_nonzero(br)),
                   b_FN=int(np.count_nonzero(bf & in_s)), b_FP=int(np.count_nonzero(br & in_s)),
                   h_TP=int(np.count_nonzero(ht & in_s)), h_TN=int(np.count_nonzero(hn & in_s)),
                   K=len(selected), selected_count=len(selected))
        # An independent full-mask reconstruction verifies the flip-count formula.
        routed = p.copy().reshape(-1)
        routed[c_indices[in_s]] = after[in_s]
        routed = routed.reshape(h, w)
        _require(int(np.count_nonzero(routed & g)) == row['I']+row['b_FN']-row['h_TP']
                 and int(np.count_nonzero(routed | g)) == row['U']-row['b_FP']+row['h_TN'],
                 'Actual Omega-restricted full mask and action counts disagree')
        regions = dict(pair_id=row['pair_id'], query_index=q, raw_GT_index=raw_index,
                       raw_GT_annotation_id=ann_ids[raw_index])
        for label, region, c_region in (('Omega_inside', omega, in_s),
                                       ('Omega_outside', ~omega, ~in_s)):
            counts, repair_fp_c = _region(np, region, c_region, p, g, union_gt, on_c_gt, before, after)
            on_c_union_gt = union_gt.reshape(-1)[c_indices]
            counts['pr_beneficial_fp_other_real_GT'] = int(np.count_nonzero(repair_fp_c & on_c_union_gt))
            counts['pr_beneficial_fp_pure_background'] = int(np.count_nonzero(repair_fp_c & ~on_c_union_gt))
            _require(counts['native_fp'] == counts['native_fp_other_real_GT']+counts['native_fp_pure_background']
                     and counts['pr_beneficial_fp'] == counts['pr_beneficial_fp_other_real_GT']
                     +counts['pr_beneficial_fp_pure_background'], 'FP descriptive strata do not reconcile')
            regions[label] = counts
        for key in ('native_fn', 'native_fp', 'pr_beneficial_fn', 'pr_beneficial_fp'):
            _require(regions['Omega_inside'][key]+regions['Omega_outside'][key] == row[key],
                     'Omega descriptive strata do not reconcile with main counts')
        _require(regions['Omega_inside']['C_count']+regions['Omega_outside']['C_count'] == n,
                 'Regional C count does not reconcile')
        _require(regions['Omega_inside']['pr_harmful_tp'] == row['h_TP']
                 and regions['Omega_inside']['pr_harmful_tn'] == row['h_TN'],
                 'Omega harmful PR counts do not reconcile with actual R')
        records.append(row)
        regional_counts.append(regions)
    check()
    return dict(dataset_index=dataset_index, identity=packet['identity'], records=records,
                audit_counts=audit, regional_counts=regional_counts,
                checks=dict(all_query_count=100, all_query_C_points=total_c,
                            all_query_Omega_selected_points=total_selected,
                            cpu_fp32_combined_sign_mismatches=0,
                            cpu_gpu_combined_max_abs_difference=max_difference,
                            used_actual_GPU_combined_signs_for_actions=True,
                            CPU_addition_overwrote_GPU_actions=False,
                            matched_IoU_zero_pairs_kept=True, GT_used_to_select_C_or_S=False))


def _load_source(ref, name, control):
    control.check()
    path = control.closed(ref)
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    control.check()
    return module


def run(torch, np, spec, control):
    """Owned CPU body; writes fresh counts/reports, leaves qualification pending."""
    control.check()
    _require(spec['cuda_visible_devices'] == '', 'CPU operation requires an empty CUDA device list')
    selected = spec['selected_dataset_indices']
    identities = spec['selected_identities']
    refs = spec['captured_packet_refs']
    _require(type(selected) is list and len(selected) == 32
             and all(type(x) is int and x >= 0 for x in selected)
             and len(set(selected)) == 32 and len(identities) == 32 and len(refs) == 32,
             'Expected exactly the prelocked 32 unique FIT identities')
    output = Path(spec['output_directory'])
    _require(output.is_absolute(), 'Output directory must be absolute')
    output.mkdir(parents=True, exist_ok=False)
    q = json.loads(control.closed(spec['GPU_capture_Q']).read_text(encoding='utf-8'))
    capture = json.loads(control.closed(spec['GPU_capture_REPORT']).read_text(encoding='utf-8'))
    _require(q['schema'] == 'N22-owned-pending-terminal-result-v1'
             and q['operation'] == 'GPU_CAPTURE'
             and q['status'] == 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
             'GPU capture Q shape/status mismatch; it is not a self-qualification')
    _require(q['result']['manifest_ref'] == spec['GPU_capture_REPORT']
             and q['result']['packet_refs'] == refs and capture['packet_refs'] == refs,
             'GPU Q/REPORT/spec packet or report bindings disagree')
    _require(capture['schema'] == 'N22-FIT32-natural100-fullC-capture-report-v1'
             and len(capture['rows']) == 32, 'GPU report population mismatch')
    analysis = _load_source(spec['analysis_source'], 'N22_bound_CPU_analysis', control)
    oracle = _load_source(spec['oracle_source'], 'N22_bound_CPU_oracle', control)
    all_records, regions, image_checks = [], [], []
    audit = dict(image_count=0, collected_queries=0, unassigned_queries=0,
                 ignored_queries=0, unmatched_gt=0, total_real_gt=0)
    for i, ref in enumerate(refs):
        control.check()
        source_row = capture['rows'][i]
        _require(source_row['packet_ref'] == ref and source_row['dataset_index'] == selected[i]
                 and source_row['identity'] == identities[i]
                 and source_row['image_id'] == identities[i]['image_id'],
                 'GPU report row differs from prelocked FIT identity')
        packet = torch.load(control.closed(ref), map_location='cpu', weights_only=True)
        _require(packet['dataset_index'] == selected[i] and packet['identity'] == identities[i],
                 'Saved packet identity differs from spec and GPU report')
        verified = verify_packet(torch, np, packet, check=control.check)
        all_records.extend(verified['records'])
        regions.extend(verified['regional_counts'])
        image_checks.append(dict(dataset_index=verified['dataset_index'], identity=verified['identity'],
                                 packet_ref=ref, checks=verified['checks'], audit_counts=verified['audit_counts']))
        for key in audit:
            audit[key] += verified['audit_counts'][key]
        del packet, verified
    summarized = analysis.analyze_instances(all_records, audit, oracle=oracle)
    control.check()
    analysis_ref = control.write_json(output/'ANALYSIS.json', summarized)
    summary = dict(images=32, natural_queries=audit['collected_queries'],
                   paired_instances=len(all_records),
                   native_iou_zero_pairs=summarized['population']['native_iou_zero_pairs'],
                   max_cpu_gpu_combined_abs_difference=max(
                       row['checks']['cpu_gpu_combined_max_abs_difference'] for row in image_checks),
                   cpu_fp32_combined_sign_mismatches=0,
                   method_table=summarized['method_table'],
                   descriptive_direction_table=summarized['descriptive_direction_table'])
    report = dict(schema=REPORT_SCHEMA,
        status='CPU_COUNTS_RECONSTRUCTED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        GPU_capture_Q=spec['GPU_capture_Q'], GPU_capture_REPORT=spec['GPU_capture_REPORT'],
        packet_refs=refs, analysis_source=spec['analysis_source'], oracle_source=spec['oracle_source'],
        analysis_ref=analysis_ref, summary=summary, population=audit,
        image_checks=image_checks, regional_counts=regions,
        regional_scope='Descriptive Omega in/out and other-real-GT/background FP strata; no separate region K or oracle',
        comparator_scope='R is fixed Omega restriction of the same actual full-C PR forward; not old S-only forward or AP reproduction',
        original_matcher_metadata_scope='Counts use preserved original assignments and real raw GT; ten-stage metadata presence checked, matcher not rerun',
        qualified=False, ROOT_actual_CPU0_verified=False, self_certified=False,
        is_AP=False, is_AP_bound=False, GT_used_to_select_C_or_S=False,
        teacher_loaded=False, PR_model_loaded=False, F_loaded=False,
        retraining_performed=False, research_direction_selected=False)
    report_ref = control.write_json(output/'CPU_REPORT.json', report)
    control.check()
    return dict(schema=REPORT_SCHEMA, report_ref=report_ref, analysis_ref=analysis_ref, summary=summary,
                qualified=False, ROOT_actual_CPU0_verified=False, self_certified=False,
                is_AP_bound=False, research_direction_selected=False)
