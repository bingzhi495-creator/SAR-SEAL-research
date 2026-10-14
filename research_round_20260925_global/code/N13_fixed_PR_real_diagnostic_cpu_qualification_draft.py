"""UNBOUND_DRAFT: independent CPU checks for the real fixed-PR GPU diagnostic.

Not run, not source-bound, not an admission, and not a qualification result.
ROOT must review and bind a separate immutable source and complete the explicit
external ABI and numeric-location contract before one owner-controlled attempt.
No launch, subprocess, GPU calculation, training, publication, AP evaluation, or
automatic retry is implemented. The CPU output cannot certify its own terminal.

The exact frozen GPU native L determines A/domains/statistics. CPU rendering is
used only for F_original, through the original renderer/N10 feature path. Separate
ascending 2048-point old-LOG/halo partitions independently evaluate the eight
frozen weights with functional LN/linear/ReLU. Stored GPU C remains the candidate
for every scientific statistic, even where a CPU/GPU near-zero sign differs.
Legacy numeric tolerances are compatibility checks, not certified error bounds.
The pinned pure statistics/oracle source is reused; oracle sorting is therefore
not claimed to be an independently implemented ordering algorithm.

External ABI required, not inspected while authoring this draft:
  lineage.check_recipe / own_gate(..., cpu=True) / owner_ref_join /
  endpoint_gate / prior_gate / selector_parent_gate;
  evaluation.load_heads / NAMES / digest; N10.packet_check / original_feature /
  segmentation; reader.population / rle_bytes; renderer original feature ABI;
  common.imported_files. Every missing ABI fails explicitly; no guessed fallback.

CPU spec schema: ROOT-N13-fixed-PR-real-diagnostic-CPU-spec-v1. In addition to the
GPU diagnostic's carrier references, diagnostic_sources, fixed_population and
diagnostic_protocol, ROOT supplies sources, RAW_inputs (normal94), eight preserved
failed RAWs, identity pins102, CPU_admission_file, entry_source, phase, Q_path,
details_path, deadline_seconds=1800, and diagnostic_GPU_{report,spec,source,RAW,
admission}. cpu_check_contract requires legacy_numeric_source (the already locked
carrier GPU spec), legacy_numeric_object_path (the actual numeric_protocol JSON
location), exact refined_support_atol/rtol copied from that object, and separate
caller-prelocked formula_atol/rtol. No absolute near-zero limit or mismatch-count
cap replaces the legacy rule: a CPU/GPU sign difference is rejected precisely
when abs(CPU_refined_logit) > atol + rtol * abs(CPU_refined_logit). Within-range
counts are recorded per pair and overall. This is a practical allowance, not a
certified bound. ROOT must verify the actual JSON location and formula tolerances.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types


SCHEMA = 'ROOT-N13-fixed-PR-real-diagnostic-CPU-spec-v1'
Q_SCHEMA = 'N13-fixed-PR-real-diagnostic-CPU-qualification-v1'
Q_STATUS = 'ROOT_N13_FIXED_PR_REAL_DIAGNOSTIC_CPU_QUALIFIED_V1'
GPU_SCHEMA = 'N13-fixed-PR-real-diagnostic-GPU-report-v1'
GPU_STATUS = 'N13_FIXED_PR_REAL_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_CPU'
LOCKED = {'matched_count': 374, 'unmatched_gt_count': 6, 'total_gt_count': 380}
DISTANCE_NAMES = ('D_le_1', 'D_gt_1_le_2', 'D_gt_2_le_4', 'D_gt_4_or_infinite')
DISTANCE_SOURCE = 'native_in_domain_opposite_chessboard_D;empty_full_infinite;fixed_bands1_2_4'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
CURRENT_KEYS = ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt',
                'additional_failed_RAW_inputs', 'terminal_identity_pins', 'phase',
                'entry_source', 'CPU_admission_file', 'Q_path', 'details_path',
                'deadline_seconds')
WEIGHT_SHAPES = ((91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,))
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def api(module, name):
    value = getattr(module, name, None)
    require(callable(value), 'ROOT must adapt and review missing external ABI: ' + name)
    return value


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and
            type(ref['sha256']) is str and len(ref['sha256']) == 64 and
            set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed absolute pinned reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def closed(closure, ref):
    ref_shape(ref)
    require(closure.get(ref['path']) == ref, 'Input is absent from complete pinned SOURCE closure: ' + ref['path'])
    return bound(ref)


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
            'SOURCE changed between binding and import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code=0):
    # Original RAW JSON only. No science, SOURCE hashing/import, or process probe.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and
            path.parent.name == phase, 'Original RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
            raw['returncode'] == code and type(raw['original_identity']) is dict and
            raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and
            raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Every ancestor needs actual held exit/reap/original birth absence before the first SHA/import')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'RAW phase differs')
    peak, sha = raw['whole_GPU_observed_peak_MiB'], raw['admission_sha256']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Original RAW resource/admission scope differs')
    return raw


def raw_only_pre_gate(spec):
    # Mirrors GPU pre_gate's entire RAW portion before it is safe to import that
    # pinned module. The actual inherited GPU pre_gate is called after this gate.
    rows = spec['RAW_inputs']
    require(type(spec['normal_RAW_count']) is int and spec['normal_RAW_count'] == 94 and
            type(rows) is list and len(rows) == 94 and
            len({row['receipt']['path'] for row in rows}) == 94, 'Exact normal94 ancestry required')
    normals = [terminal_raw_only(row['receipt'], row['phase']) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    first_failed = terminal_raw_only(failed_ref, failed_phase, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extra),
            'Exact unchanged extra failed7 rows, eight failures in total, required')
    failures = [first_failed] + [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extra]
    failed_paths = [failed_ref['path']] + [row['receipt']['path'] for row in extra]
    require(len(set(failed_paths)) == 8 and
            not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
            'Normal94 and preserved failed8 owners must be distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref,
                phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and len(pins) == 102 and
            set(pins) == {row['receipt']['path'] for row in all_rows}, 'Exact pins102 required')
    for row, raw in zip(all_rows, normals + failures):
        pin = pins[row['receipt']['path']]
        require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
                pin['phase'] == row['phase'] and type(pin['returncode']) is int and
                pin['returncode'] == row['returncode'] and pin['original_identity'] == raw['original_identity'] and
                pin['admission_sha256'] == raw['admission_sha256'],
                'Actual original identity/admission differs from its prelocked pin')
    return normals


def numeric_contract(spec, closure):
    contract = spec['cpu_check_contract']
    keys = {'legacy_numeric_source', 'legacy_numeric_object_path', 'refined_support_atol',
            'refined_support_rtol', 'formula_atol', 'formula_rtol', 'certified_error_bound'}
    require(type(contract) is dict and set(contract) == keys and contract['certified_error_bound'] is False,
            'ROOT must explicitly prelock the legacy numeric-location and formula comparison contract')
    legacy_ref = contract['legacy_numeric_source']
    require(legacy_ref == spec['carrier_GPU_spec'],
            'Legacy support values must come from the already locked carrier GPU numeric_protocol')
    legacy = read(closed(closure, legacy_ref))
    locations = contract['legacy_numeric_object_path']
    require(type(locations) is list and all(type(key) is str and key for key in locations),
            'Explicit actual legacy numeric JSON object path required; no inferred fallback')
    for key in locations:
        require(type(legacy) is dict and key in legacy,
                'ROOT must adapt the actual legacy numeric JSON location: ' + key)
        legacy = legacy[key]
    require(type(legacy) is dict and all(key in legacy for key in ('refined_support_atol', 'refined_support_rtol')),
            'Actual legacy refined_support_atol/rtol object required')
    for key in ('refined_support_atol', 'refined_support_rtol', 'formula_atol', 'formula_rtol'):
        require(type(contract[key]) in (int, float) and math.isfinite(contract[key]) and contract[key] >= 0,
                'Finite prelocked compatibility tolerance required: ' + key)
    require(all(contract[key] == legacy[key] for key in ('refined_support_atol', 'refined_support_rtol')),
            'CPU compatibility tolerance differs from actual legacy locked range')
    return contract


def source_gate(spec_path, spec):
    require(spec['schema'] == SCHEMA and type(spec['deadline_seconds']) is int and
            spec['deadline_seconds'] == 1800, 'Exact CPU schema and prelocked 1800-second deadline required')
    protocol = spec['diagnostic_protocol']
    require(protocol['carrier_seed'] == 202610120701 and protocol['carrier_epoch'] == 48 and
            protocol['arm'] == 'LOG_PR' and protocol['population'] == LOCKED and
            protocol['distance_names'] == list(DISTANCE_NAMES), 'Exact first701 LOG E48 fixed diagnostic required')
    raw_normals = raw_only_pre_gate(spec)
    # FIRST HASH occurs only after ALL normal94, failed8 and pins102 pass.
    require(type(spec['sources']) is list, 'Complete pinned sources list required')
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in spec['RAW_inputs'] for key in RAW_KEYS if key != 'phase') and
            spec['prior_failed_CPU_receipt'] in closure.values() and
            all(row['receipt'] in closure.values() for row in spec['additional_failed_RAW_inputs']),
            'All original owner metadata and failed RAWs must belong to SOURCE closure')
    for key in ('entry_source', 'diagnostic_GPU_source', 'diagnostic_GPU_spec', 'diagnostic_GPU_report',
                'diagnostic_GPU_RAW', 'diagnostic_GPU_admission', 'carrier_GPU_spec', 'carrier_CPU_spec',
                'carrier_CPU_qualification', 'carrier_evaluation_source', 'carrier_CPU_source',
                'carrier_GPU_RAW', 'carrier_CPU_RAW', 'carrier_CPU_admission', 'fixed_population'):
        closed(closure, spec[key])
    require(Path(spec['entry_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True),
            'Only ROOT-bound current scientific execution source may pass entry identity')
    gpu = load(spec['diagnostic_GPU_source'], '_N13_real_diagnostic_CPU_inherited_GPU')
    inherited_normals, inherited_closure = api(gpu, 'pre_gate')(spec, 94)
    require(inherited_normals == raw_normals and inherited_closure == closure,
            'The actual pinned GPU pre_gate must confirm the same RAW/source closure')
    gpu_spec = read(closed(closure, spec['diagnostic_GPU_spec']))
    report = read(closed(closure, spec['diagnostic_GPU_report']))
    original = read(closed(closure, spec['carrier_GPU_spec']))
    carrier_cpu = read(closed(closure, spec['carrier_CPU_spec']))
    carrier_q = read(closed(closure, spec['carrier_CPU_qualification']))
    require(gpu_spec['schema'] == gpu.SCHEMA and report['schema'] == GPU_SCHEMA and
            report['status'] == GPU_STATUS and report['source'] == spec['diagnostic_GPU_source'] and
            report['spec'] == spec['diagnostic_GPU_spec'] and report['GPU_admission'] == spec['diagnostic_GPU_admission'] and
            report['phase'] == gpu_spec['phase'] and gpu_spec['entry_source'] == spec['diagnostic_GPU_source'] and
            Path(gpu_spec['GPU_admission_file']).resolve(strict=True) == Path(spec['diagnostic_GPU_admission']['path']).resolve(strict=True),
            'Original diagnostic GPU report/spec/source/admission identities differ')
    require(type(gpu_spec['normal_RAW_count']) is int and gpu_spec['normal_RAW_count'] == 93 and
            type(report['normal_RAW_count']) is int and report['normal_RAW_count'] == 93 and
            len(gpu_spec['terminal_identity_pins']) == 101 and
            gpu_spec['RAW_inputs'] == report['RAW_inputs'] == spec['RAW_inputs'][:93] and
            gpu_spec['terminal_identity_pins'] == report['terminal_identity_pins'] ==
            {key: value for key, value in spec['terminal_identity_pins'].items()
             if key != spec['diagnostic_GPU_RAW']['path']} and
            gpu_spec['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
            gpu_spec['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'],
            'Exact GPU inheritedprefix93 and unchanged eight failures required')
    latest = spec['RAW_inputs'][93]
    require(latest['receipt'] == spec['diagnostic_GPU_RAW'] and latest['admission'] == spec['diagnostic_GPU_admission'] and
            latest['phase'] == gpu_spec['phase'] and raw_normals[93]['admission_sha256'] ==
            spec['diagnostic_GPU_admission']['sha256'], 'Actual just-terminated diagnostic GPU owner must be appended as owner94')
    for key in ('carrier_GPU_spec', 'carrier_CPU_spec', 'carrier_CPU_qualification', 'carrier_evaluation_source',
                'carrier_CPU_source', 'carrier_GPU_RAW', 'carrier_CPU_RAW', 'carrier_CPU_admission',
                'fixed_population', 'diagnostic_sources', 'diagnostic_protocol'):
        require(gpu_spec[key] == spec[key], 'CPU/GPU fixed protocol or inherited carrier reference differs: ' + key)
    require(report['fixed_population'] == spec['fixed_population'] and report['diagnostic_protocol'] == protocol and
            report['carrier_CPU_qualification'] == spec['carrier_CPU_qualification'] and
            report['images'] == 187 and report['matched_pairs'] == 374 and report['unmatched_GT'] == 6 and report['raw_GT'] == 380 and
            report['original_all18700_native_export_exact'] is True and
            report['all374_original_LOG_full_binary_reconstructed'] is True and
            report['author_width2_all374_independent_integer_checked'] is True and
            report['candidate_generated_without_G_or_S_or_K'] is True and report['independent_CPU_qualified'] is False and
            report['current_owned_GPU_terminal_self_certified'] is False and report['no_TEST'] is True and
            report['AP_evaluated'] is False, 'Original GPU diagnostic completeness/scope differs')
    require(original['seed'] == carrier_cpu['seed'] == carrier_q['seed'] == 202610120701 and
            original['head_endpoint_epoch'] == carrier_cpu['head_endpoint_epoch'] == carrier_q['head_endpoint_epoch'] == 48 and
            carrier_q['status'] == 'ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1' and
            carrier_q['evaluation_spec'] == spec['carrier_GPU_spec'] and carrier_q['spec'] == spec['carrier_CPU_spec'] and
            carrier_q['source'] == spec['carrier_evaluation_source'] and carrier_q['reader_source'] == spec['carrier_CPU_source'] and
            carrier_q['evaluation_GPU_RAW'] == spec['carrier_GPU_RAW'] and carrier_q['CPU_admission'] == spec['carrier_CPU_admission'] and
            carrier_q['original_native_374pair_6FN_cohort_preserved'] is True and
            carrier_q['exact_GPU_L_regions_all18700_CPU_checked'] is True and
            carrier_q['independent_two_PR_formula_all_selected_pixels_CPU_checked'] is True,
            'Previously qualified actual carrier identities differ')
    require(all(ref in closure.values() for payload in (original, carrier_cpu, gpu_spec) for ref in payload['sources']),
            'Every original carrier and diagnostic source must remain pinned')
    context = dict(original)
    for key in CURRENT_KEYS:
        context[key] = spec[key]
    if 'output_directory' in spec:
        context['output_directory'] = spec['output_directory']
    lineage = load(original['endpoint_lineage_source'], '_N13_real_diagnostic_CPU_lineage')
    api(lineage, 'check_recipe')(original)
    gate = api(lineage, 'own_gate')(spec_path, context, __file__, closure, raw_normals, cpu=True)
    join = api(lineage, 'owner_ref_join')
    gi = join(context, gate['owners'], spec['carrier_GPU_RAW'], carrier_q['evaluation_GPU_admission'],
              carrier_q['source'], carrier_q['evaluation_spec'], '0')
    ci = join(context, gate['owners'], spec['carrier_CPU_RAW'], carrier_q['CPU_admission'],
              carrier_q['reader_source'], carrier_q['spec'], '')
    di = join(context, gate['owners'], spec['diagnostic_GPU_RAW'], spec['diagnostic_GPU_admission'],
              spec['diagnostic_GPU_source'], spec['diagnostic_GPU_spec'], '0')
    require(gi < ci < di and type(di) is int and di == 93 and
            original['RAW_inputs'] == spec['RAW_inputs'][:gi] and carrier_cpu['RAW_inputs'] == spec['RAW_inputs'][:ci] and
            gpu_spec['RAW_inputs'] == spec['RAW_inputs'][:di], 'Actual carrier/GPU diagnostic chronological owner joins differ')
    gate.update(lineage=lineage, context=context, closure=closure, carrier_Q=carrier_q,
                carrier_CPU=carrier_cpu, gpu_report=report, gpu_spec=gpu_spec)
    gate.update(api(lineage, 'endpoint_gate')(context, closure, gate['owners']))
    prior, prior_q = api(lineage, 'prior_gate')(context, closure, gate['owners'])
    require(prior['teacher_checkpoint'] == gate['cache_report']['teacher_checkpoint'], 'Exact shared teacher required')
    gate.update(prior=prior, prior_q=prior_q, report=prior, packets=prior_q['packet_refs'],
                selector=api(lineage, 'selector_parent_gate')(context, closure, gate['owners']))
    fixed = read(closed(closure, spec['fixed_population']))
    carrier_details = read(closed(closure, carrier_q['details']))
    pairing_key = 'LOG_PR_seed202610120701_E48_fixed_native_score'
    selected = [row for row in carrier_details['fixed_native_query_GT_pairs'][pairing_key]
                if abs(row['threshold'] - .5) < 1e-12]
    require(len(selected) == 1 and fixed['rows'] == selected[0]['GT_rows'] and len(fixed['rows']) == 380 and
            sum(row['status'] == 'native_matched_fixed_query' for row in fixed['rows']) == 374 and
            sum(row['status'] == 'native_FN' for row in fixed['rows']) == 6 and
            all(row['status'] in ('native_matched_fixed_query', 'native_FN') for row in fixed['rows']) and
            all((row['query_id'] is not None) == (row['status'] == 'native_matched_fixed_query') for row in fixed['rows']),
            'Only the original locked native .50 JSON pairing may provide matched374 and unmatched6')
    require(len({(row['image_id'], row['gt_id']) for row in fixed['rows']}) == 380 and
            len({(row['image_id'], row['query_id']) for row in fixed['rows'] if row['query_id'] is not None}) == 374,
            'Unique actual fixed pairing required')
    require(type(report['candidate_packets']) is list and len(report['candidate_packets']) == 374,
            'Every matched pair needs its actual GPU candidate packet')
    for key in ('population', 'pair_diagnostics', 'image_checks', 'trace', 'start'):
        closed(closure, report[key])
    for ref in report['candidate_packets']:
        closed(closure, ref)
    for ref in spec['diagnostic_sources'].values():
        closed(closure, ref)
    gate.update(fixed=fixed, numeric=numeric_contract(spec, closure),
                evaluation=load(spec['carrier_evaluation_source'], '_N13_real_diagnostic_CPU_evaluation'),
                statistics=load(spec['diagnostic_sources']['statistics'], '_N13_real_diagnostic_CPU_statistics'),
                oracle=load(spec['diagnostic_sources']['oracle'], '_N13_real_diagnostic_CPU_oracle'))
    # Adapter, domains, pipeline and author sources stay pinned but are not called
    # by the independent arithmetic/domain paths below.
    return gate


def deadline(started, spec):
    require(time.monotonic() - started < spec['deadline_seconds'],
            'Prelocked CPU diagnostic deadline exceeded; no automatic retry')


def cpu_tensor(t, value, name, shape, dtype):
    require(isinstance(value, t.Tensor) and value.device.type == 'cpu' and value.dtype == dtype and
            value.layout == t.strided and tuple(value.shape) == tuple(shape), 'CPU tensor shape/dtype/device differs: ' + name)
    if dtype != t.bool:
        require(bool(t.isfinite(value).all().item()), 'Nonfinite CPU tensor: ' + name)
    return value


def cpu_tree(t, value):
    if isinstance(value, t.Tensor):
        require(value.device.type == 'cpu', 'Stored GPU packet must deserialize to CPU-only tensors')
    elif isinstance(value, dict):
        for item in value.values():
            cpu_tree(t, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            cpu_tree(t, item)


def independent_domains(np, P, L):
    padded = np.pad(P.astype(np.int64), ((2, 2), (2, 2)), mode='constant', constant_values=0)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode='constant').cumsum(0).cumsum(1)
    count = integral[5:, 5:] - integral[:-5, 5:] - integral[5:, :-5] + integral[:-5, :-5]
    require(count.shape == P.shape and bool(((count >= 0) & (count <= 25)).all()),
            'Independent integer 5x5 zero-pad count/25 differs')
    old = np.abs(L) <= 2
    width = (P & (count <= 22)) | (~P & (count >= 3))
    halo = width & ~old
    return dict(native_P=P, old_LOG=old, native_width2=width, new_halo=halo, Omega=old | halo)


def independent_distance_bands(np, cv2, ndimage, P):
    if not P.any() or P.all():
        distance = np.full(P.shape, np.inf, dtype=np.float64)
        scipy_distance = distance.copy()
    else:
        opposite_for_P = cv2.distanceTransform(P.astype(np.uint8), cv2.DIST_C, 3)
        opposite_for_not_P = cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_C, 3)
        distance = np.where(P, opposite_for_P, opposite_for_not_P).astype(np.float64)
        scipy_distance = np.where(P, ndimage.distance_transform_cdt(P, metric='chessboard'),
                                  ndimage.distance_transform_cdt(~P, metric='chessboard')).astype(np.float64)
        require(bool((distance >= 1).all()) and np.array_equal(distance, np.floor(distance)),
                'OpenCV DIST_C must yield positive integer in-domain opposite-label distances')
    require(np.array_equal(distance, scipy_distance), 'Independent OpenCV DIST_C differs from original SciPy native distances')
    bands = (distance <= 1, (distance > 1) & (distance <= 2),
             (distance > 2) & (distance <= 4), distance > 4)
    require(np.array_equal(sum(mask.astype(np.int8) for mask in bands), np.ones(P.shape, dtype=np.int8)),
            'Prelocked native bands must partition the full raster')
    return {name: dict(mask=mask, source=DISTANCE_SOURCE) for name, mask in zip(DISTANCE_NAMES, bands)}


def independent_residual(t, weights, features, native):
    # Independent of the GPU adapter/head calls; no old-LOG internal gate.
    coarse = native.clamp(-8.0, 8.0).unsqueeze(1) / 8.0
    hidden = t.nn.functional.layer_norm(features, (256,), weight=None, bias=None, eps=1e-5)
    for offset in (0, 2, 4):
        hidden = t.nn.functional.relu(t.nn.functional.linear(t.cat((hidden, coarse), dim=1),
                                                            weights[offset], weights[offset + 1]))
    return t.nn.functional.linear(t.cat((hidden, coarse), dim=1), weights[6], weights[7]).squeeze(1)


def check_partition(t, np, weights, feature_points, Lflat, domain, stored, label, numeric, started, spec):
    indices = t.from_numpy(np.flatnonzero(domain).astype(np.int64, copy=False))
    count = int(indices.numel())
    require(type(stored) is dict and set(stored) == {'flat_indices', 'raw_residual', 'refined_logit',
                'binary_candidate', 'chunk_count'}, 'Stored partition schema differs: ' + label)
    cpu_tensor(t, stored['flat_indices'], label + '.flat_indices', (count,), t.int64)
    for name in ('raw_residual', 'refined_logit'):
        cpu_tensor(t, stored[name], label + '.' + name, (count,), t.float32)
    cpu_tensor(t, stored['binary_candidate'], label + '.binary_candidate', (count,), t.bool)
    require(t.equal(indices, stored['flat_indices']) and type(stored['chunk_count']) is int and
            stored['chunk_count'] == (count + 2047) // 2048, 'Exact ascending separate2048 partition differs: ' + label)
    require(t.equal(stored['refined_logit'], Lflat[indices] + stored['raw_residual']) and
            t.equal(stored['binary_candidate'], stored['refined_logit'] > 0),
            'Stored GPU partition logit/binary self-consistency differs: ' + label)
    max_residual_error = max_logit_error = 0.0
    mismatches = 0
    max_near_zero_CPU_abs = max_near_zero_GPU_abs = 0.0
    for start in range(0, count, 2048):
        deadline(started, spec)
        end = min(count, start + 2048)
        chosen = indices[start:end]
        residual = independent_residual(t, weights, feature_points[chosen], Lflat[chosen])
        refined = Lflat[chosen] + residual
        gpu_residual, gpu_refined = stored['raw_residual'][start:end], stored['refined_logit'][start:end]
        require(bool(t.isfinite(residual).all().item()) and bool(t.isfinite(refined).all().item()),
                'Independent CPU formula produced nonfinite values')
        atol, rtol = numeric['formula_atol'], numeric['formula_rtol']
        require(t.allclose(residual, gpu_residual, atol=atol, rtol=rtol) and
                t.allclose(refined, gpu_refined, atol=atol, rtol=rtol),
                'Independent CPU residual/logit exceeds caller-prelocked formula compatibility range: ' + label)
        changed_sign = (refined > 0) != stored['binary_candidate'][start:end]
        differences = int(changed_sign.sum().item())
        if differences:
            CPU_abs = refined[changed_sign].abs()
            tau = numeric['refined_support_atol'] + numeric['refined_support_rtol'] * CPU_abs
            require(not bool((CPU_abs > tau).any().item()),
                    'CPU/GPU sign difference lies outside the actual legacy practical support allowance: ' + label)
            max_near_zero_CPU_abs = max(max_near_zero_CPU_abs, float(CPU_abs.max().item()))
            max_near_zero_GPU_abs = max(max_near_zero_GPU_abs, float(gpu_refined[changed_sign].abs().max().item()))
            del CPU_abs, tau
        mismatches += differences
        max_residual_error = max(max_residual_error, float((residual - gpu_residual).abs().max().item()))
        max_logit_error = max(max_logit_error, float((refined - gpu_refined).abs().max().item()))
        del residual, refined, gpu_residual, gpu_refined, changed_sign, chosen
    return dict(points=count, chunk_count=(count + 2047) // 2048, max_raw_residual_abs_error=max_residual_error,
                max_refined_logit_abs_error=max_logit_error, near_zero_sign_differences=mismatches,
                max_observed_compatible_sign_difference_CPU_abs=max_near_zero_CPU_abs,
                max_observed_compatible_sign_difference_GPU_abs=max_near_zero_GPU_abs,
                practical_support_rule='reject_sign_difference_only_if_CPU_abs_gt_atol_plus_rtol_times_CPU_abs',
                compatibility_check=True, certified_error_bound=False)


def pixels(np, A, G):
    return dict(pixel_TP=int(np.count_nonzero(A & G)), pixel_FP=int(np.count_nonzero(A & ~G)),
                pixel_FN=int(np.count_nonzero(~A & G)))


def pair_identity(pair):
    return {key: pair[key] for key in ('image_id', 'gt_id', 'query_id')}


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec)
    deadline(started, spec)
    Q_path, details_path = Path(spec['Q_path']), Path(spec['details_path'])
    require(Q_path.is_absolute() and details_path.is_absolute() and Q_path != details_path and
            not Q_path.exists() and not details_path.exists() and Q_path.parent.is_dir() and details_path.parent.is_dir(),
            'Distinct fresh absolute qualification/details files with existing ROOT-owned parents required')
    require(all(os.environ.get(key) == '2' for key in THREAD_ENV), 'Pre-import numeric thread environment must be2')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1] and psutil.virtual_memory().available >= 16 * 1024**3,
            'Actual CPU01/RAM16GiB required')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified cold Torch CPU runtime required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0' and scipy.__version__ == '1.10.1',
            'Qualified numerical library versions required')
    context, evaluation, closure = gate['context'], gate['evaluation'], gate['closure']
    heads, states, proofs = api(evaluation, 'load_heads')(torch, context, gate)
    require(isinstance(evaluation.NAMES, (tuple, list)) and len(evaluation.NAMES) == 8 and len(set(evaluation.NAMES)) == 8,
            'ROOT must confirm actual eight checkpoint keys in exact architecture order')
    weights = tuple(states['LOG_PR'][key] for key in evaluation.NAMES)
    for index, (weight, shape) in enumerate(zip(weights, WEIGHT_SHAPES)):
        cpu_tensor(torch, weight, 'LOG_PR.weight' + str(index), shape, torch.float32)
    require(sum(weight.numel() for weight in weights) == 40497, 'Exact frozen LOG head must have40497 scalars')
    n10 = load(context['N10_evaluation_source'], '_N13_real_diagnostic_CPU_N10')
    reader = load(context['reader_source'], '_N13_real_diagnostic_CPU_native_reader')
    renderer = load(context['renderer_source'], '_N13_real_diagnostic_CPU_original_renderer')
    coco = COCO(str(closed(closure, context['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1}, 'Actual original DEV187/GT380 required')
    require({row['gt_id'] for row in gate['fixed']['rows']} == set(coco.anns) and
            all(coco.anns[row['gt_id']]['image_id'] == row['image_id'] for row in gate['fixed']['rows']),
            'Locked fixed population must contain every actual DEV GT exactly once')
    native_ref = context['native_reference_predictions']
    native_rows = read(closed(closure, native_ref))
    native_pop = api(reader, 'population')(coco, native_rows, ids)
    original_report = read(closed(closure, gate['carrier_Q']['evaluation_report']))
    log_ref = original_report['predictions']['LOG_PR_seed202610120701_E48_fixed_native_score']
    log_rows = read(closed(closure, log_ref))
    log_pop = api(reader, 'population')(coco, log_rows, ids)
    require(len(native_rows) == len(log_rows) == 18700 and all(
            {key: value for key, value in native.items() if key != 'segmentation'} ==
            {key: value for key, value in refined.items() if key != 'segmentation'}
            for native, refined in zip(native_rows, log_rows)),
            'Original full native/LOG query/class/score/order/empty-score identity differs')
    image_rows = read(closed(closure, gate['prior']['image_rows']))
    manifest = gate['selector']['manifest']
    require(len(image_rows) == len(manifest['rows']) == len(gate['packets']) == 187 and
            [row['image_id'] for row in image_rows] == ids, 'Canonical full187 packet/selector population required')
    paired = {iid: [row for row in gate['fixed']['rows'] if row['image_id'] == iid and row['query_id'] is not None] for iid in ids}
    ordered_pairs = [row for iid in ids for row in paired[iid]]
    unmatched = [dict(image_id=row['image_id'], gt_id=row['gt_id']) for row in gate['fixed']['rows'] if row['query_id'] is None]
    gpu_report = gate['gpu_report']
    gpu_diagnostics = read(closed(closure, gpu_report['pair_diagnostics']))
    gpu_population = read(closed(closure, gpu_report['population']))
    gpu_images = read(closed(closure, gpu_report['image_checks']))
    require(len(ordered_pairs) == len(gpu_diagnostics) == len(gpu_report['candidate_packets']) == 374 and
            [row['identity'] for row in gpu_diagnostics] == [pair_identity(pair) for pair in ordered_pairs] and
            len(unmatched) == 6 and len(gpu_images) == 187, 'All real fixed pairs and six identity-only unmatched GT required')
    gpu_image_pairs = [record for image in gpu_images for record in image['pairs']]
    require([row['image_id'] for row in gpu_images] == ids and
            all(row['dataset_index'] == index and row['native_all100_exact'] is True and row['selector_exact'] is True
                for index, row in enumerate(gpu_images)) and
            [record['identity'] for record in gpu_image_pairs] == [pair_identity(pair) for pair in ordered_pairs] and
            [record['candidate'] for record in gpu_image_pairs] == gpu_report['candidate_packets'],
            'GPU trace image/pair/candidate correspondence differs')
    libraries = api(gate['common'], 'imported_files')(torch, np, cv2, masks, COCO, closure, gate['admission'], ndimage)
    require(not torch.cuda.is_initialized(), 'CPU gates/imports must leave CUDA uninitialized')
    outside_rng = torch.get_rng_state().clone()
    frozen_weights = tuple(weight.clone() for weight in weights)
    diagnostics, checked_images, numeric_checks = [], [], []
    candidate_cursor = native_checked = 0
    logical_points = 0
    chunks = {'old_LOG': 0, 'new_halo': 0}
    with torch.no_grad():
        for index, (image_row, packet_ref, selector_row) in enumerate(zip(image_rows, gate['packets'], manifest['rows'])):
            deadline(started, spec)
            packet = torch.load(str(closed(closure, packet_ref)), map_location='cpu')
            fields, identity, qids = api(n10, 'packet_check')(torch, reader, coco, packet, image_row, index, packet_ref, gate)
            cpu_tree(torch, fields)
            carrier = torch.load(str(closed(closure, selector_row['carrier'])), map_location='cpu')
            cpu_tree(torch, carrier)
            expected_selector = dict(dataset_index=index, image_id=image_row['image_id'], identity=identity,
                native_packet=packet_ref, natural_query_ids=list(range(100)), native_export_query_ids=qids,
                renderer_source=context['renderer_source'])
            require(carrier['schema'] == gate['lineage'].SELECTOR_PACKET_SCHEMA and
                    all(carrier[key] == selector_row[key] == value for key, value in expected_selector.items()) and
                    carrier['source'] == gate['selector']['report']['source'] and
                    carrier['spec'] == gate['selector']['report']['spec'] and
                    carrier['GPU_admission'] == gate['selector']['report']['GPU_admission'] and
                    api(evaluation, 'digest')(carrier['native_L']) == carrier['L_digest'] == selector_row['L_digest'],
                    'Exact frozen GPU selector L/source identity differs')
            L = carrier['native_L']
            require(isinstance(L, torch.Tensor) and L.ndim == 3 and L.shape[0] == 100, 'Frozen native L must have100 natural queries')
            cpu_tensor(torch, L, 'exact frozen GPU native L', tuple(L.shape), torch.float32)
            require(len(qids) == 100 and set(qids) == set(range(100)) and
                    [record['query_id'] for record in native_pop[image_row['image_id']]] == qids and
                    [record['query_id'] for record in log_pop[image_row['image_id']]] == qids,
                    'Actual natural100/export query identities differ')
            for ordinal, qid in enumerate(qids):
                segmentation = api(n10, 'segmentation')(np, masks, (L[qid] > 0).numpy().astype(np.uint8))
                require(segmentation == native_pop[image_row['image_id']][ordinal]['segmentation'],
                        'Frozen exact GPU L native full binary differs from original all100 output')
                native_checked += 1
            # Re-render F on CPU through the original N10/renderer path. Never
            # render a replacement CPU L or use one for eligibility/statistics.
            F_original = api(n10, 'original_feature')(torch, renderer, fields['F256'], identity)
            cpu_tensor(torch, F_original, 'CPU re-rendered original F', (256, int(L.shape[1]), int(L.shape[2])), torch.float32)
            feature_points = F_original.reshape(256, -1).transpose(0, 1)
            log_lookup = {record['query_id']: record for record in log_pop[image_row['image_id']]}
            per_image = []
            for pair in paired[image_row['image_id']]:
                deadline(started, spec)
                candidate_ref = gpu_report['candidate_packets'][candidate_cursor]
                candidate = torch.load(str(closed(closure, candidate_ref)), map_location='cpu')
                cpu_tree(torch, candidate)
                require(candidate['schema'] == 'N13-fixed-PR-real-candidate-v1' and
                        candidate['source'] == spec['diagnostic_GPU_source'] and candidate['spec'] == spec['diagnostic_GPU_spec'] and
                        candidate['GPU_admission'] == spec['diagnostic_GPU_admission'] and candidate['phase'] == gate['gpu_spec']['phase'] and
                        candidate['identity'] == pair_identity(pair) and candidate['native_packet'] == packet_ref and
                        candidate['selector'] == selector_row['carrier'] and
                        candidate['head_checkpoint'] == context['endpoint']['checkpoints']['LOG_PR'],
                        'Stored actual GPU candidate source/spec/identity/head differs')
                z = L[pair['query_id']]
                A = (z > 0).numpy().copy()
                domains = independent_domains(np, A, z.numpy())
                stored_domains = candidate['domains']
                require(set(stored_domains) == set(domains), 'Stored actual GPU domain schema differs')
                for name, domain in domains.items():
                    cpu_tensor(torch, stored_domains[name], 'stored GPU.' + name, A.shape, torch.bool)
                    require(np.array_equal(domain, stored_domains[name].numpy()),
                            'Independent frozen-L/integer-width2 domain differs from GPU: ' + name)
                counts = {name: int(np.count_nonzero(mask)) for name, mask in domains.items()}
                counts['outside'] = int(A.size) - counts['Omega']
                require(counts == gpu_image_pairs[candidate_cursor]['domain_counts'] and
                        counts['old_LOG'] == image_row['band_counts_all100'][pair['query_id']],
                        'Actual old-LOG K and independently reconstructed domain counts differ')
                bands = independent_distance_bands(np, cv2, ndimage, A)
                stored = candidate['candidates']
                require(type(stored) is dict and set(stored) == {'status', 'refined_binary_rule', 'point_chunk_size',
                    'originalLOG', 'addedHalo', 'native_support_L_gt_zero', 'merged_C'} and
                    stored['refined_binary_rule'] == 'strict_gt_zero' and stored['point_chunk_size'] == 2048,
                    'Stored GPU candidate fixed strict>0/separate2048 contract differs')
                cpu_tensor(torch, stored['native_support_L_gt_zero'], 'stored GPU.native_A', A.shape, torch.bool)
                cpu_tensor(torch, stored['merged_C'], 'stored GPU.frozen_C', A.shape, torch.bool)
                require(np.array_equal(A, stored['native_support_L_gt_zero'].numpy()), 'Stored candidate native support differs')
                arithmetic = {}
                for label, domain_name in (('originalLOG', 'old_LOG'), ('addedHalo', 'new_halo')):
                    arithmetic[label] = check_partition(torch, np, weights, feature_points, z.reshape(-1),
                        domains[domain_name], stored[label], label, gate['numeric'], started, spec)
                merged = (z > 0).contiguous().clone()
                for label in ('originalLOG', 'addedHalo'):
                    merged.reshape(-1)[stored[label]['flat_indices']] = stored[label]['binary_candidate']
                require(torch.equal(merged, stored['merged_C']), 'Frozen GPU C fails exact partition assembly')
                C = stored['merged_C'].numpy().copy()
                reconstructed = A.copy()
                reconstructed[domains['old_LOG']] = C[domains['old_LOG']]
                original_LOG = masks.decode(api(reader, 'rle_bytes')(log_lookup[pair['query_id']]['segmentation'])).astype(bool)
                require(np.array_equal(reconstructed, original_LOG), 'Full original LOG binary reconstruction differs')
                G = masks.decode(api(reader, 'rle_bytes')(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(bool)
                require(G.shape == A.shape and pixels(np, A, G) ==
                        {key: pair['native'][key] for key in ('pixel_TP', 'pixel_FP', 'pixel_FN')} and
                        pixels(np, reconstructed, G) == {key: pair['refined'][key] for key in ('pixel_TP', 'pixel_FP', 'pixel_FN')},
                        'Native/LOG pixel counts differ from the original locked pair JSON')
                metadata = dict(carrier_seed=202610120701, carrier_epoch=48, carrier_arm='LOG_PR',
                    comparison='offline_privileged_fixed374_not_AP',
                    distance_definition='native_in_domain_opposite_chessboard_D;empty_full_infinite;bands1_2_4',
                    source=spec['diagnostic_GPU_source'], spec=spec['diagnostic_GPU_spec'], native_packet=packet_ref,
                    selector=selector_row['carrier'], original_LOG_predictions=log_ref, actual_pair_source=spec['fixed_population'])
                require(candidate['diagnostic'] == gpu_diagnostics[candidate_cursor] and
                        candidate['diagnostic']['metadata'] == metadata, 'Stored per-pair GPU JSON/provenance differs')
                diagnostic = api(gate['statistics'], 'diagnose_instance')(gate['oracle'], A=A, C=C, G=G,
                    old_LOG=domains['old_LOG'], new_halo=domains['new_halo'], S=domains['old_LOG'], K=counts['old_LOG'],
                    image_id=pair['image_id'], gt_id=pair['gt_id'], query_id=pair['query_id'],
                    candidate_independent_of_selection=True, metadata=metadata, distance_bands=bands)
                require(diagnostic == candidate['diagnostic'], 'Every actual C/A/G/S/K statistic must exactly reproduce GPU pair JSON')
                diagnostics.append(diagnostic)
                record = dict(identity=pair_identity(pair), candidate=candidate_ref, partition_checks=arithmetic,
                    domain_counts=counts, full_original_LOG_binary_exact=True, locked_native_LOG_pixel_counts_exact=True,
                    independent_integer_width2_exact=True, independent_OpenCV_chessboard_exact=True,
                    frozen_GPU_candidate_statistics_exact=True)
                per_image.append(record)
                numeric_checks.append(dict(identity=pair_identity(pair), partitions=arithmetic))
                logical_points += counts['Omega']
                chunks['old_LOG'] += arithmetic['originalLOG']['chunk_count']
                chunks['new_halo'] += arithmetic['addedHalo']['chunk_count']
                candidate_cursor += 1
                del candidate, stored, merged, C, G, reconstructed, original_LOG, domains, stored_domains, bands, A, z
            checked_images.append(dict(dataset_index=index, image_id=image_row['image_id'], native_all100_binary_exact=True,
                                       exact_frozen_GPU_selector_checked=True, pairs=per_image))
            del packet, fields, carrier, L, F_original, feature_points, log_lookup, per_image
            gc.collect()
            require(not torch.cuda.is_initialized(), 'Independent CPU check initialized CUDA unexpectedly')
            print(json.dumps(dict(CPU_checked_images=index + 1, CPU_checked_pairs=candidate_cursor)), flush=True)
    require(native_checked == 18700 and candidate_cursor == len(diagnostics) == 374 and len(checked_images) == 187,
            'Full DEV187/all18700 native support/fixed374 verification incomplete')
    population = api(gate['statistics'], 'diagnose_population')(gate['oracle'], matched_diagnostics=diagnostics,
        unmatched_gt=unmatched, metadata=dict(protocol=spec['diagnostic_protocol'], fixed_population=spec['fixed_population']),
        locked_population=LOCKED)
    require(population == gpu_population and logical_points == gpu_report['logical_candidate_points'] and
            chunks == gpu_report['actual_PR_chunk_calls'], 'Complete fixed374+6 population JSON/logical count/chunk totals differ')
    numeric_totals = {}
    for label in ('originalLOG', 'addedHalo'):
        records = [row['partitions'][label] for row in numeric_checks]
        numeric_totals[label] = dict(
            points=sum(row['points'] for row in records), chunk_count=sum(row['chunk_count'] for row in records),
            near_zero_sign_differences=sum(row['near_zero_sign_differences'] for row in records),
            pairs_with_near_zero_sign_differences=sum(row['near_zero_sign_differences'] > 0 for row in records),
            max_raw_residual_abs_error=max(row['max_raw_residual_abs_error'] for row in records),
            max_refined_logit_abs_error=max(row['max_refined_logit_abs_error'] for row in records),
            max_observed_compatible_sign_difference_CPU_abs=max(
                row['max_observed_compatible_sign_difference_CPU_abs'] for row in records),
            max_observed_compatible_sign_difference_GPU_abs=max(
                row['max_observed_compatible_sign_difference_GPU_abs'] for row in records),
            certified_error_bound=False)
    require(all(torch.equal(before, after) for before, after in zip(frozen_weights, weights)) and
            all(torch.equal(value, head.state_dict()[key]) for arm, head in heads.items() for key, value in states[arm].items()) and
            torch.equal(outside_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'Frozen endpoint weights/head/RNG or CPU-only execution invariant changed')
    deadline(started, spec)
    details = dict(schema='N13-fixed-PR-real-diagnostic-CPU-details-v1', source=gate['source'], spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'], phase=spec['phase'], diagnostic_GPU_report=spec['diagnostic_GPU_report'],
        diagnostic_GPU_RAW=spec['diagnostic_GPU_RAW'], fixed_population=spec['fixed_population'],
        images=checked_images, numeric_checks=numeric_checks, numeric_totals=numeric_totals, numeric_contract=gate['numeric'],
        recomputed_pair_diagnostics=diagnostics, recomputed_population=population,
        native_support_from_exact_frozen_GPU_L=True, statistics_candidate='actual_frozen_GPU_C',
        CPU_L_used_for_domains_or_statistics=False, CPU_near_zero_signs_replace_GPU_C=False,
        independent_oracle_ordering_implementation=False, legacy_numeric_is_certified_error_bound=False)
    write_new(details_path, details)
    deadline(started, spec)
    qualification = dict(schema=Q_SCHEMA, status=Q_STATUS, source=gate['source'], spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'], phase=spec['phase'], details=reference(details_path),
        RAW_inputs=spec['RAW_inputs'], normal_RAW_count=94, terminal_identity_pins=spec['terminal_identity_pins'],
        diagnostic_GPU_report=spec['diagnostic_GPU_report'], diagnostic_GPU_spec=spec['diagnostic_GPU_spec'],
        diagnostic_GPU_source=spec['diagnostic_GPU_source'], diagnostic_GPU_RAW=spec['diagnostic_GPU_RAW'],
        diagnostic_GPU_admission=spec['diagnostic_GPU_admission'], carrier_CPU_qualification=spec['carrier_CPU_qualification'],
        fixed_population=spec['fixed_population'], diagnostic_protocol=spec['diagnostic_protocol'],
        exact_GPU_inheritedprefix93_and_actual_owner94_joined=True, original_native_374pair_6FN_cohort_preserved=True,
        exact_frozen_GPU_L_all18700_native_support_checked=True, independent_CPU_original_feature_path_checked=True,
        independent_eight_weight_functional_PR_formula_all_candidate_points_checked=True,
        independent_integer_width2_all374_checked=True, independent_OpenCV_native_chessboard_all374_checked=True,
        all374_original_LOG_full_binary_reconstructed=True, all374_locked_native_LOG_pixel_counts_exact=True,
        all374_frozen_GPU_C_pair_statistics_exact=True, complete_fixed374_plus6_population_statistics_exact=True,
        images=187, matched_pairs=374, unmatched_GT=6, raw_GT=380, head_state_proofs=proofs,
        logical_candidate_points=logical_points, actual_PR_chunk_calls=chunks,
        numeric_totals=numeric_totals, numeric_contract=gate['numeric'],
        actual_imported_files=libraries, total_elapsed_seconds=time.monotonic() - started,
        CPU_only_no_CUDA_initialized=True, independent_oracle_ordering_implementation=False,
        legacy_numeric_is_certified_error_bound=False, no_TEST=True, AP_evaluated=False,
        deployment_gain_established=False, innovation_established=False,
        current_owned_CPU_terminal_self_certified=False, whole_leaf_VRAM_certified=False,
        evidence_limits='First701 LOG E48, locked native .50 matched374 plus identity-only unmatched6; privileged local IoU diagnostics. '
                        'Compatibility tolerances are actual legacy ranges, not certified bounds. Reused oracle/statistics arithmetic; '
                        'no independent sorting-code claim, AP, causality, deployment, cross-data or innovation conclusion.')
    write_new(Q_path, qualification)
    print(json.dumps(dict(status=Q_STATUS, qualification=reference(Q_path),
                          current_owned_CPU_terminal_self_certified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    options = parser.parse_args()
    require(options.owner_execute, 'UNBOUND_DRAFT: ROOT must first bind/review a separate scientific source and own the single attempt')
    run(options.spec)
