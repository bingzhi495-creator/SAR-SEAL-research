"""UNBOUND_DRAFT: complete fixed701 LOG E24 action-opportunity GPU diagnostic.
Not scientifically run; ROOT alone binds SOURCE/admission/actual metadata and owns execution.
Only authorized ordinary excerpts, pure helper and predata recipe were read.
No AP/training/TEST; numeric failure differs from scientific STOP; no automatic retry.
"""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

OWNER_SHA="ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924"
RAW_KEYS=("receipt","phase","admission","phase_binding","bootstrap")
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def api(module, name):
    value = getattr(module, name, None)
    require(callable(value), 'ROOT must adapt/review missing external ABI: ' + name)
    return value


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
            'Malformed absolute pinned reference')


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
    require(closure.get(ref['path']) == ref, 'Missing exact complete SOURCE reference: ' + ref['path'])
    return bound(ref)


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code):
    # Reads only original RAW JSON; no SOURCE hash/import or process probe.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name == phase,
            'Original RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
            raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held exit/reap/original birth absence required before any SHA/import')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Preserved RAW resource/admission scope differs')
    return raw


def check_pin(pin, raw, phase, code):
    require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and pin['admission_sha256'] == raw['admission_sha256'],
            'Original actual owner differs from its immutable identity/admission pin')


def ninth_failed_raw_only(spec):
    extra = spec['additional_preserved_independent_CPU_failure']
    require(type(extra) is dict and set(extra) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(extra['returncode']) is int and extra['returncode'] == 1,
            'Independent failed9 needs exactly RAW_ref/phase/pin/returncode1')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= 97 and
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 9 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 9 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 is separate; total terminal pins must equal normal+9')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9 has ALREADY passed its RAW-only identity check.
    rows = spec['RAW_inputs']
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= minimum and type(rows) is list and len(rows) == expected and
            len({row['receipt']['path'] for row in rows}) == expected, 'All original normal RAW ancestors required')
    normals = [terminal_raw_only(row['receipt'], row['phase'], 0) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    failed = terminal_raw_only(failed_ref, failed_phase, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extra),
            'Do not alter the original extra failed7 list or preserved failed8')
    failures = [failed] + [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extra]
    failed_paths = [failed_ref['path']] + [row['receipt']['path'] for row in extra]
    require(len(set(failed_paths)) == 8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
            'Normal owners and preserved failed8 owners must remain distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in all_rows},
            'Original normal-plus-failed8 pins must remain unchanged')
    for row, raw in zip(all_rows, normals + failures):
        check_pin(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    require(len(pins) == expected + 8 and
            spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'] not in
            {row['receipt']['path'] for row in all_rows},
            'Legacy normal+failed8 pins required; independent failed9 owner distinct')
    # FIRST HASH: failed9 and every normal/legacy failed8 RAW owner passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




from fractions import Fraction

EXPECTED_PROTOCOL = {
  "schema": "N15-fixed-E24-action-opportunity-locked-v1",
  "locked_UTC": "2026-10-14T04:31:14.389Z",
  "locked_before_N15_actual_data_execution": True,
  "carrier_seed": 202610120701,
  "carrier_epoch": 24,
  "arm": "LOG_PR",
  "carrier_label": "LOG_PR_seed202610120701_E24_fixed_native_score",
  "population": {
    "matched_count": 374,
    "unmatched_gt_count": 6,
    "total_gt_count": 380
  },
  "native_matching_threshold": 0.5,
  "unmatched_semantics": "UNEVALUABLE_NO_QUERY_NOT_ZERO_IOU_EXCLUDED_FROM374_MEAN",
  "candidate": {
    "native_binary": "P = original_GPU_native_L > 0",
    "old": "Omega = abs(original_GPU_native_L) <= 2",
    "wide": "C = Omega union P union Chebyshev_radius8_dilation(P), clipped to original raster",
    "radius_pixels": 8,
    "budget": "K = |Omega|",
    "GT_independent": True,
    "empty_P": "C = Omega",
    "K_zero": "No edits; native result"
  },
  "action": {
    "definition": "A = (original_GPU_native_L + fixed_E24_PR_ungated_residual) > 0",
    "pool_order": "sorted_unique_flat_original_raster_indices",
    "chunk": 2048,
    "common_action": "C_forward restricted to Omega and C for fair action oracle",
    "strong_carrier_check": "independent original Omega-only forward reconstructs exact fixed E24 prediction and refined counts",
    "overlap_check": "C versus Omega-only residual and combined absolute tolerance1e-4; every final binary sign identical; failure is numeric qualification failure with no automatic retry"
  },
  "comparisons": [
    "native",
    "actual_PR_Omega_original",
    "actual_PR_Omega_common_C",
    "action_oracle_atmostK_Omega_common_C",
    "action_oracle_atmostK_C",
    "action_oracle_exactK_C_with_real_noops",
    "GT_replacement_oracle_atmostK_C_same_pool_same_budget"
  ],
  "oracle": {
    "class_counts": {
      "a": "FP->TN",
      "b": "FN->TP",
      "c": "TP->FN",
      "d": "TN->FP",
      "z": "physical no-op"
    },
    "atmost_order": [
      "b",
      "a"
    ],
    "exact_order": [
      "b",
      "a",
      "z",
      "d",
      "c"
    ],
    "GT_classes": "a=C_FP,b=C_FN,c=d=0,z=C_TP+C_TN",
    "ground_truth_only_privileged_diagnostic": True
  },
  "main_comparison": "action_oracle_atmostK_C minus action_oracle_atmostK_Omega_common_C",
  "secondary_comparison": "action_oracle_atmostK_Omega_common_C minus actual_PR_Omega_common_C",
  "GO": {
    "all_conditions_required": True,
    "mean_delta_iou_min_fraction": {
      "numerator": 1,
      "denominator": 200
    },
    "mean_over_matched": 374,
    "sum_delta_iou_min_fraction": {
      "numerator": 374,
      "denominator": 200
    },
    "minimum_pairs_with_delta_ge_1_over200": 20,
    "increase_in_pairs_iou_ge_9_over10_min": 4,
    "rational_decision": True,
    "meaning": "Permit one fair selector hypothesis preregistration; not AP success, learnability, novelty or TGRS readiness"
  },
  "not_GO": "Stop only this fixed E24 PR/radius8 candidate-action selector recipe; no radius/budget sweep or late gate change",
  "resources": {
    "ROOT_single_owner": True,
    "deadline_seconds": 1800,
    "CPU_affinity": [
      0,
      1
    ],
    "threads": 2,
    "allocator_limit_MiB": 4096,
    "whole_GPU_observed_limit_MiB": 8192,
    "available_RAM_min_GiB": 16,
    "automatic_retry": False,
    "hidden_process": True
  },
  "qualification": {
    "GPU_original_held_handle_actual_exit0_required": True,
    "all_103_success_9_failed_ancestors_RAW_first": True,
    "full_bound_SOURCE_before_import": True,
    "CPU_same_actual_GPU_sampled_inputs": True,
    "CPU_two_independent_PR_formulae": True,
    "CPU_residual_combined_absolute_tolerance": 0.0001,
    "CPU_exact_final_sign_and_oracle_counts": True,
    "ROOT_after_CPU_actual_exit0_required": True
  },
  "outputs": {
    "conditional_IoU_only": True,
    "all_natural100_native_and_selector_provenance_checks": True,
    "PR_diagnostic_only_matched374": True,
    "full_AP_evaluated": False,
    "training_started": False,
    "no_TEST": True,
    "no_cross_dataset": True,
    "main_cost": "candidate fullC and extra originalOmega PR are real diagnostic cost; equal K is not equal compute"
  },
  "claim": "Test whether a wider GT-independent pool exposes helpful actual frozen E24 actions at the same K beyond what simply rejecting harmful actions in Omega can explain",
  "anti_claim": "More native errors or a high GT oracle do not imply this refiner can correct them or that a cheap selector is learnable",
  "future_selector_only_if_GO": "Fixed candidate M, scout capacity/input/labels/initialization/FIT sampling/trainbudget and full18,700 prediction evaluation; error-prediction + known fixed action control; action-label generation cost and sameFIT exposure disclosed; repeat seeds and strong E24 reference required separately"
}

SCHEMA = 'ROOT-N15-fixed-E24-action-opportunity-GPU-spec-v1'
CPU_SCHEMA = 'ROOT-N15-fixed-E24-action-opportunity-CPU-spec-v1'
REPORT_SCHEMA = 'N15-fixed-E24-action-opportunity-GPU-report-v1'
STATUS = 'N15_FIXED_E24_ACTION_OPPORTUNITY_COMPLETE_PENDING_INDEPENDENT_CPU'
CAPTURE_SCHEMA = 'N15-fixed-action-opportunity-actual-GPU-input-image-v1'
CAPTURE_MANIFEST_SCHEMA = 'N15-fixed-action-opportunity-GPU-capture-manifest-v1'
PAIR_SCHEMA = 'N15-fixed-E24-action-opportunity-pair-v1'
LOCKED_POPULATION = dict(matched_count=374, unmatched_gt_count=6, total_gt_count=380)
CARRIER_LABEL = 'LOG_PR_seed202610120701_E24_fixed_native_score'
CARRIER_KEYS = ('carrier_GPU_spec', 'carrier_GPU_report', 'carrier_GPU_RAW', 'carrier_GPU_admission',
    'carrier_CPU_spec', 'carrier_CPU_source', 'carrier_CPU_qualification', 'carrier_CPU_RAW',
    'carrier_CPU_admission', 'carrier_evaluation_source')
CLASS_KEYS = ('a', 'b', 'c', 'd', 'z')

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def deadline(started, spec):
    require(time.monotonic() - started < spec['deadline_seconds'], 'Locked N15 deadline exceeded; no automatic retry')

def digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())

def validate_protocol(spec, closure):
    protocol = read(closed(closure, spec['locked_protocol']))
    require(protocol == EXPECTED_PROTOCOL, 'Exact ROOT predata locked N15 protocol differs; no late gate, radius, budget or tolerance change')
    return protocol

def source_gate(path, spec, cpu=False):
    require(spec['schema'] == (CPU_SCHEMA if cpu else SCHEMA) and
        type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] >= 103 + int(cpu) and
        spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] and spec['deadline_seconds'] == 1800 and
        spec['automatic_retry'] is False and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
        'New N15 actual full ancestry/schema/owner resource recipe required')
    ninth_failed_raw_only(spec)
    raws, closure = pre_gate(spec, spec['expected_normal_RAW_count'])
    for key in CARRIER_KEYS + ('entry_source', 'locked_protocol', 'count_oracle_source', 'array_oracle_source', 'fixed_population'):
        closed(closure, spec[key])
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    if not cpu:
        require(Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True), 'Current N15 entry differs')
    protocol = validate_protocol(spec, closure)
    original = read(closed(closure, spec['carrier_GPU_spec']))
    cpu_spec = read(closed(closure, spec['carrier_CPU_spec']))
    q = read(closed(closure, spec['carrier_CPU_qualification']))
    report = read(closed(closure, spec['carrier_GPU_report']))
    require(original['seed'] == cpu_spec['seed'] == q['seed'] == 202610120701 and
        original['head_endpoint_epoch'] == cpu_spec['head_endpoint_epoch'] == q['head_endpoint_epoch'] == 24 and
        q['status'] == 'ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1' and
        q['evaluation_spec'] == report['spec'] == spec['carrier_GPU_spec'] and q['spec'] == spec['carrier_CPU_spec'] and
        q['source'] == report['source'] == original['entry_source'] == spec['carrier_evaluation_source'] and
        q['reader_source'] == cpu_spec['entry_source'] == spec['carrier_CPU_source'] and
        q['evaluation_report'] == spec['carrier_GPU_report'] and
        q['evaluation_GPU_RAW'] == spec['carrier_GPU_RAW'] and
        q['evaluation_GPU_admission'] == report['GPU_admission'] == spec['carrier_GPU_admission'] and
        q['CPU_admission'] == spec['carrier_CPU_admission'] and
        q['original_native_374pair_6FN_cohort_preserved'] is True and
        q['exact_GPU_L_regions_all18700_CPU_checked'] is True and
        q['independent_two_PR_formula_all_selected_pixels_CPU_checked'] is True,
        'Only original qualified strongest701 LOG E24 carrier is permitted; no N14/E48 substitution')
    require(Path(original['GPU_admission_file']).resolve(strict=True) == Path(spec['carrier_GPU_admission']['path']).resolve(strict=True) and
        Path(cpu_spec['CPU_admission_file']).resolve(strict=True) == Path(spec['carrier_CPU_admission']['path']).resolve(strict=True),
        'Carrier admission locators must resolve to actual bound refs')
    require(all(ref in closure.values() for payload in (original, cpu_spec) for ref in payload['sources']),
        'Full original carrier GPU/CPU Source closure required')
    context = dict(original)
    for key in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'terminal_identity_pins', 'phase', 'entry_source', 'output_directory', 'deadline_seconds'):
        context[key] = spec[key]
    context['CPU_admission_file' if cpu else 'GPU_admission_file'] = spec['CPU_admission_file' if cpu else 'GPU_admission_file']
    lineage = load(original['endpoint_lineage_source'], '_N15_original_E24_lineage')
    lineage.check_recipe(original)
    gate = lineage.own_gate(path, context, spec['entry_source']['path'] if cpu else __file__, closure, raws, cpu=cpu)
    gi = lineage.owner_ref_join(context, gate['owners'], spec['carrier_GPU_RAW'], spec['carrier_GPU_admission'],
        spec['carrier_evaluation_source'], spec['carrier_GPU_spec'], '0')
    ci = lineage.owner_ref_join(context, gate['owners'], spec['carrier_CPU_RAW'], spec['carrier_CPU_admission'],
        spec['carrier_CPU_source'], spec['carrier_CPU_spec'], '')
    require(gi < ci < spec['normal_RAW_count'] and original['RAW_inputs'] == spec['RAW_inputs'][:gi] and
        cpu_spec['RAW_inputs'] == spec['RAW_inputs'][:ci], 'Carrier actual GPU0/CPU0 chronological prefixes differ')
    for ancestor in (original, cpu_spec):
        require(all(spec['terminal_identity_pins'].get(key) == value for key, value in ancestor['terminal_identity_pins'].items()),
            'Preserved carrier identity/admission pins differ')
    gate.update(lineage=lineage, context=context, carrier_Q=q, carrier_CPU=cpu_spec, carrier_report=report, protocol=protocol)
    gate.update(lineage.endpoint_gate(context, closure, gate['owners']))
    prior, prior_q = lineage.prior_gate(context, closure, gate['owners'])
    require(prior['teacher_checkpoint'] == gate['cache_report']['teacher_checkpoint'], 'Exact qualified FIT/DEV teacher provenance differs')
    selector = lineage.selector_parent_gate(context, closure, gate['owners'])
    fixed = read(closed(closure, spec['fixed_population']))
    details = read(closed(closure, q['details']))
    matches = [row for row in details['fixed_native_query_GT_pairs'][CARRIER_LABEL] if abs(row['threshold'] - .5) < 1e-12]
    require(len(matches) == 1 and fixed['rows'] == matches[0]['GT_rows'] and len(fixed['rows']) == 380 and
        sum(row['status'] == 'native_matched_fixed_query' for row in fixed['rows']) == 374 and
        sum(row['status'] == 'native_FN' for row in fixed['rows']) == 6 and
        all(row['status'] in ('native_matched_fixed_query', 'native_FN') for row in fixed['rows']) and
        len({(row['image_id'], row['gt_id']) for row in fixed['rows']}) == 380 and
        len({(row['image_id'], row['query_id']) for row in fixed['rows'] if row['query_id'] is not None}) == 374 and
        all((row['query_id'] is None) == (row['status'] == 'native_FN') for row in fixed['rows']),
        'Exact strongE24 native .5 fixed374 matched plus6 unevaluable population required')
    closed(closure, report['predictions'][CARRIER_LABEL])
    gate.update(prior=prior, prior_q=prior_q, report=prior, packets=prior_q['packet_refs'], selector=selector,
        fixed=fixed, evaluation=load(spec['carrier_evaluation_source'], '_N15_original_fixed_E24_evaluation'),
        count_oracle=load(spec['count_oracle_source'], '_N15_pure_count_oracle'),
        array_oracle=load(spec['array_oracle_source'], '_N15_original_array_action_oracle'))
    return gate

def confusion(np, output, G):
    require(output.dtype == G.dtype == np.bool_ and output.shape == G.shape, 'Exact Boolean instance masks required')
    return dict(TP=int(np.count_nonzero(output & G)), FP=int(np.count_nonzero(output & ~G)), FN=int(np.count_nonzero(~output & G)))

def fixed_confusion(row):
    return dict(TP=row['pixel_TP'], FP=row['pixel_FP'], FN=row['pixel_FN'])

def rational(value):
    return dict(numerator=value.numerator, denominator=value.denominator)

def fraction(metric):
    value = metric['iou_fraction']
    return Fraction(value['numerator'], value['denominator'])

def classes_in(np, P, A, G, domain):
    return dict(a=int(np.count_nonzero(domain & P & ~A & ~G)),
        b=int(np.count_nonzero(domain & ~P & A & G)),
        c=int(np.count_nonzero(domain & P & ~A & G)),
        d=int(np.count_nonzero(domain & ~P & A & ~G)),
        z=int(np.count_nonzero(domain & (P == A))))

def branch(np, P, output, G, selected):
    counts = confusion(np, output, G)
    union = sum(counts.values())
    ratio = Fraction(counts['TP'], union) if union else Fraction(1)
    changes = classes_in(np, P, output, G, selected)
    return dict(after_counts=counts, intersection=counts['TP'], union=union, empty_union=(union == 0),
        iou=float(ratio), iou_fraction=rational(ratio), selected_counts=changes,
        selected_support=int(np.count_nonzero(selected)), actual_flips=sum(changes[key] for key in ('a','b','c','d')),
        selected_noops=changes['z'], helpful_flips=changes['a'] + changes['b'], harmful_flips=changes['c'] + changes['d'])

def domains_from_native(np, ndimage, native_L):
    require(native_L.dtype == np.float32 and native_L.ndim == 2 and bool(np.isfinite(native_L).all()), 'Exact GPU frozen L input required')
    P, Omega = native_L > 0, np.abs(native_L) <= 2
    dilation = ndimage.binary_dilation(P, structure=np.ones((17,17), dtype=np.bool_), iterations=1, border_value=0)
    C = Omega | P | dilation
    require(not np.any(Omega & ~C) and not np.any(P & ~C), 'C must contain physical Omega and native P')
    idx = np.flatnonzero(C).astype(np.int64)
    positions = np.flatnonzero(Omega.ravel()[idx]).astype(np.int64)
    require(len(positions) == int(np.count_nonzero(Omega)), 'Exact sorted Omega-to-C index mapping required')
    return P, Omega, C, idx, positions

def array_check(np, array_oracle, P, A, G, domain, K, mode, result):
    selected = (array_oracle.at_most_k_repair_oracle if mode == 'atmost' else array_oracle.exact_k_action_oracle)(P,A,G,domain,K)
    require(selected.dtype == np.bool_ and selected.shape == P.shape and not np.any(selected & ~domain), 'Original array oracle selected invalid support')
    output = array_oracle.apply_fixed_candidate(P,A,selected)
    actual = branch(np,P,output,G,selected)
    for key in ('after_counts','intersection','union','empty_union','iou_fraction','selected_counts',
                'selected_support','actual_flips','selected_noops','helpful_flips','harmful_flips'):
        require(actual[key] == result[key], 'Count versus qualified original array oracle differs: ' + key)

def describe_pair(np, count_oracle, array_oracle, P, G, A, original_Omega_output, Omega, C, pair, batch_compatibility):
    require(bool(G.any()) and G.shape == P.shape and not np.any(Omega & ~C), 'Nonempty fixed matched GT and nested physical domains required')
    native = confusion(np,P,G)
    require(native == fixed_confusion(pair['native']), 'Actual native fixedpair pixel counts differ')
    K = int(np.count_nonzero(Omega))
    omega_classes, C_classes = classes_in(np,P,A,G,Omega), classes_in(np,P,A,G,C)
    require(sum(omega_classes.values()) == K and sum(C_classes.values()) == int(np.count_nonzero(C)) and
        all(omega_classes[key] <= C_classes[key] for key in CLASS_KEYS), 'Actual nested domains/count classes disagree')
    within = dict(TP=int(np.count_nonzero(C & P & G)), FP=int(np.count_nonzero(C & P & ~G)),
        FN=int(np.count_nonzero(C & ~P & G)), TN=int(np.count_nonzero(C & ~P & ~G)))
    GT_classes = dict(a=within['FP'], b=within['FN'], c=0, d=0, z=within['TP'] + within['TN'])
    call = lambda classes,mode: count_oracle.oracle_counts(native['TP'],native['FP'],native['FN'],classes,K,mode)
    omega_oracle, C_oracle, exact_C, GT_C = call(omega_classes,'atmost'), call(C_classes,'atmost'), call(C_classes,'exact'), call(GT_classes,'atmost')
    for domain, mode, result, candidate in ((Omega,'atmost',omega_oracle,A), (C,'atmost',C_oracle,A),
            (C,'exact',exact_C,A), (C,'atmost',GT_C,G)):
        array_check(np,array_oracle,P,candidate,G,domain,K,mode,result)
    common_output = array_oracle.apply_fixed_candidate(P,A,Omega)
    original_actual = branch(np,P,original_Omega_output,G,Omega)
    common_actual = branch(np,P,common_output,G,Omega)
    require(original_actual['after_counts'] == fixed_confusion(pair['refined']) and np.array_equal(original_Omega_output,common_output),
        'Strong E24 original Omega-only/fullmask/refinedpair counts and same-C overlap sign must remain exact')
    zero = np.zeros(P.shape,dtype=np.bool_)
    baseline = branch(np,P,P,G,zero)
    main = fraction(C_oracle) - fraction(omega_oracle)
    secondary = fraction(omega_oracle) - fraction(common_actual)
    require(Fraction(1,2) <= fraction(baseline) <= fraction(omega_oracle) <= fraction(C_oracle) <= fraction(GT_C) and
        fraction(common_actual) <= fraction(exact_C) <= fraction(C_oracle) and main >= 0 and secondary >= 0,
        'Matched .5/native/oracle feasibility monotonicity differs')
    minimum_harm = max(0,K-C_classes['a']-C_classes['b']-C_classes['z'])
    require(exact_C['harmful_flips'] == minimum_harm, 'Exact-C physical noop/minimum unavoidable harm differs')
    return dict(schema=PAIR_SCHEMA, identity={key:pair[key] for key in ('image_id','gt_id','query_id')},
        budget_K=K, C_count=int(np.count_nonzero(C)), native=baseline, native_confusion_within_C=within,
        classes_Omega=omega_classes, classes_C=C_classes,
        branches=dict(actual_PR_Omega_original=original_actual, actual_PR_Omega_common_C=common_actual,
            action_oracle_atmostK_Omega_common_C=omega_oracle, action_oracle_atmostK_C=C_oracle,
            action_oracle_exactK_C_with_real_noops=exact_C, GT_replacement_oracle_atmostK_C_same_pool_same_budget=GT_C),
        main_B_delta_fraction=rational(main), secondary_A_delta_fraction=rational(secondary),
        exact_C_unavoidable_harms=minimum_harm, actual_Omega_harmful_flips=common_actual['harmful_flips'],
        additional_action_classes_C_minus_Omega={key:C_classes[key]-omega_classes[key] for key in CLASS_KEYS},
        batch_compatibility=batch_compatibility, array_count_crosscheck_exact=True,
        GT_privileged_action_selection=True, AP_result=False)

def summarize(diagnostics, unmatched):
    require(len(diagnostics) == 374 and len(unmatched) == 6 and
        len({tuple(row['identity'][key] for key in ('image_id','gt_id','query_id')) for row in diagnostics}) == 374,
        'Complete374 plus separately unevaluable6 required')
    gains = [Fraction(**row['main_B_delta_fraction']) for row in diagnostics]
    secondary = [Fraction(**row['secondary_A_delta_fraction']) for row in diagnostics]
    total = sum(gains,Fraction(0))
    count_delta = sum(value >= Fraction(1,200) for value in gains)
    omega90 = sum(fraction(row['branches']['action_oracle_atmostK_Omega_common_C']) >= Fraction(9,10) for row in diagnostics)
    C90 = sum(fraction(row['branches']['action_oracle_atmostK_C']) >= Fraction(9,10) for row in diagnostics)
    conditions = dict(sum_delta_at_least_374_over200=total >= Fraction(374,200),
        at_least20_pairs_delta_ge_1_over200=count_delta >= 20, iou_ge_9_over10_paircount_gain_at_least4=C90-omega90 >= 4)
    return dict(schema='N15-fixed-E24-action-opportunity-population-v1', matched_count=374, unmatched_gt_count=6,
        unmatched=unmatched, unmatched_semantics='UNEVALUABLE_NO_QUERY_NOT_ZERO_IOU_EXCLUDED_FROM374_MEAN',
        primary_B=dict(sum_delta_fraction=rational(total), mean_delta_fraction=rational(total/374),
            pairs_delta_ge_1_over200=count_delta, Omega_oracle_pairs_iou_ge_9_over10=omega90,
            C_oracle_pairs_iou_ge_9_over10=C90, high_iou_paircount_gain=C90-omega90,
            GO_conditions=conditions, all_conditions_required=True,
            diagnostic_decision='GO_PENDING_INDEPENDENT_CPU' if all(conditions.values()) else 'STOP_THIS_FIXED_RECIPE_PENDING_INDEPENDENT_CPU'),
        secondary_A=dict(sum_gain_fraction=rational(sum(secondary,Fraction(0))), mean_gain_fraction=rational(sum(secondary,Fraction(0))/374), GO_gate=False),
        scope='Privileged fixed374 IoU action opportunity only; no AP, selector learnability, innovation or general method failure claim')

def tree_equal(torch, left, right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.device.type == right.device.type == 'cpu' and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left,right)
    if type(left) is dict:
        return type(right) is dict and set(left) == set(right) and all(tree_equal(torch,v,right[k]) for k,v in left.items())
    if type(left) in (list,tuple):
        return type(right) is type(left) and len(left) == len(right) and all(tree_equal(torch,a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left == right

def tensor_bytes(torch, value):
    if isinstance(value,torch.Tensor):
        require(value.device.type == 'cpu' and not value.requires_grad and value.layout == torch.strided and
            value.dtype in (torch.float32,torch.int64,torch.uint8,torch.bool) and bool(torch.isfinite(value).all()), 'Capture must be CPU-only finite exact dtype')
        return value.numel()*value.element_size()
    if type(value) is dict: return sum(tensor_bytes(torch,v) for v in value.values())
    if type(value) in (list,tuple): return sum(tensor_bytes(torch,v) for v in value)
    return 0

class CaptureSink:
    def __init__(self, torch, root, budget, process, check_deadline):
        self.torch,self.root,self.budget,self.process,self.check_deadline = torch,Path(root).resolve(strict=True),budget,process,check_deadline
        self.refs,self.payload_bytes,self.serialized_bytes = [],0,0
    def write(self, name, packet):
        self.check_deadline()
        target=(self.root/name).resolve(strict=False)
        require(target.parent == self.root and not target.exists(), 'New real image capture filename must remain within ROOT output')
        payload=tensor_bytes(self.torch,packet)
        require(payload == packet['raw_tensor_payload_bytes'] and payload <= self.budget['maximum_image_raw_tensor_payload_bytes'] and
            self.payload_bytes+payload <= self.budget['raw_tensor_payload_upper_bound_bytes'] and
            self.process.memory_info().rss+payload <= self.budget['maximum_process_RSS_bytes'], 'Prelocked actual capture/RSS budget exceeded')
        with target.open('xb') as stream: self.torch.save(packet,stream)
        replay=self.torch.load(str(target),map_location='cpu')
        require(tree_equal(self.torch,packet,replay), 'Exact real whole-image capture saved readback differs')
        del replay
        ref=reference(target)
        self.payload_bytes+=payload
        self.serialized_bytes+=ref['bytes']
        require(self.serialized_bytes <= self.budget['raw_tensor_payload_upper_bound_bytes']+self.budget['serialized_capture_overhead_budget_bytes'],
            'Prelocked serialized capture disk budget exceeded')
        self.refs.append(ref)
        return ref

def validate_budget(spec,coco,ids,rows,paired,process,psutil,output):
    budget=spec['capture_budget']
    fields={'total_original_pixels','total_pair_pixels','total_K_points','total_GT_bitset_bytes','raw_tensor_payload_upper_bound_bytes',
        'maximum_image_raw_tensor_payload_bytes','serialized_capture_overhead_budget_bytes','auxiliary_output_budget_bytes',
        'minimum_free_output_bytes','required_available_RAM_bytes','maximum_process_RSS_bytes','capture_readback_peak_RAM_upper_bound_bytes',
        'union_points_upper_bound','C_points_upper_bound'}
    require(type(budget) is dict and set(budget)==fields and all(type(v) is int and v>=0 for v in budget.values()), 'ROOT must prelock concrete14 count/disk/RAM budget fields')
    Ns=[coco.imgs[i]['height']*coco.imgs[i]['width'] for i in ids]
    counts=[len(paired[i]) for i in ids]
    Ks=[sum(row['band_counts_all100'][pair['query_id']] for pair in paired[row['image_id']]) for row in rows]
    gtbytes=sum(m*((N+7)//8) for N,m in zip(Ns,counts))
    totalpair=sum(N*m for N,m in zip(Ns,counts))
    raw=1032*sum(Ns)+21*totalpair+16*sum(Ks)+187*102400+gtbytes
    maximum=max(1032*N+21*m*N+16*K+102400+m*((N+7)//8) for N,m,K in zip(Ns,counts,Ks))
    require(budget['total_original_pixels']==budget['union_points_upper_bound']==sum(Ns) and
        budget['total_pair_pixels']==budget['C_points_upper_bound']==totalpair and budget['total_K_points']==sum(Ks) and
        budget['total_GT_bitset_bytes']==gtbytes and budget['raw_tensor_payload_upper_bound_bytes']==raw and
        budget['maximum_image_raw_tensor_payload_bytes']==maximum and budget['capture_readback_peak_RAM_upper_bound_bytes']>=2*maximum and
        budget['maximum_process_RSS_bytes']>=budget['capture_readback_peak_RAM_upper_bound_bytes'] and
        budget['required_available_RAM_bytes']>=max(16*1024**3,budget['capture_readback_peak_RAM_upper_bound_bytes']) and
        budget['minimum_free_output_bytes']>=raw+budget['serialized_capture_overhead_budget_bytes']+budget['auxiliary_output_budget_bytes']+2*1024**3 and
        psutil.disk_usage(str(output.parent)).free>=budget['minimum_free_output_bytes'] and
        psutil.virtual_memory().available>=budget['required_available_RAM_bytes'] and process.memory_info().rss<budget['maximum_process_RSS_bytes'],
        'Concrete paired N/K, conservative U/C, two-packet readback, total-process RAM/free disk budget differs')
    return dict(prelocked=budget, U_upper_is_N=True, C_upper_is_pair_pixels=True, raw_tensor_payload_upper_bound_bytes=raw,
        whole_image_readback_copies=2, capture_files=187, total_process_peak_RAM_certified=False)

def forward_points(torch,head,F_original,e,native_L,indices,check_deadline):
    residual=torch.empty(len(indices),dtype=torch.float32,device='cpu')
    combined=torch.empty_like(residual)
    feature=F_original.flatten(1)
    calls=0
    for first in range(0,len(indices),2048):
        check_deadline()
        idx=torch.from_numpy(indices[first:first+2048]).to(device=native_L.device,dtype=torch.long)
        native=native_L.flatten()[idx]
        output=head(feature[:,idx].T,e.expand(len(idx),256),native,native_L.new_zeros((len(idx),49)),arm='PR')
        u=output['residual']
        require(u.dtype==torch.float32 and u.device==native_L.device and u.shape==native.shape and bool(torch.isfinite(u).all()),
            'Original head must emit finite actual ungated PR residual')
        logits=native+u
        residual[first:first+len(idx)].copy_(u.detach().cpu())
        combined[first:first+len(idx)].copy_(logits.detach().cpu())
        calls+=1
        del idx,native,output,u,logits
    return residual,combined,calls

def overlap_check(torch,C_u,C_logits,positions,Omega_u,Omega_logits):
    mapped=torch.from_numpy(positions)
    uerr=(C_u[mapped]-Omega_u).abs()
    lerr=(C_logits[mapped]-Omega_logits).abs()
    sign=(C_logits[mapped]>0)!=(Omega_logits>0)
    return dict(atol=1e-4,rtol=0.,absolute_not_certified_error_bound=True,
        residual_max_abs=float(uerr.max().item()) if len(uerr) else 0.,
        combined_max_abs=float(lerr.max().item()) if len(lerr) else 0.,
        residual_outside_absolute_range=int((uerr>1e-4).sum().item()),
        combined_outside_absolute_range=int((lerr>1e-4).sum().item()), sign_differences=int(sign.sum().item()),
        qualified=bool((uerr<=1e-4).all() and (lerr<=1e-4).all() and not sign.any()))

def run(spec_path):
    started=time.monotonic()
    path=Path(spec_path).resolve(strict=True)
    spec=read(path)
    gate=source_gate(path,spec)
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh ROOT-owned absolute output required')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1], 'Actual CPU01 required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified cold Torch runtime required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and scipy.__version__=='1.10.1', 'Qualified numerical versions required')
    context,evaluation=gate['context'],gate['evaluation']
    heads,states,proofs=evaluation.load_heads(torch,context,gate)
    n10=load(context['N10_evaluation_source'],'_N15_original_N10')
    reader=load(context['reader_source'],'_N15_original_native_reader')
    renderer=load(context['renderer_source'],'_N15_original_renderer')
    kernels=n10.original_export_kernels(context['export_source'])
    coco=COCO(str(closed(gate['closure'],context['DEV'])))
    ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and ids==[row['id'] for row in coco.dataset['images']] and len(coco.anns)==380 and set(coco.cats)=={1}, 'Historical fullDEV187/GT380/category1 required')
    native=read(closed(gate['closure'],context['native_reference_predictions']))
    native_pop=reader.population(coco,native,ids)
    logref=gate['carrier_report']['predictions'][CARRIER_LABEL]
    logrows=read(closed(gate['closure'],logref))
    logpop=reader.population(coco,logrows,ids)
    require(len(native)==len(logrows)==18700 and all({k:v for k,v in a.items() if k!='segmentation'}==
        {k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(native,logrows)), 'Original all100 scores/classes/query order differs')
    rows=read(closed(gate['closure'],gate['prior']['image_rows']))
    selectors=gate['selector']['manifest']['rows']
    require(len(rows)==len(selectors)==len(gate['packets'])==187 and [row['image_id'] for row in rows]==ids, 'All187 canonical packet/selector order required')
    paired={iid:[row for row in gate['fixed']['rows'] if row['image_id']==iid and row['query_id'] is not None] for iid in ids}
    unmatched=[dict(image_id=row['image_id'],gt_id=row['gt_id'],status='UNEVALUABLE_NO_QUERY') for row in gate['fixed']['rows'] if row['query_id'] is None]
    require(sum(len(values) for values in paired.values())==374 and len(unmatched)==6, 'Complete fixed population images differ')
    budget=validate_budget(spec,coco,ids,rows,paired,process,psutil,output)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    fraction_ref=reference(inspect.getsourcefile(Fraction))
    require(fraction_ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(fraction_ref['path'])==fraction_ref['sha256'], 'Actual stdlib Fraction source must be newly admitted')
    libraries['stdlib_Fraction']=fraction_ref
    require(not torch.cuda.is_initialized(), 'All RAW/source/native/head/library/budget checks precede CUDA')
    torch.cuda.init()
    require(torch.cuda.current_device()==0, 'GPU0 only')
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    outside=[torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone()]
    head=heads['LOG_PR'].to('cuda:0')
    output.mkdir()
    (output/'CAPTURES').mkdir()
    check=lambda: deadline(started,spec)
    sink=CaptureSink(torch,output/'CAPTURES',spec['capture_budget'],process,check)
    binding=dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        locked_protocol=spec['locked_protocol'],carrier_CPU_qualification=spec['carrier_CPU_qualification'],
        carrier_GPU_report=spec['carrier_GPU_report'],head_checkpoint=context['endpoint']['checkpoints']['LOG_PR'],fixed_population=spec['fixed_population'])
    write(output/'START.json',dict(schema='N15-fixed-E24-action-opportunity-start-v1',**binding,
        head_state_proofs=proofs,protocol=gate['protocol'],capture_budget=budget,privileged_GT_diagnostic=True,
        no_TEST=True,AP_evaluated=False,training_started=False,optimizer_created=False,native_model_loaded=False))
    diagnostics,image_checks,capture_rows=[],[],[]
    totalC=totalK=old_calls=C_calls=0
    costs={name:0. for name in ('native','selector','feature','domain','original_Omega_PR','common_C_PR','oracle_statistics','capture_inclusive_inputs_PR_oracle')}
    with torch.no_grad(),(output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index,(row,ref,selector_row) in enumerate(zip(rows,gate['packets'],selectors)):
            check()
            require(process.memory_info().rss<spec['capture_budget']['maximum_process_RSS_bytes'], 'Actual process RSS exceeded prelock')
            tick=time.monotonic()
            packet=torch.load(str(closed(gate['closure'],ref)),map_location='cpu')
            fields,identity,qids=n10.packet_check(torch,reader,coco,packet,row,index,ref,gate)
            F=fields['F256'].to('cuda:0')
            e=fields['e0'].to('cuda:0')
            M=fields['M200'].to('cuda:0')
            cls=fields['final_class_logits'].to('cuda:0')
            require(torch.equal(torch.einsum('bqc,bchw->bqhw',e,F),M), 'Original native M=eF exact GPU projection differs')
            L=renderer.render_original(torch,M,identity)
            require(L.dtype==torch.float32 and L.device.type=='cuda' and tuple(L.shape)==(100,)+tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),
                'Original natural100 GPU L shape/dtype differs')
            top,order=kernels.order(torch,cls,types.SimpleNamespace(num_classes=1))
            require(order.cpu().tolist()==qids, 'Original sorted=False all100 query order differs')
            binary,quality,scores=kernels.quality(torch,L[order],top)
            for ordinal,qid in enumerate(qids):
                actual=dict(image_id=row['image_id'],query_id=qid,category_id=1,score=float(scores[ordinal].item()),
                    segmentation=n10.segmentation(np,masks,binary[ordinal].cpu().numpy().astype(np.uint8)))
                require(actual==native_pop[row['image_id']][ordinal], 'Original all100 native mask/score/class/order differs')
            torch.cuda.synchronize(0)
            costs['native']+=time.monotonic()-tick
            tick=time.monotonic()
            checked=evaluation.selector_packet(torch,context,gate,index,row,ref,identity,qids,L,None)
            require(checked==selector_row, 'All100 exact original frozen GPU selector differs')
            costs['selector']+=time.monotonic()-tick
            tick=time.monotonic()
            F_original=n10.original_feature(torch,renderer,F,identity)
            require(F_original.dtype==torch.float32 and F_original.device==L.device and tuple(F_original.shape)==(256,)+tuple(identity['original_shape']) and bool(torch.isfinite(F_original).all()),
                'Original GPU feature renderer ABI differs')
            torch.cuda.synchronize(0)
            costs['feature']+=time.monotonic()-tick
            F=M=cls=top=order=binary=quality=scores=None
            tick=time.monotonic()
            domains={}
            union=np.zeros(L.shape[1]*L.shape[2],dtype=np.bool_)
            for pair in paired[row['image_id']]:
                qid=pair['query_id']
                P,Omega,C,idx,positions=domains_from_native(np,ndimage,L[qid].detach().cpu().contiguous().numpy())
                require(len(positions)==row['band_counts_all100'][qid], 'Physical old Omega K differs from closed original population')
                union[idx]=True
                domains[qid]=(P,Omega,C,idx,positions)
            union_idx=np.flatnonzero(union).astype(np.int64)
            del union
            costs['domain']+=time.monotonic()-tick
            tick=time.monotonic()
            sampled=torch.empty((len(union_idx),256),dtype=torch.float32,device='cpu')
            feature=F_original.flatten(1)
            for first in range(0,len(union_idx),2048):
                check()
                idx_gpu=torch.from_numpy(union_idx[first:first+2048]).to(L.device)
                block=feature[:,idx_gpu].T.detach().cpu().contiguous()
                sampled[first:first+len(idx_gpu)].copy_(block)
                del idx_gpu,block
            shared=dict(dataset_index=index,image_id=row['image_id'],identity=identity,native_packet=ref,selector=selector_row,
                union_flat_indices=torch.from_numpy(union_idx),sampled_F=sampled,e0=e.detach().cpu().contiguous())
            del sampled,feature,union_idx
            capture_pairs=[]
            failed=[]
            image_C=image_K=0
            loglookup={item['query_id']:item for item in logpop[row['image_id']]}
            for pair in paired[row['image_id']]:
                check()
                qid=pair['query_id']
                P,Omega,C,idx,positions=domains[qid]
                omega_idx=idx[positions]
                tick2=time.monotonic()
                old_u,old_logits,calls=forward_points(torch,head,F_original,e[0,qid],L[qid],omega_idx,check)
                old_calls+=calls
                costs['original_Omega_PR']+=time.monotonic()-tick2
                tick2=time.monotonic()
                C_u,C_logits,calls=forward_points(torch,head,F_original,e[0,qid],L[qid],idx,check)
                C_calls+=calls
                costs['common_C_PR']+=time.monotonic()-tick2
                overlap=overlap_check(torch,C_u,C_logits,positions,old_u,old_logits)
                original=P.copy()
                original.ravel()[omega_idx]=(old_logits>0).numpy()
                expected=masks.decode(reader.rle_bytes(loglookup[qid]['segmentation'])).astype(np.bool_)
                exact_original=np.array_equal(original,expected)
                require(coco.anns[pair['gt_id']]['image_id']==row['image_id'], 'Fixed GT annotation image identity differs')
                G=masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(np.bool_)
                require(G.shape==P.shape and bool(G.any()), 'Actual matched GT must be nonempty and original-shape aligned')
                refined_exact=confusion(np,original,G)==fixed_confusion(pair['refined'])
                overlap.update(original_Omega_full_binary_exact=bool(exact_original), original_Omega_refined_counts_exact=bool(refined_exact))
                overlap['qualified']=overlap['qualified'] and bool(exact_original) and bool(refined_exact)
                A=P.copy()
                A.ravel()[idx]=(C_logits>0).numpy()
                tick2=time.monotonic()
                if overlap['qualified']:
                    diagnostic=describe_pair(np,gate['count_oracle'],gate['array_oracle'],P,G,A,original,Omega,C,pair,overlap)
                    diagnostics.append(diagnostic)
                else:
                    diagnostic=dict(schema=PAIR_SCHEMA,identity={key:pair[key] for key in ('image_id','gt_id','query_id')},
                        status='NUMERICAL_QUALIFICATION_FAILURE',batch_compatibility=overlap,scientific_STOP=False,GO_evaluated=False)
                    failed.append(diagnostic)
                costs['oracle_statistics']+=time.monotonic()-tick2
                native_selected=L[qid].flatten()[torch.from_numpy(idx).to(L.device)].detach().cpu().contiguous()
                GT_bits=torch.from_numpy(np.packbits(G.ravel(),bitorder='little'))
                capture_pairs.append(dict(identity={key:pair[key] for key in ('image_id','gt_id','query_id')},
                    native_counts=confusion(np,P,G),C_indices=torch.from_numpy(idx),Omega_C_positions=torch.from_numpy(positions),
                    GPU_native_L=native_selected,GPU_C_residual=C_u,GPU_C_combined=C_logits,GPU_C_positive=(C_logits>0),
                    GPU_Omega_residual=old_u,GPU_Omega_combined=old_logits,GT_packbits=GT_bits,
                    GT_bits_count=int(G.size),GT_bitorder='little',diagnostic=diagnostic,batch_compatibility=overlap))
                image_C+=len(idx)
                image_K+=len(positions)
                del P,Omega,C,idx,positions,omega_idx,old_u,old_logits,C_u,C_logits,original,expected,G,A,native_selected,GT_bits,diagnostic
            totalC+=image_C
            totalK+=image_K
            torch.cuda.synchronize(0)
            payload=1032*len(shared['union_flat_indices'])+21*image_C+16*image_K+102400+sum(item['GT_packbits'].numel() for item in capture_pairs)
            image_packet=dict(schema=CAPTURE_SCHEMA,binding=binding,shared=shared,pairs=capture_pairs,
                natural_query_ids=list(range(100)),native_export_query_ids=qids,native_all100_exact=True,selector_exact=True,
                U=len(shared['union_flat_indices']),C_points=image_C,K_points=image_K,raw_tensor_payload_bytes=payload,
                candidate_generated_without_GT=True,all_pr_pairs_prelocked=True,independent_CPU_qualified=False,current_terminal_self_certified=False)
            capture_ref=sink.write('IMAGE_%03d_IMAGE%d.pth'%(index,row['image_id']),image_packet)
            costs['capture_inclusive_inputs_PR_oracle']+=time.monotonic()-tick
            image_record=dict(dataset_index=index,image_id=row['image_id'],capture=capture_ref,pairs=[item['identity'] for item in capture_pairs],
                U=image_packet['U'],C_points=image_C,K_points=image_K,raw_tensor_payload_bytes=payload)
            capture_rows.append(image_record)
            image_check=dict(dataset_index=index,image_id=row['image_id'],native_all100_exact=True,selector_exact=True,
                matched_pairs=len(capture_pairs),capture=capture_ref,numerical_qualification_passed=not failed)
            image_checks.append(image_check)
            trace.write(json.dumps(image_check,allow_nan=False)+'\n')
            trace.flush()
            if failed:
                write(output/'NUMERICAL_QUALIFICATION_FAILURE.json',dict(schema='N15-fixed-action-numerical-qualification-failure-v1',**binding,
                    capture=capture_ref,failures=failed,scientific_STOP=False,GO_evaluated=False,automatic_retry=False,
                    rule='absolute1e-4 residual/combined and strict sign; no posterior relaxation; original strong E24 remains the carrier'))
                raise RuntimeError('NUMERICAL_QUALIFICATION_FAILURE; preserved actual GPU capture; not scientific STOP; ROOT forensic review required')
            del packet,fields,L,F_original,e,domains,shared,capture_pairs,image_packet,loglookup
            gc.collect()
            print(json.dumps(dict(N15_native_images=index+1,matched_pairs=len(diagnostics),C_points=totalC)),flush=True)
    population=summarize(diagnostics,unmatched)
    require(all(torch.equal(v.cpu(),states['LOG_PR'][key]) for key,v in head.state_dict().items()) and
        all(not p.requires_grad and p.grad is None for p in head.parameters()) and
        torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),
        'Frozen original E24 LOG head/grad/RNG changed')
    peak=torch.cuda.max_memory_allocated(0)
    head.cpu()
    heads.clear()
    del heads,head
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2, 'Actual4GiB allocator cap/cleanup required')
    require(len(sink.refs)==len(capture_rows)==len(image_checks)==187 and len(diagnostics)==374 and
        totalK==spec['capture_budget']['total_K_points'], 'Complete187 captures/374pairs/original K sum required')
    write(output/'POPULATION.json',population)
    write(output/'PAIR_DIAGNOSTICS.json',diagnostics)
    write(output/'IMAGE_CHECKS.json',image_checks)
    manifest=dict(schema=CAPTURE_MANIFEST_SCHEMA,**binding,capture_schema=CAPTURE_SCHEMA,images=187,matched_pairs=374,
        unmatched_GT=6,rows=capture_rows,actual_raw_tensor_payload_bytes=sink.payload_bytes,
        actual_serialized_capture_bytes=sink.serialized_bytes,capture_budget=budget,actual_GPU_inputs_no_CPU_rerender=True)
    write(output/'CAPTURE_MANIFEST.json',manifest)
    for ref in spec['sources']:
        check()
        bound(ref)
    current_libs=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    current_libs['stdlib_Fraction']=reference(inspect.getsourcefile(Fraction))
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['GPU_admission_file'])==gate['admission_ref'] and current_libs==libraries and
        process.cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2, 'Final actual source/library/resource binding changed')
    gate['lineage'].final_endpoint_check(gate['parent'])
    validate_protocol(spec,gate['closure'])
    check()
    report=dict(schema=REPORT_SCHEMA,status=STATUS,**binding,RAW_inputs=spec['RAW_inputs'],normal_RAW_count=spec['normal_RAW_count'],
        terminal_identity_pins=spec['terminal_identity_pins'],additional_preserved_independent_CPU_failure=spec['additional_preserved_independent_CPU_failure'],
        total_failed_RAW_count=9,total_terminal_identity_pin_count=spec['normal_RAW_count']+9,
        carrier_refs={key:spec[key] for key in CARRIER_KEYS},count_oracle_source=spec['count_oracle_source'],array_oracle_source=spec['array_oracle_source'],
        head_state_proofs=proofs,carrier_seed=202610120701,carrier_epoch=24,arm='LOG_PR',parameters=40497,parameter_tensors=8,
        images=187,matched_pairs=374,unmatched_GT=6,raw_GT=380,unmatched_unevaluable_not_zero=True,
        native_all18700_exact=True,selector_all18700_exact=True,original_Omega_full_binary_all374_exact=True,
        original_Omega_refined_counts_all374_exact=True,C_Omega_absolute_1e4_and_strict_sign_all374_exact=True,
        main_comparison=gate['protocol']['main_comparison'],secondary_comparison=gate['protocol']['secondary_comparison'],
        diagnostic_decision=population['primary_B']['diagnostic_decision'],GO_is_privileged_pending_independent_CPU=True,
        population=reference(output/'POPULATION.json'),pair_diagnostics=reference(output/'PAIR_DIAGNOSTICS.json'),
        image_checks=reference(output/'IMAGE_CHECKS.json'),capture_manifest=reference(output/'CAPTURE_MANIFEST.json'),capture_refs=sink.refs,
        capture_schema=CAPTURE_SCHEMA,capture_budget=budget,logical_C_points=totalC,logical_Omega_K_points=totalK,
        actual_PR_chunk_calls=dict(original_Omega=old_calls,common_C=C_calls),cost_seconds=costs,
        cost_components_are_not_additive=True,
        capture_cost_scope='capture_inclusive_inputs_PR_oracle includes sampled input gathering, both PR paths, pair checks, oracle calculation and save/readback; overlaps PR/oracle components',
        actual_raw_capture_tensor_bytes=sink.payload_bytes,actual_serialized_capture_bytes=sink.serialized_bytes,
        packet_refs=gate['packets'],selector_manifest=gate['selector']['manifest_ref'],native_reference_predictions=context['native_reference_predictions'],
        original_E24_LOG_predictions=logref,DEV=context['DEV'],start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),
        total_elapsed_seconds=time.monotonic()-started,peak_allocator_bytes=peak,cleanup=cleanup,actual_imported_files=libraries,
        independent_CPU_qualified=False,current_owned_GPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        AP_evaluated=False,training_started=False,optimizer_created=False,native_model_loaded=False,no_TEST=True,no_cross_dataset=True,
        task_efficacy_established=False,innovation_established=False,
        evidence_limits='Fixed strongest701 LOG E24/radius8 frozen actual PR actions and fixed374 native .5 pairs; six unmatched unevaluable. '
            'GT privileged conditional IoU action opportunity, equal K not equal compute. GO permits a future selector hypothesis only, '
            'not AP success, learnability, causal attribution, innovation/TGRS readiness or general method failure. ROOT actual terminal and independent CPU still required.')
    write(output/'REPORT.json',report)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'),decision=report['diagnostic_decision'],independent_CPU_qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args()
    require(args.owner_execute,'UNBOUND_DRAFT: ROOT must bind full103 history, locked protocol, actual carrier and budget before one owned attempt')
    run(args.spec)
