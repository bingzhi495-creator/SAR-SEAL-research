"""UNBOUND_DRAFT: standalone N17 fixedDEV LOCAL/CENTER ERROR GPU shard entry.
Copied immutable ordinary RAW9..19/carrier front; controlled20/independent CPU SOURCE-failure21 retained; ROOT owns binding/execution.
No actual scientific source/payload was read or executed by this author.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 21 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 21 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 stays separate; CPU03 includes explicit failure21 and normal+21 pins only')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9..20 and independent CPU SOURCE21 already passed RAW-only checks.
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
    # FIRST HASH: failed9..20, explicit typed21, and every normal/legacy failed8 passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(closure.get(row[key]['path']) == row[key] for row in rows for key in RAW_KEYS if key != 'phase') and
            closure.get(failed_ref['path']) == failed_ref and all(closure.get(row['receipt']['path']) == row['receipt'] for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N17-fixedDEV-local-center-error-GPU-shard-spec-v1"
CPU_SCHEMA = "ROOT-N18-fixedDEV-matched374-routing-diagnostic-CPU-spec-v1"
PRIOR_CPU_SCHEMA = "ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v2"
CARRIER_KEYS = tuple(["carrier_GPU_spec","carrier_GPU_report","carrier_GPU_RAW","carrier_GPU_admission","carrier_CPU_spec","carrier_CPU_source","carrier_CPU_qualification","carrier_CPU_RAW","carrier_CPU_admission","carrier_evaluation_source"])
EXPECTED_PROTOCOL = json.loads("{\"schema\":\"N16-two-scout-fixed-E24-FIT-selector-locked-v1\",\"locked_UTC\":\"2026-10-14T06:02:44.021Z\",\"locked_before_N16_actual_FIT_cache\":true,\"teacher\":\"qualified native0 E24\",\"fixed_PR\":{\"seed\":202610120701,\"epoch\":24,\"parameters\":40497,\"strong_DEV_AP\":68.09938494745545},\"population\":{\"FIT_images\":741,\"FIT_instances\":1661,\"natural_queries_per_image\":100,\"DEV_images\":187,\"DEV_natural_queries\":18700},\"targets\":{\"positive\":\"last original native matcher assigned raw instance; original mask_weight1\",\"negative\":\"explicit NEW common background-emptyset proxy, only original sampler.neg_inds with background class and mask_weight0\",\"ignored\":\"never assigned empty; invalid target excluded\",\"no_extra_annotation\":true},\"candidate\":{\"P\":\"actual original GPU L>0\",\"Omega\":\"abs(actual original GPU L)<=2\",\"C\":\"Omega union P union Chebyshev radius8 dilation(P)\",\"K\":\"count(Omega)\",\"query_filter\":\"none by score or GT\",\"K0\":\"no edit; no training candidate samples\"},\"sampler\":{\"point_seed_base\":202610140160000,\"matching_loss_seed_base\":202610122001,\"max_points_per_image\":4096,\"proposal\":\"2048 per active matched/background stratum; 4096 if only one; uniform active query then uniform C point, replacement\",\"target_distribution\":\"uniform valid active query within image, then uniform its C point; each image one equal update\",\"importance\":\"(active_query_count_in_stratum/all_active_query_count)/proposal_stratum_probability\",\"loss\":\"mean(weight_i * cross_entropy_i); no per-batch sum-weight renormalization or additional class weights\",\"algorithm\":{\"RNG\":\"Torch CPU private Generator seeded point_seed_base+dataset_index\",\"group_order\":[\"matched\",\"background\"],\"calls\":\"per existing group one randint(n_active_queries,(count,)); then one randint(|C_query|,(1,)) per sampled draw in draw order\",\"sample_order\":\"matched samples followed by background samples; replacement duplicates preserved\",\"weights_dtype\":\"float32\",\"GT_point_sampling\":\"only matcher stratum; no utility/help/harm or pixel GT resampling\"}},\"scout\":{\"inputs\":\"nonaffine LN(F256) and LN(e256), eps1e-5; unclipped raw L/8\",\"layers\":[513,32,3],\"activation\":\"ReLU\",\"parameters\":16547,\"tensors\":4,\"paired_identical_initial_state\":true,\"normal_Linear_init\":true,\"utility_classes\":[\"help\",\"harm\",\"noop\"],\"error_classes\":[\"FP\",\"FN\",\"correct\"]},\"training\":{\"seed\":202610140161,\"order_seed\":202610140160001,\"epochs\":24,\"curve_endpoints\":[8,24],\"AdamW_lr\":0.0003,\"AdamW_weight_decay\":0.0001,\"AdamW_betas\":[0.9,0.999],\"AdamW_eps\":1e-8,\"LR_schedule\":\"constant\",\"update_unit\":\"one same cached image batch per arm; same order/points/weights\",\"empty_image\":\"both skip; count actual updates\",\"mixed_precision\":false,\"gradient_clip\":null},\"selection\":{\"canonical\":\"FP32 scout logits transferred to CPU, stable NumPy FP64 softmax/score\",\"utility\":\"p_help-p_harm\",\"error\":\"2*(p_FP+p_FN)-1\",\"main\":\"atmost K, strictly score>0\",\"supplement\":\"exact K real points only\",\"ties\":[\"score descending\",\"abs native L ascending\",\"flat index ascending\"],\"rank_tolerance\":0,\"qualification\":\"selected indices exact; no allclose substitution\"},\"main_GATE\":{\"metric\":\"official COCO mask AP full natural all100 at fixed E24 scout endpoint; all original class/S0/order unchanged\",\"reference\":\"max(error_same24_AP,68.09938494745545)\",\"GO\":\"utility-minus-reference >=1 AP point and utility>paired error\",\"nonpositive\":\"stop current recipe, no extra budget/seed\",\"positive_under1\":\"at most one prelocked extra seed then original goal repeat rule\",\"no_DEV_epoch_or_threshold_selection\":true},\"strong_controls\":[\"original strong fixed E24\",\"same-trained error scout\",\"dense error+actual_flip indicator: delta*(2p_error-1)\",\"adapted two-class MagNet: 2sigmoid(-absL)*abs(tanh((L+u)/2))\"],\"cost\":{\"canonical_unshared_first\":true,\"shared_projection\":\"future engineering only after both-arm equal numeric qualification\",\"all100_include_background\":true,\"count_all_C_PR_for_dense_controls\":true,\"report_label_generation_separately\":true},\"numeric\":{\"absolute\":0.0001,\"relative\":0,\"actual_after_sign_exact\":true},\"execution\":{\"ROOT_single_leaf\":true,\"CPU_affinity\":[0,1],\"threads\":2,\"allocator_MiB\":4096,\"whole_GPU_observed_MiB\":8192,\"deadline_seconds\":1800,\"automatic_retry\":false},\"claims\":{\"training_started\":false,\"new_AP\":false,\"new_primitive\":false,\"TGRS_ready\":false}}")

def write(path, value):
    with Path(path).open("x",encoding="utf8",newline="\n") as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n")

def validate_protocol(spec,closure):
    protocol=read(closed(closure,spec["locked_protocol"]))
    require(protocol==EXPECTED_PROTOCOL,"Exact locked N16 protocol differs")
    return protocol


def tenth_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_cache_failure']
    require(extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path']!=spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
        'Original N16 failed owner10 must remain separate from normal and old failure owners')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw


def eleventh_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_cache_target_dtype_failure']
    require(extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in (spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_failure']['RAW_ref']['path']),
        'Original N16 target-dtype failed owner11 must remain separate')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw

"""UNBOUND pure canonical controlled-failure RAW predicate; no I/O or hash.

ROOT must read and bind the actual record/pin, prove original ownership and
record locator phase, and invoke this BEFORE any Source read/hash. This helper
does not supply those proofs or qualify partial caches. Its embedded tests use
only invented dictionaries and do not establish a scientific execution result.
"""
import math

ABORT = 'QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE'
GPU_SCOPE = 'whole GPU observed, not per-process WDDM allocation'
MIN_RAM_BYTES = 16 * 1024**3
GPU_LIMIT_MIB = 8192


class RawGateError(ValueError):
    pass


def helpercheck(raw, pin):
    """Check canonical owned controlled-failure structure; allow extra fields.

    pin.phase is a record-locator obligation of the ROOT wrapper: the canonical
    actual RAW has no phase field. No synthetic field may be added to obtain a
    pass. This function never opens a record, computes SHA, or reads Source.
    """
    def need(condition, field):
        if not condition:
            raise RawGateError('Controlled resource RAW gate failed: '+field)

    def field(obj,key):
        need(type(obj) is dict and key in obj,key+' missing')
        return obj[key]

    def finite_number(value):
        return type(value) is int or (type(value) is float and math.isfinite(value))

    def same_json(left,right):
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            return (all(type(k) is str for k in left) and set(left)==set(right)
                    and all(same_json(left[k],right[k]) for k in left))
        if type(left) is list:
            return len(left)==len(right) and all(same_json(a,b) for a,b in zip(left,right))
        if type(left) is float:
            return math.isfinite(left) and math.isfinite(right) and left==right
        return type(left) in (str,int,bool,type(None)) and left==right

    need(type(raw) is dict and type(pin) is dict,'raw/pin dictionaries')
    need(field(raw,'status')=='ACTUAL_OWNED_TERMINAL','status')
    rc=field(raw,'returncode')
    need(type(rc) is int and rc==1,'returncode must be exact failure1')
    need(field(raw,'root_wait_reaped') is True,'root_wait_reaped')
    held=field(raw,'held_handle_exit')
    need(field(held,'wait_state')=='WAIT_OBJECT_0','held wait_state')
    actual_exit=field(held,'actual_exit_code')
    need(type(actual_exit) is int and actual_exit==1,'held actual_exit_code must be failure1')
    birth=field(raw,'numeric_birth_absence')
    need(field(birth,'original_birth_absent') is True,'original birth absence')
    need(field(raw,'error') is None,'error must be None')
    need(field(raw,'abort')==ABORT,'exact controlled resource/deadline abort')
    need(field(raw,'wait_errors') is False,'wait_errors must be False')
    gpu_errors=field(raw,'GPU_observation_errors')
    need(type(gpu_errors) is list and gpu_errors==[],'GPU_observation_errors must be empty list')
    identity,pinned_identity=field(raw,'original_identity'),field(pin,'original_identity')
    need(type(identity) is dict and bool(identity) and type(pinned_identity) is dict
         and bool(pinned_identity) and same_json(identity,pinned_identity),'original_identity pin exact')
    sha,pinned_sha=field(raw,'admission_sha256'),field(pin,'admission_sha256')
    need(type(sha) is str and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
         and type(pinned_sha) is str and sha==pinned_sha,'admission SHA pin exact; no SHA computed')
    if 'returncode' in pin:
        need(type(pin['returncode']) is int and pin['returncode']==1,'optional pin returncode must be failure1')
    if 'phase' in raw and 'phase' in pin:
        need(same_json(raw['phase'],pin['phase']),'optional RAW phase matches pin')
    minimum_ram=field(raw,'minimum_ram_bytes')
    need(finite_number(minimum_ram) and minimum_ram>=MIN_RAM_BYTES,'minimum_ram_bytes >=16GiB')
    peak=field(raw,'whole_GPU_observed_peak_MiB')
    need(finite_number(peak) and 0<=peak<=GPU_LIMIT_MIB,'whole GPU observed peak <=8192MiB')
    limit=field(raw,'whole_GPU_used_limit_MiB')
    need(type(limit) is int and limit==GPU_LIMIT_MIB,'whole GPU limit must remain8192MiB')
    need(field(raw,'whole_leaf_VRAM_certified') is False,'per-leaf VRAM certification must remain False')
    need(field(raw,'GPU_scope')==GPU_SCOPE,'whole GPU observation scope')
    elapsed=field(raw,'elapsed_seconds')
    need(finite_number(elapsed) and 1800<=elapsed<=1805,'controlled resource/deadline elapsed1800..1805')
    return dict(passed=True,normal_success=False,resource_or_deadline_failure=True,
                numeric_or_method_failure_proven=False,partial_caches_qualified=False,
                returncode=1,held_actual_exit_code=1,abort=ABORT,
                minimum_ram_bytes=minimum_ram,whole_GPU_observed_peak_MiB=peak,
                elapsed_seconds=elapsed,phase_locator_proof_required_from_ROOT_wrapper=True,
                record_IO_performed=False,Source_read_or_hash_performed=False)




def twelfth_resource_failure_raw_only(spec):
    extra=spec['additional_preserved_N16_controlled_resource_failure']
    ref_shape(extra['RAW_ref']);p=Path(extra['RAW_ref']['path'])
    require(p.name=='PARENT_RECEIPT.json' and p.parent.name==extra['phase']==extra['pin']['phase'] and
        extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in (spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_target_dtype_failure']['RAW_ref']['path']),
        'Original resource-failed12 locator/pin must remain distinct')
    raw=read(p)
    decision=helpercheck(raw,extra['pin'])
    require(decision['passed'] and decision['normal_success'] is False and decision['partial_caches_qualified'] is False,
        'Controlled failure is preserved, never promoted to scientific cache success')
    return raw

def thirteenth_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_fullDEV_packet_ABI_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in (
            'additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
            'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure')),
        'Original fullDEV packet-ABI failed13 must remain distinct actual1/report-absent owner')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw


def fourteenth_controlled_storage_failure_raw_only(spec):
    extra=spec['additional_preserved_N16_fullDEV_controlled_storage_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True,'Original controlled storage-failed14 exact binding required')
    ref_shape(extra['RAW_ref']);p=Path(extra['RAW_ref']['path'])
    require(p.name=='PARENT_RECEIPT.json' and p.parent.name==extra['phase']==extra['pin']['phase'] and
        extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in (
            'additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
            'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure',
            'additional_preserved_N16_fullDEV_packet_ABI_failure')),
        'Original controlled storage-failed14 locator/owner must remain distinct')
    raw=read(p);decision=helpercheck(raw,extra['pin'])
    require(decision['passed'] and decision['normal_success'] is False and decision['partial_caches_qualified'] is False,
        'Controlled storage/deadline failure14 is not normal0, abortNone or numerical/method success')
    return raw


def fifteenth_controlled_bundle_deadline_failure_raw_only(spec):
    extra=spec['additional_preserved_N16_fullDEV_controlled_bundle_deadline_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True,'Original controlled storage-failed15 exact binding required')
    ref_shape(extra['RAW_ref']);p=Path(extra['RAW_ref']['path'])
    require(p.name=='PARENT_RECEIPT.json' and p.parent.name==extra['phase']==extra['pin']['phase'] and
        extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in (
            'additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
            'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure',
            'additional_preserved_N16_fullDEV_packet_ABI_failure','additional_preserved_N16_fullDEV_controlled_storage_failure')),
        'Original controlled storage-failed15 locator/owner must remain distinct')
    raw=read(p);decision=helpercheck(raw,extra['pin'])
    require(decision['passed'] and decision['normal_success'] is False and decision['partial_caches_qualified'] is False,
        'Controlled storage/deadline failure15 is not normal0, abortNone or numerical/method success')
    return raw



FAILURE16_PRIOR_KEYS=('additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
    'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure',
    'additional_preserved_N16_fullDEV_packet_ABI_failure','additional_preserved_N16_fullDEV_controlled_storage_failure',
    'additional_preserved_N16_fullDEV_controlled_bundle_deadline_failure')


def sixteenth_controlled_CPU_proof_deadline_failure_raw_only(spec):
    extra=spec['additional_preserved_N16_fullDEV_controlled_CPU_proof_deadline_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_Q_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_Q_absent'] is True,'Original CPU controlled failure16 exact10 fields/Qabsence required')
    ref_shape(extra['RAW_ref']);p=Path(extra['RAW_ref']['path'])
    require(p.name=='PARENT_RECEIPT.json' and p.parent.name==extra['phase']==extra['pin']['phase'] and
        extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in FAILURE16_PRIOR_KEYS),
        'Original controlled CPU16 locator/pin/owner distinct from all preserved failures')
    raw=read(p);decision=helpercheck(raw,extra['pin'])
    require(decision['passed'] and decision['normal_success'] is False and decision['partial_caches_qualified'] is False,
        'Controlled CPU16 remains failure1, never partial numeric/Q success')
    return raw


FAILURE17_PRIOR_KEYS=FAILURE16_PRIOR_KEYS+('additional_preserved_N16_fullDEV_controlled_CPU_proof_deadline_failure',)

def seventeenth_final_CPU_reference_ABI_failure_raw_only(spec):
    extra=spec['additional_preserved_N16_final_CPU_reference_ABI_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_Q_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
            extra['original_Q_absent'] is True,'Original finalCPU reference-ABI failure17 exact10 fields/exit1/Qabsence required')
    for key in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):ref_shape(extra[key])
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
            extra['RAW_ref']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']} and
            extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in FAILURE17_PRIOR_KEYS),
            'Original finalCPU17 locator/owner distinct from every normal and preserved failure')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw

FAILURE18_PRIOR_KEYS=FAILURE17_PRIOR_KEYS+('additional_preserved_N16_final_CPU_reference_ABI_failure',)

def eighteenth_firstDEV_reference_ABI_failure_raw_only(spec):
    extra=spec['additional_preserved_N17_firstDEV_reference_ABI_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True,'Original firstDEV reference-ABI failure18 exact10 fields/actual1/report absence')
    for key in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):ref_shape(extra[key])
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']} and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in FAILURE18_PRIOR_KEYS),
        'Failure18 original owner distinct from all normal and preserved failed17')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw

FAILURE19_PRIOR_KEYS=FAILURE18_PRIOR_KEYS+('additional_preserved_N17_firstDEV_reference_ABI_failure',)

def nineteenth_secondDEV_SOURCE_closure_failure_raw_only(spec):
    extra=spec['additional_preserved_N17_secondDEV_SOURCE_closure_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True,'Original secondDEV SOURCE-closure failure19 exact10 fields/actual1/report absence')
    for key in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):ref_shape(extra[key])
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']} and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in FAILURE19_PRIOR_KEYS),
        'Failure19 original owner distinct from all normal and preserved failed18')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    minimum_ram,elapsed=raw['minimum_ram_bytes'],raw['elapsed_seconds']
    require(type(minimum_ram) in (int,float) and math.isfinite(minimum_ram) and minimum_ram>=MIN_RAM_BYTES and
        type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=1800 and
        type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
        raw['error'] is None and raw['abort'] is None,'Failure19 original ordinary terminal RAM/GPU/deadline/no-abort scope')
    return raw

FAILURE20_PRIOR_KEYS=FAILURE19_PRIOR_KEYS+('additional_preserved_N17_secondDEV_SOURCE_closure_failure',)

def twentieth_DEV_GPU_controlled_deadline_failure_raw_only(spec):
    extra=spec['additional_preserved_N17_DEV_GPU_controlled_deadline_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True,'Original controlled DEV GPU20 exact10 fields/failure1/REPORT absence')
    for key in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):ref_shape(extra[key])
    p=Path(extra['RAW_ref']['path'])
    require(p.name=='PARENT_RECEIPT.json' and p.parent.name==extra['phase']==extra['pin']['phase'] and
        extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']} and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in FAILURE20_PRIOR_KEYS),
        'Controlled20 original locator/pin distinct from normal and failed1..19')
    raw=read(p);decision=helpercheck(raw,extra['pin']);check_pin(extra['pin'],raw,extra['phase'],1)
    require(decision['passed'] is True and decision['normal_success'] is False and decision['partial_caches_qualified'] is False,
        'Controlled20 remains original failed1; no partial GPU capture promoted or reused')
    return raw

FAILURE21_PRIOR_KEYS=FAILURE20_PRIOR_KEYS+('additional_preserved_N17_DEV_GPU_controlled_deadline_failure',)

def twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only(spec):
    extra=spec['additional_preserved_N17_independent_CPU_SOURCE_closure_failure']
    keys={'RAW_ref','phase','pin','returncode','admission','phase_binding','bootstrap','entry_source','spec','original_report_absent'}
    require(type(extra) is dict and set(extra)==keys and type(extra['returncode']) is int and extra['returncode']==1 and
        extra['original_report_absent'] is True,'Original independent CPU SOURCE failure21 exact10 fields/actual1/REPORT absence')
    for key in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):ref_shape(extra[key])
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']} and
        extra['RAW_ref']['path'] not in tuple(spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS),
        'Failure21 original owner distinct from all normal and preserved failed1..20')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1);check_pin(extra['pin'],raw,extra['phase'],1)
    require(raw['error'] is None and raw['abort'] is None,'Original CPU SOURCE21 was ordinary exit1, never controlled resource abort')
    return raw

def closed_independent_CPU_SOURCE_failure21(spec,closure,lineage,raw):
    extra=spec['additional_preserved_N17_independent_CPU_SOURCE_closure_failure']
    for key in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,extra[key])
    row=dict(receipt=extra['RAW_ref'],phase=extra['phase'],admission=extra['admission'],phase_binding=extra['phase_binding'],bootstrap=extra['bootstrap'])
    owner=lineage.owner_join(row,raw,closure)
    require(owner['scientific_spec']==extra['spec'] and owner['entry']==extra['entry_source']['path'] and owner['cuda_visible_devices']=='',
        'Preserved failed CPU21 original CPUempty/source/spec/ad/phase/bootstrap exact')
    old=read(closed(closure,extra['spec']));absent=Path(old['output_directory'])/'REPORT.json'
    require(old['schema']==PRIOR_CPU_SCHEMA and old['entry_source']==extra['entry_source'] and old['phase']==extra['phase'] and
        old['cpu_shard_id']==0 and old['cpu_dataset_ordinal_range']==[0,47] and old['cpu_shard_owner_index']==127 and
        old['normal_RAW_count']==old['expected_normal_RAW_count']==127 and old['total_failed_RAW_count']==20 and
        old['total_terminal_identity_pin_count']==147 and old['RAW_inputs']==spec['RAW_inputs'][:127] and
        all(closure.get(r['path'])==r for r in old['sources']) and
        all(spec['terminal_identity_pins'].get(key)==value for key,value in old['terminal_identity_pins'].items()) and
        all(old[key]==spec[key] for key in FAILURE21_PRIOR_KEYS) and not absent.exists() and not absent.is_symlink() and
        extra['original_report_absent'] is True,'Original failed CPU02 v2/127prefix/failed20/pins147/SOURCE as recorded/REPORT absence unchanged')
    # Preserve its incomplete SOURCE as a failure; do not assert the missing adapter belonged to old.sources.
    # Current source_gate still strictly closes every CPU_NUMERIC_REFS, including the adapter.
    return owner

def inherited_source_gate(path, spec, cpu=False):
    require(spec['schema'] == (CPU_SCHEMA if cpu else SCHEMA) and
        type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] >= 109 + int(cpu) and
        spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] and spec['deadline_seconds'] == 1800 and
        spec['automatic_retry'] is False and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
        'New N15 actual full ancestry/schema/owner resource recipe required')
    ninth_failed_raw_only(spec)
    failed10=tenth_failed_raw_only(spec)
    failed11=eleventh_failed_raw_only(spec)
    failed12=twelfth_resource_failure_raw_only(spec)
    failed13=thirteenth_failed_raw_only(spec)
    failed14=fourteenth_controlled_storage_failure_raw_only(spec)
    failed15=fifteenth_controlled_bundle_deadline_failure_raw_only(spec)
    failed16=sixteenth_controlled_CPU_proof_deadline_failure_raw_only(spec)
    failed17=seventeenth_final_CPU_reference_ABI_failure_raw_only(spec)
    failed18=eighteenth_firstDEV_reference_ABI_failure_raw_only(spec)
    failed19=nineteenth_secondDEV_SOURCE_closure_failure_raw_only(spec)
    failed20=twentieth_DEV_GPU_controlled_deadline_failure_raw_only(spec)
    failed21=twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only(spec)
    raws, closure = pre_gate(spec, spec['expected_normal_RAW_count'])
    for key in CARRIER_KEYS + ('entry_source', 'locked_protocol', 'cache_body_source', 'labels_source', 'FIT', 'original_fit_cache_report', 'native_context_spec', 'region_common_source', 'original_fit_cache_entry_source', 'action_diagnostic_source', 'C_engine_source'):
        closed(closure, spec[key])
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    if not cpu:
        require(Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True), 'Current N15 entry differs')
    protocol = validate_protocol(spec, closure)
    result=read(closed(closure,spec['opportunity_ROOT_result']))
    opportunityQ=read(closed(closure,spec['opportunity_CPU_qualification']))
    require(result['status']=='ROOT_N15_FIXED_E24_ACTION_OPPORTUNITY_FULLY_QUALIFIED' and result['GO'] is True and
        result['CPU_Q']==spec['opportunity_CPU_qualification'] and result['CPU_actual_exit_code']==0 and
        result['GPU_actual_exit_code']==0 and opportunityQ['status']=='ROOT_N15_FIXED_E24_ACTION_OPPORTUNITY_SAME_GPU_INPUT_CPU_QUALIFIED_V1' and
        opportunityQ['same_GPU_input_numerical_qualified'] is True, 'Actual qualified opportunity GO required')
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
    require(all(closure.get(ref['path']) == ref for payload in (original, cpu_spec) for ref in payload['sources']),
        'Full original carrier GPU/CPU Source closure required')
    context = dict(original)
    for key in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'terminal_identity_pins', 'phase', 'entry_source', 'output_directory', 'deadline_seconds'):
        context[key] = spec[key]
    context['CPU_admission_file' if cpu else 'GPU_admission_file'] = spec['CPU_admission_file' if cpu else 'GPU_admission_file']
    lineage = load(original['endpoint_lineage_source'], '_N15_original_E24_lineage')
    lineage.check_recipe(original)
    gate = lineage.own_gate(path, context, spec['entry_source']['path'] if cpu else __file__, closure, raws, cpu=cpu)
    fail10=spec['additional_preserved_N16_cache_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail10[k])
    failrow=dict(receipt=fail10['RAW_ref'],phase=fail10['phase'],admission=fail10['admission'],
        phase_binding=fail10['phase_binding'],bootstrap=fail10['bootstrap'])
    failed_owner=lineage.owner_join(failrow,failed10,closure)
    require(failed_owner['scientific_spec']==fail10['spec'] and failed_owner['entry']==fail10['entry_source']['path'] and
        failed_owner['cuda_visible_devices']=='0','Preserved N16 actual failure10 original SOURCE/spec/ad owner differs')
    fail11=spec['additional_preserved_N16_cache_target_dtype_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail11[k])
    failrow11=dict(receipt=fail11['RAW_ref'],phase=fail11['phase'],admission=fail11['admission'],
        phase_binding=fail11['phase_binding'],bootstrap=fail11['bootstrap'])
    failed_owner11=lineage.owner_join(failrow11,failed11,closure)
    require(failed_owner11['scientific_spec']==fail11['spec'] and failed_owner11['entry']==fail11['entry_source']['path'] and
        failed_owner11['cuda_visible_devices']=='0','Preserved N16 actual failure11 original SOURCE/spec/ad owner differs')
    fail12=spec['additional_preserved_N16_controlled_resource_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail12[k])
    failrow12=dict(receipt=fail12['RAW_ref'],phase=fail12['phase'],admission=fail12['admission'],
        phase_binding=fail12['phase_binding'],bootstrap=fail12['bootstrap'])
    failed_owner12=lineage.owner_join(failrow12,failed12,closure)
    require(failed_owner12['scientific_spec']==fail12['spec'] and failed_owner12['entry']==fail12['entry_source']['path'] and
        failed_owner12['cuda_visible_devices']=='0','Preserved resource failure12 original owner differs')
    failed_spec12=read(closed(closure,fail12['spec']))
    absent_report12=Path(failed_spec12['output_directory'])/'REPORT.json'
    require(not absent_report12.exists() and not absent_report12.is_symlink() and fail12['original_report_absent'] is True,
        'Original controlled-abort report must remain absent, no fabricated success')
    fail13=spec['additional_preserved_N16_fullDEV_packet_ABI_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail13[k])
    failrow13=dict(receipt=fail13['RAW_ref'],phase=fail13['phase'],admission=fail13['admission'],
        phase_binding=fail13['phase_binding'],bootstrap=fail13['bootstrap'])
    failed_owner13=lineage.owner_join(failrow13,failed13,closure)
    require(failed_owner13['scientific_spec']==fail13['spec'] and failed_owner13['entry']==fail13['entry_source']['path'] and
        failed_owner13['cuda_visible_devices']=='0','Preserved fullDEV packet-ABI failure13 original owner differs')
    failed_spec13=read(closed(closure,fail13['spec']))
    absent_report13=Path(failed_spec13['output_directory'])/'REPORT.json'
    require(not absent_report13.exists() and not absent_report13.is_symlink() and fail13['original_report_absent'] is True,
        'Original failed13 REPORT must remain absent, never a qualified fullDEV success')
    fail14=spec['additional_preserved_N16_fullDEV_controlled_storage_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail14[k])
    failrow14=dict(receipt=fail14['RAW_ref'],phase=fail14['phase'],admission=fail14['admission'],
        phase_binding=fail14['phase_binding'],bootstrap=fail14['bootstrap'])
    failed_owner14=lineage.owner_join(failrow14,failed14,closure)
    require(failed_owner14['scientific_spec']==fail14['spec'] and failed_owner14['entry']==fail14['entry_source']['path'] and
        failed_owner14['cuda_visible_devices']=='0','Preserved controlled storage-failure14 original owner differs')
    failed_spec14=read(closed(closure,fail14['spec']))
    absent_report14=Path(failed_spec14['output_directory'])/'REPORT.json'
    require(not absent_report14.exists() and not absent_report14.is_symlink() and fail14['original_report_absent'] is True,
        'Original controlled storage-failure14 REPORT must remain absent, never AP/cache success')
    fail15=spec['additional_preserved_N16_fullDEV_controlled_bundle_deadline_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail15[k])
    failrow15=dict(receipt=fail15['RAW_ref'],phase=fail15['phase'],admission=fail15['admission'],
        phase_binding=fail15['phase_binding'],bootstrap=fail15['bootstrap'])
    failed_owner15=lineage.owner_join(failrow15,failed15,closure)
    require(failed_owner15['scientific_spec']==fail15['spec'] and failed_owner15['entry']==fail15['entry_source']['path'] and
        failed_owner15['cuda_visible_devices']=='0','Preserved controlled bundle-deadline15 original owner differs')
    failed_spec15=read(closed(closure,fail15['spec']))
    absent_report15=Path(failed_spec15['output_directory'])/'REPORT.json'
    require(not absent_report15.exists() and not absent_report15.is_symlink() and fail15['original_report_absent'] is True,
        'Original controlled bundle-deadline15 REPORT must remain absent, never AP success')
    fail16=spec['additional_preserved_N16_fullDEV_controlled_CPU_proof_deadline_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail16[k])
    failrow16=dict(receipt=fail16['RAW_ref'],phase=fail16['phase'],admission=fail16['admission'],phase_binding=fail16['phase_binding'],bootstrap=fail16['bootstrap'])
    failed_owner16=lineage.owner_join(failrow16,failed16,closure)
    require(failed_owner16['scientific_spec']==fail16['spec'] and failed_owner16['entry']==fail16['entry_source']['path'] and
        failed_owner16['cuda_visible_devices']=='','Original failedCPU16 Source/spec/entry/CPUempty actual owner exact')
    failed_spec16=read(closed(closure,fail16['spec']));absent_Q16=Path(failed_spec16['output_directory'])/'Q.json'
    require(not absent_Q16.exists() and not absent_Q16.is_symlink() and fail16['original_Q_absent'] is True,
        'Original failedCPU16 Q must remain absent; previous77 image proofs never reused')
    fail17=spec['additional_preserved_N16_final_CPU_reference_ABI_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail17[k])
    failrow17=dict(receipt=fail17['RAW_ref'],phase=fail17['phase'],admission=fail17['admission'],
                   phase_binding=fail17['phase_binding'],bootstrap=fail17['bootstrap'])
    failed_owner17=lineage.owner_join(failrow17,failed17,closure)
    require(failed_owner17['scientific_spec']==fail17['spec'] and failed_owner17['entry']==fail17['entry_source']['path'] and
            failed_owner17['cuda_visible_devices']=='','Original failed finalCPU17 Source/spec/entry/CPUempty owner exact')
    failed_spec17=read(closed(closure,fail17['spec']));absent_Q17=Path(failed_spec17['output_directory'])/'Q.json'
    require(not absent_Q17.exists() and not absent_Q17.is_symlink() and fail17['original_Q_absent'] is True,
            'Original failed finalCPU17 Q remains absent; four qualified numeric CPU leaves are preserved')
    fail18=spec['additional_preserved_N17_firstDEV_reference_ABI_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail18[k])
    failrow18=dict(receipt=fail18['RAW_ref'],phase=fail18['phase'],admission=fail18['admission'],
        phase_binding=fail18['phase_binding'],bootstrap=fail18['bootstrap'])
    failed_owner18=lineage.owner_join(failrow18,failed18,closure)
    require(failed_owner18['scientific_spec']==fail18['spec'] and failed_owner18['entry']==fail18['entry_source']['path'] and
        failed_owner18['cuda_visible_devices']=='0','Failure18 actual GPU0 owner/source/spec/admission binding exact')
    failed_spec18=read(closed(closure,fail18['spec']));absent_report18=Path(failed_spec18['output_directory'])/'REPORT.json'
    require(failed_spec18['schema']==SCHEMA and failed_spec18['shard_id']==0 and failed_spec18['dataset_ordinal_range']==[0,47] and
        failed_spec18['entry_source']==fail18['entry_source'] and failed_spec18['phase']==fail18['phase'] and
        failed_spec18['normal_RAW_count']==122 and failed_spec18['total_failed_RAW_count']==17 and
        failed_spec18['total_terminal_identity_pin_count']==139 and failed_spec18['RAW_inputs']==spec['RAW_inputs'][:122] and
        all(closure.get(r['path'])==r for r in failed_spec18['sources']) and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in failed_spec18['terminal_identity_pins'].items()) and
        not absent_report18.exists() and not absent_report18.is_symlink() and fail18['original_report_absent'] is True,
        'Original failed01/source/all122prefix/failed17/pins139 unchanged and REPORT still absent; never normal success')
    fail19=spec['additional_preserved_N17_secondDEV_SOURCE_closure_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail19[k])
    failrow19=dict(receipt=fail19['RAW_ref'],phase=fail19['phase'],admission=fail19['admission'],
        phase_binding=fail19['phase_binding'],bootstrap=fail19['bootstrap'])
    failed_owner19=lineage.owner_join(failrow19,failed19,closure)
    require(failed_owner19['scientific_spec']==fail19['spec'] and failed_owner19['entry']==fail19['entry_source']['path'] and
        failed_owner19['cuda_visible_devices']=='0','Failure19 actual GPU0 owner/source/spec/phase/admission/bootstrap exact')
    failed_spec19=read(closed(closure,fail19['spec']));absent_report19=Path(failed_spec19['output_directory'])/'REPORT.json'
    require(failed_spec19['schema']==SCHEMA and failed_spec19['shard_id']==0 and failed_spec19['dataset_ordinal_range']==[0,47] and
        failed_spec19['entry_source']==fail19['entry_source'] and failed_spec19['phase']==fail19['phase'] and
        failed_spec19['normal_RAW_count']==122 and failed_spec19['total_failed_RAW_count']==18 and
        failed_spec19['total_terminal_identity_pin_count']==140 and failed_spec19['RAW_inputs']==spec['RAW_inputs'][:122] and
        failed_spec19['deadline_seconds']==1800 and failed_spec19['automatic_retry'] is False and
        failed_spec19['allocator_limit_MiB']==4096 and failed_spec19['whole_GPU_limit_MiB']==8192 and
        all(closure.get(r['path'])==r for r in failed_spec19['sources']) and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in failed_spec19['terminal_identity_pins'].items()) and
        all(failed_spec19[k]==spec[k] for k in FAILURE19_PRIOR_KEYS) and
        not absent_report19.exists() and not absent_report19.is_symlink() and fail19['original_report_absent'] is True,
        'Original failed02/source/all122prefix/failed18/pins140 unchanged and REPORT still absent; never normal success')
    fail20=spec['additional_preserved_N17_DEV_GPU_controlled_deadline_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail20[k])
    failrow20=dict(receipt=fail20['RAW_ref'],phase=fail20['phase'],admission=fail20['admission'],
        phase_binding=fail20['phase_binding'],bootstrap=fail20['bootstrap'])
    failed_owner20=lineage.owner_join(failrow20,failed20,closure)
    require(failed_owner20['scientific_spec']==fail20['spec'] and failed_owner20['entry']==fail20['entry_source']['path'] and
        failed_owner20['cuda_visible_devices']=='0','Controlled20 original GPU0 owner/source/spec/ad/phase/bootstrap exact')
    failed_spec20=read(closed(closure,fail20['spec']));absent_report20=Path(failed_spec20['output_directory'])/'REPORT.json'
    require(failed_spec20['schema']==SCHEMA and failed_spec20['shard_id']==1 and failed_spec20['dataset_ordinal_range']==[47,94] and
        failed_spec20['entry_source']==fail20['entry_source'] and failed_spec20['phase']==fail20['phase'] and
        failed_spec20['normal_RAW_count']==123 and failed_spec20['total_failed_RAW_count']==19 and
        failed_spec20['total_terminal_identity_pin_count']==142 and failed_spec20['RAW_inputs']==spec['RAW_inputs'][:123] and
        failed_spec20['deadline_seconds']==1800 and failed_spec20['automatic_retry'] is False and
        failed_spec20['allocator_limit_MiB']==4096 and failed_spec20['whole_GPU_limit_MiB']==8192 and
        all(closure.get(r['path'])==r for r in failed_spec20['sources']) and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in failed_spec20['terminal_identity_pins'].items()) and
        all(failed_spec20[k]==spec[k] for k in FAILURE20_PRIOR_KEYS) and
        not absent_report20.exists() and not absent_report20.is_symlink() and fail20['original_report_absent'] is True,
        'Original failed GPU1 Source03/123prefix/failed19/pins142/REPORT absence preserved; partials never reused')
    closed_independent_CPU_SOURCE_failure21(spec,closure,lineage,failed21)
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
    gate["evaluation"] = load(spec["carrier_evaluation_source"], "_N16_strong_E24_loader")
    return gate

# N18 suffix: captured-action counts only; no head, scout, teacher, formula replay or COCO metrics.
N18_STATUS='N18_MATCHED374_CAPTURED_ACTION_DIAGNOSTIC_COMPLETE_PENDING_ROOT_ACTUAL0'
def exact(a,b):
    return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)

def diagnostic_header(spec):
    require(spec['schema']==CPU_SCHEMA and Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True) and
        all(type(spec[k]) is int for k in ('normal_RAW_count','expected_normal_RAW_count','total_failed_RAW_count','total_terminal_identity_pin_count')) and
        spec['normal_RAW_count']==spec['expected_normal_RAW_count'] and spec['normal_RAW_count']>=136 and spec['total_failed_RAW_count']==21 and
        spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+21 and spec['maximum_process_RSS_bytes']==16*1024**3 and
        spec['deadline_seconds']==1800 and spec['automatic_retry'] is False and spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192,
        'Own N18 CPUempty Source/136-or-later explicit prefix/21 failures/unchanged resource contract')

def completed_record(record,output_key,gate,spec):
    require(type(record) is dict and set(record)=={'source','spec',output_key,'RAW','admission','phase_binding','bootstrap','owner_index'} and
        type(record['owner_index']) is int and record['owner_index']<spec['normal_RAW_count'],'Exact completed original CPU record')
    c=gate['closure'];i=record['owner_index']
    for k in set(record)-{'owner_index'}:closed(c,record[k])
    row=spec['RAW_inputs'][i]
    require(all(row[k]==record[v] for k,v in (('receipt','RAW'),('admission','admission'),('phase_binding','phase_binding'),('bootstrap','bootstrap'))),
        'Original completed RAW/ad/phase/bootstrap prefix row')
    joined=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],record['RAW'],record['admission'],record['source'],record['spec'],'')
    parent=read(closed(c,record['spec']));result=read(closed(c,record[output_key]))
    require(type(joined) is int and joined==i==parent['normal_RAW_count'] and parent['RAW_inputs']==spec['RAW_inputs'][:i] and
        parent['entry_source']==record['source'] and parent['phase']==row['phase'] and parent['total_failed_RAW_count']==21 and
        parent['total_terminal_identity_pin_count']==i+21 and all(c.get(r['path'])==r for r in parent['sources']), 'Completed original CPU owner/full Source/prefix')
    return parent,result,terminal_raw_only(record['RAW'],row['phase'],0)

def diagnostic_gate(path,spec):
    diagnostic_header(spec);gate=inherited_source_gate(path,spec,cpu=True);c=gate['closure']
    for k in ('n17_root_qualified_result_ref','diagnostic_body_source','fixed_population','CPU_adapter_source','GPU_outer_source'):closed(c,spec[k])
    root=read(closed(c,spec['n17_root_qualified_result_ref']))
    require(exact(root,spec['expected_n17_root_qualified_result']) and root['schema']=='Root-N17-full187-qualified-result-after-final-and-post-actual0-v1' and
        root['status']=='FULL187_FIXED24_MAIN_RESULT_QUALIFIED_CURRENT_RECIPE_STOP' and root['qualified'] is True and
        root['original_final_CPU_actual0_verified'] is True and root['original_post_CPU_actual0_verified'] is True and root['normal_RAW_count']==136 and
        root['total_failed_RAW_count']==21 and root['total_terminal_identity_pin_count']==157 and root['decision']=='STOP' and root['allowed_next_seeds']==[] and
        all(root[k] is False for k in ('additional_seed_or_fusion_executed','formal_TEST_evaluated','cross_dataset_evaluated','innovation_established','TGRS_readiness_established')),
        'Exact completed Root qualification and immutable main STOP; no replacement outcome')
    final_spec,Q,raw_final=completed_record(root['final_record'],'Q',gate,spec)
    post_spec,post,raw_post=completed_record(root['post_record'],'ROOT_RESULT',gate,spec)
    require(root['final_record']['owner_index']==134 and root['post_record']['owner_index']==135 and
        exact(raw_final,root['original_final_actual']) and exact(raw_post,root['original_post_actual']) and
        post['final_record']==root['final_record'] and post['final_qualified_result']==root['final_qualified_result'] and
        post['qualified'] is False and post['ROOT_post_actual_CPU0_verified'] is False and
        root['final_qualified_result']['qualified'] is True and root['final_qualified_result']['final_Q']==root['final_record']['Q'] and
        root['final_qualified_result']['control_results']==Q['control_results'] and Q['qualified'] is False and
        root['primary_interpretation']==root['final_qualified_result']['primary_interpretation'], 'Original final/post outputs and actual0 attestations exact')
    original=read(closed(c,spec['carrier_GPU_spec']));closed(c,original['DEV']);closed(c,original['reader_source'])
    fixed=read(closed(c,spec['fixed_population']));details=read(closed(c,gate['carrier_Q']['details']))
    label='LOG_PR_seed202610120701_E24_fixed_native_score'
    matches=[v for v in details['fixed_native_query_GT_pairs'][label] if abs(v['threshold']-.5)<1e-12]
    require(set(fixed)=={'schema','creator','carrier_CPU_qualification','carrier_details','carrier_label','threshold','rows','total','matched','unmatched'} and
        fixed['schema']=='N15-fixed-native-E24-374-plus6-v1' and fixed['carrier_CPU_qualification']==spec['carrier_CPU_qualification'] and
        fixed['carrier_details']==gate['carrier_Q']['details'] and fixed['carrier_label']==label and fixed['threshold']==.5 and
        fixed['total']==380 and fixed['matched']==374 and fixed['unmatched']==6 and len(matches)==1 and fixed['rows']==matches[0]['GT_rows'], 'Original fixed374+6 population exact')
    rows=fixed['rows'];rowkeys={'image_id','gt_id','status','query_id','native','refined','refined_minus_native'}
    require(len(rows)==380 and all(type(r) is dict and set(r)==rowkeys for r in rows) and len({(r['image_id'],r['gt_id']) for r in rows})==380 and
        sum(r['status']=='native_matched_fixed_query' for r in rows)==374 and sum(r['status']=='native_FN' for r in rows)==6 and
        all((r['query_id'] is None)==(r['status']=='native_FN') for r in rows) and
        len({(r['image_id'],r['query_id']) for r in rows if r['query_id'] is not None})==374, 'Fixed matched identities, six unevaluable GTs never background')
    gpu=load(spec['GPU_outer_source'],'_N18_frozen_GPU_metadata_only');gpu.legacy_gate(spec,gate)
    body=load(spec['diagnostic_body_source'],'_N18_pure_counts');adapter=load(spec['CPU_adapter_source'],'_N18_frozen_capture_descriptor')
    require(body.SCHEMA=='N18-fixed-PR-routing-capability-counts-v1' and tuple(body.POLICIES)==('E24_LOCAL_atmost','E24_CENTER_atmost','Omega_strong_R'), 'Exact count body contract')
    origins={};parent_ids=set()
    for descriptor in Q['numeric_CPU_shards']:
        cpu_spec=read(closed(c,descriptor['spec']));item=cpu_spec['GPU_shard'];sid=item['shard_id']
        if sid in parent_ids:continue
        parent_ids.add(sid)
        for k in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):closed(c,item[k])
        joined=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],item['RAW'],item['admission'],item['source'],item['spec'],'0')
        parent=read(closed(c,item['spec']));report=read(closed(c,item['report']));manifest=read(closed(c,report['capture_manifest']))
        require(joined==122+sid==item['shard_owner_index'] and parent['RAW_inputs']==spec['RAW_inputs'][:joined] and
            report['source']==item['source'] and report['spec']==item['spec'] and report['GPU_admission']==item['admission'] and report['phase']==parent['phase'] and
            manifest['physical_capture_refs']==report['capture_refs']==[r['bundle'] for r in manifest['rows']], 'Original five qualified GPU parents/capture refs exact')
        binding={k:manifest[k] for k in ('source','spec','GPU_admission','phase','ROOTtrain_result','legacy_qualified_result','experiment_lock','new_eval_body_source','checkpoints','shard_id','dataset_ordinal_range')}
        if sid:binding['execution_partition_supplement']=manifest['execution_partition_supplement']
        for row in manifest['rows']:
            require(row['image_id'] not in origins,'One original image per GPU parent');closed(c,row['bundle']);closed(c,row['input_carrier']['bundle'])
            origins[row['image_id']]=dict(row=row,binding=binding)
    require(parent_ids==set(range(5)) and len(origins)==187 and sorted(v['row']['dataset_index'] for v in origins.values())==list(range(187)), 'Original five parents cover all187 images')
    gate.update(root_result=root,final_Q=Q,fixed=fixed,original=original,origins=origins,counts_body=body,capture_adapter=adapter)
    return gate

def rates(counts):
    def ratio(n,d):return None if d is None or d==0 else n/d
    repair=counts['PR_repairable_fraction'];budget=counts['budget_repairable_denominator']
    errors=counts['PR_capability_E_denominator'];B=counts['B']
    return dict(PR_unrepairable_fraction=None if repair is None else 1-repair,
        PR_unrepairable_numerator=None if errors is None else errors-B,PR_unrepairable_denominator=errors,
        routing_miss_rate={p:None if budget is None or budget==0 else 1-v['helpful_selected']/budget for p,v in counts['selections'].items()},
        routing_miss_numerator={p:None if budget is None else budget-v['helpful_selected'] for p,v in counts['selections'].items()},routing_miss_denominator=budget,
        selected_flip_harm_fraction={p:ratio(v['harmful_selected'],v['actual_flips']) for p,v in counts['selections'].items()},
        selected_flip_harm_numerator={p:v['harmful_selected'] for p,v in counts['selections'].items()},
        selected_flip_harm_denominator={p:v['actual_flips'] for p,v in counts['selections'].items()})

def count_pair(body,pair,P_C,GT_C,A_C,K,selections):
    require(pair['status']=='native_matched_fixed_query' and type(pair['query_id']) is int and 0<=pair['query_id']<100 and type(pair['gt_id']) is int,
        'Only original matched fixed query has an evaluable GT, never unmatched background')
    counts=body.count_query(P_C,GT_C,A_C,K,selections)
    return dict(identity={k:pair[k] for k in ('image_id','gt_id','query_id','status')},counts=counts,rates=rates(counts))

def captured_query(torch,np,adapter,bundle,shared,row,q,native,R,decode):
    require(bundle['schema']=='N17-actual-GPU-selector-image-bundle-v1' and bundle['input_carrier']==row['input_carrier'] and
        bundle['dataset_index']==row['dataset_index'] and bundle['image_id']==row['image_id'] and
        list(bundle['members'])==list(bundle['member_registry'])==row['member_locators'],'Exact qualified image/member registry, no full numerical replay')
    L=shared['native_L'][q].numpy();require(L.dtype==np.float32 and L.ndim==2 and np.isfinite(L).all(),'Actual FP32 native query grid')
    P=L>0;C=shared['C_indices'][q].numpy();K=shared['K_counts'][q];flat=P.reshape(-1);N=flat.size
    require(C.dtype==np.int64 and C.ndim==1 and bool((C[1:]>C[:-1]).all()) and bool(((C>=0)&(C<N)).all()) and
        type(K) is int and K==int((np.abs(L)<=2).sum()), 'Captured C and original FP32 physical Omega budget')
    omega=np.flatnonzero((np.abs(L)<=2).reshape(-1));op=np.searchsorted(C,omega)
    require(len(omega)==0 or (bool((op<len(C)).all()) and np.array_equal(C[op],omega)), 'Physical Omega subset of stored C')
    require(np.array_equal(decode(native['segmentation']),P),'Original native RLE equals actual native L>0')
    roster=row['query_profiles'][q];require(roster['query_id']==q and roster['K']==K and roster['C_count']==len(C),'Natural query roster')
    members=bundle['members'];used=set()
    def member(name):
        value=members[name];require(bundle['member_registry'][name]==adapter._registry(torch,value),'Exact consumed member descriptor');used.add(name);return value
    profiles={a:member(roster['profiles']['E24_'+a+'_atmost']) for a in ('LOCAL_ERROR','CENTER_ERROR')}
    common=profiles['LOCAL_ERROR']['common_C_PR_chunks'];require(common==profiles['CENTER_ERROR']['common_C_PR_chunks'],'Same captured full-C action for both policies')
    bits=[];offset=0
    for name in common:
        v=member(name);idx=v['flat_indices'].numpy();combined=v['GPU_combined'].numpy()
        require(K>0 and v['stage']=='full_C_PR' and v['control']=='COMMON_C_REFERENCE' and v['query_id']==q and v['scout_epoch'] is None and v['evalmode']=='shared' and
            v['actual_device_type']=='cuda' and v['synthetic_only'] is False and v['point_chunk']==2048 and v['chunk_first']==offset and
            0<len(idx)<=2048 and np.array_equal(idx,C[offset:offset+len(idx)]) and combined.dtype==np.float32 and combined.shape==idx.shape and np.isfinite(combined).all(),
            'Original ascending2048 actual full-C combined outputs, no new inference')
        bits.append(combined>0);offset+=len(idx)
    require(offset==(len(C) if K else 0),'All observed C action points, K0 remains unobserved')
    A=np.concatenate(bits) if K else None;supports={}
    for arm,v in profiles.items():
        require(v['query_id']==q and v['epoch']==24 and v['arm']==arm and v['mode']=='atmost' and v['K']==K and
            v['original_prediction_nonsegmentation_fields']=={k:x for k,x in native.items() if k!='segmentation'}, 'Actual fixed main policy and original nonseg fields')
        idx=v['actual_selected_flat_indices'].numpy();pos=np.searchsorted(C,idx)
        require(idx.dtype==np.int64 and len(idx)<=K and bool((idx[1:]>idx[:-1]).all()) and
            (len(idx)==0 or (bool((pos<len(C)).all()) and np.array_equal(C[pos],idx))), 'Actual S stays within real C/K')
        packed=v['actual_mask_packbits'];require(packed.dtype==torch.uint8 and packed.ndim==1 and len(packed)==(N+7)//8 and
            v['actual_mask_shape']==list(P.shape) and v['actual_mask_bitorder']=='little','Actual mask storage shape')
        maskbits=np.unpackbits(packed.numpy(),bitorder='little');require(not bool(maskbits[N:].any()),'Zero mask padding')
        mask=maskbits[:N].astype(np.bool_);expected=flat.copy()
        if K:expected[idx]=A[pos]
        require(np.array_equal(mask,expected) and np.array_equal(decode(v['actual_RLE']).reshape(-1),mask),
            'N18_SAME_ACTION_COMPATIBILITY_FAILURE: selected actual mask must apply captured common A and preserve native outside S')
        support=np.zeros(len(C),dtype=np.bool_);support[pos]=True;supports['E24_'+arm.split('_')[0]+'_atmost']=support
    rmask=decode(R['segmentation']).reshape(-1);expected=flat.copy()
    if K:expected[omega]=A[op]
    require(np.array_equal(rmask,expected),'N18_SAME_ACTION_COMPATIBILITY_FAILURE: original strong R must equal same common-C A on Omega and native outside')
    os=np.zeros(len(C),dtype=np.bool_);os[op]=True;supports['Omega_strong_R']=os
    return P,C,A,K,supports,len(used)

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);require(Path(spec_path).is_absolute(),'Absolute own diagnostic spec')
    spec=read(path);gate=diagnostic_gate(path,spec);c=gate['closure'];output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh admitted diagnostic output')
    import psutil
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    process=psutil.Process();require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV), 'CPU01/emptyCUDA/threads2')
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and not torch.cuda.is_initialized(),'Closed cold CPU runtime')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2);outside=torch.get_rng_state().clone()
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and psutil.virtual_memory().available>=16*1024**3 and
            process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
            torch.get_num_threads()==torch.get_num_interop_threads()==2 and not torch.cuda.is_initialized(),'Original diagnostic CPU resources/deadline/no retry')
    check();require(psutil.disk_usage(str(output.parent)).free>=1024**3,'Free output1GiB')
    reader=load(gate['original']['reader_source'],'_N18_original_GT_RLE_bytes');coco=COCO(str(closed(c,gate['original']['DEV'])))
    require(len(coco.imgs)==187 and len(coco.anns)==380 and set(coco.cats)=={1} and
        {r['gt_id'] for r in gate['fixed']['rows']}==set(coco.anns),'Exact original DEV annotation IDs, not proxy or background')
    def decode(rle):return masks.decode(reader.rle_bytes(rle)).astype(np.bool_)
    groups={}
    for pair in gate['fixed']['rows']:
        if pair['status']=='native_matched_fixed_query':groups.setdefault(pair['image_id'],[]).append(pair)
    rows=[];consumed=[];members=0
    for image_id,pairs in groups.items():
        check();origin=gate['origins'][image_id];row=origin['row'];carrier=row['input_carrier']
        old=torch.load(str(closed(c,carrier['bundle'])),map_location='cpu');bundle=torch.load(str(closed(c,row['bundle'])),map_location='cpu')
        require(old['binding']==carrier['original_GPU_binding'] and old['image_id']==image_id and old['dataset_index']==row['dataset_index'] and
            carrier['shared_member'] in old['members'] and bundle['binding']==origin['binding'],'Exact original old/shared and new GPU input bindings')
        shared=old['members'][carrier['shared_member']];require(shared['identity']==row['identity'] and shared['schema']=='N16-actual-GPU-image-shared-inputs-v1','Qualified original shared actual inputs')
        native={p['query_id']:p for p in gate['populations']['native_official'][image_id]};R={p['query_id']:p for p in gate['populations']['strong_fixed_E24_Omega'][image_id]}
        for pair in pairs:
            check();q=pair['query_id'];P,C,A,K,S,nused=captured_query(torch,np,gate['capture_adapter'],bundle,shared,row,q,native[q],R[q],decode)
            ann=coco.anns[pair['gt_id']];require(ann['image_id']==image_id,'Original matched annotation belongs to image')
            G=decode(coco.annToRLE(ann));require(G.shape==P.shape and bool(G.any()),'True nonempty original instance GT mask')
            result=count_pair(gate['counts_body'],pair,P.reshape(-1)[C],G.reshape(-1)[C],A,K,S)
            result.update(dataset_index=row['dataset_index'],fixed_population_row=pair,bundle=row['bundle'],actual_member_ids_consumed=nused,same_action_masks_exact=True)
            rows.append(result);members+=nused;del P,C,A,K,S,G,result
        consumed.append(row['bundle']);del old,bundle,shared;gc.collect()
    require(len(rows)==374 and len({(r['identity']['image_id'],r['identity']['gt_id']) for r in rows})==374,'Exactly original374 matched instances')
    aggregate=gate['counts_body'].aggregate_counts([r['counts'] for r in rows]);aggregate_rates=rates(aggregate)
    unmatched=[r for r in gate['fixed']['rows'] if r['status']=='native_FN'];require(len(unmatched)==6,'Six unmatched remain unevaluable, outside primary mean')
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['CPU_admission_file'])==gate['admission_ref'] and
        torch.equal(outside,torch.get_rng_state()) and libraries==gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage),'Final Source/ad/import/RNG exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check();output.mkdir()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],n17_root_qualified_result_ref=spec['n17_root_qualified_result_ref'],
        fixed_population=spec['fixed_population'],diagnostic_body_source=spec['diagnostic_body_source'],CPU_adapter_source=spec['CPU_adapter_source'],DEV=gate['original']['DEV'])
    write(output/'DETAILS.json',dict(schema='N18-fixed374-routing-diagnostic-details-v1',**binding,instances=rows,unmatched_unevaluable=unmatched,
        captures_consumed=consumed,member_consumption_scope='Only common-C and main two profiles for matched queries; no repeat full numerical/member qualification',
        actual_member_ids_consumed=members,actual_imported_files=libraries))
    write(output/'Q.json',dict(schema='N18-fixed374-routing-diagnostic-pending-v1',status=N18_STATUS,**binding,details=reference(output/'DETAILS.json'),
        conditional_matched_instances=374,unmatched_unevaluable=6,full_reference_GT=380,aggregate=aggregate,rates=aggregate_rates,
        summary=dict(counts=aggregate,**aggregate_rates),K0_separate=True,
        normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=21,total_terminal_identity_pin_count=spec['normal_RAW_count']+21,
        original_main_decision='STOP',new_AP_evaluated=False,models_loaded=False,numerical_formulas_replayed=False,GT_used_for_ranking=False,
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,elapsed_seconds=time.monotonic()-started,
        evidence_scope='Conditional fixed374 candidate-local routing/action counts only; unmatched6 unevaluable; no AP bound, TEST, new seed or causal innovation claim.'))
    print(json.dumps(dict(status=N18_STATUS,Q=reference(output/'Q.json'),qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND: Root alone binds existing qualified captures and original GT; no auto start');run(args.spec)




"""Ordinary construction fragment only; embedded into standalone CPU entry."""
