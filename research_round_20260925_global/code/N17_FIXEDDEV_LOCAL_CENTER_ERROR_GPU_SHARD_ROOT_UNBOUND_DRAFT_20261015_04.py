"""UNBOUND_DRAFT: standalone N17 fixedDEV LOCAL/CENTER ERROR GPU shard entry.
Copied immutable ordinary RAW9..19/carrier front; controlled20 and physical-five execution retained; ROOT owns binding/execution.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 20 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 20 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 stays separate; version04 includes explicit failure20 and normal+20 pins only')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9..19 and controlled DEV GPU20 already passed RAW-only checks.
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
    # FIRST HASH: failed9..19, explicit controlled20, and every normal/legacy failed8 passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(closure.get(row[key]['path']) == row[key] for row in rows for key in RAW_KEYS if key != 'phase') and
            closure.get(failed_ref['path']) == failed_ref and all(closure.get(row['receipt']['path']) == row['receipt'] for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N17-fixedDEV-local-center-error-GPU-shard-spec-v1"
CPU_SCHEMA = "ROOT-N17-local-center-error-paired-training-endpoint-CPU-spec-v1"
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



N17_SCHEMA=SCHEMA
REPORT_SCHEMA='N17-local-native-L-paired-error-training-GPU-report-v1'
STATUS='N17_LOCAL_CENTER_ERROR_TRAINING_COMPLETE_PENDING_ROOT_ACTUAL_GPU0_AND_CPU_QUALIFICATION'
ARMS=('LOCAL_ERROR','CENTER_ERROR')
SEED=202610150171
ORDER_SEED=202610140160001
OFFSETS=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
NEW_REFS=('experiment_lock','neighbor_cache_ROOT_result','neighbor_cache_CPU_source','neighbor_cache_CPU_spec',
    'neighbor_cache_CPU_Q','neighbor_cache_CPU_RAW','neighbor_cache_CPU_admission','neighbor_cache_manifest',
    'neighbor_kernel_source','neighbor_cache_body_source','original_N16_cache_report','scout_module_source',
    'base_scout_source','paired_training_body_source','original_paired_training_body_source')
ROOT_CACHE_BINDINGS={'CPU_Q':'neighbor_cache_CPU_Q','manifest':'neighbor_cache_manifest','RAW':'neighbor_cache_CPU_RAW',
    'source':'neighbor_cache_CPU_source','kernel':'neighbor_kernel_source','body':'neighbor_cache_body_source',
    'spec':'neighbor_cache_CPU_spec','CPU_admission':'neighbor_cache_CPU_admission','experiment_lock':'experiment_lock'}

def validate_experiment_lock(spec,closure):
    lock=read(closed(closure,spec['experiment_lock']))
    require(type(spec['expected_experiment_lock']) is dict and lock==spec['expected_experiment_lock'] and
        lock['schema']=='N17-local-native-logit-error-paired-experiment-locked-v1' and
        lock['parent_protocol']==spec['locked_protocol'] and lock['previous_N16_decision']=='STOP',
        'Entire independent N17 prelocked experiment must match; old N16 lock remains ancestry only')
    for key in ('parent_protocol','qualified_parent_cache','qualified_previous_result'):closed(closure,lock[key])
    architecture=lock['architecture'];training=lock['training']
    require(architecture['first_seed']==SEED and architecture['repeat_seeds']==[202610150172,202610150173] and
        architecture['parameters']==16803 and architecture['layers']==[521,32,3] and architecture['tensors']==4 and
        architecture['old_factory_SHA256']==spec['base_scout_source']['sha256'] and
        architecture['neighbor_module_SHA256']==spec['scout_module_source']['sha256'],
        'Frozen lifted ERROR architecture, first/repeat seeds and actual factories differ')
    require(training['epochs']==24 and training['updates_each_arm']==17784 and training['point_exposure_each_arm']==72843264 and
        training['order_seed']==ORDER_SEED and training['AdamW_lr']==3e-4 and training['weight_decay']==1e-4 and
        training['betas']==[.9,.999] and training['eps']==1e-8 and training['AMP'] is False,
        'N17 fixed24/741 updates/4096 points and unchanged optimizer recipe required')
    closed(closure,architecture['initialization_metadata'])
    return lock

def validate_neighbor_cache(spec,gate,lock):
    closure=gate['closure'];result=read(closed(closure,spec['neighbor_cache_ROOT_result']))
    require(result['schema']=='ROOT-N17-fullFIT-neighbor-cache-qualified-result-v1' and
        result['status']=='ROOT_N17_FULLFIT_NATIVE_EIGHT_NEIGHBOR_CACHE_FULLY_QUALIFIED' and
        result['qualified'] is True and result['paired_training_eligible'] is True and
        all(result[key]==spec[value] for key,value in ROOT_CACHE_BINDINGS.items()) and
        result['qualified_original_FIT_cache']==lock['qualified_parent_cache'] and
        result['initialization_metadata']==lock['architecture']['initialization_metadata'] and
        result['images']==741 and result['natural_queries']==74100 and result['sampled_points']==3035136 and
        result['payload_bytes']==97124352 and result['feature_files']==741 and
        result['normal_ancestors_after_actual0']==120 and result['preserved_failed_ancestors']==17 and
        result['total_terminal_pins_after_actual0']==137,'Only complete actual-ROOT-qualified CPU119 neighbor cache permits paired training')
    parent=read(closed(closure,spec['neighbor_cache_CPU_spec']));q=read(closed(closure,spec['neighbor_cache_CPU_Q']))
    joined=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],spec['neighbor_cache_CPU_RAW'],
        spec['neighbor_cache_CPU_admission'],spec['neighbor_cache_CPU_source'],spec['neighbor_cache_CPU_spec'],'')
    require(type(joined) is int and joined==119 and parent['normal_RAW_count']==119 and
        parent['RAW_inputs']==spec['RAW_inputs'][:119] and parent['entry_source']==spec['neighbor_cache_CPU_source'] and
        parent['total_failed_RAW_count']==17 and parent['total_terminal_identity_pin_count']==136 and
        spec['RAW_inputs'][119]['receipt']==spec['neighbor_cache_CPU_RAW'] and
        spec['RAW_inputs'][119]['admission']==spec['neighbor_cache_CPU_admission'] and
        spec['RAW_inputs'][119]['phase_binding']==result['phase_binding'] and spec['RAW_inputs'][119]['bootstrap']==result['bootstrap'] and
        Path(parent['CPU_admission_file']).resolve(strict=True)==Path(spec['neighbor_cache_CPU_admission']['path']).resolve(strict=True),
        'New cache is original CPUempty0 owner119 with exact119-prefix; never relative old105/106 owner arithmetic')
    require(all(closure.get(ref['path'])==ref for ref in parent['sources']) and
        all(spec['terminal_identity_pins'].get(key)==value for key,value in parent['terminal_identity_pins'].items()),
        'Complete CPU119 Source/pins inherited unchanged')
    require(q['schema']=='N17-fullFIT-native-eight-neighbor-cache-CPU-proof-v1' and
        q['status']=='N17_FULLFIT_NEIGHBOR_CACHE_COMPLETE_PENDING_ROOT_ACTUAL0' and
        q['source']==spec['neighbor_cache_CPU_source'] and q['spec']==spec['neighbor_cache_CPU_spec'] and
        q['CPU_admission']==spec['neighbor_cache_CPU_admission'] and q['phase']==parent['phase'] and
        q['neighbor_experiment_lock']==spec['experiment_lock'] and q['neighbor_kernel_source']==spec['neighbor_kernel_source'] and
        q['neighbor_cache_body_source']==spec['neighbor_cache_body_source'] and q['qualified_parent_cache']==lock['qualified_parent_cache'] and
        q['manifest']==spec['neighbor_cache_manifest'] and q['normal_RAW_count']==119 and
        q['total_failed_RAW_count']==17 and q['total_terminal_identity_pin_count']==136,
        'Original CPU119 pending Q plus separately actual qualified ROOT result must bind exactly')
    manifest=read(closed(closure,spec['neighbor_cache_manifest']));old=read(closed(closure,spec['original_N16_cache_report']))
    require(manifest['schema']=='N17-FIT-native-L-neighbors8-cache-manifest-v1' and
        manifest['original_cache_report']==spec['original_N16_cache_report'] and manifest['original_locked_protocol']==spec['locked_protocol'] and
        manifest['offsets']==[list(v) for v in OFFSETS] and manifest['summary']['images']==741 and
        manifest['summary']['queries']==74100 and manifest['summary']['points']==3035136 and
        manifest['summary']['payload_bytes']==97124352 and manifest['summary']['feature_files']==741,
        'Complete original FIT741 neighbor manifest/counts/offsets required')
    for key,oldkey in (('original_source','source'),('original_spec','spec'),('original_GPU_admission','GPU_admission'),('original_phase','phase')):
        require(manifest[key]==old[oldkey],'Neighbor manifest cannot replace original N16 cache binding: '+key)
    for key in ('fixed_PR_checkpoint','teacher_checkpoint'):
        require(manifest[key]==old[key],'Original teacher/fixed E24 PR references unchanged');closed(closure,old[key])
    for key in ('source','spec','GPU_admission'):closed(closure,old[key])
    rows=manifest['rows'];oldrows=old['rows']
    require(type(rows) is list and type(oldrows) is list and len(rows)==len(oldrows)==len(old['packet_refs'])==len(old['native_L_refs'])==741 and
        [r['dataset_index'] for r in rows]==[r['dataset_index'] for r in oldrows]==list(range(741)) and
        len({r['image_id'] for r in rows})==741 and len({r['neighbors8_ref']['path'] for r in rows})==741,
        'Exactly complete ordered original FIT741 physical arrays required')
    flags=('same_query_native_center_byte_exact','all_samples_NumPy_neighbors_byte_exact','center_control_byte_exact',
        'original_labels_weights_points_preserved_by_packet_ref')
    for index,(row,oldrow) in enumerate(zip(rows,oldrows)):
        require(row['image_id']==oldrow['image_id'] and row['original_shape']==list(oldrow['identity']['original_shape']) and
            row['packet_ref']==oldrow['packet']==old['packet_refs'][index] and
            row['native_L_ref']==oldrow['natural_native_L_carrier']==old['native_L_refs'][index] and
            row['queries']==100 and row['points']==4096 and row['payload_bytes']==131072 and all(row[k] is True for k in flags),
            'Every neighbor file binds original full row/point count and exact mechanical proofs')
        for key in ('packet_ref','native_L_ref','neighbors8_ref'):
            ref_shape(row[key]);require(closure.get(row[key]['path'])==row[key],'Every physical original/generated array must be SOURCE-closed')
    gate.update(neighbor_cache_ROOT_result=result,neighbor_cache_CPU_spec=parent,neighbor_cache_CPU_Q=q,
        neighbor_manifest=manifest,original_cache_report=old,neighbor_rows=rows,original_cache_rows=oldrows)
    return gate

def state_digest(torch,state):
    whole=hashlib.sha256();tensors=[]
    for name,tensor in sorted(state.items()):
        require(isinstance(tensor,torch.Tensor) and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),'Finite FP32 state to hash')
        info=dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype))
        header=json.dumps(info,sort_keys=True,separators=(',',':')).encode('utf8')
        raw=tensor.detach().cpu().contiguous().numpy().tobytes(order='C')
        whole.update(len(header).to_bytes(8,'little'));whole.update(header);whole.update(raw)
        tensors.append(dict(info,raw_bytes_sha256=hashlib.sha256(raw).hexdigest()))
    return whole.hexdigest(),tensors

def load_training_image(torch,np,index,row,oldrow,oldreport,closure,locked_protocol,check):
    check();require(index==row['dataset_index']==oldrow['dataset_index'] and row['packet_ref']==oldrow['packet'] and
        row['native_L_ref']==oldrow['natural_native_L_carrier'],'Qualified image/packet/nativeL reference association exact')
    packet=torch.load(str(closed(closure,row['packet_ref'])),map_location='cpu')
    require(type(packet) is dict and packet['dataset_index']==index and packet['identity']==oldrow['identity'] and
        packet['identity']['image_id']==row['image_id'] and list(packet['identity']['original_shape'])==row['original_shape'] and
        packet['natural_native_L_carrier']==row['native_L_ref'] and packet['protocol']==locked_protocol,
        'Actual old thin packet full identity/protocol/nativeL pointer exact; no full native-L payload opened')
    for key in ('source','spec','GPU_admission','phase','fixed_PR_checkpoint','teacher_checkpoint'):
        require(packet[key]==oldreport[key],'Original frozen N16 thin-packet binding differs: '+key)
    sample=packet['samples'];q,flat=sample['query_ids'],sample['flat_indices'];H,W=row['original_shape']
    require(q.device.type==flat.device.type=='cpu' and q.dtype==flat.dtype==torch.int64 and
        tuple(q.shape)==tuple(flat.shape)==(4096,) and bool(((q>=0)&(q<100)).all()) and bool(((flat>=0)&(flat<H*W)).all()),
        'Exactly original4096 native query/point indices required')
    digest=hashlib.sha256(q.numpy().tobytes()+flat.numpy().tobytes()).hexdigest()
    require(digest==row['point_order_sha256'],'Qualified NPY must preserve original duplicate sample/query order')
    neighbors=np.load(str(closed(closure,row['neighbors8_ref'])),allow_pickle=False)
    require(isinstance(neighbors,np.ndarray) and neighbors.dtype==np.float32 and neighbors.shape==(4096,8) and
        bool(np.isfinite(neighbors).all()),'Actual same-order qualified FP324096x8 NPY required, no recomputation/resize')
    check();return dict(packet=packet,neighbors8=neighbors)

"""Construction fragment only; joined into independent CPU121 entry, never imported by it."""

def old_tensor_bytes(torch,value):
    if isinstance(value,torch.Tensor):
        require(value.device.type=='cpu' and not value.requires_grad and value.layout==torch.strided and
            value.dtype in (torch.float32,torch.int64,torch.bool,torch.uint8) and bool(torch.isfinite(value).all()),'Capture tensors must be actual finite CPU copies')
        return value.numel()*value.element_size()
    if type(value) is dict:return sum(old_tensor_bytes(torch,v) for v in value.values())
    if type(value) in (tuple,list):return sum(old_tensor_bytes(torch,v) for v in value)
    return 0

def old_structural_payload_bytes(torch,value):
    if isinstance(value,torch.Tensor):
        require(value.device.type=='cpu' and not value.requires_grad and value.layout==torch.strided and
            value.dtype in (torch.float32,torch.int64,torch.bool,torch.uint8),'Actual CPU bundle tensor type required')
        return value.numel()*value.element_size()
    if type(value) is dict:return sum(old_structural_payload_bytes(torch,v) for v in value.values())
    if type(value) in (list,tuple):return sum(old_structural_payload_bytes(torch,v) for v in value)
    return 0

def old_bundle_descriptor(torch,value,validate_tensors=True):
    schema=value.get('schema')
    if schema=='N16-actual-GPU-image-shared-inputs-v1':
        kind='shared_inputs';q=epoch=mode=control=stage=first=points=None
    elif schema=='N16-actual-GPU-query-core-and-bundle-member-IDs-v3':
        kind='query_core';q=value['query_id'];epoch=value['epoch'];mode=value['evalmode'];control=stage=first=points=None
    else:
        stage=value['stage'];kind='scout_chunk' if stage=='scout_C_logits' else 'PR_chunk'
        require(stage in ('scout_C_logits','selected_PR','full_C_PR','original_Omega_PR'),'Only actual known point chunk role accepted')
        q=value['query_id'];epoch=value.get('scout_epoch');mode=value['evalmode'];control=value['control'];first=value['chunk_first']
        points=len(value['flat_indices'])
    return dict(kind=kind,query_id=q,epoch=epoch,evalmode=mode,control=control,stage=stage,chunk_first=first,
        points=points,raw_tensor_payload_bytes=(old_tensor_bytes(torch,value) if validate_tensors else old_structural_payload_bytes(torch,value)))

def old_same_float_bits(torch,a,b):
    return a.shape==b.shape and a.dtype==b.dtype==torch.float32 and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))

def old_tree_equal(torch,left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and (old_same_float_bits(torch,left,right) if left.dtype==torch.float32 else torch.equal(left,right))
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(old_tree_equal(torch,v,right[k]) for k,v in left.items())
    if type(left) in (tuple,list):return type(right) is type(left) and len(left)==len(right) and all(old_tree_equal(torch,a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right

def old_validate_shared_for_CPU(torch,np,cv2,shared):
    """Pure actual-input validation; ROOT authenticates physical refs/closure."""
    require(shared['schema']=='N16-actual-GPU-image-shared-inputs-v1' and
        shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False and
        shared['natural_query_ids']==list(range(100)) and sorted(shared['native_export_query_ids'])==list(range(100)),
        'Whole-image actual input provenance/natural100 required')
    L=shared['native_L'];F=shared['F_unique'];e=shared['e0'];union=shared['union_flat_indices']
    require(L.device.type==F.device.type==e.device.type==union.device.type=='cpu' and
        L.dtype==F.dtype==e.dtype==torch.float32 and L.ndim==3 and L.shape[0]==100 and min(L.shape[1:])>0 and
        e.shape==(100,256) and union.dtype==torch.int64 and union.ndim==1 and F.shape==(len(union),256) and
        all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in (L,F,e)) and
        shared['U']==len(union) and len(shared['C_indices'])==len(shared['C_counts'])==len(shared['K_counts'])==100,
        'Exact finite CPU shared FP32 shapes and actual union required')
    merged=np.zeros(L.shape[1:],dtype=np.bool_)
    for q in range(100):
        native=L[q].numpy();P=native>0;Omega=np.abs(native)<=2
        if bool(P.any()):
            dilation=cv2.dilate(np.ascontiguousarray(P,dtype=np.uint8),np.ones((17,17),dtype=np.uint8),
                anchor=(8,8),iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
            C=Omega|P|dilation
        else:C=Omega.copy()
        idx=np.flatnonzero(C).astype(np.int64);saved=shared['C_indices'][q]
        require(saved.device.type=='cpu' and saved.dtype==torch.int64 and saved.shape==(len(idx),) and
            np.array_equal(saved.numpy(),idx) and type(shared['C_counts'][q]) is int and shared['C_counts'][q]==len(idx) and
            type(shared['K_counts'][q]) is int and shared['K_counts'][q]==int(np.count_nonzero(Omega)),
            'Independent native FP32 rectangular C/K/all100 support mismatch')
        merged|=C
    require(np.array_equal(union.numpy(),np.flatnonzero(merged).astype(np.int64)),
        'Actual GPU feature union must equal independently recomputed all100 C union exactly')
    return dict(natural_queries=100,all_C_K_exact=True,union_mapping_exact=True,CPU_F_rerendered=False,
        actual_Source_REF_provenance_proven_here=False)

def old_candidate_engine(np,cv2):
    def rectangular_dilation(P,*,structure,iterations,border_value):
        require(P.dtype==np.bool_ and P.ndim==2 and structure.shape==(17,17) and
            bool(np.all(structure)) and iterations==1 and border_value==0,'Only original centered17/constant0 binary dilation permitted')
        if not bool(P.any()):return np.zeros(P.shape,dtype=np.bool_)
        return cv2.dilate(np.ascontiguousarray(P,dtype=np.uint8),np.ones((17,17),dtype=np.uint8),
            anchor=(8,8),iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
    return types.SimpleNamespace(binary_dilation=rectangular_dilation)

"""Ordinary construction fragment; embedded into standalone GPU shard entry."""
TRAIN_SCHEMA='ROOT-N17-local-native-L-paired-error-training-GPU-spec-v1'
TRAIN_CPU_SCHEMA='ROOT-N17-local-center-error-paired-training-endpoint-CPU-spec-v1'
LEGACY_SHARD_RANGES=((0,47),(47,94),(94,141),(141,187))
SHARD_RANGES=((0,47),(47,70),(70,94),(94,141),(141,187))
STATE_SHAPES={'hidden.weight':(32,521),'hidden.bias':(32,),'output.weight':(3,32),'output.bias':(3,)}
NEW_BUNDLE_SCHEMA='N17-actual-GPU-selector-image-bundle-v1'
NEW_MANIFEST_SCHEMA='N17-fixedDEV-GPU-shard-image-manifest-v1'
NEW_REPORT_SCHEMA='N17-fixedDEV-local-center-error-GPU-shard-report-v1'
NEW_STATUS='N17_FIXEDDEV_GPU_SHARD_CAPTURE_COMPLETE_PENDING_ROOT_ACTUAL0_AND_INDEPENDENT_CPU'
OLD_BUNDLE_SCHEMA='N16-actual-GPU-image-capture-bundle-v1'
OLD_CHECKS=('original_held0_reaped_birth_absent','whole_GPU_and_RAM_and_deadline','final_Q_Source_spec_admission_exact',
    'all_four_completed_parent_ref_and_legacy_counts_exact','all187_image_proof_refs_coordinates_all100_profiles_and_member_counts',
    'original_GPU_bindings_preserved','all12_full_COCO_semantics_and_AP113_stats_exact','all12_prediction_refs_exact','immutable_pending_Q_unchanged')

def validate_gpu_header(spec):
    require(type(spec) is dict and spec['schema']==SCHEMA and type(spec['shard_id']) is int and 1<=spec['shard_id']<5 and
        spec['dataset_ordinal_range']==list(SHARD_RANGES[spec['shard_id']]) and
        all(type(spec[k]) is int for k in ('normal_RAW_count','expected_normal_RAW_count','total_failed_RAW_count',
            'total_terminal_identity_pin_count','initialization_seed','deadline_seconds')) and
        spec['normal_RAW_count']==spec['expected_normal_RAW_count']==122+spec['shard_id'] and
        spec['total_failed_RAW_count']==20 and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+20 and
        spec['initialization_seed']==SEED and spec['deadline_seconds']==1800 and spec['automatic_retry'] is False and
        spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192 and
        Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True),
        'Own newGPU fixed shard/count/schema/resources required; no future owner prediction after failure')
    return True

def policy_names():
    return ('native_official','strong_fixed_E24_Omega')+tuple('E%d_%s_%s'%(e,a,m) for e in (24,8) for a in ARMS for m in ('atmost','exact'))

def training_gate(spec,gate):
    c=gate['closure'];root=read(closed(c,spec['ROOTtrain_result']))
    require(root['schema']=='ROOT-N17-paired24-GPU-CPU-qualified-training-result-v1' and
        root['status']=='ROOT_N17_PAIRED24_GPU_CPU_TRAINING_FULLY_QUALIFIED' and root['qualified'] is True and
        root['eligible_for_fixed_DEV_evaluation'] is True,'Actual ROOT complete N17 training qualification mandatory')
    mapping={'GPU_source':'training_GPU_source','GPU_spec':'training_GPU_spec','GPU_admission':'training_GPU_admission_refs',
        'GPU_report':'training_GPU_report','GPU_RAW':'training_GPU_RAW','CPU_source':'training_CPU_source','CPU_spec':'training_CPU_spec',
        'CPU_admission':'training_CPU_admission_refs','CPU_Q':'training_CPU_qualification','CPU_RAW':'training_CPU_RAW',
        'experiment_lock':'experiment_lock','qualified_neighbor_cache':'neighbor_cache_ROOT_result',
        'scout_module_source':'scout_module_source','base_scout_source':'base_scout_source','training_body_source':'paired_training_body_source'}
    for k,v in mapping.items():require(root[k]==spec[v],'ROOT training exact reference mismatch: '+k);closed(c,spec[v])
    for key in ('GPU_phase_binding','GPU_bootstrap','CPU_phase_binding','CPU_bootstrap','CPU_details','initialization_metadata','qualifier_body_source'):closed(c,root[key])
    gpu=read(closed(c,root['GPU_spec']));cpu=read(closed(c,root['CPU_spec']));report=read(closed(c,root['GPU_report']));q=read(closed(c,root['CPU_Q']))
    gi=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],root['GPU_RAW'],root['GPU_admission'],root['GPU_source'],root['GPU_spec'],'0')
    ci=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],root['CPU_RAW'],root['CPU_admission'],root['CPU_source'],root['CPU_spec'],'')
    require(type(gi) is int and type(ci) is int and (gi,ci)==(120,121) and gpu['schema']==TRAIN_SCHEMA and cpu['schema']==TRAIN_CPU_SCHEMA and
        gpu['normal_RAW_count']==120 and cpu['normal_RAW_count']==121 and gpu['RAW_inputs']==spec['RAW_inputs'][:120] and
        cpu['RAW_inputs']==spec['RAW_inputs'][:121],'N17 original GPU120/CPU121 actual0 prefixes, never relative owner arithmetic')
    for ancestor in (gpu,cpu):
        require(all(c.get(r['path'])==r for r in ancestor['sources']) and
            all(spec['terminal_identity_pins'].get(k)==v for k,v in ancestor['terminal_identity_pins'].items()),'Full N17 parent SOURCE/pins inherited')
    for idx,prefix in ((120,'GPU'),(121,'CPU')):
        require(spec['RAW_inputs'][idx]['phase_binding']==root[prefix+'_phase_binding'] and spec['RAW_inputs'][idx]['bootstrap']==root[prefix+'_bootstrap'],
            'Original training phase binding/bootstrap exact')
    require(report['schema']==REPORT_SCHEMA and report['status']==STATUS and report['source']==root['GPU_source'] and
        report['spec']==root['GPU_spec'] and report['GPU_admission']==root['GPU_admission'] and report['phase']==gpu['phase'] and
        q['schema']=='N17-local-center-error-paired-endpoint-CPU-qualification-v1' and
        q['status']=='N17_LOCAL_CENTER_ERROR_E8_E24_ENDPOINT_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0' and
        q['source']==root['CPU_source'] and q['spec']==root['CPU_spec'] and q['CPU_admission']==root['CPU_admission'] and
        q['phase']==cpu['phase'] and q['details']==root['CPU_details'] and q['qualified'] is False and q['ROOT_actual_CPU0_verified'] is False and
        q['GPU_pointwise_logits_qualified'] is False and q['selected_indices_qualified'] is False and q['AP_evaluated'] is False and
        q['optimizer_trajectory_replayed'] is False and type(q['checks']) is dict and bool(q['checks']) and all(v is True for v in q['checks'].values()),
        'Immutable endpoint Q scope and separately actual qualified ROOT result required')
    require(root['checkpoints']==report['checkpoints']==q['training_GPU_checkpoints'] and root['FIT_endpoints']==report['FIT_endpoints']==q['training_GPU_FIT_endpoints'] and
        set(root['checkpoints'])==set(root['FIT_endpoints'])=={'8','24'} and root['initialization_metadata']==gate['experiment_lock']['architecture']['initialization_metadata'],
        'Both exact actual trained checkpoints and locked initialization, no DEV endpoint selection')
    for key in ('START','UPDATE_TRACE'):closed(c,report[key])
    for ref in tuple(root['checkpoints'].values())+tuple(root['FIT_endpoints'].values()):closed(c,ref)
    gate['training']=dict(root=root,gpu=gpu,cpu=cpu,report=report,q=q);return gate['training']

COCO_METRIC_NAMES=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL')

def validate_legacy_COCO(result):
    stats=result['stats'];coco=result['COCO']
    require(type(stats) is list and len(stats)==12 and
        all(type(v) in (int,float) and math.isfinite(v) and (v==-1 or 0<=v<=1) for v in stats) and
        type(coco) is dict and set(coco)=={'stats','metrics','iouType','maxDets','original_GT_preserved','score_filter','official_COCO_crowd_ignore_semantics'} and
        type(coco['stats']) is list and coco['stats']==stats and type(coco['metrics']) is dict and
        set(coco['metrics'])==set(COCO_METRIC_NAMES) and
        all(type(coco['metrics'][k]) in (int,float) and math.isfinite(coco['metrics'][k]) and coco['metrics'][k]==stats[i] for i,k in enumerate(COCO_METRIC_NAMES)) and
        coco['iouType']=='segm' and coco['maxDets']==[1,10,100] and coco['original_GT_preserved'] is True and
        coco['score_filter'] is None and coco['official_COCO_crowd_ignore_semantics'] is True,
        'Original inline full COCO12 dict: exact stats/metric names/raw scale/segm/maxDets/GT/score/crowd semantics')
    return True

def legacy_gate(spec,gate):
    c=gate['closure'];root=read(closed(c,spec['legacy_qualified_result']))
    require(root['schema']=='ROOT-N16-fixed-E24-all100-scout-fullDEV-qualified-result-v1' and
        root['status']=='ROOT_QUALIFIED_COMPLETE_GPU_NUMERIC_CPU_FULL_COCO' and root['qualified'] is True and
        root['locked_protocol']==spec['locked_protocol'] and set(root['checks'])==set(OLD_CHECKS) and all(v is True for v in root['checks'].values()),
        'Original complete N16 fourGPU/numericCPU/fullCOCO qualification, not a pending report')
    for k in ('final_source','final_spec','final_CPU_qualification','final_RAW','final_phase_binding','final_bootstrap','final_admission'):closed(c,root[k])
    final=read(closed(c,root['final_spec']));require(type(final['shards']) is list and len(final['shards'])==4 and final['shards']==spec['legacy_shards'],
        'Explicit inherited four original shards from actual final02 spec required')
    fi=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],root['final_RAW'],root['final_admission'],root['final_source'],root['final_spec'],'')
    require(type(fi) is int and fi<120 and final['RAW_inputs']==spec['RAW_inputs'][:fi] and final['entry_source']==root['final_source'] and
        spec['RAW_inputs'][fi]['phase_binding']==root['final_phase_binding'] and spec['RAW_inputs'][fi]['bootstrap']==root['final_bootstrap'],
        'Original finalCPU owner/source/prefix exact, not count-minus guesses')
    require(all(c.get(r['path'])==r for r in final['sources']),'Original final CPU complete SOURCE closure inherited')
    require(type(root['numeric_CPU_shards']) is list and len(root['numeric_CPU_shards'])==4,'Original four independent numeric CPU refs retained')
    for item in root['numeric_CPU_shards']:
        for k in ('source','spec','report','RAW','admission'):closed(c,item[k])
    rows=[];reports=[];manifests=[];seen=set()
    for i,item in enumerate(spec['legacy_shards']):
        require(item['shard_id']==i and item['dataset_ordinal_range']==list(LEGACY_SHARD_RANGES[i]),'Original fixed shard partition exact')
        for k in ('source','spec','report','RAW','admission'):closed(c,item[k])
        parent=read(closed(c,item['spec']));report=read(closed(c,item['report']))
        owner=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],item['RAW'],item['admission'],item['source'],item['spec'],'0')
        require(type(owner) is int and owner==item['shard_owner_index'] and parent['RAW_inputs']==spec['RAW_inputs'][:owner] and
            parent['entry_source']==item['source'] and all(c.get(r['path'])==r for r in parent['sources']), 'Original shard owner and full parent SOURCE exact')
        require(report['schema']=='N16-fixed-E24-all100-scout-fullDEV-GPU-shard-report-v1' and
            report['status']=='N16_FIXED_E24_ALL100_SCOUT_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0' and report['source']==item['source'] and
            report['spec']==item['spec'] and report['GPU_admission']==item['admission'] and report['phase']==parent['phase'] and
            report['locked_protocol']==spec['locked_protocol'] and report['scope_dataset_indices']==list(range(*LEGACY_SHARD_RANGES[i])) and
            report['shard_owner_index']==owner,'Original source-bound shard report, never imported as new entry')
        manifest=read(closed(c,report['capture_manifest']));binding=report['capture_binding']
        require(manifest['schema']=='N16-GPU-local-ordinal-shard-image-bundle-manifest-v1' and manifest['capture_binding']==binding and
            manifest['physical_capture_refs']==report['capture_refs'] and manifest['images']==LEGACY_SHARD_RANGES[i][1]-LEGACY_SHARD_RANGES[i][0] and
            manifest['queries']==manifest['images']*100 and manifest['global_images']==187 and manifest['global_queries']==18700 and
            len(manifest['rows'])==len(report['capture_refs'])==manifest['images'],'All four actual strict manifest and physical refs')
        for offset,row in enumerate(manifest['rows']):
            idx=LEGACY_SHARD_RANGES[i][0]+offset
            require(row['dataset_index']==idx and type(row['image_id']) is int and row['image_id'] not in seen and
                row['bundle']==report['capture_refs'][offset] and row['shared_actual_image_inputs']=='i%03d/SHARED_ACTUAL_INPUTS.pth'%idx and
                len(row['qids'])==100 and all(type(q) is int for q in row['qids']) and sorted(row['qids'])==list(range(100)) and
                row['selector']['native_export_query_ids']==row['qids'] and row['natural_queries']==100,
                'Canonical original image/ref/shared member/native export order')
            seen.add(row['image_id']);closed(c,row['bundle']);closed(c,row['native_packet']);closed(c,row['diagnostics'])
            rows.append(dict(row=row,binding=binding,shard_id=i));
        reports.append(report);manifests.append(manifest)
    require(len(rows)==len(seen)==187 and sum(r['raw_GT'] for r in reports)==380,'Original complete all187/380/18700 population')
    populations={}
    for name in ('native_official','strong_fixed_E24_Omega'):
        result=root['control_results'][name];validate_legacy_COCO(result);values=read(closed(c,result['predictions']))
        require(type(result['stats']) is list and len(result['stats'])==12 and all(type(v) in (int,float) and math.isfinite(v) for v in result['stats']) and
            type(values) is list and len(values)==18700,'Original complete predictions and full12COCO evidence')
        grouped={}
        for value in values:grouped.setdefault(value['image_id'],[]).append(value)
        for item in rows:
            row=item['row'];group=grouped.get(row['image_id'],[])
            require([v['query_id'] for v in group]==row['qids'],'Original full prediction order must equal original manifest qids')
        populations[name]=grouped
    require(root['control_results']['strong_fixed_E24_Omega']['stats'][0]*100==68.09938494745545,'Exact fixed strong reference AP unchanged')
    gate.update(legacy_root=root,legacy_rows=rows,legacy_reports=reports,legacy_manifests=manifests,populations=populations);return gate

PARTITION_SUPPLEMENT_SCHEMA='N17-fixedDEV-GPU-five-partition-execution-supplement-v1'

def validate_execution_partition_supplement(spec,closure):
    value=read(closed(closure,spec['execution_partition_supplement']))
    expected=spec['expected_execution_partition_supplement']
    require(type(value) is dict and type(expected) is dict and
        json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False),
        'Exact caller-prelocked execution partition supplement required')
    required=dict(schema=PARTITION_SUPPLEMENT_SCHEMA,logical_partition_count=4,physical_partition_count=5,
        logical_partition_ranges=[list(v) for v in LEGACY_SHARD_RANGES],physical_partition_ranges=[list(v) for v in SHARD_RANGES],
        shard0_reused_from_qualified_Source03=True,failed_partials_reused=False,method_training_data_metric_stopping_unchanged=True,
        single_active_leaf=True,deadline_seconds=1800,automatic_retry=False,owner_indices=[122,123,124,125,126],total_evaluation_walltime_increased=True)
    for key,wanted in required.items():
        require(key in value and json.dumps(value[key],sort_keys=True,allow_nan=False)==json.dumps(wanted,sort_keys=True,allow_nan=False),
            'Execution-only supplement semantic/type mismatch: '+key)
    return value

def source_gate(path,spec):
    validate_gpu_header(spec);gate=inherited_source_gate(path,spec,cpu=False);c=gate['closure']
    for k in NEW_REFS+('ROOTtrain_result','legacy_qualified_result','new_eval_body_source','legacy_eval_body_source'):closed(c,spec[k])
    lock=validate_experiment_lock(spec,c);validate_neighbor_cache(spec,gate,lock);gate['experiment_lock']=lock
    training_gate(spec,gate);legacy_gate(spec,gate);gate['execution_partition_supplement']=validate_execution_partition_supplement(spec,c);return gate

def new_point_member_id(index,event):
    q=event['query_id'];stage=event['stage'];control=event['control'];epoch=event['scout_epoch'];mode=event['evalmode'];first=event['chunk_first']
    require(type(index) is int and 0<=index<187 and type(q) is int and 0<=q<100 and type(first) is int and first>=0 and first%2048==0,
        'New actual natural image/query/chunk metadata')
    if stage=='full_C_PR':require(control=='COMMON_C_REFERENCE' and epoch is None and mode=='shared','One actual common-C reference');role='commonC'
    else:
        require(control in ARMS and type(epoch) is int and epoch in (8,24) and
            ((stage=='scout_C_logits' and mode=='shared-atmost-exact') or (stage=='selected_PR' and mode in ('atmost','exact'))),
            'N17 actual LOCAL/CENTER controls only; never old UTILITY names')
        role='e%d_%s_%s_%s'%(epoch,control,mode,stage)
    return 'i%03d/q%03d/%s_c%06d.pth'%(index,q,role,first)

def new_profile_member_id(index,q,epoch,arm,mode):
    require(type(index) is int and 0<=index<187 and type(q) is int and 0<=q<100 and type(epoch) is int and epoch in (8,24) and arm in ARMS and mode in ('atmost','exact'),
        'New fixed profile identifiers only')
    return 'i%03d/q%03d/e%d_%s_%s_PROFILE.pth'%(index,q,epoch,arm,mode)

def validate_actual_shared(torch,np,cv2,bundle,item):
    row=item['row'];require(type(bundle) is dict and set(bundle)=={'schema','binding','dataset_index','image_id','members','member_registry','member_count','raw_tensor_payload_bytes'} and
        bundle['schema']==OLD_BUNDLE_SCHEMA and bundle['binding']==item['binding'] and bundle['dataset_index']==row['dataset_index'] and
        bundle['image_id']==row['image_id'] and type(bundle['members']) is dict and set(bundle['members'])==set(bundle['member_registry']) and
        list(bundle['members'])==row['member_locators'] and bundle['member_count']==len(bundle['members']), 'Actual original registered physical image bundle and immutable binding')
    name=row['shared_actual_image_inputs'];shared=bundle['members'][name]
    require(shared['schema']=='N16-actual-GPU-image-shared-inputs-v1' and shared['dataset_index']==row['dataset_index'] and shared['image_id']==row['image_id'] and
        shared['identity']==row['identity'] and shared['native_packet']==row['native_packet'] and shared['selector']==row['selector'] and
        shared['native_export_query_ids']==row['qids'] and shared['natural_query_ids']==list(range(100)) and
        shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False,'Shared exact identity/selector/native order, not marker-only eligibility')
    old_validate_shared_for_CPU(torch,np,cv2,shared)
    descriptor=old_bundle_descriptor(torch,shared)
    require(bundle['member_registry'][name]==descriptor,'Actual shared registered role/size exact')
    for member,value in bundle['members'].items():require(bundle['member_registry'][member]==old_bundle_descriptor(torch,value,validate_tensors=False),'Original member registry metadata unchanged')
    return shared

def tensor_bytes(torch,value):
    if isinstance(value,torch.Tensor):
        require(value.device.type=='cpu' and value.layout==torch.strided and not value.requires_grad and value.dtype in (torch.float32,torch.int64,torch.uint8,torch.bool) and
            bool(torch.isfinite(value).all()),'Finite actual CPU capture tensor')
        return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(torch,v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(torch,v) for v in value)
    return 0

def same_bits(torch,left,right):
    return left.dtype==right.dtype==torch.float32 and left.shape==right.shape and torch.equal(left.contiguous().view(torch.int32),right.contiguous().view(torch.int32))

def capture_event_cpu(torch,np,event,shared,kernel):
    require(event['actual_device_type']=='cuda' and event['synthetic_only'] is False,'Only actual CUDA callback; no synthetic science')
    value={k:(v.detach().cpu().clone().contiguous() if isinstance(v,torch.Tensor) else v) for k,v in event.items()}
    flat=value['flat_indices'];q=value['query_id'];require(flat.device.type=='cpu' and flat.dtype==torch.int64 and flat.ndim==1,'Actual CPU original flat slots')
    positions=np.searchsorted(shared['union_flat_indices'].numpy(),flat.numpy())
    require(bool(np.all(positions<len(shared['union_flat_indices']))) and np.array_equal(shared['union_flat_indices'].numpy()[positions],flat.numpy()),'Every observed input F slot covered exactly')
    expectedF=shared['F_unique'][torch.from_numpy(positions.astype(np.int64))]
    require(same_bits(torch,value['sampled_F'],expectedF) and same_bits(torch,value['e0'],shared['e0'][q]) and
        same_bits(torch,value['native_L'],shared['native_L'][q].reshape(-1)[flat]),'Actual CUDA F/e/L readback bitexact to old qualified shared input')
    if value['stage']=='scout_C_logits':
        function=api(kernel,'neighbor_features_numpy' if value['control']=='LOCAL_ERROR' else 'center_control_features_numpy')
        extra=function(np,shared['native_L'].numpy(),np.full(len(flat),q,dtype=np.int64),flat.numpy())
        require(same_bits(torch,value['extra8'],torch.from_numpy(extra)) and same_bits(torch,value['actual_GPU_logits'],value['scout_logits']),
            'Actual GPU eight-input and logits alias independently exact')
    value={k:v for k,v in value.items() if k not in ('sampled_F','e0','native_L')}
    value.update(schema='N17-actual-GPU-selector-point-chunk-v1',actual_observed_F_e_L_bitexact_to_input_carrier=True)
    return value

class NewImageWriter:
    def __init__(self,torch,root,spec,binding,check):self.torch,self.root,self.spec,self.binding,self.check=torch,Path(root),spec,binding,check;self.refs=[];self.raw=0;self.disk=0
    def begin(self,item):
        self.item=item;self.members={};self.registry={};self.image_raw=0
        row=item['row'];self.input_carrier=dict(bundle=row['bundle'],shared_member=row['shared_actual_image_inputs'],original_GPU_binding=item['binding'])
    def put(self,name,value):
        self.check();require(name not in self.members,'Unique new capture role/member')
        size=tensor_bytes(self.torch,value);b=self.spec['capture_budget']
        require(self.image_raw+size<=b['maximum_perimage_raw_capture_bytes'] and self.raw+self.image_raw+size<=b['maximum_total_raw_capture_payload_bytes'],'Admitted raw capture cap')
        self.members[name]=value;self.registry[name]=dict(schema=value['schema'],query_id=value['query_id'],stage=value.get('stage','profile'),
            scout_epoch=value.get('scout_epoch',value.get('epoch')),control=value.get('control',value.get('arm')),evalmode=value.get('evalmode',value.get('mode')),
            chunk_first=value.get('chunk_first'),raw_tensor_payload_bytes=size);self.image_raw+=size;return name
    def finish(self):
        row=self.item['row'];target=self.root/('IMAGE_%03d_ACTUAL_GPU_BUNDLE.pth'%row['dataset_index']);self.check()
        payload=dict(schema=NEW_BUNDLE_SCHEMA,binding=self.binding,input_carrier=self.input_carrier,dataset_index=row['dataset_index'],image_id=row['image_id'],
            members=self.members,member_registry=self.registry,member_count=len(self.members),raw_tensor_payload_bytes=self.image_raw)
        with target.open('xb') as stream:self.torch.save(payload,stream)
        replay=self.torch.load(str(target),map_location='cpu');require(old_tree_equal(self.torch,payload,replay),'Exact new bundle serialization/readback');del replay
        ref=reference(target);require(self.disk+ref['bytes']<=self.spec['capture_budget']['maximum_serialized_capture_bytes'],'Serialized capture cap')
        self.refs.append(ref);self.raw+=self.image_raw;self.disk+=ref['bytes'];locators=list(self.members)
        self.members={};self.registry={};return ref,locators

def model_load_cpu(torch,spec,gate):
    require(not torch.cuda.is_initialized(),'Strict factory/checkpoint loads before CUDA')
    outside=torch.get_rng_state().clone();heads,states,proofs=gate['evaluation'].load_heads(torch,gate['context'],gate)
    strong=heads['LOG_PR'];pr_state=states['LOG_PR'];require(len(pr_state)==8 and sum(v.numel() for v in pr_state.values())==40497 and not tuple(strong.buffers()),'Original qualified LOG E24 PR only')
    strong.eval().requires_grad_(False);del heads,states
    scout=load(spec['scout_module_source'],'_N17_actual_four_input_scout');base=load(spec['base_scout_source'],'_N17_original_cold_factory')
    body=load(spec['new_eval_body_source'],'_N17_new_fixedDEV_numerical_body');legacy=load(spec['legacy_eval_body_source'],'_N16_frozen_point_numerical_helpers')
    helper=load(spec['labels_source'],'_N16_original_FP64_error_score_selector')
    require(tuple(body.ARMS)==ARMS and tuple(body.EPOCHS)==(24,8) and tuple(body.MODES)==('atmost','exact') and body.STATE_SHAPES==STATE_SHAPES and
        body.POINT_CHUNK==2048 and body.ABSOLUTE_TOLERANCE==1e-4,'Frozen new eval body ABI')
    models={};frozen={'PR':{k:v.detach().cpu().clone() for k,v in pr_state.items()}};training=gate['training'];prefix={}
    for epoch in (8,24):
        cp=torch.load(str(closed(gate['closure'],training['root']['checkpoints'][str(epoch)])),map_location='cpu')
        require(cp['schema']=='N17-paired-local-center-error-training-CPU-snapshot-v1' and cp['completed_epoch']==epoch and cp['parameters_per_arm']==16803 and
            cp['initialization_seed']==SEED and set(cp['models'])==set(ARMS) and cp['metadata']['source']==training['root']['GPU_source'] and
            cp['metadata']['spec']==training['root']['GPU_spec'] and cp['metadata']['GPU_admission']==training['root']['GPU_admission'] and
            cp['metadata']['phase']==training['report']['phase'] and cp['metadata']['endpoint_epoch']==epoch,'Actual top-level qualified trained snapshot provenance')
        prefix[epoch]=(cp['history'],cp['epoch_orders'])
        for arm in ARMS:
            state=cp['models'][arm];require(set(state)==set(STATE_SHAPES) and all(v.device.type=='cpu' and v.dtype==torch.float32 and
                tuple(v.shape)==STATE_SHAPES[k] and bool(torch.isfinite(v).all()) for k,v in state.items()),'Full four finite trained matrices, no trained-eight-zero restriction')
            head=api(scout,'make_scout')(torch,SEED,api(base,'make_scout'));head.load_state_dict(state,strict=True);head.eval().requires_grad_(False)
            models[(epoch,arm)]=head;frozen[(epoch,arm)]={k:v.detach().clone() for k,v in state.items()}
        del cp
    require(prefix[24][0][:8]==prefix[8][0] and prefix[24][1][:8]==prefix[8][1] and torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized(),
        'Both fixed endpoints/history/order prefix and private CPU construction')
    return strong,models,frozen,body,legacy,helper,scout,proofs

def capture_budget_gate(spec,gate):
    b=spec['capture_budget'];keys=('maximum_process_RSS_bytes','maximum_perimage_raw_capture_bytes','maximum_total_raw_capture_payload_bytes',
        'maximum_serialized_capture_bytes','minimum_free_output_bytes','auxiliary_output_budget_bytes')
    require(type(b) is dict and all(type(b[k]) is int and b[k]>0 for k in keys) and b['maximum_process_RSS_bytes']<=16*1024**3 and
        b['maximum_perimage_raw_capture_bytes']<=1024**3 and b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']<=72*1024**3 and
        b['minimum_free_output_bytes']>=b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']+16*1024**3,'ROOT concrete bounded disk/RAM capture admission')
    a,z=SHARD_RANGES[spec['shard_id']];bound_bytes=0
    for item in gate['legacy_rows'][a:z]:
        row=item['row'];N=math.prod(row['identity']['original_shape']);s=row['shared_capture_cost'];C=s['sumC'];K=s['sumK']
        require(type(C) is int and type(K) is int and C>=K>=0,'Original per-image all100 counts')
        per=272*C+192*K+800*((N+7)//8)+16*1024**2
        require(per<=b['maximum_perimage_raw_capture_bytes'],'New extra8/logit aliases/profile-mask raw upper bound')
        bound_bytes+=per
    require(bound_bytes<=b['maximum_total_raw_capture_payload_bytes'] and b['maximum_serialized_capture_bytes']>=bound_bytes,
        'Explicit new capture bound, never old104C formula')
    return bound_bytes

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);require(Path(spec_path).is_absolute(),'Absolute new ROOT shard spec')
    spec=read(path);gate=source_gate(path,spec);raw_upper=capture_budget_gate(spec,gate)
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh new shard output')
    import psutil
    process=psutil.Process();require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and
        all(os.environ.get(k)=='2' for k in THREAD_ENV) and psutil.virtual_memory().available>=16*1024**3 and
        psutil.disk_usage(str(output.parent)).free>=spec['capture_budget']['minimum_free_output_bytes'],'Actual admitted GPU0/CPU01/thread/disk/RAM')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and not torch.cuda.is_initialized(),'Closed cold runtime')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    strong,models,frozen,body,legacy,helper,kernel,model_proofs=model_load_cpu(torch,spec,gate)
    engine=old_candidate_engine(np,cv2);libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    torch.cuda.set_device(0);torch.cuda.init();require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Only actual CUDA0')
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0);torch.cuda.reset_peak_memory_stats(0)
    outside=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone());strong.to('cuda:0')
    for head in models.values():head.to('cuda:0')
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<spec['capture_budget']['maximum_process_RSS_bytes'] and
            process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
            torch.get_num_threads()==torch.get_num_interop_threads()==2 and torch.cuda.current_device()==0 and torch.cuda.memory_reserved(0)<=4096*1024**2,
            'Bounded single actual GPU shard; no retry')
    binding=dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],ROOTtrain_result=spec['ROOTtrain_result'],
        legacy_qualified_result=spec['legacy_qualified_result'],experiment_lock=spec['experiment_lock'],new_eval_body_source=spec['new_eval_body_source'],
        checkpoints=gate['training']['root']['checkpoints'],shard_id=spec['shard_id'],dataset_ordinal_range=spec['dataset_ordinal_range'],
        execution_partition_supplement=spec['execution_partition_supplement'])
    output.mkdir();(output/'CAPTURES').mkdir();writer=NewImageWriter(torch,output/'CAPTURES',spec,binding,check)
    predictions={name:[] for name in policy_names()};manifest_rows=[];first,last=SHARD_RANGES[spec['shard_id']];image_costs=[];auxiliary=0
    def auxiliary_write(path,value):
        nonlocal auxiliary
        write(path,value);auxiliary+=path.stat().st_size;require(auxiliary<=spec['capture_budget']['auxiliary_output_budget_bytes'],'Actual auxiliary cap')
    def encode(mask):
        encoded=masks.encode(np.asfortranarray(mask.astype(np.uint8)));return dict(size=[int(v) for v in encoded['size']],counts=encoded['counts'].decode('ascii'))
    auxiliary_write(output/'START.json',dict(schema='N17-fixedDEV-GPU-shard-start-v1',**binding,model_proofs=model_proofs,raw_capture_upper_bound=raw_upper,
        main_epoch=24,curve_epoch=8,policies=policy_names(),AP_evaluated=False,native_model_loaded=False))
    for item in gate['legacy_rows'][first:last]:
        check();row=item['row'];index=row['dataset_index'];tick=time.monotonic()
        bundle=torch.load(str(closed(gate['closure'],row['bundle'])),map_location='cpu');shared=validate_actual_shared(torch,np,cv2,bundle,item);writer.begin(item)
        L=shared['native_L'].to('cuda:0');e=shared['e0'].to('cuda:0');F=shared['F_unique'].to('cuda:0')
        require(same_bits(torch,L.cpu(),shared['native_L']) and same_bits(torch,e.cpu(),shared['e0']) and same_bits(torch,F.cpu(),shared['F_unique']),
            'Actual reused inputs survive GPU transfer bitexactly')
        baseline=gate['populations']['native_official'][row['image_id']];strong_predictions=gate['populations']['strong_fixed_E24_Omega'][row['image_id']]
        by_q={v['query_id']:v for v in baseline};current={name:{} for name in policy_names()[2:]};refs={};query_members=[]
        def capture(event):
            value=capture_event_cpu(torch,np,event,shared,kernel);name=writer.put(new_point_member_id(index,value),value)
            refs.setdefault((value['query_id'],value['scout_epoch'],value['control'],value['evalmode'],value['stage']),[]).append(name)
        for result in api(body,'evaluate_all100')(torch,np,engine,helper,legacy,kernel,L,e,shared['union_flat_indices'],F,models,strong,
                point_callback=capture,check_callback=check,synthetic_only=False):
            q=result['query_id'];profiles={}
            for name,profile in result['profiles'].items():
                epoch,arm,mode=profile['epoch'],profile['arm'],profile['mode'];encoded=encode(profile['mask']);prediction=dict(by_q[q],segmentation=encoded);current[name][q]=prediction
                value=dict(schema='N17-actual-GPU-selector-query-profile-v1',query_id=q,epoch=epoch,arm=arm,mode=mode,K=result['K'],
                    C_scout_chunks=refs.get((q,epoch,arm,'shared-atmost-exact','scout_C_logits'),[]),
                    common_C_PR_chunks=refs.get((q,None,'COMMON_C_REFERENCE','shared','full_C_PR'),[]),
                    selected_PR_chunks=refs.get((q,epoch,arm,mode,'selected_PR'),[]),
                    actual_selected_flat_indices=torch.from_numpy(profile['selected_flat_indices'].copy()),
                    actual_mask_packbits=torch.from_numpy(np.packbits(profile['mask'].ravel(),bitorder='little')),actual_mask_shape=list(profile['mask'].shape),
                    actual_mask_bitorder='little',actual_RLE=encoded,original_prediction_nonsegmentation_fields={k:v for k,v in by_q[q].items() if k!='segmentation'},
                    PR_compatibility=profile['PR_compatibility'])
                profiles[name]=writer.put(new_profile_member_id(index,q,epoch,arm,mode),value)
            query_members.append(dict(query_id=q,profiles=profiles,K=result['K'],C_count=len(result['C_flat_indices'])))
            del result,profiles,profile
        require(len(query_members)==100 and all(set(values)==set(range(100)) for values in current.values()),'All natural100 actual outputs')
        predictions['native_official'].extend(baseline);predictions['strong_fixed_E24_Omega'].extend(strong_predictions)
        for name,values in current.items():
            ordered=[values[q] for q in row['qids']]
            require(all({k:v for k,v in x.items() if k!='segmentation'}=={k:v for k,v in y.items() if k!='segmentation'} for x,y in zip(ordered,baseline)),
                'Never rescore/change class/query/export order')
            predictions[name].extend(ordered)
        ref,members=writer.finish();manifest_rows.append(dict(dataset_index=index,image_id=row['image_id'],identity=row['identity'],qids=row['qids'],
            input_carrier=writer.input_carrier,bundle=ref,member_locators=members,query_profiles=query_members,natural_queries=100))
        image_costs.append(dict(dataset_index=index,inclusive_seconds=time.monotonic()-tick))
        del bundle,shared,L,e,F,baseline,strong_predictions,by_q,current,refs,query_members;gc.collect();check()
    require(all(len(v)==(last-first)*100 for v in predictions.values()),'Complete fixed shard, no local AP averaging')
    for key,head in [('PR',strong)]+list(models.items()):
        require(all(same_bits(torch,v.detach().cpu(),frozen[key][k]) for k,v in head.state_dict().items()) and
            all(not p.requires_grad and p.grad is None for p in head.parameters()),'All actual model states/grads frozen exactly')
    require(torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),'Evaluation RNG unchanged')
    peak=torch.cuda.max_memory_allocated(0);strong.cpu()
    for head in models.values():head.cpu()
    models.clear();del head,strong,models;gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    require(torch.cuda.memory_allocated(0)==0 and peak<=4096*1024**2,'Actual GPU allocation cleanup')
    outputs={}
    for name,values in predictions.items():
        target=output/(name+'_PREDICTIONS.json');auxiliary_write(target,values);outputs[name]=dict(predictions=reference(target))
    auxiliary_write(output/'CAPTURE_MANIFEST.json',dict(schema=NEW_MANIFEST_SCHEMA,**binding,images=last-first,queries=(last-first)*100,global_images=187,global_queries=18700,
        rows=manifest_rows,physical_capture_refs=writer.refs,internal_member_locators_are_strings=True,raw_tensor_payload_bytes=writer.raw,serialized_bytes=writer.disk))
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['GPU_admission_file'])==gate['admission_ref'] and
        gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)==libraries,'Final actual source/spec/admission/imported files')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    auxiliary_write(output/'REPORT.json',dict(schema=NEW_REPORT_SCHEMA,status=NEW_STATUS,**binding,controls=policy_names(),control_results=outputs,
        scope_dataset_indices=list(range(first,last)),scope_image_ids=[v['image_id'] for v in manifest_rows],images=last-first,queries=(last-first)*100,
        capture_manifest=reference(output/'CAPTURE_MANIFEST.json'),capture_refs=writer.refs,START=reference(output/'START.json'),
        normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=20,total_terminal_identity_pin_count=spec['normal_RAW_count']+20,
        original_input_bindings_preserved=True,original_scores_classes_query_order_preserved=True,native_and_R_refs_bound_reused=True,
        new_actual_extra8_and_logits_and_selected_PR_and_afterMasks_captured=True,actual_imported_files=libraries,image_inclusive_costs=image_costs,
        elapsed_seconds=time.monotonic()-started,peak_torch_allocated_bytes=peak,qualified=False,current_owned_terminal_self_certified=False,
        independent_CPU_proof_complete=False,AP_evaluated=False,local_AP_not_computed=True,standalone_end_to_end_FPS_claimed=False,failed_partials_reused=False,
        task_efficacy_established=False,innovation_established=False,evidence_limits='One complete fixed partition only; need reused Source03 shard0 plus four Source04 actual0 shards, independent CPU and unified full187/18700 COCO10.'))
    print(json.dumps(dict(status=NEW_STATUS,report=reference(output/'REPORT.json'),qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND: ROOT alone binds complete N17 training/old shared qualification/resources/current Source')
    run(args.spec)
