"""UNBOUND_DRAFT: standalone N17 full187 CPU COCO final02 entry.
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
CPU_SCHEMA = "ROOT-N17-fixedDEV-full187-CPU-COCO-final-spec-v1"
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



"""Ordinary final construction fragment; no model or numerical proof rerun."""
import sys
RANGES=((0,47),(47,70),(70,94),(94,141),(141,187))
CPU_DESCRIPTOR_KEYS={'cpu_shard_id','cpu_dataset_ordinal_range','source','spec','report','RAW','admission','phase_binding','bootstrap','cpu_shard_owner_index'}
FINAL_SCHEMA='N17-fixedDEV-full187-CPU-COCO-qualification-v1'
FINAL_STATUS='N17_ALL18700_FIXED_DEV_CPU_NUMERIC_AND_FULL_COCO_COMPLETE_PENDING_ROOT_ACTUAL0'
ROOT_SCHEMA='ROOT-N17-fixedDEV-full187-GPU-CPU-COCO-qualified-result-v1'
ROOT_STATUS='ROOT_N17_ALL18700_FIXED_DEV_GPU_CPU_FULL_COCO_QUALIFIED'
LOCAL_TRUE_FLAGS=('parent_original_GPU0_CUDA0_bound','all_actual_GPU_scout_logits_two_CPU_formulas_absolute1e4',
    'all_actual_extra8_native_grid_byte_exact','all_selected_indices_masks_RLE_exact','all_PR_two_formulas_absolute1e4_strictsign',
    'all_native_R_and_new_score_class_query_order_exact','independent_CPU_proof_complete')
LOCAL_FALSE_FLAGS=('qualified','ROOT_actual_CPU0_verified','current_owned_CPU_terminal_self_certified','DEV_winner_selected',
    'CPU_scout_forward_used_for_ranking','CPU_F_rerendered','AP_evaluated','official_COCO_called','failed_partials_reused','CUDA_initialized')

def validate_final_header(spec):
    require(type(spec) is dict and spec['schema']==CPU_SCHEMA and
        all(type(spec[k]) is int for k in ('normal_RAW_count','expected_normal_RAW_count','total_failed_RAW_count',
            'total_terminal_identity_pin_count','deadline_seconds','initialization_seed','maximum_process_RSS_bytes')) and
        spec['normal_RAW_count']==spec['expected_normal_RAW_count'] and spec['normal_RAW_count']>126 and
        spec['total_failed_RAW_count']==20 and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+20 and
        spec['initialization_seed']==202610150171 and spec['maximum_process_RSS_bytes']==16*1024**3 and spec['deadline_seconds']==1800 and
        spec['automatic_retry'] is False and spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192 and
        Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True),
        'Own final CPU Source/schema/explicit ROOT prefix/failed19/resources required')
    return True

def validated_CPU_descriptors(records,current_normal):
    require(type(records) is list and len(records)==5 and type(current_normal) is int,'Exactly five explicit actual CPU descriptors')
    previous=126
    for i,record in enumerate(records):
        require(type(record) is dict and set(record)==CPU_DESCRIPTOR_KEYS and type(record['cpu_shard_id']) is int and record['cpu_shard_id']==i and
            record['cpu_dataset_ordinal_range']==list(RANGES[i]) and type(record['cpu_shard_owner_index']) is int and
            previous<record['cpu_shard_owner_index']<current_normal,'Five prelocked scopes and explicit chronological CPU owners after original GPU122..126')
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):ref_shape(record[key])
        previous=record['cpu_shard_owner_index']
    for key in ('spec','report','RAW','admission','phase_binding','bootstrap'):
        require(len({r[key]['path'] for r in records})==5,'Distinct completed CPU artifact: '+key)
    return records

def merge_policy_predictions(pieces,image_ids,native,original_GPU_predictions):
    require(type(pieces) is list and len(pieces)==5 and type(image_ids) is list and len(image_ids)==187 and len(set(image_ids))==187 and
        type(native) is list and len(native)==18700 and type(original_GPU_predictions) is list and len(original_GPU_predictions)==18700,
        'Exact complete original18700 population and five pieces required')
    combined=[]
    for values,(first,last) in zip(pieces,RANGES):
        expected=native[first*100:last*100]
        require(type(values) is list and len(values)==(last-first)*100 and
            [v['image_id'] for v in values]==[image for image in image_ids[first:last] for _ in range(100)] and
            all(type(v) is dict and 'segmentation' in v and {k:x for k,x in v.items() if k!='segmentation'}==
                {k:x for k,x in n.items() if k!='segmentation'} for v,n in zip(values,expected)),
            'Merged CPU shard cannot alter original class/S0/native-query/ordinal order')
        combined.extend(values)
    require(combined==original_GPU_predictions and len(combined)==18700,'Every complete mask/RLE must exactly match original four GPU prediction refs')
    return combined

def AP_percent(value):
    require(type(value) in (int,float) and math.isfinite(value) and (value==-1 or 0<=value<=1),'COCO AP raw scale or legal undefined -1 required')
    return -1 if value==-1 else value*100

def main_preview(interpreter,results):
    require(interpreter.STRONG_AP==68.09938494745545 and tuple(interpreter.SEEDS)==(202610150171,202610150172,202610150173),
        'Exact prelocked independent main interpreter ABI')
    return api(interpreter,'interpret_fixed_main')([dict(seed=202610150171,epoch=24,mode='atmost',AP_unit='percent',
        LOCAL_AP=AP_percent(results['E24_LOCAL_ERROR_atmost']['stats'][0]),CENTER_AP=AP_percent(results['E24_CENTER_ERROR_atmost']['stats'][0]),
        strong_AP=AP_percent(results['strong_fixed_E24_Omega']['stats'][0]))])

def validate_PR_stats(stats,points):
    require(type(stats) is dict and set(stats)=={'original_head','independent_PR'},'Two preserved PR formula summaries')
    for value in stats.values():
        require(type(value) is dict and value['calls']==(points+2047)//2048 and value['points']==points and value['strict_sign_exact'] is True and
            all(type(value[k]) in (int,float) and math.isfinite(value[k]) and 0<=value[k]<=1e-4 for k in ('residual_max_absolute_error','combined_max_absolute_error')),
            'Original complete PR formula counts/absolute1e-4/strict sign summaries')

def validate_image_proof(record,row,names):
    require(type(record) is dict and set(record)=={'dataset_index','image_id','queries','bundle','input_carrier','consumption','details'} and
        record['dataset_index']==row['dataset_index'] and record['image_id']==row['image_id'] and record['queries']==100 and
        record['bundle']==row['bundle'] and record['input_carrier']==row['input_carrier'],'Exact numeric image/ref/carrier/all100 binding')
    consumption=record['consumption'];members=len(row['member_locators'])
    require(consumption['all_registered_members_consumed'] is True and type(consumption['members']) is int and consumption['members']==members and
        consumption['queries']==100 and consumption['ROOT_scientific_qualification_conferred'] is False and members>0,
        'All and only registered members numerically consumed, no self qualification')
    details=record['details'];require(type(details) is list and len(details)==100 and [d['query_id'] for d in details]==list(range(100)),
        'Complete original natural100 query proofs')
    scout_names=tuple('E%d_%s'%(e,a) for e in (24,8) for a in ('LOCAL_ERROR','CENTER_ERROR'))
    for q,item in enumerate(details):
        query=row['query_profiles'][q];K=query['K'];C=query['C_count'];active=C if K else 0
        require(item['K']==K and item['C_points']==C and item['all_selected_indices_exact'] is True and item['all_full_masks_and_RLE_exact'] is True and
            item['actual_extra8_native_grid_byte_exact'] is True and item['CPU_scout_outputs_used_for_ranking'] is False and
            item['atol']==1e-4 and item['rtol']==0 and item['AP_evaluated'] is False and item['ROOT_scientific_qualification_conferred'] is False and
            set(item['profiles'])==set(names) and set(item['scout_numeric'])==set(scout_names),'Fixed all8 masks and four actual-logit formula proofs')
        common=item['common_C_PR'];require(common['strict_sign_exact'] is True,'Actual common C strict sign')
        validate_PR_stats(common['formula_stats'],active)
        require(common['formula_chunk_calls']=={k:v['calls'] for k,v in common['formula_stats'].items()},'Exact common PR chunk summary')
        for stat in item['scout_numeric'].values():
            require(stat['chunk_calls']==(active+2047)//2048 and stat['points']==active and stat['actual_extra8_native_grid_byte_exact'] is True and
                all(type(stat[k]) in (int,float) and math.isfinite(stat[k]) and 0<=stat[k]<=1e-4 for k in
                    ('factory_vs_actual_GPU_max_absolute_error','independent_vs_actual_GPU_max_absolute_error')),
                'Both saved CPU formulas vs same actualGPU absolute1e-4, original all-C chunk counts')
        for name,profile in item['profiles'].items():
            epoch,arm,mode=name.split('_',1)[0],profile['arm'],profile['mode']
            require(name=='E%d_%s_%s'%(profile['epoch'],arm,mode) and profile['epoch'] in (24,8) and arm in ('LOCAL_ERROR','CENTER_ERROR') and mode in ('atmost','exact') and
                type(profile['selected_points']) is int and 0<=profile['selected_points']<=K and (mode!='exact' or profile['selected_points']==K) and
                all(profile[k] is True for k in ('selected_indices_exact','full_mask_exact','outside_selected_native_exact','RLE_exact','PR_strict_sign_exact')) and
                profile['CPU_scout_outputs_used_for_ranking'] is False and profile['CPU_approximate_PR_used_for_ranking'] is False and
                profile['main_result']==(profile['epoch']==24 and mode=='atmost'),'Original epoch/arm/mode/counts/exact mask/RLE and actualGPU ranking flags')
            validate_PR_stats(profile['selected_PR_formula_stats'],profile['selected_points'])
    return members

def source_gate(path,spec):
    validate_final_header(spec);records=validated_CPU_descriptors(spec['numeric_CPU_shards'],spec['normal_RAW_count'])
    gate=inherited_source_gate(path,spec,cpu=True);c=gate['closure']
    for key in ('numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source','execution_partition_supplement'):closed(c,spec[key])
    numeric=load(spec['numeric_CPU_shard_gate_source'],'_N17_closed_numeric_CPU_report_contract')
    gpu=load(spec['GPU_outer_source'],'_N17_closed_GPU03_metadata_helpers')
    require(numeric.CPU_SCHEMA!=CPU_SCHEMA and gpu.SCHEMA==SCHEMA and numeric.CPU_REPORT_SCHEMA=='N17-fixedDEV-local-center-error-CPU-numeric-shard-report-v1',
        'Distinct own final schema and exact closed N17 CPU/GPU report ABIs')
    for key in gpu.NEW_REFS+('ROOTtrain_result','legacy_qualified_result','new_eval_body_source','legacy_eval_body_source'):closed(c,spec[key])
    lock=gpu.validate_experiment_lock(spec,c);gpu.validate_neighbor_cache(spec,gate,lock);gate['experiment_lock']=lock
    gpu.training_gate(spec,gate);gpu.legacy_gate(spec,gate)
    gate['execution_partition_supplement']=gpu.validate_execution_partition_supplement(spec,c)
    evidence=[];proof_refs=[];all_rows=[];members=0;GPUowners=[]
    for descriptor in records:
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):closed(c,descriptor[key])
        require(descriptor['source']['sha256']==spec['numeric_CPU_shard_gate_source']['sha256'] and descriptor['source']['bytes']==spec['numeric_CPU_shard_gate_source']['bytes'],
            'All five completed CPU shards execute same frozen qualified numerical source')
        parent=read(closed(c,descriptor['spec']));report=read(closed(c,descriptor['report']));i=descriptor['cpu_shard_id'];first,last=RANGES[i];owner=descriptor['cpu_shard_owner_index']
        completed_numeric=load(descriptor['source'],'_N17_completed_CPU_scope_%d'%i)
        completed_numeric.validate_cpu_header(parent)
        joined=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],descriptor['RAW'],descriptor['admission'],descriptor['source'],descriptor['spec'],'')
        require(type(joined) is int and joined==owner==parent['normal_RAW_count'] and parent['RAW_inputs']==spec['RAW_inputs'][:owner] and
            parent['entry_source']==descriptor['source'] and parent['cpu_shard_id']==i and parent['cpu_dataset_ordinal_range']==list(RANGES[i]) and
            parent['total_failed_RAW_count']==20 and parent['total_terminal_identity_pin_count']==owner+20 and
            spec['RAW_inputs'][owner]['receipt']==descriptor['RAW'] and spec['RAW_inputs'][owner]['admission']==descriptor['admission'] and
            spec['RAW_inputs'][owner]['phase_binding']==descriptor['phase_binding'] and spec['RAW_inputs'][owner]['bootstrap']==descriptor['bootstrap'] and
            all(c.get(r['path'])==r for r in parent['sources']) and all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
            'Actual CPUempty0/source/spec/ad/phase/bootstrap/fullSource/pins and explicit chronological prefix exact')
        for key in gpu.NEW_REFS+tuple(k for k in numeric.CPU_NUMERIC_REFS if k!='GPU_outer_source')+('ROOTtrain_result','legacy_qualified_result','legacy_shards','new_eval_body_source','legacy_eval_body_source','execution_partition_supplement'):
            require(parent[key]==spec[key],'Five CPU leaves preserve same qualified provenance/formulas/frozen configuration: '+key)
        closed(c,parent['GPU_outer_source'])
        parent_gpu=load(parent['GPU_outer_source'],'_N17_completed_parent_GPU_metadata_%d'%i)
        require(parent['GPU_outer_source']==parent['GPU_shard']['source'] and parent_gpu.policy_names()==gpu.policy_names(),
            'Each physical GPU source remains exact; old03 physical0 and new04 others are distinct')
        child=dict(gate);numeric.parent_GPU_gate(parent,child,parent_gpu);GPUowners.append(parent['GPU_shard']['shard_owner_index'])
        rows=child['parent_GPU_manifest']['rows'];ids=[r['image_id'] for r in rows]
        require(report['schema']==numeric.CPU_REPORT_SCHEMA and report['status']==numeric.CPU_STATUS and report['source']==descriptor['source'] and
            report['spec']==descriptor['spec'] and report['CPU_admission']==descriptor['admission'] and report['phase']==parent['phase'] and
            report['GPU_shard']==parent['GPU_shard'] and report['capture_manifest']==child['parent_GPU_report']['capture_manifest'] and
            report['cpu_shard_id']==i and report['cpu_dataset_ordinal_range']==list(RANGES[i]) and report['cpu_shard_owner_index']==owner and
            report['normal_RAW_count']==owner and report['total_failed_RAW_count']==20 and report['total_terminal_identity_pin_count']==owner+20 and
            report['scope_dataset_indices']==list(range(first,last)) and report['scope_image_ids']==ids and report['images']==last-first and report['queries']==(last-first)*100 and
            report['global_images']==187 and report['global_queries']==18700 and report['global_GT']==380 and tuple(report['controls'])==gpu.policy_names() and
            report['fixed_main_epoch']==24 and report['fixed_curve_epoch']==8 and report['numeric_atol']==1e-4 and report['numeric_rtol']==0 and
            all(report[k] is True for k in LOCAL_TRUE_FLAGS) and all(report[k] is False for k in LOCAL_FALSE_FLAGS) and
            report['capture_refs']==child['parent_GPU_report']['capture_refs'] and report['actual_capture_files_consumed']==last-first and
            len(report['image_proofs'])==last-first and set(report['control_results'])==set(gpu.policy_names()),
            'Immutable complete original numerical Q/report, fixed scope, original actual captured GPU parent, no local AP/self terminal claims')
        for key in ('ROOTtrain_result','legacy_qualified_result','experiment_lock','execution_partition_supplement','CPU_adapter_source','selector_cpu_proof_source','legacy_selector_proof_source','endpoint_body_source','PR_formula_source'):
            require(report[key]==spec[key],'Numerical report exact binding: '+key)
        require(report['checkpoints']==gate['training']['root']['checkpoints'],'Same actual E8/E24 qualified checkpoint refs')
        count=0
        for proof_ref,row in zip(report['image_proofs'],rows):
            target=closed(c,proof_ref);require(target.resolve(strict=True)==(Path(parent['output_directory'])/('IMAGE_%03d_PROOF.json'%row['dataset_index'])).resolve(strict=True),
                'Only actual successful numeric owner imageproof output, no failed partials')
            count+=validate_image_proof(read(target),row,gpu.policy_names()[2:]);proof_refs.append(proof_ref)
        require(count==report['registered_bundle_members_consumed']==sum(len(row['member_locators']) for row in rows),'Exact complete numerical member count')
        for name in gpu.policy_names():
            result=report['control_results'][name];require(type(result) is dict and set(result)=={'predictions'},'Local predictions only, no local COCO')
            values=read(closed(c,result['predictions']));require(values==child['expected_predictions'][name],'CPU full local masks/RLE/class/S0/order exact original actualGPU refs')
        all_rows.extend(rows);members+=count;evidence.append(dict(descriptor=descriptor,parent=parent,report=report,child=child))
    require(GPUowners==[122,123,124,125,126] and len(all_rows)==187 and [row['dataset_index'] for row in all_rows]==list(range(187)) and
        len({row['image_id'] for row in all_rows})==187 and len(proof_refs)==len({r['path'] for r in proof_refs})==187,
        'All five exact GPU122..126 and CPU completed partitions cover original187 whole images/18700 queries')
    gate.update(gpu_helpers=gpu,numeric_CPU_evidence=evidence,full_rows=all_rows,image_proofs=proof_refs,registered_members=members);return gate

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);require(Path(spec_path).is_absolute(),'Own absolute final CPU spec')
    spec=read(path);gate=source_gate(path,spec);c=gate['closure'];context=gate['context'];gpu=gate['gpu_helpers']
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh admitted final CPU output')
    import psutil
    process=psutil.Process();require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
        psutil.virtual_memory().available>=16*1024**3 and psutil.disk_usage(str(output.parent)).free>=1024**3,'Actual CPU01/CUDAempty/threads2/16GiB RAM/free1GiB')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and not torch.cuda.is_initialized(),'Closed cold final CPU runtime; no model load')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    reader=load(context['reader_source'],'_N17_original_canonical_population');official=load(context['COCO_source'],'_N17_original_official_full_COCO')
    interpreter=load(spec['interpreter_source'],'_N17_frozen_main_only_interpreter')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    cocoeval_module=sys.modules[COCOeval.__module__];closed(c,reference(cocoeval_module.__file__))
    outside=torch.get_rng_state().clone()
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and not torch.cuda.is_initialized() and
            process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
            torch.get_num_threads()==torch.get_num_interop_threads()==2,'Final CPU bounds/no retry/no model/trajectory rerun')
    coco=COCO(str(closed(c,context['DEV'])));ids=list(coco.imgs)
    require(len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1} and ids==[item['id'] for item in coco.dataset['images']] and
        [row['image_id'] for row in gate['full_rows']]==ids,'Original full187/GT380/category1/native image order, no annotation normalization')
    original_results=gate['legacy_root']['control_results'];native=read(closed(c,original_results['native_official']['predictions']))
    strong=read(closed(c,original_results['strong_fixed_E24_Omega']['predictions']));reader.population(coco,native,ids);reader.population(coco,strong,ids)
    output.mkdir();results={}
    for name in gpu.policy_names():
        check();pieces=[read(closed(c,item['report']['control_results'][name]['predictions'])) for item in gate['numeric_CPU_evidence']]
        original=[value for item in gate['numeric_CPU_evidence'] for value in item['child']['expected_predictions'][name]]
        values=merge_policy_predictions(pieces,ids,native,original);reader.population(coco,values,ids)
        if name=='native_official':require(values==native,'Original complete native raw predictions exact')
        if name=='strong_fixed_E24_Omega':require(values==strong,'Original complete R raw predictions exact')
        target=output/(name+'_PREDICTIONS.json');write(target,values)
        metrics=api(official,'coco_metrics')(coco,values,ids,COCOeval);result=dict(predictions=reference(target),COCO=metrics,stats=metrics['stats'])
        gpu.validate_legacy_COCO(result)
        if name in original_results:require(result['stats']==original_results[name]['stats'] and result['COCO']==original_results[name]['COCO'],
            'Native/R all12 stats and inline official COCO semantics exact original qualified ROOT; raw0..1 not percent')
        results[name]=result;del values,pieces,original,metrics,result;gc.collect()
    preview=main_preview(interpreter,results)
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['CPU_admission_file'])==gate['admission_ref'] and
        torch.equal(outside,torch.get_rng_state()) and gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)==libraries,
        'Own final Source/spec/admission/RNG/imported libraries exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],numeric_CPU_shards=spec['numeric_CPU_shards'],
        numeric_CPU_shard_gate_source=spec['numeric_CPU_shard_gate_source'],GPU_outer_source=spec['GPU_outer_source'],ROOTtrain_result=spec['ROOTtrain_result'],
        legacy_qualified_result=spec['legacy_qualified_result'],experiment_lock=spec['experiment_lock'],execution_partition_supplement=spec['execution_partition_supplement'],interpreter_source=spec['interpreter_source'],
        checkpoints=gate['training']['root']['checkpoints'])
    details=dict(schema='N17-fixedDEV-full187-CPU-COCO-details-v1',**binding,control_results=results,image_proofs=gate['image_proofs'],
        capture_manifests=[item['child']['parent_GPU_report']['capture_manifest'] for item in gate['numeric_CPU_evidence']],
        capture_refs=[ref for item in gate['numeric_CPU_evidence'] for ref in item['child']['parent_GPU_report']['capture_refs']],
        registered_bundle_members_consumed=gate['registered_members'],actual_imported_files=libraries,numerical_forward_rerun=False,head_loaded=False,local_AP_averaged=False)
    write(output/'DETAILS.json',details)
    write(output/'Q.json',dict(schema=FINAL_SCHEMA,status=FINAL_STATUS,**binding,details=reference(output/'DETAILS.json'),images=187,GT=380,natural_queries=18700,
        policies=gpu.policy_names(),control_results=results,initialization_seed=202610150171,normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=20,
        total_terminal_identity_pin_count=spec['normal_RAW_count']+20,primary_interpretation_preview_pending_ROOT_actual0=preview,
        checks=dict(five_original_GPU0_CPU0_Source_spec_ad_phase_prefix_exact=True,all187_all100_registered_numerical_imageproofs_complete=True,
            all_actual_GPU_logits_two_CPU_formulas_absolute1e4_and_exact_selected_PR_masks_RLE_preserved=True,
            all10_full18700_predictions_scores_classes_native_query_order_exact=True,all10_original_official_full_COCO12_semantics=True,
            native_R_all12stats_inline_COCO_and_full_predictions_exact=True,seed171_main_only_prelocked_interpreter_Source_exact=True,all20_failures_preserved_not_normal=True),
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,numerical_forward_rerun=False,head_loaded=False,
        CUDA_initialized=False,local_AP_averaged=False,CPU_scout_forward_used_for_ranking=False,AP_results_interpreted_as_qualified=False,
        task_efficacy_established=False,innovation_established=False,TGRS_readiness_established=False,elapsed_seconds=time.monotonic()-started,
        evidence_scope='Internal DEV187 under frozen teacher/PR, same paired seed171/budget/proxy labels; one qualified seed cannot establish innovation or TEST/generalization.'))
    print(json.dumps(dict(status=FINAL_STATUS,Q=reference(output/'Q.json'),qualified=False)),flush=True)

def root_qualified_result(final_record,completed_gate):
    """ROOT only AFTER original final RAW0 and completed Source-gated prefix; never called by run."""
    keys={'source','spec','Q','RAW','admission','phase_binding','bootstrap','owner_index'}
    require(type(final_record) is dict and set(final_record)==keys and type(final_record['owner_index']) is int,'Exact explicit completed original final-owner record')
    for key in keys-{'owner_index'}:ref_shape(final_record[key])
    c=completed_gate['closure'];context=completed_gate['context'];owner=final_record['owner_index']
    require(context['normal_RAW_count']>owner and len(context['RAW_inputs'])==context['normal_RAW_count'],'ROOT completed normal prefix must actually include original final owner')
    row=context['RAW_inputs'][owner]
    require(row['receipt']==final_record['RAW'] and row['admission']==final_record['admission'] and row['phase_binding']==final_record['phase_binding'] and
        row['bootstrap']==final_record['bootstrap'],'Completed original final RAW/ad/phase/bootstrap exact, never live self certification')
    raw=terminal_raw_only(final_record['RAW'],row['phase'],0);check_pin(context['terminal_identity_pins'][final_record['RAW']['path']],raw,row['phase'],0)
    for key in keys-{'owner_index'}:closed(c,final_record[key])
    joined=completed_gate['lineage'].owner_ref_join(context,completed_gate['owners'],final_record['RAW'],final_record['admission'],final_record['source'],final_record['spec'],'')
    spec=read(closed(c,final_record['spec']));Q=read(closed(c,final_record['Q']))
    require(type(joined) is int and joined==owner==spec['normal_RAW_count'] and spec['schema']==CPU_SCHEMA and spec['entry_source']==final_record['source'] and
        spec['RAW_inputs']==context['RAW_inputs'][:owner] and all(c.get(r['path'])==r for r in spec['sources']) and
        Q['schema']==FINAL_SCHEMA and Q['status']==FINAL_STATUS and Q['source']==final_record['source'] and Q['spec']==final_record['spec'] and
        Q['CPU_admission']==final_record['admission'] and Q['phase']==row['phase']==spec['phase'] and Q['normal_RAW_count']==owner and
        Q['total_failed_RAW_count']==20 and Q['total_terminal_identity_pin_count']==owner+20 and Q['qualified'] is False and
        Q['ROOT_actual_CPU0_verified'] is False and Q['current_owned_CPU_terminal_self_certified'] is False and
        Q['images']==187 and Q['GT']==380 and Q['natural_queries']==18700 and Q['initialization_seed']==202610150171 and
        type(Q['checks']) is dict and bool(Q['checks']) and all(v is True for v in Q['checks'].values()),
        'Exact immutable completed pending Q and Source-gated actual final CPUempty0 required; current run cannot qualify itself')
    details=read(closed(c,Q['details']));require(details['control_results']==Q['control_results'] and details['image_proofs'] and len(details['image_proofs'])==187,
        'Original completed DETAILS/control/proof refs exact')
    gpu=load(spec['GPU_outer_source'],'_N17_ROOT_completed_fixed_COCO_semantics');interpreter=load(spec['interpreter_source'],'_N17_ROOT_completed_main_interpreter')
    closed(c,Q['execution_partition_supplement'])
    require(Q['execution_partition_supplement']==spec['execution_partition_supplement'],'Qualified final retains exact locked execution supplement')
    gpu.validate_execution_partition_supplement(spec,c)
    for result in Q['control_results'].values():gpu.validate_legacy_COCO(result);closed(c,result['predictions'])
    for ref in details['image_proofs']+details['capture_manifests']+details['capture_refs']:closed(c,ref)
    interpreted=main_preview(interpreter,Q['control_results']);require(interpreted==Q['primary_interpretation_preview_pending_ROOT_actual0'],'ROOT qualified main interpretation must equal frozen pending preview')
    return dict(schema=ROOT_SCHEMA,status=ROOT_STATUS,qualified=True,original_final_CPU_actual0_verified=True,final_owner=final_record,
        final_Q=final_record['Q'],final_details=Q['details'],numeric_CPU_shards=Q['numeric_CPU_shards'],ROOTtrain_result=Q['ROOTtrain_result'],
        legacy_qualified_result=Q['legacy_qualified_result'],experiment_lock=Q['experiment_lock'],execution_partition_supplement=Q['execution_partition_supplement'],checkpoints=Q['checkpoints'],control_results=Q['control_results'],
        images=187,GT=380,natural_queries=18700,initialization_seed=202610150171,primary_interpretation=interpreted,
        normal_after=context['normal_RAW_count'],preserved_failures=20,pins_after=context['normal_RAW_count']+20,
        eligible_for_locked_main_rule=True,formal_TEST_evaluated=False,numerical_forward_rerun=False,local_AP_averaged=False,
        optimizer_trajectory_replayed=False,innovation_established=False,TGRS_readiness_established=False,evidence_scope=Q['evidence_scope'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND: ROOT binds original five GPU0/CPU0, exact Source/20 failures/official full COCO; no auto start')
    run(args.spec)
