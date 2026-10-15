# UNBOUND_NOT_EXECUTED: 32 CPU post, ordinary text assembly only.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 26 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 26 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Original failed9 and typed21/22/23/24/25/26 stay separate; normal+26 terminal identities')
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
CPU_SCHEMA = "ROOT-N19-fixedDEV-full187-fixed-half-readout-CPU-spec-v1"
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
    require(spec['schema'] == (N20_FINAL_CPU_SCHEMA if cpu else N20_DEV_SCHEMA) and
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


"""UNBOUND typed failure22 extension; supplied helpers belong to original RAW front.

Raw-only hook must precede inherited_source_gate and every SOURCE hash.
Closed hook runs only after all original normal/failed terminals have passed.
No execution, IO or retry takes place on import. The original failed source,
specification, receipt and absent training outputs are retained unchanged.
"""


def n20_failed22_raw_only(spec):
    extra = spec['additional_preserved_N20_training_output_precondition_failure']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding',
        'bootstrap', 'entry_source', 'spec', 'original_report_absent'},
        'Exact original N20 output-precondition failure22 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True, 'Original failure22 actual1 and report absence')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        ref_shape(extra[key])
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior.append(spec['additional_preserved_N17_independent_CPU_SOURCE_closure_failure']['RAW_ref']['path'])
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Original failure22 owner distinct from all normal and failed1..21')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(raw['error'] is None and raw['abort'] is None,
        'Ordinary startup failure22, never model failure or controlled resource abort')
    return raw


def n20_failed22_closed(spec, gate, raw):
    extra = spec['additional_preserved_N20_training_output_precondition_failure']
    closure, lineage = gate['closure'], gate['lineage']
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, extra[key])
    row = dict(receipt=extra['RAW_ref'], phase=extra['phase'], admission=extra['admission'],
        phase_binding=extra['phase_binding'], bootstrap=extra['bootstrap'])
    owner = lineage.owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path']
        and owner['cuda_visible_devices'] == '0', 'Original failure22 exact GPU0 owner/source/spec')
    original = read(closed(closure, extra['spec']))
    original_output = Path(original['output_directory'])
    require(original['schema'] == 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
        and original['entry_source'] == extra['entry_source'] and original['phase'] == extra['phase']
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 140
        and original['total_failed_RAW_count'] == 21
        and original['total_terminal_identity_pin_count'] == 161
        and original['RAW_inputs'] == spec['RAW_inputs'][:140]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in original['terminal_identity_pins'].items())
        and all(original[key] == spec[key] for key in (
            'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
            'N20_initialization_lock', 'expected_N20_initialization_digest',
            'N20_locked_protocol', 'expected_N20_locked_protocol', 'locked_protocol',
            'scout_module_source', 'base_scout_source', 'paired_training_body_source',
            'original_paired_training_body_source', 'N20_training_tail_source', 'initialization_seed'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and not original_output.exists() and not original_output.is_symlink()
        and Path(original['GPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True),
        'Original failed22 source/prefix/pins/unchanged science and absent outputs preserved')
    return owner

"""UNBOUND typed failure23 extension; supplied helpers belong to original RAW front.

Raw-only hook must precede inherited_source_gate and every SOURCE hash.
Closed hook runs only after all original normal/failed terminals have passed.
No execution, IO or retry takes place on import. The original failed source,
specification, receipt and absent training outputs are retained unchanged.
"""


def n20_failed23_raw_only(spec):
    extra = spec['additional_preserved_N20_current_failure_counter_failure']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding',
        'bootstrap', 'entry_source', 'spec', 'original_report_absent'},
        'Exact original N20 output-precondition failure23 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True, 'Original failure23 actual1 and report absence')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        ref_shape(extra[key])
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior.append(spec['additional_preserved_N17_independent_CPU_SOURCE_closure_failure']['RAW_ref']['path'])
    prior.append(spec['additional_preserved_N20_training_output_precondition_failure']['RAW_ref']['path'])
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Original failure23 owner distinct from all normal and failed1..21')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(raw['error'] is None and raw['abort'] is None,
        'Ordinary startup failure23, never model failure or controlled resource abort')
    return raw


def n20_failed23_closed(spec, gate, raw):
    extra = spec['additional_preserved_N20_current_failure_counter_failure']
    closure, lineage = gate['closure'], gate['lineage']
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, extra[key])
    row = dict(receipt=extra['RAW_ref'], phase=extra['phase'], admission=extra['admission'],
        phase_binding=extra['phase_binding'], bootstrap=extra['bootstrap'])
    owner = lineage.owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path']
        and owner['cuda_visible_devices'] == '0', 'Original failure23 exact GPU0 owner/source/spec')
    original = read(closed(closure, extra['spec']))
    original_output = Path(original['output_directory'])
    require(original['schema'] == 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
        and original['entry_source'] == extra['entry_source'] and original['phase'] == extra['phase']
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 140
        and original['total_failed_RAW_count'] == 22
        and original['total_terminal_identity_pin_count'] == 162
        and original['RAW_inputs'] == spec['RAW_inputs'][:140]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in original['terminal_identity_pins'].items())
        and all(original[key] == spec[key] for key in (
            'additional_preserved_N20_training_output_precondition_failure',
            'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
            'N20_initialization_lock', 'expected_N20_initialization_digest',
            'N20_locked_protocol', 'expected_N20_locked_protocol', 'locked_protocol',
            'scout_module_source', 'base_scout_source', 'paired_training_body_source',
            'original_paired_training_body_source', 'N20_training_tail_source', 'initialization_seed'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and not original_output.exists() and not original_output.is_symlink()
        and Path(original['GPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True),
        'Original failure23 counter-check exit1 source/prefix/pins/unchanged science and absent outputs preserved')
    return owner

"""UNBOUND terminal/source closure for original trained GPU cleanup failure24.

This preserves original actual1 and missing REPORT. It does not certify saved
weights, regenerate an original report, run training or turn failed GPU into0.
CPU recovery must independently validate all saved tensors and endpoints.
"""


def n20_failed24_raw_only(spec):
    extra = spec['N20_failed_training_record']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding',
        'bootstrap', 'entry_source', 'spec', 'original_report_absent'},
        'Exact preserved failed training24 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True, 'Original GPU exit1/report absence retained')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        ref_shape(extra[key])
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior += [spec[key]['RAW_ref']['path'] for key in (
        'additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure')]
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Failed24 remains separate from every normal and failed1..23 owner')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(raw['error'] is None and raw['abort'] is None,
        'Original cleanup exception is ordinary exit1, never successful or resource-aborted training')
    return raw


def n20_failed24_closed(spec, gate, raw):
    extra, closure = spec['N20_failed_training_record'], gate['closure']
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, extra[key])
    row = dict(receipt=extra['RAW_ref'], phase=extra['phase'], admission=extra['admission'],
        phase_binding=extra['phase_binding'], bootstrap=extra['bootstrap'])
    owner = gate['lineage'].owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path'] and owner['cuda_visible_devices'] == '0',
        'Original failed24 GPU0 owner/source/spec/phase/bootstrap association')
    original = read(closed(closure, extra['spec']))
    output = Path(original['output_directory'])
    require(original['schema'] == 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
        and original['phase'] == extra['phase'] and original['entry_source'] == extra['entry_source']
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 140
        and original['total_failed_RAW_count'] == 23 and original['total_terminal_identity_pin_count'] == 163
        and original['RAW_inputs'] == spec['RAW_inputs'][:140]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in original['terminal_identity_pins'].items())
        and all(original[key] == spec[key] for key in (
            'additional_preserved_N20_training_output_precondition_failure',
            'additional_preserved_N20_current_failure_counter_failure',
            'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
            'N20_initialization_lock', 'expected_N20_initialization_digest',
            'N20_locked_protocol', 'expected_N20_locked_protocol', 'locked_protocol',
            'scout_module_source', 'base_scout_source', 'paired_training_body_source',
            'original_paired_training_body_source', 'N20_training_tail_source', 'initialization_seed'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and output.is_dir() and not output.is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink()
        and Path(original['GPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True),
        'Original failed24 source/history/science and saved directory with missing REPORT preserved')
    stderr_ref = spec['N20_failed_training_stderr']
    stderr_path = Path(closed(closure, stderr_ref))
    require(stderr_ref['sha256'] == raw['logs']['stderr.log']
        and stderr_path.resolve(strict=True) ==
            (Path(extra['RAW_ref']['path']).parent / 'stderr.log').resolve(strict=True),
        'Original failed24 actual terminal-bound stderr locator and bytes')
    error_text = stderr_path.read_text('utf8')
    require(error_text.rstrip().endswith(
        'RuntimeError: All model/optimizer CUDA tensors must be released after training')
        and extra['entry_source']['path'] in error_text,
        'Actual original source failed at the final CUDA cleanup check; no invented GPU success report')
    return owner

"""UNBOUND saved-artifact recovery CPU front; no prefix, tail, CLI or execution.

ROOT replaces only the inherited CURRENT CPU schema/counts: own140/failed24/
pins164. Historical cache GPU138/CPU139 remain21; the failed training GPU
specification remains own140/failed23/pins163 and is never a normal completion.
Required prefix hooks run raw22/raw23/raw24 before inherited_source_gate(...,
cpu=True), then their closed joins after it. Helper24 directly consumes the
exact10-field N20_failed_training_record and closes N20_failed_training_stderr.

New required spec fields: N20_failed_training_record, N20_failed_training_stderr,
N20_saved_training_artifacts={START,checkpoints:{'8','24'},FIT_endpoints:{'8','24'},
UPDATE_TRACE}. All six saved artifacts are closed refs to prescribed filenames
inside the preserved original output. Its REPORT.json must remain absent.
Cache/model/init/protocol fields keep the original CPU front contract, including
N20_training_tail_source (the original GPU tail, never this recovery tail).

Returns recovery_artifact_refs, recovery_start, recovery_training_binding (the
original20 fields), recovery_failed_GPU_record, recovery_training_parent, and
unchanged cache/common/parent bindings. No GPU REPORT is created or inferred.
ROOT alone qualifies the new CPU terminal; this front never imports Torch.
"""

import json
from pathlib import Path


N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'

"""UNBOUND typed preservation of original N20 DEV GPU141 capture failure25.

Append after ordinary typed22/23/24 helpers. n20_failed25_raw_only(spec) must run
before inherited_source_gate/first SOURCE SHA; n20_failed25_closed afterwards.
ROOT updates only CURRENT entry/ninth counts to25 (first new attempt141/25/166).
Historical failed GPU141 spec remains141/24/165; prior failures stay unchanged.
No import-time IO, retry, tensor read or scientific eligibility is supplied.
The existing START/CAPTURES are preserved partials, never completed evidence.
"""


N20_FAILURE25_IDENTITY = dict(pid=73252, birth_filetime_100ns='134365182757714398',
                             birth_unix=1792044675.7714405)
N20_FAILURE25_ADMISSION_SHA = '3097ddd35fdb5a4c204f836e9e89c5ce3c9ff3f291048ec276c9acbe3b5ac26b'
N20_FAILURE25_STDERR_SHA = 'a51f6d29d9de97350d5b25ba71c8d4900b8c7d136fb683e072730ff2caac1ff4'
N20_FAILURE25_STDERR_LAST_LINE = (
    'RuntimeError: self.stride(-1) must be 1 to view Long as Byte (different element sizes), but got 0')


def n20_failed25_raw_only(spec):
    extra = spec['N20_failed_DEV_capture_record']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding',
        'bootstrap', 'entry_source', 'spec', 'original_report_absent'},
        'Exact10 preserved N20 DEV capture failure25 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True,
        'Original DEV GPU exit1/report absence retained; never completed normal owner')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        ref_shape(extra[key])
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior += [spec[key]['RAW_ref']['path'] for key in (
        'additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure',
        'N20_failed_training_record')]
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Failed25 stays separate from every normal and failed1..24 owner')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(raw['original_identity'] == N20_FAILURE25_IDENTITY
        and raw['admission_sha256'] == extra['admission']['sha256'] == N20_FAILURE25_ADMISSION_SHA
        and raw['logs']['stderr.log'] == N20_FAILURE25_STDERR_SHA
        and raw['error'] is None and raw['abort'] is None,
        'Exact original pid/birth/admission/stderr; ordinary exit1, held/reaped/birth absent, no abort')
    return raw


def n20_failed25_closed(spec, gate, raw):
    extra, closure = spec['N20_failed_DEV_capture_record'], gate['closure']
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, extra[key])
    row = dict(receipt=extra['RAW_ref'], phase=extra['phase'], admission=extra['admission'],
        phase_binding=extra['phase_binding'], bootstrap=extra['bootstrap'])
    owner = gate['lineage'].owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path'] and owner['cuda_visible_devices'] == '0',
        'Original failed25 GPU0/source/spec/phase/bootstrap exact original owner')
    original = read(closed(closure, extra['spec']))
    output = Path(original['output_directory'])
    preserved_keys = (
        'N20_failed_training_record', 'N20_failed_training_stderr',
        'N20_root_training_qualified_result', 'expected_N20_root_training_qualified_result',
        'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_initialization_lock', 'expected_N20_initialization_digest',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'locked_protocol',
        'scout_module_source', 'base_scout_source', 'paired_training_body_source',
        'original_paired_training_body_source', 'N20_training_tail_source', 'initialization_seed',
        'cache_GPU_report', 'region_common_source', 'legacy_qualified_result', 'legacy_shards',
        'n17_root_qualified_result_ref', 'expected_n17_root_qualified_result',
        'new_eval_body_source', 'legacy_eval_body_source', 'capture_body_source',
        'coordinates_source', 'labels_source')
    require(original['schema'] == 'ROOT-N20-fixedDEV-local-center-feature-error-GPU-shard-spec-v1'
        and original['phase'] == extra['phase'] and original['entry_source'] == extra['entry_source']
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 141
        and original['total_failed_RAW_count'] == 24
        and original['total_terminal_identity_pin_count'] == 165
        and original['shard_id'] == 0 and original['dataset_ordinal_range'] == [0, 47]
        and original['RAW_inputs'] == spec['RAW_inputs'][:141]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in original['terminal_identity_pins'].items())
        and all(original[key] == spec[key] for key in preserved_keys)
        and all(original[key] == spec[key] for key in original if key.startswith('additional_preserved_'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['maximum_process_RSS_bytes'] == 16 * 1024**3
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and output.is_dir() and not output.is_symlink()
        and (output / 'START.json').is_file() and not (output / 'START.json').is_symlink()
        and (output / 'CAPTURES').is_dir() and not (output / 'CAPTURES').is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink()
        and Path(original['GPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True),
        'Original failed141/24/165 source/prefix/science and START/CAPTURES partials with absent REPORT')
    # CPU_adapter_source deliberately is not compared with the new own CPU's
    # F9 decoder: the original failed GPU used the old CEA2 common-PR decoder.
    # Entry/DEV-tail source may change for the isolated engineering fix; every
    # frozen model, scoring/capture body, protocol and prior carrier stays exact.
    stderr_ref = spec['N20_failed_DEV_capture_stderr']
    stderr_path = Path(closed(closure, stderr_ref))
    require(stderr_ref['bytes'] == 4679
        and stderr_ref['sha256'] == raw['logs']['stderr.log'] == N20_FAILURE25_STDERR_SHA
        and stderr_path.resolve(strict=True) ==
            (Path(extra['RAW_ref']['path']).parent / 'stderr.log').resolve(strict=True),
        'Original failed25 actual RAW-bound stderr locator/size/SHA exact')
    error_text = stderr_path.read_text('utf8')
    require(error_text.rstrip().endswith(N20_FAILURE25_STDERR_LAST_LINE)
        and extra['entry_source']['path'] in error_text and 'n20_same_bits' in error_text
        and '.view(torch.uint8)' in error_text,
        'Original exact Long-stride byte-view exception; no report reconstruction or partial qualification')
    return owner


"""UNBOUND_DRAFT: preserve original N20 CPU145 numeric deadline failure26.

Append after ordinary typed22..25 helpers. Call n20_failed26_raw_only(spec)
before inherited_source_gate/first SOURCE SHA; call n20_failed26_closed after.
ROOT supplies existing require/ref_shape/read/closed/check_pin/helpercheck and
FAILURE21_PRIOR_KEYS. No import-time IO, scientific tensor load or retry.
Current count is normal>=145 / failed26 / pins=normal+26. The old CPU spec stays
145/25/170. terminal_identity_pins contains normal+legacy8 only; record.pin is
independent, so its dictionary length is never compared with total pin count.
Thirteen failed partial proofs remain unqualified and are not read or reused.
"""
from pathlib import Path

N20_FAILURE26_IDENTITY = dict(pid=77760, birth_filetime_100ns='134365278288498169',
                            birth_unix=1792054228.8498173)
N20_FAILURE26_EMPTY_LOG_SHA = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
N20_FAILURE26_REF_KEYS = ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec')
N20_FAILURE26_PRESERVED_KEYS = (
    'N20_failed_training_record','N20_failed_training_stderr',
    'N20_failed_DEV_capture_record','N20_failed_DEV_capture_stderr',
    'N20_root_training_qualified_result','expected_N20_root_training_qualified_result',
    'N20_root_cache_qualified_result','expected_N20_root_cache_qualified_result',
    'N20_initialization_lock','expected_N20_initialization_digest',
    'N20_locked_protocol','expected_N20_locked_protocol','locked_protocol',
    'scout_module_source','base_scout_source','paired_training_body_source',
    'original_paired_training_body_source','N20_training_tail_source','initialization_seed',
    'cache_GPU_report','region_common_source','legacy_qualified_result','legacy_shards',
    'n17_root_qualified_result_ref','expected_n17_root_qualified_result',
    'new_eval_body_source','legacy_eval_body_source','capture_body_source',
    'coordinates_source','labels_source')


def n20_failed26_raw_only(spec):
    extra = spec['N20_failed_CPU_numeric_record']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref','phase','pin','returncode','admission','phase_binding',
        'bootstrap','entry_source','spec','original_report_absent'},
        'Exact10 original CPU numeric controlled-deadline failure26 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True and spec['failed26_partials_reused'] is False,
        'Failure26 stays exit1 with absent REPORT; failed partial numeric proofs never reused')
    for key in N20_FAILURE26_REF_KEYS:
        ref_shape(extra[key])
    ref_shape(spec['N20_failed26_helper_source'])
    normal = spec['normal_RAW_count']
    require(type(normal) is int and normal >= 145
        and type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] == normal
        and type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 26
        and type(spec['total_terminal_identity_pin_count']) is int
        and spec['total_terminal_identity_pin_count'] == normal + 26,
        'Current own count is normal>=145 plus preserved failed26; old parent counts are immutable')
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior += [spec[key]['RAW_ref']['path'] for key in (
        'additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure',
        'N20_failed_training_record','N20_failed_DEV_capture_record')]
    path = Path(extra['RAW_ref']['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(extra['phase']) is str and extra['phase']
        and path.parent.name == extra['phase']
        and extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Failure26 original RAW locator distinct from all normal owners and failed1..25')
    # terminal_raw_only deliberately is not used: that ordinary-exit helper rejects aborts.
    raw = read(path)
    decision = helpercheck(raw, extra['pin'])
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(decision['passed'] is True and decision['normal_success'] is False
        and decision['partial_caches_qualified'] is False
        and raw['original_identity'] == N20_FAILURE26_IDENTITY
        and raw['admission_sha256'] == extra['admission']['sha256']
        and raw['abort'] == 'QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE'
        and raw['elapsed_seconds'] == 1800.342999999877
        and raw['minimum_ram_bytes'] == 30790778880
        and raw['whole_GPU_observed_peak_MiB'] == 2146
        and raw['logs']['stdout.log'] == raw['logs']['stderr.log'] == N20_FAILURE26_EMPTY_LOG_SHA,
        'Exact original CPU pid/birth, controlled deadline, resources and empty logs; never normal0')
    return raw


def n20_failed26_closed(spec, gate, raw):
    extra, closure = spec['N20_failed_CPU_numeric_record'], gate['closure']
    for key in N20_FAILURE26_REF_KEYS:
        closed(closure, extra[key])
    closed(closure, spec['N20_failed26_helper_source'])
    row = dict(receipt=extra['RAW_ref'],phase=extra['phase'],admission=extra['admission'],
               phase_binding=extra['phase_binding'],bootstrap=extra['bootstrap'])
    owner = gate['lineage'].owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path'] and owner['cuda_visible_devices'] == '',
        'Original failed26 CPU-empty owner/source/spec/admission/phase/bootstrap exact')
    original = read(closed(closure, extra['spec']))
    output = Path(original['output_directory'])
    require(original['schema'] == 'ROOT-N20-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1'
        and original['phase'] == extra['phase'] and original['entry_source'] == extra['entry_source']
        and extra['entry_source']['bytes'] == 159357
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 145
        and original['total_failed_RAW_count'] == 25 and original['total_terminal_identity_pin_count'] == 170
        and original['cpu_shard_id'] == 0 and original['cpu_shard_owner_index'] == 145
        and original['CPU_parent_GPU_id'] == 0 and original['cpu_dataset_ordinal_range'] == [0,24]
        and original['RAW_inputs'] == spec['RAW_inputs'][:145]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key,value in original['terminal_identity_pins'].items())
        and original['completed_GPU_shards'] == spec['completed_GPU_shards']
        and all(original[key] == spec[key] for key in N20_FAILURE26_PRESERVED_KEYS)
        and all(original[key] == spec[key] for key in original if key.startswith('additional_preserved_'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['maximum_process_RSS_bytes'] == 16*1024**3
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and output.is_dir() and not output.is_symlink()
        and not (output/'REPORT.json').exists() and not (output/'REPORT.json').is_symlink()
        and Path(original['CPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True)
        and spec['failed26_partials_reused'] is False,
        'Original CPU145/25/170 source/prefix/frozen science retained; absent REPORT and no partial reuse')
    return owner


LEGACY_SHARD_RANGES=((0,47),(47,94),(94,141),(141,187))


OLD_CHECKS=('original_held0_reaped_birth_absent','whole_GPU_and_RAM_and_deadline','final_Q_Source_spec_admission_exact',
    'all_four_completed_parent_ref_and_legacy_counts_exact','all187_image_proof_refs_coordinates_all100_profiles_and_member_counts',
    'original_GPU_bindings_preserved','all12_full_COCO_semantics_and_AP113_stats_exact','all12_prediction_refs_exact','immutable_pending_Q_unchanged')


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


"""UNBOUND post-only N20 front+tail. ROOT appends typed26 prefix; no CLI.

The inherited current CPU dispatch uses N20_FINAL_CPU_SCHEMA; historical schemas
remain intact. Original final177 must actually complete before binding this own
CPU178 spec. Parent modules are called only through pure metadata/interpretation
helpers. No parent own gate, model, numeric replay, training or COCO is called.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time

N20_FINAL_CPU_SCHEMA = 'ROOT-N20-fixedDEV-full187-CPU-COCO-final-spec-v1'
N20_POST_RECORD_KEYS = {'source','spec','Q','RAW','admission','phase_binding','bootstrap','owner_index'}
N20_POST_SCHEMA = 'N20-fixedDEV-post-actual-final-ROOT-CPU-output-v1'
N20_POST_STATUS = 'N20_POST_ACTUAL_FINAL_ROOT_RESULT_WRITTEN_PENDING_POST_OWNER_ACTUAL0'
N20_POST_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
N20_POST_FINAL_CHECKS = (
    'four_original_DEV_GPU0_thirty_two_CPU0_Source_spec_ad_phase_prefix_exact',
    'all187_all100_registered_F9_FP64_score_numerical_imageproofs_complete',
    'all_actual_GPU_logits_two_CPU_formulas_absolute1e4_and_exact_selected_PR_masks_RLE_preserved',
    'all10_full18700_predictions_scores_classes_native_query_order_exact',
    'all10_original_official_full_COCO12_semantics','native_R_all12stats_inline_COCO_and_full_predictions_exact',
    'seed201_main_only_prelocked_interpreter_Source_exact','all26_failures_preserved_not_normal',
    'original_training_GPU_exit1_and_independent_CPU_recovery_ancestry_preserved')


def n20_validate_post_header(spec):
    require(type(spec) is dict and spec['schema'] == N20_FINAL_CPU_SCHEMA
        and all(type(spec[k]) is int for k in ('normal_RAW_count','expected_normal_RAW_count','total_failed_RAW_count',
            'total_terminal_identity_pin_count','deadline_seconds','allocator_limit_MiB','whole_GPU_limit_MiB',
            'maximum_process_RSS_bytes','initialization_seed'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 178
        and spec['total_failed_RAW_count'] == 26 and spec['total_terminal_identity_pin_count'] == 204
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and spec['maximum_process_RSS_bytes'] == 16*1024**3 and spec['automatic_retry'] is False
        and spec['failed26_partials_reused'] is False,
        'Independent post CPU178/26/204 resources and unchanged first201 recipe')
    record = spec['final_record']
    require(type(record) is dict and set(record) == N20_POST_RECORD_KEYS
        and type(record['owner_index']) is int and record['owner_index'] == 177,
        'Exact original completed final record, explicit owner177, Q output key')
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        ref_shape(record[key])
    ref_shape(spec['completed_final_CPU_source']); ref_shape(spec['entry_source'])
    require(spec['completed_final_CPU_source'] == record['source']
        and spec['entry_source']['path'] != record['source']['path']
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True)
        and Path(spec['CPU_admission_file']).is_absolute()
        and Path(spec['CPU_admission_file']).resolve() != Path(record['admission']['path']).resolve()
        and len(spec['RAW_inputs']) == 178 and type(spec['phase']) is str and bool(spec['phase'])
        and all(row['phase'] != spec['phase'] for row in spec['RAW_inputs']),
        'Fresh own post source/admission/phase; live post absent from the completed normal prefix')
    return True


def n20_post_parent_metadata(final, parent, gate):
    """Minimal pure metadata portion of frozen final front, no final own gate."""
    records = final.n20_validate_final_header(parent); c = gate['closure']
    for key in ('numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source','CPU_execution_partition_supplement'):
        closed(c, parent[key])
    for item in parent['completed_GPU_shards']+records:
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
            closed(c, item[key])
    numeric = load(parent['numeric_CPU_shard_gate_source'], '_N20_post_completed_CPU_metadata')
    require(numeric.N20_DEV_CPU_SCHEMA == final.N20_FINAL_NUMERIC_SCHEMA
        and tuple(numeric.N20_CPU_RANGES) == final.N20_FINAL_CPU_RANGES
        and tuple(numeric.N20_CPU_PARENT_GPU_IDS) == final.N20_FINAL_PARENT_IDS,
        'Original completed standalone numerical CPU metadata ABI')
    original = read(closed(c, parent['completed_GPU_shards'][0]['spec']))
    numeric.n20_recovered_training_metadata_gate(original, gate)
    partition = read(closed(c, parent['CPU_execution_partition_supplement']))
    require(type(partition) is dict and partition['schema'] == final.N20_FINAL_PARTITION_SCHEMA
        and final.n20_final_exact(partition,parent['expected_CPU_execution_partition_supplement']),
        'Original complete prelocked execution supplement')
    gate['CPU_execution_partition_supplement'] = partition
    for key in numeric.N20_CPU_SOURCE_KEYS:
        closed(c, parent[key])
    GPUs = [numeric.n20_completed_DEV_GPU_record(parent,gate,item) for item in parent['completed_GPU_shards']]
    require(parent['GPU_outer_source'] in [r['source'] for r in parent['completed_GPU_shards']], 'Actual completed GPU helper source')
    gpu = load(parent['GPU_outer_source'], '_N20_post_original_inline_COCO_semantics')
    require(tuple(gpu.n20_policy_names()) == final.n20_final_policy_names()
        and callable(getattr(gpu,'validate_legacy_COCO',None)), 'Ten original policies/inline COCO metadata validator')
    evidence = []; rows = []; proofs = []; members = 0
    for record in records:
        item, count = final.n20_final_completed_CPU(parent,gate,record,numeric,GPUs)
        evidence.append(item); rows.extend(item['child']['scope_rows']); proofs.extend(item['report']['image_proofs']); members += count
    require([row['dataset_index'] for row in rows] == list(range(187)) and len({row['image_id'] for row in rows}) == 187
        and len(proofs) == len({ref['path'] for ref in proofs}) == 187
        and [item['parent']['GPU_shard']['shard_owner_index'] for item in evidence] == [141]*8+[142]*8+[143]*8+[144]*8,
        'Four original GPU/32 CPU actual0 scopes, complete187 images, no failed partial reuse')
    gate.update(gpu_helpers=gpu,numeric_helpers=numeric,completed_GPU_evidence=GPUs,numeric_CPU_evidence=evidence,
        full_rows=rows,image_proofs=proofs,registered_members=members)
    return gate


def n20_post_validate_final(record, spec, gate, final):
    c = gate['closure']; context = gate['context']; owner = record['owner_index']; row = spec['RAW_inputs'][owner]
    require(context['normal_RAW_count'] == 178 and context['RAW_inputs'] == spec['RAW_inputs']
        and all(row[k] == record[v] for k,v in (('receipt','RAW'),('admission','admission'),
            ('phase_binding','phase_binding'),('bootstrap','bootstrap'))), 'Real completed own prefix includes original final177')
    raw = terminal_raw_only(record['RAW'],row['phase'],0)
    check_pin(spec['terminal_identity_pins'][record['RAW']['path']],raw,row['phase'],0)
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        closed(c,record[key])
    joined = gate['lineage'].owner_ref_join(context,gate['owners'],record['RAW'],record['admission'],record['source'],record['spec'],'')
    parent = read(closed(c,record['spec'])); Q = read(closed(c,record['Q']))
    final.n20_validate_final_header(parent)
    require(type(joined) is int and joined == owner == 177 and parent['entry_source'] == record['source']
        and parent['phase'] == row['phase'] and parent['RAW_inputs'] == spec['RAW_inputs'][:177]
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k,v in parent['terminal_identity_pins'].items())
        and Path(parent['CPU_admission_file']).resolve(strict=True) == Path(record['admission']['path']).resolve(strict=True)
        and Path(record['Q']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'Q.json').resolve(strict=True)
        and all(final.n20_final_exact(parent[k],spec[k]) for k in final.N20_FINAL_SHARED_KEYS+(
            'numeric_CPU_shards','numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source'))
        and all(final.n20_final_exact(v,spec[k]) for k,v in parent.items() if k.startswith('additional_preserved_')),
        'Original final CPUempty0 actual/source/spec/ad/phase/prefix/full Sources/pins and immutable bindings')
    n20_post_parent_metadata(final,parent,gate)
    binding = dict(source=record['source'],spec=record['spec'],CPU_admission=record['admission'],phase=parent['phase'],
        numeric_CPU_shards=parent['numeric_CPU_shards'],completed_GPU_shards=parent['completed_GPU_shards'],
        numeric_CPU_shard_gate_source=parent['numeric_CPU_shard_gate_source'],CPU_execution_partition_supplement=parent['CPU_execution_partition_supplement'],
        N20_root_training_qualified_result=parent['N20_root_training_qualified_result'],N20_locked_protocol=parent['N20_locked_protocol'],
        legacy_qualified_result=parent['legacy_qualified_result'],interpreter_source=parent['interpreter_source'],checkpoints=gate['trained_checkpoints'],
        N20_failed_training_record=parent['N20_failed_training_record'],N20_failed_training_stderr=parent['N20_failed_training_stderr'],
        N20_failed_DEV_capture_record=parent['N20_failed_DEV_capture_record'],N20_failed_DEV_capture_stderr=parent['N20_failed_DEV_capture_stderr'],
        N20_failed_CPU_numeric_record=parent['N20_failed_CPU_numeric_record'],N20_failed26_helper_source=parent['N20_failed26_helper_source'])
    require(Q['schema'] == 'N20-fixedDEV-full187-CPU-COCO-qualification-v1'
        and Q['status'] == 'N20_ALL18700_FIXED_DEV_CPU_NUMERIC_AND_FULL_COCO_COMPLETE_PENDING_ROOT_ACTUAL0'
        and all(Q[k] == v for k,v in binding.items()) and Q['images'] == 187 and Q['GT'] == 380 and Q['natural_queries'] == 18700
        and Q['initialization_seed'] == 202610150201 and Q['normal_RAW_count'] == 177
        and Q['total_failed_RAW_count'] == 26 and Q['total_terminal_identity_pin_count'] == 203
        and tuple(Q['policies']) == final.n20_final_policy_names()
        and set(Q['checks']) == set(N20_POST_FINAL_CHECKS) and all(Q['checks'][k] is True for k in N20_POST_FINAL_CHECKS)
        and all(Q[k] is False for k in ('qualified','ROOT_actual_CPU0_verified','current_owned_CPU_terminal_self_certified',
            'numerical_forward_rerun','head_loaded','CUDA_initialized','local_AP_averaged','CPU_scout_forward_used_for_ranking',
            'AP_results_interpreted_as_qualified','original_training_GPU_actual0_verified','task_efficacy_established',
            'innovation_established','TGRS_readiness_established','failed26_partials_reused')) and Q['training_artifacts_recovery_qualified'] is True,
        'Immutable original pending final Q/full187/ten policies/checks/limits; actual0 comes from original RAW join')
    details = read(closed(c,Q['details']))
    require(Path(Q['details']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'DETAILS.json').resolve(strict=True)
        and details['schema'] == 'N20-fixedDEV-full187-CPU-COCO-details-v1'
        and all(details[k] == v for k,v in binding.items()) and details['control_results'] == Q['control_results']
        and details['image_proofs'] == gate['image_proofs'] and details['registered_bundle_members_consumed'] == gate['registered_members']
        and all(details[k] is False for k in ('numerical_forward_rerun','head_loaded','local_AP_averaged','original_training_GPU_actual0_verified','failed26_partials_reused'))
        and details['training_artifacts_recovery_qualified'] is True, 'Original DETAILS and187 complete proof refs/flags')
    evidence = gate['numeric_CPU_evidence']
    manifests = list({item['child']['parent_GPU_report']['capture_manifest']['path']:
        item['child']['parent_GPU_report']['capture_manifest'] for item in evidence}.values())
    captures = [ref for item in evidence for ref in item['child']['scope_capture_refs']]
    require(details['capture_manifests'] == manifests and len(manifests) == 4
        and details['capture_refs'] == captures and len(captures) == len({r['path'] for r in captures}) == 187,
        'Original four full manifests/187 actual physical captures, correct origin bindings')
    for ref in details['image_proofs']+manifests+captures:
        closed(c,ref)
    require(set(Q['control_results']) == set(final.n20_final_policy_names()), 'Exactly ten complete final policies')
    for name in final.n20_final_policy_names():
        result = Q['control_results'][name]
        require(type(result) is dict and set(result) == {'predictions','COCO','stats'}, 'Inline COCO metadata, strictref predictions only')
        gate['gpu_helpers'].validate_legacy_COCO(result)
        values = read(closed(c,result['predictions']))
        expected = [value for item in evidence for value in item['child']['scope_expected_predictions'][name]]
        require(len(values) == 18700 and values == expected
            and Path(result['predictions']['path']).resolve(strict=True) ==
                (Path(parent['output_directory'])/(name+'_PREDICTIONS.json')).resolve(strict=True),
            'Original full18700 predictions/RLE/class/S0/export order exact all32 CPU/all4 GPU')
        if name in ('native_official','strong_fixed_E24_Omega'):
            baseline = gate['legacy_root']['control_results'][name]
            require(values == read(closed(c,baseline['predictions'])) and result['stats'] == baseline['stats']
                and result['COCO'] == baseline['COCO'], 'Native/R full predictions/all12 stats/inline COCO exact original reference')
        del values,expected
    interpreter = load(parent['interpreter_source'],'_N20_post_frozen_first201_interpreter')
    final.n20_validate_final_interpreter(interpreter,gate['N20_protocol'],parent['interpreter_source'])
    interpreted = final.n20_final_main_preview(interpreter,Q['control_results'])
    require(interpreted == Q['primary_interpretation_preview_pending_ROOT_actual0'], 'Only prelocked E24 atmost percent interpretation, unchanged from pending preview')
    return dict(schema='ROOT-N20-fixedDEV-full187-GPU-CPU-COCO-qualified-result-v1',
        status='ROOT_N20_ALL18700_FIXED_DEV_GPU_CPU_FULL_COCO_QUALIFIED',qualified=True,
        original_final_CPU_actual0_verified=True,final_owner=record,final_Q=record['Q'],final_details=Q['details'],
        **{k:binding[k] for k in binding if k not in ('source','spec','CPU_admission','phase')},
        control_results=Q['control_results'],images=187,GT=380,natural_queries=18700,initialization_seed=202610150201,
        primary_interpretation=interpreted,normal_after=178,preserved_failures=26,pins_after=204,
        original_training_GPU_actual0_verified=False,training_artifacts_recovery_qualified=True,failed26_partials_reused=False,
        formal_TEST_evaluated=False,cross_dataset_evaluated=False,numerical_forward_rerun=False,COCO_rerun=False,
        task_efficacy_established=False,innovation_established=False,TGRS_readiness_established=False,evidence_scope=Q['evidence_scope'])


def source_gate(path,spec):
    n20_validate_post_header(spec)
    failed22 = n20_failed22_raw_only(spec); failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec); failed25 = n20_failed25_raw_only(spec)
    failed26 = n20_failed26_raw_only(spec)
    gate = inherited_source_gate(path,spec,cpu=True)
    n20_failed22_closed(spec,gate,failed22); n20_failed23_closed(spec,gate,failed23)
    n20_failed24_closed(spec,gate,failed24); n20_failed25_closed(spec,gate,failed25)
    n20_failed26_closed(spec,gate,failed26)
    c = gate['closure']
    closed(c,spec['N20_failed26_helper_source'])
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        closed(c,spec['final_record'][key])
    closed(c,spec['completed_final_CPU_source'])
    final = load(spec['completed_final_CPU_source'],'_N20_completed_final_pure_metadata_API')
    require(final.N20_FINAL_CPU_SCHEMA == N20_FINAL_CPU_SCHEMA, 'Frozen final metadata API schema')
    qualified = n20_post_validate_final(spec['final_record'],spec,gate,final)
    gate.update(completed_final_module=final,final_qualified_result=qualified)
    return gate


def run(spec_path):
    started = time.monotonic(); require(Path(spec_path).is_absolute(),'Independent absolute post specification')
    path = Path(spec_path).resolve(strict=True); spec = read(path); n20_validate_post_header(spec)
    require(path != Path(spec['final_record']['spec']['path']).resolve(strict=True), 'New post spec cannot be the frozen final spec')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(), 'Fresh post output')
    gate = source_gate(path,spec)
    import psutil
    process = psutil.Process(); peak = 0; minimum = None
    def check():
        nonlocal peak,minimum
        rss = process.memory_info().rss; available = psutil.virtual_memory().available
        peak = max(peak,rss); minimum = available if minimum is None else min(minimum,available)
        require(time.monotonic()-started < 1800 and process.cpu_affinity() == [0,1]
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '2' for k in N20_POST_THREAD_ENV)
            and rss < 16*1024**3 and available >= 16*1024**3, 'Post CPU01/emptyCUDA/two-thread/RAM/RSS/1800s bounds')
        if 'torch' in sys.modules:
            require(not sys.modules['torch'].cuda.is_initialized(),'Post metadata must never initialize CUDA')
    check(); require(psutil.disk_usage(str(output.parent)).free >= 1024**3,'Post output free1GiB')
    check()
    require(gate['source'] == spec['entry_source'] and gate['spec_ref'] == reference(path)
        and gate['admission_ref']['path'] == str(Path(spec['CPU_admission_file']).resolve(strict=True))
        and gate['context']['normal_RAW_count'] == 178 and gate['context']['RAW_inputs'] == spec['RAW_inputs'],
        'Real current own post context/source/spec/admission; original final context was not changed')
    for ref in spec['sources']:
        check(); bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source']
        and reference(spec['CPU_admission_file']) == gate['admission_ref'], 'Final post Source/spec/admission immutable')
    gate['lineage'].final_endpoint_check(gate['parent']); check(); gc.collect(); output.mkdir()
    value = dict(schema=N20_POST_SCHEMA,status=N20_POST_STATUS,source=gate['source'],spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'],phase=spec['phase'],completed_final_CPU_source=spec['completed_final_CPU_source'],
        final_record=spec['final_record'],final_qualified_result=gate['final_qualified_result'],normal_RAW_count=178,
        total_failed_RAW_count=26,total_terminal_identity_pin_count=204,post_only=True,original_final_actual0_checked=True,
        qualified=False,ROOT_post_actual_CPU0_verified=False,current_post_owned_terminal_self_certified=False,
        COCO_rerun=False,numerical_forward_rerun=False,head_loaded=False,old_context_modified=False,failed26_partials_reused=False,
        live_post_added_to_normal=False,peak_observed_process_RSS_bytes=peak,minimum_observed_available_RAM_bytes=minimum,
        elapsed_seconds=time.monotonic()-started,evidence_limits='Inner result is completed original final177; ROOT must separately verify original post178 actual0 and Source before publishing outer qualification.')
    write(output/'ROOT_RESULT.json',value); check()
    print(json.dumps(dict(status=N20_POST_STATUS,result=reference(output/'ROOT_RESULT.json'),qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'Single actual ROOT owner required');run(args.spec)
