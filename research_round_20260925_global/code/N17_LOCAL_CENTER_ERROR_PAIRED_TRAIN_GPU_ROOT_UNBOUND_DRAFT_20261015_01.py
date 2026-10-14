"""UNBOUND_DRAFT: standalone N17 GPU120 paired LOCAL/CENTER ERROR training entry.
Copied immutable ordinary RAW9..17/carrier front; ROOT owns actual binding/execution.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 17 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 17 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 stays separate; all terminal pins must equal normal+17')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9/10/11/controlled12/packet-ABI13/controlled-storage14/controlled-bundle15/controlled-CPU-proof16/reference-ABI17 already passed RAW-only checks.
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
    # FIRST HASH: failed9/10/11/controlled12/packet-ABI13/controlled-storage14/controlled-bundle15/controlled-CPU-proof16/reference-ABI17 and every normal/legacy failed8 passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(closure.get(row[key]['path']) == row[key] for row in rows for key in RAW_KEYS if key != 'phase') and
            closure.get(failed_ref['path']) == failed_ref and all(closure.get(row['receipt']['path']) == row['receipt'] for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N17-local-native-L-paired-error-training-GPU-spec-v1"
CPU_SCHEMA = "ROOT-N16-fixed-E24-all100-scout-CPU-numeric-shards-aggregate-spec-v1"
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

def source_gate(path,spec):
    require(spec['schema']==N17_SCHEMA and spec['normal_RAW_count']==spec['expected_normal_RAW_count']==120 and
        spec['total_failed_RAW_count']==17 and spec['total_terminal_identity_pin_count']==137 and
        spec['initialization_seed']==SEED and Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True),
        'Standalone current GPU120 own entry/120+17pins137/first N17 seed required')
    gate=inherited_source_gate(path,spec,cpu=False)
    for key in NEW_REFS:closed(gate['closure'],spec[key])
    require(spec['scout_module_source']==spec['neighbor_kernel_source'],'Train exact CPU119-qualified neighbor/lift module')
    lock=validate_experiment_lock(spec,gate['closure']);validate_neighbor_cache(spec,gate,lock)
    metadata=read(closed(gate['closure'],lock['architecture']['initialization_metadata']))
    require(metadata['schema']=='N17-private-initialization-metadata-v1' and
        metadata['old_factory_SHA256']==spec['base_scout_source']['sha256'] and metadata['lift_module_SHA256']==spec['scout_module_source']['sha256'] and
        metadata['synthetic_parameters_only'] is True and metadata['training_executed'] is False and
        metadata['actual_SAR_data_or_checkpoint_loaded'] is False and metadata['CPU_RNG_restored'] is True and metadata['CUDA_initialized'] is False,
        'Only prelocked synthetic/private lifted initialization metadata allowed')
    require([row['seed'] for row in metadata['rows']]==[202610150171,202610150172,202610150173], 'All prelocked initialization seeds retained')
    gate.update(experiment_lock=lock,initialization_metadata=metadata)
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

def verify_initialization(torch,state,gate):
    row=next(row for row in gate['initialization_metadata']['rows'] if row['seed']==SEED)
    expected=gate['experiment_lock']['architecture']['initialization_digest_by_seed'][str(SEED)]
    require(row['combined_digest']==expected=='5b9f58a3dece9720f84b257d03ae6928a7129e6f364e218f6bd49475d5fb7aa7' and
        row['paired_state_exact'] is True and row['paired_storage_independent'] is True and row['new_columns_zero'] is True,
        'Frozen N17 first-seed paired lifted initialization proof differs')
    actual={}
    for arm in ARMS:
        digest,tensors=state_digest(torch,state['heads'][arm].state_dict())
        require(digest==expected and tensors==row['tensors'],'Actual new scout state must equal prelocked whole/tensor initial digests')
        actual[arm]=dict(combined_digest=digest,tensors=tensors)
    return actual

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

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);require(Path(spec_path).is_absolute(),'Absolute GPU120 spec required')
    spec=read(path);gate=source_gate(path,spec);closure=gate['closure']
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh owned N17 GPU120 output required')
    import psutil
    process=psutil.Process();require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and
        all(os.environ.get(key)=='2' for key in THREAD_ENV),'GPU0/CPU01/two-thread actual own environment')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and
        torch.cuda.is_available() and torch.cuda.device_count()==1,'Closed FP32 GPU runtime required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    torch.cuda.set_device(0);torch.cuda.init()
    scout=load(spec['scout_module_source'],'_N17_closed_four_input_scout');base=load(spec['base_scout_source'],'_N17_closed_original_scout_factory')
    body=load(spec['paired_training_body_source'],'_N17_closed_paired_error_body');old_body=load(spec['original_paired_training_body_source'],'_N17_closed_original_paired_body')
    outside_CPU=torch.get_rng_state().clone();outside_CUDA=torch.cuda.get_rng_state(0).cpu().clone()
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and process.cpu_affinity()==[0,1] and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(key)=='2' for key in THREAD_ENV) and
            torch.get_num_threads()==torch.get_num_interop_threads()==2 and torch.cuda.current_device()==0 and
            torch.cuda.memory_reserved(0)<=4096*1024**2,'GPU120 bounded resources/deadline/env; no retry')
    state=api(body,'make_paired_state')(torch,scout,api(base,'make_scout'),old_body,device='cuda:0',seed=SEED)
    initial=verify_initialization(torch,state,gate);require(torch.equal(outside_CPU,torch.get_rng_state()),'Private scout constructor must restore CPU RNG')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)
    binding=dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        experiment_lock=spec['experiment_lock'],locked_protocol=spec['locked_protocol'],neighbor_cache_ROOT_result=spec['neighbor_cache_ROOT_result'],
        neighbor_cache_CPU_Q=spec['neighbor_cache_CPU_Q'],neighbor_cache_manifest=spec['neighbor_cache_manifest'],neighbor_cache_CPU_RAW=spec['neighbor_cache_CPU_RAW'],
        neighbor_cache_CPU_source=spec['neighbor_cache_CPU_source'],neighbor_cache_CPU_spec=spec['neighbor_cache_CPU_spec'],neighbor_cache_CPU_admission=spec['neighbor_cache_CPU_admission'],
        original_N16_cache_report=spec['original_N16_cache_report'],scout_module_source=spec['scout_module_source'],base_scout_source=spec['base_scout_source'],
        paired_training_body_source=spec['paired_training_body_source'],original_paired_training_body_source=spec['original_paired_training_body_source'],
        initialization_metadata=gate['experiment_lock']['architecture']['initialization_metadata'],initialization_seed=SEED,order_seed=ORDER_SEED,
        frozen_teacher_checkpoint=gate['original_cache_report']['teacher_checkpoint'],frozen_PR_checkpoint=gate['original_cache_report']['fixed_PR_checkpoint'])
    output.mkdir();write(output/'START.json',dict(schema='N17-local-native-L-paired-error-training-start-v1',**binding,
        initial_states=initial,parameters_per_arm=16803,tensors_per_arm=4,arms=list(ARMS),teacher_or_PR_loaded=False,full_native_L_loaded=False,
        fixed_epochs=24,updates_each_arm=17784,point_exposure_each_arm=72843264,AMP=False,DEV_AP_evaluated=False,qualified=False))
    trace_path=output/'UPDATE_TRACE.jsonl';checkpoints={};endpoint_files={};loads=0
    with trace_path.open('x',encoding='utf8',newline='\n') as trace:
        def observer(event):
            check()
            if event['event']=='paired_image_update':
                trace.write(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n')
                if event['dataset_index']%32==0:trace.flush()
        def loader(index):
            nonlocal loads
            loaded=load_training_image(torch,np,index,gate['neighbor_rows'][index],gate['original_cache_rows'][index],
                gate['original_cache_report'],closure,spec['locked_protocol'],check);loads+=1;return loaded
        def checkpoint(epoch,endpoint,snapshot):
            check();require(epoch in (8,24),'Only prelocked E8/E24 snapshots')
            snapshot=dict(snapshot,metadata=dict(binding,endpoint_epoch=epoch),external_CPU_RNG=outside_CPU.clone(),external_CUDA_RNG=outside_CUDA.clone())
            filename=output/('E%02d_COMPLETE_PAIRED_SNAPSHOT.pth'%epoch)
            with filename.open('xb') as stream:torch.save(snapshot,stream)
            checkpoints[str(epoch)]=reference(filename);write(output/('E%02d_FIT_ENDPOINT.json'%epoch),dict(binding,endpoint=endpoint))
            endpoint_files[str(epoch)]=reference(output/('E%02d_FIT_ENDPOINT.json'%epoch));check()
        result=api(body,'train_until')(torch,state,loader,last_epoch=24,image_count=741,observer=observer,checkpoint_callback=checkpoint)
        trace.flush()
    require(result['completed_epoch']==24 and result['updates']=={arm:17784 for arm in ARMS} and state['sampled_points']==72843264 and
        state['visited_images']==17784 and state['empty_images']==0 and len(state['history'])==len(state['epoch_orders'])==24 and
        set(checkpoints)==set(endpoint_files)=={'8','24'} and loads==741*26,'Complete same-sample paired fixed24 trajectory/endpoints required')
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['GPU_admission_file'])==gate['admission_ref'] and
        torch.equal(outside_CPU,torch.get_rng_state()) and torch.equal(outside_CUDA,torch.cuda.get_rng_state(0).cpu()) and
        gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)==libraries,
        'Final Source/spec/admission/imported libraries/external RNG unchanged')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    write(output/'REPORT.json',dict(schema=REPORT_SCHEMA,status=STATUS,**binding,START=reference(output/'START.json'),
        checkpoints=checkpoints,FIT_endpoints=endpoint_files,UPDATE_TRACE=reference(trace_path),history=state['history'],epoch_orders=state['epoch_orders'],
        images=741,natural_queries=74100,sampled_points_per_epoch=3035136,epochs=24,paired_updates_per_arm=17784,
        point_exposure_per_arm=72843264,parameters_per_arm=16803,tensors_per_arm=4,arms=list(ARMS),initial_states=initial,
        normal_RAW_count=120,total_failed_RAW_count=17,total_terminal_identity_pin_count=137,actual_imported_files=libraries,
        qualified=False,ROOT_actual_GPU0_verified=False,current_owned_terminal_self_certified=False,CPU_endpoint_qualification_pending=True,
        teacher_or_PR_loaded=False,full_native_L_loaded=False,DEV_AP_evaluated=False,best_endpoint_selected=False,
        old_cache_Source_or_protocol_modified=False,additional_FIT_annotations=False,new_primitive_claimed=False,
        old_point_query_label_weight_order_preserved=True,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'),qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND_DRAFT: ROOT must bind source/spec/admission/CPU119 qualification before actual execution')
    run(args.spec)
