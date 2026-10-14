"""UNBOUND_DRAFT: full CPU118 certificate from four freshly owned numeric CPU0 shards.
No scientific execution or actual payload access by the author. Own actualCPU0 remains ROOT-only.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 16 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 16 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 stays separate; all terminal pins must equal normal+16')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9/10/11/controlled12/packet-ABI13/controlled-storage14/controlled-bundle15/controlled-CPU-proof16 already passed RAW-only checks.
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
    # FIRST HASH: failed9/10/11/controlled12/packet-ABI13/controlled-storage14/controlled-bundle15/controlled-CPU-proof16 and every normal/legacy failed8 passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(closure.get(row[key]['path']) == row[key] for row in rows for key in RAW_KEYS if key != 'phase') and
            closure.get(failed_ref['path']) == failed_ref and all(closure.get(row['receipt']['path']) == row['receipt'] for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N16-fixed-E24-all100-scout-fullDEV-GPU-shard-spec-v1"
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



RANGES=((0,47),(47,94),(94,141),(141,187))
CPU_DESCRIPTOR_KEYS={'cpu_shard_id','cpu_dataset_ordinal_range','source','spec','report','RAW','admission','cpu_shard_owner_index'}

def validated_CPU_descriptors(records):
    require(type(records) is list and len(records)==4,'Exactly four complete fresh numeric CPU0 shards required')
    for i,record in enumerate(records):
        require(type(record) is dict and set(record)==CPU_DESCRIPTOR_KEYS and type(record['cpu_shard_id']) is int and
            record['cpu_shard_id']==i and type(record['cpu_dataset_ordinal_range']) is list and
            len(record['cpu_dataset_ordinal_range'])==2 and all(type(n) is int for n in record['cpu_dataset_ordinal_range']) and
            record['cpu_dataset_ordinal_range']==list(RANGES[i]) and type(record['cpu_shard_owner_index']) is int and
            record['cpu_shard_owner_index']==114+i,'Fixed CPU shard ordinal scope and explicit owner114..117 required')
        for key in ('source','spec','report','RAW','admission'):ref_shape(record[key])
    for key in ('spec','report','RAW','admission'):
        require(len({r[key]['path'] for r in records})==4,'Distinct new numeric CPU owner artifacts required: '+key)
    return records

def merge_policy_predictions(pieces,image_ids,native,AP_reference):
    require(type(pieces) is list and len(pieces)==4 and type(image_ids) is list and len(image_ids)==187 and
            len(set(image_ids))==187 and type(native) is list and len(native)==18700 and
            type(AP_reference) is list and len(AP_reference)==18700,'Exact complete four-piece original18700 population required')
    combined=[]
    for values,(first,last) in zip(pieces,RANGES):
        expected=native[first*100:last*100]
        require(type(values) is list and len(values)==(last-first)*100 and
                [v['image_id'] for v in values]==[image for image in image_ids[first:last] for _ in range(100)] and
                all(type(v) is dict and 'segmentation' in v and
                    {k:x for k,x in v.items() if k!='segmentation'}=={k:x for k,x in n.items() if k!='segmentation'}
                    for v,n in zip(values,expected)),'Numeric shard cannot alter native query/class/S0/ordinal order')
        combined.extend(values)
    require(len(combined)==18700 and combined==AP_reference,'All merged masks/predictions must exactly reproduce original AP aggregate113')
    return combined

SCHEMA=CPU_SCHEMA
REPORT_SCHEMA='N16-fixed-E24-all100-scout-CPU-numeric-shards-qualification-v1'
STATUS='N16_ALL18700_SCOUT_CPU_NUMERIC_SHARDS_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0'

IMAGE_PROOF_KEYS={'dataset_index','image_id','queries','bundle','registered_members','consumed_members','details'}
LOCAL_PROOF_FLAGS=('all_actual_GPU_logits_selected_indices_exact','all_PR_two_formulas_absolute1e4_strictsign',
    'all_native_strong_and_new_mask_RLE_exact')

def validate_image_proof(record,row):
    members=len(row['member_locators'])
    require(type(record) is dict and set(record)==IMAGE_PROOF_KEYS and type(record['dataset_index']) is int and
            record['dataset_index']==row['dataset_index'] and record['image_id']==row['image_id'] and
            type(record['queries']) is int and record['queries']==100 and record['bundle']==row['bundle'] and
            type(record['registered_members']) is int and type(record['consumed_members']) is int and
            record['registered_members']==record['consumed_members']==members and members>0,
            'Owned numeric proof must cover exact original image/bundle/all registered members')
    details=record['details']
    require(type(details) is list and len(details)==100 and [d['query_id'] for d in details]==list(range(100)),
            'Full natural100 numeric query proofs required, never failure16 partials')
    for q,item in enumerate(details):
        require(type(item) is dict and set(item)=={'query_id','strong_original_Omega','fixed_profiles'} and
                type(item['strong_original_Omega']) is dict and type(item['fixed_profiles']) is list and
                len(item['fixed_profiles'])==4 and [(v['epoch'],v['mode']) for v in item['fixed_profiles']]==
                [(24,'atmost'),(24,'exact'),(8,'atmost'),(8,'exact')],
                'Original strong Omega and all four fixed numeric profiles required')
        for profile in item['fixed_profiles']:
            require(type(profile) is dict and set(profile)=={'epoch','mode','proof'} and type(profile['proof']) is dict,
                    'Saved original numeric proof object must be retained without invented replacement')
    return members

def source_gate(path,spec):
    require(spec['schema']==SCHEMA and type(spec['normal_RAW_count']) is int and spec['normal_RAW_count']==118 and
            spec['expected_normal_RAW_count']==118 and spec['total_failed_RAW_count']==16 and
            spec['total_terminal_identity_pin_count']==134 and type(spec['aggregate_CPU_owner_index']) is int and
            spec['aggregate_CPU_owner_index']==113 and Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True),
            'Own final CPU118 schema/entry after original AP113 and four numeric114..117 required')
    gate=inherited_source_gate(path,spec,cpu=True);closure=gate['closure']
    numeric=load(closed(closure,spec['numeric_CPU_shard_gate_source']),'_N16_closed_numeric_CPU_shard_contract')
    api(numeric,'check_common_completed_aggregate')(spec,gate)
    api(numeric,'validate_CPU_proof_execution_supplement')(spec,gate)
    records=validated_CPU_descriptors(spec['numeric_CPU_shards']);evidence=[];all_proofs=[];all_bindings=[];total_members=0
    manifest=gate['manifest'];rows=manifest['rows'];names=gate['shard'].policy_names()
    require(len(rows)==187 and [r['dataset_index'] for r in rows]==list(range(187)), 'Complete AP113 original GPU image roster required')
    common=tuple(dict.fromkeys(gate['shard'].ROOT_FIELDS+numeric.CARRIER_KEYS+('shards','shard_gate_source','locked_protocol',
        'aggregate_CPU_source','aggregate_CPU_spec','aggregate_CPU_report','aggregate_CPU_RAW','aggregate_CPU_admission','aggregate_CPU_owner_index',
        'CPU_adapter_source','selector_cpu_proof_source','PR_formula_source','CPU_proof_execution_supplement',
        'prior_failed_CPU_receipt','additional_failed_RAW_inputs')+numeric.FAILURE16_PRIOR_KEYS+
        ('additional_preserved_N16_fullDEV_controlled_CPU_proof_deadline_failure',)))
    for descriptor in records:
        for key in ('source','spec','report','RAW','admission'):closed(closure,descriptor[key])
        require(descriptor['source']['sha256']==spec['numeric_CPU_shard_gate_source']['sha256'] and
                descriptor['source']['bytes']==spec['numeric_CPU_shard_gate_source']['bytes'],'All four numeric CPU leaves execute the same frozen qualified code')
        parent=read(closed(closure,descriptor['spec']));report=read(closed(closure,descriptor['report']))
        i=descriptor['cpu_shard_id'];first,last=RANGES[i];owner=114+i;local_rows=rows[first:last];count=last-first
        joined=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],descriptor['RAW'],descriptor['admission'],
                    descriptor['source'],descriptor['spec'],'')
        require(type(joined) is int and joined==descriptor['cpu_shard_owner_index']==parent['normal_RAW_count']==owner and
                parent['RAW_inputs']==spec['RAW_inputs'][:owner] and spec['RAW_inputs'][owner]['receipt']==descriptor['RAW'] and
                parent['schema']==numeric.CPU_SCHEMA and parent['entry_source']==descriptor['source'] and
                parent['cpu_shard_id']==i and parent['cpu_dataset_ordinal_range']==list(RANGES[i]) and parent['cpu_shard_owner_index']==owner and
                Path(parent['CPU_admission_file']).resolve(strict=True)==Path(descriptor['admission']['path']).resolve(strict=True),
                'Each fresh numeric CPUempty0 owner114..117/source/spec/ad/range/prefix must close exactly')
        require(all(closure.get(r['path'])==r for r in parent['sources']) and
                all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
                'Full numeric parent Source/pin closure required')
        for key in common:
            if key!='capture_budget':require(parent[key]==spec[key],'Numeric leaf cannot replace shared frozen provenance/protocol/failure field: '+key)
        api(numeric,'validate_CPU_proof_execution_supplement')(parent,gate)
        require(report['schema']==numeric.REPORT_SCHEMA and report['status']==numeric.STATUS and
                report['source']==descriptor['source'] and report['spec']==descriptor['spec'] and report['CPU_admission']==descriptor['admission'] and
                report['phase']==parent['phase'] and report['cpu_shard_id']==i and report['cpu_dataset_ordinal_range']==list(RANGES[i]) and
                report['cpu_shard_owner_index']==owner and report['normal_RAW_count']==owner and report['total_failed_RAW_count']==16 and
                report['total_terminal_identity_pin_count']==owner+16 and report['scope_dataset_indices']==list(range(first,last)) and
                report['scope_image_ids']==[r['image_id'] for r in local_rows] and report['images']==count and report['queries']==count*100 and
                report['global_images']==187 and report['global_queries']==18700 and report['global_GT']==380 and report['queries_per_image']==100 and
                tuple(report['controls'])==names and report['fixed_main_epoch']==24 and report['fixed_curve_epoch']==8 and
                report['DEV_winner_selected'] is False and report['AP_evaluated'] is False and report['official_COCO_called'] is False and
                report['failed_partials_reused'] is False and report['qualified'] is False and report['ROOT_actual_CPU0_verified'] is False and
                report['current_owned_CPU_terminal_self_certified'] is False and report['CUDA_initialized'] is False and
                report['CPU_scout_forward_used_for_ranking'] is False and report['CPU_F_rerendered'] is False and
                report['numeric_atol']==1e-4 and report['numeric_rtol']==0 and all(report[k] is True for k in LOCAL_PROOF_FLAGS),
                'Complete fixed-scope numeric report only; no localCOCO/AP, relaxed numeric/sign or self-terminal claims')
        binding_keys=('aggregate_CPU_source','aggregate_CPU_spec','aggregate_CPU_report','aggregate_CPU_RAW','aggregate_CPU_admission',
            'shards','locked_protocol','execution_supplement','CPU_adapter_source','selector_cpu_proof_source','PR_formula_source','CPU_proof_execution_supplement')
        require(all(report[k]==spec[k] for k in binding_keys) and report['capture_manifest']==gate['aggregate_report']['capture_manifest'] and
                report['shard_reports']==gate['aggregate_report']['shard_reports'] and report['shard_manifests']==gate['aggregate_report']['shard_manifests'] and
                report['capture_refs']==[r['bundle'] for r in local_rows] and
                report['original_GPU_bindings']==[r['original_GPU_binding'] for r in local_rows] and
                type(report['actual_capture_files_consumed']) is int and report['actual_capture_files_consumed']==count and
                type(report['registered_bundle_members_consumed']) is int and
                report['registered_bundle_members_consumed']==sum(len(r['member_locators']) for r in local_rows) and
                type(report['image_proofs']) is list and len(report['image_proofs'])==count and
                len({r['path'] for r in report['image_proofs']})==count,
                'Original per-GPU bindings/full-member physical bundle evidence must stay unchanged')
        for ref,row in zip(report['image_proofs'],local_rows):
            target=closed(closure,ref)
            require(target.resolve(strict=True)==(Path(parent['output_directory'])/('IMAGE_%03d_PROOF.json'%row['dataset_index'])).resolve(strict=True),
                    'Only new owned numeric-leaf proof files, never failure16 partials')
            proof_record=read(target);total_members+=validate_image_proof(proof_record,row);all_proofs.append(ref)
        for name in names:
            result=report['control_results'][name]
            require(type(result) is dict and set(result)=={'predictions'},'Only local complete raw predictions allowed, never local or averaged COCO')
            closed(closure,result['predictions'])
        all_bindings.extend(report['original_GPU_bindings']);evidence.append(dict(descriptor=descriptor,parent=parent,report=report))
    require(len(all_proofs)==187 and len({r['path'] for r in all_proofs})==187 and
            all_bindings==[r['original_GPU_binding'] for r in rows] and
            total_members==sum(len(r['member_locators']) for r in rows),'All187 independent numeric image/member proofs and original GPU bindings exact')
    gate.update(numeric_CPU_module=numeric,numeric_CPU_evidence=evidence,numeric_image_proofs=all_proofs,
                original_GPU_bindings=all_bindings,total_numeric_members=total_members)
    return gate

def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(),'Absolute ROOT final CPU118 spec required')
    spec=read(path);gate=source_gate(path,spec);closure=gate['closure'];context=gate['context'];gpu=gate['shard']
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh CPU118 qualification output')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            all(os.environ.get(k)=='2' for k in THREAD_ENV),'CPUempty/CPU01/two-thread own environment')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified cold CPU Torch; no model/head load')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0','Original numeric runtime')
    cv2.setNumThreads(2)
    n10=load(context['N10_evaluation_source'],'_N16_final_numeric_original_export')
    reader=load(context['reader_source'],'_N16_final_numeric_original_population')
    official=load(context['COCO_source'],'_N16_final_numeric_official_COCO')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)
    outside=torch.get_rng_state().clone()
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and
                not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and
                os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
                torch.get_num_threads()==torch.get_num_interop_threads()==2,'CPU118 bounded environment/no retry')
    coco=COCO(str(closed(closure,context['DEV'])));ids=list(coco.imgs)
    require(len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1} and
            ids==[v['id'] for v in coco.dataset['images']],'Complete original DEV187/380/category1')
    report=gate['aggregate_report'];native=read(closed(closure,report['native_reference']))
    strong=read(closed(closure,report['strong_fixed_reference']))
    reader.population(coco,native,ids);reader.population(coco,strong,ids)
    require([r['image_id'] for r in gate['manifest']['rows']]==ids,'Original AP113 canonical full image roster')
    results={};output.mkdir()
    for name in gpu.policy_names():
        check();original=read(closed(closure,report['control_results'][name]['predictions']))
        pieces=[read(closed(closure,v['report']['control_results'][name]['predictions'])) for v in gate['numeric_CPU_evidence']]
        values=merge_policy_predictions(pieces,ids,native,original)
        reader.population(coco,values,ids)
        if name=='native_official':require(values==native,'Full native original predictions exact')
        if name=='strong_fixed_E24_Omega':require(values==strong,'Full strong fixed original predictions exact')
        pred_path=output/(name+'_PREDICTIONS.json');write(pred_path,values)
        metrics=official.coco_metrics(coco,values,ids,COCOeval)
        require(metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100] and metrics['score_filter'] is None and
                metrics['original_GT_preserved'] is True and metrics['official_COCO_crowd_ignore_semantics'] is True,
                'Full original official COCO12 semantics, never averaged local AP')
        n10.stats_exact(metrics['stats'],report['control_results'][name]['stats'],'Full CPU118 COCO stats exactly reproduce AP aggregate113')
        results[name]=dict(predictions=reference(pred_path),COCO=metrics,stats=metrics['stats'])
        del values,pieces,original;gc.collect()
    n10.stats_exact(results['native_official']['stats'],gate['prior_q']['expected_E24_COCO_stats'],'Original native gold stats exact')
    carrier=gate['carrier_report']['control_results'][spec['strong_fixed_carrier_control_name']]
    require(carrier['predictions']==spec['strong_fixed_reference_predictions'],'Original strong carrier ref unchanged')
    n10.stats_exact(results['strong_fixed_E24_Omega']['stats'],carrier['stats'],'Original strong E24 gold stats exact')
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
            reference(spec['CPU_admission_file'])==gate['admission_ref'] and torch.equal(outside,torch.get_rng_state()) and
            gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)==libraries,
            'Final CPU118 own Source/spec/admission/RNG/libraries exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        aggregate_CPU_source=spec['aggregate_CPU_source'],aggregate_CPU_spec=spec['aggregate_CPU_spec'],aggregate_CPU_report=spec['aggregate_CPU_report'],
        aggregate_CPU_RAW=spec['aggregate_CPU_RAW'],aggregate_CPU_admission=spec['aggregate_CPU_admission'],
        numeric_CPU_shard_gate_source=spec['numeric_CPU_shard_gate_source'],numeric_CPU_shards=spec['numeric_CPU_shards'],
        numeric_CPU_reports=[v['descriptor']['report'] for v in gate['numeric_CPU_evidence']],
        capture_manifest=report['capture_manifest'],capture_refs=report['capture_refs'],shards=spec['shards'],
        shard_reports=report['shard_reports'],shard_manifests=report['shard_manifests'],
        locked_protocol=spec['locked_protocol'],execution_supplement=spec['execution_supplement'],
        CPU_proof_execution_supplement=spec['CPU_proof_execution_supplement'],ROOTtrain_result=spec['ROOTtrain_result'],
        training_GPU_report=spec['training_GPU_report'],training_CPU_qualification=spec['training_CPU_qualification'],
        scout_checkpoints=spec['scout_checkpoints'],CPU_adapter_source=spec['CPU_adapter_source'],
        selector_cpu_proof_source=spec['selector_cpu_proof_source'],PR_formula_source=spec['PR_formula_source'])
    details=dict(schema='N16-fixed-E24-all100-scout-CPU-numeric-shards-aggregate-details-v1',**binding,
        control_results=results,image_proofs=gate['numeric_image_proofs'],original_GPU_bindings=gate['original_GPU_bindings'],
        registered_bundle_members_consumed=gate['total_numeric_members'],actual_imported_files=libraries,
        numerical_forward_rerun=False,head_loaded=False,CUDA_initialized=False,local_AP_averaged=False,failed77_partial_reused=False)
    write(output/'DETAILS.json',details)
    write(output/'Q.json',dict(schema=REPORT_SCHEMA,status=STATUS,**binding,details=reference(output/'DETAILS.json'),
        images=187,GT=380,natural_queries=18700,policies=list(gpu.policy_names()),control_results=results,
        normal_RAW_count=118,total_failed_RAW_count=16,total_terminal_identity_pin_count=134,
        checks=dict(four_owned_CPU0_numeric_shards_exact=True,all187_registered_members_numeric_proven=True,
            all100_C_K_and_actual_GPUlogits_selection_exact=True,original_PR_two_CPU_formulas_absolute1e4_strictsign=True,
            full_native_strong_and_scout_masks_RLE_exact=True,all12_official_COCO12_exact=True,
            original_AP113_predictions_and_stats_exact=True,failed77_partial_never_reused=True),
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,
        numerical_forward_rerun=False,head_loaded=False,CUDA_initialized=False,local_AP_averaged=False,
        CPU_scout_forward_used_for_ranking=False,CPU_F_rerendered=False,AP_results_interpreted=False,
        task_efficacy_established=False,innovation_established=False,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=STATUS,Q=reference(output/'Q.json'),qualified=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND_DRAFT: ROOT must bind four actualCPU0 shards and failure16/supplement')
    run(args.spec)
