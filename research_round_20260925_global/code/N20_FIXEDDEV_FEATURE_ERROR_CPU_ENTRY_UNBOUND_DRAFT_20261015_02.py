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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 25 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 25 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Original failed9 and typed21/22/23/24/25 stay separate; normal+25 terminal identities')
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
    require(spec['schema'] == (N20_DEV_CPU_SCHEMA if cpu else N20_DEV_SCHEMA) and
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


"""UNBOUND N20 eight-scope DEV CPU metadata front; no RAW prefix, run or CLI.

ROOT appends the existing typed25 RAW-first prefix and changes only its current
CPU_SCHEMA branch to N20_DEV_CPU_SCHEMA. Historical SCHEMA/recovery schemas and
failed-parent specs stay unchanged. No GPU source_gate is called or monkeypatched.
Four original GPU141..144 actual0 records must precede CPU145+cid. This draft does
not certify any future execution. CPU source/admission/phase/context remain own.

The shared recovery/legacy/prior metadata function below is copied from ordinary
GPU front02, without its current-owner GPU gate. It takes a CLOSED ORIGINAL GPU
spec: that spec's CPU_adapter_source is the old common-PR decoder. The own CPU
spec's CPU_adapter_source is the new F9 decoder and is never substituted into the
old hook. This is an explicit parent argument, not an edited scientific view.

ROOT supplies inherited_source_gate, typed22/23/24/25 helpers, legacy_gate and
n20_prior_DEV_carrier_gate exactly as for the original GPU front02. The prefix
continues to require all normal/failed RAW and pin checks before first Source SHA.
Only stdlib is imported here. Runtime tensor/model/numeric/RLE proof is in tail.
"""
import json
from pathlib import Path

N20_DEV_CPU_SCHEMA = 'ROOT-N20-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1'
N20_CPU_RANGES = ((0,24),(24,47),(47,71),(71,94),(94,118),(118,141),(141,164),(164,187))
N20_CPU_PARENT_GPU_IDS = (0,0,1,1,2,2,3,3)
N20_COMPLETED_GPU_KEYS = {'shard_id','dataset_ordinal_range','source','spec','report','RAW',
    'admission','phase_binding','bootstrap','shard_owner_index'}
N20_CPU_SOURCE_KEYS = ('GPU_outer_source','CPU_adapter_source','selector_cpu_proof_source',
    'legacy_selector_proof_source','endpoint_body_source','PR_formula_source','labels_source',
    'N20_failed_DEV_capture_stderr')
N20_GPU_BINDING_KEYS = ('source','spec','GPU_admission','phase','N20_root_training_qualified_result',
    'N20_locked_protocol','legacy_qualified_result','new_eval_body_source','capture_body_source',
    'coordinates_source','checkpoints','shard_id','dataset_ordinal_range')

N20_DEV_SCHEMA = 'ROOT-N20-fixedDEV-local-center-feature-error-GPU-shard-spec-v1'
N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
N20_TRAIN_CPU_SCHEMA = 'ROOT-N20-paired-feature-error-saved-artifact-recovery-CPU-spec-v1'
N20_DEV_TRAINING_BINDING_KEYS = (
    'source', 'spec', 'GPU_admission', 'phase', 'N20_locked_protocol', 'locked_protocol',
    'parent_cache_report', 'neighbor_report', 'scout_module_source', 'base_scout_source',
    'paired_training_body_source', 'original_paired_training_body_source',
    'initialization_seed', 'order_seed', 'expected_N20_initialization_digest',
    'frozen_teacher_checkpoint', 'frozen_PR_checkpoint', 'N20_root_cache_qualified_result',
    'N20_initialization_lock', 'N20_cache_CPU_Q')
N20_DEV_RANGES = ((0, 47), (47, 94), (94, 141), (141, 187))


def _n20_dev_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)

def n20_completed_cache_record(record, output_key, owner_index, cuda, gate, spec):
    """Close cache GPU138/CPU139 or recovery CPU140; never completed GPU140."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner_index
        and (owner_index, output_key, cuda) in ((138, 'report', '0'), (139, 'Q', ''), (140, 'Q', '')),
        'Exact eight-field original completion record required')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner_index]
    require(all(row[key] == record[record_key] for key, record_key in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Completed record must equal its original RAW prefix row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        record['RAW'], record['admission'], record['source'], record['spec'], cuda)
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    expected_failed = {138: 21, 139: 21, 140: 24}[owner_index]
    admission_key = 'GPU_admission' if cuda == '0' else 'CPU_admission'
    filename = 'REPORT.json' if output_key == 'report' else 'Q.json'
    require(type(joined) is int and joined == owner_index == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner_index
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner_index]
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['total_failed_RAW_count'] == expected_failed
        and parent['total_terminal_identity_pin_count'] == owner_index + expected_failed
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and Path(parent[admission_key + '_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / filename).resolve(strict=True),
        'Completed original owner/source/spec/admission/pins/prefix/output locator must be exact')
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Completed pending output must bind the actual original owner')
    return parent, output


def n20_recovered_training_metadata_gate(spec, gate):
    """Read-only parent metadata; spec is the original closed GPU spec, not own."""
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_training_qualified_result', 'N20_root_cache_qualified_result',
            'N20_initialization_lock', 'N20_locked_protocol', 'locked_protocol', 'cache_GPU_report',
            'region_common_source', 'new_eval_body_source', 'legacy_qualified_result',
            'N20_training_tail_source', 'N20_failed_training_stderr'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_dev_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'Complete original N20 protocol literal, separate from inherited N16 protocol')
    metadata = read(closed(closure, spec['N20_initialization_lock']))
    digest = spec['expected_N20_initialization_digest']
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        and digest == 'a13802fc0bf4f23a8ab368b11b15a24b60552baadf1ab8264daf242173854a68'
        and metadata['schema'] == 'Root-N20-source-only-private-initialization-lock-v1'
        and metadata['seed'] == 202610150201 and metadata['combined_digest'] == digest
        and metadata['CUDA_initialized'] is False and metadata['training_executed'] is False
        and all(metadata[key] is True for key in ('paired_state_exact', 'paired_storage_independent',
            'spatial_weights_zero', 'external_CPU_RNG_unchanged')),
        'Use original prelocked source-only initialization; never synthesize unknown digest')
    initialization_sources = [spec[key] for key in model_keys]
    if 'N20_training_tail_source' in spec:
        closed(closure, spec['N20_training_tail_source'])
        initialization_sources.append(spec['N20_training_tail_source'])
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'All original initialization source basenames and hashes')
    cache = read(closed(closure, spec['N20_root_cache_qualified_result']))
    require(_n20_dev_exact(cache, spec['expected_N20_root_cache_qualified_result'])
        and cache['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and cache['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and cache['qualified'] is True and cache['original_GPU_CPU_actual0_verified'] is True
        and cache['full_proofs_ROOT_verified'] is True and cache['images'] == 741
        and cache['sampled_points'] == 3035136 and cache['unique_centers'] == 1786820
        and cache['parent_cache'] == spec['cache_GPU_report'] and cache['protocol'] == spec['N20_locked_protocol']
        and cache['GPU_report'] == cache['gpu_record']['report'] and cache['CPU_Q'] == cache['CPU_record']['Q'],
        'Original qualified cache envelope and GPU138/CPU139 refs remain unchanged')
    cache_gpu, cache_report = n20_completed_cache_record(cache['gpu_record'], 'report', 138, '0', gate, spec)
    cache_cpu, cache_q = n20_completed_cache_record(cache['CPU_record'], 'Q', 139, '', gate, spec)
    require(cache_gpu['N20_locked_protocol'] == cache_cpu['N20_locked_protocol'] == spec['N20_locked_protocol']
        and cache_report['protocol'] == cache_q['protocol'] == spec['N20_locked_protocol']
        and cache_q['capture_GPU_report'] == cache['GPU_report'] and cache_q['qualified'] is False
        and cache_q['ROOT_actual_CPU0_verified'] is False,
        'Do not rewrite old cache pending outputs as current qualified records')
    root = read(closed(closure, spec['N20_root_training_qualified_result']))
    require(_n20_dev_exact(root, spec['expected_N20_root_training_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-paired-feature-error-fixed24-saved-artifact-recovery-qualified-v1'
        and root['status'] == 'N20_PAIRED_E8_E24_SAVED_ARTIFACTS_CPU_RECOVERY_QUALIFIED'
        and all(root[key] is True for key in ('qualified', 'original_GPU_exit1_preserved',
            'recovery_CPU_actual0_verified', 'full_saved_endpoint_ROOT_verified'))
        and root['original_GPU_actual0_verified'] is False
        and root['images'] == 741 and root['epochs'] == 24
        and root['sampled_points_per_epoch'] == 3035136 and root['updates_each_arm'] == 17784
        and root['parameters_per_arm'] == 18851 and root['tensors_per_arm'] == 5
        and root['cache_qualified_result'] == spec['N20_root_cache_qualified_result']
        and root['protocol'] == spec['N20_locked_protocol']
        and root['initialization_lock'] == spec['N20_initialization_lock']
        and root['initialization_digest'] == digest
        and root['CPU_Q'] == root['CPU_record']['Q']
        and root['failed_training_record'] == spec['N20_failed_training_record']
        and not any(key in root for key in ('GPU_report', 'gpu_record', 'training_GPU_report')),
        'Mandatory saved-artifact ROOT qualification preserves GPU actual1; no GPU report/completion')
    cpu, q = n20_completed_cache_record(root['CPU_record'], 'Q', 140, '', gate, spec)
    record = root['failed_training_record']
    require(type(record) is dict and set(record) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding', 'bootstrap',
        'entry_source', 'spec', 'original_report_absent'}
        and type(record['returncode']) is int and record['returncode'] == 1
        and record['original_report_absent'] is True,
        'Preserved failed training record exact10; never completed GPU140')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, record[key])
    gpu = read(closed(closure, record['spec']))
    require(gpu['schema'] == N20_TRAIN_SCHEMA and cpu['schema'] == N20_TRAIN_CPU_SCHEMA
        and gpu['normal_RAW_count'] == gpu['expected_normal_RAW_count'] == 140
        and gpu['total_failed_RAW_count'] == 23 and gpu['total_terminal_identity_pin_count'] == 163
        and gpu['RAW_inputs'] == spec['RAW_inputs'][:140]
        and gpu['entry_source'] == record['entry_source'] and gpu['phase'] == record['phase']
        and all(closure.get(ref['path']) == ref for ref in gpu['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in gpu['terminal_identity_pins'].items())
        and Path(gpu['GPU_admission_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and cpu['N20_failed_training_record'] == record,
        'Original failed GPU140/23/163 and independent recovery CPU140/24/164 identities stay separate')
    shared_keys = model_keys + ('N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_initialization_lock', 'expected_N20_initialization_digest', 'N20_locked_protocol',
        'expected_N20_locked_protocol', 'locked_protocol', 'cache_GPU_report', 'region_common_source',
        'initialization_seed', 'N20_training_tail_source')
    require(all(_n20_dev_exact(parent[key], spec[key]) for parent in (gpu, cpu) for key in shared_keys),
        'Preserved failed/recovery parents retain identical original cache/model/init/protocol refs')
    require(q['schema'] == 'N20-paired-local-center-feature-error-training-recovery-CPU-qualification-v1'
        and q['status'] == 'N20_PAIRED_FEATURE_ERROR_E8_E24_RECOVERY_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and q['normal_RAW_count'] == 140 and q['total_failed_RAW_count'] == 24
        and q['total_terminal_identity_pin_count'] == 164
        and q['preserved_failed_training_GPU'] == record
        and q['recovered_from_preserved_failed_GPU'] is True
        and q['original_GPU_actual0_verified'] is False and q['original_GPU_REPORT_created'] is False
        and q['images'] == 741 and q['fixed_endpoints'] == [8, 24]
        and q['natural_queries'] == 74100 and q['sampled_points_per_endpoint'] == 3035136
        and q['CPU_packet_pair_reads'] == 1482 and q['total_CPU_endpoint_sampled_points'] == 6070272
        and q['parameters_per_arm'] == 18851 and q['tensors_per_arm'] == 5
        and q['E8_updates_per_arm'] == 5928 and q['E24_updates_per_arm'] == 17784
        and q['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR']
        and q['qualified'] is False and q['ROOT_actual_CPU0_verified'] is False
        and q['original_owned_terminal_self_certified'] is False
        and all(q[key] is False for key in ('GPU_pointwise_logits_qualified', 'optimizer_trajectory_replayed',
            'selected_indices_qualified', 'mask_RLE_qualified', 'AP_evaluated',
            'teacher_or_PR_loaded', 'full_native_L_loaded'))
        and type(q['checks']) is dict and bool(q['checks']) and all(v is True for v in q['checks'].values()),
        'Immutable recovery CPU Q is endpoint-only; preserved GPU actual1 is not upgraded')
    closed(closure, q['details'])
    closed(closure, cpu['qualifier_body_source'])
    require(q['qualifier_body_source'] == cpu['qualifier_body_source'], 'Original CPU numerical body exact')
    for key in model_keys + ('N20_locked_protocol', 'locked_protocol', 'N20_root_cache_qualified_result',
            'N20_initialization_lock', 'initialization_seed'):
        require(q[key] == spec[key], 'Original recovery CPU training binding differs: ' + key)
    require(q['neighbor_report'] == cache['GPU_report']
        and q['parent_cache_report'] == spec['cache_GPU_report']
        and q['N20_cache_CPU_Q'] == cache['CPU_Q']
        and q['initialization_seed'] == 202610150201 and q['order_seed'] == 202610140160001
        and q['expected_N20_initialization_digest'] == digest,
        'Actual training uses the same independently qualified extra-F and original center caches')
    artifacts = root['saved_artifacts']
    require(type(artifacts) is dict and set(artifacts) == {'START', 'checkpoints', 'FIT_endpoints', 'UPDATE_TRACE'}
        and type(artifacts['checkpoints']) is dict and type(artifacts['FIT_endpoints']) is dict
        and set(artifacts['checkpoints']) == set(artifacts['FIT_endpoints']) == {'8', '24'}
        and cpu['N20_saved_training_artifacts'] == q['recovery_artifact_refs'] == artifacts,
        'Exactly original saved E8/E24 references; no DEV-selected replacement')
    output = Path(gpu['output_directory'])
    require(output.is_absolute() and output.is_dir() and not output.is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink(),
        'Preserved failed-training output exists while original REPORT remains absent')
    named_refs = [
        (artifacts['START'], 'START.json'), (artifacts['UPDATE_TRACE'], 'UPDATE_TRACE.jsonl'),
        (artifacts['checkpoints']['8'], 'E08_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['checkpoints']['24'], 'E24_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['FIT_endpoints']['8'], 'E08_FIT_ENDPOINT.json'),
        (artifacts['FIT_endpoints']['24'], 'E24_FIT_ENDPOINT.json')]
    require(len({ref['path'] for ref, _ in named_refs}) == 6, 'Six distinct original saved artifact refs')
    for ref, filename in named_refs:
        require(Path(closed(closure, ref)).resolve(strict=True) == (output / filename).resolve(strict=True),
            'Saved artifact exact original failed-output filename: ' + filename)
    old_report = read(closed(closure, spec['cache_GPU_report']))
    binding = dict(source=record['entry_source'], spec=record['spec'],
        GPU_admission=record['admission'], phase=record['phase'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=spec['cache_GPU_report'], neighbor_report=cache['GPU_report'],
        initialization_seed=202610150201, order_seed=202610140160001,
        expected_N20_initialization_digest=digest,
        frozen_teacher_checkpoint=old_report['teacher_checkpoint'],
        frozen_PR_checkpoint=old_report['fixed_PR_checkpoint'],
        N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
        N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=cache['CPU_Q'])
    binding.update({key: spec[key] for key in model_keys})
    start = read(closed(closure, artifacts['START']))
    require(set(binding) == set(N20_DEV_TRAINING_BINDING_KEYS)
        and q['original_training_binding'] == binding
        and type(start) is dict and all(start[key] == value for key, value in binding.items())
        and start['schema'] == 'N20-paired-feature-error-training-start-v1',
        'Original saved START and recovery Q have exact20 failed-owner/cache/model/init bindings')
    require(callable(globals().get('legacy_gate')), 'ROOT must append the known ordinary N17 legacy metadata gate')
    require(legacy_gate(spec, gate) is gate, 'Preserve original fully qualified DEV shared carrier metadata gate')
    require(len(gate['legacy_rows']) == 187 and len(gate['legacy_reports']) == 4
        and len(gate['legacy_manifests']) == 4 and set(gate['populations']) == {
            'native_official', 'strong_fixed_E24_Omega'}, 'Original full187 native/R and physical carrier ABI')
    common = load(spec['region_common_source'], '_N20_DEV_original_region_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original library gate ABI remains exact')
    gate.update(common=common, N20_protocol=protocol, initialization_metadata=metadata,
        N20_cache_root=cache, N20_training_root=root,
        training=dict(root=root, failed_gpu_spec=gpu, failed_gpu_record=record, cpu=cpu, q=q,
            saved_artifacts=artifacts, start=start, original_binding=binding,
            checkpoints=artifacts['checkpoints'], FIT_endpoints=artifacts['FIT_endpoints']),
        trained_checkpoints=artifacts['checkpoints'], N20_training_CPU_Q_ref=root['CPU_Q'])
    require('parent' in gate and 'lineage' in gate and 'evaluation' in gate,
            'Inherited original endpoint parent/lineage/fixed PR loader must remain available')
    require(callable(globals().get('n20_prior_DEV_carrier_gate')),
        'UNBOUND: ROOT must supply exact metadata-only native/shared/common-PR/decoder reference gate')
    require(n20_prior_DEV_carrier_gate(spec, gate) is gate,
        'ROOT prior DEV carrier hook must preserve this verified gate; no default true or guessed fields')
    return gate

def n20_cpu_policy_names():
    return ('native_official', 'strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch, arm, mode)
        for epoch in (24, 8) for arm in ('LOCAL_ERROR', 'CENTER_ERROR')
        for mode in ('atmost', 'exact'))


def n20_validate_cpu_header(spec):
    require(type(spec) is dict and spec['schema'] == N20_DEV_CPU_SCHEMA
        and type(spec['cpu_shard_id']) is int and 0 <= spec['cpu_shard_id'] < 8
        and type(spec['CPU_parent_GPU_id']) is int
        and spec['CPU_parent_GPU_id'] == N20_CPU_PARENT_GPU_IDS[spec['cpu_shard_id']]
        and spec['cpu_dataset_ordinal_range'] == list(N20_CPU_RANGES[spec['cpu_shard_id']])
        and all(type(spec[k]) is int for k in ('normal_RAW_count', 'expected_normal_RAW_count',
            'cpu_shard_owner_index', 'total_failed_RAW_count', 'total_terminal_identity_pin_count',
            'initialization_seed', 'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] ==
            spec['cpu_shard_owner_index'] == 145 + spec['cpu_shard_id']
        and spec['total_failed_RAW_count'] == 25
        and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + 25
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent CPU145+cid/failed25 own entry/schema/resources and fixed eight whole-image scopes')
    records = spec['completed_GPU_shards']
    require(type(records) is list and len(records) == 4, 'Four original completed GPU descriptors required before any CPU scope')
    for sid, item in enumerate(records):
        require(type(item) is dict and set(item) == N20_COMPLETED_GPU_KEYS
            and type(item['shard_id']) is int and item['shard_id'] == sid
            and item['dataset_ordinal_range'] == list(N20_DEV_RANGES[sid])
            and type(item['shard_owner_index']) is int and item['shard_owner_index'] == 141 + sid
            and item['shard_owner_index'] < spec['normal_RAW_count'],
            'Exact ten-key original GPU141..144 descriptors in global ordinal order')
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            ref_shape(item[key])
    require(spec['GPU_shard'] == records[spec['CPU_parent_GPU_id']]
        and spec['GPU_outer_source'] == spec['GPU_shard']['source'],
        'Mapped complete original GPU descriptor/source; never normal-minus-one ownership guesses')
    return True


def n20_completed_DEV_GPU_record(spec, gate, item):
    c = gate['closure']
    for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
        closed(c, item[key])
    parent = read(closed(c, item['spec']))
    report = read(closed(c, item['report']))
    owner = item['shard_owner_index']; sid = item['shard_id']; row = spec['RAW_inputs'][owner]
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        item['RAW'], item['admission'], item['source'], item['spec'], '0')
    require(type(joined) is int and joined == owner
        and parent['schema'] == N20_DEV_SCHEMA and parent['entry_source'] == item['source']
        and parent['normal_RAW_count'] == parent['expected_normal_RAW_count'] == owner
        and parent['total_failed_RAW_count'] == 25 and parent['total_terminal_identity_pin_count'] == owner + 25
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and parent['shard_id'] == sid and parent['dataset_ordinal_range'] == list(N20_DEV_RANGES[sid])
        and all(row[k] == item[v] for k, v in (('receipt', 'RAW'), ('admission', 'admission'),
            ('phase_binding', 'phase_binding'), ('bootstrap', 'bootstrap')))
        and row['phase'] == parent['phase']
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k, v in parent['terminal_identity_pins'].items())
        and Path(parent['GPU_admission_file']).resolve(strict=True) == Path(item['admission']['path']).resolve(strict=True)
        and Path(item['report']['path']).resolve(strict=True) == (Path(parent['output_directory']) / 'REPORT.json').resolve(strict=True),
        'Original owned GPU actual0 CUDA0/source/spec/ad/bootstrap/prefix/full Sources/pins/report locator')
    common_keys = ('N20_root_training_qualified_result', 'expected_N20_root_training_qualified_result',
        'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_failed_DEV_capture_record', 'N20_failed_DEV_capture_stderr',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'N20_initialization_lock',
        'expected_N20_initialization_digest', 'N20_failed_training_record', 'N20_failed_training_stderr',
        'N20_training_tail_source', 'locked_protocol', 'cache_GPU_report', 'region_common_source',
        'scout_module_source', 'base_scout_source', 'paired_training_body_source',
        'original_paired_training_body_source', 'initialization_seed', 'legacy_qualified_result',
        'legacy_shards', 'n17_root_qualified_result_ref', 'expected_n17_root_qualified_result',
        'new_eval_body_source', 'legacy_eval_body_source', 'capture_body_source', 'coordinates_source', 'labels_source')
    require(all(_n20_dev_exact(parent[k], spec[k]) for k in common_keys)
        and all(_n20_dev_exact(v, spec[k]) for k, v in parent.items() if k.startswith('additional_preserved_')),
        'CPU retains all original frozen model/cache/lock/native/prior and failed-record semantics')
    binding = dict(source=item['source'], spec=item['spec'], GPU_admission=item['admission'], phase=parent['phase'],
        N20_root_training_qualified_result=parent['N20_root_training_qualified_result'], N20_locked_protocol=parent['N20_locked_protocol'],
        legacy_qualified_result=parent['legacy_qualified_result'], new_eval_body_source=parent['new_eval_body_source'],
        capture_body_source=parent['capture_body_source'], coordinates_source=parent['coordinates_source'],
        checkpoints=gate['trained_checkpoints'], shard_id=sid, dataset_ordinal_range=list(N20_DEV_RANGES[sid]))
    names = n20_cpu_policy_names(); first, last = N20_DEV_RANGES[sid]
    require(set(binding) == set(N20_GPU_BINDING_KEYS)
        and report['schema'] == 'N20-fixedDEV-feature-error-GPU-shard-report-v1'
        and report['status'] == 'N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
        and all(report[k] == v for k, v in binding.items())
        and report['normal_RAW_count'] == owner and report['total_failed_RAW_count'] == 25
        and report['total_terminal_identity_pin_count'] == owner + 25
        and report['qualified'] is False and report['ROOT_actual_GPU0_verified'] is False
        and report['original_training_GPU_actual0_verified'] is False
        and report['training_artifacts_recovery_qualified'] is True
        and report['CPU_numerical_qualification_pending'] is True and report['models_frozen'] is True
        and report['teacher_forward_calls'] == report['new_fullC_PR_forward_calls'] == 0
        and report['fresh_selected_PR'] is True and report['reused_common_C_reference'] is True
        and report['AP_evaluated'] is False and report['local_AP_computed'] is False
        and report['full187_qualification_claimed'] is False and report['standalone_FPS_measured'] is False
        and report['cleanup'] == {'allocated_after': 0, 'reserved_after': 0}
        and report['controls'] == list(names) and set(report['control_results']) == set(names),
        'Immutable pending GPU report; terminal0 comes from original owner, never output self-certification')
    manifest = read(closed(c, report['capture_manifest'])); start = read(closed(c, report['START']))
    output = Path(parent['output_directory'])
    require(Path(report['capture_manifest']['path']).resolve(strict=True) == (output / 'CAPTURE_MANIFEST.json').resolve(strict=True)
        and Path(report['START']['path']).resolve(strict=True) == (output / 'START.json').resolve(strict=True)
        and manifest['schema'] == 'N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
        and start['schema'] == 'N20-fixedDEV-feature-GPU-shard-start-v1'
        and all(manifest[k] == start[k] == v for k, v in binding.items())
        and start['original_GPU_training_actual0'] is False and start['recovery_CPU_qualified'] is True
        and start['policies'] == list(names)
        and manifest['images'] == report['images'] == last - first
        and manifest['queries'] == report['queries'] == (last - first) * 100
        and manifest['global_images'] == 187 and manifest['global_queries'] == 18700
        and manifest['internal_member_locators_are_strings'] is True
        and report['scope_dataset_indices'] == list(range(first, last))
        and len(manifest['rows']) == last - first
        and manifest['physical_capture_refs'] == report['capture_refs'] == [r['bundle'] for r in manifest['rows']],
        'Complete original GPU ordinal scope, START, manifest and actual physical captures')
    for i, captured in enumerate(manifest['rows'], start=first):
        origin = gate['legacy_rows'][i]; old = origin['row']; common = gate['prior_common_rows'][i]
        carrier = dict(bundle=old['bundle'], shared_member=old['shared_actual_image_inputs'], original_GPU_binding=origin['binding'],
            common_PR_bundle=common['row']['bundle'], common_PR_original_GPU_binding=common['binding'])
        require(captured['dataset_index'] == i and captured['image_id'] == old['image_id']
            and captured['identity'] == old['identity'] and captured['qids'] == old['qids']
            and captured['input_carrier'] == carrier and captured['natural_queries'] == 100
            and captured['neighbor_input_member'] == 'i%03d/FEATURE_NEIGHBORS_INPUTS.pth' % i
            and type(captured['member_locators']) is list
            and all(type(v) is str for v in captured['member_locators'])
            and len(captured['member_locators']) == len(set(captured['member_locators']))
            and type(captured['query_profiles']) is list and len(captured['query_profiles']) == 100
            and [v['query_id'] for v in captured['query_profiles']] == list(range(100)),
            'Full natural100 new-F9 rows with original input/native/prior-common provenance')
        closed(c, captured['bundle']); closed(c, carrier['bundle']); closed(c, carrier['common_PR_bundle'])
    image_ids = [r['image_id'] for r in manifest['rows']]
    require(report['scope_image_ids'] == image_ids and len(set(image_ids)) == last - first, 'Original canonical image namespace/order')
    expected = {}; grouped = {}
    for name in names:
        value = report['control_results'][name]
        require(type(value) is dict and set(value) == {'predictions'}, 'Actual predictions are strict physical refs, not inline COCO')
        values = read(closed(c, value['predictions']))
        require(type(values) is list and len(values) == (last - first) * 100, 'Complete original GPU local prediction list')
        groups = {}
        for prediction in values:
            groups.setdefault(prediction['image_id'], []).append(prediction)
        require(list(groups) == image_ids and all(
            [p['query_id'] for p in groups[r['image_id']]] == r['qids'] for r in manifest['rows']),
            'All100 query export order; never force natural order onto exported predictions')
        for captured in manifest['rows']:
            baseline = gate['populations']['native_official'][captured['image_id']]
            observed = groups[captured['image_id']]
            require(all({k: v for k, v in p.items() if k != 'segmentation'} ==
                        {k: v for k, v in q.items() if k != 'segmentation'} for p, q in zip(observed, baseline)),
                'Original class/S0/query identity and all nonsegmentation fields stay unchanged')
            if name in ('native_official', 'strong_fixed_E24_Omega'):
                require(observed == gate['populations'][name][captured['image_id']], 'Original native/strong R predictions exact')
        expected[name] = values; grouped[name] = groups
    return dict(descriptor=item, spec=parent, report=report, manifest=manifest, START=start,
        binding=binding, expected_predictions=expected, expected_prediction_groups=grouped)


def n20_scope_parent_GPU(spec, gate):
    cid = spec['cpu_shard_id']; pid = spec['CPU_parent_GPU_id']
    first, last = N20_CPU_RANGES[cid]; pa, pz = N20_DEV_RANGES[pid]
    rows = gate['parent_GPU_manifest']['rows']
    require(pa <= first < last <= pz and spec['cpu_dataset_ordinal_range'] == [first, last]
        and [r['dataset_index'] for r in rows] == list(range(pa, pz)), 'Immutable whole parent before CPU child slicing')
    selected = rows[first-pa:last-pa]; ids = [r['image_id'] for r in selected]
    groups = {name: {image_id: parent[image_id] for image_id in ids}
              for name, parent in gate['expected_prediction_groups'].items()}
    expected = {name: [p for r in selected for p in parent[r['image_id']]] for name, parent in groups.items()}
    require(len(selected) == last-first and len(set(ids)) == last-first
        and all(len(v) == (last-first)*100 for v in expected.values()), 'Complete child images/all100; no failed partials')
    gate.update(scope_rows=selected, scope_capture_refs=[r['bundle'] for r in selected], scope_image_ids=ids,
        scope_expected_predictions=expected, scope_expected_prediction_groups=groups)
    return gate


def source_gate(path, spec):
    n20_validate_cpu_header(spec)
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec)
    failed25 = n20_failed25_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    n20_failed24_closed(spec, gate, failed24)
    n20_failed25_closed(spec, gate, failed25)
    c = gate['closure']
    for key in N20_CPU_SOURCE_KEYS:
        closed(c, spec[key])
    partition = read(closed(c, spec['CPU_execution_partition_supplement']))
    require(type(partition) is dict and type(spec['expected_CPU_execution_partition_supplement']) is dict
        and _n20_dev_exact(partition, spec['expected_CPU_execution_partition_supplement']),
        'Exact complete ROOT-prelocked eight-scope CPU execution supplement, not a derived scientific spec')
    gate['CPU_execution_partition_supplement'] = partition
    for item in spec['completed_GPU_shards']:
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            closed(c, item[key])
    original_parent_spec = read(closed(c, spec['GPU_shard']['spec']))
    # This pure parent function never calls original GPU source_gate/own_gate.
    n20_recovered_training_metadata_gate(original_parent_spec, gate)
    parents = [n20_completed_DEV_GPU_record(spec, gate, item) for item in spec['completed_GPU_shards']]
    parent = parents[spec['CPU_parent_GPU_id']]
    gpu = load(spec['GPU_outer_source'], '_N20_completed_DEV_GPU_metadata_helpers')
    require(Path(gpu.__file__).resolve(strict=True) == Path(spec['GPU_outer_source']['path']).resolve(strict=True)
        and callable(getattr(gpu, 'n20_model_load_cpu', None))
        and callable(getattr(gpu, 'n20_policy_names', None))
        and tuple(gpu.n20_policy_names()) == n20_cpu_policy_names(), 'Closed original GPU pure CPU model loader/policy ABI')
    gate.update(gpu_helpers=gpu, completed_GPU_shards=spec['completed_GPU_shards'],
        parent_GPU=parent['spec'], parent_GPU_report=parent['report'], parent_GPU_manifest=parent['manifest'],
        parent_GPU_START=parent['START'], parent_GPU_binding=parent['binding'],
        expected_predictions=parent['expected_predictions'], expected_prediction_groups=parent['expected_prediction_groups'])
    n20_scope_parent_GPU(spec, gate)
    return gate


"""UNBOUND metadata-only prior DEV hook, appended after N20 DEV front02.

Requires existing globals require/read/closed and inherited lineage/context/
owners. legacy_gate must already have populated legacy_rows/root/populations.
No source_gate/training_gate from N16/N17 is called; no Torch or tensor load.

Required existing spec fields (all refs must be in complete SOURCE):
  n17_root_qualified_result_ref, expected_n17_root_qualified_result (whole JSON),
  CPU_adapter_source (the same decoder ref as all seven original numeric CPUs).
Optional aliases prior_common_decoder_source/prior_shared_validator_source must
equal that decoder/original Source04 respectively. The latter is derived from
qualified parent1.source, never a fabricated locator. Neither old entry nor
training_gate may run. legacy_qualified_result/legacy_shards still mean N16;
N17 physical parents are obtained from qualified finalQ.numeric_CPU_shards.
Existing inherited context supplies N10_evaluation_source, reader_source,
renderer_source, DEV, COCO_source, export_source, native_reference_predictions.
Their values are not guessed or copied from an unrelated cache.

Outputs required by DEV tail: DEV_packet_module_refs exact{n10,reader,renderer},
DEV_packet_gate (separate gate copy whose report is the original prior report),
DEV_annotation_ref, DEV_packet_rows, DEV_packet_refs, prior_common_rows (187
items exact{row,binding}), prior_common_decoder_source. Actual shared/bundle
tensors, registry, common-C roles and same-input PR still require owned checks.
This hook preserves original gate parent/evaluation/common/context identities.
"""

import json
from pathlib import Path


N20_PRIOR_GPU_RANGES = ((0, 47), (47, 70), (70, 94), (94, 141), (141, 187))
N20_PRIOR_CPU_RANGES = ((0, 47), (47, 70), (70, 94), (94, 117), (117, 141), (141, 164), (164, 187))
N20_PRIOR_CPU_PARENT_IDS = (0, 1, 2, 3, 3, 4, 4)
N20_PRIOR_GPU_KEYS = {'shard_id', 'dataset_ordinal_range', 'source', 'spec', 'report',
    'RAW', 'admission', 'phase_binding', 'bootstrap', 'shard_owner_index'}
N20_PRIOR_CPU_KEYS = {'cpu_shard_id', 'cpu_dataset_ordinal_range', 'source', 'spec', 'report',
    'RAW', 'admission', 'phase_binding', 'bootstrap', 'cpu_shard_owner_index'}
N20_PRIOR_BINDING_KEYS = ('source', 'spec', 'GPU_admission', 'phase', 'ROOTtrain_result',
    'legacy_qualified_result', 'experiment_lock', 'new_eval_body_source', 'checkpoints',
    'shard_id', 'dataset_ordinal_range')


def _n20_prior_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _n20_prior_completed_CPU(record, output_key, owner, gate, spec):
    """Only metadata joins of already-normal original N17 final134/post135."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner
        and owner < spec['normal_RAW_count'], 'Explicit completed original N17 final/post record')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner]
    require(all(row[key] == record[other] for key, other in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Original final/post exact normal RAW row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], record['RAW'],
        record['admission'], record['source'], record['spec'], '')
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    require(type(joined) is int and joined == owner == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and parent['total_failed_RAW_count'] == 21 and parent['total_terminal_identity_pin_count'] == owner + 21
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and output['source'] == record['source'] and output['spec'] == record['spec']
        and output['CPU_admission'] == record['admission'] and output['phase'] == row['phase']
        and Path(parent['CPU_admission_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / ('Q.json' if output_key == 'Q' else 'ROOT_RESULT.json')).resolve(strict=True),
        'Original final/post owner/full-source/prefix/pins/admission/output metadata exact')
    return parent, output, read(closed(closure, record['RAW']))


def n20_prior_DEV_carrier_gate(spec, gate):
    closure = gate['closure']
    preserved = {key: gate[key] for key in ('parent', 'evaluation', 'common', 'context')}
    require(len(gate['legacy_rows']) == 187 and len(gate['legacy_reports']) == 4
        and len(gate['legacy_manifests']) == 4, 'Existing qualified N16 legacy metadata gate must run first')
    for key in ('n17_root_qualified_result_ref', 'CPU_adapter_source'):
        closed(closure, spec[key])
    decoder_ref = spec['CPU_adapter_source']
    if 'prior_common_decoder_source' in spec:
        closed(closure, spec['prior_common_decoder_source'])
        require(spec['prior_common_decoder_source'] == decoder_ref, 'Explicit decoder alias equals original CPU_adapter_source')

    # Reproduce only N16's prior/selector metadata setup, never its training_gate.
    context = dict(gate['context'])
    context['selector_mode'] = 'reuse'
    prior, prior_q = gate['lineage'].prior_gate(context, closure, gate['owners'])
    selector = gate['lineage'].selector_parent_gate(context, closure, gate['owners'])
    prior_rows = read(closed(closure, prior['image_rows']))
    packets = prior_q['packet_refs']
    cache = read(closed(closure, spec['cache_GPU_report']))
    require(prior['teacher_checkpoint'] == cache['teacher_checkpoint']
        and type(prior_rows) is list and type(packets) is list
        and len(prior_rows) == len(packets) == len(selector['manifest']['rows']) == 187
        and sum(row['raw_GT'] for row in prior_rows) == 380,
        'Original DEV prior teacher/187 packet rows/380 raw GT and selector population')
    for index, (row, packet, item) in enumerate(zip(prior_rows, packets, gate['legacy_rows'])):
        legacy = item['row']
        require(index == legacy['dataset_index'] and row['image_id'] == legacy['image_id']
            and packet == legacy['native_packet']
            and selector['manifest']['rows'][index] == legacy['selector'],
            'Original prior row/packet, frozen selector and qualified shared carrier stay associated')
        closed(closure, packet)
    module_refs = {name: context[key] for name, key in (
        ('n10', 'N10_evaluation_source'), ('reader', 'reader_source'), ('renderer', 'renderer_source'))}
    for ref in tuple(module_refs.values()) + tuple(context[key] for key in (
            'DEV', 'COCO_source', 'export_source', 'native_reference_predictions')):
        closed(closure, ref)
    packet_gate = dict(gate, context=context, report=prior, prior=prior, prior_q=prior_q,
                       packets=packets, selector=selector)

    # Existing published N18 metadata ABI for the fully qualified N17 carriers.
    root = read(closed(closure, spec['n17_root_qualified_result_ref']))
    require(_n20_prior_exact(root, spec['expected_n17_root_qualified_result'])
        and root['schema'] == 'Root-N17-full187-qualified-result-after-final-and-post-actual0-v1'
        and root['status'] == 'FULL187_FIXED24_MAIN_RESULT_QUALIFIED_CURRENT_RECIPE_STOP'
        and root['qualified'] is True and root['original_final_CPU_actual0_verified'] is True
        and root['original_post_CPU_actual0_verified'] is True and root['normal_RAW_count'] == 136
        and root['total_failed_RAW_count'] == 21 and root['total_terminal_identity_pin_count'] == 157
        and root['decision'] == 'STOP' and root['allowed_next_seeds'] == []
        and all(root[key] is False for key in ('additional_seed_or_fusion_executed', 'formal_TEST_evaluated',
            'cross_dataset_evaluated', 'innovation_established', 'TGRS_readiness_established')),
        'Exact completed original N17 Root scope/STOP; no new scientific qualification')
    final_spec, final_q, final_raw = _n20_prior_completed_CPU(root['final_record'], 'Q', 134, gate, spec)
    post_spec, post, post_raw = _n20_prior_completed_CPU(root['post_record'], 'ROOT_RESULT', 135, gate, spec)
    qualified = root['final_qualified_result']
    require(_n20_prior_exact(final_raw, root['original_final_actual'])
        and _n20_prior_exact(post_raw, root['original_post_actual'])
        and post['final_record'] == root['final_record'] and post['final_qualified_result'] == qualified
        and post['qualified'] is False and post['ROOT_post_actual_CPU0_verified'] is False
        and qualified['qualified'] is True and qualified['final_Q'] == root['final_record']['Q']
        and qualified['control_results'] == final_q['control_results']
        and qualified['numeric_CPU_shards'] == final_q['numeric_CPU_shards'] == final_spec['numeric_CPU_shards']
        and final_q['qualified'] is False and final_q['images'] == 187 and final_q['GT'] == 380
        and final_q['natural_queries'] == 18700 and root['primary_interpretation'] == qualified['primary_interpretation']
        and qualified['legacy_qualified_result'] == final_q['legacy_qualified_result'] == spec['legacy_qualified_result'],
        'Completed immutable final/post Qs and separately qualified Root carrier ancestry exact')

    descriptors = final_q['numeric_CPU_shards']
    require(type(descriptors) is list and len(descriptors) == 7, 'Existing seven qualified N17 numeric CPU scopes')
    parents, origins = {}, {}
    for cpu_id, descriptor in enumerate(descriptors):
        require(type(descriptor) is dict and set(descriptor) == N20_PRIOR_CPU_KEYS
            and descriptor['cpu_shard_id'] == cpu_id
            and descriptor['cpu_dataset_ordinal_range'] == list(N20_PRIOR_CPU_RANGES[cpu_id]),
            'Original canonical seven numeric CPU descriptors, never inferred owner arithmetic')
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            closed(closure, descriptor[key])
        cpu_parent = read(closed(closure, descriptor['spec']))
        item = cpu_parent['GPU_shard']
        require(cpu_parent['entry_source'] == descriptor['source']
            and cpu_parent['CPU_adapter_source'] == decoder_ref
            and cpu_parent['cpu_shard_id'] == cpu_id
            and cpu_parent['cpu_dataset_ordinal_range'] == descriptor['cpu_dataset_ordinal_range']
            and type(item) is dict and set(item) == N20_PRIOR_GPU_KEYS
            and item['shard_id'] == N20_PRIOR_CPU_PARENT_IDS[cpu_id],
            'Qualified numeric CPU parent names the exact shared common-PR decoder and physical GPU')
        sid = item['shard_id']
        if sid in parents:
            require(parents[sid]['descriptor'] == item, 'Repeated split CPU scopes retain identical whole GPU parent')
            continue
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            closed(closure, item[key])
        owner = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], item['RAW'],
            item['admission'], item['source'], item['spec'], '0')
        parent = read(closed(closure, item['spec']))
        report = read(closed(closure, item['report']))
        manifest = read(closed(closure, report['capture_manifest']))
        first, last = N20_PRIOR_GPU_RANGES[sid]
        failed = 19 if sid == 0 else 20
        row = spec['RAW_inputs'][122 + sid]
        require(type(owner) is int and owner == item['shard_owner_index'] == 122 + sid
            and parent['normal_RAW_count'] == parent['expected_normal_RAW_count'] == owner
            and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
            and parent['entry_source'] == item['source'] and parent['total_failed_RAW_count'] == failed
            and parent['total_terminal_identity_pin_count'] == owner + failed
            and all(closure.get(ref['path']) == ref for ref in parent['sources'])
            and all(spec['terminal_identity_pins'].get(key) == value
                    for key, value in parent['terminal_identity_pins'].items())
            and row['receipt'] == item['RAW'] and row['admission'] == item['admission']
            and row['phase_binding'] == item['phase_binding'] and row['bootstrap'] == item['bootstrap']
            and item['dataset_ordinal_range'] == parent['dataset_ordinal_range'] == [first, last]
            and report['schema'] == 'N17-fixedDEV-local-center-error-GPU-shard-report-v1'
            and report['status'] == 'N17_FIXEDDEV_GPU_SHARD_CAPTURE_COMPLETE_PENDING_ROOT_ACTUAL0_AND_INDEPENDENT_CPU'
            and manifest['schema'] == 'N17-fixedDEV-GPU-shard-image-manifest-v1'
            and manifest['physical_capture_refs'] == report['capture_refs'] == [r['bundle'] for r in manifest['rows']]
            and manifest['images'] == report['images'] == last - first
            and manifest['queries'] == report['queries'] == (last - first) * 100
            and manifest['global_images'] == 187 and manifest['global_queries'] == 18700
            and report['scope_dataset_indices'] == list(range(first, last)),
            'Original five GPU actual0/source/prefix/19-or20/whole-manifest refs remain exact')
        binding = {key: manifest[key] for key in N20_PRIOR_BINDING_KEYS}
        if sid:
            binding['execution_partition_supplement'] = manifest['execution_partition_supplement']
            require(binding['execution_partition_supplement'] == qualified['execution_partition_supplement']
                and report['failed_partials_reused'] is False, 'Original failed partials never common-PR evidence')
        expected = dict(source=item['source'], spec=item['spec'], GPU_admission=item['admission'],
            phase=parent['phase'], ROOTtrain_result=qualified['ROOTtrain_result'],
            legacy_qualified_result=spec['legacy_qualified_result'], experiment_lock=qualified['experiment_lock'],
            new_eval_body_source=parent['new_eval_body_source'], checkpoints=qualified['checkpoints'],
            shard_id=sid, dataset_ordinal_range=[first, last])
        require(all(binding[key] == report[key] == value for key, value in expected.items())
            and all(report[key] == value for key, value in binding.items()), 'Exact original GPU/manifest binding, not current N20 training')
        for index, current in enumerate(manifest['rows'], start=first):
            old = gate['legacy_rows'][index]
            legacy = old['row']
            carrier = dict(bundle=legacy['bundle'], shared_member=legacy['shared_actual_image_inputs'],
                           original_GPU_binding=old['binding'])
            require(current['dataset_index'] == index and current['image_id'] == legacy['image_id']
                and current['identity'] == legacy['identity'] and current['qids'] == legacy['qids']
                and current['input_carrier'] == carrier and current['natural_queries'] == 100
                and len(current['query_profiles']) == 100
                and [entry['query_id'] for entry in current['query_profiles']] == list(range(100))
                and current['image_id'] not in origins, 'N17 common-C carrier has original identity/namespace/export order')
            closed(closure, current['bundle'])
            closed(closure, carrier['bundle'])
            origins[current['image_id']] = dict(row=current, binding=binding)
        parents[sid] = dict(descriptor=item, report=report, manifest=manifest)
    ordered = [origins[item['row']['image_id']] for item in gate['legacy_rows']]
    require(set(parents) == set(range(5)) and len(origins) == 187
        and [item['row']['dataset_index'] for item in ordered] == list(range(187)),
        'Five qualified physical parents cover exactly187 images and natural18700 queries')
    validator_ref = parents[1]['descriptor']['source']
    if 'prior_shared_validator_source' in spec:
        closed(closure, spec['prior_shared_validator_source'])
        require(spec['prior_shared_validator_source'] == validator_ref,
            'Explicit shared-validator alias equals original qualified Source04 parent source')
    gate.update(DEV_packet_module_refs=module_refs, DEV_packet_gate=packet_gate,
        DEV_annotation_ref=context['DEV'], DEV_packet_rows=prior_rows, DEV_packet_refs=packets,
        prior_common_rows=ordered, prior_common_decoder_source=decoder_ref,
        N17_common_PR_root=root, N17_common_PR_parents=parents,
        prior_shared_validator_source=validator_ref)
    require(all(gate[key] is value for key, value in preserved.items()),
        'Original N20 inherited parent/evaluation/common/context must not be mutated or replaced')
    return gate


# UNBOUND_DRAFT: N20 eight-scope DEV CPU numeric run tail; no RAW/admission front or CLI.
"""ROOT's source_gate authenticates all four actual completed GPU shards first,
recovered-qualified training/cache/native/strong refs and the current own CPU.
Only ordinary source is prepared here. Importing this file performs no run.
"""
import gc
import json
import os
from pathlib import Path
import time

N20_CPU_RANGES = ((0,24),(24,47),(47,71),(71,94),(94,118),(118,141),(141,164),(164,187))
N20_CPU_PARENT_GPU_IDS = (0,0,1,1,2,2,3,3)
N20_CPU_GPU_RANGES = ((0,47),(47,94),(94,141),(141,187))
N20_CPU_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
N20_CPU_REPORT_SCHEMA = 'N20-fixedDEV-feature-error-CPU-numeric-shard-report-v1'
N20_CPU_STATUS = 'N20_FIXEDDEV_FEATURE_ERROR_CPU_NUMERIC_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_FULL_COCO'


def n20_cpu_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch,arm,mode) for epoch in (24,8)
        for arm in ('LOCAL_ERROR','CENTER_ERROR') for mode in ('atmost','exact'))


def n20_cpu_scope(spec, gate):
    """Verify front-provided immutable whole-parent and exact child metadata."""
    sid,pid=spec['cpu_shard_id'],spec['CPU_parent_GPU_id']
    require(type(sid) is int and 0<=sid<8 and type(pid) is int and pid==N20_CPU_PARENT_GPU_IDS[sid]
        and type(spec['normal_RAW_count']) is int and spec['normal_RAW_count']==145+sid
        and spec['cpu_shard_owner_index']==spec['normal_RAW_count'], 'Fixed CPU0..7 own145..152 and parent mapping')
    first,last=N20_CPU_RANGES[sid];a,z=N20_CPU_GPU_RANGES[pid]
    require(spec['cpu_dataset_ordinal_range']==[first,last] and a<=first<last<=z,
            'Prelocked whole-image child range, no changing the parent partition')
    completed=gate['completed_GPU_shards']
    require(type(completed) is list and len(completed)==4 and all(
        type(item) is dict and type(item['shard_id']) is int and item['shard_id']==i
        and item['dataset_ordinal_range']==list(N20_CPU_GPU_RANGES[i])
        and type(item['shard_owner_index']) is int and item['shard_owner_index']==141+i
        for i,item in enumerate(completed)), 'All four original GPU actual0 identities must have been joined by the front before any CPU')
    require(spec['GPU_shard']==completed[pid], 'Keep original complete parent GPU descriptor verbatim')
    report,manifest=gate['parent_GPU_report'],gate['parent_GPU_manifest']
    require(report['schema']=='N20-fixedDEV-feature-error-GPU-shard-report-v1'
        and report['status']=='N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
        and report['shard_id']==pid and report['dataset_ordinal_range']==[a,z]
        and report['scope_dataset_indices']==list(range(a,z))
        and manifest['schema']=='N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
        and manifest['dataset_ordinal_range']==[a,z] and manifest['shard_id']==pid
        and [row['dataset_index'] for row in manifest['rows']]==list(range(a,z))
        and report['capture_refs']==manifest['physical_capture_refs']==[row['bundle'] for row in manifest['rows']],
        'Complete actual original GPU parent report/manifest/capture population')
    rows=manifest['rows'][first-a:last-a]
    require(rows==gate['scope_rows'] and gate['scope_capture_refs']==[row['bundle'] for row in rows]
        and gate['scope_image_ids']==[row['image_id'] for row in rows]
        and len(set(gate['scope_image_ids']))==last-first and all(row['natural_queries']==100 for row in rows),
        'Own rows/physical refs/whole natural100 images are an exact immutable parent slice')
    require(tuple(report['controls'])==n20_cpu_policy_names()
        and set(gate['scope_expected_predictions'])==set(gate['scope_expected_prediction_groups'])==set(n20_cpu_policy_names()),
        'All ten fixed policies, no DEV winner substitution')
    for name in n20_cpu_policy_names():
        groups=gate['expected_prediction_groups'][name]
        expected=[value for row in rows for value in groups[row['image_id']]]
        require(expected==gate['scope_expected_predictions'][name] and len(expected)==(last-first)*100
            and gate['scope_expected_prediction_groups'][name]=={row['image_id']:groups[row['image_id']] for row in rows}
            and all([v['query_id'] for v in groups[row['image_id']]]==row['qids'] for row in rows),
            'Exact complete scoped original query/class/S0/prediction export order: '+name)
    return first,last


def n20_json_query_proof(value):
    return dict(value,profiles={name:{key:item for key,item in profile.items() if key!='reconstructed_mask'}
                               for name,profile in value['profiles'].items()})


def run(spec_path):
    started=time.monotonic()
    require(Path(spec_path).is_absolute(), 'Absolute independent N20 CPU specification')
    path=Path(spec_path).resolve(strict=True);spec=read(path);gate=source_gate(path,spec);c=gate['closure']
    first,last=n20_cpu_scope(spec,gate)
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(),
            'Fresh owned CPU output with an existing parent')
    require(spec['deadline_seconds']==1800 and spec['automatic_retry'] is False
        and spec['maximum_process_RSS_bytes']==16*1024**3, 'Unchanged single1800s CPU leaf/no retry/RSS16GiB')
    source_keys=('GPU_outer_source','CPU_adapter_source','selector_cpu_proof_source','legacy_selector_proof_source',
                 'endpoint_body_source','PR_formula_source','labels_source','scout_module_source')
    for key in source_keys:closed(c,spec[key])
    closed(c,gate['prior_shared_validator_source'])
    for key in ('N20_root_training_qualified_result','N20_locked_protocol','legacy_qualified_result','CPU_execution_partition_supplement'):
        closed(c,spec[key])
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and all(os.environ.get(key)=='2' for key in N20_CPU_THREAD_ENV)
        and psutil.virtual_memory().available>=16*1024**3 and psutil.disk_usage(str(output.parent)).free>=1024**3,
        'Actual independent CPU01/emptyCUDA/two threads/minRAM16GiB/output free1GiB')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0'
        and not torch.cuda.is_initialized(), 'Original closed libraries/cold CUDA')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    gpu=gate['gpu_helpers']
    require(Path(gpu.__file__).resolve(strict=True)==Path(spec['GPU_outer_source']['path']).resolve(strict=True)
        and callable(getattr(gpu,'n20_model_load_cpu',None)) and tuple(gpu.n20_policy_names())==n20_cpu_policy_names(),
        'Closed actual N20 GPU helper module; never call its GPU source_gate or run')
    adapter=load(spec['CPU_adapter_source'],'_N20_CPU_actual_F9_decoder')
    core=load(spec['selector_cpu_proof_source'],'_N20_CPU_actual_selected_numeric_core')
    endpoint=load(spec['endpoint_body_source'],'_N20_CPU_actual_independent_head_formula')
    legacy=load(spec['legacy_selector_proof_source'],'_N16_CPU_original_PR_numeric_kernel')
    manual=load(spec['PR_formula_source'],'_N20_CPU_original_PR_explicit_formula')
    helper=load(spec['labels_source'],'_N16_CPU_frozen_FP64_scoring')
    validator=load(gate['prior_shared_validator_source'],'_N17_CPU_actual_shared_validator')
    strong,models,frozen,model_proofs=gpu.n20_model_load_cpu(torch,spec,gate)
    require(core.ATOL==1e-4 and core.POINT_CHUNK==2048 and tuple(core.FORMULA_NAMES)==('original_head','independent_PR')
        and tuple(adapter.POLICIES)==n20_cpu_policy_names()[2:]
        and json.loads(json.dumps(model_proofs,allow_nan=False))==gate['parent_GPU_START']['model_proofs'],
        'New F9/five-tensor CPU core, fixed PR and original actual parent model proof bindings')
    states={key:frozen[key] for key in models};outside=torch.get_rng_state().clone()
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    peak_RSS=0;minimum_available_RAM=None
    def check():
        nonlocal peak_RSS,minimum_available_RAM
        rss=process.memory_info().rss;available=psutil.virtual_memory().available
        peak_RSS=max(peak_RSS,rss);minimum_available_RAM=available if minimum_available_RAM is None else min(minimum_available_RAM,available)
        require(time.monotonic()-started<1800 and rss<16*1024**3 and available>=16*1024**3
            and not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and all(os.environ.get(key)=='2' for key in N20_CPU_THREAD_ENV)
            and torch.get_num_threads()==torch.get_num_interop_threads()==cv2.getNumThreads()==2,
            'Bounded independent CPU/coldCUDA/affinity/thread/deadline; no retry')
    def original_head(F,e,L):
        check();value=strong(F,e,L,L.new_zeros((len(L),49)),arm='PR');u=value['residual'];return u,L+u
    def independent_PR(F,e,L):
        check();return manual.independent_PR(torch,frozen['PR'],F,L)
    formulas=dict(original_head=original_head,independent_PR=independent_PR)
    def encode(mask):
        encoded=masks.encode(np.asfortranarray(mask.astype(np.uint8)))
        return dict(size=[int(v) for v in encoded['size']],counts=encoded['counts'].decode('ascii'))
    check();output.mkdir()
    all_predictions={name:[] for name in n20_cpu_policy_names()};image_proofs=[];used=set();member_count=0
    for row in gate['scope_rows']:
        check();index=row['dataset_index'];origin=gate['legacy_rows'][index];carrier=row['input_carrier']
        require(row['bundle']['path'] not in used,'Each new physical bundle is consumed once within this CPU scope')
        old_bundle=torch.load(str(closed(c,carrier['bundle'])),map_location='cpu')
        shared=validator.validate_actual_shared(torch,np,cv2,old_bundle,origin)
        bundle=torch.load(str(closed(c,row['bundle'])),map_location='cpu');used.add(row['bundle']['path'])
        native=gate['populations']['native_official'][row['image_id']]
        fixed=gate['populations']['strong_fixed_E24_Omega'][row['image_id']]
        require(native==gate['scope_expected_prediction_groups']['native_official'][row['image_id']]
            and fixed==gate['scope_expected_prediction_groups']['strong_fixed_E24_Omega'][row['image_id']],
            'Reuse only exact original qualified native/R predictions; no new baseline/GT/AP')
        decoder=adapter.decode_image(torch,np,shared,bundle,row,gate['parent_GPU_binding'],carrier,native);details=[]
        for q in range(100):
            check();kwargs=decoder.query(q)
            proof=core.proof_query(torch,np,cv2,helper,legacy,endpoint,heads=models,states=states,
                formulas=formulas,rle_encoder=encode,check_callback=check,**kwargs)
            require(proof['query_id']==q and all(proof[key] is True for key in ('all_selected_indices_exact','all_full_masks_and_RLE_exact',
                'actual_true_F9_or_center_inputs_byte_exact','actual_saved_FP64_scores_canonical_exact'))
                and proof['CPU_scout_outputs_used_for_ranking'] is False and proof['CPU_feature_rerender_equivalence_proven'] is False,
                'Actual natural-query F9/two-formula/saved-score/selection/PR/mask/RLE proof completed')
            details.append(n20_json_query_proof(proof));del proof,kwargs
        for name in n20_cpu_policy_names()[2:]:
            values=decoder.predictions(name)
            require(values==gate['scope_expected_prediction_groups'][name][row['image_id']],
                    'Exact actual GPU masks/RLE/classes/S0/query export order: '+name)
            all_predictions[name].extend(values)
        all_predictions['native_official'].extend(native);all_predictions['strong_fixed_E24_Omega'].extend(fixed)
        consumed=decoder.require_all_consumed();member_count+=consumed['members']
        target=output/('IMAGE_%03d_PROOF.json'%index)
        write(target,dict(dataset_index=index,image_id=row['image_id'],queries=100,bundle=row['bundle'],
            input_carrier=carrier,consumption=consumed,details=details));image_proofs.append(reference(target))
        del old_bundle,shared,bundle,decoder,details,native,fixed,consumed,values;gc.collect();check()
    require(used=={ref['path'] for ref in gate['scope_capture_refs']} and len(image_proofs)==last-first
        and all(len(values)==(last-first)*100 for values in all_predictions.values()),
        'Complete owned child scope/natural100/all8profiles/registered members consumed')
    outputs={}
    for name,values in all_predictions.items():
        require(values==gate['scope_expected_predictions'][name], 'Whole local original prediction sequence exact: '+name)
        target=output/(name+'_PREDICTIONS.json');write(target,values);outputs[name]=dict(predictions=reference(target))
    for key,head in [('PR',strong)]+list(models.items()):
        require(not head.training and all(not p.requires_grad and p.grad is None for p in head.parameters())
            and all(gpu.n20_same_bits(torch,value.detach().cpu(),frozen[key][name]) for name,value in head.state_dict().items()),
            'Original frozen PR and all four recovered N20 five-tensor trained states unchanged')
    require(torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized(), 'External CPU RNG/CUDAcold unchanged')
    del head,strong,models,states,formulas;gc.collect()
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source']
        and reference(spec['CPU_admission_file'])==gate['admission_ref']
        and gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)==libraries,
        'Final complete Source/currentCPU entry/spec/admission/actual libraries exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        GPU_shard=spec['GPU_shard'],CPU_parent_GPU_id=spec['CPU_parent_GPU_id'],
        capture_manifest=gate['parent_GPU_report']['capture_manifest'],parent_GPU_binding=gate['parent_GPU_binding'],
        N20_root_training_qualified_result=spec['N20_root_training_qualified_result'],N20_locked_protocol=spec['N20_locked_protocol'],
        legacy_qualified_result=spec['legacy_qualified_result'],CPU_execution_partition_supplement=spec['CPU_execution_partition_supplement'],
        checkpoints=gate['trained_checkpoints'],CPU_adapter_source=spec['CPU_adapter_source'],
        selector_cpu_proof_source=spec['selector_cpu_proof_source'],legacy_selector_proof_source=spec['legacy_selector_proof_source'],
        endpoint_body_source=spec['endpoint_body_source'],PR_formula_source=spec['PR_formula_source'],labels_source=spec['labels_source'],
        scout_module_source=spec['scout_module_source'],GPU_outer_source=spec['GPU_outer_source'])
    write(output/'REPORT.json',dict(schema=N20_CPU_REPORT_SCHEMA,status=N20_CPU_STATUS,**binding,
        cpu_shard_id=spec['cpu_shard_id'],cpu_dataset_ordinal_range=[first,last],cpu_shard_owner_index=spec['normal_RAW_count'],
        normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        scope_dataset_indices=list(range(first,last)),scope_image_ids=gate['scope_image_ids'],images=last-first,queries=(last-first)*100,
        global_images=187,global_queries=18700,global_GT=380,controls=n20_cpu_policy_names(),control_results=outputs,
        image_proofs=image_proofs,capture_refs=gate['scope_capture_refs'],actual_capture_files_consumed=len(used),
        registered_bundle_members_consumed=member_count,parent_original_DEV_GPU0_CUDA0_bound=True,
        all_actual_GPU_scout_logits_two_CPU_formulas_absolute1e4=True,all_actual_true_F9_or_center_inputs_byte_exact=True,
        all_actual_saved_FP64_scores_canonical_exact=True,all_selected_indices_masks_RLE_exact=True,
        all_PR_two_formulas_absolute1e4_strictsign=True,all_native_R_and_new_score_class_query_order_exact=True,
        independent_CPU_proof_complete=True,qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,
        original_training_GPU_actual0_verified=False,training_artifacts_recovery_qualified=True,
        fixed_main_epoch=24,fixed_curve_epoch=8,DEV_winner_selected=False,CPU_scout_forward_used_for_ranking=False,
        CPU_F_rerendered=False,teacher_forward_calls=0,full_native_teacher_model_loaded=False,
        AP_evaluated=False,official_COCO_called=False,failed_partials_reused=False,CUDA_initialized=False,numeric_atol=1e-4,numeric_rtol=0.,
        actual_imported_files=libraries,peak_observed_process_RSS_bytes=peak_RSS,minimum_observed_available_RAM_bytes=minimum_available_RAM,
        elapsed_seconds=time.monotonic()-started,evidence_limits='One exact prelocked CPU child scope; combine all8 after originalactualCPU0 and Source checks for unified187COCO; not originalGPUtraining0 or CPU rerender equivalence'))
    print(json.dumps(dict(status=N20_CPU_STATUS,report=reference(output/'REPORT.json'),qualified=False,local_COCO_called=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'Single actual ROOT owner required');run(args.spec)
