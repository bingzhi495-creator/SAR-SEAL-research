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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 24 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 24 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Original failed9 and typed21/22/23/24 stay separate; normal+24 terminal identities')
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
    require(spec['schema'] == (N20_TRAIN_CPU_SCHEMA if cpu else N20_DEV_SCHEMA) and
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


"""UNBOUND N20 saved-artifact-recovery DEV FRONT; no typed prefix, tail or CLI.

ROOT replaces only inherited_source_gate's current GPU schema with N20_DEV_SCHEMA
and current ninth counts with24/+24. Historical schemas/counts remain intact.
Helpers22/23/24 must precede this front. No Torch or IO occurs on import.
First DEV owner is141+sid. Only recovery CPU140 is a completed training-related
normal owner. The preserved failed GPU140 is an additional failure, not a
completed normal GPU owner, and its absent REPORT is never created or inferred.
Models come only from qualified_root.saved_artifacts.checkpoints. The recovery
CPU Q remains pending/unchanged below ROOT's distinct actual0-qualified envelope.

Required ordinary legacy helper: legacy_gate(spec,gate) copied from N17 Source04,
with its validate_legacy_COCO/OLD_CHECKS/COCO_METRIC_NAMES/LEGACY_SHARD_RANGES.
It consumes existing spec.legacy_qualified_result and spec.legacy_shards and
returns legacy_root/rows/reports/manifests/populations. Do not invoke old N17
source_gate/training_gate/model_load_cpu or change the old four logical ranges.

Required ROOT-defined METADATA-ONLY hook n20_prior_DEV_carrier_gate(spec,gate)
returns the same gate after binding the actual old native_packet/shared carrier,
priorpacket_gate (original prior-report argument), N17 qualified common-PR bundles
and decoder refs. Actual field names/refs are not guessed here. No default true
hook is provided. Actual tensors/neighbors/PR still need the owned leaf's checks.
This front leaves gate.protocol as the inherited N16 protocol; N20_protocol is new.
"""

import json
from pathlib import Path

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


def source_gate(path, spec):
    require(spec['schema'] == N20_DEV_SCHEMA and type(spec['shard_id']) is int
        and 0 <= spec['shard_id'] < 4
        and spec['dataset_ordinal_range'] == list(N20_DEV_RANGES[spec['shard_id']])
        and all(type(spec[key]) is int for key in ('normal_RAW_count', 'expected_normal_RAW_count',
            'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'initialization_seed',
            'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 141 + spec['shard_id']
        and spec['total_failed_RAW_count'] == 24
        and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + 24
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'N20 recovery-qualified DEV GPU141+sid/failed24/fixed four ranges/resources')
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=False)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    n20_failed24_closed(spec, gate, failed24)
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


"""UNBOUND / NOT_EXECUTED N20 fused DEV GPU tail; no admission front or CLI.

Concatenate after ROOT's typed24 RAW front, recovery-qualified DEV front and
prior DEV metadata hook. Existing helpers: require/read/closed/bound/load/
reference/write. No Source/ref is invented here; old frozen files stay unchanged.

Additional spec source refs: capture_body_source (N20 DEV capture02),
coordinates_source, legacy_eval_body_source (N16 numerical body), labels_source;
new_eval_body_source/scout_module_source already belong to the DEV front.
capture_budget uses the six original N17 disk/RSS fields. Additional gate refs
prior_shared_validator_source/prior_common_decoder_source come from the prior
hook, not guessed spec aliases. Required gate runtime metadata:
DEV_packet_module_refs={n10,reader,renderer}, DEV_packet_gate (original prior
report), DEV_annotation_ref, DEV_packet_rows/DEV_packet_refs[187],
prior_common_rows[187] items={row,binding}; legacy_rows/populations/training/
trained_checkpoints/context/evaluation/common/lineage/parent and own ref fields.

Per-image: read old N16 bundle once, read old N17 common bundle once, capture
extra/map into an in-memory member, score, then save/readback/hash one physical
bundle. No teacher forward or new full-C PR; selected PR is freshly executed.
The new reader must understand F9/FP64-score schemas; old extra8 decoder is used
only to authenticate/read the PRIOR common PR, never the new N20 bundle.
"""

import gc
import json
import math
import os
from pathlib import Path
import time

N20_BUNDLE_SCHEMA = 'N20-actual-GPU-feature-neighbors-selector-image-bundle-v1'
N20_POINT_SCHEMA = 'N20-actual-GPU-feature-neighbors-selector-point-chunk-v1'
N20_PROFILE_SCHEMA = 'N20-actual-GPU-feature-neighbors-selector-query-profile-v1'
N20_SCORE_SCHEMA = 'N20-actual-GPU-logits-canonical-FP64-route-scores-v1'
N20_MANIFEST_SCHEMA = 'N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
N20_REPORT_SCHEMA = 'N20-fixedDEV-feature-error-GPU-shard-report-v1'
N20_STATUS = 'N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
N20_ARMS = ('LOCAL_ERROR', 'CENTER_ERROR')
N20_EPOCHS = (24, 8)
N20_MODES = ('atmost', 'exact')
N20_SHAPES = {'spatial_weight': (9,256), 'hidden.weight': (32,513),
              'hidden.bias': (32,), 'output.weight': (3,32), 'output.bias': (3,)}
N20_RANGES = ((0,47), (47,94), (94,141), (141,187))
N20_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def n20_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch,arm,mode) for epoch in N20_EPOCHS for arm in N20_ARMS for mode in N20_MODES)


def n20_tensor_bytes(torch, value):
    if isinstance(value, torch.Tensor):
        require(value.device.type=='cpu' and value.layout==torch.strided and not value.requires_grad
            and value.dtype in (torch.float32,torch.float64,torch.int64,torch.uint8,torch.bool)
            and bool(torch.isfinite(value).all()), 'Finite detached CPU bundle tensor')
        return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(n20_tensor_bytes(torch,v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(n20_tensor_bytes(torch,v) for v in value)
    return 0


def n20_same_bits(torch, left, right):
    return left.dtype==right.dtype and left.shape==right.shape and torch.equal(
        left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


def n20_tree_equal(torch, left, right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and n20_same_bits(torch,left,right)
    if isinstance(left,dict):return isinstance(right,dict) and list(left)==list(right) and all(n20_tree_equal(torch,left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return type(left) is type(right) and len(left)==len(right) and all(n20_tree_equal(torch,a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right


def n20_member_descriptor(torch, value):
    return dict(schema=value['schema'],query_id=value.get('query_id'),stage=value.get('stage','profile'),
        scout_epoch=value.get('scout_epoch',value.get('epoch')),control=value.get('control',value.get('arm')),
        evalmode=value.get('evalmode',value.get('mode')),chunk_first=value.get('chunk_first'),
        raw_tensor_payload_bytes=n20_tensor_bytes(torch,value))


def n20_point_member_id(index,value):
    q,first=value['query_id'],value['chunk_first'];stage=value['stage'];epoch=value['scout_epoch'];arm=value['control'];mode=value['evalmode']
    require(type(q) is int and 0<=q<100 and type(first) is int and first>=0 and first%2048==0,'Actual point role/query/chunk')
    if stage=='full_C_PR':
        require(epoch is None and arm=='COMMON_C_REFERENCE' and mode=='shared','Original common-C role');role='commonC'
    else:
        require(epoch in N20_EPOCHS and arm in N20_ARMS and ((stage=='scout_C_logits' and mode=='shared-atmost-exact')
            or (stage=='selected_PR' and mode in N20_MODES)),'Fixed N20 scout/selected role')
        role='e%d_%s_%s_%s'%(epoch,arm,mode,stage)
    return 'i%03d/q%03d/%s_c%06d.pth'%(index,q,role,first)


def n20_capture_event_cpu(torch,np,event,shared,combined_CPU,mapping,input_member,common_origin):
    require(event['actual_device_type']=='cuda' and event['synthetic_only'] is False,'Actual CUDA callback only')
    value={k:(v.detach().cpu().clone().contiguous() if isinstance(v,torch.Tensor) else v) for k,v in event.items()}
    flat,q=value['flat_indices'],value['query_id'];union=shared['union_flat_indices'].numpy()
    require(flat.dtype==torch.int64 and flat.ndim==1 and 0<len(flat)<=2048,'Canonical nonempty callback block')
    positions=np.searchsorted(union,flat.numpy())
    require(bool(np.all(positions<len(union))) and np.array_equal(union[positions],flat.numpy()),'Actual callback union coverage')
    positions_tensor=torch.from_numpy(positions.astype(np.int64))
    expected_F=shared['F_unique'][positions_tensor]
    require(n20_same_bits(torch,value['sampled_F'],expected_F) and n20_same_bits(torch,value['e0'],shared['e0'][q])
        and n20_same_bits(torch,value['native_L'],shared['native_L'][q].reshape(-1)[flat]),'Actual F/e/L must equal old input bytes')
    if value['stage']=='scout_C_logits':
        expected=(combined_CPU[mapping[positions_tensor]] if value['control']=='LOCAL_ERROR'
                  else expected_F[:,None,:].repeat(1,9,1))
        require(value['neighbor_center_column']==4 and torch.equal(value['neighbor_union_positions'],positions_tensor)
            and n20_same_bits(torch,value['neighbors_F'],expected)
            and n20_same_bits(torch,value['actual_GPU_logits'],value['scout_logits']), 'Actual nine-F inputs/logit alias byteexact before deduplication')
        for key in ('neighbors_F','neighbor_union_positions','neighbor_center_column','scout_logits'):del value[key]
        value.update(neighbor_input_member=input_member,actual_observed_neighbors_F_bitexact_to_input_carrier=True,
                     actual_GPU_logits_alias_byte_exact=True)
    elif value['stage']=='full_C_PR':
        require(value['reused_qualified_common_PR'] is True and value['fresh_PR_forward_executed'] is False
            and value['result_origin']=='qualified_prior_GPU_common_C_capture','Common is original qualified reference, never new execution')
        origin=common_origin[(q,value['chunk_first'])]
        value.update(original_common_bundle=origin['bundle'],original_common_member=origin['member'],
                     original_common_GPU_binding=origin['binding'])
    else:require(value['stage']=='selected_PR','Fresh PR only on the new selected support')
    for key in ('sampled_F','e0','native_L'):del value[key]
    value.update(schema=N20_POINT_SCHEMA,actual_observed_F_e_L_bitexact_to_input_carrier=True)
    return value


class N20ImageWriter:
    def __init__(self,torch,output,spec,binding,check):
        self.torch,self.output,self.spec,self.binding,self.check=torch,Path(output),spec,binding,check
        self.refs=[];self.raw=0;self.disk=0
    def begin(self,item,common):
        self.item=item;self.members={};self.registry={};self.image_raw=0;row=item['row']
        self.input_carrier=dict(bundle=row['bundle'],shared_member=row['shared_actual_image_inputs'],original_GPU_binding=item['binding'],
            common_PR_bundle=common['row']['bundle'],common_PR_original_GPU_binding=common['binding'])
    def put(self,name,value):
        self.check();require(type(name) is str and len(name.encode('utf8'))<=256 and name not in self.members,'Unique bounded internal member ID')
        size=n20_tensor_bytes(self.torch,value);b=self.spec['capture_budget']
        require(self.image_raw+size<=b['maximum_perimage_raw_capture_bytes'] and self.raw+self.image_raw+size<=b['maximum_total_raw_capture_payload_bytes'],'Actual raw tensor cap')
        self.members[name]=value;self.registry[name]=n20_member_descriptor(self.torch,value);self.image_raw+=size
        return name
    def finish(self):
        index=self.item['row']['dataset_index'];target=self.output/('IMAGE_%03d_ACTUAL_GPU_BUNDLE.pth'%index);self.check()
        payload=dict(schema=N20_BUNDLE_SCHEMA,binding=self.binding,input_carrier=self.input_carrier,dataset_index=index,
            image_id=self.item['row']['image_id'],members=self.members,member_registry=self.registry,
            member_count=len(self.members),raw_tensor_payload_bytes=self.image_raw)
        with target.open('xb') as stream:self.torch.save(payload,stream)
        replay=self.torch.load(str(target),map_location='cpu');require(n20_tree_equal(self.torch,payload,replay),'Exact physical bundle readback')
        del replay
        ref=reference(target);require(self.disk+ref['bytes']<=self.spec['capture_budget']['maximum_serialized_capture_bytes'],'Actual serialized capture cap')
        self.refs.append(ref);self.raw+=self.image_raw;self.disk+=ref['bytes'];members=list(self.members)
        self.members={};self.registry={};return ref,members


def n20_capture_budget_gate(spec,gate):
    b=spec['capture_budget'];keys=('maximum_process_RSS_bytes','maximum_perimage_raw_capture_bytes','maximum_total_raw_capture_payload_bytes',
        'maximum_serialized_capture_bytes','minimum_free_output_bytes','auxiliary_output_budget_bytes')
    require(type(b) is dict and all(type(b[k]) is int and b[k]>0 for k in keys)
        and b['maximum_process_RSS_bytes']<=16*1024**3 and b['maximum_perimage_raw_capture_bytes']<=1024**3
        and b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']<=72*1024**3
        and b['minimum_free_output_bytes']>=b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']+16*1024**3,'Concrete unchanged disk/RAM admission')
    first,last=N20_RANGES[spec['shard_id']];rows=[];total=0
    for item in gate['legacy_rows'][first:last]:
        row=item['row'];N=math.prod(row['identity']['original_shape']);cost=row['shared_capture_cost'];U,C,K=cost['U'],cost['sumC'],cost['sumK']
        require(all(type(v) is int for v in (N,U,C,K)) and 0<=U<=N and C>=K>=0,'Qualified native geometry/C/K/union metadata')
        E_upper=min(8*U,N-U)
        per=1032*E_upper+72*U+160*C+192*K+800*((N+7)//8)+16*1024**2
        require(per<=b['maximum_perimage_raw_capture_bytes'],'F9 difference/FP64 score/profile raw upper bound')
        rows.append(dict(dataset_index=row['dataset_index'],N=N,U=U,C=C,K=K,extra_count_upper=E_upper,raw_and_metadata_upper_bytes=per));total+=per
    require(total<=b['maximum_total_raw_capture_payload_bytes'] and total<=b['maximum_serialized_capture_bytes'],'Full fixed-shard conservative bound before any scout/PR')
    return dict(rows=rows,total_upper_bytes=total,extra_count_bound_is_conservative=True,
                formula='1032Eupper+72U+160C+192K+800ceil(N/8)+16MiB/image')


def n20_model_load_cpu(torch,spec,gate):
    require(not torch.cuda.is_initialized(),'Cold CUDA through all original PR/checkpoint loads')
    outside=torch.get_rng_state().clone();heads,states,proofs=gate['evaluation'].load_heads(torch,gate['context'],gate)
    strong=heads['LOG_PR'];pr=states['LOG_PR'];require(len(pr)==8 and sum(v.numel() for v in pr.values())==40497 and not tuple(strong.buffers()),'Frozen original LOG E24 PR40497 only')
    strong.eval().requires_grad_(False);frozen={'PR':{k:v.detach().cpu().clone() for k,v in pr.items()}};del heads,states,pr
    scout=load(spec['scout_module_source'],'_N20_DEV_feature_adapter');models={};prefix={}
    original_binding=gate['training']['original_binding']
    for epoch in (8,24):
        cp=torch.load(str(closed(gate['closure'],gate['trained_checkpoints'][str(epoch)])),map_location='cpu')
        require(cp['schema']=='N20-paired-local-center-feature-error-training-CPU-snapshot-v1' and cp['completed_epoch']==epoch
            and cp['parameters_per_arm']==18851 and cp['initialization_seed']==202610150201 and set(cp['models'])==set(N20_ARMS)
            and cp['metadata']==dict(original_binding,endpoint_epoch=epoch),'Actual recovery-qualified original top-level checkpoint binding')
        prefix[epoch]=(cp['history'],cp['epoch_orders'])
        for arm in N20_ARMS:
            state=cp['models'][arm];require(set(state)==set(N20_SHAPES) and all(v.device.type=='cpu' and v.dtype==torch.float32
                and tuple(v.shape)==N20_SHAPES[k] and bool(torch.isfinite(v).all()) for k,v in state.items()),'Five finite trained N20 tensors, no zero/divergence restriction')
            head=scout.create_head(torch);head.load_state_dict(state,strict=True);head.eval().requires_grad_(False)
            models[(epoch,arm)]=head;frozen[(epoch,arm)]={k:v.detach().clone() for k,v in state.items()}
        del cp,state,head
    require(prefix[24][0][:8]==prefix[8][0] and prefix[24][1][:8]==prefix[8][1] and torch.equal(outside,torch.get_rng_state())
        and not torch.cuda.is_initialized(),'Both fixed trajectory endpoints and private CPU construction')
    return strong,models,frozen,proofs


def run(spec_path):
    started=time.monotonic();require(Path(spec_path).is_absolute(),'Absolute N20 DEV shard specification');path=Path(spec_path).resolve(strict=True)
    spec=read(path);gate=source_gate(path,spec);c=gate['closure'];budget=n20_capture_budget_gate(spec,gate)
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(),'Fresh shard directory with existing parent')
    import psutil
    process=psutil.Process();require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and all(os.environ.get(k)=='2' for k in N20_THREAD_ENV) and psutil.virtual_memory().available>=16*1024**3
        and psutil.disk_usage(str(output.parent)).free>=spec['capture_budget']['minimum_free_output_bytes'],'Actual admitted CPU01/GPU0/thread/disk/RAM')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and not torch.cuda.is_initialized(),'Closed cold runtime')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    sources=('capture_body_source','coordinates_source','legacy_eval_body_source','labels_source','new_eval_body_source')
    for key in sources:closed(c,spec[key])
    for key in ('prior_shared_validator_source','prior_common_decoder_source'):closed(c,gate[key])
    refs=gate['DEV_packet_module_refs'];require(type(refs) is dict and set(refs)=={'n10','reader','renderer'},'Explicit original native packet/renderer refs')
    for ref in refs.values():closed(c,ref)
    closed(c,gate['DEV_annotation_ref'])
    require(len(gate['DEV_packet_rows'])==len(gate['DEV_packet_refs'])==len(gate['legacy_rows'])==len(gate['prior_common_rows'])==187,'Full187 original/prior carrier records')
    capture_body=load(spec['capture_body_source'],'_N20_DEV_extra_capture');coordinates=load(spec['coordinates_source'],'_N20_DEV_coordinates')
    body=load(spec['new_eval_body_source'],'_N20_DEV_numerical');legacy=load(spec['legacy_eval_body_source'],'_N16_original_numerical')
    helper=load(spec['labels_source'],'_N16_FP64_selection');validator=load(gate['prior_shared_validator_source'],'_N17_actual_shared_validator')
    decoder=load(gate['prior_common_decoder_source'],'_N17_prior_common_decoder')
    packet_modules={k:load(ref,'_N20_original_'+k) for k,ref in refs.items()};coco=COCO(str(closed(c,gate['DEV_annotation_ref'])))
    require(tuple(body.ARMS)==N20_ARMS and tuple(body.EPOCHS)==N20_EPOCHS and tuple(body.MODES)==N20_MODES
        and body.STATE_SHAPES==N20_SHAPES and body.POINT_CHUNK==2048 and body.ABSOLUTE_TOLERANCE==1e-4,'Exact fixed N20 evalbody ABI')
    strong,models,frozen,model_proofs=n20_model_load_cpu(torch,spec,gate)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    require(not torch.cuda.is_initialized(),'All closed modules and recovered models precede CUDA')
    torch.cuda.set_device(0);torch.cuda.init();require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Only actual CUDA0')
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0);torch.cuda.reset_peak_memory_stats(0)
    outside=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone());strong.to('cuda:0')
    for head in models.values():head.to('cuda:0')
    del head
    memory_observations=[]
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<spec['capture_budget']['maximum_process_RSS_bytes']
            and psutil.virtual_memory().available>=16*1024**3 and process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
            and all(os.environ.get(k)=='2' for k in N20_THREAD_ENV) and torch.get_num_threads()==torch.get_num_interop_threads()==2
            and cv2.getNumThreads()==2 and torch.cuda.current_device()==0 and torch.cuda.memory_reserved(0)<=4096*1024**2,'Bounded single leaf/resource/deadline; no retry')
    binding=dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        N20_root_training_qualified_result=spec['N20_root_training_qualified_result'],N20_locked_protocol=spec['N20_locked_protocol'],
        legacy_qualified_result=spec['legacy_qualified_result'],new_eval_body_source=spec['new_eval_body_source'],capture_body_source=spec['capture_body_source'],
        coordinates_source=spec['coordinates_source'],checkpoints=gate['trained_checkpoints'],shard_id=spec['shard_id'],dataset_ordinal_range=spec['dataset_ordinal_range'])
    output.mkdir();(output/'CAPTURES').mkdir();writer=N20ImageWriter(torch,output/'CAPTURES',spec,binding,check)
    predictions={name:[] for name in n20_policy_names()};rows=[];costs=[];auxiliary=0
    def auxiliary_write(target,value):
        nonlocal auxiliary
        write(target,value);auxiliary+=target.stat().st_size;require(auxiliary<=spec['capture_budget']['auxiliary_output_budget_bytes'],'Actual auxiliary output cap')
    def encode(mask):
        encoded=masks.encode(np.asfortranarray(mask.astype(np.uint8)));return dict(size=[int(v) for v in encoded['size']],counts=encoded['counts'].decode('ascii'))
    auxiliary_write(output/'START.json',dict(schema='N20-fixedDEV-feature-GPU-shard-start-v1',**binding,model_proofs=model_proofs,
        capture_budget=budget,policies=n20_policy_names(),original_GPU_training_actual0=False,recovery_CPU_qualified=True,
        main_epoch=24,curve_epoch=8,native_model_loaded=False,AP_evaluated=False))
    first,last=N20_RANGES[spec['shard_id']]
    # Reuse the exact original CV2-domain facade from the same known validator source.
    engine=validator.old_candidate_engine(np,cv2)
    for item in gate['legacy_rows'][first:last]:
        check();row=item['row'];index=row['dataset_index'];tick=time.monotonic();common_item=gate['prior_common_rows'][index]
        require(common_item['row']['dataset_index']==index and common_item['row']['image_id']==row['image_id']
            and common_item['row']['identity']==row['identity'] and common_item['row']['qids']==row['qids'],'N16/N17 same whole-image/common carrier')
        bundle=torch.load(str(closed(c,row['bundle'])),map_location='cpu');shared=validator.validate_actual_shared(torch,np,cv2,bundle,item)
        native_ref=gate['DEV_packet_refs'][index];require(native_ref==row['native_packet'],'Original packet ref must equal the qualified shared carrier')
        packet=torch.load(str(closed(c,native_ref)),map_location='cpu')
        extra_row,extra_payload=capture_body.capture_image(torch,np,packet_modules,packet,row,index,native_ref,shared,
            gate['DEV_packet_gate'],coordinates,coco=coco,packet_row=gate['DEV_packet_rows'][index],check=check)
        del packet
        writer.begin(item,common_item);input_member=writer.put('i%03d/FEATURE_NEIGHBORS_INPUTS.pth'%index,extra_payload)
        baseline=gate['populations']['native_official'][row['image_id']];strong_predictions=gate['populations']['strong_fixed_E24_Omega'][row['image_id']]
        old_common=torch.load(str(closed(c,common_item['row']['bundle'])),map_location='cpu')
        prior=decoder.decode_image(torch,np,shared,old_common,common_item['row'],common_item['binding'],common_item['row']['input_carrier'],baseline)
        require(common_item['row']['input_carrier']['bundle']==row['bundle'] and common_item['row']['input_carrier']['shared_member']==row['shared_actual_image_inputs'],
            'Old common PR and new scouts use identical original shared carrier')
        L=shared['native_L'].to('cuda:0');e=shared['e0'].to('cuda:0');F=shared['F_unique'].to('cuda:0');extra_GPU=extra_payload['extra_F'].to('cuda:0')
        require(all(n20_same_bits(torch,a.cpu(),b) for a,b in ((L,shared['native_L']),(e,shared['e0']),(F,shared['F_unique']),(extra_GPU,extra_payload['extra_F']))),'Actual old/new features survive one GPU transfer byteexact')
        neighbor_inputs=dict(extra_F=extra_GPU,extra_flat_indices=extra_payload['extra_flat_indices'],neighbor_to_combined=extra_payload['neighbor_to_combined'])
        combined_CPU=torch.cat((shared['F_unique'],extra_payload['extra_F']),0);common_origin={};refs_by_role={};query_members=[]
        current={name:{} for name in n20_policy_names()[2:]};by_q={v['query_id']:v for v in baseline}
        def common_provider(q,active,K):
            query=prior.query(q);require(query['K']==K and np.array_equal(query['C_flat_indices'].numpy(),shared['C_indices'][q].numpy()),'Prior common query/C/K unchanged')
            profile_id=common_item['row']['query_profiles'][q]['profiles']['E24_LOCAL_ERROR_atmost']
            names=old_common['members'][profile_id]['common_C_PR_chunks']
            for i,name in enumerate(names):common_origin[(q,i*2048)]=dict(bundle=common_item['row']['bundle'],member=name,binding=common_item['binding'])
            return query['common_PR_chunks']
        def capture(event):
            value=n20_capture_event_cpu(torch,np,event,shared,combined_CPU,extra_payload['neighbor_to_combined'],input_member,common_origin)
            name=writer.put(n20_point_member_id(index,value),value)
            refs_by_role.setdefault((value['query_id'],value['scout_epoch'],value['control'],value['evalmode'],value['stage']),[]).append(name)
        for result in body.evaluate_all100(torch,np,engine,helper,legacy,L,e,shared['union_flat_indices'],F,neighbor_inputs,models,strong,
                point_callback=capture,check_callback=check,common_PR_provider=common_provider,synthetic_only=False):
            q=result['query_id'];profiles={};scores_members={}
            for name,profile in result['profiles'].items():
                epoch,arm,mode=profile['epoch'],profile['arm'],profile['mode'];scout_chunks=refs_by_role.get((q,epoch,arm,'shared-atmost-exact','scout_C_logits'),[])
                model_key=(epoch,arm)
                if model_key not in scores_members:
                    score_value=dict(schema=N20_SCORE_SCHEMA,query_id=q,epoch=epoch,arm=arm,K=result['K'],C_count=len(result['C_flat_indices']),
                        flat_indices=torch.from_numpy((result['C_flat_indices'] if result['K'] else np.empty(0,dtype=np.int64)).copy()),
                        C_scout_chunks=scout_chunks,canonical_CPU_FP64_C_scores=torch.from_numpy(profile['canonical_CPU_FP64_C_scores'].copy()),actual_GPU_logits_used=bool(result['K']))
                    scores_members[model_key]=writer.put('i%03d/q%03d/e%d_%s_SCORES.pth'%(index,q,epoch,arm),score_value)
                encoded=encode(profile['mask']);prediction=dict(by_q[q],segmentation=encoded);current[name][q]=prediction
                value=dict(schema=N20_PROFILE_SCHEMA,query_id=q,epoch=epoch,arm=arm,mode=mode,K=result['K'],C_scout_chunks=scout_chunks,
                    common_C_PR_chunks=refs_by_role.get((q,None,'COMMON_C_REFERENCE','shared','full_C_PR'),[]),
                    selected_PR_chunks=refs_by_role.get((q,epoch,arm,mode,'selected_PR'),[]),
                    route_scores_member=scores_members[model_key],neighbor_input_member=input_member,
                    actual_selected_flat_indices=torch.from_numpy(profile['selected_flat_indices'].copy()),
                    actual_mask_packbits=torch.from_numpy(np.packbits(profile['mask'].ravel(),bitorder='little')),actual_mask_shape=list(profile['mask'].shape),actual_mask_bitorder='little',
                    actual_RLE=encoded,original_prediction_nonsegmentation_fields={k:v for k,v in by_q[q].items() if k!='segmentation'},PR_compatibility=profile['PR_compatibility'])
                profiles[name]=writer.put('i%03d/q%03d/e%d_%s_%s_PROFILE.pth'%(index,q,epoch,arm,mode),value)
            query_members.append(dict(query_id=q,profiles=profiles,K=result['K'],C_count=len(result['C_flat_indices'])))
            del result,profile,profiles,scores_members
        require(len(query_members)==100 and all(set(v)==set(range(100)) for v in current.values()),'Full natural100/8profiles actual outputs')
        predictions['native_official'].extend(baseline);predictions['strong_fixed_E24_Omega'].extend(strong_predictions)
        for name,values in current.items():
            ordered=[values[q] for q in row['qids']]
            require(all({k:v for k,v in x.items() if k!='segmentation'}=={k:v for k,v in y.items() if k!='segmentation'} for x,y in zip(ordered,baseline)),'Original class/S0/query/export ordering unchanged')
            predictions[name].extend(ordered)
        ref,members=writer.finish();rows.append(dict(dataset_index=index,image_id=row['image_id'],identity=row['identity'],qids=row['qids'],
            input_carrier=writer.input_carrier,bundle=ref,member_locators=members,query_profiles=query_members,natural_queries=100,neighbor_input_member=input_member))
        costs.append(dict(dataset_index=index,inclusive_seconds=time.monotonic()-tick,feature_capture=extra_row,standalone_FPS_measured=False))
        memory_observations.append(dict(dataset_index=index,allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0)))
        del bundle,shared,old_common,prior,extra_payload,neighbor_inputs,combined_CPU,L,e,F,extra_GPU,current,refs_by_role,common_origin,query_members
        gc.collect();check()
    require(all(len(v)==(last-first)*100 for v in predictions.values()),'Complete fixed shard; no local AP')
    for key,head in [('PR',strong)]+list(models.items()):
        require(all(n20_same_bits(torch,v.detach().cpu(),frozen[key][k]) for k,v in head.state_dict().items())
            and all(not p.requires_grad and p.grad is None for p in head.parameters()),'Frozen PR/five-tensor scouts unchanged')
    require(torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),'Evaluation external RNG unchanged')
    peak=torch.cuda.max_memory_allocated(0);before=dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0));strong.cpu()
    for head in models.values():head.cpu()
    models.clear();del head,strong,models;gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup==dict(allocated_after=0,reserved_after=0) and peak<=4096*1024**2,'Actual complete GPU cleanup including cuBLAS workspaces')
    outputs={}
    for name,values in predictions.items():
        target=output/(name+'_PREDICTIONS.json');auxiliary_write(target,values);outputs[name]=dict(predictions=reference(target))
    auxiliary_write(output/'CAPTURE_MANIFEST.json',dict(schema=N20_MANIFEST_SCHEMA,**binding,images=last-first,queries=(last-first)*100,
        global_images=187,global_queries=18700,rows=rows,physical_capture_refs=writer.refs,internal_member_locators_are_strings=True,
        raw_tensor_payload_bytes=writer.raw,serialized_bytes=writer.disk))
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['GPU_admission_file'])==gate['admission_ref']
        and gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)==libraries,'Final Source/spec/admission/imported libraries unchanged')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    auxiliary_write(output/'REPORT.json',dict(schema=N20_REPORT_SCHEMA,status=N20_STATUS,**binding,controls=n20_policy_names(),control_results=outputs,
        scope_dataset_indices=list(range(first,last)),scope_image_ids=[r['image_id'] for r in rows],images=last-first,queries=(last-first)*100,
        capture_manifest=reference(output/'CAPTURE_MANIFEST.json'),capture_refs=writer.refs,START=reference(output/'START.json'),
        normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=spec['total_failed_RAW_count'],total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        capture_budget=budget,serialized_bytes=writer.disk,auxiliary_bytes_before_REPORT=auxiliary,image_costs=costs,memory_observations=memory_observations,
        GPU_peak_allocated_bytes=peak,pre_cleanup=before,cleanup=cleanup,models_frozen=True,qualified=False,ROOT_actual_GPU0_verified=False,
        original_training_GPU_actual0_verified=False,training_artifacts_recovery_qualified=True,CPU_numerical_qualification_pending=True,
        teacher_forward_calls=0,new_fullC_PR_forward_calls=0,fresh_selected_PR=True,reused_common_C_reference=True,
        native_model_loaded=False,AP_evaluated=False,local_AP_computed=False,full187_qualification_claimed=False,standalone_FPS_measured=False,
        shared_diagnostic_costs_nonadditive=True,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=N20_STATUS,report=reference(output/'REPORT.json'),qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'Single actual ROOT owner required');run(args.spec)
