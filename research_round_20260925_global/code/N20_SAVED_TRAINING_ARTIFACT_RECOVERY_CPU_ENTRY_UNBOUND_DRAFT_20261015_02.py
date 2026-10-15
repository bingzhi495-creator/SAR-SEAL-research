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
    require(spec['schema'] == (N20_TRAIN_CPU_SCHEMA if cpu else N20_GPU_SCHEMA) and
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
N20_TRAIN_CPU_SCHEMA = 'ROOT-N20-paired-feature-error-saved-artifact-recovery-CPU-spec-v1'
N20_TRAIN_BINDING_KEYS = (
    'source', 'spec', 'GPU_admission', 'phase', 'N20_locked_protocol', 'locked_protocol',
    'parent_cache_report', 'neighbor_report', 'scout_module_source', 'base_scout_source',
    'paired_training_body_source', 'original_paired_training_body_source',
    'initialization_seed', 'order_seed', 'expected_N20_initialization_digest',
    'frozen_teacher_checkpoint', 'frozen_PR_checkpoint', 'N20_root_cache_qualified_result',
    'N20_initialization_lock', 'N20_cache_CPU_Q')


def _n20_cpu_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def n20_completed_cache_record(record, output_key, owner_index, cuda, gate, spec):
    """Only original qualified cache GPU138/CPU139; no completed GPU140 exists."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner_index
        and (owner_index, output_key, cuda) in ((138, 'report', '0'), (139, 'Q', '')),
        'Exact eight-field original qualified cache completion record required')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner_index]
    require(all(row[key] == record[record_key] for key, record_key in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Completed cache record must equal its original RAW prefix row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        record['RAW'], record['admission'], record['source'], record['spec'], cuda)
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    admission_key = 'GPU_admission' if cuda == '0' else 'CPU_admission'
    filename = 'REPORT.json' if output_key == 'report' else 'Q.json'
    require(type(joined) is int and joined == owner_index == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner_index
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner_index]
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['total_failed_RAW_count'] == 21
        and parent['total_terminal_identity_pin_count'] == owner_index + 21
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and Path(parent[admission_key + '_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / filename).resolve(strict=True),
        'Cache completion exact original owner/source/spec/admission/pins/prefix/output')
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Cache pending output must bind the actual original owner')
    return parent, output


def source_gate(path, spec):
    require(spec['schema'] == N20_TRAIN_CPU_SCHEMA
        and all(type(spec[key]) is int for key in ('normal_RAW_count', 'expected_normal_RAW_count',
            'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'initialization_seed',
            'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 140
        and spec['total_failed_RAW_count'] == 24 and spec['total_terminal_identity_pin_count'] == 164
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['maximum_process_RSS_bytes'] == 16 * 1024**3 and spec['automatic_retry'] is False
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent recovery CPU requires own140/failed24/pins164 and original resources')
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    n20_failed24_closed(spec, gate, failed24)
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
            'N20_locked_protocol', 'cache_GPU_report', 'region_common_source', 'qualifier_body_source',
            'N20_training_tail_source', 'N20_failed_training_stderr'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_cpu_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'Exact complete prelocked N20 protocol; old N16 protocol stays separate')
    root = read(closed(closure, spec['N20_root_cache_qualified_result']))
    require(_n20_cpu_exact(root, spec['expected_N20_root_cache_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and root['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and root['qualified'] is True and root['original_GPU_CPU_actual0_verified'] is True
        and root['full_proofs_ROOT_verified'] is True
        and root['images'] == 741 and root['sampled_points'] == 3035136
        and root['unique_centers'] == 1786820 and root['parent_cache'] == spec['cache_GPU_report']
        and root['protocol'] == spec['N20_locked_protocol']
        and root['GPU_report'] == root['gpu_record']['report'] and root['CPU_Q'] == root['CPU_record']['Q'],
        'Keep exact independently qualified GPU138/CPU139 cache ROOT envelope')
    gpu_spec, cache_report = n20_completed_cache_record(root['gpu_record'], 'report', 138, '0', gate, spec)
    cpu_spec, cpu_q = n20_completed_cache_record(root['CPU_record'], 'Q', 139, '', gate, spec)
    old_report = read(closed(closure, spec['cache_GPU_report']))
    require(cache_report['schema'] == 'N20-true-feature-neighbor-cache-GPU-report-v1'
        and cache_report['status'] == 'N20_FULLFIT_TRUE_FEATURE_NEIGHBORS_COMPLETE_PENDING_INDEPENDENT_CPU'
        and cache_report['parent_cache'] == spec['cache_GPU_report']
        and cache_report['protocol'] == spec['N20_locked_protocol']
        and cpu_q['schema'] == 'N20-fullFIT-feature-neighbor-cache-CPU-qualification-v1'
        and cpu_q['status'] == 'N20_FULLFIT_FEATURE_NEIGHBOR_CACHE_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and cpu_q['capture_GPU_report'] == root['GPU_report']
        and cpu_q['protocol'] == spec['N20_locked_protocol']
        and cpu_q['qualified'] is False and cpu_q['ROOT_actual_CPU0_verified'] is False
        and cpu_q['original_owned_terminal_self_certified'] is False
        and cpu_q['images'] == 741 and cpu_q['sampled_points'] == 3035136
        and cpu_q['unique_centers'] == 1786820
        and cpu_q['extra_feature_bytes'] == cache_report['extra_feature_bytes']
        and gpu_spec['N20_locked_protocol'] == cpu_spec['N20_locked_protocol'] == spec['N20_locked_protocol'],
        'Cache pending GPU report/CPU Q stay unchanged under original ROOT envelope')
    closed(closure, cpu_q['image_proofs'])
    require(old_report['images'] == cache_report['images'] == 741
        and old_report['sampled_points'] == cache_report['sampled_points'] == 3035136
        and len(old_report['rows']) == len(old_report['packet_refs']) ==
            len(cache_report['rows']) == len(cache_report['packet_refs']) == 741,
        'Complete unchanged original/new FIT741 physical cache populations')
    for index, (oldrow, row) in enumerate(zip(old_report['rows'], cache_report['rows'])):
        require(index == oldrow['dataset_index'] == row['dataset_index']
            and oldrow['packet'] == old_report['packet_refs'][index]
            and row['packet'] == cache_report['packet_refs'][index] and row['parent_packet'] == oldrow['packet']
            and row['identity'] == oldrow['identity'] and row['image_id'] == oldrow['image_id']
            and row['points'] == oldrow['points'] == 4096
            and row['unique_centers'] == oldrow['unique_sampled_F_points'],
            'Qualified N20 packet retains original identity/sample row')
        closed(closure, row['packet'])
        closed(closure, oldrow['packet'])
    metadata = read(closed(closure, spec['N20_initialization_lock']))
    require(metadata['schema'] == 'Root-N20-source-only-private-initialization-lock-v1'
        and type(metadata['seed']) is int and metadata['seed'] == 202610150201
        and metadata['combined_digest'] == spec['expected_N20_initialization_digest']
        and all(metadata[key] is True for key in ('paired_state_exact', 'paired_storage_independent',
            'spatial_weights_zero', 'external_CPU_RNG_unchanged'))
        and metadata['CUDA_initialized'] is False and metadata['training_executed'] is False
        and metadata['torch_version'] == '2.0.1+cu118' and metadata['CPU_affinity'] == [0, 1]
        and type(metadata['threads']) is int and metadata['threads'] == 2,
        'Exact original source-only five-tensor initialization metadata')
    initialization_sources = [spec[key] for key in model_keys] + [spec['N20_training_tail_source']]
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'Preserved original initialization source basenames/hashes')

    record = spec['N20_failed_training_record']
    require(type(record) is dict and set(record) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding', 'bootstrap',
        'entry_source', 'spec', 'original_report_absent'}
        and type(record['returncode']) is int and record['returncode'] == 1
        and record['original_report_absent'] is True,
        'Only preserved actual1 training record; no normal GPU140 completion')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, record[key])
    training_parent = read(closed(closure, record['spec']))
    output = Path(training_parent['output_directory'])
    shared_keys = model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'expected_N20_root_cache_qualified_result',
        'expected_N20_initialization_digest', 'locked_protocol', 'cache_GPU_report',
        'region_common_source', 'initialization_seed', 'N20_training_tail_source',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure')
    require(training_parent['schema'] == N20_TRAIN_SCHEMA
        and training_parent['entry_source'] == record['entry_source']
        and training_parent['phase'] == record['phase']
        and training_parent['normal_RAW_count'] == training_parent['expected_normal_RAW_count'] == 140
        and training_parent['total_failed_RAW_count'] == 23
        and training_parent['total_terminal_identity_pin_count'] == 163
        and training_parent['RAW_inputs'] == spec['RAW_inputs'][:140]
        and all(closure.get(ref['path']) == ref for ref in training_parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in training_parent['terminal_identity_pins'].items())
        and all(_n20_cpu_exact(training_parent[key], spec[key]) for key in shared_keys)
        and training_parent['deadline_seconds'] == 1800 and training_parent['automatic_retry'] is False
        and training_parent['maximum_process_RSS_bytes'] == 16 * 1024**3
        and training_parent['allocator_limit_MiB'] == 4096 and training_parent['whole_GPU_limit_MiB'] == 8192
        and Path(training_parent['GPU_admission_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and output.is_absolute() and output.is_dir() and not output.is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink(),
        'Preserved failed GPU original23 prefix/source/admission/science/output and absent REPORT')

    artifacts = spec['N20_saved_training_artifacts']
    require(type(artifacts) is dict and set(artifacts) == {'START', 'checkpoints', 'FIT_endpoints', 'UPDATE_TRACE'}
        and type(artifacts['checkpoints']) is dict and type(artifacts['FIT_endpoints']) is dict
        and set(artifacts['checkpoints']) == set(artifacts['FIT_endpoints']) == {'8', '24'},
        'Six actual saved artifacts only; no absent GPU REPORT reference')
    named_refs = [
        (artifacts['START'], 'START.json'), (artifacts['UPDATE_TRACE'], 'UPDATE_TRACE.jsonl'),
        (artifacts['checkpoints']['8'], 'E08_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['checkpoints']['24'], 'E24_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['FIT_endpoints']['8'], 'E08_FIT_ENDPOINT.json'),
        (artifacts['FIT_endpoints']['24'], 'E24_FIT_ENDPOINT.json')]
    require(len({ref['path'] for ref, _ in named_refs}) == 6,
        'Saved START, trace and two snapshots/endpoints must be six distinct physical artifacts')
    for ref, filename in named_refs:
        actual = closed(closure, ref)
        require(Path(actual).resolve(strict=True) == (output / filename).resolve(strict=True),
            'Saved artifact must be its prescribed original failed-training output: ' + filename)
    binding = dict(source=record['entry_source'], spec=record['spec'],
        GPU_admission=record['admission'], phase=record['phase'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=spec['cache_GPU_report'], neighbor_report=root['GPU_report'],
        initialization_seed=202610150201, order_seed=202610140160001,
        expected_N20_initialization_digest=spec['expected_N20_initialization_digest'],
        frozen_teacher_checkpoint=old_report['teacher_checkpoint'],
        frozen_PR_checkpoint=old_report['fixed_PR_checkpoint'],
        N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
        N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q'])
    binding.update({key: spec[key] for key in model_keys})
    start = read(closed(closure, artifacts['START']))
    require(set(binding) == set(N20_TRAIN_BINDING_KEYS) and type(start) is dict
        and all(start[key] == value for key, value in binding.items())
        and start['schema'] == 'N20-paired-feature-error-training-start-v1'
        and start['parameters_per_arm'] == 18851 and start['tensors_per_arm'] == 5
        and start['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR'] and start['fixed_epochs'] == 24
        and start['updates_each_arm'] == 17784 and start['point_exposure_each_arm'] == 72843264
        and all(start[key] is False for key in ('AMP', 'DEV_AP_evaluated', 'qualified',
                                               'teacher_or_PR_loaded', 'full_native_L_loaded')),
        'Actual original START uses locked five-tensor initialization and fixed24 recipe/binding')
    require('parent' in gate and 'lineage' in gate,
        'Preserve original inherited endpoint parent for final_endpoint_check')
    common = load(spec['region_common_source'], '_N20_recovery_CPU_original_region_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original imported_files ABI required')
    gate.update(common=common, parent_cache_report=old_report,
        parent_cache_report_ref=spec['cache_GPU_report'], neighbor_report=cache_report,
        neighbor_report_ref=root['GPU_report'], protocol=protocol, initialization_metadata=metadata,
        recovery_training_parent=training_parent, recovery_failed_GPU_record=record,
        recovery_artifact_refs=artifacts, recovery_start=start, recovery_training_binding=binding,
        training_binding_keys=N20_TRAIN_BINDING_KEYS,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q']))
    return gate

# UNBOUND_DRAFT: N20 artifact-recovery CPU run tail only; no RAW/admission front or CLI.
"""Concatenate after the frozen ROOT-owned typed front and N20 CPU FRONT.
ROOT supplies require/read/closed/bound/load/reference/write/source_gate; all
module load() calls receive strict three-field refs after closed() membership.
Source/actual-terminal qualification belongs to ROOT, never this pending Q.
"""
import gc
import json
import os
from pathlib import Path
import time

N20_CPU_ARMS = ('LOCAL_ERROR', 'CENTER_ERROR')
N20_CPU_SEED = 202610150201
N20_CPU_ORDER_SEED = 202610140160001
N20_CPU_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
N20_CPU_Q_SCHEMA = 'N20-paired-local-center-feature-error-training-recovery-CPU-qualification-v1'
N20_CPU_STATUS = 'N20_PAIRED_FEATURE_ERROR_E8_E24_RECOVERY_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'


def n20_load_endpoint_CPU_image(torch, index, gate, spec, check):
    """Read two qualified sparse packets; do not open the full native-L carrier."""
    check()
    closure = gate['closure']
    old_report, report = gate['parent_cache_report'], gate['neighbor_report']
    old_row, row = old_report['rows'][index], report['rows'][index]
    old_ref, neighbor_ref = old_report['packet_refs'][index], report['packet_refs'][index]
    require(type(index) is int and index == old_row['dataset_index'] == row['dataset_index']
            and old_row['packet'] == old_ref and row['packet'] == neighbor_ref
            and row['parent_packet'] == old_ref and row['image_id'] == old_row['image_id']
            and row['identity'] == old_row['identity']
            and row['points'] == old_row['points'] == 4096
            and row['unique_centers'] == old_row['unique_sampled_F_points'],
            'Qualified sparse neighbor/old packet rows must bind one exact FIT image')
    packet = torch.load(str(closed(closure, old_ref)), map_location='cpu')
    neighbor = torch.load(str(closed(closure, neighbor_ref)), map_location='cpu')
    require(type(packet) is dict and packet['dataset_index'] == index
            and packet['identity'] == old_row['identity']
            and packet['protocol'] == spec['locked_protocol'],
            'Original qualified thin packet identity and N16 protocol must remain exact')
    for key in ('source', 'spec', 'GPU_admission', 'phase',
                'fixed_PR_checkpoint', 'teacher_checkpoint'):
        require(packet[key] == old_report[key], 'Original packet/report binding differs: ' + key)
    require(type(neighbor) is dict and neighbor['schema'] == 'N20-true-F-neighbor-capture-body-v1'
            and neighbor['dataset_index'] == index and neighbor['identity'] == packet['identity']
            and neighbor['parent_packet'] == old_ref
            and neighbor['protocol'] == spec['N20_locked_protocol']
            and neighbor['teacher_checkpoint'] == packet['teacher_checkpoint']
            and neighbor['matching_loss_seed'] == packet['matching_loss_seed'] == row['matching_loss_seed']
            and neighbor['point_seed'] == packet['samples']['seed'] == row['point_seed']
            and neighbor['original_unique_count'] == row['unique_centers']
            and neighbor['extra_count'] == row['extra_count']
            and neighbor['sampled_points'] == 4096 and neighbor['center_column'] == 4,
            'Actual qualified N20 extra-F payload must bind old sample order and protocol')
    for key in ('source', 'spec', 'GPU_admission', 'phase'):
        require(neighbor[key] == report[key], 'N20 capture packet/report binding differs: ' + key)
    sample = packet['samples']
    query, flat = sample['query_ids'], sample['flat_indices']
    H, W = map(int, packet['identity']['original_shape'])
    require(query.device.type == flat.device.type == 'cpu'
            and query.dtype == flat.dtype == torch.int64
            and tuple(query.shape) == tuple(flat.shape) == (4096,)
            and bool(((query >= 0) & (query < 100)).all())
            and bool(((flat >= 0) & (flat < H * W)).all()),
            'Exactly original4096 natural-query/point slots, including duplicates')
    # The body validates sparse tensor shapes/ranges and gathers the original
    # sample_to_unique order. All other payload metadata remains outside its ABI.
    loaded = dict(packet=packet, neighbor_packet={
        'extra_F': neighbor['extra_F'],
        'neighbor_to_combined': neighbor['neighbor_to_combined']})
    check()
    return loaded



def n20_read_actual_update_trace(path, check):
    """Only the saved paired_image_update substream, with no invented events."""
    records = []
    with path.open('r', encoding='utf8') as stream:
        for line in stream:
            check()
            require(bool(line.strip()), 'No blank line in the original update trace')
            records.append(json.loads(line))
            require(len(records) <= 17784, 'No extra original paired updates')
    require(len(records) == 17784, 'Full fixed24 original17784 paired updates required')
    return records


def n20_validate_recovery_evidence(torch, body, scout, base, start, snapshots,
                                   fit_payloads, expected_binding, cache_rows, trace, expected_digest):
    """Check saved artifacts directly; never manufacture a training GPU REPORT."""
    keys = tuple(body.BINDING_KEYS)
    require(type(expected_binding) is dict and set(expected_binding) == set(keys)
            and all(start[key] == expected_binding[key] for key in keys),
            'Actual failed-training START exact20-field binding must be front-authenticated')
    require(start['schema'] == 'N20-paired-feature-error-training-start-v1'
            and start['parameters_per_arm'] == 18851 and start['tensors_per_arm'] == 5
            and start['arms'] == list(N20_CPU_ARMS) and start['fixed_epochs'] == 24
            and start['updates_each_arm'] == 17784 and start['point_exposure_each_arm'] == 72843264
            and start['initialization_seed'] == N20_CPU_SEED
            and start['order_seed'] == N20_CPU_ORDER_SEED
            and start['expected_N20_initialization_digest'] == expected_digest
            and all(start[key] is False for key in ('AMP', 'DEV_AP_evaluated', 'qualified',
                                                    'teacher_or_PR_loaded', 'full_native_L_loaded')),
            'Original START complete fixed24/error/five-tensor initialization recipe')
    initialization = body.validate_initialization_evidence(
        torch, scout, base, start['initial_states'], expected_digest)
    states, endpoints = {}, {}
    for epoch in (8, 24):
        snapshot = snapshots[epoch]
        metadata = snapshot['metadata']
        require(type(metadata) is dict and set(metadata) == set(keys) | {'endpoint_epoch'}
                and metadata['endpoint_epoch'] == epoch
                and all(metadata[key] == start[key] for key in keys),
                'Top-level checkpoint metadata exact20 bindings plus fixed endpoint')
        for key in ('external_CPU_RNG', 'external_CUDA_RNG'):
            value = snapshot[key]
            require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                    and value.device.type == 'cpu' and value.dtype == torch.uint8
                    and value.ndim == 1 and value.numel() > 0 and not value.requires_grad,
                    'Original saved external RNG vectors; no CUDA RNG initialization')
        payload = fit_payloads[epoch]
        require(type(payload) is dict and set(payload) == set(keys) | {'endpoint'}
                and all(payload[key] == start[key] for key in keys),
                'Actual saved FIT JSON complete original binding, not a reconstructed report')
        endpoints[epoch] = payload['endpoint']
        states[str(epoch)] = body.validate_state_evidence(
            torch, scout, base, snapshot, cache_rows, epoch,
            expected_initialization_digest=expected_digest)
        body.validate_trace(trace[:epoch * 741], snapshot, cache_rows)
    body.validate_snapshot_pair(torch, snapshots[8], snapshots[24])
    return initialization, states, endpoints


def run(spec_path):
    started = time.monotonic()
    require(Path(spec_path).is_absolute(), 'Absolute independent CPU specification required')
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec)
    closure = gate['closure']
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink()
            and output.parent.is_dir(), 'Fresh owned CPU output with existing parent required')
    require(spec['deadline_seconds'] == 1800 and spec['automatic_retry'] is False
            and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
            and spec['initialization_seed'] == N20_CPU_SEED,
            'Original first-seed CPU resources and fixed recipe required')
    source_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                   'original_paired_training_body_source', 'qualifier_body_source')
    for key in source_keys:
        closed(closure, spec[key])
    for report_key in ('parent_cache_report', 'neighbor_report'):
        require(read(closed(closure, gate[report_key + '_ref'])) == gate[report_key],
                'Decoded qualified cache report equals its closed physical JSON: ' + report_key)
    artifacts = gate['recovery_artifact_refs']
    require(type(artifacts) is dict and set(artifacts) == {'START', 'checkpoints', 'FIT_endpoints', 'UPDATE_TRACE'}
            and type(artifacts['checkpoints']) is dict and type(artifacts['FIT_endpoints']) is dict
            and set(artifacts['checkpoints']) == set(artifacts['FIT_endpoints']) == {'8', '24'},
            'Six saved recovery artifacts only; no absent training GPU REPORT ref')
    for ref in [artifacts['START'], artifacts['UPDATE_TRACE']] + list(artifacts['checkpoints'].values()) + list(artifacts['FIT_endpoints'].values()):
        closed(closure, ref)
    start = read(closed(closure, artifacts['START']))
    require(start == gate['recovery_start'], 'Front-decoded original START must equal actual closed JSON')
    cache_rows = gate['parent_cache_report']['rows']
    require(len(cache_rows) == len(gate['neighbor_report']['rows']) == 741,
            'Complete qualified sparse FIT741 inputs required')

    import psutil
    process = psutil.Process()
    require(process.cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(key) == '2' for key in N20_CPU_THREAD_ENV),
            'Independent CPU01/two threads/empty CUDA actual environment')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(),
            'Original Torch with cold CUDA required after the complete front')
    import numpy as np
    import cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0',
            'Closed original numerical-library versions required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    # These modules are already closed and bound by the front. The paired bodies
    # are provenance only here: no training/optimizer step is called.
    scout = load(spec['scout_module_source'], '_N20_CPU_closed_feature_adapter')
    base = load(spec['base_scout_source'], '_N20_CPU_closed_original_scout_factory')
    paired = load(spec['paired_training_body_source'], '_N20_CPU_closed_training_body_provenance')
    original = load(spec['original_paired_training_body_source'], '_N20_CPU_closed_original_training_provenance')
    body = load(spec['qualifier_body_source'], '_N20_CPU_closed_independent_endpoint_body')
    require(tuple(paired.ARMS) == tuple(body.ARMS) == N20_CPU_ARMS
            and paired.INIT_SEED == body.INIT_SEED == N20_CPU_SEED
            and paired.ORDER_SEED == body.ORDER_SEED == N20_CPU_ORDER_SEED
            and paired.EPOCHS == 24 and tuple(paired.ENDPOINTS) == (8, 24)
            and callable(getattr(original, 'train_one_epoch', None))
            and callable(getattr(scout, 'create_head', None))
            and callable(getattr(base, 'make_scout', None))
            and body.ABS_LOGITS == body.ABS_MEAN_CE == 1e-4,
            'Exact same four model sources and independent two-formula body ABI')
    libraries = gate['common'].imported_files(
        torch, np, cv2, masks, COCO, closure, gate['admission'], ndimage)
    require(not torch.cuda.is_initialized(), 'Source-qualified imports must keep CUDA cold')
    outside_CPU = torch.get_rng_state().clone()
    peak_RSS = 0
    minimum_available_RAM = None

    def check():
        nonlocal peak_RSS, minimum_available_RAM
        rss = process.memory_info().rss
        available = psutil.virtual_memory().available
        peak_RSS = max(peak_RSS, rss)
        minimum_available_RAM = available if minimum_available_RAM is None else min(minimum_available_RAM, available)
        require(time.monotonic() - started < spec['deadline_seconds']
                and rss < spec['maximum_process_RSS_bytes'] and available >= 16 * 1024**3
                and process.cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
                and all(os.environ.get(key) == '2' for key in N20_CPU_THREAD_ENV)
                and torch.get_num_threads() == torch.get_num_interop_threads() == 2
                and cv2.getNumThreads() == 2 and not torch.cuda.is_initialized(),
                'CPU bounded resources/deadline/affinity/thread/cold-CUDA invariant')

    check()
    expected_digest = spec['expected_N20_initialization_digest']
    trace = n20_read_actual_update_trace(closed(closure, artifacts['UPDATE_TRACE']), check)
    snapshots, fit_payloads = {}, {}
    trained_digests = {}
    for epoch in (8, 24):
        check()
        # Original torch snapshot is top-level; preserve all saved bytes unchanged.
        snapshots[epoch] = torch.load(str(closed(closure, artifacts['checkpoints'][str(epoch)])), map_location='cpu')
        fit_payloads[epoch] = read(closed(closure, artifacts['FIT_endpoints'][str(epoch)]))
        check()
    initialization_checks, state_checks, actual_endpoints = n20_validate_recovery_evidence(
        torch, body, scout, base, start, snapshots, fit_payloads,
        gate['recovery_training_binding'], cache_rows, trace, expected_digest)
    for epoch in (8, 24):
        trained_digests[str(epoch)] = {
            arm: body.state_digest(torch, snapshots[epoch]['models'][arm])[0] for arm in N20_CPU_ARMS}
    # This is artifact-derived history/order, explicitly not an original GPU report.
    artifact_history = dict(source_checkpoint=artifacts['checkpoints']['24'],
        provenance='E24 top-level snapshot only; no GPU REPORT created or reconstructed',
        history=snapshots[24]['history'], epoch_orders=snapshots[24]['epoch_orders'])
    output.mkdir()
    input_trace_path = output / 'CPU_INPUT_READS.jsonl'
    endpoints = {}
    loads = 0
    with input_trace_path.open('x', encoding='utf8', newline='\n') as input_trace:
        for epoch in (8, 24):
            check()
            next_index = 0

            def observer(event):
                check()
                require(event['event'] == 'before_endpoint_CPU_image'
                        and event['epoch'] == epoch and event['dataset_index'] == next_index,
                        'CPU endpoint must visit each original image exactly once in canonical order')

            def loader(index):
                nonlocal loads, next_index
                require(type(index) is int and index == next_index and 0 <= index < 741,
                        'Canonical complete FIT741 same-input endpoint traversal')
                loaded = n20_load_endpoint_CPU_image(torch, index, gate, spec, check)
                input_trace.write(json.dumps(dict(epoch=epoch, dataset_index=index,
                    image_id=cache_rows[index]['image_id'],
                    parent_packet=cache_rows[index]['packet'],
                    neighbor_packet=gate['neighbor_report']['rows'][index]['packet']),
                    separators=(',', ':'), allow_nan=False) + '\n')
                loads += 1
                next_index += 1
                return loaded

            endpoints[str(epoch)] = body.validate_endpoint(
                torch, scout, base, snapshots[epoch], actual_endpoints[epoch],
                loader, cache_rows, epoch, observer=observer,
                expected_initialization_digest=expected_digest)
            require(next_index == 741, 'Only complete actual741-image CPU endpoint qualifies')
            require(all(body.state_digest(torch, snapshots[epoch]['models'][arm])[0] ==
                        trained_digests[str(epoch)][arm] for arm in N20_CPU_ARMS),
                    'Trained checkpoint tensors must remain unchanged during numerical reads')
            input_trace.flush()
            check()
    require(loads == 1482 and set(endpoints) == {'8', '24'},
            'Both actual fixed endpoints, each full741 images/3035136 same input slots')
    del trace, snapshots, fit_payloads
    gc.collect()
    for ref in spec['sources']:
        check()
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source']
            and reference(spec['CPU_admission_file']) == gate['admission_ref']
            and torch.equal(outside_CPU, torch.get_rng_state())
            and gate['common'].imported_files(torch, np, cv2, masks, COCO,
                closure, gate['admission'], ndimage) == libraries,
            'Final complete Source/spec/current CPU entry/admission/imports/CPU RNG unchanged')
    gate['lineage'].final_endpoint_check(gate['parent'])
    check()
    binding = dict(source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'],
        phase=spec['phase'], preserved_failed_training_GPU=gate['recovery_failed_GPU_record'],
        recovery_artifact_refs=artifacts, original_training_binding=gate['recovery_training_binding'],
        parent_cache_report=gate['parent_cache_report_ref'], neighbor_report=gate['neighbor_report_ref'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        qualifier_body_source=spec['qualifier_body_source'],
        scout_module_source=spec['scout_module_source'], base_scout_source=spec['base_scout_source'],
        paired_training_body_source=spec['paired_training_body_source'],
        original_paired_training_body_source=spec['original_paired_training_body_source'],
        initialization_seed=N20_CPU_SEED, order_seed=N20_CPU_ORDER_SEED,
        expected_N20_initialization_digest=expected_digest)
    binding.update(gate['training_binding_refs'])
    checks = dict(initialization_digest_and_all_five_tensors_exact=True,
        both_E8_E24_full_snapshot_START_and_FIT_bindings_exact=True,
        all_five_AdamW_states_moments_and_exact_steps_5928_17784=True,
        private_orders_rng_history_and_saved_paired_update_substream_exact=True,
        complete_same_input_FIT741_CPU_two_formulas_absolute1e4=True,
        recorded_GPU_endpoint_mean_weighted_CE_absolute1e4=True,
        original_source_spec_admission_imports_rng_and_cpu_resources_unchanged=True)
    details_path = output / 'DETAILS.json'
    write(details_path, dict(schema='N20-paired-feature-error-recovery-endpoint-CPU-details-v1',
        **binding, checks=checks, initialization_checks=initialization_checks, state_checks=state_checks,
        endpoints=endpoints, trained_state_digests=trained_digests,
        artifact_derived_history=artifact_history,
        checkpoints=artifacts['checkpoints'], FIT_endpoints=artifacts['FIT_endpoints'],
        UPDATE_TRACE=artifacts['UPDATE_TRACE'], CPU_input_reads=reference(input_trace_path),
        saved_paired_updates=17784, E8_saved_trace_prefix_updates=5928,
        trace_scope='Only the saved paired_image_update substream; no missing before/endpoint event replay claimed',
        actual_imported_files=libraries, qualified=False, ROOT_actual_CPU0_verified=False,
        GPU_pointwise_logits_qualified=False, optimizer_trajectory_replayed=False,
        selected_indices_qualified=False, mask_RLE_qualified=False, AP_evaluated=False,
        same_actual_GPU_input_F_reused=True, extra_F_independently_rerendered=False,
        teacher_or_PR_loaded=False, full_native_L_loaded=False,
        recovered_from_preserved_failed_GPU=True, original_GPU_actual0_verified=False,
        original_GPU_REPORT_created=False,
        scope='Artifact-recovery fixed E8/E24 CPU state/moments/orders/trace and same-input two formulas plus actually saved GPU endpoint CE only; preserved failed GPU is not originalGPU0'))
    check()
    q_path = output / 'Q.json'
    write(q_path, dict(schema=N20_CPU_Q_SCHEMA, status=N20_CPU_STATUS, **binding,
        details=reference(details_path), checks=checks, endpoints=endpoints,
        recovered_from_preserved_failed_GPU=True, original_GPU_actual0_verified=False,
        original_GPU_REPORT_created=False,
        images=741, natural_queries=74100, sampled_points_per_endpoint=3035136,
        CPU_packet_pair_reads=loads, total_CPU_endpoint_sampled_points=2 * 3035136,
        fixed_endpoints=[8, 24], parameters_per_arm=18851, tensors_per_arm=5,
        E8_updates_per_arm=5928, E24_updates_per_arm=17784, arms=list(N20_CPU_ARMS),
        normal_RAW_count=spec['normal_RAW_count'], total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        qualified=False, ROOT_actual_CPU0_verified=False, original_owned_terminal_self_certified=False,
        GPU_pointwise_logits_qualified=False, optimizer_trajectory_replayed=False,
        selected_indices_qualified=False, mask_RLE_qualified=False, AP_evaluated=False,
        teacher_or_PR_loaded=False, full_native_L_loaded=False,
        peak_observed_process_RSS_bytes=peak_RSS, minimum_observed_available_RAM_bytes=minimum_available_RAM,
        actual_imported_files=libraries, elapsed_seconds=time.monotonic() - started))
    check()
    print(json.dumps(dict(status=N20_CPU_STATUS, Q=reference(q_path), qualified=False)), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'Single actual ROOT owner required');run(args.spec)
