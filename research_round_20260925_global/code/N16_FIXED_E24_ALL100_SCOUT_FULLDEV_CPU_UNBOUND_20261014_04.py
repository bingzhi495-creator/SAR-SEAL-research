"""UNBOUND N16 fullDEV CPU image-bundle proof entry v04.
RAW-first14 inherited from GPU05 before any SOURCE SHA; same numeric proof.
Natural18700 queries, fixed12 COCO policies; ROOT verifies actual CPU0 later.
No actual scientific input/state reads or execution/Git by this draft author.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 14 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 14 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 stays separate; all terminal pins must equal normal+14')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9/10/11/controlled12/packet-ABI13/controlled-storage14 already passed RAW-only checks.
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
    # FIRST HASH: failed9/10/11/controlled12/packet-ABI13/controlled-storage14 and every normal/legacy failed8 passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N16-fixed-E24-all100-scout-fullDEV-GPU-spec-v1"
CPU_SCHEMA = "ROOT-N16-fixed-E24-all100-scout-fullDEV-CPU-spec-v1"
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


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.device.type == right.device.type == 'cpu' and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(tree_equal(torch, v, right[k]) for k, v in left.items())
    if type(left) in (tuple, list):
        return type(right) is type(left) and len(left) == len(right) and all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def qualified_cache_gate(spec, gate):
    """ROOT's final actual0 result, CPU checks and owners are all mandatory.

    ROOT_cache_result_schema/status are explicit SPEC pins because the final
    ROOT record ABI is owned by ROOT. They cannot replace the actual RAW joins.
    """
    closure = gate['closure']
    keys = ('cache_gpu_source', 'cache_gpu_spec', 'cache_GPU_report', 'cache_GPU_RAW',
            'cache_GPU_admission_refs', 'cache_CPU_source', 'cache_CPU_spec', 'cache_CPU_RAW',
            'cache_CPU_admission', 'cache_CPU_qualification', 'ROOTcache_result',
            'scouter_source', 'training_body_source')
    for key in keys:
        closed(closure, spec[key])
    gpu_spec = read(spec['cache_gpu_spec']['path'])
    cpu_spec = read(spec['cache_CPU_spec']['path'])
    report = read(spec['cache_GPU_report']['path'])
    q = read(spec['cache_CPU_qualification']['path'])
    result = read(spec['ROOTcache_result']['path'])
    require(type(spec['normal_RAW_count']) is int and spec['normal_RAW_count'] >= 107,
            'Actual qualified cache GPU and independent CPU must precede training')
    gi = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], spec['cache_GPU_RAW'],
        spec['cache_GPU_admission_refs'], spec['cache_gpu_source'], spec['cache_gpu_spec'], '0')
    ci = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], spec['cache_CPU_RAW'],
        spec['cache_CPU_admission'], spec['cache_CPU_source'], spec['cache_CPU_spec'], '')
    require(type(gi) is int and type(ci) is int and gi == spec['normal_RAW_count']-2 and ci == spec['normal_RAW_count']-1 and
        gpu_spec['RAW_inputs'] == spec['RAW_inputs'][:gi] and cpu_spec['RAW_inputs'] == spec['RAW_inputs'][:ci],
        'Cache actual GPU0/CPU0 owner joins and chronological prefixes must close')
    for ancestor in (gpu_spec, cpu_spec):
        require(all(r in closure.values() for r in ancestor['sources']) and
            all(spec['terminal_identity_pins'].get(k) == v for k, v in ancestor['terminal_identity_pins'].items()),
            'Complete cache SOURCE closure and immutable parent pins required')
    require(gpu_spec['entry_source'] == report['source'] == spec['cache_gpu_source'] and
        report['spec'] == spec['cache_gpu_spec'] and report['GPU_admission'] == spec['cache_GPU_admission_refs'] and
        report['phase'] == spec['cachephase'] and cpu_spec['entry_source'] == spec['cache_CPU_source'] and
        Path(gpu_spec['GPU_admission_file']).resolve(strict=True) == Path(spec['cache_GPU_admission_refs']['path']).resolve(strict=True) and
        Path(cpu_spec['CPU_admission_file']).resolve(strict=True) == Path(spec['cache_CPU_admission']['path']).resolve(strict=True),
        'Cache original source/spec/admission/phase must be exact')
    require(q['schema'] == 'N16-all100-FIT-action-cache-CPU-qualification-v1' and
        q['status'] == 'N16_FULLFIT_ALL100_ACTION_CACHE_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0' and
        q['source'] == spec['cache_CPU_source'] and q['spec'] == spec['cache_CPU_spec'] and
        q['CPU_admission'] == spec['cache_CPU_admission'] and q['phase'] == cpu_spec['phase'] and
        q['cache_GPU_source'] == spec['cache_gpu_source'] and q['cache_GPU_spec'] == spec['cache_gpu_spec'] and
        q['cache_GPU_report'] == spec['cache_GPU_report'] and q['cache_GPU_RAW'] == spec['cache_GPU_RAW'] and
        q['cache_GPU_admission'] == spec['cache_GPU_admission_refs'] and q['locked_protocol'] == spec['locked_protocol'] and
        type(q['checks']) is dict and bool(q['checks']) and all(v is True for v in q['checks'].values()),
        'Independent full-cache CPU check evidence must bind the same actual cache')
    closed(closure, q['details'])
    require(type(spec['ROOT_cache_result_schema']) is str and type(spec['ROOT_cache_result_status']) is str and
        result['schema'] == spec['ROOT_cache_result_schema'] and result['status'] == spec['ROOT_cache_result_status'] and
        result['CPU_Q'] == spec['cache_CPU_qualification'] and result['GPU_report'] == spec['cache_GPU_report'] and
        result['GPU_RAW'] == spec['cache_GPU_RAW'] and result['CPU_RAW'] == spec['cache_CPU_RAW'] and
        type(result['GPU_actual_exit_code']) is int and result['GPU_actual_exit_code'] == 0 and
        type(result['CPU_actual_exit_code']) is int and result['CPU_actual_exit_code'] == 0 and result['fully_qualified'] is True,
        'ROOT final closure must independently verify both actual exits0, not promote CPU check self-report')
    require(report['schema'] == 'N16-all100-FIT-action-cache-GPU-report-v1' and
        report['status'] == 'N16_FULLFIT_ALL100_ACTION_CACHE_COMPLETE_PENDING_INDEPENDENT_CPU' and
        report['recipe'] == gate['protocol'] and report['images'] == q['images'] == 741 and
        report['raw_GT'] == q['raw_GT'] == 1661 and report['natural_queries'] == q['natural_queries'] == 74100 and
        report['sampled_points'] == q['sampled_points'] and report['original_native610_unchanged'] is True and
        report['original_PR8_unchanged'] is True and report['new_background_empty_set_supervision'] is True,
        'Full741/1661 natural100 fixed-label cache qualification required')
    rows, packets, carriers = report['rows'], report['packet_refs'], report['native_L_refs']
    require(len(rows) == len(packets) == len(carriers) == 741 and
        len({r['path'] for r in packets + carriers}) == 1482, 'Complete unique paired packet/L refs required')
    for index, (row, packet_ref, carrier_ref) in enumerate(zip(rows, packets, carriers)):
        require(row['dataset_index'] == index and row['packet'] == packet_ref and
            row['natural_native_L_carrier'] == carrier_ref and row['natural_queries'] == 100 and
            type(row['points']) is int and 0 <= row['points'] <= 4096, 'Ordered full-cache row identity required')
        closed(closure, packet_ref)
        closed(closure, carrier_ref)  # before CUDA only; never torch.load full L during training
    require(sum(r['raw_GT'] for r in rows) == 1661 and sum(r['points'] for r in rows) == report['sampled_points'],
            'Cache sample/GT count identity required')
    return dict(report=report, q=q, ROOT_result=result, rows=rows, packets=packets, carriers=carriers,
                nonempty_images=sum(r['points'] > 0 for r in rows), total_sample_points=report['sampled_points'])


def source_gate(path,spec):
    gate=inherited_source_gate(path,spec,cpu=True)
    for key in ('completed_GPU_source','completed_GPU_spec','completed_GPU_report','completed_GPU_RAW',
                'completed_GPU_admission_refs','CPU_adapter_source','selector_cpu_proof_source','PR_formula_source'):
        closed(gate['closure'],spec[key])
    gpu=load(spec['completed_GPU_source'],'_N16_completed_actual_FULLDEV_GPU_contract')
    require(gpu.CPU_SCHEMA==CPU_SCHEMA and spec['normal_RAW_count']>=110 and spec['selector_mode']=='reuse', 'New independent CPU normal110/schema/reuse required')
    parent=read(spec['completed_GPU_spec']['path']);report=read(spec['completed_GPU_report']['path'])
    index=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],spec['completed_GPU_RAW'],
        spec['completed_GPU_admission_refs'],spec['completed_GPU_source'],spec['completed_GPU_spec'],'0')
    require(type(index) is int and index==spec['normal_RAW_count']-1 and parent['normal_RAW_count']==index and
        parent['RAW_inputs']==spec['RAW_inputs'][:index] and parent['schema']==gpu.SCHEMA and
        parent['entry_source']==report['source']==spec['completed_GPU_source'] and report['spec']==spec['completed_GPU_spec'] and
        report['GPU_admission']==spec['completed_GPU_admission_refs'] and report['phase']==parent['phase'] and
        Path(parent['GPU_admission_file']).resolve(strict=True)==Path(spec['completed_GPU_admission_refs']['path']).resolve(strict=True),
        'Latest completed fullDEV GPU0 owner109 and exact source/spec/ad/phase must close')
    require(all(r in gate['closure'].values() for r in parent['sources']) and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()), 'Complete parent SOURCE/pins required')
    for key in gpu.ROOT_FIELDS:
        require(spec[key]==parent[key], 'CPU cannot change fixed GPU contract: '+key)
        value=spec[key]
        if type(value) is dict and set(value)=={'path','bytes','sha256'}:closed(gate['closure'],value)
    gate['context']['selector_mode']='reuse'
    prior,prior_q=gate['lineage'].prior_gate(gate['context'],gate['closure'],gate['owners'])
    selector=gate['lineage'].selector_parent_gate(gate['context'],gate['closure'],gate['owners'])
    require(prior['teacher_checkpoint']==gate['cache_report']['teacher_checkpoint'], 'Old native FIT/DEV teacher lineage exact')
    gate.update(prior=prior,prior_q=prior_q,packets=prior_q['packet_refs'],selector=selector)
    gpu.training_gate(spec,gate,cpu=True)  # trainingGPU107/CPU108 in current110; separate fullDEVGPU109 above
    require(report['schema']==gpu.REPORT_SCHEMA and report['status']==gpu.STATUS and
        report['images']==187 and report['raw_GT']==380 and report['queries_per_image']==100 and
        report['predictions_per_control']==18700 and tuple(report['controls'])==gpu.policy_names() and
        report['fixed_main_epoch']==24 and report['fixed_curve_epoch']==8 and report['DEV_winner_selected'] is False and
        report['score_class_query_order_preserved'] is True and report['GT_used_for_prediction'] is False and
        report['independent_CPU_selection_proof_complete'] is False, 'Completed fixed full-scope GPU report, not Q promotion')
    binding={k:report[k] for k in ('source','spec','GPU_admission','phase','training_GPU_report','training_CPU_qualification',
                                 'ROOTtrain_result','scout_checkpoints','locked_protocol')}
    manifest=read(closed(gate['closure'],report['capture_manifest']))
    require(manifest['schema']=='N16-GPU-all18700-image-bundle-capture-manifest-v2' and manifest['images']==187 and
        manifest['queries']==18700 and all(manifest[k]==v for k,v in binding.items()) and len(manifest['rows'])==187 and
        [row['bundle'] for row in manifest['rows']]==report['capture_refs'] and len(report['capture_refs'])==187 and
        manifest['physical_capture_refs']==report['capture_refs'] and manifest['capture_files']==187 and
        manifest['internal_capture_locators_are_strings'] is True and report['capture_storage']=='one_physical_actual_GPU_bundle_per_image' and
        len({r['path'] for r in report['capture_refs']})==187,
        'Exact187 actual physical image bundles and complete manifest required')
    for ref in report['capture_refs']:closed(gate['closure'],ref)
    for row in manifest['rows']:closed(gate['closure'],row['diagnostics'])
    for key in ('START','trace','native_reference','strong_fixed_reference','selector_manifest'):closed(gate['closure'],report[key])
    require(report['native_reference']==gate['context']['native_reference_predictions'] and report['native_packets']==gate['packets'] and
        report['selector_manifest']==selector['manifest_ref'] and report['strong_fixed_reference']==spec['strong_fixed_reference_predictions'],
        'Old native/source/selector and original strongest E24 lineage must remain exact')
    for name in gpu.policy_names():closed(gate['closure'],report['control_results'][name]['predictions'])
    gate.update(completed_GPU=gpu,completed_parent=parent,completed_report=report,manifest=manifest,capture_binding=binding)
    return gate


def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);spec=read(path)
    require(Path(spec_path).is_absolute() and Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True), 'Own absolute CPU entry/spec required')
    gate=source_gate(path,spec);gpu=gate['completed_GPU'];report=gate['completed_report'];context=gate['context'];closure=gate['closure']
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh ownCPU output')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and psutil.virtual_memory().available>=16*1024**3, 'CPU01/cudaempty/16GiB available')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified CPU Torch cold')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0', 'Original numerical runtime')
    cv2.setNumThreads(2)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)
    adapter=load(spec['CPU_adapter_source'],'_N16_actual_CPU_fullDEV_adapter')
    proof=load(spec['selector_cpu_proof_source'],'_N16_frozen_GPUlogit_CPUproof')
    manual=load(spec['PR_formula_source'],'_N16_closed_independent_PR_formula')
    require(proof.ATOL==1e-4 and proof.POINT_CHUNK==2048 and gate['protocol']['numeric']==dict(absolute=1e-4,relative=0,actual_after_sign_exact=True), 'Prelocked PR numeric/sign/pointchunk scope')
    n10=load(context['N10_evaluation_source'],'_N16_original_N10_CPUexport')
    reader=load(context['reader_source'],'_N16_original_population_reader')
    official=load(context['COCO_source'],'_N16_original_official_COCO_CPU')
    strong,models,states,model_proofs,evalbody,labels=gpu.model_load_cpu(torch,spec,gate)
    outside=torch.get_rng_state().clone()
    def check():
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and not torch.cuda.is_initialized(), 'Bounded CPU-only whole leaf, no retry')
    def original_head(F,e,L):
        check();result=strong(F,e,L,L.new_zeros((len(L),49)),arm='PR');u=result['residual'];return u,L+u
    def independent_PR(F,e,L):
        check();return api(manual,'independent_PR')(torch,states['LOG_PR'],F,L)
    formulas=dict(original_head=original_head,independent_PR=independent_PR)
    coco=COCO(str(closed(closure,context['DEV'])));ids=list(coco.imgs)
    require(len(ids)==187 and len(coco.anns)==380 and ids==[v['id'] for v in coco.dataset['images']] and set(coco.cats)=={1}, 'Original full187/GT380/category1')
    native=read(closed(closure,report['native_reference']));fixed=read(closed(closure,report['strong_fixed_reference']))
    native_pop=reader.population(coco,native,ids);fixed_pop=reader.population(coco,fixed,ids)
    expected={name:read(closed(closure,report['control_results'][name]['predictions'])) for name in gpu.policy_names()}
    expected_pop={name:reader.population(coco,rows,ids) for name,rows in expected.items()}
    require(all(len(v)==18700 for v in expected.values()), 'Complete twelve18700 GPU export populations')
    images=gate['manifest']['rows'];native_rows=read(closed(closure,gate['prior']['image_rows']))
    require(len(native_rows)==187 and [r['image_id'] for r in images]==ids, 'Canonical image roster')
    all_predictions={name:[] for name in gpu.policy_names()};used=set();physical={r['path']:r for r in report['capture_refs']};total_members=0
    output.mkdir();image_reports=[]
    for index,row in enumerate(images):
        check();require(row['dataset_index']==index and row['natural_queries']==100 and row['native_packet']==gate['packets'][index] and
            row['selector']==gate['selector']['manifest']['rows'][index] and row['qids']==[v['query_id'] for v in native_pop[row['image_id']]], 'Actual image/native packet/selector/order identity')
        bundle_ref=row['bundle'];ref_shape(bundle_ref)
        require(physical.get(bundle_ref['path'])==bundle_ref and closure.get(bundle_ref['path'])==bundle_ref and bundle_ref['path'] not in used,
            'Each image loads one distinct SOURCE-closed actual physical bundle exactly once')
        bundle=torch.load(str(bound(bundle_ref)),map_location='cpu');used.add(bundle_ref['path']);member_use=set()
        shared_id=row['shared_actual_image_inputs']
        shared_value=gpu.lookup_CPU_bundle_member(torch,bundle,shared_id,gate['capture_binding'],member_use=member_use)
        require(bundle['dataset_index']==index and bundle['image_id']==row['image_id'] and
            type(row['member_locators']) is list and all(type(v) is str for v in row['member_locators']) and
            len(row['member_locators'])==len(set(row['member_locators']))==bundle['member_count'] and
            set(row['member_locators'])==set(bundle['members'])==set(bundle['member_registry']),
            'Actual whole-bundle image/member registry and manifest locators must match exactly')
        require(shared_value['dataset_index']==index and shared_value['image_id']==row['image_id'] and shared_value['identity']==row['identity'] and
            shared_value['native_packet']==row['native_packet'] and shared_value['selector']==row['selector'] and
            shared_value['native_export_query_ids']==row['qids'] and tuple(shared_value['native_L'].shape[1:])==(coco.imgs[row['image_id']]['height'],coco.imgs[row['image_id']]['width']),
            'Shared actual capture must belong to exact old native image/query namespace')
        values,detail=adapter.replay_image_bundle(torch,np,cv2,masks,n10,gpu,proof,labels,formulas,bundle,shared_id,row['query_profiles'],
            gate['capture_binding'],native_pop[row['image_id']],fixed_pop[row['image_id']],check=check,member_use=member_use)
        require(member_use==set(row['member_locators']), 'All and only actual registered members consumed by proof')
        total_members+=len(member_use)
        for name,rows in values.items():
            require(rows==expected_pop[name][row['image_id']], 'Exact full mask/RLE/original score/class/order CPU/GPU readback differs: '+name)
            all_predictions[name].extend(rows)
        target=output/('IMAGE_%03d_PROOF.json'%index);write(target,dict(dataset_index=index,image_id=row['image_id'],queries=100,bundle=bundle_ref,registered_members=bundle['member_count'],consumed_members=len(member_use),details=detail))
        image_reports.append(reference(target));del bundle,shared_value,member_use,values,detail
        gc.collect()
    require(used==set(physical) and all(len(v)==18700 for v in all_predictions.values()), 'All187 physical bundles consumed once; complete all18700 for all12 policies')
    COCO_results={}
    for name,values in all_predictions.items():
        check();require(values==expected[name], 'Complete ordered prediction file readback exact')
        metrics=official.coco_metrics(coco,values,ids,COCOeval)
        require(metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100] and metrics['score_filter'] is None and metrics['original_GT_preserved'] is True and
            metrics['official_COCO_crowd_ignore_semantics'] is True, 'Original official COCO adapter scope')
        n10.stats_exact(metrics['stats'],report['control_results'][name]['stats'],'All12 COCO stats exact CPU/GPU readback')
        COCO_results[name]=metrics
    require(torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and
        torch.get_num_threads()==torch.get_num_interop_threads()==2 and all(torch.equal(v,states['LOG_PR'][k]) for k,v in strong.state_dict().items()), 'CPU-only/frozen PR/globalRNG/resources unchanged')
    del strong,models,states,formulas
    gc.collect()
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['CPU_admission_file'])==gate['admission_ref'], 'Final ownCPU Source/spec/admission exact')
    require(gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)==libraries, 'Actual closed CPU imported libraries unchanged')
    check();binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        completed_GPU_source=spec['completed_GPU_source'],completed_GPU_spec=spec['completed_GPU_spec'],completed_GPU_report=spec['completed_GPU_report'],
        completed_GPU_RAW=spec['completed_GPU_RAW'],completed_GPU_admission=spec['completed_GPU_admission_refs'],capture_manifest=report['capture_manifest'],
        locked_protocol=spec['locked_protocol'],CPU_adapter_source=spec['CPU_adapter_source'],selector_cpu_proof_source=spec['selector_cpu_proof_source'],PR_formula_source=spec['PR_formula_source'])
    write(output/'DETAILS.json',dict(schema='N16-fixed-fullDEV-CPU-proof-details-v1',**binding,image_proofs=image_reports,COCO_results=COCO_results,
        actual_capture_files_consumed=len(used),registered_bundle_members_consumed=total_members,all_actual_GPU_logits_selected_indices_exact=True,all_PR_two_formulas_absolute1e4_strictsign=True,
        all_native_strong_and_new_mask_RLE_exact=True,all12_official_COCO12_exact=True,CPU_scout_forward_used_for_ranking=False,actual_imported_files=libraries))
    write(output/'Q.json',dict(schema='N16-fixed-E24-all100-scout-fullDEV-CPU-qualification-v1',
        status='N16_ALL18700_SCOUT_FULLDEV_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0',**binding,details=reference(output/'DETAILS.json'),
        images=187,GT=380,natural_queries=18700,policies=list(gpu.policy_names()),checks=dict(actual_capture_closure_all_consumed=True,all_registered_bundle_members_consumed=True,
            all100_C_K_exact=True,actual_GPUlogits_selection_exact=True,original_PR_two_CPU_formulas_absolute1e4_strictsign=True,
            full_native_strong_and_scout_masks_RLE_exact=True,all12_official_COCO12_exact=True),
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,CUDA_initialized=False,
        CPU_scout_forward_used_for_ranking=False,CPU_F_rerendered=False,CPU_renderer_equivalence_claimed=False,
        AP_results_interpreted=False,task_efficacy_established=False,innovation_established=False,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status='N16_ALL18700_SCOUT_FULLDEV_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0',Q=reference(output/'Q.json'),qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND_DRAFT: ROOT completed GPU0/Source/capture/old-native/scout gates required')
    run(args.spec)


