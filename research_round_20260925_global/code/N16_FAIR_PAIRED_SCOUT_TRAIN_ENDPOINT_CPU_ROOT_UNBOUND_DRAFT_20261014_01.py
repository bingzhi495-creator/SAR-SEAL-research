"""N16 all100 FIT cache. ROOT-owned once; no scout training or AP.
New background emptyset proxy is shared by both future scouts, not native maskGT.
Frozen original lineage/source/runtime is preserved; no automatic retry.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 12 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 12 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 is separate; total terminal pins must equal normal+9')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9 has ALREADY passed its RAW-only identity check.
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
    # FIRST HASH: failed9 and every normal/legacy failed8 RAW owner passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N16-fair-paired-scout-training-GPU-spec-v1"
CPU_SCHEMA = "ROOT-N16-fair-paired-scout-training-endpoint-CPU-spec-v1"
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

def source_gate(path, spec, cpu=False):
    require(spec['schema'] == (CPU_SCHEMA if cpu else SCHEMA) and
        type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] >= 105 + int(cpu) and
        spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] and spec['deadline_seconds'] == 1800 and
        spec['automatic_retry'] is False and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
        'New N15 actual full ancestry/schema/owner resource recipe required')
    ninth_failed_raw_only(spec)
    failed10=tenth_failed_raw_only(spec)
    failed11=eleventh_failed_raw_only(spec)
    failed12=twelfth_resource_failure_raw_only(spec)
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


def training_input_gate(spec,gate):
    closure=gate['closure']
    for key in ('training_GPU_source','training_GPU_spec','training_GPU_report','training_GPU_RAW',
                'training_GPU_admission_refs','qualifier_body_source','scouter_source','scout_formula_source','CPU_endpoint_numeric_protocol'):
        closed(closure,spec[key])
    parent=read(spec['training_GPU_spec']['path']);report=read(spec['training_GPU_report']['path'])
    require(type(spec['normal_RAW_count']) is int and spec['normal_RAW_count']>=108,
            'Completed training GPU owner must precede endpoint CPU attempt')
    index=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],spec['training_GPU_RAW'],
        spec['training_GPU_admission_refs'],spec['training_GPU_source'],spec['training_GPU_spec'],'0')
    require(type(index) is int and index==spec['normal_RAW_count']-1 and
        parent['RAW_inputs']==spec['RAW_inputs'][:index] and parent['normal_RAW_count']==index,
        'Actual training GPU0 latest owner and exact ancestor prefix required')
    require(parent['schema']==SCHEMA and parent['entry_source']==spec['training_GPU_source'] and
        report['source']==spec['training_GPU_source'] and report['spec']==spec['training_GPU_spec'] and
        report['GPU_admission']==spec['training_GPU_admission_refs'] and report['phase']==parent['phase'] and
        Path(parent['GPU_admission_file']).resolve(strict=True)==Path(spec['training_GPU_admission_refs']['path']).resolve(strict=True) and
        all(r in closure.values() for r in parent['sources']) and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
        'Training source/spec/admission/phase/fullSOURCE/owner pins must close')
    require(report['schema']=='N16-fair-paired-scout-training-GPU-report-v1' and
        report['status']=='N16_PAIRED_SCOUT_FIXED24_TRAINED_PENDING_INDEPENDENT_CPU_QUALIFICATION' and
        report['completed_epoch']==24 and report['parameters_per_arm']==16547 and report['tensors_per_arm']==4 and
        report['paired_initial_state_exact'] is True and report['global_RNG_preserved'] is True and
        report['AP_evaluated'] is False and report['teacher_loaded'] is False and report['PR_head_loaded'] is False,
        'Fixed24 qualified-source training report required, never AP success')
    for key in ('scouter_source','training_body_source','cache_CPU_qualification','ROOTcache_result','cache_GPU_report','locked_protocol'):
        require(report[key]==parent[key], 'Actual training report parent binding differs: '+key)
    require(parent['scouter_source']==spec['scouter_source'] and spec['scout_formula_source']==spec['scouter_source'] and
        parent['locked_protocol']==spec['locked_protocol'], 'Canonical independent formula source and same locked scientific protocol')
    # Parent GPU spec has107 normals; current CPU spec has108. Do not shift the
    # two cache owner joins by applying the GPU helper to current CPU counts.
    cache=qualified_cache_gate(parent,gate)
    require(parent['CPU_endpoint_numeric_protocol']==spec['CPU_endpoint_numeric_protocol'] and
        spec['CPU_endpoint_numeric_protocol'] in parent['sources'], 'Numeric protocol must have been SOURCE-pinned before actual GPU training')
    numeric=read(spec['CPU_endpoint_numeric_protocol']['path'])
    require(numeric['schema']=='N16-paired-scout-endpoint-CPU-numeric-protocol-v1' and
        numeric['status']=='LOCKED_BEFORE_GPU_TRAINING' and numeric['locked_before_GPU_training'] is True and
        numeric['locked_before_CPU_execution'] is True and numeric['abs_logits']==numeric['abs_meanCE']==1e-4 and
        numeric['rtol']==0 and numeric['epochs']==[8,24], 'ROOT must prelock explicit numeric scope/tolerances before GPU training')
    for key in ('initial_state','start','trace'):
        closed(closure,report[key])
    require(set(report['checkpoints'])==set(report['fixed_FIT_endpoints'])=={'8','24'}, 'Only full fixed8/24 actual checkpoint refs')
    for ref in report['checkpoints'].values():closed(closure,ref)
    start=read(report['start']['path'])
    require(start['source']==report['source'] and start['spec']==report['spec'] and start['GPU_admission']==report['GPU_admission'] and
        start['phase']==report['phase'] and start['cache_CPU_qualification']==parent['cache_CPU_qualification'] and
        start['ROOTcache_result']==parent['ROOTcache_result'] and start['paired_initial_state_exact'] is True,
        'Original actual START provenance and common init proof must remain exact')
    return dict(parent=parent,report=report,cache=cache,numeric=numeric)


def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(), 'Absolute ROOT CPU SPEC required')
    spec=read(path)
    require(Path(__file__).resolve(strict=True)==Path(spec['entry_source']['path']).resolve(strict=True), 'CPU entry must be actual current source')
    gate=source_gate(path,spec,cpu=True)
    inputs=training_input_gate(spec,gate);parent,report,cache=inputs['parent'],inputs['report'],inputs['cache']
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh ROOT CPU output required')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        psutil.virtual_memory().available>=16*1024**3, 'CPU01/cudaempty/minavailable16GiB required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Cold qualified Torch, no CUDA init')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    scout=load(spec['scouter_source'],'_N16_CPU_actual_scout_factory')
    formula=load(spec['scout_formula_source'],'_N16_CPU_independent_scout_formula')
    qualifier=load(spec['qualifier_body_source'],'_N16_CPU_actual_training_endpoint_qualifier')
    require(qualifier.ABS_LOGITS==qualifier.ABS_MEAN_CE==1e-4 and qualifier.INIT_SEED==202610140161 and
        qualifier.ORDER_SEED==202610140160001, 'Actual qualifier must implement prelocked numeric/seed recipe')
    imports={}
    for name,obj in (('torch',torch),('functional',torch.nn.functional),('AdamW',torch.optim.AdamW),
                     ('LayerNorm',torch.nn.LayerNorm),('Linear',torch.nn.Linear)):
        ref=reference(inspect.getsourcefile(obj))
        require(ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256'],
            'Actual CPU Torch source must be admitted: '+name)
        imports[name]=ref
    outside=torch.get_rng_state().clone()
    def check(event=None):
        require(time.monotonic()-started<1800 and process.memory_info().rss<16*1024**3 and not torch.cuda.is_initialized(),
                'One bounded CPU-only endpoint attempt; no automatic retry')
    def payload_binding(value):
        for key in ('source','spec','GPU_admission','phase','locked_protocol','cache_CPU_qualification',
                    'ROOTcache_result','cache_GPU_report','scouter_source','training_body_source'):
            require(value[key]==report[key], 'Actual training checkpoint provenance differs: '+key)
    initial_payload=torch.load(str(closed(gate['closure'],report['initial_state'])),map_location='cpu')
    payload_binding(initial_payload);initial=initial_payload['snapshot']
    qualifier.validate_initial(torch,scout,initial)
    checkpoints={}
    for epoch in (8,24):
        payload=torch.load(str(closed(gate['closure'],report['checkpoints'][str(epoch)])),map_location='cpu')
        payload_binding(payload)
        require(payload['schema']=='N16-fair-paired-scout-training-checkpoint-v1' and
            payload['parameters_per_arm']==16547 and payload['tensors_per_arm']==4 and payload['initial_state']==report['initial_state'] and
            tree_equal(torch,payload['fixed_FIT_endpoint'],report['fixed_FIT_endpoints'][str(epoch)]),
            'Fixed checkpoint full state/GPU endpoint/initial provenance required')
        checkpoints[epoch]=payload['snapshot'];del payload
    require(checkpoints[24]['history'][:8]==checkpoints[8]['history'] and
        checkpoints[24]['epoch_orders'][:8]==checkpoints[8]['epoch_orders'], 'E8 checkpoint must be exact prefix of24 history/orders')
    traces=[json.loads(line) for line in Path(report['trace']['path']).read_text(encoding='utf-8-sig').splitlines()]
    qualifier.validate_trace(traces,checkpoints[24],cache['rows'])
    del traces
    def loader(index):
        check()
        row,ref=cache['rows'][index],cache['packets'][index]
        packet=torch.load(str(closed(gate['closure'],ref)),map_location='cpu')
        require(packet['schema']=='N16-all100-FIT-action-sampled-cache-body-v1' and packet['dataset_index']==index and
            packet['identity']==row['identity'] and packet['source']==parent['cache_gpu_source'] and packet['spec']==parent['cache_gpu_spec'] and
            packet['GPU_admission']==parent['cache_GPU_admission_refs'] and packet['phase']==parent['cachephase'] and
            packet['protocol']==spec['locked_protocol'] and 'natural_native_L' not in packet and
            packet['natural_native_L_carrier']==cache['carriers'][index] and len(packet['samples']['query_ids'])==row['points'] and
            packet['samples']['seed']==202610140160000+index and packet['matching_loss_seed']==202610122001+index,
            'Qualified same-input thin cache metadata/points/seeds must remain exact')
        require(gate['closure'].get(packet['natural_native_L_carrier']['path'])==packet['natural_native_L_carrier'],
                'CPU endpoint uses frozen carrier reference only; no native L rerender/load')
        thin=dict(sampled_F_unique=packet['sampled_F_unique'],sample_to_unique=packet['sample_to_unique'],native_e0=packet['native_e0'],
            samples={k:packet['samples'][k] for k in ('query_ids','native_L','importance_weights','utility_labels','error_labels','stratum_ids','delta')})
        del packet;return thin
    output.mkdir()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        training_GPU_source=spec['training_GPU_source'],training_GPU_spec=spec['training_GPU_spec'],training_GPU_report=spec['training_GPU_report'],
        training_GPU_RAW=spec['training_GPU_RAW'],training_GPU_admission=spec['training_GPU_admission_refs'],
        cache_CPU_qualification=parent['cache_CPU_qualification'],ROOTcache_result=parent['ROOTcache_result'],
        locked_protocol=spec['locked_protocol'],CPU_endpoint_numeric_protocol=spec['CPU_endpoint_numeric_protocol'],
        qualifier_body_source=spec['qualifier_body_source'],scouter_source=spec['scouter_source'],scout_formula_source=spec['scout_formula_source'])
    details={}
    for epoch in (8,24):
        check()
        details[str(epoch)]=qualifier.validate_endpoint(torch,scout,formula,initial,checkpoints[epoch],
            report['fixed_FIT_endpoints'][str(epoch)],loader,cache['rows'],epoch,observer=check)
    require(torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized() and
        process.cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2, 'CPU RNG/runtime/resources changed')
    del initial_payload,initial,checkpoints
    import gc
    gc.collect()
    for ref in spec['sources']:bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['CPU_admission_file'])==gate['admission_ref'], 'Final actual CPU SOURCE/spec/admission changed')
    validate_protocol(spec,gate['closure']);check()
    write(output/'DETAILS.json',dict(schema='N16-paired-scout-training-endpoint-CPU-details-v1',**binding,
        endpoints=details,complete_trace_order_update_history_exact=True,common_initialization_exact=True,
        CPU_factory_independent_logits_absolute1e4=True,CPU_meanCE_GPU_aggregate_absolute1e4=True,
        no_GPU_point_logits_captured=True,GPU_pointwise_logits_qualified=False,selected_indices_qualified=False,
        optimizer_trajectory_replayed=False,AP_evaluated=False))
    write(output/'Q.json',dict(schema='N16-paired-scout-training-endpoint-CPU-qualification-v1',
        status='N16_PAIRED_SCOUT_E8_E24_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0',**binding,
        details=reference(output/'DETAILS.json'),checkpoints=report['checkpoints'],initial_state=report['initial_state'],
        parameters_per_arm=16547,tensors_per_arm=4,epochs=[8,24],
        checks=dict(common_initialization_exact=True,four_FP32_state_tensors_exact=True,AdamW_state_readback_shapes_finite_steps_hyperparameters_exact=True,
            all_private_epoch_orders_RNG_and_counts_exact=True,complete_actual_trace_history_identity_exact=True,
            same_GPU_cache_inputs_CPU_two_formula_logits_absolute1e4=True,CPU_weighted_meanCE_GPU_aggregates_absolute1e4=True),
        actual_imported_files=imports,elapsed_seconds=time.monotonic()-started,CUDA_initialized=False,
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,
        GPU_pointwise_logits_qualified=False,selected_indices_qualified=False,optimizer_trajectory_replayed=False,
        AP_evaluated=False,task_efficacy_established=False,innovation_established=False,
        scope='Endpoint state/optimizer/order/count consistency and independent CPU numeric readback only; ROOT original actual0/fullSOURCE closure still required.'))
    print(json.dumps(dict(status='N16_PAIRED_SCOUT_E8_E24_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0',Q=reference(output/'Q.json'),qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND_DRAFT: ROOT alone binds completed actual GPU/cache owners and prelocked numeric protocol')
    run(args.spec)


