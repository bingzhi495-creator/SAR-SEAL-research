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
CPU_SCHEMA = "ROOT-N16-all100-FIT-action-cache-CPU-spec-v1"
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
    if type(left) is dict:
        return type(right) is dict and set(left) == set(right) and all(tree_equal(torch, v, right[k]) for k, v in left.items())
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


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(), 'ROOT absolute training SPEC required')
    spec = read(path)
    gate = source_gate(path, spec)
    cache = qualified_cache_gate(spec, gate)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh ROOT-owned output required')
    import psutil
    process = psutil.Process()
    require(process.cpu_affinity() == [0, 1] and psutil.virtual_memory().available >= 16*1024**3, 'CPU01 and16GiB availableRAM required')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified Torch cold runtime required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    scout = load(spec['scouter_source'], '_N16_actual_paired_scout')
    body = load(spec['training_body_source'], '_N16_actual_paired_training_body')
    require(body.INIT_SEED == 202610140161 and body.ORDER_SEED == 202610140160001 and body.EPOCHS == 24 and
        tuple(body.ENDPOINTS) == (8,24) and body.ADAMW == dict(lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8),
        'Exact paired scout protocol required')
    imported = {}
    for name, obj in (('torch',torch), ('functional',torch.nn.functional), ('AdamW',torch.optim.AdamW),
                       ('LayerNorm',torch.nn.LayerNorm), ('Linear',torch.nn.Linear)):
        ref = reference(inspect.getsourcefile(obj))
        require(ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(ref['path']) == ref['sha256'],
                'Actual Torch source file must be admitted: ' + name)
        imported[name] = ref
    require(not torch.cuda.is_initialized(), 'Cache/source/library gates must precede CUDA')
    torch.cuda.init(); require(torch.cuda.current_device() == 0, 'GPU0 only')
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    external = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone())
    state = body.make_paired_state(torch, scout, device='cuda:0')
    initial = body.cpu_snapshot(torch, state)
    require(tree_equal(torch, initial['models']['UTILITY'], initial['models']['ERROR']), 'Identical normal initialization proof required')
    output.mkdir()
    binding = dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        locked_protocol=spec['locked_protocol'],cache_CPU_qualification=spec['cache_CPU_qualification'],
        ROOTcache_result=spec['ROOTcache_result'],cache_GPU_report=spec['cache_GPU_report'],
        scouter_source=spec['scouter_source'],training_body_source=spec['training_body_source'])
    write(output/'START.json',dict(schema='N16-fair-paired-scout-training-start-v1',**binding,
        initialization_seed=202610140161,order_seed=202610140160001,parameters_per_arm=16547,tensors_per_arm=4,
        paired_initial_state_exact=True,cache_sample_points=cache['total_sample_points'],cache_nonempty_images=cache['nonempty_images'],
        fixed_refiner_has_seen_FIT=True,label_generation_cost_not_ignored=True,teacher_loaded=False,PR_head_loaded=False,
        native_L_loaded=False,DEV_loaded=False,TEST_loaded=False,AP_evaluated=False,qualification_pending=True))
    initial_path = output/'INITIAL_PAIRED_STATE.pth'
    with initial_path.open('xb') as stream: torch.save(dict(**binding,snapshot=initial),stream)
    initial_replay = torch.load(str(initial_path),map_location='cpu')
    require(tree_equal(torch,dict(**binding,snapshot=initial),initial_replay), 'Initial paired actual CPU readback differs')
    initial_ref = reference(initial_path)
    del initial, initial_replay
    checkpoint_refs, endpoint_metrics = {}, {}
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        def check(event=None):
            require(time.monotonic()-started < 1800 and process.memory_info().rss < 16*1024**3,
                    'Bounded1800s/process16GiB single owned training attempt')
            if event is not None:
                trace.write(json.dumps(event,allow_nan=False)+'\n');trace.flush()
        def loader(index):
            check()
            row, packet_ref = cache['rows'][index], cache['packets'][index]
            packet = torch.load(str(closed(gate['closure'],packet_ref)),map_location='cpu')
            require(packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1' and packet['dataset_index'] == index and
                packet['identity'] == row['identity'] and packet['identity']['image_id'] == row['image_id'] and
                packet['source'] == spec['cache_gpu_source'] and packet['spec'] == spec['cache_gpu_spec'] and
                packet['GPU_admission'] == spec['cache_GPU_admission_refs'] and packet['phase'] == spec['cachephase'] and
                packet['protocol'] == spec['locked_protocol'] and 'natural_native_L' not in packet and
                packet['natural_native_L_carrier'] == cache['carriers'][index] and
                len(packet['samples']['query_ids']) == row['points'] and
                packet['samples']['seed'] == 202610140160000+index and packet['matching_loss_seed'] == 202610122001+index,
                'Fresh thin-cache source/spec/ad/phase/sampling/identity must be exact')
            carrier_ref = packet['natural_native_L_carrier']
            require(gate['closure'].get(carrier_ref['path']) == carrier_ref, 'Frozen L ref only, no rerender/load')
            thin = dict(sampled_F_unique=packet['sampled_F_unique'],sample_to_unique=packet['sample_to_unique'],
                native_e0=packet['native_e0'],samples={k:packet['samples'][k] for k in
                    ('query_ids','native_L','importance_weights','utility_labels','error_labels','stratum_ids','delta')})
            del packet
            return thin
        def checkpoint(epoch, endpoint, snapshot):
            check()
            require(epoch in (8,24) and epoch not in checkpoint_refs and snapshot['completed_epoch'] == epoch and
                snapshot['updates'] == {arm:epoch*cache['nonempty_images'] for arm in ('UTILITY','ERROR')} and
                snapshot['visited_images'] == epoch*741 and snapshot['sampled_points'] == epoch*cache['total_sample_points'] and
                endpoint['nonempty_images'] == cache['nonempty_images'] and endpoint['sampled_points'] == cache['total_sample_points'],
                'Complete fixed8/24 paired updates/cache exposure identity required')
            payload = dict(schema='N16-fair-paired-scout-training-checkpoint-v1',**binding,
                parameters_per_arm=16547,tensors_per_arm=4,snapshot=snapshot,fixed_FIT_endpoint=endpoint,
                initial_state=initial_ref,AP_evaluated=False,CPU_qualification_pending=True)
            target = output/('PAIRED_SCOUT_E%d.pth'%epoch)
            with target.open('xb') as stream:torch.save(payload,stream)
            replay = torch.load(str(target),map_location='cpu')
            require(tree_equal(torch,payload,replay), 'Actual complete checkpoint double-readback differs')
            checkpoint_refs[str(epoch)] = reference(target)
            endpoint_metrics[str(epoch)] = endpoint
            del replay, payload
        summary = body.train_until(torch,state,loader,last_epoch=24,image_count=741,observer=check,checkpoint_callback=checkpoint)
    require(set(checkpoint_refs) == {'8','24'} and summary['completed_epoch'] == 24 and
        state['updates'] == {arm:24*cache['nonempty_images'] for arm in ('UTILITY','ERROR')} and
        torch.equal(external[0],torch.get_rng_state()) and torch.equal(external[1],torch.cuda.get_rng_state(0)),
        'Fixed24 completion, paired counters and outside RNG exactness required')
    final_updates, sampled_points, empty_images = dict(state['updates']),state['sampled_points'],state['empty_images']
    peak = torch.cuda.max_memory_allocated(0)
    for head in state['heads'].values():head.cpu()
    del head
    for optimizer in state['optimizers'].values():optimizer.state.clear()
    del optimizer
    state['optimizers'].clear();state['heads'].clear();state.clear();del state
    import gc
    gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup = dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after'] == 0 and peak <= 4096*1024**2, 'Exact allocator0 cleanup and4GiB cap required')
    for ref in spec['sources']:bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
        reference(spec['GPU_admission_file']) == gate['admission_ref'] and process.cpu_affinity() == [0,1] and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Final SOURCE/spec/admission/resources changed')
    validate_protocol(spec,gate['closure'])
    require(time.monotonic()-started < 1800, 'Final bounded deadline')
    write(output/'REPORT.json',dict(schema='N16-fair-paired-scout-training-GPU-report-v1',
        status='N16_PAIRED_SCOUT_FIXED24_TRAINED_PENDING_INDEPENDENT_CPU_QUALIFICATION',**binding,
        initial_state=initial_ref,checkpoints=checkpoint_refs,fixed_FIT_endpoints=endpoint_metrics,
        completed_epoch=24,parameters_per_arm=16547,tensors_per_arm=4,updates_per_arm=final_updates,
        training_point_exposures_per_arm=sampled_points,empty_image_visits=empty_images,images_per_epoch=741,
        actual_cached_points=cache['total_sample_points'],nonempty_cache_images=cache['nonempty_images'],
        training_seed=202610140161,order_seed=202610140160001,point_seed_base=202610140160000,matching_loss_seed_base=202610122001,
        weighted_CE='mean(weight*CE), no sumweights normalization',fixed_refiner_has_seen_FIT=True,
        label_generation_cost_not_ignored=True,teacher_loaded=False,PR_head_loaded=False,native_L_loaded=False,
        DEV_loaded=False,TEST_loaded=False,AP_evaluated=False,task_efficacy_established=False,innovation_established=False,
        paired_initial_state_exact=True,global_RNG_preserved=True,independent_CPU_qualified=False,
        current_owned_GPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),elapsed_seconds=time.monotonic()-started,
        peak_allocator_bytes=peak,cleanup=cleanup,actual_imported_files=imported,
        evidence_limits='Fixed shared qualified FIT cache and24epochs; training/weightedFIT statistics are not AP or scientific success. '
            'ROOT original held actual0 and full SOURCE closure plus independent CPU training qualification remain required.'))
    print(json.dumps(dict(status='N16_PAIRED_SCOUT_FIXED24_TRAINED_PENDING_INDEPENDENT_CPU_QUALIFICATION',report=reference(output/'REPORT.json'))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args()
    require(args.owner_execute,'UNBOUND_DRAFT: ROOT alone binds qualified cache+full actual ancestry and owns one attempt')
    run(args.spec)

