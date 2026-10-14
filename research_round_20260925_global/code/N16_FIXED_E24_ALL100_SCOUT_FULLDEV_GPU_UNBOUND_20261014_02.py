"""UNBOUND_DRAFT: complete N16 fixed E24/curve E8 DEV GPU evaluation entry.
NOT SCIENTIFICALLY RUN. No original/live SCI/data/state/source chain was read
in development. RAW-first source prefix is copied from the PUBLISHED ordinary
N16 endpoint CPU root with NEW evaluation schemas/min normal109. Actual train
GPU/CPU0 and ROOT result fields plus real budgets MUST be supplied by ROOT.
Reuses qualified original teacher.test_step native packets; no native forward.
Fixed all187/all100 original class/S0/order and native/strong reference exact;
no winner or new scientific conclusion. Captures stream per2048-point chunk,
never concatenate full-C256D F or hold all100 policy masks. CPU proof pending.
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


TRAIN_SCHEMA='ROOT-N16-fair-paired-scout-training-GPU-spec-v1'
TRAIN_CPU_SCHEMA='ROOT-N16-fair-paired-scout-training-endpoint-CPU-spec-v1'
REPORT_SCHEMA='N16-fixed-E24-all100-scout-fullDEV-GPU-report-v1'
STATUS='N16_FIXED_E24_ALL100_SCOUT_GPU_COMPLETE_PENDING_INDEPENDENT_CPU_PROOF'
ENDPOINTS=(8,24)
SCOUT_SHAPES={'hidden.weight':(32,513),'hidden.bias':(32,), 'output.weight':(3,32),'output.bias':(3,)}
ROOT_FIELDS=('training_GPU_source','training_GPU_spec','training_GPU_report','training_GPU_RAW','training_GPU_admission_refs',
    'training_CPU_source','training_CPU_spec','training_CPU_qualification','training_CPU_RAW','training_CPU_admission_refs',
    'ROOTtrain_result',
    'scouter_source','selector_eval_body_source','selector_cpu_proof_source','labels_source',
    'scout_checkpoints','capture_budget','strong_fixed_reference_predictions','strong_fixed_carrier_control_name')


def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],'Actual1800-second full evaluation deadline exceeded; no retry')


def nested(value,path):
    require(type(path) is list and path and all(type(p) in (str,int) for p in path),
            'NEEDS_ROOT_BINDING_FIELDS: exact nonempty actual JSON field path required')
    for part in path:
        value=value[part]
    return value


def source_gate(path,spec,cpu=False):
    missing=[key for key in ROOT_FIELDS if key not in spec or spec[key] in (None,'NEEDS_ROOT_BINDING_FIELDS')]
    require(not missing,'NEEDS_ROOT_BINDING_FIELDS: '+','.join(missing))
    require(spec['selector_mode']=='reuse' and spec['normal_RAW_count']>=109+int(cpu),
            'Completed original trainingGPU/CPU ancestry and exact reuse selector required')
    gate=inherited_source_gate(path,spec,cpu=cpu)
    gate['context']['selector_mode']='reuse'
    prior,prior_q=gate['lineage'].prior_gate(gate['context'],gate['closure'],gate['owners'])
    selector=gate['lineage'].selector_parent_gate(gate['context'],gate['closure'],gate['owners'])
    require(prior['teacher_checkpoint']==gate['cache_report']['teacher_checkpoint'], 'Original FIT/DEV teacher provenance differs')
    gate.update(prior=prior,prior_q=prior_q,packets=prior_q['packet_refs'],selector=selector)
    for key in ROOT_FIELDS:
        value=spec[key]
        if type(value) is dict and set(value)=={'path','bytes','sha256'}:closed(gate['closure'],value)
    training_gate(spec,gate,cpu=cpu)
    return gate


def training_gate(spec,gate,cpu=False):
    """Actual original GPU0/CPU0 joins precede any trained checkpoint loading."""
    closure,context,owners=gate['closure'],gate['context'],gate['owners']
    parent=read(closed(closure,spec['training_GPU_spec']))
    report=read(closed(closure,spec['training_GPU_report']))
    cpu_spec=read(closed(closure,spec['training_CPU_spec']))
    q=read(closed(closure,spec['training_CPU_qualification']))
    gi=gate['lineage'].owner_ref_join(context,owners,spec['training_GPU_RAW'],spec['training_GPU_admission_refs'],
        spec['training_GPU_source'],spec['training_GPU_spec'],'0')
    ci=gate['lineage'].owner_ref_join(context,owners,spec['training_CPU_RAW'],spec['training_CPU_admission_refs'],
        spec['training_CPU_source'],spec['training_CPU_spec'],'')
    require((gi,ci)==(spec['normal_RAW_count']-2-int(cpu),spec['normal_RAW_count']-1-int(cpu)) and
        parent['RAW_inputs']==spec['RAW_inputs'][:gi] and cpu_spec['RAW_inputs']==spec['RAW_inputs'][:ci] and
        parent['normal_RAW_count']==gi and cpu_spec['normal_RAW_count']==ci,
        'Actual original trainingGPU0/CPU0 latest chronological prefixes required')
    require(parent['schema']==TRAIN_SCHEMA and cpu_spec['schema']==TRAIN_CPU_SCHEMA and
        parent['entry_source']==report['source']==spec['training_GPU_source'] and report['spec']==spec['training_GPU_spec'] and
        report['GPU_admission']==spec['training_GPU_admission_refs'] and report['phase']==parent['phase'] and
        Path(parent['GPU_admission_file']).resolve(strict=True)==Path(spec['training_GPU_admission_refs']['path']).resolve(strict=True),
        'Completed original GPU training bindings differ')
    require(q['schema']=='N16-paired-scout-training-endpoint-CPU-qualification-v1' and
        q['status']=='N16_PAIRED_SCOUT_E8_E24_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0' and
        q['source']==cpu_spec['entry_source']==spec['training_CPU_source'] and q['spec']==spec['training_CPU_spec'] and
        q['CPU_admission']==spec['training_CPU_admission_refs'] and q['phase']==cpu_spec['phase'] and
        Path(cpu_spec['CPU_admission_file']).resolve(strict=True)==Path(spec['training_CPU_admission_refs']['path']).resolve(strict=True),
        'Independent CPU training-Q actual binding differs')
    correspondence=dict(training_GPU_source='training_GPU_source',training_GPU_spec='training_GPU_spec',
        training_GPU_report='training_GPU_report',training_GPU_RAW='training_GPU_RAW',training_GPU_admission='training_GPU_admission_refs')
    for key,target in correspondence.items():require(q[key]==spec[target],'CPUQ binds different original training '+key)
    require(report['schema']=='N16-fair-paired-scout-training-GPU-report-v1' and
        report['status']=='N16_PAIRED_SCOUT_FIXED24_TRAINED_PENDING_INDEPENDENT_CPU_QUALIFICATION' and
        report['completed_epoch']==24 and report['parameters_per_arm']==q['parameters_per_arm']==16547 and
        report['tensors_per_arm']==q['tensors_per_arm']==4 and q['epochs']==[8,24] and
        report['paired_initial_state_exact'] is True and report['global_RNG_preserved'] is True and
        report['AP_evaluated'] is False and report['teacher_loaded'] is False and report['PR_head_loaded'] is False and
        q['qualified'] is False and q['ROOT_actual_CPU0_verified'] is False and q['current_owned_CPU_terminal_self_certified'] is False and
        q['AP_evaluated'] is False and q['task_efficacy_established'] is False and q['innovation_established'] is False,
        'CPU self-report is only endpoint checks; ROOT actual0 is separately mandatory, never AP success')
    require(set(report['checkpoints'])==set(q['checkpoints'])==set(spec['scout_checkpoints'])=={'8','24'} and
        report['checkpoints']==q['checkpoints']==spec['scout_checkpoints'] and
        parent['scouter_source']==report['scouter_source']==q['scouter_source']==spec['scouter_source'] and
        parent['locked_protocol']==report['locked_protocol']==q['locked_protocol']==spec['locked_protocol'],
        'Both fixed8/24 checkpoints/canonical scout/protocol must be bound, no DEV winner')
    require(type(q['checks']) is dict and q['checks'] and all(value is True for value in q['checks'].values()),'Every declared endpoint CPU check required')
    for ancestor in (parent,cpu_spec):
        require(all(closure.get(ref['path'])==ref for ref in ancestor['sources']) and
            all(spec['terminal_identity_pins'].get(key)==value for key,value in ancestor['terminal_identity_pins'].items()),
            'Complete training original SOURCE/pin closure differs')
    for ref in list(report['checkpoints'].values())+[report['initial_state'],report['start'],report['trace'],q['details']]:closed(closure,ref)
    actual=read(closed(closure,spec['ROOTtrain_result']))
    expect=dict(schema='ROOT-N16-paired-scout-training-qualified-result-v1',status='ROOT_N16_PAIRED_SCOUT_E8_E24_FULLY_QUALIFIED_FOR_DEV',
        GPU_report=spec['training_GPU_report'],GPU_spec=spec['training_GPU_spec'],GPU_source=spec['training_GPU_source'],
        GPU_RAW=spec['training_GPU_RAW'],GPU_admission=spec['training_GPU_admission_refs'],
        CPU_Q=spec['training_CPU_qualification'],CPU_spec=spec['training_CPU_spec'],CPU_source=spec['training_CPU_source'],
        CPU_RAW=spec['training_CPU_RAW'],CPU_admission=spec['training_CPU_admission_refs'],locked_protocol=spec['locked_protocol'],
        CPU_endpoint_numeric_protocol=q['CPU_endpoint_numeric_protocol'],epochs=[8,24],checkpoints=report['checkpoints'],
        initial_state=report['initial_state'],updates_per_arm={'UTILITY':17784,'ERROR':17784},parameters_per_arm=16547,tensors_per_arm=4,
        fully_qualified=True,GPU_actual_exit_code=0,CPU_actual_exit_code=0,DEV_evaluation_eligible=True,
        checks=q['checks'],GPU_pointwise_logits_qualified=False,selected_indices_qualified=False,optimizer_trajectory_replayed=False,AP_evaluated=False)
    for key,value in expect.items():
        found=actual[key]
        require(type(found) is type(value) and found==value,'ROOT original actual0/frozen non-efficacy result differs: '+key)
    gate['training']=dict(parent=parent,report=report,CPU_spec=cpu_spec,CPU_Q=q,ROOT_result=actual)
    return gate['training']


def model_load_cpu(torch,spec,gate):
    require(not torch.cuda.is_initialized(),'All trained model loads must begin cold CUDA')
    outside=torch.get_rng_state().clone()
    heads,old_states,old_proofs=gate['evaluation'].load_heads(torch,gate['context'],gate)
    strong=heads['LOG_PR'];strong_state=old_states['LOG_PR']
    require(len(strong_state)==8 and sum(t.numel() for t in strong_state.values())==40497 and
        not list(strong.named_buffers()),'Only original LOG_PR8/40497, no memory/new8tensor reinterpretation')
    strong.eval().requires_grad_(False)
    del heads,old_states
    scout=load(spec['scouter_source'],'_N16_fixed_endpoint_scout_factory')
    evalbody=load(spec['selector_eval_body_source'],'_N16_fixed_selector_eval_body')
    labels=load(spec['labels_source'],'_N16_frozen_canonical_FP64_selector')
    require(scout.PARAMETERS==16547 and scout.STATE_SHAPES==SCOUT_SHAPES and evalbody.POINT_CHUNK==2048 and evalbody.ABSOLUTE_TOLERANCE==1e-4,
        'Fixed scout and evaluation algorithm ABI differs')
    models,states={},{'LOG_PR':strong_state}
    report=gate['training']['report'];prefix={}
    for epoch in ENDPOINTS:
        cp=torch.load(str(closed(gate['closure'],spec['scout_checkpoints'][str(epoch)])),map_location='cpu')
        require(cp['schema']=='N16-fair-paired-scout-training-checkpoint-v1' and cp['parameters_per_arm']==16547 and
            cp['tensors_per_arm']==4 and cp['source']==spec['training_GPU_source'] and cp['spec']==spec['training_GPU_spec'] and
            cp['GPU_admission']==spec['training_GPU_admission_refs'] and cp['phase']==report['phase'] and
            cp['initial_state']==report['initial_state'] and cp['fixed_FIT_endpoint']==report['fixed_FIT_endpoints'][str(epoch)],
            'Frozen actual endpoint checkpoint provenance differs')
        snapshot=cp['snapshot'];require(snapshot['completed_epoch']==epoch and set(snapshot['models'])=={'UTILITY','ERROR'},'Fixed trained snapshot endpoints required')
        prefix[epoch]=dict(history=snapshot['history'],orders=snapshot['epoch_orders'])
        for arm in ('UTILITY','ERROR'):
            state=snapshot['models'][arm]
            require(isinstance(state,dict) and set(state)==set(SCOUT_SHAPES) and
                all(t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==SCOUT_SHAPES[name] and
                    bool(torch.isfinite(t).all()) for name,t in state.items()) and sum(t.numel() for t in state.values())==16547,
                'Actual four FP32 scout tensors must be distinct from original PR state')
            head=scout.make_scout(torch,202610140161);head.load_state_dict(state,strict=True);head.eval().requires_grad_(False)
            models[(epoch,arm)]=head;states[(epoch,arm)]={name:t.detach().clone() for name,t in state.items()}
        del cp,snapshot
    require(prefix[24]['history'][:8]==prefix[8]['history'] and prefix[24]['orders'][:8]==prefix[8]['orders'] and
        torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized(),'Fixed E8 exact prefix/private construction changed RNG/CUDA')
    return strong,models,states,dict(original_PR=old_proofs,scout_shapes=SCOUT_SHAPES,parameters_per_scout=16547,
        checkpoint_refs=spec['scout_checkpoints']),evalbody,labels


def policy_names():
    return ('native_official','strong_fixed_E24_Omega')+tuple(
        'E%d_%s_%s'%(epoch,arm,mode) for epoch in (24,8) for mode in ('atmost','exact') for arm in ('UTILITY','ERROR'))+(
        'E24_DENSE_ERROR_FLIP_atmost','E24_MAGNET_atmost')


def tensor_bytes(torch,value):
    if isinstance(value,torch.Tensor):
        require(value.device.type=='cpu' and not value.requires_grad and value.layout==torch.strided and
            value.dtype in (torch.float32,torch.int64,torch.bool,torch.uint8) and bool(torch.isfinite(value).all()),'Capture tensors must be actual finite CPU copies')
        return value.numel()*value.element_size()
    if type(value) is dict:return sum(tensor_bytes(torch,v) for v in value.values())
    if type(value) in (tuple,list):return sum(tensor_bytes(torch,v) for v in value)
    return 0


def candidate_engine(np,cv2):
    def rectangular_dilation(P,*,structure,iterations,border_value):
        require(P.dtype==np.bool_ and P.ndim==2 and structure.shape==(17,17) and
            bool(np.all(structure)) and iterations==1 and border_value==0,'Only original centered17/constant0 binary dilation permitted')
        if not bool(P.any()):return np.zeros(P.shape,dtype=np.bool_)
        return cv2.dilate(np.ascontiguousarray(P,dtype=np.uint8),np.ones((17,17),dtype=np.uint8),
            anchor=(8,8),iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
    return types.SimpleNamespace(binary_dilation=rectangular_dilation)


def firstpass_counts(torch,np,engine,evalbody,gate,coco,ids,rows,process,check,RSS_cap):
    """Read closed original selector L only; no scout/PR/CUDA/native forward."""
    started=time.monotonic();measured=[]
    for index,row in enumerate(rows):
        check();require(process.memory_info().rss<RSS_cap,'Firstpass actual RSS cap exceeded')
        selector=gate['selector']['manifest']['rows'][index]
        packet=torch.load(str(closed(gate['closure'],selector['carrier'])),map_location='cpu')
        L=packet['native_L'];N=coco.imgs[ids[index]]['height']*coco.imgs[ids[index]]['width']
        require(L.device.type=='cpu' and L.dtype==torch.float32 and L.shape==(100,coco.imgs[ids[index]]['height'],coco.imgs[ids[index]]['width']) and
            bool(torch.isfinite(L).all()) and packet['dataset_index']==index and packet['image_id']==ids[index] and
            packet['natural_query_ids']==list(range(100)) and packet['L_digest']==selector['L_digest'],
            'Firstpass must use actual closed natural100 original selector nativeL')
        union=np.zeros(L.shape[1:],dtype=np.bool_);C_counts=[];K_counts=[];tick=time.monotonic()
        for q in range(100):
            _,_,C,idx,K=evalbody.domains_from_native(np,engine,L[q].numpy())
            union|=C;C_counts.append(len(idx));K_counts.append(K)
        require(K_counts==row['band_counts_all100'] and process.memory_info().rss<RSS_cap,'Firstpass physical Omega/RSS gate differs')
        measured.append(dict(dataset_index=index,image_id=ids[index],N=N,C_counts=C_counts,K_counts=K_counts,
            sumC=sum(C_counts),sumK=sum(K_counts),U=int(np.count_nonzero(union)),maximum_query_C=max(C_counts),
            C_seconds=time.monotonic()-tick,RSS_bytes=process.memory_info().rss))
        del packet,L,union,C,idx,K
    require(not torch.cuda.is_initialized(),'Actual firstpass must precede all CUDA/scout/PR')
    return dict(rows=measured,totalpixels=sum(r['N'] for r in measured),sumC_all100=sum(r['sumC'] for r in measured),
        sumK_all100=sum(r['sumK'] for r in measured),sum_union=sum(r['U'] for r in measured),
        maxC=max(r['maximum_query_C'] for r in measured),seconds=time.monotonic()-started,
        maximum_firstpass_RSS_bytes=max(r['RSS_bytes'] for r in measured),actual_selector_values_used=True,
        engine='exact-CV2-17rect-anchor8-borderconstant0-emptyP',frozen_evalbody_changed=False)


def budget_gate(spec,coco,ids,rows,firstpass,psutil,process,output):
    b=spec['capture_budget']
    keys={'total_original_pixels','total_C_points_upper_bound','total_K_points','maximum_query_C_points',
        'maximum_total_raw_capture_payload_bytes','maximum_serialized_capture_bytes','maximum_capture_files',
        'minimum_free_output_bytes','required_available_RAM_bytes','maximum_process_RSS_bytes','auxiliary_output_budget_bytes',
        'serialized_chunk_overhead_bytes','serialized_query_core_overhead_bytes','serialized_shared_image_overhead_bytes'}
    require(type(b) is dict and set(b)==keys and all(type(v) is int and v>0 for v in b.values()),
        'NEEDS_ROOT_BINDING_FIELDS: concrete capture/disk/RSS upper bounds before CUDA')
    Ns=[coco.imgs[i]['height']*coco.imgs[i]['width'] for i in ids]
    T=sum(sum(row['band_counts_all100']) for row in rows)
    require(b['total_original_pixels']==sum(Ns) and b['total_K_points']==T and
        T<=b['total_C_points_upper_bound']<=100*sum(Ns),
        'ROOT C bound must follow real closed geometry/K, not invented scientific counts')
    C=firstpass['sumC_all100'];U=firstpass['sum_union'];N=sum(Ns)
    require(C<=b['total_C_points_upper_bound'] and firstpass['maxC']<=b['maximum_query_C_points'] and
        firstpass['totalpixels']==N and firstpass['sumK_all100']==T,'Actual firstpass must fit prelocked Root C/N/K bounds')
    # F/e/L once per image, four independent scout C outputs once each.
    # Atmost/exact and dense use saved actual logits; no cross-model projection.
    # Cidx8C shared + fullC action/idx16C + four scout logits/idx20C=104C.
    mask_bytes=1200*sum((n+7)//8 for n in Ns)
    raw_bound=1032*U+400*N+104*C+240*T+mask_bytes+187*102400
    nc_total=nk_total=core_ref_total=max_ref_count=0
    for image in firstpass['rows']:
        for cq,kq in zip(image['C_counts'],image['K_counts']):
            nc=(cq+2047)//2048 if kq else 0;nk=(kq+2047)//2048
            nc_total+=nc;nk_total+=nk
            # two epochs x(atmost:4nc+2nk, exact:3nc+2nk).
            core_ref_total+=14*nc+8*nk;max_ref_count=max(max_ref_count,4*nc+2*nk)
    chunk_bound=5*nc_total+9*nk_total
    core_files=4*18700;shared_files=187
    # Variable physical-ref metadata is charged per ACTUAL firstpass support,
    # rather than charging maxC references to every query. Per-file remains capped.
    core_overhead_bound=core_files*65536+8192*core_ref_total
    overhead=(chunk_bound*b['serialized_chunk_overhead_bytes']+core_overhead_bound+
        shared_files*b['serialized_shared_image_overhead_bytes'])
    largest_shared=max(1032*r['U']+400*r['N']+8*r['sumC']+102400 for r in firstpass['rows'])
    largest_image=max(1032*r['U']+400*r['N']+104*r['sumC']+240*r['sumK']+1200*((r['N']+7)//8)+102400 for r in firstpass['rows'])
    require(raw_bound+overhead<=72*1024**3 and b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']<=72*1024**3 and
        b['maximum_process_RSS_bytes']<=16*1024**3 and largest_image<=1024**3 and
        firstpass['maximum_firstpass_RSS_bytes']<b['maximum_process_RSS_bytes'] and
        b['maximum_total_raw_capture_payload_bytes']>=raw_bound and
        b['serialized_chunk_overhead_bytes']>=65536 and b['serialized_query_core_overhead_bytes']>=65536+8192*max_ref_count and
        b['serialized_shared_image_overhead_bytes']>=1048576 and
        b['maximum_serialized_capture_bytes']>=raw_bound+overhead and
        b['maximum_capture_files']>=chunk_bound+4*18700+187 and
        b['minimum_free_output_bytes']>=b['maximum_serialized_capture_bytes']+b['auxiliary_output_budget_bytes']+16*1024**3 and
        psutil.disk_usage(str(output.parent)).free>=b['minimum_free_output_bytes'] and
        b['required_available_RAM_bytes']>=16*1024**3 and psutil.virtual_memory().available>=b['required_available_RAM_bytes'] and
        process.memory_info().rss+2*largest_shared+max(Ns)*8+8*1024**2<b['maximum_process_RSS_bytes'],
        'Concrete conservative event/capture/serialization/disk/RAM budget not admitted')
    return dict(prelocked=b,raw_payload_upper_bound=raw_bound,serialized_upper_bound=raw_bound+overhead,
        chunk_files_upper_bound=chunk_bound,largest_shared_image_tensor_bytes=largest_shared,
        maximum_perimage_total_raw_capture_bytes=largest_image,query_core_ref_metadata_upper_bound=8192*core_ref_total,
        raw_components=dict(F_and_union_indices=1032*U,native_L=400*N,all100_full_masks_exact_packbits=mask_bytes,C_indices_and_four_scout_logits_and_common_action=104*C,
            K_bounded_selected_PR_and_selected_indices=240*T,e100=187*102400),
        actual_firstpass=firstpass,free_disk_before_bytes=psutil.disk_usage(str(output.parent)).free,
        free_disk_margin_bytes=psutil.disk_usage(str(output.parent)).free-(raw_bound+overhead+b['auxiliary_output_budget_bytes']),
        available_RAM_before_bytes=psutil.virtual_memory().available,process_RSS_before_bytes=process.memory_info().rss,
        max_perimage_process_RSS_cap=b['maximum_process_RSS_bytes'],standalone_speed_certified=False)


class ChunkCaptureSink:
    """One actual CPU chunk per file; exact readback; no whole-C256D F concat."""
    def __init__(self,torch,root,spec,check,process,binding):
        self.torch,self.root,self.spec,self.check,self.process,self.binding=torch,Path(root),spec,check,process,binding
        self.refs=[];self.raw_bytes=0;self.disk_bytes=0
    def write(self,name,value):
        self.check();require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,'Safe fresh capture locator required')
        target=self.root/name;require(not target.exists(),'Never overwrite any capture')
        packet=dict(schema='N16-actual-GPU-streamed-capture-member-v1',binding=self.binding,value=value)
        size=tensor_bytes(self.torch,packet);b=self.spec['capture_budget']
        require(self.raw_bytes+size<=b['maximum_total_raw_capture_payload_bytes'] and
            len(self.refs)<b['maximum_capture_files'] and self.process.memory_info().rss+size<b['maximum_process_RSS_bytes'],
            'Actual capture/process/readback bound exceeded')
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:self.torch.save(packet,stream)
        replay=self.torch.load(str(target),map_location='cpu')
        require(tree_equal(self.torch,packet,replay),'Exact actual chunk saved readback differs')
        del replay
        ref=reference(target)
        kind=value.get('schema') if type(value) is dict else None
        overhead_key='serialized_query_core_overhead_bytes' if kind=='N16-actual-GPU-query-core-and-physical-chunk-refs-v2' else (
            'serialized_shared_image_overhead_bytes' if kind=='N16-actual-GPU-image-shared-inputs-v1' else 'serialized_chunk_overhead_bytes')
        per_file_overhead=b[overhead_key]
        if kind=='N16-actual-GPU-query-core-and-physical-chunk-refs-v2':
            ref_count=len(value['common_C_PR_chunks'])+sum(len(v['C_scout_chunks'])+len(v['selected_PR_chunks']) for v in value['controls'].values())
            per_file_overhead=min(per_file_overhead,65536+8192*ref_count)
        require(ref['bytes']<=size+per_file_overhead and ref['bytes']<=2*1024**3,'Actual per-file serialization/2GiB packet cap exceeds prelock')
        require(len(ref['path'].encode('utf-8'))<=4096,'Physical capture path exceeds metadata bound')
        self.refs.append(ref);self.raw_bytes+=size;self.disk_bytes+=ref['bytes']
        require(self.disk_bytes<=b['maximum_serialized_capture_bytes'],'Serialized capture byte budget exceeded')
        return ref


def tree_equal(torch,left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and (same_float_bits(torch,left,right) if left.dtype==torch.float32 else torch.equal(left,right))
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(tree_equal(torch,v,right[k]) for k,v in left.items())
    if type(left) in (tuple,list):return type(right) is type(left) and len(left)==len(right) and all(tree_equal(torch,a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right


def original_image(torch,np,masks,renderer,kernels,reader,n10,coco,packet,row,index,ref,native,gate,spec,selector_module):
    fields,identity,qids=n10.packet_check(torch,reader,coco,packet,row,index,ref,gate)
    require([v['query_id'] for v in native]==qids,'Original all100 exported order differs')
    with torch.no_grad():
        F,e=fields['F256'].to('cuda:0'),fields['e0'].to('cuda:0')
        M,C=fields['M200'].to('cuda:0'),fields['final_class_logits'].to('cuda:0')
        require(torch.equal(torch.einsum('bqc,bchw->bqhw',e,F),M),'Original captured M=eF must be exact')
        top,order=kernels.order(torch,C,types.SimpleNamespace(num_classes=1))
        require(order.cpu().tolist()==qids,'Original all100 sorted=False order differs')
        L=renderer.render_original(torch,M,identity)
        require(L.dtype==torch.float32 and L.shape==(100,)+tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),'Actual natural100 GPU original L required')
        binary,quality,scores=kernels.quality(torch,L[order],top)
        baseline=[]
        for ordinal,q in enumerate(qids):
            item=dict(image_id=row['image_id'],query_id=q,category_id=1,score=float(scores[ordinal].item()),
                segmentation=n10.segmentation(np,masks,binary[ordinal].cpu().numpy().astype(np.uint8)))
            require(item==native[ordinal],'Original S0/class/query-order/native RLE must reconstruct exactly')
            if not bool(binary[ordinal].any()):require(item['score']==0.,'Original empty-mask score must remain0')
            baseline.append(item)
        selector=selector_module.selector_packet(torch,spec,gate,index,row,ref,identity,qids,L,None)
        counts=[int((L[q].abs()<=2).sum().item()) for q in range(100)]
        require(counts==row['band_counts_all100'],'Original natural100 Omega counts differ')
        F_original=n10.original_feature(torch,renderer,F,identity)
        require(F_original.shape==(256,)+tuple(identity['original_shape']) and e.shape==(1,100,256),'Exact original F/e100 ABI required')
        del F,M,C,top,order,binary,quality,scores,fields
    return F_original,e[0],L,baseline,selector,qids,identity


def export_mask(np,masks,n10,baseline_row,mask,cost_tracker=None):
    before=time.monotonic()
    require(mask.dtype==np.bool_ and mask.ndim==2,'Actual strict-sign Boolean mask required')
    result=dict(baseline_row);result['segmentation']=n10.segmentation(np,masks,mask.astype(np.uint8))
    require({k:v for k,v in result.items() if k!='segmentation'}=={k:v for k,v in baseline_row.items() if k!='segmentation'},'Never update class/S0/order')
    if cost_tracker is not None:
        cost_tracker['RLE_export_calls']+=1
        cost_tracker['RLE_export_seconds']+=time.monotonic()-before
    return result


def same_float_bits(torch,a,b):
    return a.shape==b.shape and a.dtype==b.dtype==torch.float32 and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


def shared_actual_image(torch,np,engine,evalbody,F,e,L,index,row,identity,selector,ref,qids,sink,measured,check):
    """Capture only the actual GPU spatial union once; no CPU renderer."""
    tick=time.monotonic();native=L.detach().cpu().clone().contiguous()
    indices=[];union=np.zeros(native.shape[1:],dtype=np.bool_);counts=[];Ks=[]
    for q in range(100):
        _,_,C,idx,K=evalbody.domains_from_native(np,engine,native[q].numpy())
        indices.append(torch.from_numpy(idx));union|=C;counts.append(len(idx));Ks.append(K)
    union_idx=np.flatnonzero(union).astype(np.int64)
    require(counts==measured['C_counts'] and Ks==measured['K_counts'] and len(union_idx)==measured['U'],
        'Actual GPU nativeL C/K/union must match admitted firstpass exactly')
    features=torch.empty((len(union_idx),256),dtype=torch.float32,device='cpu')
    with torch.no_grad():
        for first in range(0,len(union_idx),2048):
            check();ix=torch.from_numpy(union_idx[first:first+2048]).to(F.device)
            block=F[:,ix//L.shape[2],ix%L.shape[2]].t().contiguous()
            features[first:first+len(ix)].copy_(block.detach().cpu());del ix,block
    shared=dict(schema='N16-actual-GPU-image-shared-inputs-v1',dataset_index=index,image_id=row['image_id'],identity=identity,
        native_packet=ref,selector=selector,natural_query_ids=list(range(100)),native_export_query_ids=qids,
        union_flat_indices=torch.from_numpy(union_idx),F_unique=features,e0=e.detach().cpu().clone().contiguous(),
        native_L=native,C_indices=indices,C_counts=counts,K_counts=Ks,U=len(union_idx),
        actual_GPU_F_captured=True,CPU_F_rerendered=False)
    shared_ref=sink.write('i%03d/SHARED_ACTUAL_INPUTS.pth'%index,shared)
    return shared,shared_ref,dict(C_construction_and_shared_input_capture_seconds=time.monotonic()-tick,U=len(union_idx),
        sumC=sum(counts),sumK=sum(Ks),shared_tensor_bytes=tensor_bytes(torch,shared))


def strip_event_to_shared(torch,np,event,shared,shared_ref):
    """Require observed GPU input bytes==shared before discarding duplicates."""
    q=event['query_id'];idx=event['flat_indices'];flat=idx.numpy()
    positions=np.searchsorted(shared['union_flat_indices'].numpy(),flat)
    require(bool(np.all(positions<len(shared['union_flat_indices']))) and np.array_equal(shared['union_flat_indices'].numpy()[positions],flat),'Actual event outside captured union')
    expectedF=shared['F_unique'][torch.from_numpy(positions.astype(np.int64))]
    require(same_float_bits(torch,event['sampled_F'],expectedF) and same_float_bits(torch,event['e0'],shared['e0'][q]) and
        same_float_bits(torch,event['native_L'],shared['native_L'][q].reshape(-1)[idx]),
        'Do not deduplicate differing actual point inputs; F/e/L must be bitexact')
    result={key:value for key,value in event.items() if key not in ('sampled_F','e0','native_L')}
    result.update(shared_actual_image_inputs=shared_ref,actual_observed_F_e_L_bitexact_to_shared=True)
    return result


def evaluate_image(torch,np,ndimage,masks,F,e,L,baseline,qids,index,row,strong,models,evalbody,labels,n10,sink,shared,shared_ref,check):
    by_q={value['query_id']:value for value in baseline}
    exported={name:{} for name in policy_names() if name!='native_official'}
    members=[];diagnostics=[];export_cost=dict(RLE_export_calls=0,RLE_export_seconds=0.)
    with torch.no_grad():
        for q in range(100):
            check();native=shared['native_L'][q].numpy();P=native>0
            Cidx=shared['C_indices'][q].numpy();K=shared['K_counts'][q]
            common_cost=evalbody._cost();common_refs=[]
            def common_callback(event):
                compact=strip_event_to_shared(torch,np,event,shared,shared_ref)
                common_refs.append(sink.write('i%03d/q%03d/commonC_c%06d.pth'%(index,q,event['chunk_first']),compact))
            active=Cidx if K else np.empty(0,dtype=np.int64)
            common=evalbody._run_points(torch,np,F,e,L[q],q,active,strong,'PR',common_cost,common_callback,'COMMON_C_REFERENCE','shared','full_C_PR')
            omega=np.flatnonzero(np.abs(native)<=2).astype(np.int64);omega_cost=evalbody._cost();omega_refs=[]
            def omega_callback(event):
                compact=strip_event_to_shared(torch,np,event,shared,shared_ref)
                omega_refs.append(sink.write('i%03d/q%03d/strongOmega_c%06d.pth'%(index,q,event['chunk_first']),compact))
            original=evalbody._run_points(torch,np,F,e,L[q],q,omega,strong,'PR',omega_cost,omega_callback,'STRONG_FIXED_E24_OMEGA','atmost','original_Omega_PR')
            evalbody._qualify_subset(np,q,'STRONG_FIXED_E24_OMEGA',omega,original,common,Cidx)
            strong_mask=P.copy();strong_mask.ravel()[omega]=original['combined']>0
            exported['strong_fixed_E24_Omega'][q]=export_mask(np,masks,n10,by_q[q],strong_mask,export_cost)
            members.append(dict(query_id=q,kind='strong_fixed_Omega',point_chunks=omega_refs))
            scout_points={};scout_refs={};scout_cost={}
            for epoch in (24,8):
                for arm in ('UTILITY','ERROR'):
                    refs=[];cost=evalbody._cost()
                    def scout_callback(event):
                        compact=strip_event_to_shared(torch,np,event,shared,shared_ref);compact['scout_epoch']=epoch
                        refs.append(sink.write('i%03d/q%03d/e%d_%s_c%06d.pth'%(index,q,epoch,arm,event['chunk_first']),compact))
                    points=evalbody._run_points(torch,np,F,e,L[q],q,active,models[(epoch,arm)],'scout',cost,scout_callback,arm,'shared-atmost-exact','scout_C_logits')
                    scout_points[(epoch,arm)]=points;scout_refs[(epoch,arm)]=refs;scout_cost[(epoch,arm)]=cost
            Cmask=np.zeros(native.shape,dtype=np.bool_);Cmask.ravel()[Cidx]=True
            for epoch in (24,8):
                for mode in ('atmost','exact'):
                    controls=('UTILITY','ERROR','DENSE_ERROR_FLIP','MAGNET') if mode=='atmost' else ('UTILITY','ERROR')
                    outputs={};profile_cost={}
                    for control in controls:
                        cost=evalbody._cost();selection_start=time.monotonic()
                        if not K:selected=np.zeros(native.shape,dtype=np.bool_)
                        else:
                            if control=='UTILITY':scores=labels.utility_score(scout_points[(epoch,'UTILITY')]['logits'])
                            elif control=='ERROR':scores=labels.error_score(scout_points[(epoch,'ERROR')]['logits'])
                            elif control=='DENSE_ERROR_FLIP':scores=labels.dense_error_flip_score((common['native_L']>0)!=(common['combined']>0),scout_points[(epoch,'ERROR')]['logits'])
                            else:scores=labels.magnet_score(common['native_L'],common['residual'])
                            require(scores.dtype==np.float64 and np.isfinite(scores).all(),'Canonical actualGPU-input FP64 scores required')
                            fullscore=np.zeros(native.shape,dtype=np.float64);fullscore.ravel()[Cidx]=scores
                            cost['score_calls']=1;cost['score_seconds']=time.monotonic()-selection_start
                            tick=time.monotonic();selected=labels.choose_candidates(native,Cmask,fullscore,K,mode=mode)
                            cost['sorting_calls']=1;cost['sorting_seconds']=time.monotonic()-tick
                            del scores,fullscore
                        selected_idx=np.flatnonzero(selected).astype(np.int64);selected_refs=[]
                        if control in ('UTILITY','ERROR'):
                            def selected_callback(event):
                                compact=strip_event_to_shared(torch,np,event,shared,shared_ref);compact['scout_epoch']=epoch
                                selected_refs.append(sink.write('i%03d/q%03d/e%d_%s_%s_selected_c%06d.pth'%(index,q,epoch,mode,control,event['chunk_first']),compact))
                            subset=evalbody._run_points(torch,np,F,e,L[q],q,selected_idx,strong,'PR',cost,selected_callback,control,mode,'selected_PR')
                            evalbody._qualify_subset(np,q,control,selected_idx,subset,common,Cidx)
                            after=subset['combined'];del subset
                        else:after=common['combined'][np.searchsorted(Cidx,selected_idx)]
                        mask=P.copy();mask.ravel()[selected_idx]=after>0
                        require(np.array_equal(mask[~selected],P[~selected]),'Outside selected support must remain native exact')
                        if control in ('UTILITY','ERROR') or epoch==24:
                            policy='E%d_%s_%s'%(epoch,control,mode)
                            exported[policy][q]=export_mask(np,masks,n10,by_q[q],mask,export_cost)
                        refs=[] if control=='MAGNET' else scout_refs[(epoch,'UTILITY' if control=='UTILITY' else 'ERROR')]
                        outputs[control]=dict(actual_selected_indices=torch.from_numpy(selected_idx),
                            actual_mask_packbits=torch.from_numpy(np.packbits(mask.reshape(-1),bitorder='little')),
                            mask_shape=list(mask.shape),mask_bits_count=mask.size,mask_bitorder='little',
                            C_scout_chunks=refs,selected_PR_chunks=selected_refs,
                            logits_diagnostic_reuse=(control=='DENSE_ERROR_FLIP' or mode=='exact'))
                        profile_cost[control]=cost
                        del selected,selected_idx,mask,after
                    core=dict(schema='N16-actual-GPU-query-core-and-physical-chunk-refs-v2',query_id=q,evalmode=mode,epoch=epoch,
                        shared_actual_image_inputs=shared_ref,K=K,common_C_PR_chunks=common_refs,controls=outputs)
                    core_ref=sink.write('i%03d/q%03d/e%d_%s_QUERY_CORE.pth'%(index,q,epoch,mode),core)
                    members.append(dict(query_id=q,epoch=epoch,evalmode=mode,core=core_ref))
                    diagnostics.append(dict(query_id=q,epoch=epoch,evalmode=mode,C=len(Cidx),K=K,controls=profile_cost,
                        shared_diagnostic_runtime_only=True,standalone_total_time_measured=False,actual_GPU_logits_reused_for_atmost_exact_dense=True,
                        cross_model_projection_shared=False,E8_extra_strong_diagnostics_not_AP_policies=(epoch==8 and mode=='atmost')))
                    del outputs,core
            diagnostics.append(dict(query_id=q,profile='actual_shared_inputs_and_four_separate_model_forwards',common_C_cost=common_cost,
                original_Omega_cost=omega_cost,scout_cost={'E%d_%s'%key:value for key,value in scout_cost.items()},standalone_FPS_claimed=False))
            del common,original,scout_points,scout_refs,scout_cost,Cmask,P,native,strong_mask
    image={name:[values[q] for q in qids] for name,values in exported.items()}
    require(all(len(values)==100 for values in image.values()),'All fixed policies must export original natural100')
    return image,members,diagnostics,export_cost



def validate_shared_for_CPU(torch,np,cv2,shared):
    """Pure actual-input validation; ROOT authenticates physical refs/closure."""
    require(shared['schema']=='N16-actual-GPU-image-shared-inputs-v1' and
        shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False and
        shared['natural_query_ids']==list(range(100)) and sorted(shared['native_export_query_ids'])==list(range(100)),
        'Whole-image actual input provenance/natural100 required')
    L=shared['native_L'];F=shared['F_unique'];e=shared['e0'];union=shared['union_flat_indices']
    require(L.device.type==F.device.type==e.device.type==union.device.type=='cpu' and
        L.dtype==F.dtype==e.dtype==torch.float32 and L.ndim==3 and L.shape[0]==100 and min(L.shape[1:])>0 and
        e.shape==(100,256) and union.dtype==torch.int64 and union.ndim==1 and F.shape==(len(union),256) and
        all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in (L,F,e)) and
        shared['U']==len(union) and len(shared['C_indices'])==len(shared['C_counts'])==len(shared['K_counts'])==100,
        'Exact finite CPU shared FP32 shapes and actual union required')
    merged=np.zeros(L.shape[1:],dtype=np.bool_)
    for q in range(100):
        native=L[q].numpy();P=native>0;Omega=np.abs(native)<=2
        if bool(P.any()):
            dilation=cv2.dilate(np.ascontiguousarray(P,dtype=np.uint8),np.ones((17,17),dtype=np.uint8),
                anchor=(8,8),iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
            C=Omega|P|dilation
        else:C=Omega.copy()
        idx=np.flatnonzero(C).astype(np.int64);saved=shared['C_indices'][q]
        require(saved.device.type=='cpu' and saved.dtype==torch.int64 and saved.shape==(len(idx),) and
            np.array_equal(saved.numpy(),idx) and type(shared['C_counts'][q]) is int and shared['C_counts'][q]==len(idx) and
            type(shared['K_counts'][q]) is int and shared['K_counts'][q]==int(np.count_nonzero(Omega)),
            'Independent native FP32 rectangular C/K/all100 support mismatch')
        merged|=C
    require(np.array_equal(union.numpy(),np.flatnonzero(merged).astype(np.int64)),
        'Actual GPU feature union must equal independently recomputed all100 C union exactly')
    return dict(natural_queries=100,all_C_K_exact=True,union_mapping_exact=True,CPU_F_rerendered=False,
        actual_Source_REF_provenance_proven_here=False)


def decode_CPU_query_core(torch,np,core_packet,shared_packet,shared_ref,load_closed_physical,expected_binding,*,synthetic_only=False):
    """Rebuild one CPUproof packet solely from saved actual GPU input bytes.

    load_closed_physical(ref) is ROOT's closure/hash-authenticated physical .pth
    loader, returning the complete streamed wrapper. No virtual member refs,
    CPU renderer/scout forward or tolerance-based rank substitution is used.
    Validate the whole shared image once with validate_shared_for_CPU first.
    """
    require(type(expected_binding) is dict and expected_binding and callable(load_closed_physical),
        'ROOT authentic actual completed GPU bindings and physical ref loader required')
    def unwrap(packet):
        require(type(packet) is dict and set(packet)=={'schema','binding','value'} and
            packet['schema']=='N16-actual-GPU-streamed-capture-member-v1' and packet['binding']==expected_binding,
            'Every physical capture must share the closed original completed GPU report binding')
        return packet['value']
    core=unwrap(core_packet);shared=unwrap(shared_packet);q=core['query_id'];epoch=core['epoch'];mode=core['evalmode']
    require(core['schema']=='N16-actual-GPU-query-core-and-physical-chunk-refs-v2' and type(q) is int and 0<=q<100 and
        type(epoch) is int and epoch in ENDPOINTS and mode in ('atmost','exact') and core['shared_actual_image_inputs']==shared_ref and
        core['K']==shared['K_counts'][q] and shared['schema']=='N16-actual-GPU-image-shared-inputs-v1',
        'Exact current image/query/endpoint/mode binding required')
    input_cache={}
    def actual_mask(record):
        packed=record['actual_mask_packbits'];shape=list(shared['native_L'].shape[1:]);N=shape[0]*shape[1]
        require(packed.device.type=='cpu' and packed.dtype==torch.uint8 and packed.shape==((N+7)//8,) and
            record['mask_shape']==shape and type(record['mask_bits_count']) is int and record['mask_bits_count']==N and
            record['mask_bitorder']=='little','Exact actual full-mask packbits shape/count/little-bitorder required')
        bits=np.unpackbits(packed.numpy(),bitorder='little')
        require(not bool(bits[N:].any()),'Actual full-mask padding bits must remain zero')
        return torch.from_numpy(bits[:N].reshape(shape).astype(np.bool_))
    def blocks(refs,kind,control,stage,recorded_mode):
        require(type(refs) is list,'Only physical chunk ref lists accepted')
        result=[]
        for i,ref in enumerate(refs):
            require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'Real closed physical chunk ref required')
            compact=unwrap(load_closed_physical(ref));idx=compact['flat_indices']
            require(compact['query_id']==q and compact['control']==control and compact['stage']==stage and
                compact['evalmode']==recorded_mode and compact['point_chunk']==2048 and compact['chunk_first']==i*2048 and
                compact['shared_actual_image_inputs']==shared_ref and compact['actual_observed_F_e_L_bitexact_to_shared'] is True and
                compact['actual_input_scope']=='caller_actual_original_device_FP32_F_e_L' and compact['synthetic_only'] is synthetic_only and
                (stage not in ('scout_C_logits','selected_PR') or compact['scout_epoch']==epoch) and
                idx.device.type=='cpu' and idx.dtype==torch.int64 and idx.ndim==1 and 0<len(idx)<=2048,
                'Actual chunk cannot mix image/query/model endpoint, roles, mode, support or synthetic status')
            flat=idx.numpy();positions=np.searchsorted(shared['union_flat_indices'].numpy(),flat)
            require(bool(np.all(positions<len(shared['union_flat_indices']))) and
                np.array_equal(shared['union_flat_indices'].numpy()[positions],flat), 'Point outside actual captured union')
            # Reuse identical actual F blocks within this query; no entire-C F concat.
            key=flat.tobytes()
            if key not in input_cache:
                input_cache[key]=(shared['F_unique'][torch.from_numpy(positions.astype(np.int64))],shared['native_L'][q].reshape(-1)[idx])
            actualF,actualL=input_cache[key]
            block=dict(flat_indices=idx,sampled_F=actualF,e0=shared['e0'][q],native_L=actualL)
            if kind=='scout':block['scout_logits']=compact['scout_logits']
            else:block.update(GPU_raw_residual=compact['GPU_raw_residual'],GPU_combined=compact['GPU_combined'])
            result.append(block)
        return result
    expected_controls=('UTILITY','ERROR','DENSE_ERROR_FLIP','MAGNET') if mode=='atmost' else ('UTILITY','ERROR')
    require(set(core['controls'])==set(expected_controls),'All and only fixed mode controls required')
    controls={}
    for name in expected_controls:
        record=core['controls'][name]
        controls[name]=dict(C_scout_chunks=[] if name=='MAGNET' else blocks(record['C_scout_chunks'],'scout',
            'UTILITY' if name=='UTILITY' else 'ERROR','scout_C_logits','shared-atmost-exact'),
            selected_PR_chunks=blocks(record['selected_PR_chunks'],'PR',name,'selected_PR',mode),
            actual_selected_indices=record['actual_selected_indices'],actual_mask=actual_mask(record))
    return dict(schema='N16-all100-scout-selection-same-GPU-input-query-capture-v1',query_id=q,evalmode=mode,
        native_L=shared['native_L'][q],e0=shared['e0'][q],K=core['K'],C_flat_indices=shared['C_indices'][q],
        common_C_PR_chunks=blocks(core['common_C_PR_chunks'],'PR','COMMON_C_REFERENCE','full_C_PR','shared'),controls=controls)


def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);spec=read(path)
    gate=source_gate(path,spec);check=lambda:deadline(started,spec)
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh ROOT-owned output only')
    imports_start=time.monotonic()
    import psutil
    process=psutil.Process();require(process.cpu_affinity()==[0,1],'Actual CPU01 required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified coldCUDA runtime required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    strong,models,states,proofs,evalbody,labels=model_load_cpu(torch,spec,gate)
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0' and scipy.__version__=='1.10.1','Pinned actual numerical versions required')
    context=gate['context'];n10=load(context['N10_evaluation_source'],'_N16_original_N10_export')
    reader=load(context['reader_source'],'_N16_original_native_reader');renderer=load(context['renderer_source'],'_N16_original_renderer')
    official=load(context['COCO_source'],'_N16_original_official_COCO');selector_module=gate['evaluation']
    kernels=n10.original_export_kernels(context['export_source']);coco=COCO(str(closed(gate['closure'],context['DEV'])))
    ids=list(coco.imgs);require(len(ids)==187 and len(set(ids))==187 and ids==[r['id'] for r in coco.dataset['images']] and
        len(coco.anns)==380 and set(coco.cats)=={1},'Complete original DEV187/380/category1 required')
    native=read(closed(gate['closure'],context['native_reference_predictions']));population=reader.population(coco,native,ids)
    rows=read(closed(gate['closure'],gate['prior']['image_rows']))
    require(len(rows)==len(gate['packets'])==len(gate['selector']['manifest']['rows'])==187 and [r['image_id'] for r in rows]==ids and
        sum(r['raw_GT'] for r in rows)==380,'Canonical original all187 capture/selector population required')
    strong_reference=read(closed(gate['closure'],spec['strong_fixed_reference_predictions']))
    control=gate['carrier_report']['control_results'][spec['strong_fixed_carrier_control_name']]
    require(control['predictions']==spec['strong_fixed_reference_predictions'],'ROOT must name exact fixed701 E24 original strong control, no winner')
    reader.population(coco,strong_reference,ids)
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    cv2.setNumThreads(2);engine=candidate_engine(np,cv2)
    require(type(spec['capture_budget'].get('maximum_process_RSS_bytes')) is int and 0<spec['capture_budget']['maximum_process_RSS_bytes']<=16*1024**3,'Concrete firstpass RSS cap required')
    firstpass=firstpass_counts(torch,np,engine,evalbody,gate,coco,ids,rows,process,check,spec['capture_budget']['maximum_process_RSS_bytes'])
    admitted_budget=budget_gate(spec,coco,ids,rows,firstpass,psutil,process,output)
    imports_CPU_load_and_preCUDA_native_gate_seconds=time.monotonic()-imports_start
    require(not torch.cuda.is_initialized(),'RAW/Source/train/native/library/concrete budget gates precede CUDA')
    check();torch.cuda.init();require(torch.cuda.current_device()==0,'Only admitted GPU0')
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0);outside=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone())
    tick=time.monotonic();strong.to('cuda:0')
    for head in models.values():head.to('cuda:0')
    torch.cuda.synchronize(0);transfer_seconds=time.monotonic()-tick
    output.mkdir();(output/'CAPTURES').mkdir()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        training_GPU_report=spec['training_GPU_report'],training_CPU_qualification=spec['training_CPU_qualification'],
        ROOTtrain_result=spec['ROOTtrain_result'],scout_checkpoints=spec['scout_checkpoints'],locked_protocol=spec['locked_protocol'])
    sink=ChunkCaptureSink(torch,output/'CAPTURES',spec,check,process,binding)
    auxiliary_bytes=0
    def write_auxiliary(target,value):
        nonlocal auxiliary_bytes
        check();write(target,value);auxiliary_bytes+=target.stat().st_size
        require(auxiliary_bytes<=spec['capture_budget']['auxiliary_output_budget_bytes'] and
            sink.disk_bytes+auxiliary_bytes<=72*1024**3,'Actual auxiliary/full output byte budget exceeded')
    write_auxiliary(output/'START.json',dict(schema='N16-fixed-fullDEV-GPU-start-v1',**binding,model_proofs=proofs,capture_budget=admitted_budget,
        policies=policy_names(),main_epoch=24,curve_epoch=8,epoch_or_threshold_winner_selected=False,native_model_loaded=False,optimizer_created=False))
    predictions={name:[] for name in policy_names()};manifest_rows=[];image_times=[];body_capture_RLE_seconds=0.
    actual_RLE_export_seconds=0.;actual_RLE_export_calls=0;native_field_replay_seconds=0.
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index,(row,ref) in enumerate(zip(rows,gate['packets'])):
            check();require(process.memory_info().rss<spec['capture_budget']['maximum_process_RSS_bytes'],'Actual RSS cap')
            packet=torch.load(str(closed(gate['closure'],ref)),map_location='cpu');tick=time.monotonic()
            F,e,L,baseline,selector,qids,identity=original_image(torch,np,masks,renderer,kernels,reader,n10,coco,
                packet,row,index,ref,population[row['image_id']],gate,context,selector_module)
            native_field_replay_seconds+=time.monotonic()-tick
            require(selector==gate['selector']['manifest']['rows'][index],'Original exact frozen selector reuse differs')
            shared,shared_ref,shared_cost=shared_actual_image(torch,np,engine,evalbody,F,e,L,index,row,identity,selector,ref,qids,sink,firstpass['rows'][index],check)
            before=time.monotonic()
            image,members,diagnostics,export_cost=evaluate_image(torch,np,engine,masks,F,e,L,baseline,qids,index,row,strong,models,evalbody,labels,n10,sink,shared,shared_ref,check)
            body_capture_RLE_seconds+=time.monotonic()-before
            actual_RLE_export_seconds+=export_cost['RLE_export_seconds'];actual_RLE_export_calls+=export_cost['RLE_export_calls']
            predictions['native_official'].extend(baseline)
            for name,values in image.items():predictions[name].extend(values)
            diag_ref=output/('IMAGE_%03d_DIAGNOSTICS.json'%index);write_auxiliary(diag_ref,diagnostics)
            image_seconds=time.monotonic()-tick;image_times.append(image_seconds)
            manifest_rows.append(dict(dataset_index=index,image_id=row['image_id'],identity=identity,native_packet=ref,selector=selector,
                qids=qids,shared_actual_image_inputs=shared_ref,shared_capture_cost=shared_cost,
                query_profiles=members,diagnostics=reference(diag_ref),natural_queries=100))
            line=json.dumps(dict(dataset_index=index,image_id=row['image_id'],queries=100,seconds=image_seconds),allow_nan=False)+'\n'
            trace.write(line);trace.flush();auxiliary_bytes+=len(line.encode('utf-8'))
            require(auxiliary_bytes<=spec['capture_budget']['auxiliary_output_budget_bytes'],'Actual trace/auxiliary byte cap exceeded')
            del F,e,L,packet,baseline,image,members,diagnostics,export_cost,shared,shared_cost,selector,identity
            gc.collect();print(json.dumps(dict(evaluated_DEV_images=index+1,all100_queries=(index+1)*100)),flush=True)
    require(predictions['native_official']==native and predictions['strong_fixed_E24_Omega']==strong_reference and
        all(len(values)==18700 for values in predictions.values()),'Full native and strong fixed reference reconstruct EXACTLY for all18700')
    for values in predictions.values():
        reader.population(coco,values,ids)
        require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(native,values)),
            'All policy original score/class/query order must stay exact')
    for key,head in list(models.items())+[('LOG_PR',strong)]:
        require(all(torch.equal(t.detach().cpu(),states[key][name]) for name,t in head.state_dict().items()) and
            all(not p.requires_grad and p.grad is None for p in head.parameters()),'Frozen actual qualified states/grads changed')
    require(torch.equal(outside[0],torch.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),'Evaluation RNG changed')
    peak=torch.cuda.max_memory_allocated(0)
    strong.cpu()
    for head in models.values():head.cpu()
    models.clear();del head,models,strong,key
    gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup=dict(allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2,'Actual4GiB allocation/zero live CUDA cleanup required')
    results={};COCO_seconds=0.
    for name,values in predictions.items():
        check();target=output/(name+'_PREDICTIONS.json');write_auxiliary(target,values);before=time.monotonic()
        metrics=official.coco_metrics(coco,values,ids,COCOeval);COCO_seconds+=time.monotonic()-before
        require(metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100] and metrics['score_filter'] is None and
            metrics['original_GT_preserved'] is True and metrics['official_COCO_crowd_ignore_semantics'] is True,'Original official COCO12 convention differs')
        n10.stats_exact(metrics['stats'],metrics['stats'],'Finite defined COCO12/-1 required')
        results[name]=dict(predictions=reference(target),COCO=metrics,stats=metrics['stats'])
    n10.stats_exact(results['native_official']['stats'],gate['prior_q']['expected_E24_COCO_stats'],'Original native COCO12 must reconstruct')
    n10.stats_exact(results['strong_fixed_E24_Omega']['stats'],control['stats'],'Original strong fixed E24 COCO12 must reconstruct')
    write_auxiliary(output/'CAPTURE_MANIFEST.json',dict(schema='N16-streamed-GPU-all18700-point-capture-manifest-v1',**binding,images=187,queries=18700,
        rows=manifest_rows,physical_capture_refs=sink.refs,capture_files=len(sink.refs),raw_tensor_payload_bytes=sink.raw_bytes,
        serialized_bytes=sink.disk_bytes,CPU_query_core_decode_contract=spec['selector_cpu_proof_source'],current_owned_terminal_self_certified=False))
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['GPU_admission_file'])==gate['admission_ref'] and
        gate['common'].imported_files(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)==libraries and
        process.cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,'Final fullSource/current metadata/import/resource binding differs')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    report=dict(schema=REPORT_SCHEMA,status=STATUS,**binding,controls=policy_names(),control_results=results,
        images=187,raw_GT=380,queries_per_image=100,predictions_per_control=18700,normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=12,total_terminal_identity_pin_count=spec['normal_RAW_count']+12,
        native_packets=gate['packets'],selector_manifest=gate['selector']['manifest_ref'],native_reference=context['native_reference_predictions'],
        strong_fixed_reference=spec['strong_fixed_reference_predictions'],capture_manifest=reference(output/'CAPTURE_MANIFEST.json'),
        capture_refs=sink.refs,START=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),
        fixed_main_epoch=24,fixed_curve_epoch=8,DEV_winner_selected=False,score_class_query_order_preserved=True,
        native_model_loaded=False,native_forward_calls=0,optimizer_created=False,weights_updated=False,GT_used_for_prediction=False,
        independent_CPU_selection_proof_complete=False,CPU_renderer_equivalence_claimed=False,current_owned_terminal_self_certified=False,
        task_efficacy_established=False,innovation_established=False,standalone_end_to_end_FPS_claimed=False,
        cost=dict(imports_CPU_load_and_preCUDA_native_gate_seconds=imports_CPU_load_and_preCUDA_native_gate_seconds,
            head_transfer_seconds=transfer_seconds,all_image_inclusive_seconds=sum(image_times),
            native_field_replay_inclusive_seconds=native_field_replay_seconds,
            body_capture_RLE_inclusive_seconds=body_capture_RLE_seconds,
            refined_RLE_export_seconds=actual_RLE_export_seconds,refined_RLE_export_calls=actual_RLE_export_calls,
            official_COCO_seconds=COCO_seconds,cost_components_are_not_additive=True,
            physical_capture_files=len(sink.refs),raw_capture_bytes=sink.raw_bytes,serialized_capture_bytes=sink.disk_bytes,
            auxiliary_bytes_before_final_report=auxiliary_bytes,full_output_cap_bytes=72*1024**3,
            shared_full_C_diagnostic_runtime_only=True,subphase_costs_must_not_be_subtracted_for_standalone_speed=True,
            old_native_capture_replay_instead_of_fresh_teacher_forward=True),
        peak_torch_allocated_bytes=peak,cleanup=cleanup,actual_imported_files=libraries,elapsed_seconds=time.monotonic()-started,
        evidence_limits='Fixed historical DEV, all100, original native scores/order; GPU results pending ROOT owned actual0 and CPU proof. '
            'No fresh teacher forward/end-to-end FPS, blind TEST, 3seed, convergence, external SOTA, mechanism or innovation claim.')
    write_auxiliary(output/'REPORT.json',report)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'),CPU_proof_complete=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND_DRAFT: ROOT must finish original trainGPU/CPU0, fill actual bindings/budgets and admit ONE new leaf')
    run(args.spec)
