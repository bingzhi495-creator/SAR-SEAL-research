"""UNBOUND_DRAFT: aggregate four freshly owned actual0 N16 shards; ACTIVATION_PENDING_ROOT_BINDING.
Development has no scientific execution/input access. Actual own CPU0 and point proof remain ROOT obligations.
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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 15 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 15 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 stays separate; all terminal pins must equal normal+15')
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



SCHEMA='ROOT-N16-fixed-E24-all100-scout-shards-aggregate-spec-v1'
CPU_SCHEMA=SCHEMA
REPORT_SCHEMA='N16-fixed-E24-all100-scout-shards-aggregate-report-v1'
STATUS='N16_FIXED_E24_ALL100_SCOUT_SHARDS_AGGREGATE_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU_PROOF'
MANIFEST_SCHEMA='N16-GPU-all18700-shard-bundle-aggregate-manifest-v1'
RANGES=((0,47),(47,94),(94,141),(141,187))
DESCRIPTOR_KEYS={'shard_id','dataset_ordinal_range','source','spec','report','RAW','admission','shard_owner_index'}


def validated_descriptors(records):
    require(type(records) is list and len(records)==4,'Exactly four complete new actual0 shards required')
    for i,record in enumerate(records):
        require(type(record) is dict and set(record)==DESCRIPTOR_KEYS and type(record['shard_id']) is int and
            record['shard_id']==i and type(record['dataset_ordinal_range']) is list and
            len(record['dataset_ordinal_range'])==2 and all(type(n) is int for n in record['dataset_ordinal_range']) and
            record['dataset_ordinal_range']==list(RANGES[i]) and
            type(record['shard_owner_index']) is int and record['shard_owner_index']==109+i,
            'Canonical ordinal shard range/owner slot, no partial or substituted shard')
        for key in ('source','spec','report','RAW','admission'):ref_shape(record[key])
    for key in ('spec','report','RAW','admission'):
        require(len({record[key]['path'] for record in records})==4,'Distinct actual shard owner/artifact refs required: '+key)
    return records


def prediction_segment(values,image_ids,native_segment):
    require(type(values) is list and len(values)==len(image_ids)*100==len(native_segment),
            'Complete local natural100 prediction count required')
    expected=[image for image in image_ids for _ in range(100)]
    require([v['image_id'] for v in values]==expected and
            all(type(v) is dict and 'segmentation' in v and
                {k:x for k,x in v.items() if k!='segmentation'}=={k:x for k,x in n.items() if k!='segmentation'}
                for v,n in zip(values,native_segment)),
            'Prediction global ordinal image/query/class/S0 order must stay original')
    return values


def raw_first(spec):
    require(spec['schema']==SCHEMA and type(spec['normal_RAW_count']) is int and spec['normal_RAW_count']>=113 and
            spec['expected_normal_RAW_count']==spec['normal_RAW_count'] and spec['total_failed_RAW_count']==15 and
            spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+15 and
            spec['deadline_seconds']==1800 and spec['automatic_retry'] is False,'Four actual0 shards/full15 failure ancestry required')
    ninth_failed_raw_only(spec);tenth_failed_raw_only(spec);eleventh_failed_raw_only(spec)
    twelfth_resource_failure_raw_only(spec);thirteenth_failed_raw_only(spec)
    fourteenth_controlled_storage_failure_raw_only(spec);fifteenth_controlled_bundle_deadline_failure_raw_only(spec)
    return pre_gate(spec,spec['normal_RAW_count'])  # first Source SHA only after every original RAW/pin.


def source_gate(path,spec):
    raws,closure=raw_first(spec)
    for key in ('entry_source','shard_gate_source','execution_supplement'):closed(closure,spec[key])
    require(Path(spec['entry_source']['path']).resolve(strict=True)==Path(__file__).resolve(strict=True),
            'Aggregate own independent CPU entry required')
    shard=load(spec['shard_gate_source'],'_N16_aggregate_closed_shard_gate')
    require(shard.CPU_SCHEMA==SCHEMA,'Closed shard gate aggregate CPU schema ABI differs')
    gate=shard.source_gate(path,spec,cpu=True)
    require(gate['closure']==closure and gate['source']==spec['entry_source'] and gate['spec_ref']==reference(path) and
            Path(gate['admission_ref']['path']).resolve(strict=True)==Path(spec['CPU_admission_file']).resolve(strict=True),
            'Current aggregate CPUempty own source/spec/admission differs')
    records=validated_descriptors(spec['shards'])
    evidence=[];model_proofs=None
    for record in records:
        for key in ('source','spec','report','RAW','admission'):closed(closure,record[key])
        require(record['source']['sha256']==spec['shard_gate_source']['sha256'] and
                record['source']['bytes']==spec['shard_gate_source']['bytes'],'Every shard must execute the same closed new code')
        parent=read(closed(closure,record['spec']));report=read(closed(closure,record['report']))
        i=record['shard_id'];first,last=RANGES[i];owner=record['shard_owner_index']
        joined=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],record['RAW'],record['admission'],
                                            record['source'],record['spec'],'0')
        require(joined==owner==parent['normal_RAW_count']==parent['shard_owner_index']==109+i and
                parent['RAW_inputs']==spec['RAW_inputs'][:owner] and spec['RAW_inputs'][owner]['receipt']==record['RAW'],
                'Actual original shard GPU0 chronological prefix/source/spec/ad owner differs')
        require(parent['schema']==shard.SCHEMA and parent['entry_source']==record['source'] and
                parent['shard_id']==i and parent['dataset_ordinal_range']==list(RANGES[i]) and
                report['schema']==shard.REPORT_SCHEMA and report['status']==shard.STATUS and
                report['source']==record['source'] and report['spec']==record['spec'] and report['GPU_admission']==record['admission'] and
                report['phase']==parent['phase'] and report['shard_id']==i and report['dataset_ordinal_range']==list(RANGES[i]) and
                report['shard_owner_index']==owner and report['scope_dataset_indices']==list(range(first,last)) and
                report['images']==last-first and report['queries_per_image']==100 and report['queries']==100*(last-first) and
                report['predictions_per_control']==100*(last-first) and report['global_raw_GT']==380 and report['AP_evaluated'] is False,
                'Complete fresh fixed-range shard report bindings/scope required, never failed partials')
        require(Path(parent['GPU_admission_file']).resolve(strict=True)==Path(record['admission']['path']).resolve(strict=True) and
                all(closure.get(r['path'])==r for r in parent['sources']) and
                all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
                'Full shard Source and original immutable owner pins required')
        for key in shard.ROOT_FIELDS:
            if key!='capture_budget':require(parent[key]==spec[key],'Shard cannot change common model/protocol/data/supplement field: '+key)
        common_refs=shard.CARRIER_KEYS+('locked_protocol','cache_body_source','labels_source','FIT','original_fit_cache_report',
            'native_context_spec','region_common_source','original_fit_cache_entry_source','action_diagnostic_source','C_engine_source',
            'opportunity_ROOT_result','opportunity_CPU_qualification','prior_failed_CPU_receipt','additional_failed_RAW_inputs',
            'additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
            'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure',
            'additional_preserved_N16_fullDEV_packet_ABI_failure','additional_preserved_N16_fullDEV_controlled_storage_failure',
            'additional_preserved_N16_fullDEV_controlled_bundle_deadline_failure')
        require(all(parent[key]==spec[key] for key in common_refs),'Common original data/carrier/immutable failure refs differ')
        for key in ('START','trace','capture_manifest','selector_manifest','native_reference','strong_fixed_reference'):closed(closure,report[key])
        start=read(closed(closure,report['START']))
        if model_proofs is None:model_proofs=start['model_proofs']
        require(start['model_proofs']==model_proofs,'All four shards must use exact same frozen original PR/four scout proofs')
        binding=dict(source=record['source'],spec=record['spec'],GPU_admission=record['admission'],phase=parent['phase'],
            training_GPU_report=spec['training_GPU_report'],training_CPU_qualification=spec['training_CPU_qualification'],
            ROOTtrain_result=spec['ROOTtrain_result'],scout_checkpoints=spec['scout_checkpoints'],locked_protocol=spec['locked_protocol'],
            execution_supplement=spec['execution_supplement'],shard_id=i,dataset_ordinal_range=list(RANGES[i]))
        require(report['capture_binding']==binding and all(start[k]==v for k,v in binding.items()) and
                tuple(report['controls'])==shard.policy_names() and report['fixed_main_epoch']==24 and report['fixed_curve_epoch']==8 and
                report['DEV_winner_selected'] is False and report['score_class_query_order_preserved'] is True and
                report['GT_used_for_prediction'] is False and report['weights_updated'] is False and report['optimizer_created'] is False,
                'Actual per-shard capture binding and fixed method scope differ')
        manifest=read(closed(closure,report['capture_manifest']))
        require(manifest['schema']==shard.MANIFEST_SCHEMA and manifest['capture_binding']==binding and
                all(manifest[k]==v for k,v in binding.items()) and manifest['images']==last-first and manifest['queries']==100*(last-first) and
                manifest['global_images']==187 and manifest['global_queries']==18700 and
                manifest['physical_capture_refs']==report['capture_refs'] and manifest['capture_files']==last-first and
                len(manifest['rows'])==last-first and [r['dataset_index'] for r in manifest['rows']]==list(range(first,last)) and
                [r['bundle'] for r in manifest['rows']]==report['capture_refs'] and
                manifest['internal_capture_locators_are_strings'] is True,
                'Every actual local bundle/ordinal must appear exactly once')
        for ref in report['capture_refs']:closed(closure,ref)
        for row in manifest['rows']:closed(closure,row['diagnostics'])
        for name in shard.policy_names():
            result=report['control_results'][name]
            require(set(result)=={'predictions'},'Shard must export raw predictions, never local/averaged AP')
            closed(closure,result['predictions'])
        evidence.append(dict(descriptor=record,parent=parent,report=report,manifest=manifest,binding=binding))
    require([v['descriptor']['shard_owner_index'] for v in evidence]==list(range(109,113)),
            'Four new original successful shard owners must remain explicit109..112')
    gate.update(shard=shard,shard_evidence=evidence,model_proofs=model_proofs)
    return gate


def write_new(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return reference(path)


def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(),'Absolute ROOT aggregate spec required')
    spec=read(path);gate=source_gate(path,spec);closure=gate['closure'];context=gate['context'];shard=gate['shard']
    output=Path(spec['output_directory']);require(output.is_absolute() and not output.exists() and output.parent.is_dir(),'Fresh CPU aggregate output')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            all(os.environ.get(k)=='2' for k in THREAD_ENV),'CPUempty/CPU01/two thread environment required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified cold CPU runtime only, no head loading')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0','Original numeric versions required')
    def check():
        require(time.monotonic()-started<1800 and not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and
                os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV) and
                torch.get_num_threads()==torch.get_num_interop_threads()==2 and
                process.memory_info().rss<16*1024**3,'Bounded CPU-only aggregate/no retry')
    n10=load(context['N10_evaluation_source'],'_N16_aggregate_original_export')
    reader=load(context['reader_source'],'_N16_aggregate_original_reader')
    official=load(context['COCO_source'],'_N16_aggregate_official_COCO')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)
    coco=COCO(str(closed(closure,context['DEV'])));ids=list(coco.imgs)
    require(len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1} and
            ids==[v['id'] for v in coco.dataset['images']],'Complete original canonical DEV187/380/category1')
    native=read(closed(closure,context['native_reference_predictions']))
    strong=read(closed(closure,spec['strong_fixed_reference_predictions']))
    reader.population(coco,native,ids);reader.population(coco,strong,ids)
    names=shard.policy_names();predictions={name:[] for name in names};rows=[];bundle_refs=[]
    output.mkdir();outside=torch.get_rng_state().clone()
    aggregate_binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        locked_protocol=spec['locked_protocol'],execution_supplement=spec['execution_supplement'],ROOTtrain_result=spec['ROOTtrain_result'],
        training_GPU_report=spec['training_GPU_report'],training_CPU_qualification=spec['training_CPU_qualification'],scout_checkpoints=spec['scout_checkpoints'])
    start_ref=write_new(output/'START.json',dict(schema='N16-fixed-fullDEV-shards-aggregate-start-v1',**aggregate_binding,
        shards=spec['shards'],model_proofs=gate['model_proofs'],head_loaded=False,GPU_initialized=False,
        AP_is_recomputed_from_full_ordered_predictions=True,local_AP_is_never_averaged=True))
    with (output/'TRACE.jsonl').open('x',encoding='utf8',newline='\n') as trace:
        for item in gate['shard_evidence']:
            check();record=item['descriptor'];first,last=record['dataset_ordinal_range'];report=item['report'];manifest=item['manifest']
            require(report['scope_image_ids']==ids[first:last] and report['native_packets']==gate['packets'][first:last] and
                    report['selector_manifest']==gate['selector']['manifest_ref'] and
                    report['native_reference']==context['native_reference_predictions'] and
                    report['strong_fixed_reference']==spec['strong_fixed_reference_predictions'],
                    'Actual shard native/selector/global ordinal refs must preserve same original population')
            segment_native=native[first*100:last*100]
            for name in names:
                values=read(closed(closure,report['control_results'][name]['predictions']))
                predictions[name].extend(prediction_segment(values,ids[first:last],segment_native))
            for row in manifest['rows']:
                ordinal=row['dataset_index']
                require(row['image_id']==ids[ordinal] and row['natural_queries']==100 and
                        row['native_packet']==gate['packets'][ordinal] and row['selector']==gate['selector']['manifest']['rows'][ordinal] and
                        row['qids']==[v['query_id'] for v in native[ordinal*100:(ordinal+1)*100]],
                        'Actual shard bundle identity and original query order differs')
                copyrow=dict(row)
                copyrow.update(origin_shard_id=record['shard_id'],origin_shard_report=record['report'],
                    origin_shard_manifest=report['capture_manifest'],origin_shard_source=record['source'],origin_shard_spec=record['spec'],
                    origin_shard_RAW=record['RAW'],origin_shard_admission=record['admission'],original_GPU_binding=item['binding'])
                rows.append(copyrow);bundle_refs.append(row['bundle'])
            trace.write(json.dumps(dict(shard_id=record['shard_id'],dataset_ordinal_range=[first,last],images=last-first,
                original_GPU_report=record['report'],merged_images=len(rows)),allow_nan=False)+'\n');trace.flush()
    require([row['dataset_index'] for row in rows]==list(range(187)) and len(bundle_refs)==187 and
            len({r['path'] for r in bundle_refs})==187 and predictions['native_official']==native and
            sum(v['report']['raw_GT'] for v in gate['shard_evidence'])==380 and
            predictions['strong_fixed_E24_Omega']==strong and all(len(v)==18700 for v in predictions.values()),
            'Exactly full18700 native/strong/all12 predictions and187 actual new bundles; no gaps/duplicates/failed partials')
    manifest_ref=write_new(output/'CAPTURE_MANIFEST.json',dict(schema=MANIFEST_SCHEMA,**aggregate_binding,images=187,queries=18700,
        rows=rows,physical_capture_refs=bundle_refs,capture_files=187,internal_capture_locators_are_strings=True,
        shards=spec['shards'],shard_reports=[v['descriptor']['report'] for v in gate['shard_evidence']],
        shard_manifests=[v['report']['capture_manifest'] for v in gate['shard_evidence']],
        original_capture_bindings_preserved=True,bundle_bytes_rewritten=False,aggregate_CPU_binding_is_not_GPU_capture_binding=True))
    results={}
    for name,values in predictions.items():
        check();reader.population(coco,values,ids)
        target=output/(name+'_PREDICTIONS.json');pred_ref=write_new(target,values)
        metrics=official.coco_metrics(coco,values,ids,COCOeval)
        require(metrics['iouType']=='segm' and metrics['maxDets']==[1,10,100] and metrics['score_filter'] is None and
                metrics['original_GT_preserved'] is True and metrics['official_COCO_crowd_ignore_semantics'] is True,'Exact official COCO12 semantics')
        n10.stats_exact(metrics['stats'],metrics['stats'],'Finite original COCO12/-1 convention')
        results[name]=dict(predictions=pred_ref,COCO=metrics,stats=metrics['stats'])
    n10.stats_exact(results['native_official']['stats'],gate['prior_q']['expected_E24_COCO_stats'],'Full native COCO12 exact')
    control=gate['carrier_report']['control_results'][spec['strong_fixed_carrier_control_name']]
    require(control['predictions']==spec['strong_fixed_reference_predictions'],'Original strongest E24 control ref, no winner replacement')
    n10.stats_exact(results['strong_fixed_E24_Omega']['stats'],control['stats'],'Full original strong E24 COCO12 exact')
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
            reference(spec['CPU_admission_file'])==gate['admission_ref'] and torch.equal(outside,torch.get_rng_state()) and
            torch.get_num_threads()==torch.get_num_interop_threads()==2 and not torch.cuda.is_initialized() and
            process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            all(os.environ.get(k)=='2' for k in THREAD_ENV) and
            gate['common'].imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'],ndimage)==libraries,
            'Final own Source/spec/admission/RNG/imported libraries/coldCPU resources exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    report=dict(schema=REPORT_SCHEMA,status=STATUS,**aggregate_binding,controls=list(names),control_results=results,
        images=187,raw_GT=380,queries_per_image=100,predictions_per_control=18700,shards=spec['shards'],
        shard_reports=[v['descriptor']['report'] for v in gate['shard_evidence']],
        shard_manifests=[v['report']['capture_manifest'] for v in gate['shard_evidence']],
        capture_manifest=manifest_ref,capture_refs=bundle_refs,capture_storage='original_per_image_shard_GPU_bundles_preserved',
        native_reference=context['native_reference_predictions'],strong_fixed_reference=spec['strong_fixed_reference_predictions'],
        selector_manifest=gate['selector']['manifest_ref'],native_packets=gate['packets'],START=start_ref,trace=reference(output/'TRACE.jsonl'),
        fixed_main_epoch=24,fixed_curve_epoch=8,DEV_winner_selected=False,score_class_query_order_preserved=True,
        AP_recomputed_from_full_predictions=True,local_AP_averaged=False,normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=15,total_terminal_identity_pin_count=spec['normal_RAW_count']+15,
        model_proofs=gate['model_proofs'],head_loaded=False,CUDA_initialized=False,bundle_bytes_rewritten=False,
        independent_CPU_selection_proof_complete=False,current_owned_terminal_self_certified=False,
        task_efficacy_established=False,innovation_established=False,actual_imported_files=libraries,elapsed_seconds=time.monotonic()-started)
    report_ref=write_new(output/'REPORT.json',report)
    print(json.dumps(dict(status=STATUS,report=report_ref,CPU_point_proof_complete=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'UNBOUND_DRAFT: ROOT must bind activated supplement/failure15 and actual0 shard metadata')
    run(args.spec)
