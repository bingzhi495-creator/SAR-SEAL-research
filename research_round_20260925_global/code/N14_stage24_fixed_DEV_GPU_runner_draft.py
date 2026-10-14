"""UNBOUND_DRAFT: complete N14 fixed stage24 historical DEV GPU runner.

NOT RUN. ROOT must close current stage24 GPU/CPU evidence, audit actual Source,
fill concrete metadata/resource budgets and bind a NEW immutable evaluation.
Only allowed ordinary drafts/closed-source excerpts were used in development;
no actual SCI, data, weights, RAW, Q or live outputs were read.

Native provenance comes from the original qualified E24 carrier/old lineage;
that old gate qualifies ONLY its original PR/cache/native history. N14 trained
BASE40497/MEM42698 use a SEPARATE actual stage24 GPU+CPU canonical gate. No old
load_heads or LOG/GEO8-tensor qualification is applied to the memory state.
All187 images,380 GT,all100 queries,original scores/classes/order,exact frozen
GPU L selector and oldLOG K are retained; no best-DEV, TEST, native model or
training optimizer. Helper v02 writes one real CPU capture package per image;
no member virtual refs. Source B plus exact readback B is budgeted without clone.

source_gate(path,spec,cpu=False) is also the ordinary shared gate ABI for a NEW
independent CPU replay runner. cpu=True uses CPU schema/admission/cuda visibility
and still validates actual N14 training evidence; caller must additionally
join its completed GPU evaluation/source/capture manifest before any replay.
GPU report is GPU_AP_PENDING_INDEPENDENT_CPU_REPLAY. A later same-input CPU
allclose/exactmask failure is FAILED_CPU_REPLAY pending ROOT forensic review,
not automatic proof that GPU AP or the method is invalid; no post-hoc tolerance
relaxation is authorized. The current GPU report cannot certify its own terminal.
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
ARMS=("BASE_CONTINUATION","MEMORY_ADDON")
SCHEMA="ROOT-N14-stage24-fixed-fullDEV-GPU-spec-v1"
FIRST_STAGE_SCHEMA="ROOT-N14-LOG-E24-memory-paired-warm-stage-GPU-spec-v1"
STATUS="N14_STAGE24_FIXED_DEV_GPU_COMPLETE_PENDING_INDEPENDENT_CPU"

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
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 9 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 9 and
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



def imported_files(psutil,torch,np,gate):
    modules=dict(psutil=psutil,psutil_platform=psutil._psplatform,psutil_binary=psutil._psplatform.cext,
        torch=torch,torch_binary=torch._C,torch_tensor=sys.modules['torch._tensor'],torch_autograd=sys.modules['torch.autograd'],
        torch_random=torch.random,numpy=np,numpy_binary=np.core._multiarray_umath)
    result={name:reference(module.__file__) for name,module in modules.items()}
    objects=dict(torch_Module=torch.nn.Module,torch_Linear=torch.nn.Linear,AdamW=torch.optim.AdamW,
        Optimizer=torch.optim.Optimizer,point_math=torch.nn.functional.binary_cross_entropy_with_logits,
        torch_serialization=torch.load)
    result.update({name:reference(inspect.getsourcefile(obj)) for name,obj in objects.items()})
    require(all(ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256']
        for ref in result.values()),'Actual imported library/source/binary is not fully pinned')
    return result


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(value):
    a = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(torch, v) for v in value)
    return value


CPU_SCHEMA = 'ROOT-N14-stage24-fixed-fullDEV-CPU-spec-v1'
REPORT_SCHEMA = 'N14-stage24-fixed-fullDEV-GPU-report-v1'
CAPTURE_MANIFEST_SCHEMA = 'N14-stage24-fixed-DEV-capture-manifest-v2'
TRAIN_SCHEMA = 'ROOT-N14-LOG-E24-memory-paired-stage-continuation-GPU-spec-v2'
TRAIN_STATUS = 'N14_MEMORY_PAIRED_STAGE_LEAF_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION'
CPU24_SCHEMA = 'N14-stage24-training-endpoint-CPU-qualification-v1'
CPU24_STATUS = 'ROOT_N14_MEMORY_PAIRED_STAGE24_ENDPOINT_CPU_QUALIFIED_V1'
STAGE24_REF_KEYS = ('stage24_GPU_source', 'stage24_GPU_spec', 'stage24_GPU_report', 'stage24_GPU_admission', 'stage24_GPU_RAW',
    'stage24_CPU_source', 'stage24_CPU_spec', 'stage24_CPU_admission', 'stage24_CPU_RAW', 'stage24_CPU_qualification',
    'stage24_CPU_endpoint_numeric_protocol')


def deadline(started, spec):
    require(time.monotonic() - started < spec['deadline_seconds'], 'Prelocked evaluation1800-second deadline exceeded; no retry')


def source_gate(path, spec, cpu=False):
    """Complete RAW/SOURCE/owner/native gate, separate from new N14 head gate.

    CPU replay caller must separately validate/join its completed GPU evaluation
    and captures. This shared gate never treats training Q as DEV replay evidence.
    """
    require(spec['schema'] == (CPU_SCHEMA if cpu else SCHEMA) and
            type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] >= 101 and
            spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] and spec['deadline_seconds'] == 1800 and
            spec['automatic_retry'] is False and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
            'Exact new N14 evaluation schema/full latest ancestry/resource budget required')
    ninth_failed_raw_only(spec)
    raws, closure = pre_gate(spec, spec['expected_normal_RAW_count'])
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    for key in STAGE24_REF_KEYS + ('entry_source', 'carrier_spec', 'carrier_evaluation_source', 'evaluation_adapter_source', 'memory_source'):
        closed(closure, spec[key])
    require(Path(spec['entry_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True) or cpu,
            'Current GPU runner entry must be its new admitted source; CPU caller must supply its own entry_file below')
    carrier = read(closed(closure, spec['carrier_spec']))
    require(carrier['seed'] == spec['seed'] == 202610120701 and carrier['head_endpoint_epoch'] == 24 and
            carrier['entry_source'] == spec['carrier_evaluation_source'], 'Original qualified E24 common/native provenance differs')
    training_spec = read(closed(closure, spec['stage24_GPU_spec']))
    report = read(closed(closure, spec['stage24_GPU_report']))
    cpu_spec = read(closed(closure, spec['stage24_CPU_spec']))
    q = read(closed(closure, spec['stage24_CPU_qualification']))
    numeric = read(closed(closure, spec['stage24_CPU_endpoint_numeric_protocol']))
    n = training_spec['normal_RAW_count']
    require(type(n) is int and n >= 99 and cpu_spec['normal_RAW_count'] == n + 1 and
            spec['normal_RAW_count'] >= n + 2 + int(cpu) and training_spec['RAW_inputs'] == spec['RAW_inputs'][:n] and
            cpu_spec['RAW_inputs'] == spec['RAW_inputs'][:n + 1] and
            spec['RAW_inputs'][n]['receipt'] == spec['stage24_GPU_RAW'] and spec['RAW_inputs'][n + 1]['receipt'] == spec['stage24_CPU_RAW'],
            'Actual current stage24 GPU/CPU normal prefixes and receipt positions differ')
    for ancestor in (training_spec, cpu_spec):
        require(ancestor['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
                ancestor['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
                ancestor['additional_preserved_independent_CPU_failure'] == spec['additional_preserved_independent_CPU_failure'] and
                ancestor['total_failed_RAW_count'] == 9 and
                all(spec['terminal_identity_pins'].get(key) == pin for key, pin in ancestor['terminal_identity_pins'].items()) and
                all(closure.get(ref['path']) == ref for ref in ancestor['sources']), 'Preserved stage24 GPU/CPU failed history/SOURCE closure differs')
    require(training_spec['schema'] == report['schema'] == TRAIN_SCHEMA and report['status'] == TRAIN_STATUS and
            training_spec['entry_source'] == report['source'] == spec['stage24_GPU_source'] and report['spec'] == spec['stage24_GPU_spec'] and
            report['GPU_admission'] == spec['stage24_GPU_admission'] and report['phase'] == training_spec['phase'] == spec['RAW_inputs'][n]['phase'] and
            Path(training_spec['GPU_admission_file']).resolve(strict=True) == Path(spec['stage24_GPU_admission']['path']).resolve(strict=True) and
            spec['RAW_inputs'][n]['admission'] == spec['stage24_GPU_admission'] and
            spec['RAW_inputs'][n + 1]['admission'] == spec['stage24_CPU_admission'], 'Actual completed stage24 GPU report/spec/admission differs')
    require(training_spec['first_stage_epoch'] == report['first_stage_epoch'] == 17 and
            training_spec['last_stage_epoch'] == report['last_stage_epoch'] == report['stage_epochs_completed'] == 24 and
            training_spec['fixed_stage_epochs'] == report['original_parent_epochs'] == report['fixed_stage_epochs'] == 24 and
            report['new_stage_epochs'] == 8 and report['updates_per_arm'] == report['leaf_updates_per_arm'] == 13288 and
            report['cumulative_stage_updates_per_arm'] == 39864 and report['continuous_parent_plus_stage'] is False and
            report['own_terminal_self_certified'] is False and report['independent_CPU_qualified'] is False and
            report['no_DEV'] is True and report['no_TEST'] is True and report['AP_evaluated'] is False,
            'Prelocked complete new stage24 must remain distinct from original E24 provenance')
    require(q['schema'] == CPU24_SCHEMA and q['status'] == CPU24_STATUS and q['source'] == spec['stage24_CPU_source'] and
            q['spec'] == spec['stage24_CPU_spec'] and q['CPU_admission'] == spec['stage24_CPU_admission'] and
            q['phase'] == cpu_spec['phase'] == spec['RAW_inputs'][n + 1]['phase'] and
            cpu_spec['entry_source'] == spec['stage24_CPU_source'] and
            Path(cpu_spec['CPU_admission_file']).resolve(strict=True) == Path(spec['stage24_CPU_admission']['path']).resolve(strict=True) and
            q['current_owned_CPU_terminal_self_certified'] is False, 'Actual independent CPU24 qualification binding differs')
    correspondence = dict(training_GPU_source='stage24_GPU_source', training_GPU_spec='stage24_GPU_spec',
        training_GPU_report='stage24_GPU_report', training_GPU_RAW='stage24_GPU_RAW', training_GPU_admission='stage24_GPU_admission')
    for key, destination in correspondence.items():
        require(q[key] == cpu_spec[key] == spec[destination], 'CPU24 must bind this exact stage24 GPU ' + key)
    require(q['checkpoints'] == report['checkpoints'] == cpu_spec['stage24_checkpoints'] == spec['stage24_checkpoints'] and
            set(report['checkpoints']) == set(ARMS) and q['parameters'] == dict(BASE_CONTINUATION=40497, MEMORY_ADDON=42698) and
            q['original_parent_epochs'] == q['fixed_stage_epochs'] == 24 and q['new_stage_epochs'] == 8 and
            q['paired_updates_per_arm'] == 39864 and q['stage_epochs_completed'] == 24 and
            q['carrier_spec'] == training_spec['carrier_spec'] == spec['carrier_spec'] and
            q['memory_source'] == training_spec['memory_source'] == spec['memory_source'] and spec['memory_seed'] == training_spec['module_seed'],
            'Exact same two qualified stage24 checkpoints/provenance required everywhere')
    required_Q = ('common_E24_warm_initialization_exact', 'deterministic_zero_addon_initialization_exact',
        'all_new_state_FP32_finite_shape_exact', 'all_AdamW_parameter_steps_exact39864',
        'private_twentyfour741_orders_and_final_state_exact', 'complete_paired39864_trace_order_identity_online_means_exact',
        'final_stage24_full_FIT_mean_BCE_CPU_GPU_compatible', 'actual_parent_resume_state_digest_exact')
    require(all(q[key] is True for key in required_Q) and q['no_DEV'] is True and q['no_TEST'] is True and q['AP_evaluated'] is False and
            q['optimizer_trajectory_independently_replayed'] is False and q['per_instance_GPU_CPU_endpoint_losses_compared'] is False,
            'Complete actual CPU24 state/order/trace/endpoint proof and scope required')
    require(q['CPU_endpoint_numeric_protocol'] == cpu_spec['CPU_endpoint_numeric_protocol'] == cpu_spec['endpoint_numeric_source'] ==
            spec['stage24_CPU_endpoint_numeric_protocol'] and type(numeric) is dict and set(numeric) ==
            {'atol', 'rtol', 'locked_before_CPU_execution', 'locked_after_GPU_training', 'scope', 'bound_completed_GPU_report'} and
            numeric['bound_completed_GPU_report'] == spec['stage24_GPU_report'] and numeric['locked_before_CPU_execution'] is True and
            numeric['locked_after_GPU_training'] is True and numeric['atol'] == numeric['rtol'] == 1e-4,
            'Training CPU24 numeric protocol must remain separately locked after training and before CPU execution')
    details = read(closed(closure, q['details']))
    require(details['source'] == spec['stage24_CPU_source'] and details['spec'] == spec['stage24_CPU_spec'] and
            details['CPU_admission'] == spec['stage24_CPU_admission'] and details['phase'] == q['phase'] and
            details['training_GPU_report'] == spec['stage24_GPU_report'] and details['stage_epochs_completed'] == 24 and
            details['cumulative_paired_updates'] == 39864 and details['FIT_images'] == 741 and details['FIT_instances'] == 1661 and
            details['trace_chain'] == report['trace_chain'], 'Actual qualified CPU24 details differ')
    require(len(report['history']) == len(report['epoch_orders']) == 24 and len(report['trace_chain']) == 3 and
            report['trace_chain'][-1] == report['trace'] and len({ref['path'] for ref in report['trace_chain']}) == 3,
            'Complete original24 history/orders/three-leaf traces required')
    for ref in list(report['checkpoints'].values()) + report['trace_chain'] + [report['start'], report['stage_origin_START']]:
        closed(closure, ref)
    origin = read(closed(closure, report['stage_origin_START']))
    require(origin['parent'] == spec['carrier_spec'] and origin['original_parent_epochs'] == 24 and
            origin['first_stage_epoch'] == 1 and origin['last_stage_epoch'] == 8 and origin['stage_epochs'] == 24 and
            origin['optimizer_reset_both'] is True and origin['initial_state_digests'] == report['initial_state_digests'] and
            origin['behavior'] == report['behavior_preflight'] and origin['common_LOG_parent_proofs'] == report['common_LOG_parent_proofs'],
            'Original common E24 warm/first-leaf zero-addon proofs must remain immutable')
    lineage = load(carrier['endpoint_lineage_source'], '_N14_DEV_original_native_lineage')
    context = dict(carrier)
    for key in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'terminal_identity_pins', 'entry_source', 'phase', 'deadline_seconds'):
        context[key] = spec[key]
    context['CPU_admission_file' if cpu else 'GPU_admission_file'] = spec['CPU_admission_file' if cpu else 'GPU_admission_file']
    # A CPU runner supplies its OWN source file, not this loaded helper's __file__.
    entry_file = spec['entry_source']['path'] if cpu else __file__
    gate = api(lineage, 'own_gate')(path, context, entry_file, closure, raws, cpu=cpu)
    gi = api(lineage, 'owner_ref_join')(context, gate['owners'], spec['stage24_GPU_RAW'], spec['stage24_GPU_admission'],
        spec['stage24_GPU_source'], spec['stage24_GPU_spec'], '0')
    ci = api(lineage, 'owner_ref_join')(context, gate['owners'], spec['stage24_CPU_RAW'], spec['stage24_CPU_admission'],
        spec['stage24_CPU_source'], spec['stage24_CPU_spec'], '')
    require((gi, ci) == (n, n + 1), 'Actual independent stage24 GPU0/CPU0 joins must match exact full ancestry positions')
    # ONLY original carrierE24 factory/cache/native provenance is qualified here.
    gate.update(api(lineage, 'endpoint_gate')(context, closure, gate['owners']))
    prior, prior_q = api(lineage, 'prior_gate')(context, closure, gate['owners'])
    require(prior['teacher_checkpoint'] == gate['cache_report']['teacher_checkpoint'], 'Original FIT/DEV shared teacher differs')
    selector = api(lineage, 'selector_parent_gate')(context, closure, gate['owners'])
    effective = dict(carrier)
    effective.update(spec)
    require(effective['selector_mode'] == 'reuse' and effective['renderer_source'] == carrier['renderer_source'] and
            effective['DEV'] == carrier['DEV'] and effective['native_reference_predictions'] == carrier['native_reference_predictions'],
            'Original renderer/DEV/native S0/selector must remain frozen')
    adapter = load(spec['evaluation_adapter_source'], '_N14_DEV_new_stage24_capture_adapter')
    require(adapter.CAPTURE_SCHEMA == 'N14-fixed-DEV-actual-GPU-point-input-capture-v2', 'Actual whole-image capture helper v02 required')
    api(adapter, 'check_recipe')(effective)
    canonical = dict(schema=adapter.GATE_SCHEMA, GPU_source=spec['stage24_GPU_source'], GPU_spec=spec['stage24_GPU_spec'],
        GPU_admission=spec['stage24_GPU_admission'], GPU_RAW=spec['stage24_GPU_RAW'], GPU_report_ref=spec['stage24_GPU_report'],
        CPU_source=spec['stage24_CPU_source'], CPU_spec=spec['stage24_CPU_spec'], CPU_admission=spec['stage24_CPU_admission'],
        CPU_RAW=spec['stage24_CPU_RAW'], CPU_qualification_ref=spec['stage24_CPU_qualification'],
        GPU_report=report, GPU_phase=report['phase'], checkpoints=report['checkpoints'], proofs={})
    gate.update(lineage=lineage, context=context, evaluation_spec=effective, report=prior, prior=prior, prior_q=prior_q,
        packets=prior_q['packet_refs'], selector=selector, adapter=adapter, stage24_gate=canonical, stage24_GPU_report=report,
        stage24_GPU_spec=training_spec, stage24_CPU_Q=q, stage24_CPU_details=details, stage_origin=origin)
    return gate


def canonical_stage24_gate(torch, gate, spec):
    """Actual checked states/moments/orders/trace evidence, not bare booleans."""
    adapter, report = gate['adapter'], gate['stage24_GPU_report']
    outside = torch.get_rng_state().clone()
    states = {}
    order_rng = torch.Generator(device='cpu')
    order_rng.manual_seed(gate['stage24_GPU_spec']['order_seed'])
    for epoch, row in enumerate(report['epoch_orders'], 1):
        order = torch.randperm(741, generator=order_rng).tolist()
        sha = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode('ascii')).hexdigest()
        h = report['history'][epoch - 1]
        require(row == dict(stage_epoch=epoch, indices=order, sha256=sha) and h['stage_epoch'] == epoch and
                h['paired_image_order_sha256'] == sha and h['updates_per_arm'] == 1661 and h['cumulative_stage_updates'] == epoch * 1661,
                'Independent readback of all24 private orders/history differs')
    for arm in ARMS:
        cp = torch.load(str(closed(gate['closure'], report['checkpoints'][arm])), map_location='cpu')
        require(set(cp) == {'meta', 'state_dict', 'optimizer_state_dict', 'private_order_state'}, 'Complete actual trained checkpoint required')
        meta, state = cp['meta'], cp['state_dict']
        expected_shapes = dict(zip(adapter.BASE_NAMES, adapter.BASE_SHAPES))
        if arm == 'MEMORY_ADDON':
            expected_shapes = {'base.' + name: shape for name, shape in expected_shapes.items()}
            expected_shapes.update(adapter.MEMORY_SHAPES)
        require(set(state) == set(expected_shapes) and sum(value.numel() for value in state.values()) == adapter.PARAMETERS[arm] and
                all(value.device.type == 'cpu' and value.dtype == torch.float32 and tuple(value.shape) == expected_shapes[name] and
                    bool(torch.isfinite(value).all()) for name, value in state.items()), 'Separate actual8/14 and40497/42698 FP32 states required')
        require(meta['schema'] == 'N14-context-memory-stage-checkpoint-v2' and meta['arm'] == arm and
                meta['source'] == spec['stage24_GPU_source'] and meta['spec'] == spec['stage24_GPU_spec'] and
                meta['GPU_admission'] == spec['stage24_GPU_admission'] and meta['phase'] == report['phase'] and
                meta['stage_epochs_completed'] == 24 and meta['stage_updates'] == meta['cumulative_stage_updates'] == 39864 and
                meta['parameters'] == adapter.PARAMETERS[arm] and meta['history'] == report['history'] and meta['epoch_orders'] == report['epoch_orders'] and
                meta['trace_chain'] == report['trace_chain'] and meta['stage_origin_START'] == report['stage_origin_START'] and
                meta['gradient_activity'] == report['gradient_activity'][arm] and meta['initial_state_digests'] == report['initial_state_digests'][arm] and
                meta['optimizer_reset_at_stage_start'] is True and meta['optimizer_reset_this_leaf'] is False,
                'Actual trained stage24 checkpoint metadata differs')
        optimizer = cp['optimizer_state_dict']
        require(set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1 and
                optimizer['param_groups'][0]['params'] == list(range(len(expected_shapes))) and
                set(optimizer['state']) == set(range(len(expected_shapes))), 'Complete actual AdamW IDs/state required')
        # Independent CPU24 has already bound exact moment-to-actual-parameter
        # mapping. Here verify every moment finite/FP32 and exact scalar step.
        for item in optimizer['state'].values():
            require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'} and isinstance(item['step'], torch.Tensor) and
                    item['step'].device.type == 'cpu' and item['step'].dtype == torch.float32 and item['step'].shape == () and
                    bool(torch.isfinite(item['step'])) and float(item['step'].item()) == 39864.0 and
                    all(value.device.type == 'cpu' and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                        for value in (item['exp_avg'], item['exp_avg_sq'])) and bool((item['exp_avg_sq'] >= 0).all()),
                    'All actual AdamW state/39864 steps must be finite and exact')
        require(torch.equal(cp['private_order_state'], order_rng.get_state()), 'Actual complete private order state differs')
        require(set(meta['gradient_activity']) == set(state) and all(row['present'] == 39864 and 0 <= row['nonzero'] <= 39864
            for row in meta['gradient_activity'].values()), 'Complete stage24 gradient bookkeeping differs')
        states[arm] = state
        del cp
    identities = gate['stage24_CPU_details']['qualified_cache_image_instance_identities']
    require(len(identities) == 741 and sum(len(rows) for rows in identities) == 1661, 'Qualified FIT identity map must preserve741/1661')
    updates = 0
    for leaf, ref in enumerate(report['trace_chain']):
        leaf_count = 0
        with closed(gate['closure'], ref).open('r', encoding='utf-8') as stream:
            for epoch in range(8 * leaf + 1, 8 * leaf + 9):
                losses = {arm: [] for arm in ARMS}
                for image_index in report['epoch_orders'][epoch - 1]['indices']:
                    for g, identity in enumerate(identities[image_index]):
                        row = json.loads(stream.readline())
                        updates += 1
                        leaf_count += 1
                        expected = dict(stage_epoch=epoch, image_index=image_index, GT_index=g,
                            annotation_id=identity['annotation_id'], query_id=identity['query_id'], stage_update=updates)
                        require(set(row) == set(expected) | {'BCE'} and all(type(row[key]) is int and row[key] == value for key, value in expected.items()) and
                                set(row['BCE']) == set(ARMS), 'Complete actual cumulative paired trace identity differs')
                        for arm in ARMS:
                            value = row['BCE'][arm]
                            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Finite actual online trace BCE required')
                            losses[arm].append(value)
                require(all(len(values) == 1661 for values in losses.values()) and
                        {arm: math.fsum(values) / 1661 for arm, values in losses.items()} == report['history'][epoch - 1]['online_BCE'],
                        'Every actual epoch1661 trace online mean must be exact')
            require(stream.readline() == '' and leaf_count == 13288, 'Each actual leaf trace must contain13288 paired rows')
    require(updates == 39864 and torch.equal(outside, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'Full24 trace/bookkeeping readback changed outside RNG/CUDA')
    canonical = dict(gate['stage24_gate'])
    canonical['proofs'] = {key: True for key in adapter.stage24_adapter_contract()['mandatory_proofs']}
    # Every proof is set only after the concrete checks above and source_gate's
    # actual RAW/owner/Q/native/cache/selector verification. No training is replayed.
    gate['stage24_gate'] = canonical
    return canonical


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if type(left) is dict:
        return type(right) is dict and set(left) == set(right) and all(tree_equal(torch, value, right[key]) for key, value in left.items())
    if type(left) in (tuple, list):
        return type(right) is type(left) and len(left) == len(right) and all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


class CaptureSink:
    """One real CPU wholepacket per image; no cloned third payload or virtual ref."""
    def __init__(self, torch, root, budget, deadline_callback, process):
        self.torch, self.root, self.budget, self.deadline_callback, self.process = torch, Path(root).resolve(), budget, deadline_callback, process
        self.refs = []
        self.serialized_bytes = self.payload_bytes = 0

    def cpu_only(self, value):
        t = self.torch
        if isinstance(value, t.Tensor):
            require(value.device.type == 'cpu' and value.layout == t.strided and
                    value.dtype in (t.float32, t.int64, t.uint8, t.bool) and not value.requires_grad and
                    bool(t.isfinite(value).all()), 'Capture sink accepts only finite already-CPU detached tensors')
        elif type(value) is dict:
            for item in value.values(): self.cpu_only(item)
        elif type(value) in (tuple, list):
            for item in value: self.cpu_only(item)
        elif type(value) is float:
            require(math.isfinite(value), 'Capture metadata must be finite')

    def write(self, relative_name, value):
        self.deadline_callback()
        require(type(relative_name) is str and not Path(relative_name).is_absolute() and '..' not in Path(relative_name).parts,
                'Fresh relative capture path must stay inside current output')
        target = (self.root / relative_name).resolve()
        try: target.relative_to(self.root)
        except ValueError: raise RuntimeError('Capture path escaped fresh output')
        require(not target.exists(), 'Never overwrite a capture file')
        self.cpu_only(value)
        payload = value['raw_tensor_payload_bytes']
        require(type(payload) is int and 0 <= payload <= self.budget['maximum_image_raw_tensor_payload_bytes'] and
                self.payload_bytes + payload <= self.budget['raw_tensor_payload_upper_bound_bytes'], 'Actual per-image/cumulative tensor payload exceeds prelock')
        require(self.process.memory_info().rss + payload <= self.budget['maximum_process_RSS_bytes'],
                'Wholepacket exact readback would exceed ROOT prelocked peak RAM')
        target.parent.mkdir(parents=True, exist_ok=True)
        # value is already CPU; cloning it would create a third full payload.
        with target.open('xb') as stream: self.torch.save(value, stream)
        replay = self.torch.load(str(target), map_location='cpu')
        require(tree_equal(self.torch, value, replay), 'Exact saved whole-image capture readback differs')
        del replay
        ref = reference(target)
        self.payload_bytes += payload
        self.serialized_bytes += ref['bytes']
        require(self.serialized_bytes <= self.budget['raw_tensor_payload_upper_bound_bytes'] + self.budget['serialized_capture_overhead_budget_bytes'],
                'Actual capture serialization exceeded prelocked disk overhead')
        self.refs.append(ref)
        return ref

    def read(self, ref):
        self.deadline_callback()
        require(ref in self.refs, 'Sink read requires a real newly saved wholepacket ref')
        return self.torch.load(str(bound(ref)), map_location='cpu')


def validate_budget(spec, adapter, coco, ids, rows, process, psutil, output):
    budget = spec['capture_budget']
    required = {'total_original_pixels', 'total_LOG_points', 'union_points_upper_bound', 'raw_tensor_payload_upper_bound_bytes',
        'maximum_image_pixels', 'maximum_image_LOG_points', 'maximum_image_union_points_upper_bound', 'maximum_image_raw_tensor_payload_bytes',
        'serialized_capture_overhead_budget_bytes', 'auxiliary_output_budget_bytes', 'minimum_free_output_bytes',
        'required_available_RAM_bytes', 'capture_readback_peak_RAM_upper_bound_bytes', 'maximum_process_RSS_bytes'}
    require(type(budget) is dict and set(budget) == required and all(type(value) is int and value >= 0 for value in budget.values()),
            'ROOT must prelock concrete N/T/disk/readback/process RAM budget fields')
    pixels = [coco.imgs[i]['height'] * coco.imgs[i]['width'] for i in ids]
    Ts = [sum(row['band_counts_all100']) for row in rows]
    require(budget['total_original_pixels'] == sum(pixels) and budget['total_LOG_points'] == sum(Ts) and
            budget['maximum_image_pixels'] == max(pixels) and budget['maximum_image_LOG_points'] == max(Ts) and
            budget['union_points_upper_bound'] == sum(pixels) and budget['maximum_image_union_points_upper_bound'] == max(pixels),
            'Root conservative union bounds must use actual closed N/T; no invented image dimensions')
    raw = adapter.capture_payload_budget(sum(pixels), sum(Ts), total_union_points=None,
        maximum_image_pixels=max(pixels), maximum_image_union_points=max(pixels), maximum_image_LOG_points=max(Ts))
    per_image_bound = max(1032 * N + 28 * T + 102400 for N, T in zip(pixels, Ts))
    require(budget['raw_tensor_payload_upper_bound_bytes'] == raw['raw_tensor_payload_bytes'] and
            budget['maximum_image_raw_tensor_payload_bytes'] == per_image_bound and
            budget['capture_readback_peak_RAM_upper_bound_bytes'] >= 2 * per_image_bound and
            budget['maximum_process_RSS_bytes'] >= budget['capture_readback_peak_RAM_upper_bound_bytes'] and
            budget['required_available_RAM_bytes'] >= budget['capture_readback_peak_RAM_upper_bound_bytes'] and
            budget['required_available_RAM_bytes'] >= 16 * 1024**3 and
            budget['minimum_free_output_bytes'] >= raw['raw_tensor_payload_bytes'] + budget['serialized_capture_overhead_budget_bytes'] +
                budget['auxiliary_output_budget_bytes'] + 2 * 1024**3 and
            psutil.disk_usage(str(output.parent)).free >= budget['minimum_free_output_bytes'] and
            psutil.virtual_memory().available >= budget['required_available_RAM_bytes'] and
            process.memory_info().rss < budget['maximum_process_RSS_bytes'],
            'Concrete disk/two-wholepacket readback/total-process RAM prelock not met')
    return dict(raw_tensor_bound=raw, maximum_whole_image_payload_bytes=per_image_bound, prelocked=budget,
                capture_files=187, full_raster_GPU_F_copies_added=0, current_process_RAM_certificate=False)


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec)
    effective, adapter, context = gate['evaluation_spec'], gate['adapter'], gate['context']
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh absolute ROOT-owned evaluation output required')
    import psutil
    process = psutil.Process()
    require(process.cpu_affinity() == [0, 1], 'Actual CPU01 required')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified cold Torch runtime required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    canonical = canonical_stage24_gate(torch, gate, spec)
    base_factory = load(context['point_heads_source'], '_N14_DEV_qualified_original_base_factory')
    memory_factory = load(spec['memory_source'], '_N14_DEV_qualified_memory_factory')
    heads, states, head_proofs, canonical = adapter.load_stage24_heads(torch, effective,
        lambda supplied: canonical, lambda ref: closed(gate['closure'], ref), base_factory, memory_factory)
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0' and scipy.__version__ == '1.10.1', 'Qualified numerical versions required')
    n10 = load(context['N10_evaluation_source'], '_N14_DEV_original_N10')
    reader = load(context['reader_source'], '_N14_DEV_original_reader')
    renderer = load(context['renderer_source'], '_N14_DEV_original_renderer')
    official = load(context['COCO_source'], '_N14_DEV_original_official_COCO')
    original_eval = load(spec['carrier_evaluation_source'], '_N14_DEV_original_selector_verifier')
    kernels = n10.original_export_kernels(context['export_source'])
    coco = COCO(str(closed(gate['closure'], context['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [row['id'] for row in coco.dataset['images']] and len(coco.anns) == 380 and set(coco.cats) == {1},
            'Complete historical DEV187/GT380/category1 required')
    native = read(closed(gate['closure'], context['native_reference_predictions']))
    population = reader.population(coco, native, ids)
    rows = read(closed(gate['closure'], gate['prior']['image_rows']))
    trace = [json.loads(line) for line in closed(gate['closure'], gate['prior']['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(rows == trace and len(rows) == len(gate['packets']) == len(gate['selector']['manifest']['rows']) == 187 and
            [row['image_id'] for row in rows] == ids and sum(row['raw_GT'] for row in rows) == 380, 'Canonical full187 packet/selector order differs')
    budget = validate_budget(spec, adapter, coco, ids, rows, process, psutil, output)
    libraries = gate['common'].imported_files(torch, np, cv2, masks, COCO, gate['closure'], gate['admission'], ndimage)
    require(not torch.cuda.is_initialized(), 'All RAW/SOURCE/N14/cache/library/budget gates must precede CUDA')
    torch.cuda.init()
    require(torch.cuda.current_device() == 0, 'Only GPU0 admitted')
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    outside = [torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()]
    for head in heads.values(): head.to('cuda:0')
    output.mkdir()
    (output / 'CAPTURES').mkdir()
    sink = CaptureSink(torch, output / 'CAPTURES', spec['capture_budget'], lambda: deadline(started, spec), process)
    binding = dict(source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'], phase=spec['phase'],
        stage24_GPU_source=spec['stage24_GPU_source'], stage24_GPU_spec=spec['stage24_GPU_spec'], stage24_GPU_report=spec['stage24_GPU_report'],
        stage24_CPU_qualification=spec['stage24_CPU_qualification'])
    write_new(output / 'START.json', dict(schema='N14-stage24-fixed-DEV-GPU-start-v1', source=gate['source'], spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'], phase=spec['phase'], stage24_gate=canonical, head_state_proofs=head_proofs,
        stage24_checkpoints=spec['stage24_checkpoints'], fixed_endpoint=24, seed=spec['seed'], controls=effective['controls'],
        selector_mode='reuse', selector_manifest=gate['selector']['manifest_ref'], capture_budget=budget,
        native_model_loaded=False, optimizer_created=False, no_TEST=True, best_DEV_selection=False))
    predictions = {control: [] for control in effective['controls']}
    diagnostics, selector_rows, capture_rows = [], [], []
    with (output / 'TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for index, (row, ref) in enumerate(zip(rows, gate['packets'])):
            deadline(started, spec)
            require(process.memory_info().rss < spec['capture_budget']['maximum_process_RSS_bytes'], 'Actual RSS exceeded prelocked process budget')
            packet = torch.load(str(closed(gate['closure'], ref)), map_location='cpu')
            tick = time.monotonic()
            base, refined, diagnostic, selector, capture_ref = adapter.evaluate_image(torch, np, masks, renderer, kernels, reader, n10, coco,
                packet, row, index, ref, population[row['image_id']], heads, gate, effective,
                original_eval.selector_packet, sink, binding, lambda: deadline(started, spec))
            torch.cuda.synchronize(0)
            diagnostic['elapsed_seconds'] = time.monotonic() - tick
            require(selector == gate['selector']['manifest']['rows'][index] and capture_ref == sink.refs[-1], 'Actual original selector/new wholepacket reference differs')
            predictions['native_official'] += base
            for arm in ARMS: predictions[arm] += refined[arm]
            diagnostics.append(diagnostic)
            selector_rows.append(selector)
            capture_rows.append(dict(dataset_index=index, image_id=row['image_id'], capture=capture_ref, U=diagnostic['U'], T=diagnostic['T'],
                raw_tensor_payload_bytes=diagnostic['raw_capture_tensor_bytes'], packet=ref, selector=selector,
                natural_queries=100, K_natural_query_order=diagnostic['K_natural_query_order']))
            stream.write(json.dumps(diagnostic, allow_nan=False) + '\n')
            stream.flush()
            del packet, base, refined, diagnostic, selector
            gc.collect()
            print(json.dumps(dict(N14_stage24_evaluated_DEV_images=index + 1, exact_native_queries=100 * (index + 1))), flush=True)
    require(predictions['native_official'] == native and all(len(values) == 18700 for values in predictions.values()) and
            len(sink.refs) == len(capture_rows) == 187 and selector_rows == gate['selector']['manifest']['rows'],
            'All18700 query population/187 actual capture files/frozen selectors required')
    for values in predictions.values():
        reader.population(coco, values, ids)
        require(all({key: value for key, value in original.items() if key != 'segmentation'} ==
                    {key: value for key, value in current.items() if key != 'segmentation'} for original, current in zip(native, values)),
                'Original all-query score/class/order identity differs')
    require(all(torch.equal(value.cpu(), states[arm][name]) for arm, head in heads.items() for name, value in head.state_dict().items()) and
            all(not p.requires_grad and p.grad is None for head in heads.values() for p in head.parameters()) and
            torch.equal(outside[0], torch.get_rng_state()) and torch.equal(outside[1], torch.cuda.get_rng_state(0)),
            'Read-only qualified stage24 weights/grad/RNG contract changed')
    peak = torch.cuda.max_memory_allocated(0)
    for head in heads.values(): head.cpu()
    heads.clear()
    del head, heads
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(allocated_after=torch.cuda.memory_allocated(0), reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after'] == 0 and peak <= 4096 * 1024**2, 'Actual allocator4GiB cap/cleanup required')
    results, refs = {}, {}
    for control, values in predictions.items():
        target = output / (control.upper() + '_PREDICTIONS.json')
        write_new(target, values)
        refs[control] = reference(target)
        deadline(started, spec)
        metrics = official.coco_metrics(coco, values, ids, COCOeval)
        require(metrics['iouType'] == 'segm' and metrics['maxDets'] == [1, 10, 100] and metrics['score_filter'] is None and
                metrics['original_GT_preserved'] is True and metrics['official_COCO_crowd_ignore_semantics'] is True, 'Original official COCO convention differs')
        n10.stats_exact(metrics['stats'], metrics['stats'], 'Finite defined COCO12/-1 required')
        results[control] = dict(predictions=refs[control], COCO=metrics, stats=metrics['stats'])
    n10.stats_exact(results['native_official']['stats'], gate['prior_q']['expected_E24_COCO_stats'], 'Original strong native E24 COCO12 differs')
    n10.stats_exact(results['native_official']['stats'], gate['prior']['control_results']['native_official']['stats'], 'Original native official COCO12 differs')
    write_new(output / 'IMAGE_DIAGNOSTICS.json', diagnostics)
    manifest = dict(schema=CAPTURE_MANIFEST_SCHEMA, source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'], phase=spec['phase'],
        stage24_GPU_report=spec['stage24_GPU_report'], stage24_CPU_qualification=spec['stage24_CPU_qualification'],
        images=187, queries=18700, rows=capture_rows, capture_schema=adapter.CAPTURE_SCHEMA, capture_files=187,
        raw_tensor_payload_bytes=sink.payload_bytes, actual_serialized_capture_bytes=sink.serialized_bytes,
        prelocked_budget=budget, actual_GPU_inputs_no_CPU_feature_rerender=True, current_owned_terminal_self_certified=False)
    write_new(output / 'CAPTURE_MANIFEST.json', manifest)
    for ref in spec['sources']:
        deadline(started, spec)
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and reference(spec['GPU_admission_file']) == gate['admission_ref'] and
            gate['common'].imported_files(torch, np, cv2, masks, COCO, gate['closure'], gate['admission'], ndimage) == libraries and
            process.cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
            'Final own SOURCE/library/resource binding changed')
    gate['lineage'].final_endpoint_check(gate['parent'])
    deadline(started, spec)
    report = dict(schema=REPORT_SCHEMA, status=STATUS, evidence_status='GPU_AP_PENDING_INDEPENDENT_CPU_REPLAY', source=gate['source'], spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'], phase=spec['phase'], RAW_inputs=spec['RAW_inputs'], normal_RAW_count=spec['normal_RAW_count'],
        terminal_identity_pins=spec['terminal_identity_pins'], additional_preserved_independent_CPU_failure=spec['additional_preserved_independent_CPU_failure'],
        total_failed_RAW_count=9, total_terminal_identity_pin_count=spec['normal_RAW_count'] + 9,
        stage24_GPU_source=spec['stage24_GPU_source'], stage24_GPU_spec=spec['stage24_GPU_spec'], stage24_GPU_report=spec['stage24_GPU_report'],
        stage24_GPU_RAW=spec['stage24_GPU_RAW'], stage24_GPU_admission=spec['stage24_GPU_admission'],
        stage24_CPU_source=spec['stage24_CPU_source'], stage24_CPU_spec=spec['stage24_CPU_spec'], stage24_CPU_RAW=spec['stage24_CPU_RAW'],
        stage24_CPU_admission=spec['stage24_CPU_admission'], stage24_CPU_qualification=spec['stage24_CPU_qualification'],
        stage24_checkpoints=spec['stage24_checkpoints'], stage24_CPU_endpoint_numeric_protocol=spec['stage24_CPU_endpoint_numeric_protocol'],
        carrier_spec=spec['carrier_spec'], carrier_evaluation_source=spec['carrier_evaluation_source'], evaluation_adapter_source=spec['evaluation_adapter_source'],
        seed=spec['seed'], original_parent_epochs=24, stage_epochs_completed=24, stage_updates_per_arm=39864,
        head_state_proofs=head_proofs, parameters=adapter.PARAMETERS, controls=effective['controls'], predictions=refs, control_results=results,
        images=187, raw_GT=380, queries_per_image=100, predictions_per_control=18700, DEV=context['DEV'],
        native_reference_predictions=context['native_reference_predictions'], packet_refs=gate['packets'], teacher_checkpoint=gate['cache_report']['teacher_checkpoint'],
        selector_mode='reuse', selector_manifest=gate['selector']['manifest_ref'], exact_shared_GPU_L_selector=True,
        capture_manifest=reference(output / 'CAPTURE_MANIFEST.json'), capture_refs=sink.refs, capture_schema=adapter.CAPTURE_SCHEMA,
        actual_serialized_capture_bytes=sink.serialized_bytes, raw_capture_tensor_payload_bytes=sink.payload_bytes, capture_budget=budget,
        start=reference(output / 'START.json'), trace=reference(output / 'TRACE.jsonl'), image_diagnostics=reference(output / 'IMAGE_DIAGNOSTICS.json'),
        support_formula='abs(actual_original_GPU_native_L)<=2', fixed_native_scores_class_order=True,
        no_GT_query_selection=True, no_NMS=True, no_score_cutoff=True, no_TEST=True, best_DEV_selection=False,
        native_model_loaded=False, native_forward_calls=0, original_JPEG_decoded=False, optimizer_created=False, weights_updated=False,
        current_owned_GPU_terminal_self_certified=False, independent_CPU_evaluation_replayed=False, CPU_computational_portability_established=False,
        CPU_exactmask_failure_policy='FAILED_CPU_REPLAY; ROOT inspect actual GPU captured logits and GPU predictions; no posterior tolerance relaxation; '
                                    'not automatic GPU AP invalidity or method failure',
        task_efficacy_established=False, innovation_established=False, generalization_established=False, capacity_confound_not_yet_controlled=True,
        peak_torch_allocated_bytes=peak, cleanup=cleanup, actual_imported_files=libraries, elapsed_seconds=time.monotonic() - started,
        costs=dict(all187_image_evaluation_seconds=sum(row['elapsed_seconds'] for row in diagnostics),
                   capture_files=187, capture_bytes=sink.serialized_bytes, original_parent_training_epochs=24, new_stage_training_epochs=24),
        evidence_limits='One seed, historical DEV fixed stage24 BASE/MEM with original frozen scores. GPU official AP pending independent same-input CPU replay; '
                        'CPU portability failures and GPU self-consistency are distinct checks. No blind TEST, external SOTA, causal routing attribution, '
                        'capacity control, deployment or innovation conclusion; current terminal is certified only by ROOT after actual exit/reap.')
    write_new(output / 'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, report=reference(output / 'REPORT.json'), independent_CPU_evaluation_replayed=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'UNBOUND_DRAFT: ROOT must close/audit stage24 GPU/CPU and bind this NEW evaluation before one owned attempt')
    run(args.spec)

