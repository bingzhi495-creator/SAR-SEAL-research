"""UNBOUND_DRAFT: N14 stage8 paired training endpoint independent CPU checks.

NOT RUN, SOURCE-unbound; ROOT must bind/review a new immutable source AFTER the
actual training GPU terminal/reap/birth absence and full SOURCE audit. No current
live scientific chain is read while developing this draft. No retry, training,
DEV/AP evaluation or GT decoder is implemented. CPU checks use qualified FIT
packets only, both arms with LOG input ABI. Reconstruct the common E24 warm heads,
verify private zero-addon initial digests, then strictly load NEW stage8 states.
40497 base and42698 addon state counts are checked separately.

Future CPU schema ROOT-N14-LOG-E24-memory-paired-warm-stage-CPU-spec-v1. Preserve
normal96/legacy failed8 pins104 plus additional failed9 pins105, all RAW-only
before the first SHA/import. Required training_GPU_source/spec/report/RAW/admission
refs; copied carrier_spec/carrier_evaluation_source/carrier_curve_qualified_summary,
memory_source/protocol/seed/module_seed/order_seed; entry_source/phase/CPU_admission_file,
Q_path/details_path/deadline_seconds1800/automatic_retryFalse;
CPU_resource_fields={affinity,thread_limit,deadline_seconds,automatic_retry,cuda_visible_devices}
with each value the ACTUAL admission JSON key path, matching[0,1],2,1800,False,'';
CUDA visibility must also be empty in the real child environment, and Torch must
stay uninitialized on CUDA. ROOT separately owns whole-GPU inactivity admission.
CPU_endpoint_numeric_protocol is a NEW independent CPU protocol reference,
required in sources; endpoint_numeric_source must equal it. Its JSON contains
atol/rtol=1e-4, locked_before_CPU_execution=True, locked_after_GPU_training=True,
scope='same cached input endpoint mean compatibility only, not AP/statistical equivalence',
and bound_completed_GPU_report must exactly equal training_GPU_report.
endpoint_comparison={atol:1e-4,rtol:1e-4}, CPU_numeric_fields={atol:['atol'],rtol:['rtol']}.
This standard is locked after completed GPU training, before independent CPU
execution; not a training efficacy gate and not preregistered before GPU training.
No endpoint per-instance GPU losses exist in this GPU ABI: CPU saves1661 losses
per arm but compares only final stage8 full-FIT means. UPDATE_TRACE online BCE
is audited for identity/order/count and exact GPU online means, never compared
to final-head CPU per-instance BCE. AdamW state/step and trace bookkeeping checks
are not independent replay of the training optimizer/gradient trajectory.
Actual lineage own_gate(cpu=True), inherited EVAL_REF/E24 endpoint_gate, factory,
parent evaluator/common cache ABI and resource JSON paths require ROOT review.
This file never certifies its own actual CPU terminal.
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
SCHEMA="ROOT-N14-LOG-E24-memory-paired-warm-stage-CPU-spec-v1"
STATUS="ROOT_N14_MEMORY_PAIRED_STAGE8_ENDPOINT_CPU_QUALIFIED_V1"

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
    require(type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 9 and
            type(spec['total_terminal_identity_pin_count']) is int and spec['total_terminal_identity_pin_count'] == 105 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 remains separate from normal96 plus legacy failed8 pins104; total105')
    return raw


def pre_gate(spec, minimum):
    # Inherited RAW/source contract of the original GPU pre_gate, with
    # minimum94 supplied here. Additional failed9 has ALREADY passed RAW-only.
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
            'Original normal94 and failed8 owners must remain distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in all_rows},
            'Original normal-plus-failed8 pins must remain unchanged')
    for row, raw in zip(all_rows, normals + failures):
        check_pin(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    require(expected == 96 and len(pins) == 104, 'Current normal96/legacy pins104 required')
    # FIRST HASH: failed9 and all original normal94/failed8 RAW checks passed.
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


GPU_SCHEMA = 'ROOT-N14-LOG-E24-memory-paired-warm-stage-GPU-spec-v1'
GPU_STATUS = 'N14_MEMORY_PAIRED_STAGE_LEAF_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION'
CHECKPOINT_SCHEMA = 'N14-context-memory-stage-checkpoint-v1'
PARAMETERS = {'BASE_CONTINUATION': 40497, 'MEMORY_ADDON': 42698}
BASE_SHAPES = ((91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,))


def deadline(started, spec):
    require(time.monotonic() - started < spec['deadline_seconds'], 'CPU endpoint deadline exceeded; no retry')


def lookup(value, keys, label):
    require(type(keys) is list and keys and all(type(key) is str and key for key in keys),
            'ROOT must provide an actual explicit JSON key path: ' + label)
    for key in keys:
        require(type(value) is dict and key in value, 'ROOT must adapt actual JSON ABI: ' + label + '.' + key)
        value = value[key]
    return value


def cpu_tensor(torch, value, name, shape, dtype):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.layout == torch.strided and tuple(value.shape) == tuple(shape) and value.dtype == dtype and
            bool(torch.isfinite(value).all()), 'CPU finite tensor descriptor differs: ' + name)


def source_gate(path, spec):
    require(spec['schema'] == SCHEMA and spec['seed'] == 202610120701 and
            spec['module_seed'] == spec['order_seed'] == 202610140801 and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds'] == 1800 and
            spec['automatic_retry'] is False, 'Exact prelocked stage8 endpoint CPU spec required')
    ninth_failed_raw_only(spec)
    normals, closure = pre_gate(spec, 96)
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    for key in ('entry_source', 'training_GPU_source', 'training_GPU_spec', 'training_GPU_report',
                'training_GPU_RAW', 'training_GPU_admission', 'carrier_spec', 'carrier_evaluation_source',
                'carrier_curve_qualified_summary', 'memory_source', 'protocol', 'endpoint_numeric_source',
                'CPU_endpoint_numeric_protocol'):
        closed(closure, spec[key])
    require(Path(spec['entry_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True),
            'Actual new ROOT-bound CPU source identity differs')
    gpu_spec = read(closed(closure, spec['training_GPU_spec']))
    report = read(closed(closure, spec['training_GPU_report']))
    require(gpu_spec['schema'] == report['schema'] == GPU_SCHEMA and report['status'] == GPU_STATUS and
            gpu_spec['entry_source'] == report['source'] == spec['training_GPU_source'] and
            report['spec'] == spec['training_GPU_spec'] and report['GPU_admission'] == spec['training_GPU_admission'] and
            report['phase'] == gpu_spec['phase'] and
            Path(gpu_spec['GPU_admission_file']).resolve(strict=True) == Path(spec['training_GPU_admission']['path']).resolve(strict=True),
            'Actual completed training GPU report/source/spec/admission binding differs')
    require(type(gpu_spec['normal_RAW_count']) is int and gpu_spec['normal_RAW_count'] == 95 and
            len(gpu_spec['terminal_identity_pins']) == 103 and gpu_spec['RAW_inputs'] == spec['RAW_inputs'][:95] and
            gpu_spec['terminal_identity_pins'] == {key: pin for key, pin in spec['terminal_identity_pins'].items()
                if key != spec['training_GPU_RAW']['path']} and
            gpu_spec['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
            gpu_spec['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
            gpu_spec['additional_preserved_independent_CPU_failure'] == spec['additional_preserved_independent_CPU_failure'] and
            gpu_spec['total_failed_RAW_count'] == 9 and gpu_spec['total_terminal_identity_pin_count'] == 104,
            'Exact GPU normal95/legacy103 plus separate failed9 ancestry must be preserved')
    last = spec['RAW_inputs'][95]
    require(last['receipt'] == spec['training_GPU_RAW'] and last['admission'] == spec['training_GPU_admission'] and
            last['phase'] == gpu_spec['phase'] and normals[95]['admission_sha256'] == spec['training_GPU_admission']['sha256'],
            'Actual training GPU terminal must be the latest normal96 owner')
    for key in ('carrier_spec', 'carrier_evaluation_source', 'carrier_curve_qualified_summary', 'memory_source',
                'protocol', 'seed', 'module_seed', 'order_seed'):
        require(gpu_spec[key] == spec[key], 'Exact original training protocol field differs: ' + key)
    require(gpu_spec['first_stage_epoch'] == 1 and gpu_spec['last_stage_epoch'] == 8 and
            gpu_spec['fixed_stage_epochs'] == 24 and gpu_spec['optimizer_reset_both'] is True and
            gpu_spec['deadline_seconds'] == 1800 and gpu_spec['automatic_retry'] is False and
            gpu_spec['allocator_limit_MiB'] == 4096 and gpu_spec['whole_GPU_limit_MiB'] == 8192 and
            report['original_parent_epochs'] == 24 and report['new_stage_epochs'] == 8 and
            report['fixed_stage_epochs'] == 24 and report['updates_per_arm'] == 13288 and
            report['own_terminal_self_certified'] is False and report['independent_CPU_qualified'] is False and
            report['no_DEV'] is True and report['no_TEST'] is True and report['AP_evaluated'] is False and
            report['capacity_confound_not_yet_controlled'] is True and
            type(report['elapsed_seconds']) in (int, float) and 0 <= report['elapsed_seconds'] < 1800 and
            type(report['peak_torch_allocated_bytes']) is int and 0 <= report['peak_torch_allocated_bytes'] <= 4096 * 1024**2,
            'Exact paired stage8 budget, resource evidence and limited report scope required')
    require(set(report['checkpoints']) == set(ARMS), 'Exactly two actual stage8 checkpoints required')
    for ref in list(report['checkpoints'].values()) + [report['trace'], report['start']]:
        closed(closure, ref)
    carrier = read(closed(closure, spec['carrier_spec']))
    curve = read(closed(closure, spec['carrier_curve_qualified_summary']))
    curve_report = read(closed(closure, curve['evaluation_GPU_report']))
    require(carrier['seed'] == spec['seed'] and carrier['head_endpoint_epoch'] == 24 and
            carrier['entry_source'] == spec['carrier_evaluation_source'] and
            curve['status'] == 'QUALIFIED_E24_CURVE_NOT_E48_PRIMARY_CLAIM' and curve['single_seed_AP_qualified'] is True and
            curve['evaluation_GPU_actual_returncode'] == curve['evaluation_CPU_actual_returncode'] == 0 and
            curve_report['spec'] == spec['carrier_spec'] and curve_report['seed'] == spec['seed'] and
            curve_report['head_endpoint_epoch'] == 24, 'Qualified common E24 carrier parent differs')
    require(all(ref in closure.values() for payload in (gpu_spec, carrier) for ref in payload['sources']),
            'Complete original parent/training SOURCE closure required')
    context = dict(carrier)  # Keep inherited EVAL_REF and E24 recipe for parent gate.
    for key in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'terminal_identity_pins', 'entry_source', 'phase', 'CPU_admission_file', 'Q_path', 'details_path', 'deadline_seconds'):
        context[key] = spec[key]
    lineage = load(carrier['endpoint_lineage_source'], '_N14_stage8_CPU_parent_lineage')
    gate = api(lineage, 'own_gate')(path, context, __file__, closure, normals, cpu=True)
    index = api(lineage, 'owner_ref_join')(context, gate['owners'], spec['training_GPU_RAW'],
        spec['training_GPU_admission'], spec['training_GPU_source'], spec['training_GPU_spec'], '0')
    require(type(index) is int and index == 95 and gpu_spec['RAW_inputs'] == spec['RAW_inputs'][:index],
            'Actual final training GPU0 chronological source/spec/admission join differs')
    endpoint = api(lineage, 'endpoint_gate')(context, closure, gate['owners'])
    gate.update(endpoint)
    gate.update(context=context, closure=closure, lineage=lineage, cache_Q=endpoint['cache_q'], parent=endpoint['parent'],
                training_spec=gpu_spec, training_report=report)
    require(len(gate['cache_report']['packet_refs']) == len(gate['cache_Q']['packet_refs']) == 741 and
            gate['cache_report']['packet_refs'] == gate['cache_Q']['packet_refs'] and gate['cache_report']['raw_GT'] == 1661,
            'Original qualified FIT741/1661 cache/order required')
    fields = spec['CPU_resource_fields']
    wanted = dict(affinity=[0, 1], thread_limit=2, deadline_seconds=1800, automatic_retry=False, cuda_visible_devices='')
    require(type(fields) is dict and set(fields) == set(wanted), 'ROOT must map actual CPU-only admission resource JSON fields')
    for key, expected in wanted.items():
        actual = lookup(gate['admission'], fields[key], 'CPU admission ' + key)
        require(type(actual) is type(expected) and actual == expected, 'Actual CPU admission resource contract differs: ' + key)
    require(spec['endpoint_numeric_source'] == spec['CPU_endpoint_numeric_protocol'],
            'Endpoint tolerance must bind the new post-GPU/pre-CPU numeric protocol; no old GPU prelock claim')
    numeric = read(closed(closure, spec['endpoint_numeric_source']))
    numeric_fields = spec['CPU_numeric_fields']
    require(type(numeric) is dict and set(numeric) == {'atol', 'rtol', 'locked_before_CPU_execution',
                'locked_after_GPU_training', 'scope', 'bound_completed_GPU_report'} and
            numeric['bound_completed_GPU_report'] == spec['training_GPU_report'] and
            numeric['locked_before_CPU_execution'] is True and
            numeric['locked_after_GPU_training'] is True and
            numeric['scope'] == 'same cached input endpoint mean compatibility only, not AP/statistical equivalence' and
            numeric_fields == {'atol': ['atol'], 'rtol': ['rtol']} and
            spec['endpoint_comparison'] == {'atol': 1e-4, 'rtol': 1e-4} and
            all(lookup(numeric, numeric_fields[key], 'endpoint ' + key) == 1e-4 for key in ('atol', 'rtol')),
            'New post-GPU/pre-CPU endpoint numeric protocol values/timing/scope must be explicit and bound')
    start = read(closed(closure, report['start']))
    require(start['schema'] == GPU_SCHEMA and start['source'] == spec['training_GPU_source'] and
            start['spec'] == spec['training_GPU_spec'] and start['GPU_admission'] == spec['training_GPU_admission'] and
            start['phase'] == gpu_spec['phase'] and start['parent'] == spec['carrier_spec'] and
            start['first_stage_epoch'] == 1 and start['last_stage_epoch'] == 8 and start['stage_epochs'] == 24 and
            start['optimizer_reset_both'] is True and start['original_parent_epochs'] == 24 and
            start['continuous_parent_plus_stage'] is False and start['native_weights_updated'] is False and
            start['no_DEV'] is True and start['no_TEST'] is True and start['no_AP'] is True and
            start['initial_state_digests'] == report['initial_state_digests'] and start['behavior'] == report['behavior_preflight'],
            'Actual training START warm/reset/initial proof differs')
    gate.update(start=start, endpoint_numeric_protocol=numeric,
                evaluator=load(spec['carrier_evaluation_source'], '_N14_CPU_common_E24_loader'),
                factory=load(context['point_heads_source'], '_N14_CPU_original_head_factory'),
                memory_module=load(spec['memory_source'], '_N14_CPU_memory_head'))
    return gate


def reconstruct_heads(torch, gate, spec):
    outside = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        old_heads, parent_states, proofs = api(gate['evaluator'], 'load_heads')(torch, gate['context'], gate)
        require(proofs['LOG_PR'] == gate['start']['common_LOG_parent_proofs'], 'Exact common qualified E24 parent proof differs')
        parent = parent_states['LOG_PR']
        require(len(parent) == 8 and sum(v.numel() for v in parent.values()) == 40497, 'Parent E24 LOG has40497 scalars')
        for name, value in parent.items():
            cpu_tensor(torch, value, 'parent.' + name, value.shape, torch.float32)
        base = api(gate['factory'], 'make_point_head')(torch, spec['seed'])
        memory_base = api(gate['factory'], 'make_point_head')(torch, spec['seed'])
        base.load_state_dict(parent, strict=True)
        memory_base.load_state_dict(parent, strict=True)
        require(base is not memory_base, 'Paired heads must use distinct base objects')
        memory = api(gate['memory_module'], 'make_context_memory_head')(torch, memory_base,
            seed=spec['module_seed'], tokens=8, dim=8, temperature=0.25)
        heads = dict(BASE_CONTINUATION=base, MEMORY_ADDON=memory)
        require(tuple(tuple(p.shape) for p in base.parameters()) == BASE_SHAPES and
                sum(p.numel() for p in base.parameters()) == 40497 and sum(p.numel() for p in memory.parameters()) == 42698 and
                memory.base is memory_base and not tuple(base.buffers()) and not tuple(memory.buffers()),
                'Separate actual40497 base/42698 addon architectures differ')
        require(bool((memory.output.weight == 0).all()) and bool((memory.output.bias == 0).all()),
                'Actual reconstructed addon must start with zero output projection')
        initial = {arm: {name: digest(value) for name, value in head.state_dict().items()} for arm, head in heads.items()}
        require(initial == gate['training_report']['initial_state_digests'] and
                all(initial[ARMS[0]][name] == initial[ARMS[1]]['base.' + name] == digest(value) for name, value in parent.items()),
                'Both exact common E24 warm states and six deterministic addon initial tensors must reproduce GPU proof')
        del old_heads, parent_states
    require(torch.equal(outside, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'Factory/private zero-addon construction changed outside RNG/CUDA')
    return heads, initial, proofs


def checkpoint_checks(torch, gate, spec, heads, initial):
    report = gate['training_report']
    checkpoints, summaries = {}, {}
    for arm in ARMS:
        cp = torch.load(str(closed(gate['closure'], report['checkpoints'][arm])), map_location='cpu')
        require(type(cp) is dict and set(cp) == {'meta', 'state_dict', 'optimizer_state_dict', 'private_order_state'},
                'Actual stage8 checkpoint schema differs')
        meta = cp['meta']
        expected = dict(schema=CHECKPOINT_SCHEMA, arm=arm, source=spec['training_GPU_source'], spec=spec['training_GPU_spec'],
            GPU_admission=spec['training_GPU_admission'], phase=gate['training_spec']['phase'], parent_spec=spec['carrier_spec'],
            parent_head_epochs=24, common_parent_arm='LOG_PR', stage_epochs_completed=8, stage_updates=13288,
            fixed_stage_epochs=24, optimizer_reset_at_stage_start=True, memory_source=spec['memory_source'],
            head_factory_source=gate['context']['point_heads_source'], parameters=PARAMETERS[arm],
            initial_state_digests=initial[arm], history=report['history'], gradient_activity=report['gradient_activity'][arm],
            cache_CPU_qualification=gate['context']['cache_CPU_qualification'], natural_query_scores_changed=False,
            support_changed=False, no_DEV=True, no_TEST=True, no_AP=True, native_weights_updated=False, independent_qualified=False)
        require(type(meta) is dict and all(meta[key] == value and type(meta[key]) is type(value) for key, value in expected.items()),
                'Actual checkpoint stage8/parent/budget/provenance metadata differs: ' + arm)
        head = heads[arm]
        state = cp['state_dict']
        require(type(state) is dict and set(state) == set(head.state_dict()) and len(state) == (8 if arm == ARMS[0] else 14) and
                sum(v.numel() for v in state.values()) == PARAMETERS[arm], 'Arm-specific new trained parameter count differs')
        for name, expected_value in head.state_dict().items():
            cpu_tensor(torch, state[name], arm + '.' + name, expected_value.shape, torch.float32)
        head.load_state_dict(state, strict=True)
        require(all(torch.equal(value, head.state_dict()[name]) for name, value in state.items()), 'Exact loaded new state differs')
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        # Parameter traversal comes from the actual reconstructed module, never
        # state_dict key order (memory_keys/values precede base child parameters).
        named = list(head.named_parameters())
        optimizer = cp['optimizer_state_dict']
        require(type(optimizer) is dict and set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1,
                'One fresh AdamW parameter group/state required')
        group = optimizer['param_groups'][0]
        ids = group['params']
        require(type(ids) is list and ids == list(range(len(named))) and set(optimizer['state']) == set(ids),
                'Actual AdamW parameter IDs must map to complete actual module traversal')
        adam = dict(lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                    maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
        require(all(key in group and group[key] == value for key, value in adam.items()), 'Original reset AdamW recipe differs')
        activity = meta['gradient_activity']
        require(set(activity) == {name for name, p in named}, 'Complete per-parameter gradient bookkeeping required')
        for parameter_id, (name, parameter) in zip(ids, named):
            item = optimizer['state'][parameter_id]
            require(type(item) is dict and set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Exact AdamW state fields required')
            cpu_tensor(torch, item['step'], arm + '.' + name + '.step', (), torch.float32)
            require(float(item['step'].item()) == 13288.0, 'Every actual AdamW step must equal13288 exactly; no truncation')
            for field in ('exp_avg', 'exp_avg_sq'):
                cpu_tensor(torch, item[field], arm + '.' + name + '.' + field, parameter.shape, torch.float32)
            require(bool((item['exp_avg_sq'] >= 0).all()), 'AdamW second moment must be nonnegative')
            record = activity[name]
            require(type(record) is dict and set(record) == {'present', 'nonzero'} and
                    type(record['present']) is int and record['present'] == 13288 and
                    type(record['nonzero']) is int and 0 <= record['nonzero'] <= 13288,
                    'Actual gradients must be present every update; no per-step nonzero assertion is made')
        require(isinstance(cp['private_order_state'], torch.Tensor) and cp['private_order_state'].device.type == 'cpu' and
                cp['private_order_state'].dtype == torch.uint8 and cp['private_order_state'].ndim == 1, 'CPU private order RNG state required')
        checkpoints[arm] = cp
        summaries[arm] = dict(parameters=PARAMETERS[arm], parameter_tensors=len(named), exact_new_state_loaded=True,
            AdamW_all_parameter_steps=13288, AdamW_state_shape_dtype_finite_checked=True,
            gradient_activity=activity, optimizer_trajectory_independently_replayed=False)
    require(checkpoints[ARMS[0]]['meta']['epoch_orders'] == checkpoints[ARMS[1]]['meta']['epoch_orders'] and
            torch.equal(checkpoints[ARMS[0]]['private_order_state'], checkpoints[ARMS[1]]['private_order_state']),
            'Paired arms must preserve identical private orders/final private RNG')
    return checkpoints, summaries


def checked_orders(torch, checkpoints, report, spec):
    orders = checkpoints[ARMS[0]]['meta']['epoch_orders']
    history = report['history']
    require(type(orders) is list and type(history) is list and len(orders) == len(history) == 8, 'Actual eight epochs required')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(spec['order_seed'])
    for epoch, (row, h) in enumerate(zip(orders, history), 1):
        actual_order = torch.randperm(741, generator=generator).tolist()
        sha = hashlib.sha256(json.dumps(actual_order, separators=(',', ':')).encode('ascii')).hexdigest()
        require(type(row) is dict and set(row) == {'stage_epoch', 'indices', 'sha256'} and
                row == dict(stage_epoch=epoch, indices=actual_order, sha256=sha) and
                len(set(actual_order)) == 741 and set(actual_order) == set(range(741)) and
                h['stage_epoch'] == epoch and h['paired_image_order_sha256'] == sha and
                h['updates_per_arm'] == 1661 and h['cumulative_stage_updates'] == epoch * 1661 and
                set(h['online_BCE']) == set(h['full_FIT_BCE']) == set(ARMS) and
                all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for key in ('online_BCE', 'full_FIT_BCE') for value in h[key].values()), 'Exact private eight741 permutations/history differ')
    require(all(torch.equal(generator.get_state(), cp['private_order_state']) for cp in checkpoints.values()),
            'Private order RNG must match exactly after eight sequential randperm741 draws')
    return orders


def trained_memory_single_instance_behavior(torch, gate, heads, started, spec):
    """One REAL first FIT instance of the LOADED stage8 memory head, not init.

    A nonzero trained output-weight projection ensures the retrieval path is
    exercised. Compatible point permutation is an ABI/behavior check on this
    instance only; no full-FIT, causal or AP conclusion is inferred.
    """
    deadline(started, spec)
    value = api(gate['common'], 'load_packet')(torch, gate['cache_report'], 0)
    require(len(value['query_ids']) > 0, 'First actual FIT packet must contain the declared first instance')
    F, e, L, y, flag = api(gate['common'], 'instance_inputs')(torch, value, 0, 'LOG_PR', 'cpu')
    require(L.numel() > 0 and all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' for v in (F, e, L, y, flag)),
            'First qualified nonempty FIT LOG CPU instance required')
    head = heads['MEMORY_ADDON']
    projection_active = bool((head.output.weight != 0).any())
    require(projection_active, 'Loaded stage8 retrieval projection is zero; an active retrieval behavior check cannot be claimed')
    patch = L.new_zeros((L.numel(), 49))
    permutation = torch.arange(L.numel() - 1, -1, -1, dtype=torch.int64, device='cpu')
    with torch.no_grad():
        baseline = head.base(F, e, L, patch, arm='PR')
        original = head(F, e, L, patch, arm='PR')
        permuted = head(F[permutation], e[permutation], L[permutation], patch[permutation], arm='PR')
        require(bool(torch.isfinite(original['residual']).all()) and bool(torch.isfinite(permuted['residual']).all()) and
                torch.allclose(permuted['residual'], original['residual'][permutation], atol=1e-4, rtol=1e-4),
                'Loaded trained memory residual violates CPU-before-execution1e-4 point-permutation compatibility')
        gated_original = api(gate['common'], 'region_logits')(torch, head, F, e, L, flag)
        reversed_flag, reversed_L = flag[permutation], L[permutation]
        gated_permuted = api(gate['common'], 'region_logits')(torch, head, F[permutation], e[permutation], reversed_L, reversed_flag)
        require(torch.equal(gated_original['logits'][~flag], L[~flag]) and
                torch.equal(gated_permuted['logits'][~reversed_flag], reversed_L[~reversed_flag]) and
                torch.allclose(gated_permuted['logits'], gated_original['logits'][permutation], atol=1e-4, rtol=1e-4),
                'Loaded trained memory LOG exterior/native identity or permuted output differs')
        correction = original['residual'] - baseline['residual']
        record = dict(image_index=0, GT_index=0, annotation_id=int(value['identity']['ann_ids'][0]),
            query_id=int(value['query_ids'][0]), point_count=int(L.numel()), outside_LOG_points=int((~flag).sum()),
            trained_output_projection_nonzero=projection_active,
            actual_addon_correction_max_abs=float(correction.abs().max()),
            actual_addon_correction_nonzero_points=int((correction != 0).sum()),
            residual_permutation_max_abs_error=float((permuted['residual'] - original['residual'][permutation]).abs().max()),
            gated_logit_permutation_max_abs_error=float((gated_permuted['logits'] - gated_original['logits'][permutation]).abs().max()),
            atol=1e-4, rtol=1e-4, compatible=True, outside_LOG_native_exact=True,
            behavior_numeric_criterion_locked_in_CPU_source_before_execution=True,
            behavior_numeric_criterion_preregistered_before_GPU_training=False,
            checkpoint_stage=8, loaded_trained_head_checked=True, initial_zero_projection_head_used=False,
            full_FIT_permutation_behavior_checked=False,
            scope='Only first actual FIT packet/instance of the loaded stage8 MEMORY_ADDON; active projection, '
                  'point-permutation compatibility and LOG exterior identity; no global behavior/efficacy claim.')
    del value
    require(not torch.cuda.is_initialized(), 'Single-instance trained behavior check initialized CUDA')
    return record


def measure_CPU(torch, gate, heads, started, spec):
    losses = {arm: [] for arm in ARMS}
    instances = []
    image_ids = []
    with torch.no_grad():
        for index in range(741):
            deadline(started, spec)
            require(gate['cache_report']['packet_refs'][index] == gate['cache_Q']['packet_refs'][index], 'Immutable qualified cache order differs')
            value = api(gate['common'], 'load_packet')(torch, gate['cache_report'], index)
            qids = value['query_ids']
            ann_ids = value['identity']['ann_ids']
            require(len(qids) == len(ann_ids), 'Actual cache instance identity vectors differ')
            image_ids.append([dict(annotation_id=int(ann_ids[g]), query_id=int(qids[g])) for g in range(len(qids))])
            for g in range(len(qids)):
                deadline(started, spec)
                F, e, L, y, flag = api(gate['common'], 'instance_inputs')(torch, value, g, 'LOG_PR', 'cpu')
                require(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' for v in (F, e, L, y, flag)),
                        'Qualified LOG FIT operands must all be material CPU tensors')
                record = dict(image_index=index, GT_index=g, annotation_id=int(ann_ids[g]), query_id=int(qids[g]), BCE={})
                for arm in ARMS:
                    result = api(gate['common'], 'region_logits')(torch, heads[arm], F, e, L, flag)
                    require(torch.equal(result['logits'][~flag], L[~flag]) and
                            bool(torch.isfinite(result['logits']).all()), 'Actual LOG exterior/finite output contract differs')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(result['logits'], y)
                    require(bool(torch.isfinite(loss)) and not loss.requires_grad, 'Finite inference-only CPU BCE required')
                    observed = float(loss.item())
                    losses[arm].append(observed)
                    record['BCE'][arm] = observed
                instances.append(record)
            del value
            require(not torch.cuda.is_initialized(), 'CPU endpoint pass must never initialize CUDA')
            if (index + 1) % 25 == 0 or index == 740:
                print(json.dumps(dict(CPU_FIT_images=index + 1, CPU_FIT_instances=len(instances))), flush=True)
    require(len(instances) == 1661 and all(len(values) == 1661 for values in losses.values()), 'Complete actual FIT741/1661 CPU endpoint required')
    means = {arm: math.fsum(values) / 1661 for arm, values in losses.items()}
    comparisons = {}
    for arm in ARMS:
        expected = gate['training_report']['history'][-1]['full_FIT_BCE'][arm]
        error = abs(means[arm] - expected)
        allowed = 1e-4 + 1e-4 * abs(expected)
        require(error <= allowed, 'Final stage8 full-FIT CPU/GPU mean BCE exceeds new post-GPU/pre-CPU1e-4 compatibility standard: ' + arm)
        comparisons[arm] = dict(CPU_full_FIT_BCE=means[arm], GPU_full_FIT_BCE=expected,
            absolute_error=error, allowed_atol_plus_rtol_abs_GPU=allowed, passed=True,
            numeric_protocol=spec['CPU_endpoint_numeric_protocol'],
            numeric_lock_timing='locked after completed GPU training, before independent CPU execution; not training efficacy gate',
            comparison_scope='final stage8 full-FIT1661 mean only; no per-instance GPU endpoint losses stored')
    return instances, image_ids, comparisons


def trace_check(gate, orders, image_ids, started, spec):
    require(len(image_ids) == 741 and sum(len(rows) for rows in image_ids) == 1661, 'Qualified741/1661 trace identity map required')
    updates = 0
    summary = []
    def reject(value):
        raise ValueError('Nonfinite update trace JSON: ' + value)
    with closed(gate['closure'], gate['training_report']['trace']).open('r', encoding='utf-8') as stream:
        for epoch, order in enumerate(orders, 1):
            values = {arm: [] for arm in ARMS}
            for index in order['indices']:
                deadline(started, spec)
                for g, identity in enumerate(image_ids[index]):
                    line = stream.readline()
                    require(bool(line.strip()), 'Actual paired update trace ended before13288 rows')
                    row = json.loads(line, parse_constant=reject)
                    updates += 1
                    expected = dict(stage_epoch=epoch, image_index=index, GT_index=g,
                        annotation_id=identity['annotation_id'], query_id=identity['query_id'], stage_update=updates)
                    require(type(row) is dict and set(row) == set(expected) | {'BCE'} and
                            all(type(row[key]) is int and row[key] == value for key, value in expected.items()) and
                            type(row['BCE']) is dict and set(row['BCE']) == set(ARMS), 'Actual paired trace order/identity/stage update differs')
                    for arm in ARMS:
                        value = row['BCE'][arm]
                        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Finite recorded online BCE required')
                        values[arm].append(value)
            require(all(len(rows) == 1661 for rows in values.values()), 'Exactly1661 paired rows per epoch required')
            online = {arm: math.fsum(rows) / 1661 for arm, rows in values.items()}
            require(online == gate['training_report']['history'][epoch - 1]['online_BCE'], 'Actual trace online means must reproduce GPU history exactly')
            summary.append(dict(stage_epoch=epoch, paired_rows=1661, online_BCE=online,
                order_sha256=order['sha256'], trace_identity_and_count_exact=True))
        require(stream.readline() == '' and updates == 13288, 'One paired trace must contain exactly13288 rows, not26576')
    return summary


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec)
    Q_path, details_path = Path(spec['Q_path']), Path(spec['details_path'])
    require(Q_path.is_absolute() and details_path.is_absolute() and Q_path != details_path and
            not Q_path.exists() and not details_path.exists() and Q_path.parent.is_dir() and details_path.parent.is_dir(),
            'Fresh distinct ROOT-owned absolute qualification/details outputs required')
    require(all(os.environ.get(key) == '2' for key in THREAD_ENV) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Pre-import numeric threads must be2 and actual CUDA visibility must be empty')
    import psutil
    import torch
    import numpy as np
    require(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5' and
            psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'Qualified actual CPU01/cold Torch/numeric versions required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Exactly two Torch CPU threads required')
    imported = imported_files(psutil, torch, np, gate)
    outside = torch.random.get_rng_state().clone()
    heads, initial, parent_proofs = reconstruct_heads(torch, gate, spec)
    checkpoints, checkpoint_summary = checkpoint_checks(torch, gate, spec, heads, initial)
    orders = checked_orders(torch, checkpoints, gate['training_report'], spec)
    frozen = {arm: {name: value.detach().clone() for name, value in head.state_dict().items()} for arm, head in heads.items()}
    trained_behavior = trained_memory_single_instance_behavior(torch, gate, heads, started, spec)
    instances, image_ids, comparisons = measure_CPU(torch, gate, heads, started, spec)
    trace_summary = trace_check(gate, orders, image_ids, started, spec)
    require(all(torch.equal(value, heads[arm].state_dict()[name]) for arm, state in frozen.items() for name, value in state.items()) and
            torch.equal(outside, torch.random.get_rng_state()) and not torch.cuda.is_initialized() and
            psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
            'mmdet' not in sys.modules, 'Frozen new endpoint/state/outside RNG/CPU-only contract changed')
    require(imported_files(psutil, torch, np, gate) == imported, 'Actual imported library closure changed')
    for ref in spec['sources']:
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
            reference(spec['CPU_admission_file']) == gate['admission_ref'], 'Current source/spec/admission binding changed')
    deadline(started, spec)
    details = dict(schema='N14-stage8-endpoint-CPU-details-v1', source=gate['source'], spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'], phase=spec['phase'], training_GPU_report=spec['training_GPU_report'],
        checkpoint_checks=checkpoint_summary, common_E24_parent_proofs=parent_proofs['LOG_PR'],
        initial_state_digests=initial, trained_memory_first_instance_behavior=trained_behavior,
        CPU_endpoint_numeric_protocol=spec['CPU_endpoint_numeric_protocol'],
        endpoint_numeric_protocol=gate['endpoint_numeric_protocol'],
        endpoint_numeric_lock_timing='locked after completed GPU training, before independent CPU execution; not training efficacy gate',
        endpoint_numeric_preregistered_before_GPU_training=False,
        final_endpoint_BCE_comparisons=comparisons, CPU_FIT_instance_losses=instances,
        qualified_cache_image_instance_identities=image_ids, private_epoch_orders=orders, trace_epoch_checks=trace_summary,
        FIT_images=741, FIT_instances=1661, paired_updates=13288, stage_epochs=8, fixed_stage_epochs=24,
        per_instance_GPU_CPU_endpoint_losses_compared=False, optimizer_trajectory_independently_replayed=False,
        no_DEV=True, no_TEST=True, AP_evaluated=False)
    write_new(details_path, details)
    deadline(started, spec)
    qualification = dict(schema='N14-stage8-training-endpoint-CPU-qualification-v1', status=STATUS,
        source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
        details=reference(details_path), training_GPU_source=spec['training_GPU_source'], training_GPU_spec=spec['training_GPU_spec'],
        training_GPU_report=spec['training_GPU_report'], training_GPU_RAW=spec['training_GPU_RAW'],
        training_GPU_admission=spec['training_GPU_admission'], carrier_spec=spec['carrier_spec'],
        carrier_curve_qualified_summary=spec['carrier_curve_qualified_summary'], memory_source=spec['memory_source'], protocol=spec['protocol'],
        RAW_inputs=spec['RAW_inputs'], normal_RAW_count=96, terminal_identity_pins=spec['terminal_identity_pins'],
        additional_preserved_independent_CPU_failure=spec['additional_preserved_independent_CPU_failure'],
        total_failed_RAW_count=9, total_terminal_identity_pin_count=105, actual_training_GPU0_latest_owner96_joined=True,
        original_parent_epochs=24, new_stage_epochs=8, fixed_stage_epochs=24, paired_updates_per_arm=13288,
        checkpoints=gate['training_report']['checkpoints'], parameters=PARAMETERS, common_E24_warm_initialization_exact=True,
        deterministic_zero_addon_initialization_exact=True, all_new_state_FP32_finite_shape_exact=True,
        all_AdamW_parameter_steps_exact13288=True, private_eight741_orders_and_final_state_exact=True,
        trained_MEMORY_ADDON_first_FIT_instance_point_permutation_compatible=True,
        trained_memory_first_instance_behavior=trained_behavior, full_FIT_permutation_behavior_checked=False,
        complete_paired13288_trace_order_identity_online_means_exact=True, FIT_images=741, FIT_instances=1661,
        final_stage8_full_FIT_mean_BCE_CPU_GPU_compatible=True, endpoint_comparison={'atol': 1e-4, 'rtol': 1e-4},
        CPU_endpoint_numeric_protocol=spec['CPU_endpoint_numeric_protocol'], endpoint_numeric_source=spec['endpoint_numeric_source'],
        CPU_numeric_fields=spec['CPU_numeric_fields'],
        endpoint_numeric_bound_completed_GPU_report=gate['endpoint_numeric_protocol']['bound_completed_GPU_report'],
        endpoint_numeric_lock_timing='locked after completed GPU training, before independent CPU execution; not training efficacy gate',
        endpoint_numeric_scope=gate['endpoint_numeric_protocol']['scope'],
        endpoint_numeric_locked_before_CPU_execution=True, endpoint_numeric_locked_after_GPU_training=True,
        endpoint_numeric_preregistered_before_GPU_training=False, endpoint_numeric_training_efficacy_gate=False,
        endpoint_BCE=comparisons, per_instance_GPU_CPU_endpoint_losses_compared=False,
        optimizer_trajectory_independently_replayed=False, CPU_only_no_CUDA_initialization=True,
        actual_imported_library_refs=imported, elapsed_seconds=time.monotonic() - started,
        no_DEV=True, no_TEST=True, AP_evaluated=False, native_weights_updated=False,
        teacher_or_native_backbone_reinitialized=False,
        unfinished_stage9_through24_qualified=False, capacity_confound_not_yet_controlled=True,
        current_owned_CPU_terminal_self_certified=False, deployment_gain_established=False, innovation_established=False,
        scope='First701 E24 warm parent; new paired stage1--8 of fixed24,40497/42698 parameters. '
              'Independent final FIT mean BCE compatibility under a post-GPU/pre-CPU locked protocol '
              '(not a training efficacy gate or pre-GPU preregistration), and checkpoint/order/trace bookkeeping only; '
              'no independent optimizer trajectory replay, per-instance GPU endpoint comparison, DEV/AP gain, '
              'causal mechanism, completed24-stage, deployment or innovation conclusion.')
    write_new(Q_path, qualification)
    print(json.dumps(dict(status=STATUS, qualification=reference(Q_path), current_owned_CPU_terminal_self_certified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'UNBOUND_DRAFT: ROOT must terminate/audit the real GPU, review/bind a NEW CPU source and own the single attempt')
    run(args.spec)

