"""UNBOUND / UNEXECUTED N14 paired stage continuation, leaf9--16 or17--24.
Only ordinary draft text was used to prepare this file. ROOT must review/adapt
the independent CPU qualification ABI and bind the actual terminal parent.
No scientific imports/file access at module import. Original LOG mathematics,
cache, score policy and stage24 endpoint stay fixed. Both trained N14 arm states,
AdamW states, private order state and complete stage history are resumed; E24
remains provenance only. This leaf never resets the stage optimizer or reruns
the first leaf's zero-addon identity/gradient preflight on trained states.
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
SCHEMA="ROOT-N14-LOG-E24-memory-paired-stage-continuation-GPU-spec-v2"
FIRST_STAGE_SCHEMA="ROOT-N14-LOG-E24-memory-paired-warm-stage-GPU-spec-v1"
STATUS="N14_MEMORY_PAIRED_STAGE_LEAF_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION"

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


def exact_adamw_step(torch, value, expected):
    require(type(expected) is int and expected >= 0, 'Exact integer expected step required')
    require(isinstance(value, torch.Tensor) and value.ndim == 0 and value.numel() == 1 and
            value.dtype != torch.bool and not value.is_complex() and bool(torch.isfinite(value)),
            'AdamW step must be a finite real scalar tensor, never int-truncated')
    observed = value.detach().cpu().item()
    require(type(observed) in (int, float) and math.isfinite(observed) and
            observed == expected and math.floor(observed) == observed,
            'AdamW step differs from the exact cumulative stage update count')


def optimizer_snapshot(torch, state, head, expected):
    require(type(state) is dict and set(state) == {'state', 'param_groups'} and
            type(state['param_groups']) is list and len(state['param_groups']) == 1,
            'Original single AdamW group required')
    group = state['param_groups'][0]
    parameters = tuple(head.parameters())
    ids = group['params']
    require(type(ids) is list and len(ids) == len(parameters) and
            all(type(k) is int for k in ids) and len(set(ids)) == len(ids) and
            type(state['state']) is dict and set(state['state']) == set(ids),
            'Every parameter must have its original ordered AdamW state')
    require(group['lr'] == 3e-4 and group['weight_decay'] == 1e-4 and
            tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8 and
            group['amsgrad'] is False and group['maximize'] is False and
            group['capturable'] is False and group['differentiable'] is False and
            group.get('foreach') is None and group.get('fused') is None,
            'Original v1 AdamW numerical options must remain unchanged')
    # PyTorch parameter order is retained, including root memory parameters.
    # Never sort names: MEMORY root keys/values precede child base/query/output.
    for key, parameter in zip(ids, parameters):
        item = state['state'][key]
        require(type(item) is dict and set(item) == {'step', 'exp_avg', 'exp_avg_sq'},
                'Exact non-AMSGrad AdamW state fields required')
        exact_adamw_step(torch, item['step'], expected)
        for name in ('exp_avg', 'exp_avg_sq'):
            value = item[name]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                    value.dtype == parameter.dtype == torch.float32 and
                    tuple(value.shape) == tuple(parameter.shape) and bool(torch.isfinite(value).all()),
                    'Complete finite CPU FP32 moments of exact parameter shape required')
        require(bool((item['exp_avg_sq'] >= 0).all()), 'AdamW second moment must be nonnegative')


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and \
            tuple(left.shape) == tuple(right.shape) and torch.equal(left.cpu(), right.cpu())
    if type(left) is dict:
        return type(right) is dict and set(left) == set(right) and \
            all(tree_equal(torch, value, right[key]) for key, value in left.items())
    if type(left) in (list, tuple):
        return type(right) is type(left) and len(left) == len(right) and \
            all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def parent_resume_gate(spec, closure):
    """Read only after every inherited/parent actual RAW gate and first SHA.

    CPU schema/check names below intentionally match the ordinary stage8 CPU
    draft ABI. ROOT must bind/review its stage16 qualification counterpart.
    No status-only or self-terminal certification is accepted.
    """
    keys = ('parent_N14_GPU_source', 'parent_N14_GPU_spec', 'parent_N14_GPU_report',
            'parent_N14_GPU_admission', 'parent_N14_GPU_RAW', 'parent_N14_CPU_source',
            'parent_N14_CPU_spec', 'parent_N14_CPU_admission', 'parent_N14_CPU_RAW',
            'parent_N14_CPU_qualification', 'parent_N14_CPU_endpoint_numeric_protocol')
    for key in keys:
        closed(closure, spec[key])
    parent = read(closed(closure, spec['parent_N14_GPU_spec']))
    report = read(closed(closure, spec['parent_N14_GPU_report']))
    cpu_spec = read(closed(closure, spec['parent_N14_CPU_spec']))
    qualified = read(closed(closure, spec['parent_N14_CPU_qualification']))
    numeric_protocol = read(closed(closure, spec['parent_N14_CPU_endpoint_numeric_protocol']))
    first, last = spec['first_stage_epoch'], spec['last_stage_epoch']
    parent_end, previous_updates = first - 1, (first - 1) * 1661
    require(parent['schema'] in (FIRST_STAGE_SCHEMA, SCHEMA) and
            (parent['schema'], parent['first_stage_epoch'], parent['last_stage_epoch']) in
                ((FIRST_STAGE_SCHEMA, 1, 8), (SCHEMA, 9, 16)) and
            parent['last_stage_epoch'] == parent_end and parent['fixed_stage_epochs'] == 24 and
            parent['seed'] == spec['seed'] and parent['module_seed'] == spec['module_seed'] and
            parent['order_seed'] == spec['order_seed'] and parent['memory_source'] == spec['memory_source'] and
            parent['carrier_spec'] == spec['carrier_spec'] and parent['entry_source'] == spec['parent_N14_GPU_source'] and
            str(Path(parent['GPU_admission_file']).resolve()) == spec['parent_N14_GPU_admission']['path'] and
            cpu_spec['entry_source'] == spec['parent_N14_CPU_source'] and
            str(Path(cpu_spec['CPU_admission_file']).resolve()) == spec['parent_N14_CPU_admission']['path'],
            'Same N14 prelocked stage parent, seeds, module and E24 provenance required')
    expected = spec['normal_RAW_count']
    require(parent['normal_RAW_count'] == expected - 2 and
            cpu_spec['normal_RAW_count'] == expected - 1 and
            spec['RAW_inputs'][:-2] == parent['RAW_inputs'] and
            cpu_spec['RAW_inputs'] == spec['RAW_inputs'][:-1] and
            spec['RAW_inputs'][-2]['receipt'] == spec['parent_N14_GPU_RAW'] and
            spec['RAW_inputs'][-1]['receipt'] == spec['parent_N14_CPU_RAW'],
            'Normal ancestry must preserve prefix and append actual parent GPU then CPU')
    for ancestor in (parent, cpu_spec):
        require(ancestor['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
                ancestor['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
                ancestor['additional_preserved_independent_CPU_failure'] ==
                    spec['additional_preserved_independent_CPU_failure'] and
                ancestor['total_failed_RAW_count'] == spec['total_failed_RAW_count'] == 9 and
                all(spec['terminal_identity_pins'].get(k) == v
                    for k, v in ancestor['terminal_identity_pins'].items()),
                'Normal and all failed owner/pin prefixes must be immutable')
        require(type(ancestor['sources']) is list and
                all(closure.get(ref['path']) == ref for ref in ancestor['sources']),
                'Actual parent GPU and CPU complete SOURCE closures must be inherited')
    gpu_row, cpu_row = spec['RAW_inputs'][-2:]
    require(report['status'] == STATUS and report['schema'] == parent['schema'] and
            report['source'] == spec['parent_N14_GPU_source'] and
            report['spec'] == spec['parent_N14_GPU_spec'] and
            report['GPU_admission'] == spec['parent_N14_GPU_admission'] and
            report['phase'] == parent['phase'] == gpu_row['phase'] and
            gpu_row['admission'] == spec['parent_N14_GPU_admission'] and
            cpu_row['admission'] == spec['parent_N14_CPU_admission'],
            'Parent report/spec/source/admission/actual RAW linkage differs')
    require(qualified['schema'] == 'N14-stage'+str(parent_end)+'-training-endpoint-CPU-qualification-v1' and
            qualified['status'] == 'ROOT_N14_MEMORY_PAIRED_STAGE'+str(parent_end)+'_ENDPOINT_CPU_QUALIFIED_V1' and
            qualified['source'] == spec['parent_N14_CPU_source'] and
            qualified['spec'] == spec['parent_N14_CPU_spec'] and
            qualified['CPU_admission'] == spec['parent_N14_CPU_admission'] and
            qualified['phase'] == cpu_spec['phase'] == cpu_row['phase'] and
            qualified['current_owned_CPU_terminal_self_certified'] is False,
            'Exact independent CPU qualification source/spec/admission/phase required')
    correspondence = dict(training_GPU_source='parent_N14_GPU_source',
        training_GPU_spec='parent_N14_GPU_spec', training_GPU_report='parent_N14_GPU_report',
        training_GPU_RAW='parent_N14_GPU_RAW', training_GPU_admission='parent_N14_GPU_admission')
    for key, spec_key in correspondence.items():
        require(qualified[key] == cpu_spec[key] == spec[spec_key],
                'CPU qualification must bind this exact actual parent GPU '+key)
    require(qualified['CPU_endpoint_numeric_protocol'] == cpu_spec['CPU_endpoint_numeric_protocol'] ==
                spec['parent_N14_CPU_endpoint_numeric_protocol'] and
            cpu_spec['endpoint_numeric_source'] == spec['parent_N14_CPU_endpoint_numeric_protocol'] and
            type(numeric_protocol) is dict and set(numeric_protocol) ==
                {'atol', 'rtol', 'locked_before_CPU_execution', 'locked_after_GPU_training',
                 'scope', 'bound_completed_GPU_report'} and
            numeric_protocol['bound_completed_GPU_report'] == spec['parent_N14_GPU_report'] and
            numeric_protocol['locked_before_CPU_execution'] is True and
            numeric_protocol['locked_after_GPU_training'] is True and
            type(numeric_protocol['scope']) is str and bool(numeric_protocol['scope']) and
            all(type(numeric_protocol[k]) in (int, float) and math.isfinite(numeric_protocol[k]) and
                numeric_protocol[k] > 0 for k in ('atol', 'rtol')),
            'Separate after-GPU/before-CPU numerical protocol must bind this completed parent report')
    require(qualified['carrier_spec'] == spec['carrier_spec'] and
            qualified['memory_source'] == spec['memory_source'] and
            qualified['protocol'] == spec['protocol'] and
            qualified['original_parent_epochs'] == 24 and
            qualified['new_stage_epochs'] == 8 and
            qualified['parameters'] == dict(BASE_CONTINUATION=40497, MEMORY_ADDON=42698) and
            qualified['fixed_stage_epochs'] == report['fixed_stage_epochs'] == 24 and
            qualified['paired_updates_per_arm'] == previous_updates and
            qualified['common_E24_warm_initialization_exact'] is True and
            qualified['deterministic_zero_addon_initialization_exact'] is True and
            qualified['all_new_state_FP32_finite_shape_exact'] is True and
            qualified['all_AdamW_parameter_steps_exact'+str(previous_updates)] is True and
            qualified['complete_paired'+str(previous_updates)+'_trace_order_identity_online_means_exact'] is True and
            qualified['final_stage'+str(parent_end)+'_full_FIT_mean_BCE_CPU_GPU_compatible'] is True,
            'Independent numerical CPU qualification of the full parent stage required')
    order_proof_key = ('private_eight741_orders_and_final_state_exact' if parent_end == 8 else
                       'private_sixteen741_orders_and_final_state_exact')
    require(qualified[order_proof_key] is True, 'Independent complete private order proof required')
    require(set(report['checkpoints']) == set(qualified['checkpoints']) == set(ARMS) and
            report['checkpoints'] == qualified['checkpoints'],
            'Both resumed checkpoint refs must come from the same independently qualified parent')
    for ref in report['checkpoints'].values():
        closed(closure, ref)
    for ref in (report['start'], report['trace'], qualified['details']):
        closed(closure, ref)
    origin_ref = report.get('stage_origin_START', report['start'])
    origin = read(closed(closure, origin_ref))
    for key in ('source', 'spec', 'GPU_admission', 'parent'):
        closed(closure, origin[key])
    origin_spec = read(closed(closure, origin['spec']))
    require(origin['schema'] == origin_spec['schema'] == FIRST_STAGE_SCHEMA and
            origin['source'] == origin_spec['entry_source'] and
            str(Path(origin_spec['GPU_admission_file']).resolve()) == origin['GPU_admission']['path'] and
            origin['first_stage_epoch'] == 1 and origin['last_stage_epoch'] == 8 and
            origin['stage_epochs'] == 24 and origin['optimizer_reset_both'] is True and
            origin['parent'] == spec['carrier_spec'] and
            origin['initial_state_digests'] == report['initial_state_digests'] and
            origin['behavior'] == report['behavior_preflight'] and
            origin['behavior']['initial_exact_parent_identity'] is True and
            origin['behavior']['initial_permutation_equivariance'] is True and
            origin['behavior']['exterior_identity'] is True,
            'Initial warm/cold construction and preflight proof must remain the original first-leaf evidence')
    history = report['history']
    require(type(history) is list and len(history) == parent_end and
            all(h['stage_epoch'] == i+1 and h['updates_per_arm'] == 1661 and
                h['cumulative_stage_updates'] == (i+1)*1661 for i, h in enumerate(history)),
            'Complete parent epoch history required')
    inherited_traces = report.get('trace_chain', [report['trace']])
    require(type(inherited_traces) is list and len(inherited_traces) == parent_end // 8 and
            inherited_traces[-1] == report['trace'] and
            len({ref['path'] for ref in inherited_traces}) == len(inherited_traces),
            'Immutable complete parent update trace chain required')
    for ref in inherited_traces:
        closed(closure, ref)
    return dict(parent_spec=parent, report=report, CPU_qualification=qualified,
                parent_end=parent_end, previous_updates=previous_updates,
                origin_ref=origin_ref, origin=origin, inherited_traces=inherited_traces)


def gate_inputs(path, spec):
    require(spec['schema'] == SCHEMA and type(spec['first_stage_epoch']) is int and
            type(spec['last_stage_epoch']) is int and
            (spec['first_stage_epoch'], spec['last_stage_epoch']) in ((9, 16), (17, 24)) and
            spec['fixed_stage_epochs'] == 24 and
            spec['optimizer_reset_both'] is False and spec['seed'] == 202610120701 and
            spec['module_seed'] == 202610140801 and spec['order_seed'] == 202610140801 and
            spec['deadline_seconds'] == 1800 and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_limit_MiB'] == 8192 and spec['automatic_retry'] is False,
            'Only prelocked paired stage9--16 or17--24 continuation of stage24 is admitted')
    ninth_failed_raw_only(spec)
    normals, closure = pre_gate(spec, 97)
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    for k in ('entry_source', 'memory_source', 'carrier_spec', 'carrier_evaluation_source',
              'carrier_curve_qualified_summary', 'protocol'):
        closed(closure, spec[k])
    resume = parent_resume_gate(spec, closure)
    carrier = read(bound(spec['carrier_spec']))
    require(carrier['seed'] == spec['seed'] and carrier['head_endpoint_epoch'] == 24 and
            carrier['entry_source'] == spec['carrier_evaluation_source'],
            'Common qualified first701 LOG E24 carrier required')
    curve = read(closed(closure, spec['carrier_curve_qualified_summary']))
    curve_report = read(closed(closure, curve['evaluation_GPU_report']))
    require(curve['status'] == 'QUALIFIED_E24_CURVE_NOT_E48_PRIMARY_CLAIM' and
            curve['single_seed_AP_qualified'] is True and
            curve['evaluation_GPU_actual_returncode'] == curve['evaluation_CPU_actual_returncode'] == 0 and
            curve_report['spec'] == spec['carrier_spec'] and curve_report['seed'] == spec['seed'] and
            curve_report['head_endpoint_epoch'] == 24,
            'Qualified E24 curve must link to this actual common carrier')
    context = dict(carrier)
    for k in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt',
              'additional_failed_RAW_inputs', 'terminal_identity_pins', 'entry_source',
              'phase', 'GPU_admission_file'):
        context[k] = spec[k]
    lineage = load(carrier['endpoint_lineage_source'], '_N14_actual_carrier_lineage')
    gate = lineage.own_gate(path, context, __file__, closure, normals, cpu=False)
    gpu_index = api(lineage, 'owner_ref_join')(
        context, gate['owners'], spec['parent_N14_GPU_RAW'], spec['parent_N14_GPU_admission'],
        spec['parent_N14_GPU_source'], spec['parent_N14_GPU_spec'], '0')
    cpu_index = api(lineage, 'owner_ref_join')(
        context, gate['owners'], spec['parent_N14_CPU_RAW'], spec['parent_N14_CPU_admission'],
        spec['parent_N14_CPU_source'], spec['parent_N14_CPU_spec'], '')
    require(type(gpu_index) is int and type(cpu_index) is int and
            gpu_index == spec['normal_RAW_count'] - 2 and cpu_index == spec['normal_RAW_count'] - 1,
            'Qualified actual parent GPU/CPU owners must join their exact source/spec/admission at the ancestry tail')
    endpoint = lineage.endpoint_gate(context, closure, gate['owners'])
    gate.update(endpoint)
    gate.update(context=context, lineage=lineage, cache_Q=endpoint['cache_q'],
                parent=endpoint['parent'], N14_resume=resume)
    require(len(gate['cache_report']['packet_refs']) == 741 and
            gate['cache_report']['raw_GT'] == 1661,
            'Complete qualified FIT741/1661 input required')
    return gate


def packet(torch, gate, index):
    report = gate['cache_report']
    require(report['packet_refs'][index] == gate['cache_Q']['packet_refs'][index],
            'Qualified immutable packet order required')
    return gate['common'].load_packet(torch, report, index)


def forward(torch, head, gate, value, g, device):
    F, e, L, y, flag = gate['common'].instance_inputs(torch, value, g, 'LOG_PR', device)
    result = gate['common'].region_logits(torch, head, F, e, L, flag)
    require(torch.equal(result['logits'][~flag], L[~flag]), 'LOG exterior changed')
    loss = torch.nn.functional.binary_cross_entropy_with_logits(result['logits'], y)
    require(bool(torch.isfinite(loss)), 'Nonfinite point loss')
    return loss, result, (F, e, L, y, flag)




def measure(torch, gate, heads, started, spec):
    sums = {arm: [] for arm in ARMS}
    for head in heads.values():
        head.eval()
    with torch.no_grad():
        for index in range(741):
            require(time.monotonic()-started < spec['deadline_seconds'], 'Endpoint deadline exceeded')
            value = packet(torch, gate, index)
            for g in range(len(value['query_ids'])):
                for arm in ARMS:
                    loss, _, _ = forward(torch, heads[arm], gate, value, g, 'cuda:0')
                    sums[arm].append(float(loss.detach().cpu()))
            del value
    for head in heads.values():
        head.train()
    require(all(len(v) == 1661 for v in sums.values()), 'Complete endpoint1661 required')
    return {arm: math.fsum(v)/1661 for arm, v in sums.items()}


def run(path):
    started = time.monotonic()
    path = Path(path).resolve(strict=True)
    spec = read(path)
    gate = gate_inputs(path, spec)
    import psutil
    import torch
    import numpy as np
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(),
            'Actual affinity and cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2,
            'Exactly two torch threads required')
    imported = imported_files(psutil, torch, np, gate)
    factory = load(gate['context']['point_heads_source'], '_N14_original_PR_factory')
    memory_module = load(spec['memory_source'], '_N14_memory_module')
    outside_CPU = torch.random.get_rng_state().clone()
    base = factory.make_point_head(torch, spec['seed'])
    memory_base = factory.make_point_head(torch, spec['seed'])
    memory = memory_module.make_context_memory_head(torch, memory_base,
        seed=spec['module_seed'], tokens=8, dim=8)
    heads = dict(BASE_CONTINUATION=base, MEMORY_ADDON=memory)
    require(torch.equal(outside_CPU, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'Private module construction changed outside RNG/CUDA')
    require(sum(p.numel() for p in base.parameters()) == 40497 and
            sum(p.numel() for p in memory.parameters()) == 42698,
            'Prelocked40497/42698 parameter counts required')
    for head in heads.values():
        for p in head.parameters():
            p.requires_grad_(True)
    resume = gate['N14_resume']
    parent_report = resume['report']
    first, last = spec['first_stage_epoch'], spec['last_stage_epoch']
    previous_updates, final_updates = resume['previous_updates'], last * 1661
    parent_checkpoints = {}
    initial = copy.deepcopy(parent_report['initial_state_digests'])
    behavior = copy.deepcopy(parent_report['behavior_preflight'])
    for arm in ARMS:
        cp = torch.load(str(closed(gate['closure'], parent_report['checkpoints'][arm])), map_location='cpu')
        require(type(cp) is dict and set(cp) == {'meta', 'state_dict', 'optimizer_state_dict', 'private_order_state'},
                'Exact complete parent state/optimizer/private-order checkpoint required')
        meta = cp['meta']
        require(meta['schema'] in ('N14-context-memory-stage-checkpoint-v1',
                                   'N14-context-memory-stage-checkpoint-v2') and
                meta['arm'] == arm and meta['source'] == spec['parent_N14_GPU_source'] and
                meta['spec'] == spec['parent_N14_GPU_spec'] and
                meta['GPU_admission'] == spec['parent_N14_GPU_admission'] and
                meta['phase'] == resume['parent_spec']['phase'] and
                meta['parent_spec'] == spec['carrier_spec'] and meta['parent_head_epochs'] == 24 and
                meta['common_parent_arm'] == 'LOG_PR' and meta['stage_epochs_completed'] == first - 1 and
                meta['stage_updates'] == previous_updates and meta['fixed_stage_epochs'] == 24 and
                meta['optimizer_reset_at_stage_start'] is True and
                meta['memory_source'] == spec['memory_source'] and
                meta['head_factory_source'] == gate['context']['point_heads_source'] and
                meta['parameters'] == sum(p.numel() for p in heads[arm].parameters()) and
                meta['initial_state_digests'] == initial[arm] and meta['history'] == parent_report['history'] and
                meta['gradient_activity'] == parent_report['gradient_activity'][arm] and
                meta['cache_CPU_qualification'] == gate['context']['cache_CPU_qualification'] and
                all(meta[k] is True for k in ('no_DEV', 'no_TEST', 'no_AP')) and
                all(meta[k] is False for k in ('native_weights_updated', 'natural_query_scores_changed', 'support_changed')),
                'Actual resumed checkpoint metadata must match qualified parent, original protocol and complete history')
        if meta['schema'] == 'N14-context-memory-stage-checkpoint-v2':
            require(meta['trace_chain'] == resume['inherited_traces'] and
                    meta['stage_origin_START'] == resume['origin_ref'] and
                    meta['original_behavior_preflight'] == behavior and
                    meta['original_common_LOG_parent_proofs'] == resume['origin']['common_LOG_parent_proofs'] and
                    meta['original_preflight_inherited'] is True and meta['optimizer_reset_this_leaf'] is False,
                    'Continuation parent must preserve identical report/checkpoint origin and cumulative trace evidence')
        template = heads[arm].state_dict()
        require(type(cp['state_dict']) is dict and set(cp['state_dict']) == set(template),
                'Exact original model parameter names required')
        for name, value in cp['state_dict'].items():
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                    value.dtype == torch.float32 and tuple(value.shape) == tuple(template[name].shape) and
                    bool(torch.isfinite(value).all()), 'Finite exact CPU FP32 resumed head state required')
        heads[arm].load_state_dict(cp['state_dict'], strict=True)
        optimizer_snapshot(torch, cp['optimizer_state_dict'], heads[arm], previous_updates)
        names = {n for n, _ in heads[arm].named_parameters()}
        require(set(meta['gradient_activity']) == names and
                all(type(row) is dict and set(row) == {'present', 'nonzero'} and
                    type(row['present']) is type(row['nonzero']) is int and
                    row['present'] == previous_updates and 0 <= row['nonzero'] <= row['present']
                    for row in meta['gradient_activity'].values()),
                'Complete cumulative parent gradient activity required')
        parent_checkpoints[arm] = cp
    parent_orders = parent_checkpoints[ARMS[0]]['meta']['epoch_orders']
    require(parent_orders == parent_checkpoints[ARMS[1]]['meta']['epoch_orders'] and
            (parent_report.get('epoch_orders', parent_orders) == parent_orders) and
            type(parent_orders) is list and len(parent_orders) == first - 1 and
            torch.equal(parent_checkpoints[ARMS[0]]['private_order_state'],
                        parent_checkpoints[ARMS[1]]['private_order_state']),
            'Both qualified arms must share all previous orders and final private order state')
    proof_rng = torch.Generator(device='cpu')
    proof_rng.manual_seed(spec['order_seed'])
    for epoch, row in enumerate(parent_orders, 1):
        expected_order = torch.randperm(741, generator=proof_rng).tolist()
        expected_sha = hashlib.sha256(json.dumps(expected_order, separators=(',', ':')).encode('ascii')).hexdigest()
        require(row == dict(stage_epoch=epoch, indices=expected_order, sha256=expected_sha) and
                parent_report['history'][epoch-1]['paired_image_order_sha256'] == expected_sha,
                'Complete original private order prefix/hash proof differs')
    saved_order_state = parent_checkpoints[ARMS[0]]['private_order_state']
    require(isinstance(saved_order_state, torch.Tensor) and saved_order_state.device.type == 'cpu' and
            saved_order_state.dtype == torch.uint8 and saved_order_state.ndim == 1 and
            torch.equal(proof_rng.get_state(), saved_order_state), 'Exact cumulative private RNG state required')
    order_rng = torch.Generator(device='cpu')
    order_rng.set_state(saved_order_state.clone())
    history = copy.deepcopy(parent_report['history'])
    orders = copy.deepcopy(parent_orders)
    activity = copy.deepcopy(parent_report['gradient_activity'])
    resume_state_digests = {arm: {n: digest(v) for n, v in head.state_dict().items()}
                            for arm, head in heads.items()}
    require(torch.equal(outside_CPU, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'CPU resume/proof construction changed outside RNG or cold CUDA')
    output = Path(spec['output_directory'])
    require(not output.exists(), 'Never overwrite old science output')
    output.mkdir()
    write_new(output/'START.json', dict(schema=SCHEMA, phase=spec['phase'],
        source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'],
        parent=spec['carrier_spec'], common_LOG_parent_proofs=resume['origin']['common_LOG_parent_proofs'],
        initial_state_digests=initial, behavior=behavior, stage_epochs=24,
        stage_origin_START=resume['origin_ref'], original_preflight_inherited=True,
        resume_state_digests=resume_state_digests, parent_N14_checkpoint_refs=parent_report['checkpoints'],
        parent_N14_GPU_spec=spec['parent_N14_GPU_spec'], parent_N14_GPU_report=spec['parent_N14_GPU_report'],
        parent_N14_GPU_RAW=spec['parent_N14_GPU_RAW'],
        parent_N14_CPU_qualification=spec['parent_N14_CPU_qualification'], parent_N14_CPU_RAW=spec['parent_N14_CPU_RAW'],
        first_stage_epoch=first, last_stage_epoch=last, optimizer_reset_both=False,
        restored_cumulative_stage_updates=previous_updates, leaf_updates_per_arm=8*1661,
        original_parent_epochs=24, continuous_parent_plus_stage=False,
        native_weights_updated=False, no_DEV=True, no_TEST=True, no_AP=True))
    del memory_base, base, memory, cp, meta, template, proof_rng, saved_order_state
    torch.cuda.set_device(0)
    require(torch.cuda.device_count() == 1, 'Exactly CUDA0 required')
    torch.cuda.set_per_process_memory_fraction(4096*1024**2/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    outside_GPU = torch.cuda.get_rng_state(0).clone()
    opts = {}
    for arm in ARMS:
        heads[arm].to('cuda:0').train()
        opts[arm] = torch.optim.AdamW(heads[arm].parameters(), lr=3e-4, weight_decay=1e-4)
        opts[arm].load_state_dict(parent_checkpoints[arm]['optimizer_state_dict'])
        require(len(opts[arm].state) == len(tuple(heads[arm].parameters())),
                'Restored AdamW must have all trained parameter states')
        for parameter in heads[arm].parameters():
            item = opts[arm].state[parameter]
            exact_adamw_step(torch, item['step'], previous_updates)
            require(item['exp_avg'].device == parameter.device == item['exp_avg_sq'].device and
                    item['exp_avg'].dtype == item['exp_avg_sq'].dtype == torch.float32 and
                    bool(torch.isfinite(item['exp_avg']).all()) and bool(torch.isfinite(item['exp_avg_sq']).all()) and
                    bool((item['exp_avg_sq'] >= 0).all()), 'Exact finite CUDA optimizer moments required')
        require(tree_equal(torch, cpu_tree(torch, opts[arm].state_dict()),
                           parent_checkpoints[arm]['optimizer_state_dict']),
                'Optimizer restore must preserve every saved value and group option')
    del parent_checkpoints, item, parameter, parent_orders
    with (output/'UPDATE_TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as trace:
        for epoch in range(first, last+1):
            order = torch.randperm(741, generator=order_rng).tolist()
            order_sha = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode('ascii')).hexdigest()
            orders.append(dict(stage_epoch=epoch, indices=order, sha256=order_sha))
            count, losses = 0, {arm: [] for arm in ARMS}
            for index in order:
                require(time.monotonic()-started < spec['deadline_seconds'], 'Training leaf deadline exceeded')
                value = packet(torch, gate, index)
                for g in range(len(value['query_ids'])):
                    row = dict(stage_epoch=epoch, image_index=index, GT_index=g,
                        annotation_id=value['identity']['ann_ids'][g], query_id=int(value['query_ids'][g]),
                        stage_update=(epoch-1)*1661+count+1, BCE={})
                    for arm in ARMS:
                        head, opt = heads[arm], opts[arm]
                        opt.zero_grad(set_to_none=True)
                        loss, result, operands = forward(torch, head, gate, value, g, 'cuda:0')
                        require(loss.requires_grad, 'Loss must connect to both learned branches')
                        loss.backward()
                        for name, p in head.named_parameters():
                            require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Missing or nonfinite gradients')
                            activity[arm][name]['present'] += 1
                            activity[arm][name]['nonzero'] += int(bool((p.grad != 0).any()))
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'Nonfinite model state')
                        observed = float(loss.detach().cpu())
                        row['BCE'][arm] = observed
                        losses[arm].append(observed)
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    count += 1
                del value, loss, result, operands
            require(count == 1661 and all(len(x) == 1661 for x in losses.values()),
                    'Both arms must complete1661 updates per epoch')
            endpoint = measure(torch, gate, heads, started, spec)
            h = dict(stage_epoch=epoch, paired_image_order_sha256=order_sha, updates_per_arm=1661,
                cumulative_stage_updates=epoch*1661,
                online_BCE={arm: math.fsum(v)/1661 for arm, v in losses.items()},
                full_FIT_BCE=endpoint)
            history.append(h)
            trace.flush()
            print(json.dumps(h), flush=True)
    require(len(history) == len(orders) == last and
            all(row['present'] == final_updates and type(row['nonzero']) is int and
                0 <= row['nonzero'] <= final_updates for arm in ARMS for row in activity[arm].values()),
            'Complete cumulative stage history/orders/activity required')
    trace_ref = reference(output/'UPDATE_TRACE.jsonl')
    trace_chain = copy.deepcopy(resume['inherited_traces']) + [trace_ref]
    checkpoints = {}
    for arm in ARMS:
        head, opt = heads[arm], opts[arm]
        require(len(opt.state) == len(tuple(head.parameters())), 'Complete final AdamW state required')
        for parameter in head.parameters():
            exact_adamw_step(torch, opt.state[parameter]['step'], final_updates)
        saved_optimizer = cpu_tree(torch, opt.state_dict())
        optimizer_snapshot(torch, saved_optimizer, head, final_updates)
        meta = dict(schema='N14-context-memory-stage-checkpoint-v2', arm=arm,
            source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'],
            phase=spec['phase'], parent_spec=spec['carrier_spec'], parent_head_epochs=24,
            common_parent_arm='LOG_PR', stage_epochs_completed=last, stage_updates=final_updates,
            fixed_stage_epochs=24, optimizer_reset_at_stage_start=True,
            optimizer_reset_this_leaf=False, first_stage_epoch=first, last_stage_epoch=last,
            leaf_new_stage_epochs=8, leaf_updates=8*1661, cumulative_stage_updates=final_updates,
            actual_state_parent_N14_checkpoint=parent_report['checkpoints'][arm],
            actual_state_parent_N14_GPU_source=spec['parent_N14_GPU_source'],
            actual_state_parent_N14_GPU_spec=spec['parent_N14_GPU_spec'],
            actual_state_parent_N14_GPU_report=spec['parent_N14_GPU_report'],
            actual_state_parent_N14_GPU_RAW=spec['parent_N14_GPU_RAW'],
            actual_state_parent_N14_CPU_qualification=spec['parent_N14_CPU_qualification'],
            actual_state_parent_N14_CPU_RAW=spec['parent_N14_CPU_RAW'],
            stage_origin_START=resume['origin_ref'], original_behavior_preflight=behavior,
            original_common_LOG_parent_proofs=resume['origin']['common_LOG_parent_proofs'],
            original_preflight_inherited=True, resume_state_digests=resume_state_digests[arm],
            trace_chain=trace_chain, continuous_parent_plus_stage=False,
            memory_source=spec['memory_source'], head_factory_source=gate['context']['point_heads_source'],
            parameters=sum(p.numel() for p in head.parameters()), initial_state_digests=initial[arm],
            history=history, epoch_orders=orders, gradient_activity=activity[arm],
            cache_CPU_qualification=gate['context']['cache_CPU_qualification'],
            natural_query_scores_changed=False, support_changed=False, no_DEV=True, no_TEST=True,
            no_AP=True, native_weights_updated=False, independent_qualified=False,
            checkpoint_resume_eligible=True,
            resume_eligibility_scope='Complete numerical state only; independent CPU qualification and actual RAW/SOURCE closure still mandatory')
        cp = dict(meta=meta, state_dict=cpu_tree(torch, head.state_dict()),
            optimizer_state_dict=saved_optimizer,
            private_order_state=order_rng.get_state().clone())
        target = output/('HEAD_'+arm+'_STAGE'+str(last)+'.pth')
        with target.open('xb') as stream:
            torch.save(cp, stream)
        replay = torch.load(str(target), map_location='cpu')
        require(tree_equal(torch, cp, replay),
                'Exact own model/optimizer/private-order/history checkpoint readback required')
        checkpoints[arm] = reference(target)
        del cp, replay, opt, head, saved_optimizer, parameter
    require(torch.equal(outside_CPU, torch.random.get_rng_state()) and
            torch.equal(outside_GPU, torch.cuda.get_rng_state(0)), 'Outside RNG changed')
    peak = torch.cuda.max_memory_allocated(0)
    for head in heads.values():
        head.cpu()
    del head
    opts.clear()
    heads.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    require(torch.cuda.memory_allocated(0) == 0, 'Live GPU aliases remain')
    require(psutil.Process().cpu_affinity() == [0, 1] and
            torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
            'mmdet' not in sys.modules, 'Final thread/affinity/no native model contract differs')
    require(imported_files(psutil, torch, np, gate) == imported, 'Imported library closure changed')
    for ref in spec['sources']:
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
            reference(spec['GPU_admission_file']) == gate['admission_ref'], 'Own execution binding changed')
    report = dict(status=STATUS, schema=SCHEMA, source=gate['source'], spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'], phase=spec['phase'], checkpoints=checkpoints,
        original_parent_epochs=24, new_stage_epochs=8, fixed_stage_epochs=24,
        first_stage_epoch=first, last_stage_epoch=last, stage_epochs_completed=last,
        leaf_updates_per_arm=8*1661, cumulative_stage_updates_per_arm=final_updates,
        updates_per_arm=8*1661, optimizer_reset_both=False, optimizer_state_restored=True,
        private_order_state_restored=True, original_preflight_inherited=True,
        parent_N14_GPU_source=spec['parent_N14_GPU_source'], parent_N14_GPU_spec=spec['parent_N14_GPU_spec'],
        parent_N14_GPU_report=spec['parent_N14_GPU_report'], parent_N14_GPU_RAW=spec['parent_N14_GPU_RAW'],
        parent_N14_GPU_admission=spec['parent_N14_GPU_admission'],
        parent_N14_CPU_qualification=spec['parent_N14_CPU_qualification'], parent_N14_CPU_RAW=spec['parent_N14_CPU_RAW'],
        parent_N14_checkpoint_refs=parent_report['checkpoints'],
        stage_origin_START=resume['origin_ref'], trace_chain=trace_chain,
        common_LOG_parent_proofs=resume['origin']['common_LOG_parent_proofs'],
        history=history, epoch_orders=orders, behavior_preflight=behavior,
        initial_state_digests=initial, gradient_activity=activity, imported_library_refs=imported,
        resume_state_digests=resume_state_digests, continuous_parent_plus_stage=False,
        checkpoint_resume_eligible=True,
        resume_eligibility_scope='Complete numerical state only; independent CPU qualification and actual RAW/SOURCE closure still mandatory',
        trace=trace_ref, start=reference(output/'START.json'),
        elapsed_seconds=time.monotonic()-started, peak_torch_allocated_bytes=peak,
        own_terminal_self_certified=False, independent_CPU_qualified=False,
        no_DEV=True, no_TEST=True, AP_evaluated=False, capacity_confound_not_yet_controlled=True,
        scope='Exploratory single-module training only; FIT loss cannot establish mask AP gain or innovation')
    require(report['elapsed_seconds'] < 1800, 'Whole leaf deadline exceeded')
    write_new(output/'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, report=reference(output/'REPORT.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'One explicit ROOT-owned attempt required')
    run(args.spec)


