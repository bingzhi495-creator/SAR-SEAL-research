"""UNBOUND single-leaf CPU FIT741/1661 reader and common-init qualification.

Source preparation only. Root must later bind/admit the complete original
inputs; this entry never creates a binder, retries, selects a shard or runs GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

MODULE_ROLES = (
    'shared_initialization', 'rgb_matched_reader',
    'full_legacy', 'full_dsconv', 'full_head', 'full_feature',
    'plain_legacy', 'plain_feature', 'instance_adapter',
    'matched_reader', 'image_geometry',
)
INFRA_ROLES = ('entry_source', 'body_source', 'runtime_source', 'qualifier_source')
SPEC_SCHEMA = 'ROOT-U38-full-method-CPU-reader-init-spec-v1'
OPERATION = 'CPU_U38_FULL_METHOD_READER_AND_COMMON_INITIALIZATION'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def no_torch():
    need(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
         'Complete original admission and source closure must precede Torch import')


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Exact closed reference')


def path_key(value):
    return value.replace('\\', '/').casefold()


def ref(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path, name):
    path = Path(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_scope(spec):
    """Pure metadata guard, reused by the actual entry before scientific imports."""
    counts = ('owner_index', 'normal_RAW_count', 'total_failed_RAW_count',
              'total_terminal_identity_pin_count', 'legacy_pin_count')
    need(spec['schema'] == SPEC_SCHEMA and spec['operation'] == OPERATION
         and all(type(spec[key]) is int and spec[key] >= 0 for key in counts)
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] ==
             spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and type(spec['deadline_seconds']) is int and spec['deadline_seconds'] == 1800
         and spec['automatic_retry'] is False and spec['cuda_visible_devices'] == ''
         and spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16*1024**3
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['images'] == 741 and spec['GT'] == 1661
         and spec['dataset_indices'] == list(range(741))
         and all(spec[key] is True for key in ('no_new_training', 'no_AP', 'no_DEV', 'no_TEST'))
         and spec['full_method_training_eligible'] is False,
         'Single complete CPU741/1661 leaf, no automatic sharding/retry/GPU/training')
    need(type(spec['module_sources']) is dict and set(spec['module_sources']) == set(MODULE_ROLES),
         'Exactly eleven qualifier component roles; no guessed dependency aliases')
    need(type(spec['image_references']) is list and len(spec['image_references']) == 741,
         'All original JPEG references must be supplied before admission')


def owned(c, value, output, name):
    strict_ref(value)
    path = Path(value['path'])
    need(path.resolve() == (output/name).resolve() and not path.is_symlink()
         and c.ref(path) == value, 'Exact owned report/log artifact')


def run(path):
    no_torch()
    path = Path(path)
    need(path.is_absolute(), 'Absolute original Root spec')
    spec = json.loads(path.read_bytes())
    validate_scope(spec)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Original CPU-only environment')
    need(type(spec['sources']) is list and bool(spec['sources']), 'Complete closed input list')
    source_map = {}
    for value in spec['sources']:
        strict_ref(value)
        key = path_key(value['path'])
        need(key not in source_map, 'No duplicate/conflicting source paths')
        source_map[key] = value
    required = [spec[role] for role in INFRA_ROLES]+list(spec['module_sources'].values())
    required += [spec[key] for key in ('context', 'prelock', 'plan', 'parent_ROOT_qualified')]
    required += spec['image_references']
    image_keys = []
    for value in required:
        strict_ref(value)
        need(source_map.get(path_key(value['path'])) == value, 'Every direct source/input is admitted')
    for value in spec['image_references']:
        image_keys.append(path_key(value['path']))
    need(len(set(image_keys)) == 741, '741 distinct original JPEG paths')
    need(ref(__file__) == spec['entry_source'], 'Actual source-closed entry')
    spec_ref = ref(path)
    admission_path = Path(spec['admission_file'])
    need(admission_path.is_absolute(), 'Absolute original admission path')
    admission = json.loads(admission_path.read_bytes())
    keys = {'status', 'phase', 'driver_sha256', 'entry', 'entry_args', 'allowed_cpus', 'thread_limit',
            'cuda_visible_devices', 'whole_GPU_used_limit_MiB', 'qualified_leaf_deadline_seconds',
            'automatic_retry', 'scientific_spec', 'bound_source_sha256'}
    need(set(admission) == keys and len(admission) == 13
         and admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE'
         and admission['scientific_spec'] == spec_ref and admission['phase'] == spec['phase']
         and admission['entry'] == spec['entry_source']['path']
         and admission['entry_args'] == ['--spec', str(path), '--owner-execute']
         and admission['allowed_cpus'] == [0, 1] and admission['thread_limit'] == 2
         and admission['cuda_visible_devices'] == '' and admission['automatic_retry'] is False
         and admission['whole_GPU_used_limit_MiB'] == 8192
         and admission['qualified_leaf_deadline_seconds'] == 1800
         and admission['bound_source_sha256'] == {
             value['path']: value['sha256'] for value in spec['sources']+[spec_ref]},
         'Original thirteen-field admission and complete source map')
    for value in spec['sources']:
        need(ref(value['path']) == value, 'Pre-import closed source/input hash')
    no_torch()
    runtime = load(spec['runtime_source']['path'], 'U38_CPU_RESOURCE_CONTROL')
    no_torch()
    c = runtime.Control(spec)
    c.execution_spec_ref, c.execution_admission_ref = spec_ref, ref(admission_path)
    c.check()
    need(len(c.allowed) == len(spec['sources']), 'No conflicting Control direct inputs')
    for value in spec['sources']:
        c.closed(value)
    need(c.psutil.Process().cpu_affinity() == [0, 1]
         and all(os.environ.get(key) == '2' for key in (
             'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
         'Actual CPU01 and two-thread environment')
    parent = json.loads(Path(c.closed(spec['parent_ROOT_qualified'])).read_bytes())
    need(parent['qualified'] is True and all(parent[key] == spec[key] for key in (
        'normal_RAW_count', 'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'legacy_pin_count')),
        'Current actual Root-qualified terminal predecessor')
    context = json.loads(Path(c.closed(spec['context'])).read_bytes())
    need(type(context) is dict and context['schema'] == 'ROOT-U38-full-method-CPU-context-v1'
         and all(key in context for key in ('FIT_context', 'N13_qualified', 'N13_report')),
         'Original matched-reader context; no invented bridge')
    for key in ('FIT_context', 'N13_qualified', 'N13_report'):
        strict_ref(context[key])
        c.closed(context[key])
    pre = json.loads(Path(c.closed(spec['prelock'])).read_bytes())
    need(pre['schema'] == 'U38-full-method-CPU-reader-init-prelock-v1'
         and pre['images'] == 741 and pre['GT'] == 1661 and pre['dataset_indices'] == list(range(741))
         and pre['branch_size'] == 400 and pre['canvas_size'] == 800
         and pre['common_state_count'] == 104 and pre['common_parameter_tensors'] == 68
         and pre['common_BN_buffers'] == 36 and pre['initialization_seed'] == 202610173101
         and pre['native_L_baseline'] is True and pre['require_fullFIT_native100_M_or_S0'] is False
         and pre['model_forward'] is False and pre['optimizer_created'] is False
         and pre['automatic_retry'] is False and pre['automatic_sharding'] is False
         and all(pre[key] is True for key in ('no_new_training', 'no_AP', 'no_DEV', 'no_TEST'))
         and pre['plan'] == spec['plan'], 'Fixed complete reader/common-init contract')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free >= 1024**3,
         'Fresh owned report directory')
    no_torch()
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned runtime and cold CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    need(torch.get_default_dtype() == torch.float32, 'FP32 default is unchanged')
    body = load(c.closed(spec['body_source']), 'U38_CPU_READER_INIT_BODY')
    result = body.run(torch, np, spec, c, pre)
    c.check()
    need(not torch.cuda.is_initialized() and result['images'] == 741 and result['GT'] == 1661
         and result['CPU_reader_verified'] is True and result['common_initialization_verified'] is True
         and all(result[key] is False for key in ('CUDA_initialized', 'qualified', 'new_training', 'AP', 'DEV',
             'TEST', 'GPU_resource_gate_passed', 'full_method_training_eligible')),
         'Actual completed CPU scope only, never Root self-qualification')
    owned(c, result['report_ref'], output, 'REPORT.json')
    owned(c, result['log_ref'], output, 'probe.log')
    report = json.loads(Path(result['report_ref']['path']).read_bytes())
    need(report['schema'] == 'U38-full-method-CPU-reader-init-report-pending-v1'
         and report['status'] == 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
         and report['qualification']['images_completed'] == 741
         and report['qualification']['GT_completed'] == 1661
         and report['source_aliases_restored_exact'] is True and report['CPU_RNG_restored_exact'] is True
         and report['CUDA_initialization_attempts'] == [] and report['CUDA_initialized'] is False,
         'Exact worker completion report, no independent numeric replay')
    for value in required[:len(INFRA_ROLES)+len(MODULE_ROLES)]:
        need(c.ref(value['path']) == value, 'Executed source unchanged')
    c.check()
    q = dict(schema='U38-owned-pending-full-method-CPU-reader-init-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE', operation=OPERATION,
        phase=spec['phase'], source=spec['entry_source'], body_source=spec['body_source'],
        runtime_source=spec['runtime_source'], qualifier_source=spec['qualifier_source'],
        module_sources=spec['module_sources'], spec=spec_ref, admission=c.execution_admission_ref,
        context=spec['context'], prelock=spec['prelock'], plan=spec['plan'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'], result=result,
        owner_index=spec['owner_index'], normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        legacy_pin_count=spec['legacy_pin_count'], FIT=True, DEV=False, TEST=False, AP=False,
        new_training=False, model_forward_executed=False, backward_executed=False,
        optimizer_created=False, GPU_resource_gate_passed=False, full_method_training_eligible=False,
        elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available, cleanup=dict(CUDA_initialized=False),
        qualified=False, original_actual0_self_certified=False, innovation_established=False, goal_complete=False)
    qref = c.write_json(output/'Q.json', q)
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False, CPU_reader_init_only=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Only Root owns the original single CPU leaf')
    run(args.spec)
