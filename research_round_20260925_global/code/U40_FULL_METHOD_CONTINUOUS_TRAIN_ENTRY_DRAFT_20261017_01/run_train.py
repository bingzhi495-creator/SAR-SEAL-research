"""UNBOUND formal continuous-training segment entry.

Source preparation only. Root must later bind/admit the complete original
inputs; this entry never creates a binder, retries or starts formal training.
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
    'matched_reader', 'image_geometry', 'u36_helpers', 'u39_training_ops',
)
INFRA_ROLES = ('entry_source', 'body_source', 'runtime_source', 'core_source')
SPEC_SCHEMA = 'ROOT-U40-continuous-training-segment-spec-v1'
OPERATION = 'GPU_U40_FULL_METHOD_CONTINUOUS_TRAIN_SEGMENT'
ORDER_SHA256 = '2afadef62eac9f797c125fa105a3a28f94cdbdd9db4b5b5a2b018d4f0d5f194e'


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
    """Pure metadata guard; Root locks the span before original execution."""
    keys = ('owner_index', 'normal_RAW_count', 'total_failed_RAW_count',
            'total_terminal_identity_pin_count', 'legacy_pin_count')
    need(spec['schema'] == SPEC_SCHEMA and spec['operation'] == OPERATION
         and all(type(spec[k]) is int and spec[k] >= 0 for k in keys)
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and spec['deadline_seconds'] == 1800 and type(spec['deadline_seconds']) is int
         and spec['automatic_retry'] is False and spec['cuda_visible_devices'] == '0'
         and spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16*1024**3
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['full_metadata_population_images'] == 741 and spec['full_metadata_population_GT'] == 1661
         and spec['no_new_training'] is False and spec['new_training'] is True and spec['pilot_only'] is False
         and spec['formal_main_training_authorized'] is True
         and all(spec[k] is True for k in ('no_AP', 'no_DEV', 'no_TEST')),
         'Root-owned formal segment, fixed runtime/limits, no retry or AP selection')
    need(type(spec['module_sources']) is dict and set(spec['module_sources']) == set(MODULE_ROLES),
         'Exactly thirteen fixed modules including original U39 numeric operations')
    need(type(spec['image_references']) is list and len(spec['image_references']) == 741,
         'Full original JPEG refs remain pre-admitted')
    segment = spec['segment']
    need(type(segment) is dict and type(segment['seed']) is int and segment['seed'] == 202610173101
         and type(segment['start']) is int and type(segment['stop']) is int
         and 0 <= segment['start'] < segment['stop'] <= 37050
         and segment['core_source'] == spec['core_source'] and segment['order_sha256'] == ORDER_SHA256,
         'Exact common global [start,stop) span and prelocked 50-epoch order')
    need(type(spec['segment_length']) is int and spec['segment_length'] == 256
         and segment['start'] % 256 == 0 and segment['stop'] == min(segment['start']+256, 37050),
         'Root-locked 256-update partitions, with final 186-update segment')
    indices = spec['admitted_tensor_indices']
    need(type(indices) is list and indices and indices == sorted(set(indices))
         and all(type(i) is int and 0 <= i < 741 for i in indices),
         'Explicit unique tensor membership, validated against locked order inside body')
    need(type(spec['parent_checkpoint_refs']) is dict and set(spec['parent_checkpoint_refs']) == {'PLAIN', 'FULL'},
         'Explicit two-arm checkpoint references')
    if segment['start'] == 0:
        need(spec['parent_checkpoint_refs'] == {'PLAIN': None, 'FULL': None}
             and segment['order_ref'] is None and segment['parent_pair_ref'] is None
             and spec['resume_qualified'] is None, 'No hidden parent or order fallback in initial segment')
    else:
        need(all(spec['parent_checkpoint_refs'][a] is not None for a in ('PLAIN', 'FULL'))
             and segment['order_ref'] is not None and segment['parent_pair_ref'] is not None
             and spec['resume_qualified'] is not None, 'Resume requires a Root-accepted paired endpoint')
    need(type(spec['minimum_output_free_bytes']) is int and spec['minimum_output_free_bytes'] > 0,
         'Root must prelock checkpoint/storage margin without inventing a measured file size')
    need(type(spec['disk_budget']) is dict
         and spec['disk_budget']['required_free_bytes'] == spec['minimum_output_free_bytes'],
         'Explicit Root disk calculation and required free space agree')


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
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Original GPU0 environment')
    need(type(spec['sources']) is list and bool(spec['sources']), 'Complete closed input list')
    source_map = {}
    for value in spec['sources']:
        strict_ref(value)
        key = path_key(value['path'])
        need(key not in source_map, 'No duplicate/conflicting source paths')
        source_map[key] = value
    required = [spec[role] for role in INFRA_ROLES]+list(spec['module_sources'].values())
    required += [spec[key] for key in ('context', 'prelock', 'plan', 'parent_ROOT_qualified', 'U39_qualified')]
    required += spec['image_references']
    required += [r for r in [spec['resume_qualified'], spec['segment']['order_ref'], spec['segment']['parent_pair_ref']]+list(spec['parent_checkpoint_refs'].values()) if r is not None]
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
         and admission['cuda_visible_devices'] == '0' and admission['automatic_retry'] is False
         and admission['whole_GPU_used_limit_MiB'] == 8192
         and admission['qualified_leaf_deadline_seconds'] == 1800
         and admission['bound_source_sha256'] == {
             value['path']: value['sha256'] for value in spec['sources']+[spec_ref]},
         'Original thirteen-field admission and complete source map')
    for value in spec['sources']:
        need(ref(value['path']) == value, 'Pre-import closed source/input hash')
    no_torch()
    runtime = load(spec['runtime_source']['path'], 'U40_TRAIN_RESOURCE_CONTROL')
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
    u39 = json.loads(Path(c.closed(spec['U39_qualified'])).read_bytes())
    need(u39['schema'] == 'ROOT-U39-full-method-cross-image-resource-qualified-v1'
         and u39['qualified'] is True and u39['worker_six_disposable_updates_reported_complete'] is True
         and u39['successful_optimizer_updates'] == 6,
         'Actual accepted short U39 resource evidence, not automatic main-training eligibility')
    pre = json.loads(Path(c.closed(spec['prelock'])).read_bytes())
    need(pre['schema'] == 'U40-continuous-training-segment-prelock-v1'
         and pre['segment'] == spec['segment'] and pre['parent_checkpoint_refs'] == spec['parent_checkpoint_refs']
         and pre['resume_qualified'] == spec['resume_qualified'] and pre['plan'] == spec['plan']
         and pre['admitted_tensor_indices'] == spec['admitted_tensor_indices']
         and pre['formal_training_authorized'] is True and pre['pilot_only'] is False
         and pre['epochs'] == 50 and pre['images'] == 741 and pre['GT'] == 1661
         and pre['total_updates_per_arm'] == 37050 and pre['automatic_retry'] is False
         and pre['minimum_output_free_bytes'] == spec['minimum_output_free_bytes']
         and pre['segment_length'] == spec['segment_length'] and pre['disk_budget'] == spec['disk_budget']
         and all(pre[k] is True for k in ('no_AP', 'no_DEV', 'no_TEST')),
         'Root prelocks formal budget/span/order/parents/storage, no epoch/AP adaptive choice')
    if spec['segment']['start']:
        resume = json.loads(Path(c.closed(spec['resume_qualified'])).read_bytes())
        start = spec['segment']['start']
        expected_cursor = dict(global_updates=start, completed_epochs=start//741, within_epoch=start % 741)
        need(resume['schema'] == 'ROOT-U40-continuous-training-segment-qualified-v1'
             and resume['qualified'] is True and resume['paired_prefix_advanced'] is True
             and resume['result']['pair_commit_ref'] == spec['segment']['parent_pair_ref']
             and resume['result']['checkpoint_refs'] == spec['parent_checkpoint_refs']
             and resume['result']['order_ref'] == spec['segment']['order_ref']
             and resume['result']['paired_checkpoint_prefix'] == expected_cursor,
             'Only actual Root-accepted paired prefix may become a resume source')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free >= spec['minimum_output_free_bytes'],
         'Fresh owned report directory')
    need(Path(spec['segment_output_directory']).resolve() == (output/'SEGMENT').resolve()
         and not Path(spec['segment_output_directory']).exists(), 'Core owns fresh SEGMENT subdirectory')
    no_torch()
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned runtime and cold CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    need(torch.get_default_dtype() == torch.float32, 'FP32 default is unchanged')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    need(not torch.cuda.is_initialized(), 'Settings/imports must preserve cold CUDA')
    body = load(c.closed(spec['body_source']), 'U40_CONTINUOUS_TRAIN_BODY')
    result = body.run(torch, np, spec, c, pre)
    c.check()
    length = spec['segment']['stop']-spec['segment']['start']
    need(torch.cuda.is_initialized() and result['successful_optimizer_updates'] == 2*length
         and result['updates_per_arm'] == length and result['new_training'] is True
         and result['formal_main_training_started'] is True and result['pilot_only'] is False
         and all(result[k] is False for k in ('qualified', 'paired_prefix_advanced', 'AP', 'DEV', 'TEST')),
         'Two exact formal arm spans with pending Root paired-prefix acceptance')
    owned(c, result['report_ref'], output, 'REPORT.json')
    owned(c, result['log_ref'], output, 'probe.log')
    state_dir = Path(spec['segment_output_directory'])
    owned(c, result['pair_commit_ref'], state_dir, 'PAIR_COMMIT.json')
    for arm in ('PLAIN', 'FULL'):
        owned(c, result['checkpoint_refs'][arm], state_dir, arm+'.pth')
        owned(c, result['trace_refs'][arm], state_dir, arm+'.trace.jsonl')
    if spec['segment']['start'] == 0:
        owned(c, result['order_ref'], state_dir, 'ORDER.json')
    else:
        need(result['order_ref'] == spec['segment']['order_ref'], 'Same frozen full order on resume')
        c.closed(result['order_ref'])
    report = json.loads(Path(result['report_ref']['path']).read_bytes())
    core_report = report['core_report']
    need(report['status'] == 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
         and report['source_aliases_restored_exact'] is True and report['external_RNG_restored_exact'] is True
         and core_report['status'] == 'SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE'
         and core_report['paired_prefix_advanced'] is False
         and core_report['final_GPU_cleanup'] == dict(allocated=0, reserved=0)
         and all(core_report['arms'][arm]['status'] == 'CHECKPOINT_WRITTEN'
             and core_report['arms'][arm]['GPU_cleanup'] == dict(allocated=0, reserved=0)
             for arm in ('PLAIN', 'FULL')), 'Original paired checkpoint completion plus hard-zero CUDA')
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(CUDA_initialized=True, allocated=int(torch.cuda.memory_allocated(0)),
                   reserved=int(torch.cuda.memory_reserved(0)),
                   peak_allocated_bytes=report['CUDA_peak_allocated_bytes'])
    need(cleanup['allocated'] == cleanup['reserved'] == 0
         and 0 <= cleanup['peak_allocated_bytes'] <= 4096*1024**2, 'All stage peaks and final release in fixed budget')
    for value in required[:len(INFRA_ROLES)+len(MODULE_ROLES)]:
        need(c.ref(value['path']) == value, 'Executed source unchanged')
    c.check()
    q = dict(schema='U40-owned-pending-continuous-training-segment-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE', operation=OPERATION,
        phase=spec['phase'], source=spec['entry_source'], body_source=spec['body_source'],
        runtime_source=spec['runtime_source'], core_source=spec['core_source'], module_sources=spec['module_sources'],
        spec=spec_ref, admission=c.execution_admission_ref, context=spec['context'], prelock=spec['prelock'],
        plan=spec['plan'], U39_qualified=spec['U39_qualified'], resume_qualified=spec['resume_qualified'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'], result=result, segment=spec['segment'],
        owner_index=spec['owner_index'], normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'], legacy_pin_count=spec['legacy_pin_count'],
        full_metadata_population_images=741, full_metadata_population_GT=1661,
        admitted_tensor_indices=spec['admitted_tensor_indices'], pilot_only=False,
        formal_main_training_started=True, new_training=True, paired_prefix_advanced=False,
        epochs_target=50, total_updates_per_arm_target=37050, FIT=True, DEV=False, TEST=False, AP=False,
        elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available, cleanup=cleanup,
        qualified=False, original_actual0_self_certified=False, innovation_established=False, goal_complete=False)
    qref = c.write_json(output/'Q.json', q)
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False, paired_prefix_advanced=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Only Root owns the original formal segment leaf')
    run(args.spec)
