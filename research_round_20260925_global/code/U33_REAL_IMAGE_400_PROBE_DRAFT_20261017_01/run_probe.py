"""UNBOUND Root-owned real-image400 disposable two-update pilot entry.

This is source-only preparation. The pilot trains neither a complete FIT model
nor an AP-selected endpoint. Original failed tests/errors must remain visible.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

PROBE_ROLES = (
    'full_legacy', 'full_dsconv', 'full_head', 'full_feature',
    'plain_legacy', 'plain_feature', 'instance_adapter',
    'image_geometry', 'geometry_test', 'plain_test',
)
ROLES = ('entry_source', 'body_source', 'runtime_source') + PROBE_ROLES
SPEC_SCHEMA = 'ROOT-U33-real-image400-two-update-spec-v1'
OPERATION = 'GPU_U33_REAL_IMAGE400_TWO_UPDATE_PROBE'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def no_torch():
    need(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
         'All source/admission checks must precede Torch')


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def strict(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Exact closed reference')


def canonical_reference(value):
    strict(value)
    return dict(value, path=str(Path(value['path'])))


def load(path, name):
    path = Path(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def owned(c, value, folder, filename):
    strict(value)
    path = Path(value['path'])
    need(path.name == filename and folder.resolve() in path.resolve().parents
         and not path.is_symlink() and c.ref(path) == value, 'Exact newly owned output')


def recorded_allocator_peaks(value):
    """The body resets allocator peaks at stage boundaries; retain every interval."""
    peaks = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'CUDA_peak_allocated_bytes':
                need(type(item) is int and item >= 0, 'Actual recorded CUDA interval peak')
                peaks.append(item)
            else:
                peaks.extend(recorded_allocator_peaks(item))
    elif isinstance(value, list):
        for item in value:
            peaks.extend(recorded_allocator_peaks(item))
    return peaks


def run(path):
    no_torch()
    path = Path(path)
    need(path.is_absolute(), 'Absolute original admitted spec')
    spec = json.loads(path.read_bytes())
    spec_ref = reference(path)
    count_keys = ('owner_index', 'normal_RAW_count', 'total_failed_RAW_count',
                  'total_terminal_identity_pin_count', 'legacy_pin_count')
    need(spec['schema'] == SPEC_SCHEMA and spec['operation'] == OPERATION
         and all(type(spec[key]) is int and spec[key] >= 0 for key in count_keys)
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + spec['total_failed_RAW_count']
         and spec['automatic_retry'] is False and spec['deadline_seconds'] == 1800
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16*1024**3
         and spec['new_training'] is True and spec['no_new_training'] is False
         and spec['pilot_only'] is True and spec['formal_main_training_started'] is False
         and all(spec[key] is True for key in ('no_AP', 'no_DEV', 'no_TEST')),
         'Exact two-update pilot scope and unchanged resources')
    need(len(ROLES) == 13 and len(set(ROLES)) == 13, 'Three entry roles plus ten probe roles')
    need(type(spec['sources']) is list and bool(spec['sources']), 'Complete direct closure')
    source_map = {}
    for value in spec['sources']:
        strict(value)
        need(value == canonical_reference(value) and value['path'] not in source_map,
             'Canonical unique direct Source path')
        source_map[value['path']] = value
    for key in ROLES + ('native_capture_qualified', 'native_fields_packet', 'image',
                        'prelock', 'parent_ROOT_qualified', 'fresh_pure_FIRST', 'input_context', 'plan'):
        value = canonical_reference(spec[key])
        need(source_map.get(value['path']) == value, 'Every role and direct input admitted: '+key)
    need(reference(__file__) == spec['entry_source'], 'Actual original entry Source unchanged')
    admission_path = Path(spec['admission_file'])
    need(admission_path.is_absolute(), 'Absolute admission')
    ad = json.loads(admission_path.read_bytes())
    keys = {'status', 'phase', 'driver_sha256', 'entry', 'entry_args', 'allowed_cpus',
            'thread_limit', 'cuda_visible_devices', 'whole_GPU_used_limit_MiB',
            'qualified_leaf_deadline_seconds', 'automatic_retry', 'scientific_spec', 'bound_source_sha256'}
    need(set(ad) == keys and len(ad) == 13
         and ad['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE'
         and ad['scientific_spec'] == spec_ref and ad['phase'] == spec['phase']
         and ad['entry'] == spec['entry_source']['path']
         and ad['entry_args'] == ['--spec', str(path), '--owner-execute']
         and ad['allowed_cpus'] == [0, 1] and ad['thread_limit'] == 2
         and ad['cuda_visible_devices'] == '0' and ad['automatic_retry'] is False
         and ad['whole_GPU_used_limit_MiB'] == 8192 and ad['qualified_leaf_deadline_seconds'] == 1800
         and ad['bound_source_sha256'] == {value['path']: value['sha256'] for value in spec['sources']+[spec_ref]},
         'Original thirteen-field admission and complete bound Source map')
    for value in spec['sources']:
        need(reference(value['path']) == value, 'Every admitted Source unchanged before Torch')
    no_torch()
    runtime = load(spec['runtime_source']['path'], 'U33_REAL_IMAGE400_RESOURCE_CONTROL')

    class CanonicalControl(runtime.Control):
        def closed(self, value):
            return super().closed(canonical_reference(value))

    no_torch()
    c = CanonicalControl(spec)
    c.execution_spec_ref = spec_ref
    c.execution_admission_ref = reference(admission_path)
    c.check()
    need(len(c.allowed) == len(spec['sources']), 'Unique exact direct Source closure')
    for value in spec['sources']:
        c.closed(value)
    need(c.psutil.Process().cpu_affinity() == [0, 1]
         and all(os.environ.get(key) == '2' for key in (
             'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
         'Actual CPU01 and two-thread environment')
    parent = json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] is True and all(parent[key] == spec[key] for key in count_keys[1:]),
         'Current Root actual terminal parent and dynamic counts')
    capture = json.loads(c.closed(spec['native_capture_qualified']).read_bytes())
    need(capture['schema'] == 'ROOT-U32-native100-M-one-FIT-qualified-v1'
         and capture['qualified'] is True and capture['dataset_index'] == 691
         and capture['image_id'] == 871 and capture['images'] == 1
         and capture['queries'] == 100 and capture['raw_GT'] == 9
         and capture['result']['packet_ref'] == spec['native_fields_packet']
         and capture['packet_ref'] == spec['native_fields_packet']
         and capture['identity']['image'] == spec['image']
         and capture['worker_native100_M_F_capture_reported'] is True,
         'Accepted actual U32 native100 fields and image identity, no inherited training eligibility')
    pre = json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema'] == 'U33-real-image400-two-update-prelock-v1'
         and pre['arms'] == ['PLAIN', 'FULL'] and pre['dataset_index'] == 691
         and pre['image_id'] == 871 and pre['queries'] == 100 and pre['GT'] == 9
         and pre['query_chunk'] == 4 and pre['input_size'] == 400
         and pre['original_shape'] == [283, 503] and pre['content_shape'] == [450, 800]
         and pre['geometry_tests'] == pre['plain64_tests'] == 1
         and pre['stage1_atol'] == 2e-5 and pre['stage1_rtol'] == 1e-5
         and pre['optimizer_updates_per_arm'] == 2
         and pre['optimizer'] == dict(name='AdamW', lr=3e-4, weight_decay=1e-4, betas=[.9, .999], eps=1e-8)
         and pre['stage1_restore_zero_before_training'] is True and pre['new_training'] is True
         and pre['pilot_only'] is True and pre['formal_main_training_started'] is False
         and all(pre[key] is False for key in ('AP', 'DEV', 'TEST'))
         and all(type(pre[key]) is int and pre[key] >= 0 for key in (
             'backbone_initialization_seed', 'adapter_initialization_seed'))
         and pre['backbone_initialization_seed'] == 202610173330
         and pre['adapter_initialization_seed'] == 202610173331,
         'Exact original public CPU tests and fixed real-image400 disposable updates')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free >= 1024**3,
         'Fresh owned output directory and storage margin')
    no_torch()
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned runtime and cold CUDA before CPU contracts')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    body = load(c.closed(spec['body_source']), 'U33_REAL_IMAGE400_PROBE_BODY')
    need(not torch.cuda.is_initialized(), 'Body import must preserve cold CUDA')
    result = body.run(torch, np, spec, c, pre)
    c.check()
    need(result['pilot_optimizer_updates'] == {'PLAIN': 2, 'FULL': 2}
         and result['new_training'] is True and result['pilot_only'] is True
         and all(result[key] is False for key in (
             'qualified', 'formal_main_training_started', 'full_method_training_eligible', 'AP', 'DEV', 'TEST')),
         'Both disposable pilot arms complete; no full FIT or AP qualification')
    owned(c, result['report_ref'], output, 'REPORT.json')
    owned(c, result['log_ref'], output, 'probe.log')
    report = json.loads(Path(result['report_ref']['path']).read_bytes())
    need(report['schema'] == 'U33-real-image400-two-update-report-v1'
         and report['status'] == 'COMPLETED_PENDING_ROOT_REVIEW'
         and report['successful_optimizer_updates'] == 4 and report['new_training'] is True
         and all(report['arms'][arm]['status'] == 'PASS' for arm in ('PLAIN', 'FULL')),
         'Original worker pilot report, no independent neural replay claimed')
    interval_peaks = recorded_allocator_peaks(report)
    need(bool(interval_peaks), 'Actual GPU stage peaks must be preserved despite peak resets')
    need(torch.cuda.is_initialized(), 'An actual original CUDA pilot is required')
    peak = max(interval_peaks+[torch.cuda.max_memory_allocated(0)])
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(CUDA_initialized=True, allocated_after=torch.cuda.memory_allocated(0),
                   reserved_after=torch.cuda.memory_reserved(0), peak_allocated_bytes=peak,
                   peak_scope='maximum actual worker stage allocator peaks plus final interval')
    need(cleanup['allocated_after'] == cleanup['reserved_after'] == 0
         and peak <= 4096*1024**2, 'Full CUDA release and unchanged allocator ceiling')
    for value in spec['sources']:
        need(reference(value['path']) == value, 'Every Source and direct input unchanged after pilot')
    c.check()
    q = dict(schema='U33-owned-pending-real-image400-two-update-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE', operation=spec['operation'],
        phase=spec['phase'], source=spec['entry_source'], body_source=spec['body_source'],
        spec=spec_ref, admission=c.execution_admission_ref, source_refs={key: spec[key] for key in ROLES},
        prelock=spec['prelock'], parent_ROOT_qualified=spec['parent_ROOT_qualified'],
        native_capture_qualified=spec['native_capture_qualified'], native_fields_packet=spec['native_fields_packet'],
        image=spec['image'], result=result, owner_index=spec['owner_index'],
        normal_RAW_count=spec['normal_RAW_count'], total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'], legacy_pin_count=spec['legacy_pin_count'],
        cleanup=cleanup, elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available, qualified=False, original_actual0_self_certified=False,
        new_training=True, pilot_only=True, formal_main_training_started=False,
        full_method_training_eligible=False, all741_qualification=False,
        FIT=True, AP=False, DEV=False, TEST=False, utility_evaluated=False, innovation_established=False, goal_complete=False)
    qref = c.write_json(output/'Q.json', q)
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False, pilot_only=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Only the actual original Root-owned leaf may execute')
    run(args.spec)
