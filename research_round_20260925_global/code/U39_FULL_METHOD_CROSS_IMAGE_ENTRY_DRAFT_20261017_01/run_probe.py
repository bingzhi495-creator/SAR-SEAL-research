"""UNBOUND GPU cross-image disposable resource probe.

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
    'matched_reader', 'image_geometry', 'u36_helpers',
)
INFRA_ROLES = ('entry_source', 'body_source', 'runtime_source', 'probe_source')
SPEC_SCHEMA = 'ROOT-U39-full-method-cross-image-resource-spec-v1'
OPERATION = 'GPU_U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_PROBE'


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
    """Actual metadata guard; no CUDA call or science import."""
    counts = ('owner_index', 'normal_RAW_count', 'total_failed_RAW_count',
              'total_terminal_identity_pin_count', 'legacy_pin_count')
    need(spec['schema'] == SPEC_SCHEMA and spec['operation'] == OPERATION
         and all(type(spec[k]) is int and spec[k] >= 0 for k in counts)
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and type(spec['deadline_seconds']) is int and spec['deadline_seconds'] == 1800
         and spec['automatic_retry'] is False and spec['cuda_visible_devices'] == '0'
         and spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16*1024**3
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['full_metadata_population_images'] == 741 and spec['full_metadata_population_GT'] == 1661
         and spec['admitted_tensor_indices'] == [54, 710] and spec['tensor_images'] == 2
         and spec['image_order'] == [54, 710, 54]
         and spec['no_new_training'] is False and spec['new_training'] is True
         and spec['pilot_only'] is True and spec['formal_main_training_started'] is False
         and spec['formal_training_budget_credit'] == 0
         and all(spec[k] is True for k in ('no_AP', 'no_DEV', 'no_TEST'))
         and spec['full_method_training_eligible'] is False,
         'Fixed three-image two-arm disposable probe; metadata741, actual tensor images2')
    need(type(spec['module_sources']) is dict and set(spec['module_sources']) == set(MODULE_ROLES),
         'Exactly twelve resource-probe modules')
    need(type(spec['image_references']) is list and len(spec['image_references']) == 741,
         'All original JPEG refs pre-admitted without loading other image tensors')


def owned(c, value, output, name):
    strict_ref(value)
    path = Path(value['path'])
    need(path.resolve() == (output/name).resolve() and not path.is_symlink()
         and c.ref(path) == value, 'Exact owned report/log artifact')


def recorded_peaks(value):
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'CUDA_peak_allocated_bytes':
                need(type(item) is int and item >= 0, 'Actual nonnegative stage peak')
                result.append(item)
            else:
                result.extend(recorded_peaks(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(recorded_peaks(item))
    return result


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
    required += [spec[key] for key in ('context', 'prelock', 'plan', 'parent_ROOT_qualified', 'U38_qualified', 'U38_report')]
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
         and admission['cuda_visible_devices'] == '0' and admission['automatic_retry'] is False
         and admission['whole_GPU_used_limit_MiB'] == 8192
         and admission['qualified_leaf_deadline_seconds'] == 1800
         and admission['bound_source_sha256'] == {
             value['path']: value['sha256'] for value in spec['sources']+[spec_ref]},
         'Original thirteen-field admission and complete source map')
    for value in spec['sources']:
        need(ref(value['path']) == value, 'Pre-import closed source/input hash')
    no_torch()
    runtime = load(spec['runtime_source']['path'], 'U39_GPU_RESOURCE_CONTROL')
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
    u38 = json.loads(Path(c.closed(spec['U38_qualified'])).read_bytes())
    u38_report = json.loads(Path(c.closed(spec['U38_report'])).read_bytes())
    need(u38['schema'] == 'ROOT-U38-full-method-CPU-reader-init-qualified-v1'
         and u38['qualified'] is True and u38['worker_complete_CPU_reader_and_initialization_reported'] is True
         and u38['result']['report_ref'] == u38['contract_report'] == spec['U38_report']
         and u38['images'] == 741 and u38['GT'] == 1661 and u38['common_states'] == 104
         and u38['full_method_training_eligible'] is False,
         'Actual accepted U38 CPU scope, no inherited GPU or main-training eligibility')
    need(u38_report['schema'] == 'U38-full-method-CPU-reader-init-report-pending-v1'
         and u38_report['status'] == 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
         and u38_report['context'] == spec['context']
         and u38_report['qualification']['images_completed'] == 741
         and u38_report['qualification']['GT_completed'] == 1661,
         'Reuse exactly the actual U38 population/context; no new all-population pixel qualification')
    import hashlib as _hashlib
    expected_image_digest = _hashlib.sha256(json.dumps(spec['image_references'], sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    need(u38_report['qualification']['image_references_metadata_sha256'] == expected_image_digest,
         'Same full JPEG ref list as the actual U38 qualifier')
    pre = json.loads(Path(c.closed(spec['prelock'])).read_bytes())
    need(pre['schema'] == 'U39-full-method-cross-image-resource-prelock-v1'
         and pre['image_order'] == [54, 710, 54] and pre['arms'] == ['PLAIN', 'FULL']
         and pre['updates_per_arm'] == 3 and pre['total_pilot_updates'] == 6
         and pre['image_batch_size'] == pre['query_chunk'] == 1
         and pre['branch_size'] == 400 and pre['canvas_size'] == 800
         and pre['initialization_seed'] == 202610173101 and pre['dtype'] == 'float32'
         and pre['training_encode_saved_tensor_mode'] == 'save_on_cpu_pin_memory_true'
         and pre['optimizer'] == dict(name='AdamW', lr=3e-4, weight_decay=1e-4,
             betas=[.9, .999], eps=1e-8, foreach=False, fused=False)
         and all(pre[k] is False for k in ('checkpointing', 'AMP', 'VJP', 'automatic_retry', 'main_training'))
         and all(pre[k] is True for k in ('no_AP', 'no_DEV', 'no_TEST'))
         and pre['plan'] == spec['plan'], 'Exact resource_probe prelock; six disposable updates')
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    need(not torch.cuda.is_initialized(), 'Settings/imports must preserve cold CUDA')
    body = load(c.closed(spec['body_source']), 'U39_CROSS_IMAGE_RESOURCE_BODY')
    result = body.run(torch, np, spec, c, pre)
    c.check()
    need(torch.cuda.is_initialized() and result['pilot_optimizer_updates'] == {'PLAIN': 3, 'FULL': 3}
         and result['successful_optimizer_updates'] == 6 and result['unique_images'] == 2
         and result['new_training'] is True and result['pilot_only'] is True
         and result['formal_training_budget_credit'] == 0
         and all(result[k] is False for k in ('qualified', 'formal_main_training_started',
             'full_method_training_eligible', 'AP', 'DEV', 'TEST')),
         'Only six actual disposable updates complete; no formal training/efficacy claim')
    owned(c, result['report_ref'], output, 'REPORT.json')
    owned(c, result['log_ref'], output, 'probe.log')
    report = json.loads(Path(result['report_ref']['path']).read_bytes())
    p = report['resource_probe']
    need(report['schema'] == 'U39-full-method-cross-image-resource-wrapper-report-pending-v1'
         and report['status'] == 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
         and report['source_aliases_restored_exact'] is True and report['CPU_RNG_restored_exact'] is True
         and p['schema'] == 'U39-full-method-cross-image-resource-report-pending-v1'
         and p['status'] == 'COMPLETED_PENDING_ROOT_REVIEW'
         and p['successful_optimizer_updates'] == p['optimizer_steps_entered'] == 6
         and p['image_order'] == [54, 710, 54] and p['distinct_probe_images'] == 2
         and p['final_CUDA_cleanup'] == dict(CUDA_allocated_bytes=0, CUDA_reserved_bytes=0)
         and all(p['arms'][arm]['status'] == 'PASS' and p['arms'][arm]['updates_completed'] == 3
             and p['arms'][arm]['after_arm_release'] == dict(CUDA_allocated_bytes=0, CUDA_reserved_bytes=0)
             for arm in ('PLAIN', 'FULL')),
         'Original complete resource probe and both arm allocations/cache fully released')
    peaks = recorded_peaks(report)
    need(peaks and max(peaks) <= 4096*1024**2, 'All real stage allocator peaks within unchanged 4GiB')
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(CUDA_initialized=True, CUDA_allocated_bytes=int(torch.cuda.memory_allocated(0)),
                   CUDA_reserved_bytes=int(torch.cuda.memory_reserved(0)),
                   peak_allocated_bytes=max(peaks),
                   peak_scope='maximum actual recursive worker stage peaks plus final allocator interval')
    need(cleanup['CUDA_allocated_bytes'] == cleanup['CUDA_reserved_bytes'] == 0,
         'Final original entry hard-zero CUDA cleanup')
    for value in required[:len(INFRA_ROLES)+len(MODULE_ROLES)]:
        need(c.ref(value['path']) == value, 'Executed source unchanged')
    c.check()
    q = dict(schema='U39-owned-pending-full-method-cross-image-resource-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE', operation=OPERATION,
        phase=spec['phase'], source=spec['entry_source'], body_source=spec['body_source'],
        runtime_source=spec['runtime_source'], probe_source=spec['probe_source'],
        module_sources=spec['module_sources'], spec=spec_ref, admission=c.execution_admission_ref,
        context=spec['context'], prelock=spec['prelock'], plan=spec['plan'],
        U38_qualified=spec['U38_qualified'], U38_report=spec['U38_report'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'], result=result,
        owner_index=spec['owner_index'], normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'], legacy_pin_count=spec['legacy_pin_count'],
        full_metadata_population_images=741, full_metadata_population_GT=1661,
        admitted_tensor_indices=[54, 710], unique_tensor_images=2, image_order=[54, 710, 54],
        new_training=True, pilot_only=True, formal_main_training_started=False, formal_training_budget_credit=0,
        successful_optimizer_updates=6, full_method_training_eligible=False,
        FIT=True, DEV=False, TEST=False, AP=False,
        elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available, cleanup=cleanup,
        qualified=False, original_actual0_self_certified=False, innovation_established=False, goal_complete=False)
    qref = c.write_json(output/'Q.json', q)
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False, cross_image_disposable_resource_only=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Only Root owns the original single GPU resource leaf')
    run(args.spec)
