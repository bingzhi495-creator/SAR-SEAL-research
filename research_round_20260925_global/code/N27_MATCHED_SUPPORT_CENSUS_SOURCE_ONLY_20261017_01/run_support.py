"""Root-owned CPU census of fixed support for the original matched FIT instances.

This entry neither trains nor evaluates AP. Its completion remains pending the
original admitted driver's actual terminal identity and independent Root intake.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def no_torch():
    need(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
         'Original admission/closure must precede all Torch imports')


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path, name):
    module_spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def output_reference(c, ref, output, filename):
    need(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'},
         'Completed output requires an exact three-field reference')
    path = Path(ref['path'])
    need(path.is_absolute() and path.name == filename
         and output.resolve() in path.resolve().parents and not path.is_symlink()
         and c.ref(path) == ref, 'Completed owned output identity: ' + filename)
    return path


def run(path):
    no_torch()
    path = Path(path)
    need(path.is_absolute(), 'Absolute original scientific spec required')
    spec = json.loads(path.read_bytes())
    need(spec['schema'] == 'ROOT-N27-matched-support-census-spec-v1'
         and spec['operation'] == 'CPU_MATCHED_SUPPORT_CENSUS'
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] ==
             spec['normal_RAW_count'] + spec['total_failed_RAW_count']
         and all(type(spec[k]) is int and spec[k] >= 0 for k in
                 ('owner_index', 'normal_RAW_count', 'total_failed_RAW_count',
                  'total_terminal_identity_pin_count', 'legacy_pin_count'))
         and spec['automatic_retry'] is False and spec['deadline_seconds'] == 1800
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == ''
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16 * 1024**3
         and spec['no_new_training'] is True and spec['no_TEST'] is True
         and spec['no_AP'] is True
         and 'quality_source' not in spec and 'setup_source' not in spec,
         'Fixed CPU support census, original accounting and resource scope')
    need(reference(__file__) == spec['entry_source']
         and reference(spec['runtime_source']['path']) == spec['runtime_source'],
         'Actual original entry/resource source identity')
    runtime = load(spec['runtime_source']['path'], 'N27_OWNED_SUPPORT_RESOURCE_CONTROL')
    no_torch()
    spec_ref = reference(path)
    admission_path = Path(spec['admission_file'])
    need(admission_path.is_absolute(), 'Absolute own original admission')
    admission = json.loads(admission_path.read_bytes())
    admission_ref = reference(admission_path)
    admission_keys = {'status', 'phase', 'driver_sha256', 'entry', 'entry_args',
                      'allowed_cpus', 'thread_limit', 'cuda_visible_devices',
                      'whole_GPU_used_limit_MiB', 'qualified_leaf_deadline_seconds',
                      'automatic_retry', 'scientific_spec', 'bound_source_sha256'}
    need(set(admission) == admission_keys and len(admission) == 13
         and admission['scientific_spec'] == spec_ref
         and admission['entry'] == spec['entry_source']['path']
         and admission['entry_args'] == ['--spec', str(path), '--owner-execute']
         and admission['phase'] == spec['phase']
         and admission['allowed_cpus'] == [0, 1] and admission['thread_limit'] == 2
         and admission['cuda_visible_devices'] == '' and admission['automatic_retry'] is False
         and admission['whole_GPU_used_limit_MiB'] == 8192
         and admission['qualified_leaf_deadline_seconds'] == 1800
         and admission['bound_source_sha256'] ==
             {r['path']: r['sha256'] for r in spec['sources'] + [spec_ref]},
         'Exact thirteen-field own admission and complete direct binding map')
    c = runtime.Control(spec)
    c.execution_spec_ref, c.execution_admission_ref = spec_ref, admission_ref
    c.check()
    need(len(c.allowed) == len(spec['sources']), 'Unique direct input paths')
    roles = ['entry_source', 'body_source', 'runtime_source']
    for key in roles:
        c.closed(spec[key])
    pre = json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema'] == 'N27-matched-support-census-prelock-v1'
         and pre['images'] == 741 and pre['GT'] == 1661
         and pre['Omega_abs_logit'] == 2.0
         and pre['thresholds'] == [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
         and all(pre[key] is True for key in ('no_new_training', 'no_DEV', 'no_TEST', 'no_AP')),
         'Complete matched FIT population and fixed support diagnostic scope')
    context = json.loads(c.closed(spec['context']).read_bytes())
    need(context['schema'] == 'ROOT-N27-matched-support-census-context-v1'
         and all(key in context for key in ('FIT_context', 'FIT_capture_reports', 'N26_inputs_qualified'))
         and len(context['FIT_capture_reports']) == 2,
         'Exact Root matched-support context')
    parent = json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] is True
         and parent['normal_RAW_count'] == spec['normal_RAW_count']
         and parent['total_failed_RAW_count'] == spec['total_failed_RAW_count'],
         'Current actual Root-qualified terminal predecessor')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink()
         and c.psutil.disk_usage(str(output.parent)).free >= 2 * 1024**3,
         'Fresh owned support census output and storage margin')
    no_torch()
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned original runtime with cold CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    c.check()
    body = load(c.closed(spec['body_source']), 'N27_OWNED_MATCHED_SUPPORT_CENSUS_BODY')
    need(not torch.cuda.is_initialized(), 'Support body import cannot initialize CUDA')
    result = body.run(torch, np, spec, c, pre)
    c.check()
    need(type(result) is dict and output.is_dir()
         and result['images'] == 741 and result['samples'] == 1661
         and all(result[key] is False for key in
                 ('qualified', 'AP', 'new_training', 'DEV', 'TEST')),
         'Complete matched FIT census remains pending actual0 and Root acceptance')
    report_path = output_reference(c, result['report_ref'], output, 'SUPPORT_REPORT.json')
    output_reference(c, result['counts_ref'], output, 'PER_MATCHED_INSTANCE_COUNTS.json')
    report = json.loads(report_path.read_bytes())
    need(report['schema'] == 'N27-matched-fixed-support-census-report-pending-v1',
         'Saved matched fixed-support census report schema')
    gc.collect()
    need(not torch.cuda.is_initialized(), 'CPU support census never initializes CUDA')
    for key in roles:
        need(reference(spec[key]['path']) == spec[key], 'Owned source unchanged at completion')
    output_reference(c, result['report_ref'], output, 'SUPPORT_REPORT.json')
    output_reference(c, result['counts_ref'], output, 'PER_MATCHED_INSTANCE_COUNTS.json')
    c.check()
    q = dict(schema='N27-owned-pending-matched-support-census-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        operation=spec['operation'], phase=spec['phase'], source=spec['entry_source'],
        body_source=spec['body_source'], runtime_source=spec['runtime_source'],
        spec=spec_ref, admission=admission_ref, prelock=spec['prelock'],
        context=spec['context'], parent_ROOT_qualified=spec['parent_ROOT_qualified'],
        result=result, owner_index=spec['owner_index'], normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        legacy_pin_count=spec['legacy_pin_count'],
        qualified=False, original_actual0_self_certified=False,
        new_training=False, AP=False, FIT=True, DEV=False, TEST=False,
        innovation_established=False, goal_complete=False, cleanup=dict(CUDA_initialized=False),
        elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available)
    qref = c.write_json(output / 'Q.json', q)
    need(not torch.cuda.is_initialized(), 'Cold CUDA is required through final pending receipt')
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Root owns the sole scientific CPU leaf')
    run(args.spec)
