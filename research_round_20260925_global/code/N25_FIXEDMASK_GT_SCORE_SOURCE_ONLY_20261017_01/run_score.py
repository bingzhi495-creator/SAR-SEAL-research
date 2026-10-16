"""Owned CPU-only fixed-mask GT score diagnostic; no learned-policy AP.

The reused resource module is imported under a private module name; only its
Control class is used. Its training CLI and numerical entry are never called.
Root alone freezes, binds, launches, and accepts this pending-result entry.
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
         'CPU GT score diagnostic must never import Torch')


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path, name):
    module_spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    no_torch()
    return module


def run(path):
    no_torch()
    path = Path(path)
    need(path.is_absolute(), 'Absolute scientific spec required')
    spec = json.loads(path.read_bytes())
    need(spec['schema'] == 'ROOT-N25-fixedmask-GT-score-spec-v1'
         and spec['operation'] == 'CPU_ORACLE_SCORE_DIAGNOSTIC'
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
         and spec['no_new_training'] is True and spec['no_TEST'] is True,
         'Fixed CPU GT score diagnostic, accounting and resource scope')
    source_roles = ('entry_source', 'body_source', 'core_source', 'runtime_source')
    need(reference(__file__) == spec['entry_source']
         and reference(spec['runtime_source']['path']) == spec['runtime_source'],
         'Actual entry and reusable Control source identity')
    runtime = load(spec['runtime_source']['path'], 'N25_FULLDEV_OWNED_RESOURCE_CONTROL')
    admission_path = Path(spec['admission_file'])
    need(admission_path.is_absolute(), 'Absolute own admission required')
    ad = json.loads(admission_path.read_bytes())
    admission_keys = {'status', 'phase', 'driver_sha256', 'entry', 'entry_args',
                      'allowed_cpus', 'thread_limit', 'cuda_visible_devices',
                      'whole_GPU_used_limit_MiB', 'qualified_leaf_deadline_seconds',
                      'automatic_retry', 'scientific_spec', 'bound_source_sha256'}
    spec_ref, admission_ref = reference(path), reference(admission_path)
    need(set(ad) == admission_keys and len(ad) == 13
         and ad['scientific_spec'] == spec_ref
         and ad['phase'] == spec['phase'] and ad['entry'] == spec['entry_source']['path']
         and ad['entry_args'] == ['--spec', str(path), '--owner-execute']
         and ad['allowed_cpus'] == [0, 1] and ad['thread_limit'] == 2
         and ad['cuda_visible_devices'] == '' and ad['automatic_retry'] is False
         and ad['whole_GPU_used_limit_MiB'] == 8192
         and ad['qualified_leaf_deadline_seconds'] == 1800
         and ad['bound_source_sha256'] ==
             {r['path']: r['sha256'] for r in spec['sources'] + [spec_ref]},
         'Exact 13-field owned admission and complete direct binding map')
    c = runtime.Control(spec)
    c.execution_spec_ref, c.execution_admission_ref = spec_ref, admission_ref
    c.check()
    need(len(c.allowed) == len(spec['sources']), 'Unique direct input paths required')
    for key in source_roles:
        c.closed(spec[key])
    pre = json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema'] == 'N25-fixedmask-GT-score-prelock-v1'
         and pre['images'] == 187 and pre['GT'] == 380 and pre['queries'] == 18700
         and pre['models'] == ['native_official', 'R', 'A_COMMON', 'A_ACTIVE', 'B_C']
         and pre['score_modes'] == ['ORIGINAL_S0', 'GT_PRODUCT', 'GT_IOU_ONLY'],
         'Original full-population models and three fixed GT score modes')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink()
         and c.psutil.disk_usage(str(output.parent)).free >= 1024**3,
         'Fresh owned output and at least 1 GiB parent storage margin')
    parent = json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] is True
         and parent['normal_RAW_count'] == spec['normal_RAW_count']
         and parent['total_failed_RAW_count'] == spec['total_failed_RAW_count'],
         'Current actual Root-qualified terminal predecessor')
    no_torch()
    import numpy as np
    need(np.__version__ == '1.23.5', 'Pinned NumPy 1.23.5 CPU GT score diagnostic runtime')
    no_torch()
    c.check()
    body = load(c.closed(spec['body_source']), 'N25_FULLDEV_FIXEDMASK_GT_SCORE_BODY')
    result = body.run(np, spec, c, pre)
    c.check()
    no_torch()
    need(type(result) is dict and output.is_dir(), 'Completed body owns its output directory')
    report_ref = result['report_ref']
    need(type(report_ref) is dict and set(report_ref) == {'path', 'bytes', 'sha256'}
         and Path(report_ref['path']).is_absolute()
         and output.resolve() in Path(report_ref['path']).resolve().parents
         and c.ref(report_ref['path']) == report_ref,
         'Completed report reference inside this fresh output')
    for key in source_roles:
        need(reference(spec[key]['path']) == spec[key], 'Owned source unchanged at completion')
    gc.collect()
    c.check()
    no_torch()
    q = dict(schema='N25-owned-pending-fixedmask-GT-score-v1',
             status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
             operation=spec['operation'], phase=spec['phase'], source=spec['entry_source'],
             body_source=spec['body_source'], core_source=spec['core_source'],
             runtime_source=spec['runtime_source'], spec=spec_ref, admission=admission_ref,
             prelock=spec['prelock'], parent_ROOT_qualified=spec['parent_ROOT_qualified'],
             result=result, owner_index=spec['owner_index'], normal_RAW_count=spec['normal_RAW_count'],
             total_failed_RAW_count=spec['total_failed_RAW_count'],
             total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
             legacy_pin_count=spec['legacy_pin_count'], qualified=False,
             original_actual0_self_certified=False, innovation_established=False, goal_complete=False,
             elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
             minimum_available_RAM_bytes=c.minimum_available,
             cleanup=dict(Torch_imported=False, CUDA_initialized=False),
             new_training=False, DEV=True, TEST=False, AP=True,
             AP_kind='GT_ORACLE_DIAGNOSTIC_NOT_LEARNED_METHOD',
             no_new_deployable_policy_AP=True, existing_metrics_recomputed=True)
    qref = c.write_json(output / 'Q.json', q)
    no_torch()
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Root owns the sole scientific CPU leaf')
    run(args.spec)
