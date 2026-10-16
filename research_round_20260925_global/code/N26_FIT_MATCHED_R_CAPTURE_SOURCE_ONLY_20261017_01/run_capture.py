"""Root-owned data capture for one locked N26 phase, never training or AP."""
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


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    return module


def run(path):
    path = Path(path)
    need(path.is_absolute() and 'torch' not in sys.modules, 'Absolute spec before Torch import')
    spec = json.loads(path.read_bytes())
    gpu = spec['operation'] == 'GPU_FIT_QUALITY_CAPTURE'
    need(spec['operation'] in ('GPU_FIT_QUALITY_CAPTURE', 'CPU_DEV_QUALITY_FEATURES')
         and spec['schema'] == 'ROOT-N26-quality-data-capture-spec-v1'
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + spec['total_failed_RAW_count']
         and all(type(spec[k]) is int and spec[k] >= 0 for k in
                 ('owner_index','normal_RAW_count','total_failed_RAW_count','total_terminal_identity_pin_count','legacy_pin_count'))
         and spec['automatic_retry'] is False and spec['deadline_seconds'] == 1800
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['allowed_cpus'] == [0,1] and spec['thread_limit'] == 2
         and spec['cuda_visible_devices'] == ('0' if gpu else '')
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16 * 1024**3
         and spec['no_new_training'] is True and spec['no_TEST'] is True and spec['no_AP'] is True,
         'Fixed capture phase, original terminal accounting and resource scope')
    need(reference(__file__) == spec['entry_source']
         and reference(spec['runtime_source']['path']) == spec['runtime_source'], 'Actual entry/resource source')
    runtime = load(spec['runtime_source']['path'], 'N26_OWNED_CAPTURE_RESOURCE_CONTROL')
    need('torch' not in sys.modules, 'Control import cannot initialize numerical models')
    spec_ref = reference(path)
    admission_path = Path(spec['admission_file'])
    need(admission_path.is_absolute(), 'Absolute own admission')
    ad = json.loads(admission_path.read_bytes())
    admission_ref = reference(admission_path)
    keys = {'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
            'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
            'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(ad) == keys and len(ad) == 13 and ad['scientific_spec'] == spec_ref
         and ad['entry'] == spec['entry_source']['path'] and ad['entry_args'] == ['--spec',str(path),'--owner-execute']
         and ad['phase'] == spec['phase'] and ad['allowed_cpus'] == [0,1] and ad['thread_limit'] == 2
         and ad['cuda_visible_devices'] == spec['cuda_visible_devices'] and ad['automatic_retry'] is False
         and ad['whole_GPU_used_limit_MiB'] == 8192 and ad['qualified_leaf_deadline_seconds'] == 1800
         and ad['bound_source_sha256'] == {r['path']:r['sha256'] for r in spec['sources']+[spec_ref]},
         'Exact thirteen-field own admission and source closure')
    c = runtime.Control(spec)
    c.execution_spec_ref, c.execution_admission_ref = spec_ref, admission_ref
    c.check()
    need(len(c.allowed) == len(spec['sources']), 'Unique direct input sources')
    roles = ['entry_source','body_source','runtime_source','quality_source'] + (['setup_source'] if gpu else [])
    for key in roles:
        c.closed(spec[key])
    pre = json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema'] == 'N26-quality-data-capture-prelock-v1' and pre['no_new_training'] is True
         and pre['images'] == 741 and pre['GT'] == 1661 and pre['models'] == 'frozen_R_E24'
         and pre['DEV_images'] == 187 and pre['DEV_queries'] == 18700
         and pre['dataset_shards'] == [[0,371],[371,741]], 'Fixed N26 data scope')
    if gpu:
        need(spec['dataset_indices'] in [list(range(a,b)) for a,b in pre['dataset_shards']], 'One fixed whole FIT shard')
    else:
        need(spec['dataset_indices'] == list(range(187)), 'Complete DEV population')
    parent = json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] is True and parent['normal_RAW_count'] == spec['normal_RAW_count']
         and parent['total_failed_RAW_count'] == spec['total_failed_RAW_count'], 'Current Root-qualified predecessor')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink()
         and c.psutil.disk_usage(str(output.parent)).free >= 2 * 1024**3, 'Fresh output/storage margin')
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned original runtime, cold CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    c.check()
    body = load(c.closed(spec['body_source']), 'N26_OWNED_QUALITY_DATA_BODY')
    result = body.run(torch,np,spec,c,pre)
    c.check()
    need(type(result) is dict and output.is_dir(), 'Completed capture owns output')
    report_ref = result['report_ref']
    need(Path(report_ref['path']).is_absolute() and output.resolve() in Path(report_ref['path']).resolve().parents
         and c.ref(report_ref['path']) == report_ref, 'Completed capture report inside output')
    gc.collect()
    if gpu:
        need(torch.cuda.is_initialized(), 'Actual full FIT readout must execute on CUDA')
        peak = torch.cuda.max_memory_allocated(0)
        torch.cuda.synchronize(0)
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        cleanup = dict(CUDA_initialized=True,allocated_after=torch.cuda.memory_allocated(0),
                       reserved_after=torch.cuda.memory_reserved(0),peak_allocated_bytes=peak)
        need(cleanup['allocated_after'] == cleanup['reserved_after'] == 0 and peak <= 4096 * 1024**2,
             'Full CUDA release and allocator ceiling')
    else:
        need(not torch.cuda.is_initialized(), 'CPU DEV feature capture never initializes CUDA')
        cleanup = dict(CUDA_initialized=False)
    for key in roles:
        need(reference(spec[key]['path']) == spec[key], 'Actual own source unchanged at completion')
    c.check()
    q = dict(schema='N26-owned-pending-quality-data-capture-v1',
             status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
             operation=spec['operation'],phase=spec['phase'],source=spec['entry_source'],
             body_source=spec['body_source'],quality_source=spec['quality_source'],
             spec=spec_ref,admission=admission_ref,prelock=spec['prelock'],parent_ROOT_qualified=spec['parent_ROOT_qualified'],
             result=result,owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
             total_failed_RAW_count=spec['total_failed_RAW_count'],total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
             legacy_pin_count=spec['legacy_pin_count'],qualified=False,original_actual0_self_certified=False,
             new_training=False,DEV=(not gpu),FIT=gpu,TEST=False,AP=False,innovation_established=False,goal_complete=False,
             cleanup=cleanup,elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
             minimum_available_RAM_bytes=c.minimum_available)
    qref = c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args = parser.parse_args()
    need(args.owner_execute,'Root owns the sole scientific leaf')
    run(args.spec)
