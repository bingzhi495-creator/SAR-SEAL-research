"""Owned fixed-E24 cached DEV shard or full-population official COCO leaf."""
import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    return module


def run(path):
    path = Path(path)
    spec = json.loads(path.read_bytes())
    need(spec['schema'] == 'ROOT-N23-fourhead-fixedDEV-spec-v1'
         and spec['operation'] in ('GPU_DEV', 'CPU_COCO')
         and spec['owner_index'] == spec['normal_RAW_count']
         and spec['total_failed_RAW_count'] == 28
         and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + 28
         and spec['automatic_retry'] is False and spec['deadline_seconds'] == 1800
         and spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096
         and spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2
         and spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16 * 1024**3
         and spec['no_new_training'] is True and spec['no_TEST'] is True,
         'Fixed DEV operation, accounting and resource contract')
    import hashlib
    def ref(p):
        p = Path(p)
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
        return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())
    need(ref(__file__) == spec['entry_source'] and ref(spec['runtime_source']['path']) == spec['runtime_source'],
         'Entry and reusable resource-control source exact')
    runtime = load(spec['runtime_source']['path'], 'N23_FIXEDDEV_OWNED_RESOURCE_CONTROL')
    ad = json.loads(Path(spec['admission_file']).read_bytes())
    need(len(ad) == 13 and ad['scientific_spec'] == ref(path)
         and ad['phase'] == spec['phase'] and ad['entry'] == spec['entry_source']['path']
         and ad['entry_args'] == ['--spec', str(path), '--owner-execute']
         and ad['allowed_cpus'] == [0, 1] and ad['thread_limit'] == 2
         and ad['cuda_visible_devices'] == spec['cuda_visible_devices']
         and ad['automatic_retry'] is False
         and ad['whole_GPU_used_limit_MiB'] == 8192 and ad['qualified_leaf_deadline_seconds'] == 1800
         and ad['bound_source_sha256'] == {r['path']: r['sha256'] for r in spec['sources'] + [ref(path)]},
         'Exact owned admission')
    c = runtime.Control(spec)
    c.execution_spec_ref, c.execution_admission_ref = ref(path), ref(spec['admission_file'])
    need(len(c.allowed) == len(spec['sources']), 'No aliasing direct inputs')
    pre = json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema'] == 'N23-fixedE24-fourhead-DEV-prelock-v1'
         and pre['heads'] == ['A_COMMON', 'A_ACTIVE', 'B_C', 'R']
         and pre['endpoint_extra_epoch'] == 24 and pre['images'] == 187 and pre['queries'] == 18700
         and pre['GT'] == 380 and pre['point_chunk'] == 2048
         and pre['query_filter'] is None and pre['score_update'] is False
         and pre['Omega_threshold'] == 2 and pre['binary_rule'] == '>0'
         and pre['GPU_ranges'] == [[0,47], [47,94], [94,141], [141,187]]
         and pre['historical_R_AP_points'] == 68.09938494745545,
         'Original four-head, full-population, endpoint and arithmetic rule lock')
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free >= 2 * 1024**3,
         'Fresh output and storage margin')
    parent = json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] and parent['normal_RAW_count'] == spec['normal_RAW_count']
         and parent['total_failed_RAW_count'] == 28, 'Current Root-qualified terminal predecessor')
    import torch
    import numpy as np
    need(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5'
         and not torch.cuda.is_initialized(), 'Pinned cold numerical runtime')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    need(spec['cuda_visible_devices'] == ('0' if spec['operation'] == 'GPU_DEV' else ''), 'Operation CUDA scope')
    body = load(c.closed(spec['body_source']), 'N23_FIXEDDEV_CACHED_FOURHEAD_BODY')
    c.check()
    result = body.run(torch, np, spec, c, pre)
    c.check()
    if spec['operation'] == 'GPU_DEV':
        peak = torch.cuda.max_memory_allocated(0)
        gc.collect()
        torch.cuda.synchronize(0)
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        cleanup = dict(allocated_after=torch.cuda.memory_allocated(0),
                       reserved_after=torch.cuda.memory_reserved(0), peak_allocated_bytes=peak)
        need(cleanup['allocated_after'] == cleanup['reserved_after'] == 0 and peak <= 4096 * 1024**2,
             'Complete GPU cleanup including cuBLAS')
    else:
        need(not torch.cuda.is_initialized(), 'CPU COCO never initialized CUDA')
        cleanup = dict(CUDA_initialized=False)
    for r in (spec['entry_source'], spec['body_source'], spec['core_source'], spec['runtime_source']):
        need(ref(r['path']) == r, 'Owned source unchanged')
    q = dict(schema='N23-owned-pending-fixedDEV-result-v1',
             status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
             operation=spec['operation'], phase=spec['phase'], source=spec['entry_source'],
             spec=ref(path), admission=ref(spec['admission_file']), prelock=spec['prelock'],
             result=result, owner_index=spec['owner_index'], normal_RAW_count=spec['normal_RAW_count'],
             total_failed_RAW_count=28, total_terminal_identity_pin_count=spec['normal_RAW_count'] + 28,
             legacy_pin_count=spec['legacy_pin_count'], qualified=False,
             original_actual0_self_certified=False, innovation_established=False, goal_complete=False,
             elapsed_seconds=time.monotonic()-c.started, peak_RSS_bytes=c.peak_rss,
             minimum_available_RAM_bytes=c.minimum_available, cleanup=cleanup,
             new_training=False, DEV=True, TEST=False, AP=(spec['operation']=='CPU_COCO'))
    qref = c.write_json(output / 'Q.json', q)
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--owner-execute', action='store_true')
    a = p.parse_args()
    need(a.owner_execute, 'Root owns the sole scientific leaf')
    run(a.spec)
