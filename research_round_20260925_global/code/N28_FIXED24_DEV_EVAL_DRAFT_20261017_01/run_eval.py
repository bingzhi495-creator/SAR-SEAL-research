"""Unbound owned N28 fixed24 DEV prediction or full CPU COCO entry.

No actual evaluation has run. Root must create new complete bindings, publish
the final source and observe each original terminal before accepting outputs.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROLES=('entry_source','body_source','factory_source','reader_source','train_core_source',
       'population_source','geometry_source','runtime_source')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def reference(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def load(path,name):
    sp=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    return module


def run(path):
    path=Path(path);spec=json.loads(path.read_bytes())
    need(spec['schema']=='ROOT-N28-fixed24-dense-DEV-spec-v1'
         and spec['operation'] in ('GPU_N28_DENSE_FIXED24_DEV','CPU_N28_DENSE_FIXED24_COCO')
         and spec['owner_index']==spec['normal_RAW_count']
         and type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count']>=28
         and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and spec['automatic_retry'] is False and spec['deadline_seconds']==1800
         and spec['whole_GPU_limit_MiB']==8192 and spec['allocator_limit_MiB']==4096
         and spec['allowed_cpus']==[0,1] and spec['thread_limit']==2
         and spec['maximum_process_RSS_bytes']==spec['minimum_available_RAM_bytes']==16*1024**3
         and spec['no_new_training'] is True and spec['no_TEST'] is True,
         'Exact fixed24 operation, original accounting and resource contract')
    gpu=spec['operation']=='GPU_N28_DENSE_FIXED24_DEV'
    need(spec['cuda_visible_devices']==('0' if gpu else '')
         and os.environ.get('CUDA_VISIBLE_DEVICES')==spec['cuda_visible_devices'],
         'Operation-specific device scope before scientific imports')
    source_map={value['path']:value for value in spec['sources']}
    need(len(source_map)==len(spec['sources']) and reference(__file__)==spec['entry_source']
         and all(source_map.get(spec[role]['path'])==spec[role] for role in ROLES),
         'All exact owned role sources and no path alias')
    need(reference(spec['runtime_source']['path'])==spec['runtime_source'],'Exact original resource source')
    runtime=load(spec['runtime_source']['path'],'N28_FIXED24_OWNED_RESOURCE_CONTROL')
    admission=json.loads(Path(spec['admission_file']).read_bytes())
    keys={'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
          'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
          'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(admission)==keys and len(admission)==13
         and admission['scientific_spec']==reference(path) and admission['phase']==spec['phase']
         and admission['entry']==spec['entry_source']['path']
         and admission['entry_args']==['--spec',str(path),'--owner-execute']
         and admission['allowed_cpus']==[0,1] and admission['thread_limit']==2
         and admission['cuda_visible_devices']==spec['cuda_visible_devices']
         and admission['whole_GPU_used_limit_MiB']==8192
         and admission['qualified_leaf_deadline_seconds']==1800 and admission['automatic_retry'] is False
         and admission['bound_source_sha256']=={value['path']:value['sha256'] for value in spec['sources']+[reference(path)]},
         'Exact original 13-field owned admission and complete source map')
    c=runtime.Control(spec)
    c.execution_spec_ref,c.execution_admission_ref=reference(path),reference(spec['admission_file'])
    pre=json.loads(Path(c.closed(spec['prelock'])).read_bytes())
    parent=json.loads(Path(c.closed(spec['parent_ROOT_qualified'])).read_bytes())
    need(parent['qualified'] is True and parent['normal_RAW_count']==spec['normal_RAW_count']
         and parent['total_failed_RAW_count']==spec['total_failed_RAW_count'],
         'Current actual Root-qualified predecessor')
    output=Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists()
         and not output.is_symlink() and c.psutil.disk_usage(str(output.parent)).free>=2*1024**3,
         'Fresh owned output with storage margin')
    import torch
    import numpy as np
    need(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5'
         and not torch.cuda.is_initialized(),'Pinned initially cold scientific runtime')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    body=load(c.closed(spec['body_source']),'N28_FIXED24_DENSE_DEV_BODY')
    c.check();result=body.run(torch,np,spec,c,pre);c.check()
    if gpu:
        peak=int(torch.cuda.max_memory_allocated(0))
        gc.collect();torch.cuda.synchronize(0);torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache();torch.cuda.synchronize(0)
        cleanup=dict(CUDA_initialized=True,allocated_after=int(torch.cuda.memory_allocated(0)),
                     reserved_after=int(torch.cuda.memory_reserved(0)),peak_allocated_bytes=peak)
        need(cleanup['allocated_after']==cleanup['reserved_after']==0 and 0<peak<=4096*1024**2,
             'Complete actual CUDA cleanup within allocator limit')
    else:
        need(not torch.cuda.is_initialized(),'CPU evaluation never initialized CUDA')
        cleanup=dict(CUDA_initialized=False)
    for role in ROLES:need(reference(spec[role]['path'])==spec[role],'Owned source unchanged')
    q=dict(schema='N28-owned-pending-fixed24-dense-DEV-v1',
           status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
           operation=spec['operation'],phase=spec['phase'],source=spec['entry_source'],
           spec=c.execution_spec_ref,admission=c.execution_admission_ref,prelock=spec['prelock'],
           parent_ROOT_qualified=spec['parent_ROOT_qualified'],result=result,
           owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
           total_failed_RAW_count=spec['total_failed_RAW_count'],
           total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],legacy_pin_count=spec['legacy_pin_count'],
           seed=spec['seed'],fixed_epoch=24,new_training=False,DEV=True,TEST=False,AP=not gpu,
           elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
           minimum_available_RAM_bytes=c.minimum_available,cleanup=cleanup,
           qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False)
    qref=c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true');args=parser.parse_args()
    need(args.owner_execute,'Root owns the sole scientific leaf')
    run(args.spec)
