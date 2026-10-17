"""UNBOUND single-FIT actual native100 M/F capture; no execution at import.

The original admitted Root leaf alone may run this entry. It reuses a frozen
N22 construction/forward, never infers M from L or a binary mask. No optimizer,
new training, PR-point actions, DEV, TEST or AP belongs to this diagnostic.
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

ROLES=('entry_source','body_source','runtime_source','legacy_capture_source')
def need(ok,message):
    if not ok:raise RuntimeError(message)
def no_torch():
    need(not any(x=='torch' or x.startswith('torch.') for x in sys.modules),'Source/admission before Torch')
def reference(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())
def strict(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
        and type(value['path']) is str and Path(value['path']).is_absolute()
        and type(value['bytes']) is int and value['bytes']>=0
        and type(value['sha256']) is str and len(value['sha256'])==64
        and set(value['sha256'])<=set('0123456789abcdef'),'Exact direct reference')
def load(path,name):
    path=Path(path);module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
def owned(c,value,folder,filename):
    strict(value);path=Path(value['path'])
    need(path.name==filename and folder.resolve() in path.resolve().parents
        and not path.is_symlink() and c.ref(path)==value,'Exact newly owned output')

def run(path):
    no_torch();path=Path(path);need(path.is_absolute(),'Absolute admitted spec')
    spec=json.loads(path.read_bytes());spec_ref=reference(path)
    count_keys=('owner_index','normal_RAW_count','total_failed_RAW_count','total_terminal_identity_pin_count','legacy_pin_count')
    need(spec['schema']=='ROOT-U32-native100-M-one-FIT-spec-v1'
        and spec['operation']=='GPU_CAPTURE_NATIVE100_M_ONE_FIT'
        and all(type(spec[k]) is int and spec[k]>=0 for k in count_keys)
        and spec['owner_index']==spec['normal_RAW_count']
        and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+spec['total_failed_RAW_count']
        and spec['dataset_index']==691 and spec['automatic_retry'] is False
        and spec['deadline_seconds']==1800 and spec['whole_GPU_limit_MiB']==8192
        and spec['allocator_limit_MiB']==4096 and spec['allowed_cpus']==[0,1]
        and spec['thread_limit']==2 and spec['cuda_visible_devices']=='0'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and spec['maximum_process_RSS_bytes']==spec['minimum_available_RAM_bytes']==16*1024**3
        and all(spec[k] is True for k in ('no_new_training','no_AP','no_DEV','no_TEST')),
        'Exact single-image FIT capture scope and resources')
    need(type(spec['sources']) is list and bool(spec['sources']),'Complete Source closure')
    source_map={}
    for value in spec['sources']:
        strict(value);need(value['path'] not in source_map,'Unique direct sources');source_map[value['path']]=value
    for key in ROLES+('legacy_N22_spec','anchor_N22_packet','prelock','parent_ROOT_qualified'):
        strict(spec[key]);need(source_map.get(spec[key]['path'])==spec[key],'Every declared role admitted')
    need(reference(__file__)==spec['entry_source'],'Actual original entry unchanged')
    admission_path=Path(spec['admission_file']);need(admission_path.is_absolute(),'Absolute admission')
    ad=json.loads(admission_path.read_bytes())
    keys={'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit','cuda_visible_devices',
        'whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds','automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(ad)==keys and len(ad)==13 and ad['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE'
        and ad['scientific_spec']==spec_ref and ad['phase']==spec['phase']
        and ad['entry']==spec['entry_source']['path'] and ad['entry_args']==['--spec',str(path),'--owner-execute']
        and ad['allowed_cpus']==[0,1] and ad['thread_limit']==2 and ad['cuda_visible_devices']=='0'
        and ad['automatic_retry'] is False and ad['whole_GPU_used_limit_MiB']==8192
        and ad['qualified_leaf_deadline_seconds']==1800
        and ad['bound_source_sha256']=={v['path']:v['sha256'] for v in spec['sources']+[spec_ref]},
        'Original thirteen-field admission and complete current Source map')
    for value in spec['sources']:need(reference(value['path'])==value,'Every Source unchanged before Torch')
    no_torch();runtime=load(spec['runtime_source']['path'],'U32_ACTUAL_NATIVE100_RESOURCE_CONTROL')
    no_torch();c=runtime.Control(spec);c.execution_spec_ref=spec_ref;c.execution_admission_ref=reference(admission_path)
    c.check();need(len(c.allowed)==len(spec['sources']),'Unique direct sources in actual Control')
    for value in spec['sources']:c.closed(value)
    need(c.psutil.Process().cpu_affinity()==[0,1]
        and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
        'Actual CPU01 and two-thread environment')
    parent=json.loads(c.closed(spec['parent_ROOT_qualified']).read_bytes())
    need(parent['qualified'] is True and all(parent[k]==spec[k] for k in count_keys[1:]),'Current Root actual terminal parent')
    pre=json.loads(c.closed(spec['prelock']).read_bytes())
    need(pre['schema']=='U32-native100-M-one-FIT-prelock-v1' and pre['dataset_index']==691
        and pre['images']==1 and pre['queries']==100 and pre['native_L_anchor_bitexact'] is True
        and all(pre[k] is True for k in ('no_new_training','no_AP','no_DEV','no_TEST')),'Fixed actual one-image capture')
    output=Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink()
        and c.psutil.disk_usage(str(output.parent)).free>=1024**3,'Fresh owned capture output and storage margin')
    no_torch()
    import torch
    import numpy as np
    need(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),
        'Pinned original runtime, cold CUDA before original CPU construction')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    body=load(c.closed(spec['body_source']),'U32_ACTUAL_NATIVE100_CAPTURE_BODY')
    need(not torch.cuda.is_initialized(),'Body import preserves cold CUDA')
    result=body.run(torch,np,spec,c,pre);c.check()
    need(result['dataset_index']==691 and result['image_id']==871 and result['images']==1 and result['queries']==100
        and all(result[k] is False for k in ('qualified','new_training','AP','DEV','TEST')),
        'Actual single FIT capture remains pending original actual0 and Root')
    owned(c,result['packet_ref'],output,'NATIVE100_FIELDS_0691.pth')
    owned(c,result['report_ref'],output,'CAPTURE_REPORT.json')
    need(torch.cuda.is_initialized(),'Actual original CUDA forward required')
    peak=torch.cuda.max_memory_allocated(0);gc.collect();torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();torch.cuda.synchronize(0)
    cleanup=dict(CUDA_initialized=True,allocated_after=torch.cuda.memory_allocated(0),
        reserved_after=torch.cuda.memory_reserved(0),peak_allocated_bytes=peak)
    need(cleanup['allocated_after']==cleanup['reserved_after']==0 and peak<=4096*1024**2,'Full GPU release and allocator ceiling')
    for value in spec['sources']:need(reference(value['path'])==value,'All actual owned sources unchanged')
    c.check()
    q=dict(schema='U32-owned-pending-native100-M-one-FIT-v1',status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        operation=spec['operation'],phase=spec['phase'],source=spec['entry_source'],body_source=spec['body_source'],
        spec=spec_ref,admission=c.execution_admission_ref,prelock=spec['prelock'],parent_ROOT_qualified=spec['parent_ROOT_qualified'],
        result=result,source_refs={k:spec[k] for k in ROLES},
        owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],legacy_pin_count=spec['legacy_pin_count'],
        dataset_index=691,image_id=871,images=1,queries=100,cleanup=cleanup,
        elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,minimum_available_RAM_bytes=c.minimum_available,
        qualified=False,original_actual0_self_certified=False,new_training=False,FIT=True,AP=False,DEV=False,TEST=False,
        utility_evaluated=False,innovation_established=False,goal_complete=False)
    qref=c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False,images=1,queries=100)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true');args=parser.parse_args()
    need(args.owner_execute,'Root owns the original single capture');run(args.spec)
