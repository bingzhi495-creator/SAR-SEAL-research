"""Unbound owned CPU check of preserved complete N28 model/optimizer state.

No training, forward/backward, GT, evaluation or CUDA initialization is allowed.
Outputs remain pending the original actual terminal and Root metadata intake.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROLES=('entry_source','body_source','factory_source','training_reader_source','train_core_source','runtime_source')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def ref(path):
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
    need(spec['schema']=='ROOT-N28-checkpoint-recovery-CPU-spec-v1'
         and spec['operation']=='CPU_N28_PAIRED_CHECKPOINT_RECOVERY'
         and spec['owner_index']==spec['normal_RAW_count']
         and spec['total_failed_RAW_count']==29
         and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+29
         and spec['automatic_retry'] is False and spec['deadline_seconds']==1800
         and spec['whole_GPU_limit_MiB']==8192 and spec['allocator_limit_MiB']==4096
         and spec['allowed_cpus']==[0,1] and spec['thread_limit']==2
         and spec['maximum_process_RSS_bytes']==spec['minimum_available_RAM_bytes']==16*1024**3
         and spec['cuda_visible_devices']=='' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
         and all(spec[key] is True for key in ('no_new_training','no_AP','no_DEV','no_TEST')),
         'Exact original CPU-only recovery operation and resource/accounting scope')
    source_map={value['path']:value for value in spec['sources']}
    need(len(source_map)==len(spec['sources']) and ref(__file__)==spec['entry_source']
         and all(source_map.get(spec[role]['path'])==spec[role] for role in ROLES),
         'All exact owned role sources')
    need(ref(spec['runtime_source']['path'])==spec['runtime_source'],'Exact resource-control source')
    runtime=load(spec['runtime_source']['path'],'N28_CHECKPOINT_CPU_RESOURCE_CONTROL')
    ad=json.loads(Path(spec['admission_file']).read_bytes())
    keys={'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
          'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
          'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(ad)==keys and len(ad)==13 and ad['scientific_spec']==ref(path)
         and ad['phase']==spec['phase'] and ad['entry']==spec['entry_source']['path']
         and ad['entry_args']==['--spec',str(path),'--owner-execute']
         and ad['allowed_cpus']==[0,1] and ad['thread_limit']==2 and ad['cuda_visible_devices']==''
         and ad['whole_GPU_used_limit_MiB']==8192 and ad['qualified_leaf_deadline_seconds']==1800
         and ad['automatic_retry'] is False
         and ad['bound_source_sha256']=={value['path']:value['sha256'] for value in spec['sources']+[ref(path)]},
         'Exact original13-field CPU admission')
    c=runtime.Control(spec);c.execution_spec_ref,c.execution_admission_ref=ref(path),ref(spec['admission_file'])
    parent=json.loads(Path(c.closed(spec['parent_ROOT_qualified'])).read_bytes())
    need(parent['qualified'] is True and parent['normal_RAW_count']==spec['normal_RAW_count']
         and parent['total_failed_RAW_count']==29
         and parent['schema']=='ROOT-N28-dense-routing-controlled-training-failure-preserved-v1'
         and parent['partial_checkpoints_qualified'] is False,
         'Preserved actual failure is the terminal predecessor, not checkpoint qualification')
    pre=json.loads(Path(c.closed(spec['prelock'])).read_bytes())
    output=Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink(),
         'Fresh owned recovery report directory')
    import torch
    import numpy as np
    need(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5'
         and not torch.cuda.is_initialized(),'Pinned cold CPU numerical runtime')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    body=load(c.closed(spec['body_source']),'N28_CHECKPOINT_CPU_PROBE_BODY')
    c.check();result=body.run(torch,np,spec,c,pre);c.check()
    need(not torch.cuda.is_initialized(),'Recovery probe never initialized CUDA')
    for role in ROLES:need(ref(spec[role]['path'])==spec[role],'Owned recovery source unchanged')
    need(ref(spec['recovery_checkpoint']['path'])==spec['recovery_checkpoint'],
         'Preserved original checkpoint bytes never modified')
    q=dict(schema='N28-owned-pending-checkpoint-recovery-CPU-v1',
           status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
           operation=spec['operation'],phase=spec['phase'],source=spec['entry_source'],
           body_source=spec['body_source'],spec=c.execution_spec_ref,admission=c.execution_admission_ref,
           prelock=spec['prelock'],parent_ROOT_qualified=spec['parent_ROOT_qualified'],
           result=result,owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
           total_failed_RAW_count=29,total_terminal_identity_pin_count=spec['normal_RAW_count']+29,
           legacy_pin_count=spec['legacy_pin_count'],new_training=False,AP=False,DEV=False,TEST=False,
           original_training_stage_successful=False,actual_CUDA_RNG_restore_verified=False,
           elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
           minimum_available_RAM_bytes=c.minimum_available,cleanup=dict(CUDA_initialized=False),
           qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False)
    qref=c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true');args=parser.parse_args()
    need(args.owner_execute,'Root owns the only scientific leaf')
    run(args.spec)
