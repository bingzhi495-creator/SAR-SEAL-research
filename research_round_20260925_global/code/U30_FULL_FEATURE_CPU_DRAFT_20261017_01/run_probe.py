"""UNBOUND Root-owned CPU size64 real-full feature contract entry; not executed.

Only the original admitted Root leaf may run this entry. Complete source and
admission checks precede Torch import. Only size64 eval-mode real full convolutional feature modules and a disposable
synthetic adapter backward belong here. No checkpoint, real image, GT, optimizer,
400-size resource qualification, renderer or AP evaluation belongs to this task.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types


ROLES=('entry_source','body_source','runtime_source','legacy_source','dsconv_source',
       'head_source','feature_source','test_source','adapter_source')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def no_torch():
    need(not any(name=='torch' or name.startswith('torch.') for name in sys.modules),
         'Complete original admission and source closure must precede Torch import')


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Exact closed reference required')


def ref(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def load(path,name):
    path=Path(path);module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def output_ref(c,value,output):
    strict_ref(value);path=Path(value['path'])
    need(output.resolve() in path.resolve().parents and not path.is_symlink()
         and c.ref(path)==value,'Exact new artifact inside owned output')


def run(path):
    no_torch();path=Path(path)
    need(path.is_absolute(),'Absolute Root-owned contract spec')
    spec=json.loads(path.read_bytes())
    count_keys=('owner_index','normal_RAW_count','total_failed_RAW_count',
                'total_terminal_identity_pin_count','legacy_pin_count')
    need(spec['schema']=='ROOT-U30-full-feature-CPU-spec-v1'
         and spec['operation']=='CPU_U30_FULL_FEATURE_CONTRACT'
         and all(type(spec[key]) is int and spec[key]>=0 for key in count_keys)
         and spec['owner_index']==spec['normal_RAW_count']
         and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+spec['total_failed_RAW_count']
         and spec['automatic_retry'] is False and type(spec['deadline_seconds']) is int
         and 0<spec['deadline_seconds']<=1800
         and spec['whole_GPU_limit_MiB']==8192 and spec['allocator_limit_MiB']==4096
         and spec['allowed_cpus']==[0,1] and spec['thread_limit']==2
         and spec['maximum_process_RSS_bytes']==spec['minimum_available_RAM_bytes']==16*1024**3
         and spec['cuda_visible_devices']=='' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
         and all(spec[key] is True for key in ('no_new_training','no_AP','no_DEV','no_TEST')),
         'Exact size64 CPU contract scope, dynamic accounting and resource limits')
    need(type(spec['sources']) is list and bool(spec['sources']),'Complete closed inputs')
    source_map={}
    for value in spec['sources']:
        strict_ref(value);need(value['path'] not in source_map,'Unique source paths')
        source_map[value['path']]=value
    for role in ROLES:
        strict_ref(spec[role]);need(source_map.get(spec[role]['path'])==spec[role],'Every source role admitted')
    need(ref(__file__)==spec['entry_source'] and ref(spec['runtime_source']['path'])==spec['runtime_source'],
         'Actual entry and reusable Control source unchanged')
    spec_ref=ref(path);admission_path=Path(spec['admission_file'])
    need(admission_path.is_absolute(),'Absolute original admission file')
    admission=json.loads(admission_path.read_bytes())
    keys={'status','phase','driver_sha256','entry','entry_args','allowed_cpus','thread_limit',
          'cuda_visible_devices','whole_GPU_used_limit_MiB','qualified_leaf_deadline_seconds',
          'automatic_retry','scientific_spec','bound_source_sha256'}
    need(set(admission)==keys and len(admission)==13
         and admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE'
         and admission['scientific_spec']==spec_ref and admission['phase']==spec['phase']
         and admission['entry']==spec['entry_source']['path']
         and admission['entry_args']==['--spec',str(path),'--owner-execute']
         and admission['allowed_cpus']==[0,1] and admission['thread_limit']==2
         and admission['cuda_visible_devices']=='' and admission['automatic_retry'] is False
         and admission['whole_GPU_used_limit_MiB']==8192
         and admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds']
         and admission['bound_source_sha256']=={value['path']:value['sha256'] for value in spec['sources']+[spec_ref]},
         'Exact original thirteen-field CPU admission and full source map')
    # Verify every declared input, including the Control source, before loading
    # even that stdlib/resource module, and again through its actual closed ABI.
    for value in spec['sources']:need(ref(value['path'])==value,'Complete pre-Torch closed-source hash verification')
    no_torch()
    runtime=load(spec['runtime_source']['path'],'U30_FULL_FEATURE_CPU_RESOURCE_CONTROL')
    no_torch();c=runtime.Control(spec)
    c.execution_spec_ref,c.execution_admission_ref=spec_ref,ref(admission_path)
    c.check();need(len(c.allowed)==len(spec['sources']),'No conflicting direct inputs')
    for value in spec['sources']:c.closed(value)
    need(c.psutil.Process().cpu_affinity()==[0,1]
         and all(os.environ.get(key)=='2' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
             'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Actual CPU01 and two-thread environment before Torch')
    parent=json.loads(Path(c.closed(spec['parent_ROOT_qualified'])).read_bytes())
    need(parent['qualified'] is True and parent['normal_RAW_count']==spec['normal_RAW_count']
         and parent['total_failed_RAW_count']==spec['total_failed_RAW_count']
         and parent['total_terminal_identity_pin_count']==spec['total_terminal_identity_pin_count']
         and parent['legacy_pin_count']==spec['legacy_pin_count'],'Current original Root-qualified terminal predecessor')
    pre=json.loads(Path(c.closed(spec['prelock'])).read_bytes())
    need(pre['schema']=='U30-full-feature-CPU-prelock-v1'
         and pre['synthetic_inputs_only'] is True and pre['expected_real_legacy_tests']==1
         and pre['expected_additional_tests']==1 and pre['input_size']==64
         and pre['batch_size']==1 and pre['query_count']==2 and pre['mode']=='eval'
         and all(pre[key] is True for key in ('no_new_training','no_AP','no_DEV','no_TEST')),
         'Fixed real-full size64 feature contract; not 400-size or efficacy evidence')
    output=Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink()
         and c.psutil.disk_usage(str(output.parent)).free>=1024**3,'Fresh owned size64 report output')
    no_torch()
    import torch
    need(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Pinned Torch, cold CPU-only execution')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    body=load(c.closed(spec['body_source']),'U30_FULL_FEATURE_CPU_PROBE_BODY')
    need(not torch.cuda.is_initialized(),'Body import preserves cold CUDA')
    result=body.run(torch,spec,c,pre);c.check()
    need(not torch.cuda.is_initialized() and result['all_tests_passed'] is True
         and result['real_legacy_tests']==1 and result['additional_tests']==1 and result['tests_run']==2
         and result['original_full_feature_modules_executed'] is True
         and all(result[key] is False for key in ('qualified','new_training','AP','DEV','TEST',
                                                 'original_semantic_head_executed','size400_verified','train_mode_BN_verified')),
         'Actual passing size64 feature contract; no 400-size/train/efficacy result')
    output_ref(c,result['report_ref'],output);output_ref(c,result['unittest_log_ref'],output)
    for role in ROLES:need(ref(spec[role]['path'])==spec[role],'Owned source unchanged')
    c.check()
    q=dict(schema='U30-owned-pending-full-feature-CPU-v1',
        status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',operation=spec['operation'],
        phase=spec['phase'],source=spec['entry_source'],body_source=spec['body_source'],
        adapter_source=spec['adapter_source'],test_source=spec['test_source'],runtime_source=spec['runtime_source'],
         legacy_source=spec['legacy_source'],dsconv_source=spec['dsconv_source'],head_source=spec['head_source'],
         feature_source=spec['feature_source'],
        spec=spec_ref,admission=c.execution_admission_ref,prelock=spec['prelock'],
        parent_ROOT_qualified=spec['parent_ROOT_qualified'],result=result,
        owner_index=spec['owner_index'],normal_RAW_count=spec['normal_RAW_count'],
        total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],legacy_pin_count=spec['legacy_pin_count'],
         synthetic_inputs_only=True,new_training=False,FIT=False,AP=False,DEV=False,TEST=False,
         original_full_feature_modules_executed=True,original_semantic_head_executed=False,
         actual_full_architecture_verified=False,size400_verified=False,train_mode_BN_verified=False,
         full_method_training_eligible=False,
        elapsed_seconds=time.monotonic()-c.started,peak_RSS_bytes=c.peak_rss,
        minimum_available_RAM_bytes=c.minimum_available,cleanup=dict(CUDA_initialized=False),
        qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False)
    qref=c.write_json(output/'Q.json',q)
    print(json.dumps(dict(status=q['status'],Q=qref,qualified=False,size64_feature_contract_only=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true');args=parser.parse_args()
    need(args.owner_execute,'Root owns the original single CPU contract leaf')
    run(args.spec)
