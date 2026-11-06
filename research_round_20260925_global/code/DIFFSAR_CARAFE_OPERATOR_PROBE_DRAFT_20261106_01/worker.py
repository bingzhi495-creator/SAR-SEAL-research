"""Fixed synthetic CARAFE CUDA forward/backward probe; metadata-only import."""
import argparse
from copy import deepcopy
import datetime
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback

SCHEMA='Root-CARAFE-fixed-operator-probe-spec-v1'
OPERATION='CUDA_CARAFE_FIXED_FP32_FORWARD_BACKWARD_V01'
LEASE_SCHEMA='Root-CARAFE-fixed-operator-probe-lease-v1'
LIBRARY_NAMES=('torch','mmcv','carafe','mmcv_extension')
PRELOCK=dict(kernel_size=5,group_size=1,scale_factor=2,dtype='float32',
    cases=[dict(name='non_square_signed',shape=[1,4,3,5]),
           dict(name='boundary_dominated',shape=[1,4,2,3])],
    tolerances=dict(output=dict(atol=2e-5,rtol=2e-4),
        feature_gradient=dict(atol=5e-5,rtol=5e-4),mask_gradient=dict(atol=5e-5,rtol=5e-4)),
    CPU_affinity=[0,1],threads=2,GPU_allocator_cap_bytes=1024**3,
    peak_group_RSS_cap_bytes=4*1024**3,free_RAM_floor_bytes=16*1024**3,
    external_deadline_seconds=120,maximum_attempts=1,
    official_implementation='mmcv.ops.carafe.CARAFEFunction.apply',
    reference='torch unfold zero padding; repeat feature H/W only; groupwise weighted sum',
    GT_or_model_or_dataset_read=False,AP=False)


def need(condition,message):
    if not condition:raise ValueError(message)


def validate_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'},'Exact byte reference required')
    path=value['path'];need(type(path) is str,'String path')
    p=PurePosixPath(path)
    need(p.is_absolute() and str(p)==path and '..' not in p.parts,'Absolute canonical POSIX path')
    need(type(value['bytes']) is int and value['bytes']>0,'Positive integer byte count')
    sha=value['sha256']
    need(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef'),'SHA256 required')


def validate_spec(spec):
    need(type(spec) is dict and set(spec)=={'schema','operation','run_id','source_refs',
        'library_sources','root_lease_ref','output_directory','prelock'},'Exact fixed spec fields')
    need(spec['schema']==SCHEMA and spec['operation']==OPERATION,'Fixed operator schema/operation')
    run=spec['run_id'];need(type(run) is str and run.startswith('CARAFE-fixed-FP32-v01-')
        and 24<=len(run)<=128 and all(c.isalnum() or c in '_-' for c in run),'Explicit run identity')
    need(json.dumps(spec['prelock'],sort_keys=True,allow_nan=False)==json.dumps(PRELOCK,sort_keys=True,allow_nan=False),
        'Fixed typed cases, precision, tolerances and resource contract')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])=={'worker'},'One own worker source')
    need(type(spec['library_sources']) is dict and set(spec['library_sources'])==set(LIBRARY_NAMES),'Four selected library sources')
    refs=list(spec['source_refs'].values())+list(spec['library_sources'].values())+[spec['root_lease_ref']]
    for value in refs:validate_ref(value)
    need(len({x['path'] for x in refs})==len(refs),'Distinct selected source/lease paths')
    out=PurePosixPath(spec['output_directory'])
    need(out.is_absolute() and str(out)==spec['output_directory'] and '..' not in out.parts,'Explicit output directory')
    need(all(out not in PurePosixPath(x['path']).parents and str(out)!=x['path'] for x in refs),'Output cannot overlap input')


def validate_lease(spec,lease):
    need(type(lease) is dict and lease.get('schema')==LEASE_SCHEMA
         and lease.get('status')=='ROOT_GRANTED','Explicit Root lease')
    need(lease.get('operation')==spec['operation'] and lease.get('run_id')==spec['run_id']
         and lease.get('worker_ref')==spec['source_refs']['worker']
         and lease.get('prelock')==spec['prelock'] and lease.get('output_directory')==spec['output_directory'],
         'Lease joins actual selected worker, run, fixed cases and output')
    need(all(lease.get(k) is True for k in ('single_attempt','main_scientific_processes_reaped',
        'exclusive_project_scientific_leaf')) and lease.get('whole_device_exclusive') is False,
        'Single externally supervised project scientific leaf; shared device')


def reference(path):
    p=Path(path).resolve(strict=True);before=p.stat();digest=hashlib.sha256();size=0
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block);size+=len(block)
    after=p.stat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(before)==identity(after),'Selected file changed while hashing')
    return dict(path=str(p),bytes=size,sha256=digest.hexdigest())


def closed(ref):
    need(reference(ref['path'])==ref,'Selected source/input bytes differ')
    return Path(ref['path'])


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise ValueError('MemAvailable unavailable')


def reference_carafe(torch,features,masks,kernel,group,scale):
    """Explicit zero-padded local reassembly; no official/native operator use."""
    n,channels,height,width=features.shape
    patches=torch.nn.functional.unfold(features,kernel_size=kernel,padding=(kernel-1)//2)
    patches=patches.view(n,group,channels//group,kernel*kernel,height,width)
    patches=patches.repeat_interleave(scale,dim=-2).repeat_interleave(scale,dim=-1)
    weights=masks.view(n,group,1,kernel*kernel,height*scale,width*scale)
    return (patches*weights).sum(dim=3).reshape(n,channels,height*scale,width*scale)


def comparison(torch,actual,expected,kind):
    need(actual.shape==expected.shape and actual.dtype==expected.dtype==torch.float32
         and actual.device==expected.device==torch.device('cuda:0'),'FP32 CUDA comparison shape/device')
    need(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),'Finite values and gradients')
    tolerance=PRELOCK['tolerances'][kind]
    difference=(actual.detach()-expected.detach()).abs()
    allowed=tolerance['atol']+tolerance['rtol']*expected.detach().abs()
    metrics=dict(shape=list(actual.shape),max_abs_error=float(difference.max().item()),
        max_error_over_allowed=float((difference/allowed).max().item()),
        atol=tolerance['atol'],rtol=tolerance['rtol'])
    need(bool((difference<=allowed).all()),'Fixed tolerance failed: '+kind+' '+json.dumps(metrics))
    metrics['passed']=True
    return metrics


def one_case(torch,official,case,progress,check):
    n,channels,height,width=case['shape'];k=PRELOCK['kernel_size'];g=PRELOCK['group_size'];s=PRELOCK['scale_factor']
    device=torch.device('cuda:0')
    x=torch.arange(n*channels*height*width,device=device,dtype=torch.float32).reshape(n,channels,height,width)
    if case['name']=='non_square_signed':x=(x.remainder(17)-8)/7
    else:
        x=torch.zeros_like(x)
        for channel in range(channels):
            x[:,channel,0,0]=(channel+1)/3
            x[:,channel,-1,-1]=-(channel+2)/5
    count=n*g*k*k*height*s*width*s
    weights=torch.arange(count,device=device,dtype=torch.float32).reshape(n,g*k*k,height*s,width*s)
    if case['name']=='non_square_signed':weights=(weights.remainder(37)-18).div(11).softmax(dim=1)
    else:
        weights=weights.remainder(11)+1
        weights[:,0,:,:]+=37;weights[:,-1,:,:]+=19
        weights=weights/weights.sum(dim=1,keepdim=True)
    features=x.detach().contiguous().requires_grad_(True)
    masks=weights.detach().contiguous().requires_grad_(True)
    expected_features=features.detach().clone().requires_grad_(True)
    expected_masks=masks.detach().clone().requires_grad_(True)
    need(masks.is_leaf and expected_masks.is_leaf,'Mask derivatives are with respect to detached mask leaves')
    need(tuple(masks.shape)==(n,g*k*k,height*s,width*s),'Actual CARAFE mask layout')
    need(bool(torch.isfinite(masks).all()) and bool((masks>=0).all()),'Finite nonnegative fixed synthetic masks')
    check()
    progress['native_forward_entered']=True
    actual=official.CARAFEFunction.apply(features,masks,k,g,s)
    torch.cuda.synchronize(0);progress['native_forward_returned']=True
    need(tuple(actual.shape)==(n,channels,height*s,width*s),'Official output shape')
    expected=reference_carafe(torch,expected_features,expected_masks,k,g,s)
    torch.cuda.synchronize(0);progress['reference_forward_returned']=True
    results={'output':comparison(torch,actual,expected,'output')}
    upstream=torch.arange(actual.numel(),device=device,dtype=torch.float32).reshape(actual.shape)
    upstream=((upstream.remainder(19)-9)/7).contiguous()
    gradients=torch.autograd.grad(actual,(features,masks),grad_outputs=upstream,
        retain_graph=False,create_graph=False)
    torch.cuda.synchronize(0);progress['native_backward_returned']=True
    expected_gradients=torch.autograd.grad(expected,(expected_features,expected_masks),
        grad_outputs=upstream,retain_graph=False,create_graph=False)
    torch.cuda.synchronize(0);progress['reference_backward_returned']=True
    for label,actual_gradient,expected_gradient in zip(('feature_gradient','mask_gradient'),gradients,expected_gradients):
        results[label]=comparison(torch,actual_gradient,expected_gradient,label)
    check()
    return dict(name=case['name'],input_shape=list(features.shape),mask_shape=list(masks.shape),
        output_shape=list(actual.shape),kernel_size=k,group_size=g,scale_factor=s,dtype='float32',
        masks_normalized_before_detached_leaf=True,gradients_wrt_mask_leaf=True,
        caller_supplied_nonconstant_signed_output_gradient=True,checks=results)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True)
    args=parser.parse_args(argv)
    raw=Path(args.spec).read_bytes()
    need(hashlib.sha256(raw).hexdigest()==args.spec_sha,'Exact selected spec SHA')
    spec=json.loads(raw);validate_spec(spec)
    need(platform.system()=='Linux' and platform.machine()=='aarch64'
         and platform.python_version()=='3.12.14','Explicit DGX runtime')
    import resource
    os.sched_setaffinity(0,{0,1});os.nice(10)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    need(not any(k.split('.')[0] in ('torch','numpy','mmcv') for k in sys.modules),'Fresh scientific interpreter')
    out=Path(spec['output_directory']);need(out.parent.is_dir() and not out.exists(),'Fresh explicit output directory')
    out.mkdir(mode=0o700)
    stat=Path('/proc/self/stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    report=dict(schema='CARAFE-fixed-operator-probe-report-v1',operation=OPERATION,run_id=spec['run_id'],
        status='STARTED',qualified=False,worker_exit_code=None,Root_external_admission_required=True,
        started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),pgid=os.getpgrp(),
        start_ticks=int(tail[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        prelock=deepcopy(PRELOCK),cases=[],case_progress=[],new_training=False,formal_training_credit=0,
        GT_or_model_or_dataset_read=False,AP=False,TGRS_ready=False,
        whole_GPU_memory_contract_certified=False,whole_device_exclusive=False,
        source_refs=deepcopy(spec['source_refs']),library_sources=deepcopy(spec['library_sources']),
        external_deadline_and_group_RSS_and_birth_reaping_required=True)
    def save():
        path=out/'REPORT.next'
        with path.open('w',encoding='utf-8') as stream:
            json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(str(path),str(out/'REPORT.json'))
    save();started=time.monotonic();torch=None;code=1;minimum_ram=None
    def check():
        nonlocal minimum_ram
        available=available_ram();minimum_ram=available if minimum_ram is None else min(minimum_ram,available)
        need(available>=PRELOCK['free_RAM_floor_bytes'],'Free RAM floor')
        need(time.monotonic()-started<PRELOCK['external_deadline_seconds'],'Observed worker wall deadline')
        need(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024<=PRELOCK['peak_group_RSS_cap_bytes'],'Worker peak RSS bound')
        need(sorted(os.sched_getaffinity(0))==[0,1],'CPU affinity')
    try:
        check()
        refs=list(spec['source_refs'].values())+list(spec['library_sources'].values())+[spec['root_lease_ref']]
        for value in refs:closed(value)
        need(reference(__file__)==spec['source_refs']['worker'],'Actual selected own worker')
        lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());validate_lease(spec,lease)
        if 'boot_id' in lease:need(lease['boot_id']==report['boot_id'],'Lease boot identity')
        modules={}
        for role,name in (('torch','torch'),('mmcv','mmcv'),('carafe','mmcv.ops.carafe'),('mmcv_extension','mmcv._ext')):
            module=importlib.import_module(name)
            need(reference(module.__file__)==spec['library_sources'][role],'Actual imported selected module: '+role)
            modules[role]=module;check()
        torch=modules['torch'];official=modules['carafe'];mmcv=modules['mmcv']
        need(str(torch.__version__)=='2.10.0+cu130' and str(torch.version.cuda)=='13.0','Locked Torch/CUDA build')
        need(str(mmcv.__version__)=='2.1.0' and importlib.metadata.version('mmcv')=='2.1.0','Locked installed MMCV2.1.0')
        need(official.CARAFEFunction.__module__=='mmcv.ops.carafe','Official CARAFEFunction')
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_grad_enabled(True)
        need(torch.cuda.is_available(),'CUDA required; no fallback')
        props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,PRELOCK['GPU_allocator_cap_bytes']/props.total_memory),device=0)
        torch.cuda.reset_peak_memory_stats(0)
        report.update(actual_environment=dict(python=platform.python_version(),machine=platform.machine(),
            Torch=str(torch.__version__),CUDA_build=str(torch.version.cuda),MMCV=str(mmcv.__version__),
            CUDA_device_name=props.name,CUDA_compute_capability=list(torch.cuda.get_device_capability(0)),
            torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32),
            actual_library_refs={k:reference(v.__file__) for k,v in modules.items()})
        for case in PRELOCK['cases']:
            progress=dict(name=case['name'],native_forward_entered=False,native_forward_returned=False,
                reference_forward_returned=False,native_backward_returned=False,reference_backward_returned=False)
            report['case_progress'].append(progress);save()
            report['cases'].append(one_case(torch,official,case,progress,check));save()
        for value in refs:closed(value)
        check();need(len(report['cases'])==2,'Both fixed cases complete')
        report.update(status='TWO_FIXED_CARAFE_FP32_CUDA_CASES_PASSED_PENDING_ROOT',
            selected_sources_unchanged_after=True,all_outputs_and_both_first_gradients_within_fixed_tolerances=True)
        code=0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc())
    finally:
        try:
            if torch is not None and torch.cuda.is_initialized():
                gc.collect();torch.cuda.synchronize(0);torch.cuda.empty_cache()
                allocated=int(torch.cuda.memory_allocated(0));reserved=int(torch.cuda.memory_reserved(0))
                peak=int(torch.cuda.max_memory_allocated(0))
                report.update(CUDA_synchronized=True,CUDA_allocated_after_cleanup=allocated,
                    CUDA_reserved_after_cleanup=reserved,CUDA_peak_allocated_bytes=peak)
                need(allocated==reserved==0 and peak<=PRELOCK['GPU_allocator_cap_bytes'],'CUDA cleanup and allocator peak')
            check()
        except BaseException as error:
            code=1;report.update(status='FAILED_RESOURCE_OR_CLEANUP_PRESERVED',cleanup_error=str(error))
        report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started,minimum_observed_available_RAM_bytes=minimum_ram,
            worker_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
        save()
        print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','elapsed_seconds','AP','formal_training_credit')}),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())

