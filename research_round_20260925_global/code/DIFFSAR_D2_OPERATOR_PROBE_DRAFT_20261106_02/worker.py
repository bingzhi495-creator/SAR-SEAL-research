"""Fixed author D2 CUDA operator forward/backward probe; metadata-only import."""
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

SCHEMA='Root-D2-fixed-operator-probe-spec-v1'
OPERATION='CUDA_D2_FIXED_FP32_OPERATOR_PROBE_V01'
LEASE_SCHEMA='Root-D2-fixed-operator-probe-lease-v1'
LIBRARY_NAMES=('torch','detectron2','deform_conv','d2_extension','roi_align','nms',
    'torchvision','tv_roi_align','tv_boxes','tv_extension')
D2_EXTENSION_PIN=dict(bytes=20097880,sha256='8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af')
PRELOCK=dict(kernel_size=3,padding=1,dilation=1,groups=1,deformable_groups=1,bias=None,dtype='float32',
    cases=[dict(name='modulated_stride1',shape=[1,2,4,5],out_channels=3,stride=1),
           dict(name='modulated_stride2',shape=[1,2,5,7],out_channels=3,stride=2)],
    tolerances=dict(output=dict(atol=3e-5,rtol=3e-4),input_gradient=dict(atol=8e-5,rtol=8e-4),
        offset_gradient=dict(atol=1e-4,rtol=1e-3),mask_gradient=dict(atol=1e-4,rtol=1e-3),
        weight_gradient=dict(atol=8e-5,rtol=8e-4)),
    delegated_interfaces=dict(ROIAlignV2=dict(output_size=[7,7],spatial_scale=1.0,sampling_ratio=2,
        aligned=True,expected_constant_output=1.0,expected_input_gradient_sum=49.0,atol=1e-5,rtol=1e-5),
        axis_batched_NMS=dict(threshold=0.5,expected_keep=[0,2])),
    CPU_affinity=[0,1],threads=2,GPU_allocator_cap_bytes=1024**3,
    peak_group_RSS_cap_bytes=4*1024**3,free_RAM_floor_bytes=16*1024**3,
    start_RAM_bytes=48*1024**3,external_deadline_seconds=120,maximum_attempts=1,
    author_implementation='detectron2.layers.deform_conv.modulated_deform_conv',
    reference='PyTorch zero-padded bilinear grid_sample per kernel offset plus explicit mask/weight sum',
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
    run=spec['run_id'];need(type(run) is str and run.startswith('D2-fixed-FP32-v01-')
        and len('D2-fixed-FP32-v01-')<len(run)<=128 and all(c.isalnum() or c in '_-' for c in run),'Explicit run identity')
    need(json.dumps(spec['prelock'],sort_keys=True,allow_nan=False)==json.dumps(PRELOCK,sort_keys=True,allow_nan=False),
        'Fixed typed cases, precision, tolerances and resource contract')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])=={'worker'},'One own worker source')
    need(type(spec['library_sources']) is dict and set(spec['library_sources'])==set(LIBRARY_NAMES),'Ten selected library sources')
    need(all(spec['library_sources']['d2_extension'][key]==value for key,value in D2_EXTENSION_PIN.items()),'Root-accepted built D2 extension identity')
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


def reference_modulated(torch,features,offsets,masks,weight,stride,pad,dilation):
    """Independent bilinear sampler plus explicit modulation/weight contraction."""
    n,channels,height,width=features.shape;out_channels,_,kh,kw=weight.shape
    ho,wo=offsets.shape[-2:]
    oy=torch.arange(ho,dtype=features.dtype,device=features.device).view(1,ho,1)
    ox=torch.arange(wo,dtype=features.dtype,device=features.device).view(1,1,wo)
    sampled=[]
    for ky in range(kh):
        for kx in range(kw):
            index=ky*kw+kx
            y=oy*stride-pad+ky*dilation+offsets[:,2*index,:,:]
            x=ox*stride-pad+kx*dilation+offsets[:,2*index+1,:,:]
            grid=torch.stack((2*x/(width-1)-1,2*y/(height-1)-1),dim=-1)
            sampled.append(torch.nn.functional.grid_sample(features,grid,mode='bilinear',
                padding_mode='zeros',align_corners=True))
    patches=torch.stack(sampled,dim=2)*masks.unsqueeze(1)
    return (patches.unsqueeze(1)*weight.reshape(1,out_channels,channels,kh*kw,1,1)).sum(dim=(2,3))


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
    n,channels,height,width=case['shape'];o=case['out_channels'];k=PRELOCK['kernel_size']
    stride=case['stride'];pad=PRELOCK['padding'];dilation=PRELOCK['dilation']
    ho=(height+2*pad-dilation*(k-1)-1)//stride+1
    wo=(width+2*pad-dilation*(k-1)-1)//stride+1
    make=lambda count:torch.arange(count,device='cuda:0',dtype=torch.float32)
    features=((make(n*channels*height*width).remainder(19)-9)/13).reshape(n,channels,height,width)
    sequence=make(n*2*k*k*ho*wo)
    signs=torch.where(sequence.div(3,rounding_mode='floor').remainder(2)==0,1.0,-1.0)
    offsets=((.05+.021*(sequence.remainder(11)+1))*signs).reshape(n,2*k*k,ho,wo)
    masks=((make(n*k*k*ho*wo).remainder(17)+1)/19).reshape(n,k*k,ho,wo)
    weight=((make(o*channels*k*k).remainder(23)-11)/17).reshape(o,channels,k,k)
    tensors=[value.detach().contiguous().requires_grad_(True) for value in (features,offsets,masks,weight)]
    references=[value.detach().clone().requires_grad_(True) for value in tensors]
    need(all(value.is_leaf for value in tensors+references),'Independent leaves for all four first derivatives')
    check();progress['native_forward_entered']=True
    actual=official.modulated_deform_conv(*tensors,None,stride,pad,dilation,1,1)
    torch.cuda.synchronize(0);progress['native_forward_returned']=True
    need(tuple(actual.shape)==(n,o,ho,wo),'Actual author functional output shape')
    expected=reference_modulated(torch,*references,stride,pad,dilation)
    torch.cuda.synchronize(0);progress['reference_forward_returned']=True
    results={'output':comparison(torch,actual,expected,'output')}
    upstream=((make(actual.numel()).remainder(13)-6)/9).reshape(actual.shape).contiguous()
    gradients=torch.autograd.grad(actual,tensors,grad_outputs=upstream,retain_graph=False,create_graph=False)
    torch.cuda.synchronize(0);progress['native_backward_returned']=True
    reference_gradients=torch.autograd.grad(expected,references,grad_outputs=upstream,
        retain_graph=False,create_graph=False)
    torch.cuda.synchronize(0);progress['reference_backward_returned']=True
    for label,actual_gradient,expected_gradient in zip(
            ('input_gradient','offset_gradient','mask_gradient','weight_gradient'),gradients,reference_gradients):
        results[label]=comparison(torch,actual_gradient,expected_gradient,label)
    check()
    return dict(name=case['name'],input_shape=list(tensors[0].shape),offset_shape=list(tensors[1].shape),
        mask_shape=list(tensors[2].shape),weight_shape=list(tensors[3].shape),output_shape=list(actual.shape),
        stride=stride,padding=pad,dilation=dilation,kernel_size=k,groups=1,deformable_groups=1,bias=None,
        offset_order='interleaved_dy_dx_per_kernel_point',noninteger_offsets=True,
        mask_is_independent_leaf_without_sigmoid_chain=True,dtype='float32',checks=results)


def normal_interfaces(torch,modules,check):
    """Analytic connection checks; these wrappers use the same Torchvision backend."""
    check();config=PRELOCK['delegated_interfaces']['ROIAlignV2']
    image=torch.ones((1,1,5,7),device='cuda:0',dtype=torch.float32,requires_grad=True)
    rois=torch.tensor([[0,1,1,5,4]],device='cuda:0',dtype=torch.float32)
    align=modules['roi_align'].ROIAlign(tuple(config['output_size']),config['spatial_scale'],
        config['sampling_ratio'],aligned=config['aligned'])
    pooled=align(image,rois);torch.cuda.synchronize(0)
    need(tuple(pooled.shape)==(1,1,7,7) and pooled.dtype==torch.float32 and pooled.is_cuda,'ROIAlign normal CUDA interface')
    torch.testing.assert_close(pooled,torch.ones_like(pooled),atol=config['atol'],rtol=config['rtol'])
    grad=torch.autograd.grad(pooled.sum(),image)[0];torch.cuda.synchronize(0)
    need(bool(torch.isfinite(grad).all()) and abs(float(grad.sum().item())-49.0)
         <=config['atol']+config['rtol']*49.0,'Analytic interior constant ROIAlign derivative')
    roi_result=dict(status='ANALYTIC_NORMAL_INTERFACE_PASSED',backend='torchvision',
        independent_implementation_comparison=False,output_shape=list(pooled.shape),
        max_abs_constant_output_error=float((pooled-1).abs().max().item()),input_gradient_sum=float(grad.sum().item()))
    boxes=torch.tensor([[0,0,2,2],[0,0,2,2],[5,5,7,7]],device='cuda:0',dtype=torch.float32)
    scores=torch.tensor([.9,.8,.7],device='cuda:0',dtype=torch.float32)
    classes=torch.zeros(3,device='cuda:0',dtype=torch.int64)
    keep=modules['nms'].batched_nms(boxes,scores,classes,PRELOCK['delegated_interfaces']['axis_batched_NMS']['threshold'])
    torch.cuda.synchronize(0);need(keep.is_cuda and keep.tolist()==[0,2],'Analytic duplicate/far-box NMS keep set')
    check()
    return dict(ROIAlignV2=roi_result,axis_batched_NMS=dict(status='ANALYTIC_NORMAL_INTERFACE_PASSED',
        backend='torchvision',independent_implementation_comparison=False,kept_indices=keep.tolist(),
        NMS_gradient_requested=False))




def clear_blas_workspaces(torch,report):
    clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    need(callable(clear),'Locked Torch2.10 cuBLAS workspace cleanup API required')
    report['cuBLAS_workspace_cleanup_entered']=True
    clear()
    report['cuBLAS_workspaces_cleared']=True


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
    report=dict(schema='D2-fixed-operator-probe-report-v1',operation=OPERATION,run_id=spec['run_id'],
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
        check();need(available_ram()>=PRELOCK['start_RAM_bytes'],'48 GiB startup admission')
        refs=list(spec['source_refs'].values())+list(spec['library_sources'].values())+[spec['root_lease_ref']]
        for value in refs:closed(value)
        need(reference(__file__)==spec['source_refs']['worker'],'Actual selected own worker')
        lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());validate_lease(spec,lease)
        if 'boot_id' in lease:need(lease['boot_id']==report['boot_id'],'Lease boot identity')
        modules={}
        for role,name in (('torch','torch'),('torchvision','torchvision'),('detectron2','detectron2'),
                ('deform_conv','detectron2.layers.deform_conv'),('d2_extension','detectron2._C'),
                ('roi_align','detectron2.layers.roi_align'),('nms','detectron2.layers.nms'),
                ('tv_roi_align','torchvision.ops.roi_align'),('tv_boxes','torchvision.ops.boxes')):
            module=importlib.import_module(name)
            need(reference(module.__file__)==spec['library_sources'][role],'Actual imported selected module: '+role)
            modules[role]=module;check()
        torch=modules['torch'];official=modules['deform_conv']
        need(str(torch.__version__)=='2.10.0+cu130' and str(torch.version.cuda)=='13.0','Locked Torch/CUDA build')
        loaded={str(Path(path).resolve()) for path in torch.ops.loaded_libraries}
        need(spec['library_sources']['tv_extension']['path'] in loaded,'Selected Torchvision extension is actually loaded')
        closed(spec['library_sources']['tv_extension'])
        need(official._ModulatedDeformConv.__module__=='detectron2.layers.deform_conv','Selected author modulated functional')
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_grad_enabled(True)
        need(torch.cuda.is_available(),'CUDA required; no fallback')
        props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,PRELOCK['GPU_allocator_cap_bytes']/props.total_memory),device=0)
        torch.cuda.reset_peak_memory_stats(0)
        report.update(actual_environment=dict(python=platform.python_version(),machine=platform.machine(),
            Torch=str(torch.__version__),CUDA_build=str(torch.version.cuda),Torchvision=str(modules['torchvision'].__version__),Detectron2=str(modules['detectron2'].__version__),
            CUDA_device_name=props.name,CUDA_compute_capability=list(torch.cuda.get_device_capability(0)),
            torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32),
            actual_library_refs=dict({k:reference(v.__file__) for k,v in modules.items()},
                tv_extension=spec['library_sources']['tv_extension']))
        for case in PRELOCK['cases']:
            progress=dict(name=case['name'],native_forward_entered=False,native_forward_returned=False,
                reference_forward_returned=False,native_backward_returned=False,reference_backward_returned=False)
            report['case_progress'].append(progress);save()
            report['cases'].append(one_case(torch,official,case,progress,check));save()
        report['delegated_interface_checks']=normal_interfaces(torch,modules,check);save()
        for value in refs:closed(value)
        check();need(len(report['cases'])==2,'Both fixed modulated DCN cases complete')
        report.update(status='TWO_FIXED_D2_MODULATED_DEFORM_CONV_CASES_PASSED_PENDING_ROOT',
            selected_sources_unchanged_after=True,all_outputs_and_four_first_gradients_within_fixed_tolerances=True)
        code=0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc())
    finally:
        try:
            if torch is not None and torch.cuda.is_initialized():
                gc.collect();torch.cuda.synchronize(0)
                clear_blas_workspaces(torch,report)
                torch.cuda.empty_cache()
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

