"""UNBOUND CPU model-load probe; imports no scientific library at module import."""
import json
import re
import argparse
from copy import deepcopy
import datetime
import gc
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types
SCHEMA='Root-R101-DiffSAR-CPU-load-probe-spec-v1'
OPERATION='CPU_DIFFSAR_R101_EXPLICIT_BACKBONE_LOAD_PROBE_V01'
LEASE_SCHEMA='Root-R101-DiffSAR-CPU-load-probe-lease-v1'
LIBRARY_MODULES=dict(torch='torch',numpy='numpy',detectron2='detectron2',diffsarshipinst='diffsarshipinst',config='diffsarshipinst.config',model_ema='diffsarshipinst.util.model_ema',d2_config='detectron2.config',train_net='train_net',detector='diffsarshipinst.detector',head='diffsarshipinst.head',fpn='detectron2.modeling.backbone.fpn',resnet='detectron2.modeling.backbone.resnet',frozenbn='detectron2.layers.batch_norm',deform_conv='detectron2.layers.deform_conv',d2_modeling='detectron2.modeling',d2_build='detectron2.modeling.meta_arch.build',backbone_init='detectron2.modeling.backbone',d2_extension='detectron2._C',carafe='mmcv.ops.carafe',mmcv_extension='mmcv._ext',torchvision='torchvision')
LIBRARY_NAMES=tuple(LIBRARY_MODULES)+('tv_extension',)
CONFIG_PIN=dict(bytes=7065,sha256='48d6e3a67a19a737ce22d0659db02da0a878dbde9a5483e5c6ed240fbc87fddb')
ADAPTER_PIN=dict(bytes=2220,sha256='31bb4935787510af70f96f098b60a6c9eb71472ad6f4760bf20bbaa01d250619')
COVERAGE_PIN=dict(bytes=8520,sha256='a5f524cd621aaf3c094043e9053f926448706c2781a7613d77e9d500abd1875d')
ACTUAL_LOAD_PIN=dict(bytes=6489,sha256='378d7f23aeba77d73d511bc90962058a39c165e4da3eed58582bd65877ccd89a')
CHECKPOINT_PIN=dict(bytes=178793939,sha256='63fe2227b86e8f1f2063f43a75c84d195911b6a0eace650907dd3dc62dd49a0a')
D2_EXTENSION_PIN=dict(bytes=20097880,sha256='8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af')
PRELOCK=dict(seed=40244023,device='cpu',CPU_affinity=[0,1],threads=2,deadline_seconds=300,maximum_group_RSS_bytes=16*1024**3,start_RAM_bytes=48*1024**3,minimum_available_RAM_bytes=16*1024**3,weights_only=True,proposals=500,heads=6,num_classes=1,single_attempt=True,automatic_retry=False,forward=False,new_training=False,AP=False)

def need(ok,message):
    if not ok:raise ValueError(message)

def signature(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def parse_json(text):
    need(type(text) is str and len(text.encode('utf-8'))<=8*1024**2,'Bounded JSON metadata')
    def pairs(rows):
        out={}
        for k,v in rows:need(k not in out,'Duplicate JSON key: '+k);out[k]=v
        return out
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)

def valid_ref(ref):
    need(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'Exact reference')
    path=ref['path'];need(type(path) is str and path.startswith('/') and str(PurePosixPath(path))==path and path!='/' and '..' not in PurePosixPath(path).parts,'Canonical POSIX reference')
    need(type(ref['bytes']) is int and ref['bytes']>0,'Positive integer reference size')
    need(type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'Explicit SHA256')

def spec_refs(spec):
    return list(spec['source_refs'].values())+list(spec['library_sources'].values())+[spec[k] for k in ('config_ref','checkpoint_ref','root_lease_ref')]

def validate_spec(spec):
    need(type(spec) is dict and set(spec)=={'schema','operation','run_id','source_refs','library_sources','config_ref','checkpoint_ref','root_lease_ref','output_directory','prelock'},'Exact CPU load-only fields')
    need(spec['schema']==SCHEMA and spec['operation']==OPERATION,'Explicit CPU load-only operation')
    need(type(spec['prelock']) is dict and signature(spec['prelock'])==signature(PRELOCK),'Exact load-only prelock')
    need(type(spec['run_id']) is str and re.fullmatch(r'DiffSAR-R101-CPU-load-v01-[A-Za-z0-9_]{1,80}',spec['run_id']) is not None,'Separate run identity')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])=={'worker','config_adapter','actual_load','coverage'},'Explicit worker/helper sources')
    need(type(spec['library_sources']) is dict and set(spec['library_sources'])==set(LIBRARY_NAMES),'Selected runtime sources')
    refs=spec_refs(spec)
    for ref in refs:valid_ref(ref)
    need(len({r['path'] for r in refs})==len(refs),'Distinct references')
    for ref,pin in ((spec['config_ref'],CONFIG_PIN),(spec['checkpoint_ref'],CHECKPOINT_PIN),(spec['source_refs']['config_adapter'],ADAPTER_PIN),(spec['source_refs']['coverage'],COVERAGE_PIN),(spec['source_refs']['actual_load'],ACTUAL_LOAD_PIN),(spec['library_sources']['d2_extension'],D2_EXTENSION_PIN)):
        need(all(ref[k]==v for k,v in pin.items()),'Fixed selected byte pin')
    out=spec['output_directory'];need(type(out) is str and out.startswith('/') and str(PurePosixPath(out))==out and out!='/' and '..' not in PurePosixPath(out).parts,'Canonical separate output')
    need(all(out!=r['path'] and not r['path'].startswith(out+'/') for r in refs),'Output outside all sources/inputs')

def validate_lease(spec,lease):
    binds=('operation','run_id','prelock','source_refs','library_sources','config_ref','checkpoint_ref','output_directory')
    need(type(lease) is dict and set(lease)==set(binds)|{'schema','status','boot_id','single_attempt','main_scientific_processes_reaped','exclusive_project_scientific_leaf','whole_device_exclusive','main_accepted_prefix','worker_ref','prior_scientific_groups','main_acceptance_ref'},'Exact Root lease fields')
    need(lease['schema']==LEASE_SCHEMA and lease['status']=='ROOT_GRANTED','Explicit Root grant')
    for key in binds:need(signature(lease[key])==signature(spec[key]),'Root lease binding: '+key)
    for key in ('single_attempt','main_scientific_processes_reaped','exclusive_project_scientific_leaf'):need(lease[key] is True,'Root exclusive closed-main lease: '+key)
    need(lease['whole_device_exclusive'] is False,'Project-scoped scientific lease only')
    need(signature(lease['worker_ref'])==signature(spec['source_refs']['worker']),'Parent/worker lease identity')
    valid_ref(lease['main_acceptance_ref'])
    groups=lease['prior_scientific_groups'];need(type(groups) is list and groups and all(type(g) is dict and type(g.get('pid')) is int and g['pid']>0 for g in groups),'Explicit prior group identities for external parent observations')
    prefix=lease['main_accepted_prefix'];need(type(prefix) is int and 31744<=prefix<=37050 and (prefix%256==0 or prefix==37050),'Actual accepted main boundary')
    need(type(lease['boot_id']) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',lease['boot_id']) is not None,'Explicit original boot')

def configuration(modules,adapter,spec):
    cfg=modules['d2_config'].get_cfg();modules['config'].add_DiffSARShipInst_config(cfg)
    modules['model_ema'].add_model_ema_configs(cfg);adapter.register_inactive_bifpn_schema(cfg)
    cfg.merge_from_file(spec['config_ref']['path']);adapter.verify_inactive_bifpn_schema(cfg)
    r=cfg.MODEL.RESNETS;d=cfg.MODEL.DiffSARShipInst
    need(cfg.MODEL.META_ARCHITECTURE=='DiffSARShipInst' and cfg.MODEL.BACKBONE.NAME=='build_resnet_scje_fpn_backbone','Original meta architecture and SCJE route')
    need(r.DEPTH==101 and r.DEFORM_MODULATED is True and list(r.DEFORM_ON_PER_STAGE)==[True]*4 and r.DEFORM_NUM_GROUPS==r.NUM_GROUPS==1 and r.STRIDE_IN_1X1 is False and r.RES5_DILATION==1 and r.NORM=='FrozenBN','Original R101/modulated groups1/FrozenBN/stride layout')
    need(d.NUM_PROPOSALS==500 and d.NUM_HEADS==6 and d.SAMPLE_STEP==1 and d.NUM_CLASSES==1 and d.DEEP_SUPERVISION and not d.USE_FED_LOSS and not cfg.MODEL_EMA.ENABLED,'Original DiffSAR heads and proposal count')
    need(cfg.INPUT.FORMAT=='RGB' and list(cfg.MODEL.PIXEL_MEAN)==[123.675,116.28,103.53] and list(cfg.MODEL.PIXEL_STD)==[58.395,57.12,57.375],'Original RGB normalization configuration')
    need(cfg.MODEL.WEIGHTS in (None,'') and cfg.MODEL.DEVICE=='cuda','Original configuration requests no initial checkpoint')
    cfg.MODEL.DEVICE='cpu';cfg.SEED=PRELOCK['seed'];cfg.DATALOADER.NUM_WORKERS=0
    cfg.freeze();return cfg

DIFFUSION_BUFFERS=('betas','alphas_cumprod','alphas_cumprod_prev','sqrt_alphas_cumprod','sqrt_one_minus_alphas_cumprod','log_one_minus_alphas_cumprod','sqrt_recip_alphas_cumprod','sqrt_recipm1_alphas_cumprod','posterior_variance','posterior_log_variance_clipped','posterior_mean_coef1','posterior_mean_coef2')

def spatial_pair(value):
    """Author ModulatedDeformConv keeps int stride/dilation; Conv2d uses pairs."""
    if type(value) is int:
        need(value>0,'Positive native spatial scalar');return (value,value)
    need(type(value) is tuple and len(value)==2 and all(type(v) is int and v>0 for v in value),'Explicit native spatial pair')
    return value

def model_properties(torch,model,modules,prior_normalizer=None):
    """Actual objects, without calling model/normalizer forward or a data mapper."""
    need(type(model) is modules['detector'].DiffSARShipInst,'Actual author model class')
    need(model.device.type=='cpu' and model.num_classes==1 and model.num_proposals==500 and model.num_heads==6 and len(model.head.head_series)==6 and model.num_timesteps==1000 and model.sampling_timesteps==1,'Actual complete author model attributes')
    state=model.state_dict();need(state and all(type(t) is torch.Tensor and t.device.type=='cpu' for t in state.values()),'Actual model state is ordinary CPU tensors')
    need(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'CPU FP32 model parameters without .float conversion')
    buffers=dict(model.named_buffers())
    for name in DIFFUSION_BUFFERS:
        t=buffers[name];need(t.device.type=='cpu' and t.dtype==torch.float64 and tuple(t.shape)==(1000,),'Original FP64 diffusion schedule: '+name)
    bottom=model.backbone.bottom_up;need(type(bottom) is modules['resnet'].ResNet and list(bottom.stage_names)==['res2','res3','res4','res5'],'Actual full R101 bottom_up')
    frozen=modules['frozenbn'].FrozenBatchNorm2d;norms=0;offsets=[]
    def norm(m):
        nonlocal norms
        need(type(m) is frozen,'Actual FrozenBN class')
        for key in ('weight','bias','running_mean','running_var'):
            t=getattr(m,key);need(type(t) is torch.Tensor and t.device.type=='cpu' and t.dtype==torch.float32,'Actual four-component FrozenBN state')
        norms+=1
    stem=bottom.stem.conv1;need(tuple(stem.stride)==(2,2) and tuple(stem.kernel_size)==(7,7) and tuple(stem.padding)==(3,3),'Actual stem convolution')
    norm(stem.norm)
    for stage,count in zip(range(2,6),(3,4,23,3)):
        blocks=getattr(bottom,'res%d'%stage);need(len(blocks)==count,'Actual complete R101 stage count')
        for i,block in enumerate(blocks):
            stride=2 if stage>2 and i==0 else 1
            need(type(block) is modules['resnet'].DeformBottleneckBlock and block.deform_modulated is True,'Actual modulated deformable block')
            need(tuple(block.conv1.stride)==(1,1) and tuple(block.conv3.stride)==(1,1),'Actual stride outside conv1')
            c=block.conv2;need(type(c) is modules['deform_conv'].ModulatedDeformConv and spatial_pair(c.stride)==(stride,stride) and spatial_pair(c.dilation)==(1,1) and c.groups==c.deformable_groups==1,'Actual conv2 DCN/stride/groups')
            o=block.conv2_offset;need(tuple(o.stride)==(stride,stride) and tuple(o.weight.shape)==(27,64*2**(stage-2),3,3) and tuple(o.bias.shape)==(27,),'Actual offset predictor layout')
            offsets.extend([o.weight,o.bias])
            for conv in (block.conv1,block.conv2,block.conv3):norm(conv.norm)
            if i==0:need(block.shortcut is not None,'Actual first-block projection');norm(block.shortcut.norm)
            else:need(block.shortcut is None,'No extra shortcut')
    need(norms==104 and len(offsets)==66,'Actual FrozenBN/offset counts')
    need(all(torch.count_nonzero(t).item()==0 for t in offsets),'Actual offset/mask predictors remain construction zero')
    fn=model.normalizer;need(isinstance(fn,types.FunctionType) and fn.__closure__ is not None,'Actual author normalization closure')
    closure=dict(zip(fn.__code__.co_freevars,(c.cell_contents for c in fn.__closure__)))
    need(set(closure)=={'pixel_mean','pixel_std'},'Actual normalizer captured tensor names')
    clones={}
    for key,values in [('pixel_mean',[123.675,116.28,103.53]),('pixel_std',[58.395,57.12,57.375])]:
        t=closure[key];need(type(t) is torch.Tensor and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==(3,1,1),'Actual normalizer closure tensor')
        expected=torch.tensor(values,dtype=torch.float32).reshape(3,1,1);need(torch.equal(t,expected),'Actual normalizer tensor values')
        if prior_normalizer is not None:
            need(fn is prior_normalizer['function'] and t.dtype==prior_normalizer[key].dtype and torch.equal(t,prior_normalizer[key]),'Normalizer closure unchanged after loading')
        clones[key]=t.detach().clone()
    clones['function']=fn
    return dict(actual_model_class=True,actual_model_CPU=True,actual_parameters_FP32=True,actual_diffusion_FP64_buffers=len(DIFFUSION_BUFFERS),actual_full_R101_layout=True,actual_FrozenBN_modules=norms,actual_offset_parameter_count=len(offsets),offset_values_construction_zero=True,normalizer_closure_values_checked=True,normalizer_closure_unchanged_checked=prior_normalizer is not None,actual_image_pipeline_or_no_double_normalization_verified=False,forward=False),clones

def reference(path):
    p=Path(path);need(not p.is_symlink() and p.resolve(strict=True)==p and p.is_file(),'Canonical actual regular file')
    before=p.stat();h=hashlib.sha256();size=0
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):size+=len(block);h.update(block)
    after=p.stat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(before)==identity(after) and size==before.st_size,'Stable complete source bytes')
    return dict(path=str(p),bytes=size,sha256=h.hexdigest())

def verify(ref):
    valid_ref(ref);need(reference(ref['path'])==ref,'Actual byte-reference mismatch')

def source_module(ref,name):
    verify(ref);m=types.ModuleType(name);m.__file__=ref['path'];sys.modules[name]=m
    exec(compile(Path(ref['path']).read_bytes(),ref['path'],'exec'),m.__dict__);verify(ref);return m

def available_ram():
    fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    return int(fields['MemAvailable'].split()[0])*1024

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True);args=parser.parse_args()
    path=Path(args.spec);sref=reference(path);need(sref['sha256']==args.spec_sha,'Actual spec SHA')
    spec=parse_json(path.read_text(encoding='utf-8'));validate_spec(spec)
    output=Path(spec['output_directory']);need(not output.exists() and output.parent.resolve(strict=True)==output.parent,'Fresh canonical output parent');output.mkdir(mode=0o700)
    start=time.monotonic();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();report=dict(schema='Root-R101-DiffSAR-CPU-load-probe-report-v1',status='STARTED',operation=OPERATION,run_id=spec['run_id'],spec_ref=sref,source_refs=deepcopy(spec['source_refs']),library_sources=deepcopy(spec['library_sources']),config_ref=spec['config_ref'],checkpoint_ref=spec['checkpoint_ref'],prelock=deepcopy(PRELOCK),pid=os.getpid(),pgid=os.getpgid(0),boot_id=boot,qualified=False,Root_external_admission_required=True,actual_load=False,forward=False,new_training=False,training_credit=0,AP=False,DEV=False,TEST=False,checkpoint_saved=False)
    def save():
        report['elapsed_seconds']=time.monotonic()-start;tmp=output/'REPORT.next'
        with tmp.open('x',encoding='utf-8') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,output/'REPORT.json')
    def check():
        import resource
        need(sorted(os.sched_getaffinity(0))==[0,1],'Actual CPU01 affinity')
        free=available_ram();report['minimum_available_RAM_bytes']=min(free,report.get('minimum_available_RAM_bytes',free));need(free>=PRELOCK['minimum_available_RAM_bytes'],'Worker free-RAM floor')
        peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;report['worker_peak_RSS_bytes']=peak;need(peak<=PRELOCK['maximum_group_RSS_bytes'],'Worker RSS cap')
        need(time.monotonic()-start<PRELOCK['deadline_seconds'],'Worker finite deadline')
    code=1;torch=None;model=None;source_state=None;normalizer_snapshot=None;modules={};refs=[];save()
    try:
        need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14','Locked Linux aarch64 interpreter')
        check();need(available_ram()>=PRELOCK['start_RAM_bytes'],'48GiB startup RAM')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')==str(PRELOCK['seed']),'GPU-hidden process and fixed hash seed')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Bounded thread environment')
        refs=spec_refs(spec)
        for ref in refs:verify(ref)
        need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Own actual worker bytes')
        lease=parse_json(Path(spec['root_lease_ref']['path']).read_text(encoding='utf-8'));validate_lease(spec,lease);need(lease['boot_id']==boot,'Original Root same-boot lease')
        report['root_main_accepted_prefix']=lease['main_accepted_prefix'];report['lease_claim_not_self_authenticating']=True
        adapter=source_module(spec['source_refs']['config_adapter'],'_Root_R101_CPU_Config_Adapter');coverage=source_module(spec['source_refs']['coverage'],'_Root_R101_CPU_Coverage');actual=source_module(spec['source_refs']['actual_load'],'_Root_R101_CPU_Actual_Load')
        need(not any(k.split('.')[0] in ('torch','numpy','detectron2','diffsarshipinst') for k in sys.modules),'No preimported scientific runtime before selected imports')
        author_root=Path(spec['library_sources']['train_net']['path']).parent;need(Path(spec['config_ref']['path']).parent==author_root/'configs','Selected compatible config root');sys.path.insert(0,str(author_root))
        modules={k:importlib.import_module(name) for k,name in LIBRARY_MODULES.items()};torch=modules['torch']
        need(torch.__version__=='2.10.0+cu130' and torch.version.cuda=='13.0' and modules['numpy'].__version__=='1.26.4' and modules['torchvision'].__version__=='0.25.0+cu130','Locked selected runtime versions')
        for key,module in modules.items():need(reference(module.__file__)==spec['library_sources'][key],'Actual selected import: '+key)
        need(spec['library_sources']['tv_extension']['path'] in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries},'Actual loaded torchvision binary')
        need(Path(modules['detectron2'].__file__).parent.parent==author_root and Path(modules['diffsarshipinst'].__file__).parent.parent==author_root and Path(modules['train_net'].__file__).parent==author_root,'One compatible author source root')
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2,'Actual Torch thread counts')
        torch.set_default_dtype(torch.float32);need(not torch.cuda.is_initialized(),'No CUDA initialization before construction')
        random.seed(PRELOCK['seed']);modules['numpy'].random.seed(PRELOCK['seed']);torch.manual_seed(PRELOCK['seed'])
        cfg=configuration(modules,adapter,spec);report.update(config_dump_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(),configuration_overrides={'MODEL.DEVICE':'cpu','SEED':40244023,'DATALOADER.NUM_WORKERS':0},actual_environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=modules['numpy'].__version__,torchvision=modules['torchvision'].__version__,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),selected_runtime_scope='EXPLICIT_SELECTED_MODULES_NOT_ALL_TRANSITIVE_RUNTIME',actual_cuda_initialized_before=False);check();save()
        model=modules['train_net'].Trainer.build_model(cfg);check();report['model_properties_before'],normalizer_snapshot=model_properties(torch,model,modules);need(not torch.cuda.is_initialized(),'No CUDA initialization by CPU factory');save()
        verify(spec['checkpoint_ref'])
        with Path(spec['checkpoint_ref']['path']).open('rb') as f:source_state=torch.load(f,map_location='cpu',weights_only=True)
        verify(spec['checkpoint_ref']);check();report['actual_checkpoint_deserialized_weights_only_CPU']=True;save()
        proof=actual.load_backbone_verified(torch,model,source_state,coverage);need(type(proof) is dict and proof.get('actual_tensor_load_checked') is True,'Actual helper success with explicit tensor comparisons');signature(proof)
        report['actual_load_proof']=proof;report['model_properties_after'],_=model_properties(torch,model,modules,normalizer_snapshot);need(not torch.cuda.is_initialized(),'No CUDA initialization by load helper');check()
        for ref in refs:verify(ref)
        report.update(actual_load=True,selected_sources_and_checkpoint_unchanged_after=True,actual_cuda_initialized_after=False,status='CPU_R101_BACKBONE_LOAD_COMPLETE_PENDING_ROOT');code=0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),retry_permitted=False);code=1
    finally:
        model=None;source_state=None;normalizer_snapshot=None;gc.collect()
        if torch is not None and torch.cuda.is_initialized():report.update(status='FAILED_PRESERVED_NO_RETRY',cleanup_error='Unexpected CUDA initialization in GPU-hidden CPU operation');code=1
        report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),Root_original_tool_owner_group_deadline_cleanup_still_required=True);save()
        print(json.dumps({k:report[k] for k in ('status','worker_exit_code','actual_load','qualified','forward','AP','training_credit','elapsed_seconds')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
