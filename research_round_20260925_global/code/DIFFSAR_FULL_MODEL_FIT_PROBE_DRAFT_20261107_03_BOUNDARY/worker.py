"""Unexecuted full-author SSDD single-FIT loss/backward candidate. Top level stdlib only."""
import argparse
from copy import deepcopy
import datetime
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types

SCHEMA='Root-DiffSAR-SSDD-full-single-FIT-probe-spec-v1'
OPERATION='CUDA_DIFFSAR_SSDD_FULL_SINGLE_FIT_FP32_LOSS_BACKWARD_V01'
LEASE_SCHEMA='Root-DiffSAR-SSDD-full-single-FIT-probe-lease-v1'
SAMPLE_SCHEMA='Root-DiffSARShipInst-SSDD-single-FIT-sample-v1'
LIBRARY_MODULES=dict(torch='torch',numpy='numpy',detectron2='detectron2',
    diffsarshipinst='diffsarshipinst',config='diffsarshipinst.config',
    mapper='diffsarshipinst.dataset_mapper',detector='diffsarshipinst.detector',
    head='diffsarshipinst.head',loss='diffsarshipinst.loss',model_ema='diffsarshipinst.util.model_ema',d2_config='detectron2.config',
    d2_modeling='detectron2.modeling',d2_build='detectron2.modeling.meta_arch.build',
    fpn='detectron2.modeling.backbone.fpn',resnet='detectron2.modeling.backbone.resnet',
    backbone_init='detectron2.modeling.backbone',transform='detectron2.data.transforms.transform',
    structures='detectron2.structures',deform_conv='detectron2.layers.deform_conv',
    d2_extension='detectron2._C',carafe='mmcv.ops.carafe',mmcv_extension='mmcv._ext',
    torchvision='torchvision')
LIBRARY_NAMES=tuple(LIBRARY_MODULES)+('tv_extension',)
SSDD_CONFIG_PIN=dict(bytes=7065,sha256='48d6e3a67a19a737ce22d0659db02da0a878dbde9a5483e5c6ed240fbc87fddb')
D2_EXTENSION_PIN=dict(bytes=20097880,sha256='8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af')
CONFIG_ADAPTER_PIN=dict(bytes=2220,sha256='31bb4935787510af70f96f098b60a6c9eb71472ad6f4760bf20bbaa01d250619')
PRELOCK=dict(seed=40244023,batch_size=1,dtype='float32',AMP=False,TF32=False,
    initialization='AUTHOR_RANDOM_INITIALIZATION_NO_CHECKPOINT',optimizer_steps=0,
    input_format='RGB',resize='AUTHOR_MAPPER_SHORTEST_EDGE512_MAX512_NO_STRETCH',
    augmentation='UNCHANGED_AUTHOR_TRAIN_MAPPER_RANDOM_FLIP_AND_OPTIONAL_CROP',
    author_commit='bdc3362372213bc8c3a1f6d5b7b74dc224d1b871',
    model='DiffSARShipInst',backbone='build_resnet_scje_fpn_backbone',resnet_depth=101,
    proposals=500,heads=6,sampling_steps=1,num_classes=1,
    CPU_affinity=[0,1],threads=2,GPU_allocator_cap_bytes=8*1024**3,
    peak_group_RSS_cap_bytes=16*1024**3,free_RAM_floor_bytes=16*1024**3,
    start_RAM_bytes=48*1024**3,external_deadline_seconds=1800,single_attempt=True,
    whole_device_exclusive=False,exclusive_project_scientific_leaf=True)

def need(condition,message):
    if not condition:raise ValueError(message)

def valid_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'},'Exact three-field reference')
    p=value['path'];need(type(p) is str and p.startswith('/') and str(PurePosixPath(p))==p
        and '..' not in PurePosixPath(p).parts and p!='/','Canonical absolute POSIX reference')
    need(type(value['bytes']) is int and value['bytes']>0,'Positive reference bytes')
    h=value['sha256'];need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'SHA256')

def reference(path):
    original=Path(path);resolved=original.resolve(strict=True)
    need(str(original)==str(resolved) and resolved.is_file(),'Canonical regular file')
    before=resolved.stat();digest=hashlib.sha256();size=0
    with resolved.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);size+=len(block)
    after=resolved.stat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(before)==identity(after),'Source changed while hashing')
    return dict(path=str(resolved),bytes=size,sha256=digest.hexdigest())

def verify(value):
    valid_ref(value);need(reference(value['path'])==value,'Reference mismatch: '+value['path'])

def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)

def validate_sample(sample):
    fields={'schema','dataset','partition','fit_image_count','selection_rule','root_verified_fit_membership',
        'source_image_key','fit_split_origin_ref','annotations_origin_ref','image_ref','record'}
    need(type(sample) is dict and set(sample)==fields,'Exact independent sample fields')
    need(sample['schema']==SAMPLE_SCHEMA and sample['dataset']=='pseg_ssdd' and sample['partition']=='FIT'
        and type(sample['fit_image_count']) is int and sample['fit_image_count']==741,'Fixed FIT741 SSDD sample')
    need(sample['selection_rule']=='FIRST_NONEMPTY_IN_ROOT_CANONICAL_FIT_ORDER'
        and sample['root_verified_fit_membership'] is True,'Root-attested FIT selection; not independently re-proven')
    need(type(sample['source_image_key']) is str and bool(sample['source_image_key']),'Original sample identity')
    for key in ('fit_split_origin_ref','annotations_origin_ref','image_ref'):valid_ref(sample[key])
    rec=sample['record'];need(type(rec) is dict and set(rec)=={'file_name','image_id','height','width','annotations'},'One D2 dataset record')
    need(rec['file_name']==sample['image_ref']['path'] and type(rec['image_id']) in (int,str),'Image identity/path')
    need(all(type(rec[k]) is int and rec[k]>1 for k in ('height','width')),'Original image dimensions')
    need(type(rec['annotations']) is list and len(rec['annotations'])>0,'Nonempty original annotations')
    for anno in rec['annotations']:
        need(type(anno) is dict and {'bbox','bbox_mode','category_id','segmentation','iscrowd'}<=set(anno),'Full real instance annotation')
        need(type(anno['category_id']) is int and anno['category_id']==0 and type(anno['bbox_mode']) is int
            and anno['bbox_mode']==1 and type(anno['iscrowd']) is int and anno['iscrowd']==0,'Zero class / XYWH_ABS / noncrowd')
        box=anno['bbox'];need(type(box) is list and len(box)==4 and all(finite_number(v) for v in box)
            and box[2]>0 and box[3]>0,'Finite positive-area original bbox')
        seg=anno['segmentation']
        if type(seg) is list:
            need(len(seg)>0 and all(type(p) is list and len(p)>=6 and len(p)%2==0
                and all(finite_number(v) for v in p) for p in seg),'Original finite polygon coordinates')
        else:
            need(type(seg) is dict and set(seg)=={'size','counts'} and seg['size']==[rec['height'],rec['width']]
                and type(seg['counts']) is str and len(seg['counts'])>0,'JSON compressed COCO RLE accepted by original mapper')

def validate_spec(spec):
    need(type(spec) is dict and set(spec)=={'schema','operation','run_id','source_refs','library_sources',
        'config_ref','sample_ref','compatibility_receipt_ref','root_lease_ref','output_directory','prelock'},'Exact spec fields')
    need(spec['schema']==SCHEMA and spec['operation']==OPERATION,'Fixed full-model operation')
    need(type(spec['run_id']) is str and spec['run_id'].startswith('DiffSAR-SSDD-full-single-FIT-v01-'),'Run identity')
    need(spec['prelock']==PRELOCK,'Exact prelock')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])=={'worker','config_adapter'},'Own selected worker and explicit config adapter')
    need(type(spec['library_sources']) is dict and set(spec['library_sources'])==set(LIBRARY_NAMES),'Selected library sources')
    refs=list(spec['source_refs'].values())+list(spec['library_sources'].values())+[
        spec[k] for k in ('config_ref','sample_ref','compatibility_receipt_ref','root_lease_ref')]
    for value in refs:valid_ref(value)
    need(len({r['path'] for r in refs})==len(refs),'Distinct selected refs')
    for key,pin in [('config_ref',SSDD_CONFIG_PIN)]:
        need(all(spec[key][k]==v for k,v in pin.items()),'Unchanged author SSDD YAML pin')
    need(Path(spec['config_ref']['path']).name=='DiffSARShipInst-SSDD-train.yaml','SSDD train YAML only')
    need(all(spec['library_sources']['d2_extension'][k]==v for k,v in D2_EXTENSION_PIN.items()),'Actual author _C build pin')
    need(all(spec['source_refs']['config_adapter'][k]==v for k,v in CONFIG_ADAPTER_PIN.items()),'Exact separately reviewed typed config adapter')
    out=spec['output_directory'];need(type(out) is str and out.startswith('/') and str(PurePosixPath(out))==out
        and '..' not in PurePosixPath(out).parts and out!='/','Canonical new output directory')
    need(all(out!=r['path'] and not r['path'].startswith(out+'/') for r in refs),'Outputs separate from sources')

def validate_lease(spec,lease):
    need(type(lease) is dict and lease.get('schema')==LEASE_SCHEMA and lease.get('status')=='ROOT_GRANTED','Explicit Root lease')
    for k in ('operation','run_id','output_directory','prelock'):need(lease.get(k)==spec[k],'Lease '+k)
    need(lease.get('worker_ref')==spec['source_refs']['worker'] and lease.get('sample_ref')==spec['sample_ref']
        and lease.get('config_ref')==spec['config_ref'] and lease.get('config_adapter_ref')==spec['source_refs']['config_adapter'],
        'Lease source/data/config/adapter join')
    need(lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
        and lease.get('exclusive_project_scientific_leaf') is True and lease.get('whole_device_exclusive') is False,'Single scientific leaf lease')

def available_ram():
    values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    return int(values['MemAvailable'].split()[0])*1024

def config_facts(cfg):
    m=cfg.MODEL;d=m.DiffSARShipInst;i=cfg.INPUT
    return dict(meta_architecture=m.META_ARCHITECTURE,backbone=m.BACKBONE.NAME,resnet_depth=m.RESNETS.DEPTH,
        deform_modulated=m.RESNETS.DEFORM_MODULATED,deform_stages=list(m.RESNETS.DEFORM_ON_PER_STAGE),
        deform_groups=m.RESNETS.DEFORM_NUM_GROUPS,num_classes=d.NUM_CLASSES,proposals=d.NUM_PROPOSALS,
        heads=d.NUM_HEADS,sampling_steps=d.SAMPLE_STEP,deep_supervision=d.DEEP_SUPERVISION,
        input_format=i.FORMAT,min_train=list(i.MIN_SIZE_TRAIN),max_train=i.MAX_SIZE_TRAIN,
        crop_enabled=i.CROP.ENABLED,crop_type=i.CROP.TYPE,crop_size=list(i.CROP.SIZE),
        pixel_mean=list(m.PIXEL_MEAN),pixel_std=list(m.PIXEL_STD),AMP=cfg.SOLVER.AMP.ENABLED,
        USE_FED_LOSS=d.USE_FED_LOSS,IMS_PER_BATCH=cfg.SOLVER.IMS_PER_BATCH,seed=cfg.SEED,
        EMA_enabled=cfg.MODEL_EMA.ENABLED)

EXPECTED_CONFIG=dict(meta_architecture='DiffSARShipInst',backbone='build_resnet_scje_fpn_backbone',resnet_depth=101,
    deform_modulated=True,deform_stages=[True]*4,deform_groups=1,num_classes=1,proposals=500,heads=6,
    sampling_steps=1,deep_supervision=True,input_format='RGB',min_train=[512],max_train=512,
    crop_enabled=True,crop_type='absolute_range',crop_size=[512,512],pixel_mean=[123.675,116.28,103.53],
    pixel_std=[58.395,57.12,57.375],AMP=False,USE_FED_LOSS=False,IMS_PER_BATCH=1,seed=40244023,EMA_enabled=False)

def record_gradient_observation(result,report,save):
    groups=result['scopes']
    result['missing_scopes']=[name for name,g in groups.items() if g['parameters_with_gradient']==0]
    result['zero_scopes']=[name for name,g in groups.items()
        if g['parameters_with_gradient']>0 and g['nonzero_elements']==0]
    result['failed_nonzero_scopes']=[name for name,g in groups.items()
        if not (g['parameters_with_gradient']>0 and g['nonzero_elements']>0)]
    report['gradient_summary']=result
    report['gradient_summary_observed_before_scope_assertion']=True
    report['stage']='GRADIENT_SUMMARY_OBSERVED'
    save()

def gradient_summary(torch,model,check,report,save):
    scopes=['backbone.bottom_up.','backbone.','head.iamr.']+['head.head_series.%d.'%i for i in range(6)]
    groups={s:dict(parameters_with_gradient=0,numel_with_gradient=0,nonzero_elements=0) for s in scopes}
    result=dict(trainable_parameters=0,trainable_numel=0,parameters_with_gradient=0,
        parameters_without_gradient=[],all_present_gradients_finite=True,scopes=groups)
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        need(param.dtype==torch.float32 and param.device.type=='cuda','FP32 CUDA trainable parameter')
        result['trainable_parameters']+=1;result['trainable_numel']+=param.numel()
        grad=param.grad
        if grad is None:result['parameters_without_gradient'].append(name);continue
        need(grad.dtype==torch.float32 and grad.shape==param.shape and bool(torch.isfinite(grad).all().item()),'Finite first gradient: '+name)
        nz=int(torch.count_nonzero(grad).item());result['parameters_with_gradient']+=1
        for prefix,group in groups.items():
            if name.startswith(prefix):
                group['parameters_with_gradient']+=1;group['numel_with_gradient']+=grad.numel();group['nonzero_elements']+=nz
        check()
    record_gradient_observation(result,report,save)
    need(all(g['parameters_with_gradient']>0 and g['nonzero_elements']>0 for g in groups.values()),'Backbone, mask tower and every dynamic head reached by backward')
    return result

class SharedIAMROutputBoundary:
    """One non-replacing forward hook; retain only the original shared output."""
    def __init__(self,torch):
        self.torch=torch;self.calls=0;self.tensor=None;self.handle=None
        self.installed=False;self.removed=False;self.released=False;self.retained=False

    def install(self,module):
        need(not self.installed and self.handle is None,'Install shared IAMR hook once')
        self.handle=module.register_forward_hook(self.forward_hook);self.installed=True

    def forward_hook(self,module,inputs,output):
        self.calls+=1;need(self.calls==1,'Exactly one shared IAMR output call')
        need(self.torch.is_tensor(output) and output.ndim==4 and output.shape[0]==1
            and output.shape[1]==8 and all(int(n)>0 for n in output.shape),'Original shared N1/C8 IAMR Tensor')
        need(output.requires_grad and output.dtype==self.torch.float32 and output.device.type=='cuda'
            and output.device.index==0,'Original differentiable FP32 CUDA0 shared output')
        self.tensor=output;output.retain_grad();self.retained=True
        return None

    def observe_after_backward(self):
        need(self.calls==1 and self.tensor is not None and self.retained,'One retained original IAMR output')
        output=self.tensor;grad=output.grad
        observed=dict(boundary='head.iamr.output',objective='sum(original_loss_dict.values())',
            hook_calls=self.calls,return_policy='None_no_output_replacement',original_output_retained=self.retained,
            output_shape=list(output.shape),output_dtype=str(output.dtype),output_requires_grad=bool(output.requires_grad),
            gradient_present=grad is not None,gradient_shape=None,gradient_dtype=None,finite=None,
            max_abs=None,nonzero_elements=None,classification='NONE')
        if grad is not None:
            need(grad.shape==output.shape and grad.dtype==output.dtype and grad.device==output.device,'Retained gradient identity metadata')
            finite=bool(self.torch.isfinite(grad).all().item());nonzero=int(self.torch.count_nonzero(grad).item())
            observed.update(gradient_shape=list(grad.shape),gradient_dtype=str(grad.dtype),finite=finite,
                nonzero_elements=nonzero,max_abs=float(grad.abs().max().item()) if finite else None,
                classification=('FINITE_NONZERO' if nonzero else 'FINITE_ALL_ZERO') if finite else 'NONFINITE')
        return observed

    def close(self):
        handle=self.handle;self.handle=None
        try:
            if handle is not None:handle.remove()
            self.removed=True
        finally:
            self.tensor=None;self.released=True

def run_full(spec,sample,modules,adapter,report,save,check):
    torch=modules['torch'];np=modules['numpy'];model=None;batch=None;losses=None;total=None
    boundary=SharedIAMROutputBoundary(torch)
    try:
        cfg=modules['d2_config'].get_cfg();modules['config'].add_DiffSARShipInst_config(cfg)
        modules['model_ema'].add_model_ema_configs(cfg)
        registration=adapter.register_inactive_bifpn_schema(cfg)
        cfg.merge_from_file(spec['config_ref']['path'])
        verification=adapter.verify_inactive_bifpn_schema(cfg)
        facts=config_facts(cfg);need(facts==EXPECTED_CONFIG and cfg.MODEL.DEVICE=='cuda','Unchanged author SSDD configuration')
        cfg.freeze();report.update(config_facts=facts,config_dump_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(),
            config_overrides=[],weights_requested=False,weights_loaded=False,optimizer_constructed=False,
            configuration_entry_sequence='get_cfg/add_DiffSARShipInst_config/add_model_ema_configs/register_inactive_bifpn/merge_original_yaml/verify_inactive_bifpn',
            config_adapter_registration=registration,config_adapter_verification=verification)
        random.seed(PRELOCK['seed']);np.random.seed(PRELOCK['seed']);torch.manual_seed(PRELOCK['seed']);torch.cuda.manual_seed_all(PRELOCK['seed'])
        report['seed_execution_order']='SEED_AFTER_CONFIG_THEN_AUTHOR_MAPPER_THEN_BUILD_MODEL_THEN_FORWARD'
        mapper=modules['mapper'].DiffSARShipInstDatasetMapper(cfg,is_train=True)
        record=mapper(deepcopy(sample['record']));image=record['image'];instances=record['instances']
        need(image.dtype==torch.uint8 and image.ndim==3 and image.shape[0]==3,'Original mapper RGB CHW uint8')
        height,width=map(int,image.shape[-2:]);need(max(height,width)==512 and min(height,width)>1,'Author proportional final512 bound')
        need(isinstance(instances,modules['structures'].Instances) and instances.image_size==(height,width)
            and isinstance(instances.gt_boxes,modules['structures'].Boxes)
            and isinstance(instances.gt_masks,modules['structures'].BitMasks),'Real mapper Instances and BitMasks')
        count=len(instances);need(count>0 and instances.gt_classes.dtype==torch.int64
            and bool((instances.gt_classes==0).all().item()),'Nonempty real zero-class instances after original augmentation')
        need(list(instances.gt_masks.tensor.shape)==[count,height,width]
            and instances.gt_masks.tensor.dtype==torch.bool and bool(instances.gt_masks.nonempty().all().item())
            and bool(instances.gt_boxes.nonempty().all().item()),'Nonempty true masks and boxes')
        report.update(stage='AUTHOR_MAPPER_COMPLETE',input_summary=dict(image_shape=list(image.shape),image_dtype=str(image.dtype),
            input_format='RGB',original_image_id=sample['record']['image_id'],annotations_before=len(sample['record']['annotations']),
            instances_after=count,mask_type=type(instances.gt_masks).__name__,class_mapping='original_single_ship_to_0',
            original_mapper_augmentation_preserved=True,synthetic_targets=False));save();check()
        model=modules['d2_build'].build_model(cfg)
        need(type(model).__name__=='DiffSARShipInst' and type(model.backbone).__name__=='SCJE_FPN'
            and model.num_proposals==500 and model.num_heads==6 and len(model.head.head_series)==6
            and model.sampling_timesteps==1,'Full original model structure')
        model.train();need(not torch.is_autocast_enabled(),'No AMP context')
        report.update(stage='FULL_MODEL_CONSTRUCTED',model_class=type(model).__name__,backbone_class=type(model.backbone).__name__,
            model_parameter_numel=sum(p.numel() for p in model.parameters()),
            native_modulated_deform_modules=sum(type(m).__name__=='ModulatedDeformConv' for m in model.modules()),
            CARAFEPack_modules=sum(type(m).__name__=='CARAFEPack' for m in model.modules()),
            original_float64_diffusion_buffers_preserved=True)
        need(report['native_modulated_deform_modules']>0,'Original modulated DCN modules');save();check()
        boundary.install(model.head.iamr)
        batch=[record];losses=model(batch);torch.cuda.synchronize(0);check()
        expected={name+suffix for suffix in ['']+['_'+str(i) for i in range(5)] for name in ('loss_ce','loss_bbox','loss_fiou','loss_masks')}
        need(type(losses) is dict and set(losses)==expected,'All original six-head classification/box/FIoU/mask losses')
        values={}
        for key,value in losses.items():
            need(torch.is_tensor(value) and value.numel()==1 and value.requires_grad and bool(torch.isfinite(value).all().item()),'Finite differentiable original loss: '+key)
            values[key]=float(value.detach().item())
        need(values['loss_masks']>0,'Nonzero real-mask branch coverage')
        total=sum(losses.values());need(total.numel()==1 and bool(torch.isfinite(total).item()),'Finite total original weighted loss')
        report.update(stage='FULL_FORWARD_LOSS_COMPLETE',losses=values,total_loss=float(total.detach().item()));save();check()
        total.backward();torch.cuda.synchronize(0);check()
        report['IAMR_shared_output_gradient']=boundary.observe_after_backward()
        report['IAMR_shared_output_observed_before_scope_assertion']=True
        save()
        report.update(gradient_summary=gradient_summary(torch,model,check,report,save),stage='FULL_BACKWARD_COMPLETE',
            optimizer_steps=0,all_original_losses_and_present_first_gradients_finite=True)
        save()
    finally:
        try:
            boundary.close()
        finally:
            report['IAMR_shared_boundary_cleanup']=dict(hook_installed=boundary.installed,hook_calls=boundary.calls,
                hook_removed=boundary.removed,held_original_output_released=boundary.released)
            model=None;batch=None;losses=None;total=None

def clear_blas_workspaces(torch,report):
    clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    need(callable(clear),'Locked Torch2.10 cuBLAS workspace cleanup API required')
    report['cuBLAS_workspace_cleanup_entered']=True;clear();report['cuBLAS_workspaces_cleared']=True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True);args=parser.parse_args()
    spec_path=Path(args.spec).resolve(strict=True);spec_ref=reference(str(spec_path));need(spec_ref['sha256']==args.spec_sha,'Actual spec SHA')
    spec=json.loads(spec_path.read_bytes());validate_spec(spec)
    out=Path(spec['output_directory']);need(not out.exists(),'Fresh output only');out.mkdir(mode=0o700)
    start=time.monotonic();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    report=dict(schema='DiffSAR-SSDD-full-single-FIT-probe-report-v1',operation=OPERATION,run_id=spec['run_id'],
        status='STARTED',stage='ENTERED',qualified=False,Root_external_admission_required=True,worker_exit_code=None,
        pid=os.getpid(),pgid=os.getpgid(0),boot_id=boot,spec_ref=spec_ref,prelock=deepcopy(PRELOCK),
        source_refs=deepcopy(spec['source_refs']),library_sources=deepcopy(spec['library_sources']),
        config_ref=spec['config_ref'],sample_ref=spec['sample_ref'],compatibility_receipt_ref=spec['compatibility_receipt_ref'],
        initialization=PRELOCK['initialization'],efficacy_comparison=False,AP=False,optimizer_steps=0,
        formal_training_credit=0,new_training=False,TGRS_success=False,GT_read_scope='ONE_ROOT_SELECTED_FIT_RECORD_AND_IMAGE',
        model_weight_read=False,whole_device_exclusive=False,exclusive_project_scientific_leaf=True,
        full_FIT_membership_independently_recomputed=False,CPU_affinity=sorted(os.sched_getaffinity(0)))
    def save():
        report['elapsed_seconds']=time.monotonic()-start
        temp=out/'REPORT.next'
        with temp.open('w',encoding='utf-8') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,out/'REPORT.json')
    def check():
        import resource
        need(sorted(os.sched_getaffinity(0))==[0,1],'CPU01 affinity')
        free=available_ram();report['minimum_available_RAM_bytes']=min(free,report.get('minimum_available_RAM_bytes',free))
        need(free>=PRELOCK['free_RAM_floor_bytes'],'Free RAM floor')
        need(time.monotonic()-start<PRELOCK['external_deadline_seconds'],'Observed worker deadline')
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        report['worker_peak_RSS_bytes']=rss;need(rss<=PRELOCK['peak_group_RSS_cap_bytes'],'Worker peak RSS bound')
    torch=None;code=1;modules={};sample=None;refs=[];save()
    try:
        check();need(available_ram()>=PRELOCK['start_RAM_bytes'],'48 GiB startup admission')
        refs=list(spec['source_refs'].values())+list(spec['library_sources'].values())+[
            spec[k] for k in ('config_ref','sample_ref','compatibility_receipt_ref','root_lease_ref')]
        for value in refs:verify(value)
        need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Actual selected own worker')
        adapter=types.ModuleType('_Root_DiffSAR_FIT_Config_Adapter')
        adapter.__file__=spec['source_refs']['config_adapter']['path']
        exec(compile(Path(adapter.__file__).read_bytes(),adapter.__file__,'exec'),adapter.__dict__)
        lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());validate_lease(spec,lease);need(lease.get('boot_id')==boot,'Same-boot Root lease')
        sample=json.loads(Path(spec['sample_ref']['path']).read_bytes());validate_sample(sample);verify(sample['image_ref']);refs.append(sample['image_ref'])
        report['sample_provenance']={k:v for k,v in sample.items() if k not in ('record','image_ref')}
        report['image_ref']=sample['image_ref'];report['origin_refs_scope']='METADATA_ONLY_NOT_READ_FULL_SPLIT_OR_ANNOTATIONS'
        need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14','Locked runtime platform')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'External thread environment')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Exactly selected CUDA0 visibility')
        modules={k:importlib.import_module(name) for k,name in LIBRARY_MODULES.items()};torch=modules['torch']
        need(torch.__version__=='2.10.0+cu130' and torch.version.cuda=='13.0'
            and modules['torchvision'].__version__=='0.25.0+cu130','Locked Torch/Torchvision/CUDA build')
        for key,module in modules.items():need(reference(module.__file__)==spec['library_sources'][key],'Actual imported selected library '+key)
        loaded={str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
        need(spec['library_sources']['tv_extension']['path'] in loaded,'Actual loaded Torchvision native extension')
        source_root=Path(modules['detectron2'].__file__).parent.parent
        need(Path(modules['diffsarshipinst'].__file__).parent.parent==source_root
            and Path(spec['config_ref']['path']).parent==source_root/'configs','Bound compatible author source root')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Single exposed CUDA device')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,PRELOCK['GPU_allocator_cap_bytes']/props.total_memory),device=0)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        report.update(actual_library_refs={**{k:reference(m.__file__) for k,m in modules.items()},
                'tv_extension':reference(spec['library_sources']['tv_extension']['path'])},
            actual_environment=dict(python=platform.python_version(),machine=platform.machine(),torch=torch.__version__,
                torchvision=modules['torchvision'].__version__,cuda_build=torch.version.cuda,
                device=props.name,device_total_memory_bytes=props.total_memory,torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                TF32_cudnn=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled(),
                executable=sys.executable,author_source_root=str(source_root)),
            runtime_scope='SELECTED_SOURCE_REFS_NOT_ALL_TRANSITIVE_RUNTIME_OR_FULLY_ISOLATED_VENV')
        save();run_full(spec,sample,modules,adapter,report,save,check);check()
        for value in refs:verify(value)
        report.update(selected_sources_and_sample_unchanged_after=True,
            status='FULL_AUTHOR_SSDD_SINGLE_REAL_FIT_LOSS_BACKWARD_PASSED_PENDING_ROOT');code=0
    except BaseException as error:
        oom=bool(torch is not None and isinstance(error,torch.cuda.OutOfMemoryError))
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),failure_scope='RESOURCE_OOM_NOT_METHOD_INVALIDITY' if oom else 'UNRESOLVED_EXECUTION_FAILURE_NOT_EFFICACY',
            retry_permitted=False);code=1
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                gc.collect();torch.cuda.synchronize(0);clear_blas_workspaces(torch,report);torch.cuda.empty_cache()
                allocated=torch.cuda.memory_allocated(0);reserved=torch.cuda.memory_reserved(0);peak=torch.cuda.max_memory_allocated(0)
                report.update(CUDA_allocated_after_cleanup=allocated,CUDA_reserved_after_cleanup=reserved,CUDA_peak_allocated_bytes=peak)
                need(allocated==reserved==0 and peak<=PRELOCK['GPU_allocator_cap_bytes'],'CUDA final0/0 and8GiB allocator peak')
            except BaseException as error:
                report.update(cleanup_error=str(error),status='FAILED_PRESERVED_NO_RETRY');code=1
        report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            CPU_group_deadline_birth_cleanup_requires_Root_supervisor=True)
        save()
        print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','elapsed_seconds','AP','formal_training_credit')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
