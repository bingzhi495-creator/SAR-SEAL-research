"""Finite native DiffSAR training-health draft. Top level standard library only."""
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
import re
import sys
import time
import traceback
import types

def need(condition,message):
    if not condition:raise ValueError(message)

def canonical_input_plan(manifest,input_root):
    """Preserve logical rows and annotation identity even when byte objects repeat."""
    need(manifest['schema']=='SSDD-FIT741-portable-original-byte-input-manifest-v1','Portable input schema')
    p=manifest['population']
    need(p['logical_training_rows']==741 and p['unique_JPEG_byte_objects']==730
         and p['FIT_annotation_instances']==1661 and p['deduplicate_training_population'] is False,'Canonical FIT population')
    root=PurePosixPath(input_root)
    need(root.is_absolute() and str(root)==input_root and '..' not in root.parts,'Canonical private input root')
    rows=manifest['rows'];need(type(rows) is list and len(rows)==741,'All logical rows')
    plan=[]
    for i,r in enumerate(rows):
        ref=r['original_image_ref'];h=ref['sha256']
        need(r['dataset_index']==i and type(r['image_id']) is int,'Canonical logical order')
        need(type(h) is str and re.fullmatch('[0-9a-f]{64}',h) and type(ref['bytes']) is int and 0<ref['bytes']<=2*1024**2,'Bounded original byte reference')
        need(r['archive_image_entry']=='images/'+h+'.jpg','Only content-addressed images')
        need(type(r['original_basename']) is str and re.fullmatch('[0-9]{6}\\.jpg',r['original_basename']),'Original JPEG basename')
        need(type(r['original_shape']) is list and len(r['original_shape'])==2 and all(type(v) is int and v>1 for v in r['original_shape']),'Original dimensions')
        need(type(r['ann_ids']) is list and r['ann_ids'] and all(type(v) is int and v>0 for v in r['ann_ids']),'Canonical FIT annotation IDs')
        plan.append(dict(dataset_index=i,image_id=r['image_id'],ann_ids=list(r['ann_ids']),
            original_shape=list(r['original_shape']),original_basename=r['original_basename'],
            image_ref=dict(path=str(root/r['archive_image_entry']),bytes=ref['bytes'],sha256=h)))
    need(len({r['image_id'] for r in plan})==741,'Distinct logical image IDs')
    need(len({r['image_ref']['sha256'] for r in plan})==730,'Unique byte count')
    ann_ids=[a for r in plan for a in r['ann_ids']]
    need(len(ann_ids)==1661 and len(set(ann_ids))==1661,'Distinct canonical annotation identities')
    return plan

def align_coco_records(records,plan):
    """Join native COCO records by image identity, then use the fixed logical order."""
    need(len(records)==len(plan)==741,'Native loader must preserve the whole FIT population')
    by_id={r['image_id']:r for r in records}
    need(len(by_id)==741 and set(by_id)=={r['image_id'] for r in plan},'Whole FIT image identity join')
    ordered=[]
    for row in plan:
        r=deepcopy(by_id[row['image_id']])
        need([r['height'],r['width']]==row['original_shape'],'Native original shape')
        need(PurePosixPath(r['file_name']).name==row['original_basename'],'Original COCO file_name association')
        ids=[a['id'] for a in r['annotations']]
        need(len(ids)==len(set(ids)) and set(ids)==set(row['ann_ids']),'Native loader cannot drop or replace canonical annotations')
        r['file_name']=row['image_ref']['path']
        r['health_dataset_index']=row['dataset_index']
        ordered.append(r)
    return ordered

def validate_completed_steps(steps,optimizer_steps):
    need(optimizer_steps==32 and len(steps)==32,'Exactly32 actual optimizer updates required')
    need([s['dataset_index'] for s in steps]==list(range(32)),'Fixed logical sample order, no replay')
    need(len({s['image_id'] for s in steps})==32,'Distinct logical FIT cases')
    for i,s in enumerate(steps):
        need(s['optimizer_steps_completed']==i+1 and s['all_present_gradients_finite'] is True,'Actual finite completed update count')
        need(type(s['total_loss']) in (float,int) and math.isfinite(s['total_loss']),'Finite recorded total loss')

SCHEMA='Root-DiffSAR-SSDD-native-train-health-spec-v1'
OPERATION='CUDA_DIFFSAR_SSDD_NATIVE_TRAIN_HEALTH32_FP32_V01'
LEASE_SCHEMA='Root-DiffSAR-SSDD-native-train-health-lease-v1'
LIBRARY_MODULES=dict(torch='torch',numpy='numpy',detectron2='detectron2',
    diffsarshipinst='diffsarshipinst',config='diffsarshipinst.config',
    mapper='diffsarshipinst.dataset_mapper',detector='diffsarshipinst.detector',
    head='diffsarshipinst.head',loss='diffsarshipinst.loss',model_ema='diffsarshipinst.util.model_ema',d2_config='detectron2.config',
    d2_modeling='detectron2.modeling',d2_build='detectron2.modeling.meta_arch.build',
    fpn='detectron2.modeling.backbone.fpn',resnet='detectron2.modeling.backbone.resnet',
    backbone_init='detectron2.modeling.backbone',transform='detectron2.data.transforms.transform',
    structures='detectron2.structures',deform_conv='detectron2.layers.deform_conv',
    d2_extension='detectron2._C',carafe='mmcv.ops.carafe',mmcv_extension='mmcv._ext',
    torchvision='torchvision',train_net='train_net',d2_data_build='detectron2.data.build',
    d2_coco='detectron2.data.datasets.coco',d2_sampler='detectron2.data.samplers.distributed_sampler',
    d2_solver_build='detectron2.solver.build',d2_events='detectron2.utils.events')
LIBRARY_NAMES=tuple(LIBRARY_MODULES)+('tv_extension',)
SSDD_CONFIG_PIN=dict(bytes=7065,sha256='48d6e3a67a19a737ce22d0659db02da0a878dbde9a5483e5c6ed240fbc87fddb')
D2_EXTENSION_PIN=dict(bytes=20097880,sha256='8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af')
CONFIG_ADAPTER_PIN=dict(bytes=2220,sha256='31bb4935787510af70f96f098b60a6c9eb71472ad6f4760bf20bbaa01d250619')
INPUT_MANIFEST_PIN=dict(bytes=861009,sha256='5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47')
FIT_COCO_PIN=dict(bytes=498760,sha256='62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1')
PRELOCK=dict(seed=40244023,batch_size=1,dtype='float32',AMP=False,TF32=False,
    initialization='PUBLIC_CODE_CONSTRUCTION_INITIALIZATION_NO_CHECKPOINT_NOT_PAPER_RECIPE_CERTIFIED',
    optimizer_steps=32,sample_order='ROOT_CANONICAL_FIT_DATASET_INDEX_0_THROUGH_31',
    sampler='AUTHOR_D2_INFERENCE_SAMPLER_FIXED_ORDER_HEALTH_ADAPTATION',num_workers=0,
    input_format='RGB',augmentation='UNCHANGED_AUTHOR_TRAIN_MAPPER',
    author_commit='bdc3362372213bc8c3a1f6d5b7b74dc224d1b871',
    model='DiffSARShipInst',backbone='build_resnet_scje_fpn_backbone',resnet_depth=101,
    proposals=500,heads=6,sampling_steps=1,num_classes=1,
    optimizer='AUTHOR_Trainer.build_optimizer',scheduler='AUTHOR_Trainer.build_lr_scheduler',
    schedule_MAX_ITER=18560,schedule_STEPS=[14848,17632],BASE_LR=3.5e-5,
    checkpoints=[1,8,32],checkpoint_weights_saved=False,
    CPU_affinity=[0,1],threads=2,GPU_allocator_cap_bytes=8*1024**3,
    peak_group_RSS_cap_bytes=16*1024**3,free_RAM_floor_bytes=16*1024**3,
    start_RAM_bytes=48*1024**3,external_deadline_seconds=1800,single_attempt=True,
    whole_device_exclusive=False,exclusive_project_scientific_leaf=True)

def valid_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'},'Exact three-field reference')
    p=value['path'];need(type(p) is str and p.startswith('/') and str(PurePosixPath(p))==p
        and '..' not in PurePosixPath(p).parts and p!='/','Canonical absolute POSIX reference')
    need(type(value['bytes']) is int and value['bytes']>0,'Positive reference bytes')
    need(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']),'SHA256')

def reference(path):
    p=Path(path);resolved=p.resolve(strict=True)
    need(str(p)==str(resolved) and resolved.is_file(),'Canonical regular file')
    a=resolved.stat();h=hashlib.sha256();n=0
    with resolved.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
    b=resolved.stat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(a)==identity(b),'Source changed during byte verification')
    return dict(path=str(resolved),bytes=n,sha256=h.hexdigest())

def verify(value):
    valid_ref(value);need(reference(value['path'])==value,'Reference mismatch: '+value['path'])

def spec_refs(spec):
    return list(spec['source_refs'].values())+list(spec['library_sources'].values())+[
        spec[k] for k in ('config_ref','input_manifest_ref','annotations_ref','compatibility_receipt_ref','root_lease_ref')]

def validate_spec(spec):
    need(type(spec) is dict and set(spec)=={'schema','operation','run_id','source_refs','library_sources',
        'config_ref','input_manifest_ref','annotations_ref','input_root','compatibility_receipt_ref','root_lease_ref','output_directory','prelock'},'Exact native health spec fields')
    need(spec['schema']==SCHEMA and spec['operation']==OPERATION and spec['prelock']==PRELOCK,'Fixed health operation/prelock')
    need(type(spec['run_id']) is str and spec['run_id'].startswith('DiffSAR-SSDD-native-health32-v01-'),'Health run identity')
    need(set(spec['source_refs'])=={'worker','config_adapter'} and set(spec['library_sources'])==set(LIBRARY_NAMES),'Selected source identities')
    refs=spec_refs(spec)
    for r in refs:valid_ref(r)
    need(len({r['path'] for r in refs})==len(refs),'Distinct source refs')
    for key,pin in [('config_ref',SSDD_CONFIG_PIN),('input_manifest_ref',INPUT_MANIFEST_PIN),('annotations_ref',FIT_COCO_PIN)]:
        need(all(spec[key][k]==v for k,v in pin.items()),'Fixed '+key+' byte pin')
    need(all(spec['source_refs']['config_adapter'][k]==v for k,v in CONFIG_ADAPTER_PIN.items()),'Reviewed config adapter')
    need(all(spec['library_sources']['d2_extension'][k]==v for k,v in D2_EXTENSION_PIN.items()),'Author D2 binary')
    root=spec['input_root'];out=spec['output_directory']
    for p in (root,out):need(type(p) is str and PurePosixPath(p).is_absolute() and str(PurePosixPath(p))==p and '..' not in PurePosixPath(p).parts and p!='/','Canonical input/output root')
    need(spec['input_manifest_ref']['path']==root+'/manifest.json' and spec['annotations_ref']['path']==root+'/annotations/fit.json','Original extracted input entries')
    need(out!=root and not root.startswith(out+'/') and not out.startswith(root+'/'),'Output outside inputs')
    need(all(out!=r['path'] and not r['path'].startswith(out+'/') for r in refs),'Output outside sources')

def validate_lease(spec,lease):
    need(lease.get('schema')==LEASE_SCHEMA and lease.get('status')=='ROOT_GRANTED','Explicit Root lease')
    for k in ('operation','run_id','output_directory','prelock','input_root'):need(lease.get(k)==spec[k],'Lease '+k)
    joins={'worker_ref':spec['source_refs']['worker'],'config_adapter_ref':spec['source_refs']['config_adapter'],
        'config_ref':spec['config_ref'],'input_manifest_ref':spec['input_manifest_ref'],'annotations_ref':spec['annotations_ref']}
    for k,v in joins.items():need(lease.get(k)==v,'Lease input/source join '+k)
    need(lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
        and lease.get('exclusive_project_scientific_leaf') is True and lease.get('whole_device_exclusive') is False,'Single scientific leaf lease')

def available_ram():
    fields={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in Path('/proc/meminfo').read_text().splitlines()}
    return int(fields['MemAvailable'].split()[0])*1024

def configuration(modules,adapter,spec):
    cfg=modules['d2_config'].get_cfg();modules['config'].add_DiffSARShipInst_config(cfg)
    modules['model_ema'].add_model_ema_configs(cfg);adapter.register_inactive_bifpn_schema(cfg)
    cfg.merge_from_file(spec['config_ref']['path']);adapter.verify_inactive_bifpn_schema(cfg)
    d=cfg.MODEL.DiffSARShipInst;s=cfg.SOLVER
    need(cfg.MODEL.META_ARCHITECTURE=='DiffSARShipInst' and cfg.MODEL.BACKBONE.NAME=='build_resnet_scje_fpn_backbone'
         and cfg.MODEL.RESNETS.DEPTH==101 and cfg.MODEL.RESNETS.DEFORM_MODULATED
         and list(cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE)==[True]*4,'Original R101/SCJE/modulated DCN')
    need(d.NUM_PROPOSALS==500 and d.NUM_HEADS==6 and d.SAMPLE_STEP==1 and d.NUM_CLASSES==1
         and d.DEEP_SUPERVISION and not d.USE_FED_LOSS and not cfg.MODEL_EMA.ENABLED,'Original head/loss/EMA configuration')
    need(cfg.INPUT.FORMAT=='RGB' and list(cfg.INPUT.MIN_SIZE_TRAIN)==[512] and cfg.INPUT.MAX_SIZE_TRAIN==512
         and cfg.INPUT.CROP.ENABLED and cfg.INPUT.CROP.TYPE=='absolute_range' and list(cfg.INPUT.CROP.SIZE)==[512,512],'Original mapper resize/crop')
    need(list(cfg.MODEL.PIXEL_MEAN)==[123.675,116.28,103.53] and list(cfg.MODEL.PIXEL_STD)==[58.395,57.12,57.375],'Original image normalization')
    need(cfg.MODEL.WEIGHTS in (None,'') and cfg.MODEL.DEVICE=='cuda','No checkpoint requested or loaded')
    need(s.IMS_PER_BATCH==1 and s.OPTIMIZER=='ADAMW' and s.BASE_LR==3.5e-5 and s.BACKBONE_MULTIPLIER==1.0
         and s.WEIGHT_DECAY==1e-4 and s.MAX_ITER==18560 and list(s.STEPS)==[14848,17632]
         and s.WARMUP_FACTOR==1.0 and s.WARMUP_ITERS==10 and not s.AMP.ENABLED,'Native optimizer and complete schedule preserved')
    need(s.CLIP_GRADIENTS.ENABLED and s.CLIP_GRADIENTS.CLIP_TYPE=='full_model' and s.CLIP_GRADIENTS.CLIP_VALUE==1.0,'Native global gradient clip')
    cfg.SEED=PRELOCK['seed'];cfg.DATALOADER.NUM_WORKERS=0
    cfg.freeze();return cfg

def parameter_gradients(torch,model,check):
    groups={p:dict(parameters=0,parameters_with_gradient=0,parameters_without_gradient=0,elements_with_gradient=0,nonzero_gradient_elements=0,max_abs_gradient=0.0)
        for p in ['head.iamr.']+['head.head_series.%d.controller.'%i for i in range(6)]}
    all_present=0;without=0
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        need(param.dtype==torch.float32 and param.device.type=='cuda','Original FP32 CUDA parameter')
        g=param.grad
        if g is None:without+=1
        else:
            need(g.shape==param.shape and bool(torch.isfinite(g).all().item()),'All present gradients finite: '+name)
            all_present+=1
        for prefix,row in groups.items():
            if not name.startswith(prefix):continue
            row['parameters']+=1
            if g is None:row['parameters_without_gradient']+=1
            else:
                row['parameters_with_gradient']+=1;row['elements_with_gradient']+=g.numel()
                row['nonzero_gradient_elements']+=int(torch.count_nonzero(g).item())
                row['max_abs_gradient']=max(row['max_abs_gradient'],float(g.abs().max().item()))
        check()
    need(all(v['parameters']>0 for v in groups.values()),'Expected original IAMR and six mask controller scopes exist')
    return dict(all_present_gradients_finite=True,parameters_with_gradient=all_present,parameters_without_gradient=without,groups=groups)

class SharedBoundary:
    def __init__(self,torch,module):
        self.torch=torch;self.tensor=None;self.calls=0
        self.handle=module.register_forward_hook(self.hook)
    def hook(self,module,args,output):
        self.calls+=1;need(self.calls==1 and self.torch.is_tensor(output) and output.ndim==4
            and output.shape[0]==1 and output.shape[1]==8 and output.requires_grad,'One original IAMR output per step')
        self.tensor=output;output.retain_grad();return None
    def observe(self):
        need(self.calls==1 and self.tensor is not None,'Original retained boundary')
        g=self.tensor.grad;r=dict(shape=list(self.tensor.shape),classification='NONE',gradient_present=g is not None,finite=None,nonzero_elements=None,max_abs=None)
        if g is not None:
            finite=bool(self.torch.isfinite(g).all().item());nz=int(self.torch.count_nonzero(g).item())
            r.update(finite=finite,nonzero_elements=nz,max_abs=float(g.abs().max().item()) if finite else None,
                classification=('FINITE_NONZERO' if nz else 'FINITE_ALL_ZERO') if finite else 'NONFINITE')
            need(finite,'Finite total gradient at IAMR output if present')
        return r
    def reset(self):self.tensor=None;self.calls=0
    def close(self):self.handle.remove();self.tensor=None

def selected_update_snapshot(torch,model,optimizer):
    """Original tensor copies, no optimizer mutation or counterfactual optimizer."""
    settings={id(p):(float(g['lr']),float(g['weight_decay'])) for g in optimizer.param_groups for p in g['params']}
    out={}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if not (name.startswith('head.iamr.') or '.controller.' in name):continue
            lr,wd=settings[id(p)];before=p.detach().clone();decay=before.clone()
            if p.grad is not None:decay.mul_(1-lr*wd)
            out[name]=(before,decay,p.grad is not None,0 if p.grad is None else int(torch.count_nonzero(p.grad).item()))
    return out

def selected_update_observation(torch,model,before):
    out={}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name not in before:continue
            old,decay,present,nz=before[name]
            need(bool(torch.isfinite(p).all().item()),'Finite updated observed parameter')
            delta=p.detach()-old;residual=p.detach()-decay
            tol=4*torch.finfo(p.dtype).eps*max(1.0,float(decay.abs().max().item()))
            out[name]=dict(gradient_present_before_step=present,nonzero_data_gradient_elements_before_clip=nz,
                max_abs_actual_parameter_change=float(delta.abs().max().item()),changed_elements=int(torch.count_nonzero(delta).item()),
                max_abs_residual_vs_decay_only=float(residual.abs().max().item()),
                residual_elements_above_roundoff_bound=int(torch.count_nonzero(residual.abs()>tol).item()),roundoff_bound=tol)
    return out

def train_health(spec,plan,modules,adapter,report,save,check):
    torch=modules['torch'];np=modules['numpy'];model=None;optimizer=None;scheduler=None;loader=None;iterator=None
    boundary=None;batch=None;losses=None;total=None;gradients=None;before=None;cfg=None;records=None
    try:
        cfg=configuration(modules,adapter,spec)
        report.update(config_dump_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(),
            config_overrides={'SEED':PRELOCK['seed'],'DATALOADER.NUM_WORKERS':0},
            loader_overrides={'sampler':'InferenceSampler','logical_order':'canonical0..740','num_workers':0},
            full_native_schedule_MAX_ITER=cfg.SOLVER.MAX_ITER,schedule_truncated_to_health_budget=False,
            weights_requested=False,weights_loaded=False,checkpoint_saved=False,configuration_frozen=True)
        random.seed(PRELOCK['seed']);np.random.seed(PRELOCK['seed']);torch.manual_seed(PRELOCK['seed']);torch.cuda.manual_seed_all(PRELOCK['seed'])
        model=modules['train_net'].Trainer.build_model(cfg)
        need(type(model).__name__=='DiffSARShipInst' and type(model.backbone).__name__=='SCJE_FPN'
            and model.num_proposals==500 and model.num_heads==6 and len(model.head.head_series)==6,'Full unchanged native model')
        model.train();optimizer=modules['train_net'].Trainer.build_optimizer(cfg,model)
        scheduler=modules['train_net'].Trainer.build_lr_scheduler(cfg,optimizer)
        need(len(optimizer.param_groups)==len({id(p) for p in model.parameters() if p.requires_grad}),'Native one parameter per group')
        report.update(model_parameter_numel=sum(p.numel() for p in model.parameters()),
            optimizer_constructed=True,optimizer_class=type(optimizer).__name__,scheduler_class=type(scheduler).__name__,
            optimizer_defaults=json.loads(json.dumps(optimizer.defaults,allow_nan=False)),
            optimizer_group_weight_decays=sorted({float(g['weight_decay']) for g in optimizer.param_groups}),
            optimizer_group_count=len(optimizer.param_groups),native_clipping_in_optimizer_step=True,
            scheduler_step_order='AFTER_OPTIMIZER_STEP',training_order='SEED_THEN_NATIVE_MODEL_OPTIMIZER_SCHEDULER_THEN_FINITE_LOADER',
            initialization_is_formal_paper_recipe=False)
        records=modules['d2_coco'].load_coco_json(spec['annotations_ref']['path'],spec['input_root']+'/by_name',dataset_name='Root_DiffSAR_FIT741_native_health32_v01',extra_annotation_keys=['id'])
        records=align_coco_records(records,plan)
        need(all(a['category_id']==0 for r in records for a in r['annotations']),'Native single-class contiguous category mapping')
        mapper=modules['mapper'].DiffSARShipInstDatasetMapper(cfg,is_train=True)
        loader=modules['d2_data_build'].build_detection_train_loader(dataset=records,mapper=mapper,
            sampler=modules['d2_sampler'].InferenceSampler(len(records)),total_batch_size=1,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,num_workers=0)
        iterator=iter(loader);boundary=SharedBoundary(torch,model.head.iamr)
        report.update(stage='NATIVE_MODEL_OPTIMIZER_FULL_INPUT_LOADER_READY',steps=[],checkpoint_observations=[],optimizer_steps=0);save();check()
        expected={name+suffix for suffix in ['']+['_'+str(i) for i in range(5)] for name in ('loss_ce','loss_bbox','loss_fiou','loss_masks')}
        with modules['d2_events'].EventStorage(0) as storage:
            for i in range(32):
                storage.iter=i;check();batch=next(iterator)
                need(type(batch) is list and len(batch)==1 and batch[0]['health_dataset_index']==i
                     and batch[0]['image_id']==plan[i]['image_id'],'Actual canonical input order without replacement or skip')
                image=batch[0]['image'];instances=batch[0]['instances']
                need(image.dtype==torch.uint8 and image.ndim==3 and image.shape[0]==3 and max(image.shape[-2:])==512,'Original RGB mapper output')
                need(isinstance(instances,modules['structures'].Instances) and isinstance(instances.gt_masks,modules['structures'].BitMasks),'Original actual BitMasks targets')
                losses=model(batch);need(type(losses) is dict and set(losses)==expected,'24 original losses')
                values={}
                for k,v in losses.items():
                    need(torch.is_tensor(v) and v.numel()==1 and bool(torch.isfinite(v).all().item()),'Finite original loss '+k)
                    values[k]=float(v.detach().item())
                total=sum(losses.values());need(total.requires_grad and bool(torch.isfinite(total).item()),'Finite differentiable total')
                optimizer.zero_grad();total.backward();torch.cuda.synchronize(0);check()
                gradients=parameter_gradients(torch,model,check);g_m=boundary.observe()
                lrs=sorted({float(g['lr']) for g in optimizer.param_groups})
                step=dict(dataset_index=i,image_id=plan[i]['image_id'],original_annotations=len(plan[i]['ann_ids']),
                    instances_after_mapper=len(instances),mapped_image_shape=list(image.shape),losses=values,
                    total_loss=float(total.detach().item()),all_present_gradients_finite=True,learning_rates_before_step=lrs,
                    IAMR_output_gradient=g_m,mask_branch_gradients=gradients,optimizer_steps_completed=report['optimizer_steps'])
                monitor=(i+1) in PRELOCK['checkpoints']
                if monitor:
                    before=selected_update_snapshot(torch,model,optimizer)
                    report['checkpoint_observations'].append(dict(optimizer_updates_before_step=i,requested_optimizer_update=i+1,
                        stage='AFTER_ORIGINAL_BACKWARD_BEFORE_NATIVE_CLIP_AND_STEP',parameter_gradients=gradients,
                        IAMR_output_gradient=g_m));save()
                optimizer.step();torch.cuda.synchronize(0)
                report['optimizer_steps']+=1;step['optimizer_steps_completed']=report['optimizer_steps']
                scheduler.step();step['learning_rates_after_scheduler_step']=sorted({float(g['lr']) for g in optimizer.param_groups})
                if monitor:
                    observation=report['checkpoint_observations'][-1]
                    observation['selected_parameter_updates']=selected_update_observation(torch,model,before)
                    state_steps=[int(v['step'].item()) if torch.is_tensor(v['step']) else int(v['step']) for v in optimizer.state.values() if 'step' in v]
                    observation['native_optimizer_state_step_range']=None if not state_steps else [min(state_steps),max(state_steps)]
                    observation['stage']='ACTUAL_NATIVE_OPTIMIZER_UPDATE_COMPLETED'
                report['steps'].append(step);report.update(stage='NATIVE_OPTIMIZER_UPDATE_%02d_COMPLETE'%(i+1));save()
                boundary.reset();before=None;batch=None;losses=None;total=None;gradients=None;check()
        validate_completed_steps(report['steps'],report['optimizer_steps'])
        report.update(stage='NATIVE_HEALTH32_FINISHED',all32_optimizer_updates_completed=True,new_training=True,
            AP=False,DEV_evaluation=False,TEST=False,efficacy_comparison=False,formal_training_credit=0,
            checkpoint_saved=False,nonzero_mask_gradient_required=False,
            decay_comparison_limit='Deviation is not a unique fresh-gradient attribution: AdamW momentum and finite precision also affect updates.')
        save()
    finally:
        if boundary is not None:boundary.close()
        model=None;optimizer=None;scheduler=None;loader=None;iterator=None;boundary=None
        batch=None;losses=None;total=None;gradients=None;before=None;cfg=None;records=None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha',required=True);args=ap.parse_args()
    spec_path=Path(args.spec).resolve(strict=True);spec_ref=reference(str(spec_path));need(spec_ref['sha256']==args.spec_sha,'Actual spec SHA')
    spec=json.loads(spec_path.read_bytes());validate_spec(spec)
    out=Path(spec['output_directory']);need(not out.exists(),'Fresh output only');out.mkdir(mode=0o700)
    start=time.monotonic();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    report=dict(schema='DiffSAR-SSDD-native-train-health-report-v1',operation=OPERATION,run_id=spec['run_id'],
        status='STARTED',stage='ENTERED',qualified=False,Root_external_admission_required=True,
        pid=os.getpid(),pgid=os.getpgid(0),boot_id=boot,spec_ref=spec_ref,prelock=deepcopy(PRELOCK),
        source_refs=deepcopy(spec['source_refs']),library_sources=deepcopy(spec['library_sources']),
        config_ref=spec['config_ref'],input_manifest_ref=spec['input_manifest_ref'],annotations_ref=spec['annotations_ref'],
        compatibility_receipt_ref=spec['compatibility_receipt_ref'],initialization=PRELOCK['initialization'],
        AP=False,DEV_evaluation=False,TEST=False,efficacy_comparison=False,optimizer_steps=0,new_training=True,formal_training_credit=0,
        checkpoint_saved=False,model_weight_read=False,TGRS_success=False,GT_read_scope='CANONICAL_FIT741_1661_ONLY',
        whole_device_exclusive=False,exclusive_project_scientific_leaf=True,CPU_affinity=sorted(os.sched_getaffinity(0)))
    def save():
        report['elapsed_seconds']=time.monotonic()-start
        tmp=out/'REPORT.next'
        with tmp.open('w',encoding='utf-8') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,out/'REPORT.json')
    def check():
        import resource
        need(sorted(os.sched_getaffinity(0))==[0,1],'CPU01 affinity')
        free=available_ram();report['minimum_available_RAM_bytes']=min(free,report.get('minimum_available_RAM_bytes',free))
        need(free>=PRELOCK['free_RAM_floor_bytes'],'Free RAM floor')
        need(time.monotonic()-start<PRELOCK['external_deadline_seconds'],'Observed worker deadline')
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;report['worker_peak_RSS_bytes']=rss
        need(rss<=PRELOCK['peak_group_RSS_cap_bytes'],'Worker peak RSS bound')
    torch=None;code=1;modules={};refs=[];save()
    try:
        check();need(available_ram()>=PRELOCK['start_RAM_bytes'],'48GiB startup admission')
        refs=spec_refs(spec)
        for ref in refs:verify(ref)
        need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Selected own worker')
        lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());validate_lease(spec,lease);need(lease.get('boot_id')==boot,'Same-boot Root lease')
        manifest=json.loads(Path(spec['input_manifest_ref']['path']).read_bytes());plan=canonical_input_plan(manifest,spec['input_root'])
        unique={r['image_ref']['path']:r['image_ref'] for r in plan}
        for ref in unique.values():verify(ref);refs.append(ref)
        report.update(input_logical_samples=741,input_annotation_instances=1661,input_unique_JPEGs=730,
            numerical_image_source_refs=list(unique.values()),whole_FIT_byte_identity_verified=True)
        need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14','Locked platform')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Thread environment')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Only CUDA0 visible')
        adapter=types.ModuleType('_Root_DiffSAR_Health_Config_Adapter');adapter.__file__=spec['source_refs']['config_adapter']['path']
        exec(compile(Path(adapter.__file__).read_bytes(),adapter.__file__,'exec'),adapter.__dict__)
        modules={k:importlib.import_module(v) for k,v in LIBRARY_MODULES.items()};torch=modules['torch']
        need(torch.__version__=='2.10.0+cu130' and torch.version.cuda=='13.0' and modules['torchvision'].__version__=='0.25.0+cu130','Locked Torch/TV/CUDA')
        for k,m in modules.items():need(reference(m.__file__)==spec['library_sources'][k],'Actual selected import '+k)
        need(spec['library_sources']['tv_extension']['path'] in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries},'Actual Torchvision extension loaded')
        source_root=Path(modules['detectron2'].__file__).parent.parent
        need(Path(modules['diffsarshipinst'].__file__).parent.parent==source_root
             and Path(modules['train_net'].__file__).parent==source_root
             and Path(spec['config_ref']['path']).parent==source_root/'configs','Compatible native author source root')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One visible CUDA device')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,PRELOCK['GPU_allocator_cap_bytes']/props.total_memory),device=0)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        report.update(actual_library_refs={**{k:reference(m.__file__) for k,m in modules.items()},'tv_extension':reference(spec['library_sources']['tv_extension']['path'])},
            actual_environment=dict(python=platform.python_version(),torch=torch.__version__,torchvision=modules['torchvision'].__version__,
                cuda_build=torch.version.cuda,device=props.name,device_total_memory_bytes=props.total_memory,
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32),
            runtime_scope='SELECTED_SOURCE_REFS_NOT_ALL_TRANSITIVE_RUNTIME_OR_FULLY_ISOLATED_VENV');save()
        train_health(spec,plan,modules,adapter,report,save,check);check()
        for ref in refs:verify(ref)
        report.update(selected_sources_and_all_FIT_inputs_unchanged_after=True,status='NATIVE_HEALTH32_COMPLETE_PENDING_ROOT');code=0
    except BaseException as error:
        oom=bool(torch is not None and isinstance(error,torch.cuda.OutOfMemoryError))
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            failure_scope='RESOURCE_OOM_NOT_METHOD_INVALIDITY' if oom else 'EXECUTION_OR_HEALTH_FAILURE_NOT_EFFICACY',retry_permitted=False);code=1
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                gc.collect();torch.cuda.synchronize(0)
                clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None);need(callable(clear),'Workspace cleanup API')
                clear();torch.cuda.empty_cache()
                allocated=torch.cuda.memory_allocated(0);reserved=torch.cuda.memory_reserved(0);peak=torch.cuda.max_memory_allocated(0)
                report.update(CUDA_allocated_after_cleanup=allocated,CUDA_reserved_after_cleanup=reserved,CUDA_peak_allocated_bytes=peak)
                need(allocated==reserved==0 and peak<=PRELOCK['GPU_allocator_cap_bytes'],'CUDA final0/0 and peak8GiB')
            except BaseException as error:report.update(cleanup_error=str(error),status='FAILED_PRESERVED_NO_RETRY');code=1
        report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            CPU_group_deadline_birth_cleanup_requires_Root_supervisor=True);save()
        print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','elapsed_seconds','AP','formal_training_credit','optimizer_steps')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
