"""Resumable native DiffSAR full-train candidate; never executed by this builder."""
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


def need(c,m):
 if not c:raise ValueError(m)
def validate_segment(start,stop,mode):
 need(type(start) is int and type(stop) is int,'Integer update positions')
 need(mode in ('FORMAL','RESUME_VALIDATION'),'Declared trajectory purpose')
 need(0<=start<stop<=37050 and stop-start<=256,'Forward segment within fixed target')
 if mode=='RESUME_VALIDATION':need((start,stop) in ((0,8),(4,8)),'Only paired8 vs4+4 validation')
 return stop-start
def validate_checkpoint_header(c,start,mode,protocol):
 need(type(c) is dict and c.get('schema')=='DiffSAR-native-full-state-v1','Full native state checkpoint')
 need(c.get('protocol')==protocol and c.get('mode')==mode and c.get('updates')==start,'Same protocol/purpose/cursor')
 need(all(k in c for k in ('model','gradients','optimizer','scheduler','rng','loader')),'Weights alone are not a continuation')
 need(set(c['rng'])=={'python','numpy','torch_CPU','torch_CUDA'},'All numerical RNG streams')
 l=c['loader'];need(set(l)=={'consumed_indices','sampler_seed','MapDataset_rng','fallback_candidates'}
  and l['consumed_indices']==start and l['sampler_seed']==40244023,'Exact native sampler and mapper state')
def skip_native_indices(native_iterator,consumed):
 need(type(consumed) is int and consumed>=0,'Nonnegative consumed sample count')
 for _ in range(consumed):next(native_iterator)
 yield from native_iterator

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
        r['full_dataset_index']=row['dataset_index']
        ordered.append(r)
    return ordered


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
    cfg.SOLVER.MAX_ITER=37050;cfg.SOLVER.STEPS=(29640,35198);cfg.SOLVER.WARMUP_ITERS=20
    cfg.freeze();return cfg


SCHEMA='Root-DiffSAR-SSDD-native-full-train-spec-v1'
OPERATION='CUDA_DIFFSAR_SSDD_NATIVE_FULL_TRAIN_SEGMENT_FP32_V01'
LEASE_SCHEMA='Root-DiffSAR-SSDD-native-full-train-lease-v1'
LIBRARY_MODULES.update(d2_data_common='detectron2.data.common',
    d2_lr_scheduler='detectron2.solver.lr_scheduler',
    torch_dataloader='torch.utils.data.dataloader',torch_fetch='torch.utils.data._utils.fetch',
    d2_augmentation='detectron2.data.transforms.augmentation',
    d2_augmentation_impl='detectron2.data.transforms.augmentation_impl')
LIBRARY_NAMES=tuple(LIBRARY_MODULES)+('tv_extension',)
PRELOCK=dict(seed=40244023,sampler_seed=40244023,batch_size=1,dtype='float32',AMP=False,TF32=False,
    initialization='PUBLIC_CODE_CONSTRUCTION_AT_FORMAL0_OWN_PINNED_FULL_STATE_ON_RESUME_NO_PAPER_INIT_CERTIFICATION',
    fixed_target=37050,epoch_population=741,sampler='AUTHOR_NATIVE_TRAINING_SAMPLER_SHUFFLE_TRUE_FIXED_SEED',
    num_workers=0,input_format='RGB',augmentation='UNCHANGED_AUTHOR_TRAIN_MAPPER',
    author_commit='bdc3362372213bc8c3a1f6d5b7b74dc224d1b871',model='DiffSARShipInst',
    proposals=500,heads=6,num_classes=1,schedule_MAX_ITER=37050,schedule_STEPS=[29640,35198],
    schedule_WARMUP_ITERS=20,schedule_WARMUP_FACTOR=1.0,BASE_LR=3.5e-5,
    optimizer='AUTHOR_Trainer.build_optimizer',scheduler='AUTHOR_Trainer.build_lr_scheduler',
    CPU_affinity=[0,1],threads=2,GPU_allocator_cap_bytes=8*1024**3,
    peak_group_RSS_cap_bytes=16*1024**3,free_RAM_floor_bytes=16*1024**3,
    start_RAM_bytes=48*1024**3,external_deadline_seconds=1800,maximum_segment_updates=256,
    single_attempt=True,whole_device_exclusive=False,exclusive_project_scientific_leaf=True,
    endpoint_only_evaluation=True,automatic_retry=False,checkpoint_gradients=True)

def spec_refs(spec):
    refs=list(spec['source_refs'].values())+list(spec['library_sources'].values())+[
        spec[k] for k in ('config_ref','input_manifest_ref','annotations_ref','compatibility_receipt_ref','root_lease_ref')]
    if spec['resume_checkpoint_ref'] is not None:refs.append(spec['resume_checkpoint_ref'])
    return refs

def validate_spec(spec):
    need(type(spec) is dict and set(spec)=={'schema','operation','run_id','source_refs','library_sources',
        'config_ref','input_manifest_ref','annotations_ref','input_root','compatibility_receipt_ref',
        'root_lease_ref','output_directory','prelock','segment','resume_checkpoint_ref'},'Exact resumable spec fields')
    need(spec['schema']==SCHEMA and spec['operation']==OPERATION and spec['prelock']==PRELOCK,'Fixed operation/protocol')
    need(type(spec['run_id']) is str and spec['run_id'].startswith('DiffSAR-SSDD-native-full-v01-'),'Run identity')
    s=spec['segment'];need(set(s)=={'start','stop','mode'},'Segment cursor and purpose')
    validate_segment(s['start'],s['stop'],s['mode'])
    need((s['start']==0)==(spec['resume_checkpoint_ref'] is None),'Cold0 or explicit trusted own-state continuation')
    need(set(spec['source_refs'])=={'worker','config_adapter'} and set(spec['library_sources'])==set(LIBRARY_NAMES),'Selected scientific sources')
    refs=spec_refs(spec)
    for r in refs:valid_ref(r)
    need(len({r['path'] for r in refs})==len(refs),'Distinct selected refs')
    for key,pin in [('config_ref',SSDD_CONFIG_PIN),('input_manifest_ref',INPUT_MANIFEST_PIN),('annotations_ref',FIT_COCO_PIN)]:
        need(all(spec[key][k]==v for k,v in pin.items()),'Fixed '+key+' byte pin')
    need(all(spec['source_refs']['config_adapter'][k]==v for k,v in CONFIG_ADAPTER_PIN.items()),'Reviewed config adapter')
    need(all(spec['library_sources']['d2_extension'][k]==v for k,v in D2_EXTENSION_PIN.items()),'Author D2 binary')
    root=spec['input_root'];out=spec['output_directory']
    for p in (root,out):need(type(p) is str and PurePosixPath(p).is_absolute() and str(PurePosixPath(p))==p and '..' not in PurePosixPath(p).parts and p!='/','Canonical roots')
    need(spec['input_manifest_ref']['path']==root+'/manifest.json' and spec['annotations_ref']['path']==root+'/annotations/fit.json','Original data entries')
    need(out!=root and not root.startswith(out+'/') and not out.startswith(root+'/'),'Separate inputs/output')
    need(all(out!=r['path'] and not r['path'].startswith(out+'/') for r in refs),'Output outside sources')

def validate_lease(spec,lease):
    need(lease.get('schema')==LEASE_SCHEMA and lease.get('status')=='ROOT_GRANTED','Explicit Root lease')
    for k in ('operation','run_id','output_directory','prelock','input_root','segment','resume_checkpoint_ref'):
        need(lease.get(k)==spec[k],'Lease '+k)
    for k,v in {'worker_ref':spec['source_refs']['worker'],'config_adapter_ref':spec['source_refs']['config_adapter'],
        'config_ref':spec['config_ref'],'input_manifest_ref':spec['input_manifest_ref'],'annotations_ref':spec['annotations_ref']}.items():
        need(lease.get(k)==v,'Lease scientific identity '+k)
    need(lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
        and lease.get('exclusive_project_scientific_leaf') is True and lease.get('whole_device_exclusive') is False,'Single scientific leaf')

def scientific_pins(spec):
    groups={'sources':spec['source_refs'],'libraries':spec['library_sources'],
        'inputs':{k:spec[k] for k in ('config_ref','input_manifest_ref','annotations_ref')}}
    return {g:{k:{n:r[n] for n in ('bytes','sha256')} for k,r in rows.items()} for g,rows in groups.items()}

def cpu_copy(torch,value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):
        out=type(value)((k,cpu_copy(torch,v)) for k,v in value.items())
        if hasattr(value,'_metadata'):out._metadata=deepcopy(value._metadata)
        return out
    if isinstance(value,list):return [cpu_copy(torch,v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(torch,v) for v in value)
    return deepcopy(value)

def tree_digest(torch,value):
    """Exact typed CPU bytes; preserves original tensor dtype including float64 buffers."""
    h=hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous();h.update(('T'+str(a.dtype)+str(tuple(a.shape))).encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            h.update(('D'+type(v).__name__).encode())
            for k in sorted(v,key=lambda x:(type(x).__name__,repr(x))):walk(k);walk(v[k])
            if hasattr(v,'_metadata'):h.update(b'METADATA');walk(v._metadata)
        elif isinstance(v,(list,tuple)):
            h.update(('L' if isinstance(v,list) else 'U').encode())
            for x in v:walk(x)
            h.update(b'END')
        elif hasattr(v,'dtype') and hasattr(v,'shape') and hasattr(v,'tobytes'):
            h.update(('N'+str(v.dtype)+str(tuple(v.shape))).encode());h.update(v.tobytes())
        else:h.update((type(v).__name__+':'+repr(v)+';').encode())
    walk(value);return h.hexdigest()

def finite_tensor_tree(torch,value):
    count=0;numel=0
    def walk(v):
        nonlocal count,numel
        if torch.is_tensor(v):
            count+=1;numel+=v.numel()
            if v.is_floating_point() or v.is_complex():need(bool(torch.isfinite(v).all().item()),'Finite full checkpoint tensor')
        elif isinstance(v,dict):
            for x in v.values():walk(x)
        elif isinstance(v,(tuple,list)):
            for x in v:walk(x)
    walk(value);return dict(tensors=count,elements=numel,all_finite=True)

def capture_rng(modules):
    t=modules['torch'];return dict(python=random.getstate(),numpy=modules['numpy'].random.get_state(),
        torch_CPU=t.get_rng_state().clone(),torch_CUDA=[x.clone() for x in t.cuda.get_rng_state_all()])

def restore_rng(modules,state):
    random.setstate(state['python']);modules['numpy'].random.set_state(state['numpy'])
    modules['torch'].set_rng_state(state['torch_CPU']);modules['torch'].cuda.set_rng_state_all(state['torch_CUDA'])

def capture_loader(map_dataset,consumed):
    return dict(consumed_indices=consumed,sampler_seed=PRELOCK['sampler_seed'],
        MapDataset_rng=map_dataset._rng.getstate(),fallback_candidates=sorted(map_dataset._fallback_candidates))

def capture_state(modules,model,optimizer,scheduler,map_dataset,updates,mode,spec):
    t=modules['torch'];t.cuda.synchronize(0)
    c=dict(schema='DiffSAR-native-full-state-v1',protocol=deepcopy(PRELOCK),mode=mode,updates=updates,
        scientific_pins=scientific_pins(spec),model=cpu_copy(t,model.state_dict()),
        gradients={n:None if p.grad is None else cpu_copy(t,p.grad) for n,p in model.named_parameters()},
        optimizer=cpu_copy(t,optimizer.state_dict()),scheduler=cpu_copy(t,scheduler.state_dict()),
        rng=capture_rng(modules),loader=capture_loader(map_dataset,updates))
    finite=finite_tensor_tree(t,{k:c[k] for k in ('model','gradients','optimizer','scheduler')})
    return c,finite

def component_digests(torch,c):
    return {k:tree_digest(torch,c[k]) for k in ('model','gradients','optimizer','scheduler','rng','loader')}

def write_checkpoint(modules,state,out):
    t=modules['torch'];target=out/('CKPT_UPDATE%06d.pt'%state['updates']);part=target.with_suffix('.pt.part')
    need(not target.exists() and not part.exists(),'Never overwrite checkpoints')
    with part.open('xb') as f:t.save(state,f);f.flush();os.fsync(f.fileno())
    os.rename(str(part),str(target))
    # IO and CPU staging are explicitly neutral to subsequent mathematical RNG streams.
    restore_rng(modules,state['rng'])
    return reference(str(target))

def load_training_state(modules,model,optimizer,scheduler,c,spec):
    t=modules['torch'];start=spec['segment']['start'];mode=spec['segment']['mode']
    validate_checkpoint_header(c,start,mode,PRELOCK)
    need(c['scientific_pins']==scientific_pins(spec),'Same full scientific code/input pins')
    model.load_state_dict(c['model'],strict=True);optimizer.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
    names={n for n,p in model.named_parameters()};need(set(c['gradients'])==names,'Every gradient buffer or None')
    for n,p in model.named_parameters():
        g=c['gradients'][n]
        need(g is None or g.shape==p.shape and g.dtype==p.dtype,'Gradient dtype and shape preserved')
        p.grad=None if g is None else g.to(device=p.device).clone()

def mapped_batch_digest(torch,batch):
    d=batch[0];ins=d['instances'];parts={'image':d['image'],'gt_boxes':ins.gt_boxes.tensor,
        'gt_classes':ins.gt_classes,'gt_masks':ins.gt_masks.tensor}
    return tree_digest(torch,parts)

def native_loader(modules,cfg,records,start):
    t=modules['torch'];native=modules['d2_sampler'].TrainingSampler(len(records),shuffle=True,seed=PRELOCK['sampler_seed'])
    need(native._rank==0 and native._world_size==1,'Single-process native sampler partition')
    class SkippedNativeSampler(t.utils.data.Sampler):
        def __init__(self):self.consumed=start;self.trace=[]
        def __iter__(self):
            for idx in skip_native_indices(iter(native),start):
                self.consumed+=1;self.trace.append(int(idx));yield int(idx)
    sampler=SkippedNativeSampler();mapper=modules['mapper'].DiffSARShipInstDatasetMapper(cfg,is_train=True)
    grouped=modules['d2_data_build'].build_detection_train_loader(dataset=records,mapper=mapper,
        sampler=sampler,total_batch_size=1,aspect_ratio_grouping=True,num_workers=0)
    need(isinstance(grouped,modules['d2_data_common'].AspectRatioGroupedDataset),'Native aspect grouped batch1')
    dl=grouped.dataset
    need(isinstance(dl.dataset,modules['d2_data_common'].ToIterableDataset),'Native sampler-to-iterable bridge')
    md=dl.dataset.dataset
    need(isinstance(md,modules['d2_data_common'].MapDataset),'Native MapDataset state')
    # Force DataLoader base_seed initialization BEFORE restoring numerical RNG.
    # No image is fetched or mapped by iter(dl) with num_workers=0.
    eager=iter(dl)
    regrouped=modules['d2_data_common'].AspectRatioGroupedDataset(eager,batch_size=1)
    return iter(regrouped),regrouped,md,sampler

def train_segment(spec,plan,modules,adapter,report,save,check):
    t=modules['torch'];np=modules['numpy'];model=None;optimizer=None;scheduler=None
    iterator=None;grouped=None;md=None;sampler=None;loaded=None;batch=None;losses=None;total=None;checkpoint=None
    try:
        cfg=configuration(modules,adapter,spec)
        random.seed(PRELOCK['seed']);np.random.seed(PRELOCK['seed']);t.manual_seed(PRELOCK['seed']);t.cuda.manual_seed_all(PRELOCK['seed'])
        model=modules['train_net'].Trainer.build_model(cfg);model.train()
        need(type(model).__name__=='DiffSARShipInst' and type(model.backbone).__name__=='SCJE_FPN'
            and model.num_proposals==500 and len(model.head.head_series)==6,'Unchanged full author model')
        optimizer=modules['train_net'].Trainer.build_optimizer(cfg,model)
        scheduler=modules['train_net'].Trainer.build_lr_scheduler(cfg,optimizer)
        need(len(optimizer.param_groups)==len({id(p) for p in model.parameters() if p.requires_grad}),'Native per-parameter groups')
        report.update(configuration_frozen=True,config_dump_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(),
            optimizer_class=type(optimizer).__name__,scheduler_class=type(scheduler).__name__,
            optimizer_group_count=len(optimizer.param_groups),optimizer_defaults=deepcopy(optimizer.defaults),
            model_parameter_numel=sum(p.numel() for p in model.parameters()),native_clipping_in_optimizer_step=True,
            scaled_schedule=dict(MAX_ITER=37050,STEPS=[29640,35198],WARMUP_ITERS=20,WARMUP_FACTOR=1.0),
            initial_model_weights_requested=False,paper_initialization_certified=False)
        start=spec['segment']['start'];stop=spec['segment']['stop'];mode=spec['segment']['mode']
        if start:
            verify(spec['resume_checkpoint_ref'])
            # Trusted private Root-produced checkpoint, bytes pinned before deserialization.
            loaded=t.load(spec['resume_checkpoint_ref']['path'],map_location='cpu',weights_only=False)
            load_training_state(modules,model,optimizer,scheduler,loaded,spec)
        records=modules['d2_coco'].load_coco_json(spec['annotations_ref']['path'],spec['input_root']+'/by_name',
            dataset_name='Root_DiffSAR_FIT741_native_full_v01',extra_annotation_keys=['id'])
        records=align_coco_records(records,plan)
        need(all(a['category_id']==0 for r in records for a in r['annotations']),'Original single-class mapping')
        iterator,grouped,md,sampler=native_loader(modules,cfg,records,start)
        if loaded is not None:
            md._rng.setstate(loaded['loader']['MapDataset_rng']);md._fallback_candidates=set(loaded['loader']['fallback_candidates'])
            restore_rng(modules,loaded['rng'])
            actual,_=capture_state(modules,model,optimizer,scheduler,md,start,mode,spec)
            expected=component_digests(t,loaded);observed=component_digests(t,actual)
            need(expected==observed,'Exact resume-start model/grad/optimizer/scheduler/RNG/MapDataset state')
            report.update(resume_start_exact=True,resume_start_component_digests=observed)
            actual=None;loaded=None
        else:report.update(resume_start_exact=None)
        report.update(stage='FULL_NATIVE_LOADER_AND_STATE_READY',steps=[],checkpoints=[],optimizer_steps_completed=start,
            samples_consumed_before_segment=start,shuffle=True,sampler_seed=PRELOCK['sampler_seed'],
            resume_loader_order='EAGER_DATALOADER_ITERATOR_THEN_FULL_RNG_RESTORE_BEFORE_ANY_MAPPING',
            gradient_buffers_restored_on_resume=bool(start));save();check()
        expected_losses={name+suffix for suffix in ['']+['_'+str(i) for i in range(5)] for name in ('loss_ce','loss_bbox','loss_fiou','loss_masks')}
        with modules['d2_events'].EventStorage(start) as storage:
            for update in range(start,stop):
                storage.iter=update;check();batch=next(iterator)
                need(type(batch) is list and len(batch)==1 and sampler.consumed==update+1,'No native loader prefetch beyond actual batch1')
                idx=sampler.trace[-1];need(batch[0]['full_dataset_index']==idx and batch[0]['image_id']==plan[idx]['image_id'],'Actual native shuffled sample identity')
                need(all(not bucket for bucket in grouped._buckets),'Empty native aspect buckets at atomic boundary')
                ins=batch[0]['instances'];need(isinstance(ins.gt_masks,modules['structures'].BitMasks),'Native actual BitMasks')
                augmentation_digest=mapped_batch_digest(t,batch)
                losses=model(batch);need(type(losses) is dict and set(losses)==expected_losses,'Original24 losses')
                values={k:float(v.detach().item()) for k,v in losses.items()}
                need(all(math.isfinite(v) for v in values.values()),'Finite original losses')
                total=sum(losses.values());need(total.requires_grad and bool(t.isfinite(total).item()),'Finite original total')
                optimizer.zero_grad();total.backward();optimizer.step();t.cuda.synchronize(0)
                lrs=sorted({float(g['lr']) for g in optimizer.param_groups});scheduler.step()
                report['optimizer_steps_completed']=update+1
                report['steps'].append(dict(global_update=update+1,dataset_index=idx,image_id=plan[idx]['image_id'],
                    mapped_image_shape=list(batch[0]['image'].shape),mapped_instances=len(ins),
                    augmentation_and_target_sha256=augmentation_digest,losses=values,total_loss=float(total.detach().item()),
                    used_learning_rates=lrs,post_scheduler_learning_rates=sorted({float(g['lr']) for g in optimizer.param_groups})))
                report['stage']='ACTUAL_UPDATE_%06d_COMPLETE'%(update+1);save();check()
                if update+1==stop or mode=='RESUME_VALIDATION' and update+1==4:
                    need(sampler.consumed==update+1 and all(not x for x in grouped._buckets),'Atomic no-prefetch checkpoint boundary')
                    checkpoint,finite=capture_state(modules,model,optimizer,scheduler,md,update+1,mode,spec)
                    digests=component_digests(t,checkpoint);cref=write_checkpoint(modules,checkpoint,Path(spec['output_directory']))
                    report['checkpoints'].append(dict(updates=update+1,ref=cref,component_digests=digests,finite_tensor_state=finite,
                        gradients_saved=True,original_dtype_buffers_preserved=True,sampler_consumed=update+1,aspect_buckets_empty=True))
                    checkpoint=None;save();check()
                batch=None;losses=None;total=None
        need(report['optimizer_steps_completed']==stop and len(report['steps'])==stop-start,'All requested actual updates')
        report.update(stage='NATIVE_FULL_SEGMENT_COMPLETE',all_requested_updates_completed=True,
            samples_consumed_after_segment=sampler.consumed,new_optimizer_updates=stop-start,
            formal_training_credit=(stop-start if mode=='FORMAL' else 0),
            validation_training_credit=(stop-start if mode=='RESUME_VALIDATION' else 0),AP=False,DEV_evaluation=False,TEST=False,
            trajectory_end_component_digests=report['checkpoints'][-1]['component_digests']);save()
    finally:
        model=None;optimizer=None;scheduler=None;iterator=None;grouped=None;md=None;sampler=None
        loaded=None;batch=None;losses=None;total=None;checkpoint=None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--spec-sha',required=True);args=ap.parse_args()
    sp=Path(args.spec).resolve(strict=True);sref=reference(str(sp));need(sref['sha256']==args.spec_sha,'Actual spec SHA')
    spec=json.loads(sp.read_bytes());validate_spec(spec)
    out=Path(spec['output_directory']);need(not out.exists(),'Fresh output');out.mkdir(mode=0o700)
    start=time.monotonic();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    report=dict(schema='DiffSAR-SSDD-native-full-train-segment-report-v1',operation=OPERATION,run_id=spec['run_id'],
        status='STARTED',stage='ENTERED',qualified=False,Root_external_admission_required=True,
        pid=os.getpid(),pgid=os.getpgid(0),boot_id=boot,spec_ref=sref,prelock=deepcopy(PRELOCK),
        source_refs=deepcopy(spec['source_refs']),library_sources=deepcopy(spec['library_sources']),
        config_ref=spec['config_ref'],input_manifest_ref=spec['input_manifest_ref'],annotations_ref=spec['annotations_ref'],
        compatibility_receipt_ref=spec['compatibility_receipt_ref'],segment=deepcopy(spec['segment']),
        resume_checkpoint_ref=spec['resume_checkpoint_ref'],AP=False,DEV_evaluation=False,TEST=False,
        efficacy_comparison=False,TGRS_success=False,new_training=True,formal_training_credit=0,
        optimizer_steps_completed=spec['segment']['start'],steps=[],checkpoints=[],whole_device_exclusive=False,
        exclusive_project_scientific_leaf=True,CPU_affinity=sorted(os.sched_getaffinity(0)))
    def save():
        report['elapsed_seconds']=time.monotonic()-start;tmp=out/'REPORT.next'
        with tmp.open('w',encoding='utf-8') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,out/'REPORT.json')
    def check():
        import resource
        need(sorted(os.sched_getaffinity(0))==[0,1],'CPU01 affinity')
        free=available_ram();report['minimum_available_RAM_bytes']=min(free,report.get('minimum_available_RAM_bytes',free))
        need(free>=PRELOCK['free_RAM_floor_bytes'],'RAM floor')
        need(time.monotonic()-start<PRELOCK['external_deadline_seconds'],'Finite worker deadline')
        peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;report['worker_peak_RSS_bytes']=peak
        need(peak<=PRELOCK['peak_group_RSS_cap_bytes'],'Worker RSS bound')
    torch=None;code=1;modules={};refs=[];save()
    try:
        check();need(available_ram()>=PRELOCK['start_RAM_bytes'],'48GiB startup admission')
        refs=spec_refs(spec)
        for ref in refs:verify(ref)
        need(reference(str(Path(__file__).resolve()))==spec['source_refs']['worker'],'Actual own worker')
        lease=json.loads(Path(spec['root_lease_ref']['path']).read_bytes());validate_lease(spec,lease);need(lease.get('boot_id')==boot,'Same-boot Root lease')
        manifest=json.loads(Path(spec['input_manifest_ref']['path']).read_bytes());plan=canonical_input_plan(manifest,spec['input_root'])
        unique={r['image_ref']['path']:r['image_ref'] for r in plan}
        for ref in unique.values():verify(ref);refs.append(ref)
        report.update(input_logical_samples=741,input_annotation_instances=1661,input_unique_JPEGs=730,
            numerical_image_source_refs=list(unique.values()),whole_FIT_byte_identity_verified=True)
        need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14','Locked platform')
        need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Thread env')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('PYTHONHASHSEED')==str(PRELOCK['seed']),'CUDA0 and startup hash seed')
        adapter=types.ModuleType('_Root_DiffSAR_Full_Config_Adapter');adapter.__file__=spec['source_refs']['config_adapter']['path']
        exec(compile(Path(adapter.__file__).read_bytes(),adapter.__file__,'exec'),adapter.__dict__)
        modules={k:importlib.import_module(v) for k,v in LIBRARY_MODULES.items()};torch=modules['torch']
        need(torch.__version__=='2.10.0+cu130' and torch.version.cuda=='13.0' and modules['torchvision'].__version__=='0.25.0+cu130','Locked runtime')
        for k,m in modules.items():need(reference(m.__file__)==spec['library_sources'][k],'Actual selected import '+k)
        need(spec['library_sources']['tv_extension']['path'] in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries},'Actual TV extension')
        root=Path(modules['detectron2'].__file__).parent.parent
        need(Path(modules['diffsarshipinst'].__file__).parent.parent==root and Path(modules['train_net'].__file__).parent==root
            and Path(spec['config_ref']['path']).parent==root/'configs','Compatible author root')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA device')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,PRELOCK['GPU_allocator_cap_bytes']/props.total_memory),device=0)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        report.update(actual_library_refs={**{k:reference(m.__file__) for k,m in modules.items()},'tv_extension':reference(spec['library_sources']['tv_extension']['path'])},
            actual_environment=dict(python=platform.python_version(),torch=torch.__version__,torchvision=modules['torchvision'].__version__,
                cuda_build=torch.version.cuda,device=props.name,device_total_memory_bytes=props.total_memory,
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                PYTHONHASHSEED=os.environ['PYTHONHASHSEED'],TF32=False),
            runtime_scope='SELECTED_SOURCE_REFS_NOT_ALL_TRANSITIVE_RUNTIME_OR_FULLY_ISOLATED_VENV');save()
        train_segment(spec,plan,modules,adapter,report,save,check);check()
        for ref in refs:verify(ref)
        report.update(selected_sources_and_all_FIT_inputs_unchanged_after=True,status='NATIVE_FULL_SEGMENT_COMPLETE_PENDING_ROOT');code=0
    except BaseException as error:
        oom=bool(torch is not None and isinstance(error,torch.cuda.OutOfMemoryError))
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            failure_scope='RESOURCE_OOM_NOT_METHOD_INVALIDITY' if oom else 'EXECUTION_OR_STATE_FAILURE_NOT_EFFICACY',retry_permitted=False);code=1
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                gc.collect();torch.cuda.synchronize(0)
                clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None);need(callable(clear),'Workspace cleanup');clear()
                torch.cuda.empty_cache();allocated=torch.cuda.memory_allocated(0);reserved=torch.cuda.memory_reserved(0)
                peak=torch.cuda.max_memory_allocated(0)
                report.update(cuBLAS_workspaces_cleared=True,CUDA_allocated_after_cleanup=allocated,
                    CUDA_reserved_after_cleanup=reserved,CUDA_peak_allocated_bytes=peak)
                need(allocated==reserved==0 and peak<=PRELOCK['GPU_allocator_cap_bytes'],'FinalCUDA0/0 and peak8GiB')
            except BaseException as error:report.update(cleanup_error=str(error),status='FAILED_PRESERVED_NO_RETRY');code=1
        report.update(worker_exit_code=code,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            CPU_group_deadline_birth_cleanup_requires_Root_supervisor=True);save()
        print(json.dumps({k:report[k] for k in ('status','worker_exit_code','qualified','elapsed_seconds','AP','formal_training_credit','optimizer_steps_completed')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
