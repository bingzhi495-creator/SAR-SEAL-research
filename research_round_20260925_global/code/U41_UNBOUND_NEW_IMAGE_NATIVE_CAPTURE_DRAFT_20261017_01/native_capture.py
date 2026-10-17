"""Unbound single-image native capture; no entry, model loading or qualification.

Imports are stdlib only. Torch/NumPy, the already loaded frozen teacher and all
numerical dependencies are explicit arguments. No annotation/parser dependency.
The original training=True image preprocessing uses its data_samples=None guard;
no pad-mask flag, weights, scoring rule or interpolation operator is changed.
"""
import inspect
import math
from pathlib import Path


PINS={
    'observer':'d7aa8e7d504a128b54f0793c4178fbfc41a51dafcf3574058aa5f2f7cda95645',
    'renderer':'dd20f440e11b6890f0f797b8f048f6953adf2a8ee8c33e74cfe91874df78018b',
    'exporter':'114af52ba6f64e4fed3e93f74177d8920bccca0b5a84aaef692a7b1854f611d6',
    'formatting':'14f3d8dbf9189d7b97e47a9453b97aad67aa3cfd66f69a5c2cd8e330649ba16a',
    'data_preprocessor':'180d3dfeddfef1fcc6a6dacbcaa2896f9d2e068b4314e92f29d4ae81e82e1551',
    'base_preprocessor':'2c14ccdbf2970b2759d174b7c383f39cdf33d9b7619cb5508204c6c33008dd4d',
}
META_KEYS=('img_id','img_path','ori_shape','img_shape','scale_factor')
TARGET_FIELDS=('gt_instances','ignored_instances','gt_sem_seg','gt_panoptic_seg','proposals')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'},'Strict ref shape required')
    need(type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict ref identity required')
    return value


def validate_preprocessing(config):
    """Require original resolved values, never fill photometric/padding defaults."""
    required={'type','mean','std','bgr_to_rgb','pad_size_divisor','pad_mask','pad_value','batch_augments'}
    need(type(config) is dict and required<=set(config),'Missing explicit original preprocessing values')
    need(config['type']=='DetDataPreprocessor','Original DetDataPreprocessor required')
    for key in ('mean','std'):
        values=config[key]
        need(type(values) in (list,tuple) and len(values)==3
             and all(type(x) in (int,float) and math.isfinite(x) for x in values),'Explicit three-channel '+key)
    need(all(x>0 for x in config['std']) and type(config['bgr_to_rgb']) is bool,'Invalid photometry')
    need(config['pad_size_divisor']==32 and config['pad_mask'] is True
         and config['pad_value']==0,'Original divisor/pad-mask/image-pad recipe required')
    aug=config['batch_augments']
    need(type(aug) in (list,tuple) and len(aug)==1 and type(aug[0]) is dict,'One original fixed-pad augment')
    value=aug[0]
    required_aug={'type','size','img_pad_value','pad_mask','mask_pad_value','pad_seg'}
    need(required_aug<=set(value) and value['type']=='BatchFixedSizePad'
         and tuple(value['size'])==(800,800) and value['img_pad_value']==0
         and value['pad_mask'] is True and value['mask_pad_value']==0 and value['pad_seg'] is False,
         'Original fixed800 padding config missing or changed')
    return config


def pack_image(metadata,modules,closed_ref):
    """Only whitelisted image identity enters Compose; no annotations are inspected."""
    need(type(metadata) is dict and set(metadata)=={'image_id','image_ref','original_shape'},
         'Image-only metadata exact keys required; annotations are not accepted')
    need(type(metadata['image_id']) is int and metadata['image_id']>=0,'Integer image ID required')
    shape=metadata['original_shape']
    need(type(shape) in (list,tuple) and len(shape)==2
         and all(type(x) is int and x>0 for x in shape),'Original H/W required')
    reference=strict_ref(metadata['image_ref']);path=str(closed_ref(reference))
    need(Path(path).resolve()==Path(reference['path']).resolve(),'Closed original image path changed')
    pipeline=modules['Compose']([dict(type='LoadImageFromFile',to_float32=True),
                                dict(type='Resize',scale=(800,800),keep_ratio=True)])
    transformed=pipeline(dict(img_id=metadata['image_id'],img_path=path))
    need(transformed is not None and tuple(transformed['ori_shape'][:2])==tuple(shape),'Original image dimensions differ')
    hw=tuple(transformed['img_shape'][:2])
    need(len(hw)==2 and all(type(x) is int and 0<x<=800 for x in hw),'Actual resize dimensions invalid')
    need(not any(k.startswith('gt_') or k in ('instances','ann_info','proposals') for k in transformed),
         'Image pipeline returned target/proposal fields')
    packed=modules['PackDetInputs'](meta_keys=META_KEYS)(transformed)
    identity=dict(image_id=metadata['image_id'],image=dict(reference),original_shape=list(shape),actual_img_shape=list(hw))
    return packed,identity


def preprocess_images_only(model,packed,identity,modules):
    """Use the unchanged library image branch; never pass a target container to it."""
    pre=model.data_preprocessor
    batch=modules['pseudo_collate']([{'inputs':packed['inputs']}])
    batch['data_samples']=None
    original_pad_shapes=pre._get_pad_shape(batch)
    need(len(original_pad_shapes)==1,'One original requested image')
    pre_pad=tuple(original_pad_shapes[0]); aug=pre.batch_augments[0]
    data=pre(batch,training=True)
    need(data['data_samples'] is None,'Original None sample guard was not retained')
    actual=tuple(data['inputs'].shape[-2:])
    # DetDataPreprocessor sets both fields before augment. FixedSizePad updates
    # both iff its input needs padding; its early return preserves divisor shape.
    if pre_pad[0]>=aug.size[0] and pre_pad[1]>=aug.size[1]:
        expected=pre_pad
    else:
        expected=tuple(aug.size)
    need(actual==expected==(800,800),'Original fixed800 output/metadata branch differs')
    sample=packed['data_samples']
    for name in TARGET_FIELDS:
        if name in sample:delattr(sample,name)
    sample.set_metainfo(dict(batch_input_shape=actual,pad_shape=expected))
    need(not any(name in sample for name in TARGET_FIELDS),'Target fields survived metadata-only sample stripping')
    need(tuple(sample.ori_shape[:2])==tuple(identity['original_shape'])
         and tuple(sample.img_shape[:2])==tuple(identity['actual_img_shape']),'Packed image geometry changed')
    identity.update(batch_input_shape=list(actual),pad_shape=list(expected),
                    scale_factor=[float(x) for x in sample.scale_factor],native_input_gamma=1.)
    return data,[sample],dict(pre_fixed_pad_shape=list(pre_pad),final_input_shape=list(actual),
                            training_flag=True,data_samples_passed_to_preprocessor=None,
                            target_padding_branch_executed=False)


def select_native_fields(observed,classes,mask_stages):
    """Natural-axis references only; h/pre-mask-embedding cannot substitute e."""
    need('e' in observed and len(observed['F'])==len(observed['output'])==1
         and len(observed['e'])==len(classes)==len(mask_stages)==10,'Actual ten-stage output capture required')
    captured_classes,captured_masks=observed['output'][0]
    need(captured_classes is classes and captured_masks is mask_stages,'Captured actual head output differs')
    return dict(F=observed['F'][0],e=observed['e'][-1],M=mask_stages[-1],C=classes[-1],
                natural_query_ids=list(range(100)))


def _verify_dependencies(model,modules,source_refs,closed_ref):
    objects={key:modules[key] for key in ('observer','renderer','exporter','pref','masks')}
    objects.update(formatting=modules['PackDetInputs'],Compose=modules['Compose'],pseudo_collate=modules['pseudo_collate'],
                   data_preprocessor=type(model.data_preprocessor))
    bases=[kind for kind in type(model.data_preprocessor).__mro__ if kind.__name__=='ImgDataPreprocessor']
    need(len(bases)==1,'Original ImgDataPreprocessor base required');objects['base_preprocessor']=bases[0]
    need(set(source_refs)==set(objects),'Explicit numerical dependency source refs required')
    for role,obj in objects.items():
        reference=strict_ref(source_refs[role])
        if role in PINS:need(reference['sha256']==PINS[role],'Original source pin differs: '+role)
        actual=inspect.getsourcefile(obj) if inspect.isclass(obj) or inspect.isfunction(obj) else obj.__file__
        need(Path(actual).resolve()==Path(closed_ref(reference)).resolve(),'Actual loaded dependency differs: '+role)
    need(type(model.data_preprocessor).__name__=='DetDataPreprocessor','Original preprocessor class required')


def _verify_loaded_preprocessing(torch,model,config):
    validate_preprocessing(config);pre=model.data_preprocessor
    need(pre._enable_normalize is True and pre._channel_conversion is config['bgr_to_rgb'],
         'Loaded channel/normalize rule differs from original config')
    need('rgb_to_bgr' in config and config['rgb_to_bgr'] is False,
         'Resolved original rgb_to_bgr=False must be explicit, not guessed')
    for name in ('mean','std'):
        actual=getattr(pre,name)
        expected=torch.tensor(config[name],dtype=actual.dtype,device=actual.device).view(3,1,1)
        need(torch.equal(actual,expected),'Loaded original '+name+' differs')
    for name in ('pad_size_divisor','pad_value','pad_mask'):
        need(getattr(pre,name)==config[name],'Loaded preprocessing field differs: '+name)
    need(pre.pad_seg is False and len(pre.batch_augments)==1,'Unexpected semantic/random batch augment')
    aug=pre.batch_augments[0];declared=config['batch_augments'][0]
    need(type(aug).__name__=='BatchFixedSizePad','Original fixed pad class required')
    for name in ('size','img_pad_value','pad_mask','mask_pad_value','pad_seg'):
        actual=getattr(aug,name);expected=declared[name]
        need(tuple(actual)==tuple(expected) if name=='size' else actual==expected,'Loaded fixed padding differs: '+name)


def capture_native_image(torch,np,model,image_metadata,*,modules,source_refs,preprocessing,
                         provenance,closed_ref,check):
    """One future admitted CUDA0/FP32 image; returns caller-owned natural tensors.

    The caller owns loaded teacher validation, CUDA/resource bounds, lifetime and
    release. This function never constructs a model, loads a checkpoint, chooses
    a population, performs training, publishes an artifact or qualifies itself.
    """
    check();validate_preprocessing(preprocessing)
    _verify_dependencies(model,modules,source_refs,closed_ref)
    need(type(provenance) is dict and set(provenance)=={'teacher_checkpoint','loading_receipt','config_sources',
         'teacher_state_proof','resolved_preprocessing'},'Explicit loaded-teacher provenance required')
    strict_ref(provenance['teacher_checkpoint']);strict_ref(provenance['loading_receipt'])
    need(provenance['resolved_preprocessing']==preprocessing and provenance['config_sources'],
         'Original resolved config provenance missing')
    for ref in provenance['config_sources']:closed_ref(strict_ref(ref))
    _verify_loaded_preprocessing(torch,model,preprocessing)
    need(all(not module.training for module in model.modules())
         and all(not p.requires_grad and p.grad is None and p.dtype==torch.float32
                 and p.device.type=='cuda' and p.device.index==0 for p in model.parameters()),
         'Already frozen eval FP32 CUDA0 teacher required')
    need(model.panoptic_head.num_queries==100 and model.panoptic_head.num_classes==1,'Original single-class100 teacher required')
    pref=modules['pref'];before=pref.state_proof(model.state_dict())
    need(before==provenance['teacher_state_proof'],'Loaded teacher differs from caller verified state proof')
    outside=pref.rng_snapshot(torch);success=False
    try:
        packed,identity=pack_image(image_metadata,modules,closed_ref)
        with torch.no_grad():
            data,samples,padding=preprocess_images_only(model,packed,identity,modules)
            need(tuple(data['inputs'].shape)==(1,3,800,800) and data['inputs'].dtype==torch.float32
                 and bool(torch.isfinite(data['inputs']).all()),'Actual original normalized input differs')
            input_digest=pref.digest_tensor(data['inputs'])
            with torch.random.fork_rng(devices=[0]):
                seed=100000+identity['image_id']
                torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
                features=model.extract_feat(data['inputs'])
                with modules['observer'].observed_fields(torch,model.panoptic_head) as observed:
                    classes,mask_stages=model.panoptic_head(features,samples)
                fields=select_native_fields(observed,classes,mask_stages)
                L=modules['renderer'].render_original(torch,fields['M'],identity)
                need(tuple(L.shape)==(100,)+tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),'Original100 L shape/finite')
                predictions,factors,qids,class_packet=modules['exporter'].export_original(
                    torch,np,modules['masks'],model.panoptic_fusion_head,fields['C'],L,identity['image_id'],pref)
                need(type(qids) is list and len(qids)==100 and all(type(q) is int for q in qids)
                     and sorted(qids)==list(range(100)),'Export must be a permutation of natural100')
                need([row['query_id'] for row in predictions]==qids
                     and [row['query_id'] for row in factors]==qids
                     and all(row['category_id']==1 for row in predictions),'Native export query/category alignment')
                # Original N07 check against complete fusion.predict, including
                # padding/crop/rescale, not merely a duplicated scoring formula.
                padded=torch.nn.functional.interpolate(fields['M'],size=(800,800),mode='bilinear',align_corners=False)
                actual=model.panoptic_fusion_head.predict(fields['C'],padded,samples,rescale=True)[0]['ins_results']
                binary=L[qids]>0
                scores=torch.tensor([row['score'] for row in predictions],dtype=actual.scores.dtype,device=actual.scores.device)
                need(len(actual)==100 and torch.equal(actual.masks,binary)
                     and torch.equal(actual.scores,scores) and bool((actual.labels==0).all()),'Original whole fusion calibration differs')
                by_query=[None]*100
                for row in factors:by_query[row['query_id']]=dict(row)
                fields.update(L=L,identity=identity,export_query_ids=qids,native_predictions=predictions,
                              S0_by_query=[row['official_score'] for row in by_query],
                              score_factors_by_query=by_query,class_packet=class_packet,
                              category_ids_by_query=[1]*100)
        check();success=True
        fields['receipt']=dict(schema='U41-unbound-single-image-native-capture-v1',qualified=False,
            teacher_checkpoint=provenance['teacher_checkpoint'],loading_receipt=provenance['loading_receipt'],
            config_sources=provenance['config_sources'],resolved_preprocessing=preprocessing,source_refs=source_refs,
            input_digest=input_digest,padding=padding,images=1,queries=100,natural_query_ids=list(range(100)),
            e_is_mask_embed_output=True,teacher_forward_calls=1,annotations_opened=False,GT_forwarded=False,
            loss_or_matching_called=False,optimizer_created=False,new_training=False,
            original_fusion_masks_scores_labels_exact=True,score_cutoff=None,NMS=False,
            population_qualification=False,numerical_runtime_qualified=False)
        return fields
    finally:
        need(before==pref.state_proof(model.state_dict()),'Frozen teacher state changed')
        after=pref.rng_snapshot(torch)
        need(len(outside)==len(after) and all(torch.equal(x,y) for x,y in zip(outside,after)),'External RNG changed')
        need(all(p.grad is None for p in model.parameters()),'Frozen teacher gained gradients')
        if success:check()
