"""Inert fixed C endpoint DEV evaluation SOURCE. Never executes at authoring.
Direct PointHead logits; own official quality is primary, fixed own native scores diagnostic.
No faithful PointRend, GPU fit, efficacy or numerical qualification claim.
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback
import types

PREFIX='query-pixel-PointRend-style-fixed-endpoint-evaluation'
SCHEMA='ROOT-'+PREFIX+'-spec-v1'
TRAIN_PREFIX='query-pixel-PointRend-style-native-integration'
TRAIN_STATUS='ROOT_QUERY_PIXEL_POINTREND_STYLE_NATIVE_INTEGRATION_CPU_QUALIFIED'
CONTROLS=('C_native_official','C_direct_official','C_direct_native_score')
MODES=('native0','global','spatial_halves','spatial_reversed')
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
DEV_SHA='5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55'
NUMERIC=dict(class_rtol=1e-6,class_atol=1e-6,renderer_rtol=1e-4,renderer_atol=1e-4,
    quality_rtol=1e-4,quality_atol=1e-4,threshold_near_zero_rtol=1e-4,threshold_near_zero_atol=1e-4)
STAGES={'first24':dict(initial_native_epoch=24,phase_epochs=24,native_epoch=48,head_epoch=24,
    microsteps=17784,updates=8892,GT_visits=39864),
    'warm48':dict(initial_native_epoch=48,phase_epochs=48,native_epoch=96,head_epoch=72,
    microsteps=35568,updates=17784,GT_visits=79728)}
TRAIN_ROLES=('training_entry_source','training_CPU_reader_source','training_spec',
    'training_CPU_qualification','training_CPU_admission','training_CPU_receipt','training_report','training_receipt','trained_checkpoint')
PURE_ROLES=('official_export_source','renderer_source','point_head_source','point_adapter_source',
    'native_loader_source','evaluation_source','micro_preflight_source','initializer_source',
    'training_source','native_helpers','training_CPU_reader','reused_CPU_reader')
HEAD_NAMES=('fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias','predictor.weight','predictor.bias')

def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(),'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'} and Path(ref['path']).is_absolute() and
        type(ref['bytes'])is int and ref['bytes']>0 and len(ref['sha256'])==64 and
        set(ref['sha256'])<=set('0123456789abcdef') and reference(ref['path'])==ref,'Pinned input differs')
    return Path(ref['path'])


def load(ref,name):
    path=bound(ref); code=path.read_bytes()
    require(path.suffix=='.py' and len(code)==ref['bytes'] and hashlib.sha256(code).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(code,str(path),'exec'),module.__dict__)
    return module


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def terminal(raw,phase,receipt_path):
    require(type(phase) is str and bool(phase), 'STOP before hashes/imports: nonempty phase required')
    receipt = Path(receipt_path)
    require(receipt.is_absolute() and receipt.name == 'PARENT_RECEIPT.json' and receipt.parent.name == phase,
        'STOP before hashes/imports: explicit original phase-bound receipt path required')
    digest = raw.get('admission_sha256')
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
        'STOP before hashes/imports: lowercase SHA256 admission digest required')
    require('phase' not in raw or (type(raw['phase']) is str and raw['phase'] == phase),
        'STOP before hashes/imports: supplied RAW phase differs')
    require(raw.get('status')=='ACTUAL_OWNED_TERMINAL' and type(raw.get('original_identity'))is dict and bool(raw['original_identity']) and
        type(raw.get('returncode'))is int and raw['returncode']==0 and raw.get('root_wait_reaped')is True and
        raw.get('held_handle_exit')==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0) and
        type(raw['held_handle_exit']['actual_exit_code'])is int and
        raw.get('numeric_birth_absence',{}).get('original_birth_absent')is True and
        all(k in raw and raw[k]is None for k in ('abort','error')) and raw.get('wait_errors')is False and
        raw.get('GPU_observation_errors')==[],'STOP before hashes/imports: actual raw0 required')
    peak=raw.get('whole_GPU_observed_peak_MiB')
    require(type(peak)in(int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        raw.get('whole_GPU_used_limit_MiB')==8192 and raw.get('whole_leaf_VRAM_certified')is False and
        raw.get('GPU_scope')=='whole GPU observed, not per-process WDDM allocation','Actual raw resource scope differs')


def observed_inference(torch,head,call):
    """Two passive hooks observe detached inference F/e; native C/M are returned."""
    before={name:(tuple(module._forward_pre_hooks.items()),tuple(module._forward_hooks.items()))for name,module in head.named_modules()}
    cache=dict(F=[],e=[]); handles=[]
    def feature(module,args,result):
        require(isinstance(result,(tuple,list))and len(result)==2,'Native pixel decoder two outputs required')
        cache['F'].append(result[0]); return None
    def embedding(module,args,result):
        cache['e'].append(result); return None
    try:
        handles.append(head.pixel_decoder.register_forward_hook(feature))
        handles.append(head.mask_embed.register_forward_hook(embedding))
        classes,masks=call()
        require(len(cache['F'])==1 and len(cache['e'])==len(classes)==len(masks)==10,'One inference forward/ten stages required')
        F,e,M,C=cache['F'][0],cache['e'][-1],masks[-1],classes[-1]
        for value,shape in ((F,(1,256,200,200)),(e,(1,100,256)),(M,(1,100,200,200)),(C,(1,100,2))):
            require(isinstance(value,torch.Tensor)and value.dtype==torch.float32 and tuple(value.shape)==shape and
                not value.requires_grad and bool(torch.isfinite(value).all()),'Observed finite no-grad native fields required')
        require(all(v.dtype==torch.float32 and tuple(v.shape)==(1,100,2)and not v.requires_grad and bool(torch.isfinite(v).all())for v in classes)and
            all(v.dtype==torch.float32 and tuple(v.shape)==(1,100,200,200)and not v.requires_grad and bool(torch.isfinite(v).all())for v in masks),
            'All ten original finite native classes/masks required')
        require(torch.equal(torch.einsum('bqc,bchw->bqhw',e,F),M),'Actual GPU final native projection replay differs')
        return F,e,M,C,classes,masks
    finally:
        for handle in handles: handle.remove()
        cache.clear()
        require(before=={name:(tuple(module._forward_pre_hooks.items()),tuple(module._forward_hooks.items()))for name,module in head.named_modules()},
            'Inference hooks not restored')


def official_factors(rows,geometry_kind):
    return [dict(image_id=r['image_id'],query_id=r['query_id'],class_probability=r['class_probability'],
        quality=r['mask_quality'],score=r['official_score'],quality_kind='official_foreground_sigmoid_mean',
        geometry_kind=geometry_kind)for r in rows]


def raw_gate(ref,phase):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and path.parent.name==phase,
        'Absolute original raw phase locator required before read')
    raw=read(path); terminal(raw,phase,path); return raw


def bootstrap(spec):
    """Only metadata/raw locators before every prior held terminal0; no hashes/imports."""
    train=read(spec['training_spec']['path']); template=read(train['native_template_spec']['path'])
    require(template['execution_kind']=='formal_additional24' and
        set(template['training_receiptmap'])==set(template['training_phasemap'])==set(MODES),'Original native template differs')
    for mode in MODES: raw_gate(template['training_receiptmap'][mode],template['training_phasemap'][mode])
    raw_gate(template['restoration_receipt'],template['restoration_phase'])
    raw_gate(train['point_synthetic_CPU_receipt'],train['point_synthetic_CPU_phase'])
    for row in train.get('prior_receipts',[]):
        require(set(row)=={'receipt','phase'},'Explicit ancestry raw identity required'); raw_gate(row['receipt'],row['phase'])
    raw_gate(train['C_tiny_receipt'],train['C_tiny_phase'])
    raw_gate(train['C_tiny_CPU_receipt'],train['C_tiny_CPU_phase'])
    if spec['endpoint_kind']=='warm48':
        raw_gate(train['first24_receipt'],train['first24_phase'])
        raw_gate(train['first24_CPU_receipt'],train['first24_CPU_phase'])
    raw=raw_gate(spec['training_receipt'],spec['training_phase'])
    CPU_raw=raw_gate(spec['training_CPU_receipt'],spec['training_CPU_phase'])
    return train,raw,CPU_raw


def source_gate(path,spec):
    stage=STAGES[spec['endpoint_kind']]; train,raw,CPU_raw=bootstrap(spec)
    source,spec_ref=reference(__file__),reference(path); owner=read(spec['GPU_admission_file'])
    require(owner['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['automatic_retry']is False and
        owner['phase']==spec['phase'] and owner['allowed_cpus']==spec['CPU_affinity']==[0,1] and owner['cuda_visible_devices']=='0' and
        owner['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and owner['whole_GPU_used_limit_MiB']==8192 and
        owner['driver_sha256']==OWNER_SHA and Path(owner['entry']).resolve()==Path(source['path']) and
        owner['entry_args']==['--spec',str(path),'--owner-execute'],'Current C evaluation owner/entry differs')
    hashes=owner['bound_source_sha256']
    require(spec['entry_source']==source and source in spec['sources'] and spec['CPU_reader']in spec['sources'] and
        Path(spec['CPU_reader']['path']).resolve()==Path(__file__).with_name('qualify_evaluation_cpu.py').resolve() and
        len({r['path']for r in spec['sources']})==len(spec['sources']) and hashes.get(str(path))==spec_ref['sha256'],'Current C unique closure differs')
    for ref in spec['sources']: bound(ref); require(hashes.get(ref['path'])==ref['sha256'],'Complete current owner closure required')
    if 'scientific_spec'in owner: require(owner['scientific_spec']==spec_ref,'Current owner SPEC differs')
    for role in TRAIN_ROLES+PURE_ROLES+('DEV','official_evaluation_CPU_reader'):
        require(spec[role]in spec['sources'],'Required C evaluation role absent: '+role)
    require(spec['DEV']['bytes']==118319 and spec['DEV']['sha256']==DEV_SHA and
        train==read(bound(spec['training_spec'])) and all(r in spec['sources']for r in train['sources']) and
        train['stage_kind']==spec['endpoint_kind'] and spec['training_entry_source']==train['entry_source'] and
        spec['training_CPU_reader_source']==train['CPU_reader'] and spec['training_phase']==train['phase'],
        'OWN C trained closure/stage differs')
    producer=load(spec['training_entry_source'],'_C_qualified_training_provenance')
    trained_source,trained_spec,trained_admission,context=producer.source_gate(Path(spec['training_spec']['path']),train)
    require(producer.PREFIX==TRAIN_PREFIX and trained_source==spec['training_entry_source'] and
        trained_spec==spec['training_spec'] and raw['admission_sha256']==trained_admission['sha256'],'Own C actual training owner join differs')
    for role in producer.MODULES+producer.LIBRARIES+('training_CPU_reader','reused_CPU_reader','FIT','native_template_spec',
        'readout_source','geometry_source','hooks_source','point_fixture_source','point_synthetic_CPU_qualification'):
        require(spec[role]==train[role] and spec[role]in spec['sources'],'Qualified C helper role/closure differs: '+role)
    for role in ('checkpointmap','qualificationmap','training_receiptmap','training_phasemap','initializers','image_root','backbone_with_cp'):
        require(spec[role]==train[role],'Original native construction differs: '+role)
    q=read(bound(spec['training_CPU_qualification'])); report=read(bound(spec['training_report']))
    require(q['schema']=='ROOT-'+TRAIN_PREFIX+'-CPU-qualification-v1' and q['status']==TRAIN_STATUS and
        q['source']==trained_source and q['reader_source']==spec['training_CPU_reader_source'] and q['spec']==trained_spec and
        q['report']==spec['training_report'] and q['parent_receipt']==spec['training_receipt'] and
        q['GPU_admission']==trained_admission and q['phase']==spec['training_phase'] and q['stage_kind']==spec['endpoint_kind'] and
        q['CPU_admission']==spec['training_CPU_admission'] and CPU_raw['admission_sha256']==q['CPU_admission']['sha256'] and
        q['completed_phase_epochs']==stage['phase_epochs'] and q['effective_native_epoch']==stage['native_epoch'] and
        q['effective_readout_epoch']==stage['head_epoch'] and q['phase_microsteps']==stage['microsteps'] and
        q['native_updates']==q['readout_updates']==stage['updates'] and q['phase_raw_GT_visits']==stage['GT_visits'] and
        q['result']==report['result'] and set(q['result'])==set(q['arms'])=={'C'} and
        q['no_AP']is q['no_DEV']is q['no_TEST']is q['formal_initializer_qualified']is True and q['formal_resume_eligible']is False and
        q['CPU_execution']==dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=2,torch_num_interop_threads=2,CUDA_initialized=False),
        'Actual independent own C training qualification differs')
    require(q['teacher_checkpoint']==spec['checkpointmap']['native0'] and q['teacher_qualification']==spec['qualificationmap']['native0'] and
        q['CPU_numeric_protocol']==producer.NUMERIC and q['components']=={k:spec[k]for k in producer.MODULES} and
        q['head_parameter_tensors']==8 and q['head_parameter_scalars']==198402 and
        q['auxiliary_point_BCE_and_local_VJP_CPU_replayed']is q['eight_head_micro_gradient_CPU_replayed']is True and
        q['direct_point_logits']is True and q['native_L_added_to_point_logits']is False and
        q['native30_full_CPU_replay']is q['whole_CUDA_optimizer_trajectory_CPU_replay']is False,'Own C qualified math/component scope differs')
    require(report['schema']==TRAIN_PREFIX+'-report-v1' and report['status']=='POINTREND_STYLE_NATIVE_INTEGRATION_GPU_COMPLETE_PENDING_CPU' and
        report['source']==trained_source and report['spec']==trained_spec and report['GPU_admission']==trained_admission and
        report['phase']==spec['training_phase'] and report['stage_kind']==spec['endpoint_kind'] and
        report['error']is None and report['cleanup']['allocated_after']==0,'Qualified C report/cleanup differs')
    row=q['result']['C']; summary=q['arms']['C']
    require(spec['trained_checkpoint']==row['checkpoint']==summary['checkpoint'] and summary['typed_BN106_affine159_buffers_CPU_exact']is True and
        summary['endpoint_native610_CPU_finite']is True and row['readout_parameter_tensors']==8 and row['readout_parameters']==198402,
        'Own trained native610/direct Head8 differs')
    for ref in [q[k]for k in ('source','reader_source','spec','report','start','GPU_qualification','GPU_admission','CPU_admission','parent_receipt')]+[
        row[k]for k in ('checkpoint','endpoint_proof','trace','order','epochs','start')]+row['audits']:
        require(ref in spec['sources'],'Complete own C primary ref absent'); bound(ref)
    CPU_inputs=train['sources']+[q[k]for k in ('spec','report','start','GPU_qualification','GPU_admission','parent_receipt')]+[
        row[k]for k in ('checkpoint','endpoint_proof','trace','order','epochs','start')]+row['audits']
    producer.CPU_owner_join(spec,'training_CPU_receipt','training_CPU_phase','training_CPU_admission',q['reader_source'],
        ['--spec',q['spec']['path'],'--acceptance',spec['training_CPU_qualification']['path'],
            '--CPU-admission',q['CPU_admission']['path'],'--owner-execute'],CPU_inputs)
    legacy=load(spec['evaluation_source'],'_C_original_teacher_lineage_ONLY'); teachers=legacy.qualified_training(spec)
    for original in teachers.values():
        oq=original['qualification']
        require(all(r in spec['sources']for r in original['training_spec']['sources']+[oq[k]for k in
            ('source','reader_source','spec','report','start','GPU_admission','owner_driver')]+list(oq['model_proof']['proof_refs'].values())),
            'Original E24 qualified initializer/teacher closure absent')
    return source,spec_ref,reference(spec['GPU_admission_file']),dict(stage=stage,train=train,qualification=q,
        report=report,training_context=context,teachers=teachers)


def prepare(torch,pref,native,point,MODELS,revert,Config,classes,spec,lineage):
    stage,train,q=lineage['stage'],lineage['train'],lineage['qualification']
    construction=dict(lineage['teachers']['native0']['training_spec'],training_source=spec['training_source'],
        checkpointmap=spec['checkpointmap'],qualificationmap=spec['qualificationmap'],
        order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    initializer=load(spec['initializer_source'],'_C_eval_initializer'); common=load(spec['training_source'],'_C_eval_original_common')
    model,original,policy,original_loading=native.prepare_training_model(torch,pref,initializer,common,MODELS,revert,Config,
        classes,construction,lineage['teachers']['native0'])
    cp=torch.load(str(bound(spec['trained_checkpoint'])),map_location='cpu')
    require(set(cp)=={'schema','meta','native_state_dict','readout_state_dict','native_optimizer_state_dict',
        'readout_optimizer_state_dict','private_order_state','external_rng_state'} and cp['schema']==TRAIN_PREFIX+'-checkpoint-v1',
        'Own C native610+direct Head8 container required')
    meta,state,head_state=cp['meta'],cp['native_state_dict'],cp['readout_state_dict']
    require(meta['source']==q['source'] and meta['spec']==q['spec'] and meta['phase']==q['phase'] and meta['arm']=='C' and
        meta['stage_kind']==spec['endpoint_kind'] and meta['teacher_checkpoint']==spec['checkpointmap']['native0'] and
        meta['teacher_qualification']==spec['qualificationmap']['native0'] and meta['initial_native_epoch']==stage['initial_native_epoch'] and
        meta['completed_phase_epochs']==stage['phase_epochs'] and meta['effective_native_epoch']==stage['native_epoch'] and
        meta['effective_readout_epoch']==stage['head_epoch'] and meta['phase_microsteps']==stage['microsteps'] and
        meta['phase_updates']==stage['updates'] and meta['phase_raw_GT_visits']==stage['GT_visits'] and
        meta['auxiliary_weight']==1. and meta['direct_point_logits']is True and meta['native_L_added_to_point_logits']is False and
        meta['no_pending_accumulation_gradients']is True and meta['formal_resume_eligible']is False and
        all(meta[k]==train[k]for k in ('readout_seed','order_seed','loss_seed_base','point_seed_base')),'Fixed own C endpoint metadata differs')
    proof=read(bound(q['result']['C']['endpoint_proof']))
    require(proof['source']==q['source'] and proof['spec']==q['spec'] and proof['arm']=='C' and proof['checkpoint']==spec['trained_checkpoint'] and
        len(state)==len(original)==610 and state.keys()==original.keys() and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and
        v.shape==original[n].shape and v.dtype==original[n].dtype and bool(torch.isfinite(v).all())for n,v in state.items()) and
        pref.state_proof(state)==proof['after_native610'],'Own qualified C native610 differs')
    require(len(policy['frozen_names'])==106 and len(policy['BN_buffer_names'])==159 and
        all(torch.equal(state[n],original[n])for n in policy['frozen_names']+policy['BN_buffer_names']),'Original E24 typed BN106/159 must remain exact')
    loaded=model.load_state_dict(state,strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and pref.tree_equal(torch,model.state_dict(),state),'Strict own C native610 CPU load differs')
    readout=point.build_point_head(torch,train['readout_seed']); expected=readout.state_dict()
    require(tuple(head_state)==tuple(expected)==HEAD_NAMES and len(head_state)==8 and sum(v.numel()for v in head_state.values())==198402 and
        all(v.device.type=='cpu' and v.shape==expected[n].shape and v.dtype==expected[n].dtype and bool(torch.isfinite(v).all())
            for n,v in head_state.items()) and pref.state_proof(head_state)==proof['after_readout'],'Own qualified C direct Head8 differs')
    loaded=readout.load_state_dict(head_state,strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and pref.tree_equal(torch,readout.state_dict(),head_state) and
        len(list(readout.buffers()))==0,'Strict own direct Head8 CPU load differs')
    model.requires_grad_(False).eval(); readout.requires_grad_(False).eval(); model.backbone.with_cp=False
    native.imported_library_gate(torch,model,spec); fusion=model.panoptic_fusion_head
    require(model.panoptic_head.num_queries==100 and model.panoptic_head.num_classes==1 and
        fusion.num_classes==fusion.num_things_classes==1 and fusion.num_stuff_classes==0 and
        fusion.test_cfg.get('max_per_image',100)==100 and fusion.test_cfg.get('instance_on',False)is True and
        fusion.test_cfg.get('panoptic_on',True)is False and fusion.test_cfg.get('semantic_on',False)is False,'Official all100 fusion required')
    for role,obj in (('native_fusion_SOURCE',type(fusion)),('interpolation_SOURCE',torch.nn.functional.interpolate)):
        require(Path(inspect.getsourcefile(obj)).resolve()==bound(spec[role]).resolve(),'Actual inference library SOURCE differs')
    loading=dict(checkpoint=spec['trained_checkpoint'],qualification=spec['training_CPU_qualification'],arm='C',endpoint_kind=spec['endpoint_kind'],
        effective_native_epoch=stage['native_epoch'],effective_readout_epoch=stage['head_epoch'],strict_OWN_native610_loaded=True,
        strict_OWN_PointHead8_loaded=True,trained_fullstate=pref.state_proof(state),trained_readout=pref.state_proof(head_state),
        original_teacher_loading=original_loading,BN_parameter_names=policy['frozen_names'],BN_buffer_names=policy['BN_buffer_names'],
        original_typed_BN_exact=True,active_native_equal_original_E24_required=False,inference_backbone_with_cp=False,
        training_backbone_with_cp=train['backbone_with_cp'],optimizer_or_RNG_restored=False,
        checkpoint_optimizer_and_RNG_containers_deserialized=True,direct_point_logits=True,native_L_added_to_point_logits=False)
    anchors=pref.cpu_tree(torch,state),pref.cpu_tree(torch,head_state)
    del cp,original,state,head_state,proof
    return model,readout,anchors[0],anchors[1],loading


def evaluate_C(torch,np,masks_api,pref,old,diag,adapter,exporter,spec,spec_ref,prepared,dataset,coco,output,started):
    model,readout,state,head_state,loading=prepared
    directory=output/'C'; directory.mkdir(); packets=directory/'FIELD_PACKETS'; packets.mkdir()
    exports={name:dict(predictions=[],factors=[])for name in CONTROLS}; image_rows=[]
    renderer=lambda value,identity:diag.render_original(torch,value,identity)
    model.cuda().eval(); readout.cuda().eval()
    before=pref.state_proof(model.state_dict()); before_head=pref.state_proof(readout.state_dict())
    require(before==loading['trained_fullstate'] and before_head==loading['trained_readout'],'GPU-loaded own endpoint differs')
    try:
        for index in range(187):
            require(time.monotonic()-started<spec['deadline_seconds'],'Fixed endpoint deadline exceeded')
            data,samples,gray,raw,identity=old.load_image(model,dataset,coco,dataset.get_data_info(index))
            require(len(samples)==1 and not any(k in samples[0]for k in ('gt_instances','ignored_instances','gt_sem_seg','proposals')),
                'GT/proposals survived inference stripping')
            external=pref.rng_snapshot(torch); seed=100000+identity['image_id']
            with torch.random.fork_rng(devices=[0]):
                torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
                with torch.no_grad():
                    features=model.extract_feat(data['inputs'])
                    F,e,M,C,classes,masks=observed_inference(torch,model.panoptic_head,lambda:model.panoptic_head(features,samples))
                    versions=[v._version for v in classes+masks]
                    L=renderer(M,identity)
                    require(tuple(L.shape)==(100,*identity['original_shape']),'Native original geometry differs')
                    native_rows,native_factors,qids,class_packet=exporter.export_original(torch,np,masks_api,
                        model.panoptic_fusion_head,C,L,identity['image_id'],pref)
                    padded=torch.nn.functional.interpolate(M,size=(800,800),mode='bilinear',align_corners=False)
                    actual=model.panoptic_fusion_head.predict(C,padded,samples,rescale=True)[0]['ins_results']
                    ordered=L[qids]; binary=ordered>0
                    native_quality=(ordered.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
                    probabilities=C[0].softmax(-1)[:,0]; official_scores=probabilities[qids]*native_quality
                    require(len(actual)==100 and torch.equal(actual.masks,binary) and torch.equal(actual.scores,official_scores) and
                        bool((actual.labels==0).all()),'Actual native official fusion TorchEqual failed')
                    fine=renderer(F,identity).unsqueeze(0)
                    require(tuple(fine.shape)==(1,256,*identity['original_shape']),'Rendered fine256 geometry differs')
                    direct=adapter.dense_direct(torch,readout,fine,L.unsqueeze(0),pixel_chunk_size=spec['pixel_chunk_size'])
                    require(tuple(direct.shape)==tuple(L.shape) and bool(torch.isfinite(direct).all()),'Direct PointHead field differs')
                    direct_rows,direct_factors,direct_qids,direct_class_packet=exporter.export_original(torch,np,masks_api,
                        model.panoptic_fusion_head,C,direct,identity['image_id'],pref)
                    require(direct_qids==qids and direct_class_packet==class_packet,'Direct head changed own native classes/query order')
                    fixed={r['query_id']:r for r in native_factors}; fixed_rows=[]; fixed_factors=[]
                    for row in direct_rows:
                        f=fixed[row['query_id']]
                        fixed_rows.append(dict(row,score=f['official_score']))
                        fixed_factors.append(dict(image_id=row['image_id'],query_id=row['query_id'],class_probability=f['class_probability'],
                            quality=f['mask_quality'],score=f['official_score'],quality_kind='fixed_own_native_official_quality',
                            geometry_kind='direct_point_logits'))
                    exports['C_native_official']['predictions']+=native_rows
                    exports['C_native_official']['factors']+=official_factors(native_factors,'native')
                    exports['C_direct_official']['predictions']+=direct_rows
                    exports['C_direct_official']['factors']+=official_factors(direct_factors,'direct_point_logits')
                    exports['C_direct_native_score']['predictions']+=fixed_rows
                    exports['C_direct_native_score']['factors']+=fixed_factors
                    require(versions==[v._version for v in classes+masks],'Direct inference changed native stage tensors')
                    field_digests={name:pref.digest_tensor(value)for name,value in dict(native_masks=M,classes=C,feature=F).items()}
                    packet=dict(schema=PREFIX+'-fields-v1',source=reference(__file__),spec=spec_ref,arm='C',endpoint_kind=spec['endpoint_kind'],
                        checkpoint=spec['trained_checkpoint'],dataset_index=index,image_id=identity['image_id'],identity=identity,
                        native_masks200=M.detach().cpu().clone(),final_class_logits=C.detach().cpu().clone(),feature=F.detach().cpu().clone(),
                        field_digests=field_digests,prediction_query_ids=qids,GPU_original_logits_saved=False,all100_fields_saved=True,
                        direct_point_logits=True,native_L_added_to_point_logits=False)
                    packet_path=packets/('IMAGE_%04d.pth'%index)
                    with packet_path.open('xb')as stream: torch.save(packet,stream)
                    require(pref.tree_equal(torch,packet,torch.load(packet_path,map_location='cpu')),'Exact field packet CPU readback differs')
                    packet_ref=reference(packet_path)
                    del packet,F,e,M,C,classes,masks,features,L,fine,direct,padded,actual,ordered,binary,probabilities,native_quality,official_scores
                    del native_rows,native_factors,direct_rows,direct_factors,fixed_rows,fixed_factors,fixed,direct_qids,direct_class_packet
            restored=pref.rng_snapshot(torch)
            require(all(torch.equal(a,b)for a,b in zip(external,restored)),'Private inference RNG did not restore external states')
            image_rows.append(dict(arm='C',endpoint_kind=spec['endpoint_kind'],dataset_index=index,image_id=identity['image_id'],identity=identity,
                checkpoint=spec['trained_checkpoint'],qualification=spec['training_CPU_qualification'],ann_ids=identity['ann_ids'],
                annotation_count=len(identity['ann_ids']),head_seed=seed,GT_removed_from_forward=True,loader_accesses_annotations=True,
                matching_or_loss_executed=False,external_RNG_before=[pref.digest_tensor(v)for v in external],
                external_RNG_after=[pref.digest_tensor(v)for v in restored],external_RNG_restored=True,prediction_query_ids=qids,
                native_class_packet=class_packet,final_field_packet=packet_ref,final_fields=field_digests,predictions=100,
                passive_pixel_decoder_calls=1,passive_mask_embed_calls=10,native_all10_tensor_versions_unchanged=True,
                native_official_fusion_TorchEqual=True,all100_queries_retained=True,no_NMS=True,no_score_cutoff=True,
                GPU_original_logits_saved=False,direct_point_logits=True,native_L_added_to_point_logits=False,
                secondary_native_scores_exact=True,fixed_score_empty_direct_queries_retained=True))
            if(index+1)%50==0 or index==186: print(json.dumps(dict(arm='C',evaluated_images=index+1,elapsed_seconds=time.monotonic()-started)),flush=True)
            del data,samples,gray,raw,identity,external,restored,qids,class_packet
        after=pref.state_proof(model.state_dict()); after_head=pref.state_proof(readout.state_dict())
        require(before==after and before_head==after_head and all(torch.equal(v.detach().cpu(),state[n])for n,v in model.state_dict().items()) and
            all(torch.equal(v.detach().cpu(),head_state[n])for n,v in readout.state_dict().items()) and
            all(not p.requires_grad and p.grad is None for m in (model,readout)for p in m.parameters()) and
            all(not mod.training for m in (model,readout)for mod in m.modules()),'Own native610/direct Head8 evaluation state changed')
        require(len(image_rows)==187 and len({r['image_id']for r in image_rows})==187 and
            sum(r['annotation_count']for r in image_rows)==380,'Complete exposed DEV187/380 required')
        controls={}
        for name,row in exports.items():
            require(len(row['predictions'])==len(row['factors'])==18700,'All18700 original queries required')
            control_dir=directory/name; control_dir.mkdir()
            write(control_dir/'ALL_PREDICTIONS.json',row['predictions']); write(control_dir/'SCORE_FACTORS.json',row['factors'])
            controls[name]=dict(predictions=reference(control_dir/'ALL_PREDICTIONS.json'),score_factors=reference(control_dir/'SCORE_FACTORS.json'),
                images=187,raw_GT=380,predictions_count=18700,geometry='native'if name=='C_native_official'else'direct_point_logits',
                score_kind='fixed_own_native_official'if name=='C_direct_native_score'else'own_official_foreground_sigmoid_mean',
                diagnostic_only=name=='C_direct_native_score',AP_computed_on_GPU=False,all100_queries_retained=True)
        write(directory/'IMAGE_ROWS.json',image_rows)
        write(directory/'STATE_PROOF.json',dict(schema=PREFIX+'-state-proof-v1',source=reference(__file__),spec=spec_ref,arm='C',
            loading=loading,before_native610=before,after_native610=after,before_readout=before_head,after_readout=after_head,
            native610_and_readout8_CPU_identity_exact=True,params_gradNone=True,all_parameters_frozen=True,all_modules_eval=True,
            no_optimizer_or_RNG_restored=True,GT_removed_from_forward=True,direct_point_logits=True,native_L_added_to_point_logits=False))
        return dict(arm='C',checkpoint=spec['trained_checkpoint'],qualification=spec['training_CPU_qualification'],
            image_rows=reference(directory/'IMAGE_ROWS.json'),state_proof=reference(directory/'STATE_PROOF.json'),controls=controls,
            images=187,raw_GT=380,all100_fields_saved=True,field_packets=187,full_native610_exact=True,
            readout_state_tensors=8,readout_parameters=198402,effective_native_epoch=loading['effective_native_epoch'],
            effective_readout_epoch=loading['effective_readout_epoch'],no_GPU_AP=True,no_TEST=True,GT_free_inference=True)
    finally:
        model.cpu(); readout.cpu()


def run(spec_file):
    started=time.monotonic(); path=Path(spec_file)
    require(path.is_absolute(),'Absolute SPEC required'); path=path.resolve(strict=True); spec=read(path)
    require(spec['schema']==SCHEMA and spec['kind']=='mask2former' and spec['endpoint_kind']in STAGES and
        spec['arms']==['C'] and spec['controls']==list(CONTROLS) and spec['evaluation_images']==187 and spec['evaluation_raw_GT']==380 and
        spec['head_seed_base']==100000 and spec['CPU_affinity']==[0,1] and spec['allocator_limit_MiB']==4096 and
        spec['whole_GPU_owner_limit_MiB']==8192 and type(spec['deadline_seconds'])is int and 0<spec['deadline_seconds']<=21600 and
        spec['CPU_numeric_protocol']==NUMERIC and spec['pixel_chunk_size']==2048 and spec['direct_point_logits']is True and
        spec['native_L_added_to_point_logits']is False and spec['endpoint_selection']is False and
        spec['historically_exposed_DEV']is True and spec['no_TEST']is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='0',
        'Prelocked C single fixed endpoint three-control evaluation SPEC differs')
    source,spec_ref,admission,lineage=source_gate(path,spec)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(),'Fresh absolute output required'); output.mkdir(parents=True)
    write(output/'START.json',dict(schema=PREFIX+'-start-v1',source=source,spec=spec_ref,GPU_admission=admission,phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'],fixed_effective_native_epoch=lineage['stage']['native_epoch'],
        fixed_C_readout_epoch=lineage['stage']['head_epoch'],controls=list(CONTROLS),arms=['C'],
        training_CPU_qualification=spec['training_CPU_qualification'],checkpoint=spec['trained_checkpoint'],GT_free_inference=True,
        loader_accesses_annotations=True,training_performed=False,optimizer_restored=False,AP_computed_on_GPU=False,no_TEST=True,
        endpoint_selection=False,CPU_numeric_protocol=NUMERIC,direct_point_logits=True,native_L_added_to_point_logits=False))
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'CPU01 required before ML imports')
    import torch
    import numpy as np
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(not torch.cuda.is_initialized() and torch.__version__=='2.0.1+cu118','Pinned Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes'])==80 and CocoDataset.METAINFO['classes'][8]=='boat','boat8/null80 required')
    pref=load(spec['micro_preflight_source'],'_C_eval_pref'); native=load(spec['native_loader_source'],'_C_eval_native_constructor')
    point=load(spec['point_head_source'],'_C_eval_point_head'); old=load(spec['native_helpers'],'_C_eval_loader'); old.torch=torch
    diag=load(spec['renderer_source'],'_C_eval_renderer'); adapter=load(spec['point_adapter_source'],'_C_eval_direct_adapter')
    exporter=load(spec['official_export_source'],'_C_eval_official_export')
    prepared=prepare(torch,pref,native,point,MODELS,revert_sync_batchnorm,Config,tuple(CocoDataset.METAINFO['classes']),spec,lineage)
    dataset=CocoDataset(ann_file=str(bound(spec['DEV'])),data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(bound(spec['DEV']))); require(len(dataset)==len(coco.imgs)==187 and len(coco.anns)==380 and
        [int(dataset.get_data_info(i)['img_id'])for i in range(187)]==list(coco.imgs),'Locked ordered exposed DEV187/380 required')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); result,error,cleanup={},None,None
    try:
        result['C']=evaluate_C(torch,np,masks_api,pref,old,diag,adapter,exporter,spec,spec_ref,prepared,dataset,coco,output,started)
        del prepared; cleanup=pref.clear_gpu(torch); require(cleanup['allocated_after']==0,'Endpoint CUDA cleanup failed')
        result['C']['cleanup']=cleanup
        for ref in spec['sources']: bound(ref)
        require(reference(path)==spec_ref and time.monotonic()-started<spec['deadline_seconds'],'Final inputs/deadline differ')
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),no_retry=True); traceback.clear_frames(exc.__traceback__)
        if 'prepared'in locals():
            prepared[0].cpu(); prepared[1].cpu(); del prepared
        gc.collect(); cleanup=pref.clear_gpu(torch)
    controls={name:row for arm in result.values()for name,row in arm['controls'].items()}
    if error is None: require(set(controls)==set(CONTROLS),'Three complete C controls required')
    report=dict(schema=PREFIX+'-report-v1',status='POINTREND_STYLE_FIXED_ENDPOINT_DEV_EXPORT_COMPLETE_NO_AP'if error is None else
        'POINTREND_STYLE_FIXED_ENDPOINT_DEV_EXPORT_STOP',source=source,spec=spec_ref,GPU_admission=admission,phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'],effective_native_epoch=lineage['stage']['native_epoch'],effective_C_readout_epoch=lineage['stage']['head_epoch'],
        training_CPU_qualification=spec['training_CPU_qualification'],checkpoint=spec['trained_checkpoint'],result=result,controls=controls,
        error=error,cleanup=cleanup,images_per_arm=187,raw_GT_per_arm=380,queries_per_image=100,CPU_numeric_protocol=NUMERIC,
        GT_removed_from_forward=True,loader_accesses_annotations=True,matching_or_loss_executed=False,one_native_forward_per_image=True,
        direct_point_logits=True,native_L_added_to_point_logits=False,main_official_quality_recomputed_on_direct_mask=True,
        secondary_fixed_native_scores_diagnostic_only=True,fixed_score_empty_direct_queries_retained=True,
        original_teacher_E24_only_initializer_and_BN_reference=True,trained_native610_used_for_classes_feature_masks=True,
        no_TEST=True,AP_computed_on_GPU=False,official_COCO12_pending_CPU=True,training_performed=False,optimizer_or_RNG_restored=False,
        endpoint_selection=False,formal_resume_eligible=False,efficacy_claimed=False,faithful_PointRend_reproduction_claimed=False,
        capacity_matched_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False,
        CPU_renderer_GPU_bitwise_claimed=False,certified_GPU_error_bound=False,GPU_original_logits_saved=False,
        elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    write(output/'REPORT.json',report)
    if error is None:
        write(output/'QUALIFIED.json',dict(schema=PREFIX+'-aggregate-v1',status='ROOT_POINTREND_STYLE_FIXED_ENDPOINT_EVALUATION_PENDING_CPU_READER',
            source=source,spec=spec_ref,report=reference(output/'REPORT.json'),result=result,controls=controls,phase=spec['phase'],
            endpoint_kind=spec['endpoint_kind'],independently_CPU_qualified=False,AP_computed=False,formal_resume_eligible=False))
    require(error is None,'EVALUATION_STOP: '+str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned GPU execution required'); run(args.spec)


if __name__=='__main__':
    main()
