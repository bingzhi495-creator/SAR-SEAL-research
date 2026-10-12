"""Fixed joint24/E48 or own warm48/E96 DEV export; SOURCE only at authoring.

Each arm loads its OWN CPU-qualified trained native610; B additionally loads
its own external8/9441 head. B posterior geometry is direct L+u, the training
target, never the legacy bounded L+tanh(u). Six prelocked controls only.
One GT-free native forward per image/arm, no matching/loss/training/selection.
CPU qualification of fields, RLE, scores and COCO12 is a separate future job.
"""
import argparse
import copy
from functools import partial
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

PREFIX='query-pixel-joint-native-fixed-endpoint-evaluation'
SCHEMA='ROOT-'+PREFIX+'-spec-v1'
ARMS=('A','B')
MODES=('native0','global','spatial_halves','spatial_reversed')
CONTROLS=('A_native_official','A_native_zero_budget','B_native_official',
    'B_native_posterior_budget','B_posterior_official','B_posterior_budget')
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
DEV_SHA='5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55'
NUMERIC=dict(class_rtol=1e-6,class_atol=1e-6,renderer_rtol=1e-4,renderer_atol=1e-4,
    quality_rtol=1e-4,quality_atol=1e-4,threshold_near_zero_rtol=1e-4,threshold_near_zero_atol=1e-4)
STAGES={
    'joint24':dict(prefix='query-pixel-joint-native-fullFIT24',
        qual_status='ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_CPU_QUALIFIED',
        report_status='QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_GPU_COMPLETE_PENDING_CPU',
        initial_native_epoch=24,phase_epochs=24,native_epoch=48,head_epoch=24,microsteps=17784,updates=8892,GT_visits=39864),
    'warm48':dict(prefix='query-pixel-joint-native-E24warm-additional48',
        qual_status='ROOT_QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_CPU_QUALIFIED',
        report_status='QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_GPU_COMPLETE_PENDING_CPU',
        initial_native_epoch=48,phase_epochs=48,native_epoch=96,head_epoch=72,microsteps=35568,updates=17784,GT_visits=79728)}
TRAIN_ROLES=('joint_training_entry_source','joint_training_CPU_reader_source','joint_training_spec',
    'joint_training_CPU_qualification','joint_training_report','joint_training_receipt')
PURE_ROLES=('posterior_dense_source','official_export_source','renderer_source','posterior_readout_source',
    'native_adapter_source','native_loader_source','evaluation_source','micro_preflight_source','initializer_source',
    'training_source','native_helpers','training_CPU_reader','reused_CPU_reader')


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


def terminal(raw,phase):
    require(raw.get('status')=='ACTUAL_OWNED_TERMINAL' and raw.get('original_identity')is not None and
        type(raw.get('returncode'))is int and raw['returncode']==0 and raw.get('root_wait_reaped')is True and
        raw.get('held_handle_exit')==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0) and
        raw.get('numeric_birth_absence',{}).get('original_birth_absent')is True and
        all(k in raw and raw[k]is None for k in ('abort','error')) and raw.get('wait_errors')is False and
        raw.get('GPU_observation_errors')==[],'STOP before hashes/imports: actual raw0 required')
    peak=raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase',phase)==phase and type(peak)in(int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        raw.get('whole_GPU_used_limit_MiB')==8192 and raw.get('whole_leaf_VRAM_certified')is False and
        raw.get('GPU_scope')=='whole GPU observed, not per-process WDDM allocation','Actual raw resource scope differs')


def source_gate(path,spec):
    stage=STAGES[spec['endpoint_kind']]
    train=read(spec['joint_training_spec']['path']); tiny=read(train['joint_preflight_spec']['path'])
    require(set(tiny['training_receiptmap'])==set(tiny['training_phasemap'])==set(MODES),'Original four raws required')
    for mode in MODES: terminal(read(tiny['training_receiptmap'][mode]['path']),tiny['training_phasemap'][mode])
    for row in tiny['prior_receipts']:
        require(set(row)=={'receipt','phase'},'Explicit technical prior raw required'); terminal(read(row['receipt']['path']),row['phase'])
    terminal(read(train['joint_preflight_receipt']['path']),train['joint_preflight_phase'])
    if spec['endpoint_kind']=='warm48': terminal(read(train['first24_receipt']['path']),train['first24_phase'])
    raw=read(spec['joint_training_receipt']['path']); terminal(raw,spec['joint_training_phase'])
    # Every original/technical/own first24/current training raw precedes first SHA or SOURCE import.
    source,spec_ref=reference(__file__),reference(path); owner=read(spec['GPU_admission_file']); hashes=owner['bound_source_sha256']
    require(owner['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['automatic_retry']is False and
        owner['phase']==spec['phase'] and owner['allowed_cpus']==spec['CPU_affinity']==[0,1] and owner['cuda_visible_devices']=='0' and
        owner['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and owner['whole_GPU_used_limit_MiB']==8192 and
        owner['driver_sha256']==OWNER_SHA and Path(owner['entry']).resolve()==Path(source['path']) and
        owner['entry_args']==['--spec',str(path),'--owner-execute'],'Current owner/entry differs')
    require(spec['entry_source']==source and source in spec['sources'] and spec['CPU_reader']in spec['sources'] and
        Path(spec['CPU_reader']['path']).resolve()==Path(__file__).with_name('qualify_cpu.py').resolve() and
        len({r['path']for r in spec['sources']})==len(spec['sources']) and hashes.get(str(path))==spec_ref['sha256'],'Current unique closure differs')
    for ref in spec['sources']: bound(ref); require(hashes.get(ref['path'])==ref['sha256'],'Complete owner closure required')
    if 'scientific_spec'in owner: require(owner['scientific_spec']==spec_ref,'Current owner SPEC differs')
    for role in TRAIN_ROLES+PURE_ROLES+('DEV',): require(spec[role]in spec['sources'],'Required role absent: '+role)
    require(spec['DEV']['bytes']==118319 and spec['DEV']['sha256']==DEV_SHA and
        train==read(bound(spec['joint_training_spec'])) and all(r in spec['sources']for r in train['sources']) and
        spec['joint_training_entry_source']==train['entry_source'] and spec['joint_training_CPU_reader_source']==train['CPU_reader'] and
        spec['joint_training_phase']==train['phase'] and
        all(spec[k]==train[k]for k in ('checkpointmap','qualificationmap','training_receiptmap','training_phasemap',
            'initializers','FIT','image_root','native_loader_source','evaluation_source','micro_preflight_source','initializer_source',
            'training_source','native_helpers','renderer_source','posterior_readout_source','training_CPU_reader','reused_CPU_reader')),
        'Original teacher and OWN trained lineage must remain distinct and preserved')
    reader=load(spec['joint_training_CPU_reader_source'],'_joint_endpoint_qualified_training_reader')
    producer,training_refs=reader.source_gate(Path(spec['joint_training_spec']['path']),train)
    require(producer.PREFIX==stage['prefix'] and reader.QUAL_SCHEMA=='ROOT-'+stage['prefix']+'-CPU-qualification-v1' and
        training_refs['source']==spec['joint_training_entry_source'] and training_refs['reader_source']==spec['joint_training_CPU_reader_source'] and
        training_refs['spec']==spec['joint_training_spec'] and training_refs['parent_receipt']==spec['joint_training_receipt'] and
        training_refs['phase']==spec['joint_training_phase'] and raw['admission_sha256']==training_refs['GPU_admission']['sha256'],
        'Actual trained SOURCE/reader/SPEC/raw owner differs')
    for role in producer.LIBRARIES:
        require(spec[role]==train[role] and spec[role]in spec['sources'],'Qualified native library role/closure differs: '+role)
    q=read(bound(spec['joint_training_CPU_qualification'])); report=read(bound(spec['joint_training_report']))
    require(q['schema']==reader.QUAL_SCHEMA and q['status']==stage['qual_status'] and
        all(q[k]==training_refs[k]for k in training_refs) and q['report']==spec['joint_training_report'] and
        q['completed_phase_epochs']==stage['phase_epochs'] and q['effective_native_epoch']==stage['native_epoch'] and
        q['phase_microsteps_per_arm']==stage['microsteps'] and q['native_updates_per_arm']==q['B_readout_updates']==stage['updates'] and
        q['phase_raw_GT_visits_per_arm']==stage['GT_visits'] and q['result']==report['result'] and set(q['arms'])==set(ARMS) and
        q['no_AP']is q['no_DEV']is q['no_TEST']is True and q['formal_resume_eligible']is False and
        q['CPU_execution']==dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=2,torch_num_interop_threads=2,CUDA_initialized=False),
        'Actual complete trained CPU qualification differs')
    require(report['schema']==stage['prefix']+'-report-v1' and report['status']==stage['report_status'] and
        report['source']==training_refs['source'] and report['spec']==training_refs['spec'] and
        report['GPU_admission']==training_refs['GPU_admission'] and report['phase']==training_refs['phase'] and
        report['error']is None and report['cleanup']['allocated_after']==0 and set(spec['joint_checkpoints'])==set(ARMS),
        'Actual trained report/cleanup differs')
    for arm in ARMS:
        require(spec['joint_checkpoints'][arm]==q['result'][arm]['checkpoint']==q['arms'][arm]['checkpoint'] and
            q['arms'][arm]['typed_BN106_affine159_buffers_CPU_exact']is True,'Own trained arm checkpoint/BN differs')
        for key in ('checkpoint','endpoint_proof','trace','order','epochs','start'):
            require(q['result'][arm][key]in spec['sources'],'Trained artifact absent from current closure')
    for key in ('report','start','GPU_qualification','GPU_admission'):
        require(q[key]in spec['sources'],'Complete qualified training provenance absent')
    if spec['endpoint_kind']=='warm48':
        require(q['completed_joint_training_epochs']==q['effective_B_readout_epoch']==72 and q['weights_warm_start']is True and
            q['first24_CPU_qualification']==train['first24_CPU_qualification'],'Own first24->warm48 lineage differs')
    legacy=load(spec['evaluation_source'],'_joint_original_teacher_lineage_ONLY')
    teachers=legacy.qualified_training(spec)
    for row in teachers.values():
        original=row['qualification']
        require(all(ref in spec['sources']for ref in row['training_spec']['sources']+[original[k]for k in
            ('source','reader_source','spec','report','start','GPU_admission','owner_driver')]+list(original['model_proof']['proof_refs'].values())),
            'Original initializer/teacher proof closure absent')
    return source,spec_ref,reference(spec['GPU_admission_file']),dict(stage=stage,train=train,qualification=q,
        report=report,training_refs=training_refs,teachers=teachers)


def prepare_arm(torch,pref,native,posterior,MODELS,revert,Config,classes,spec,lineage,arm):
    stage,train,q=lineage['stage'],lineage['train'],lineage['qualification']
    construction=dict(lineage['teachers']['native0']['training_spec'],training_source=spec['training_source'],
        checkpointmap=spec['checkpointmap'],qualificationmap=spec['qualificationmap'],
        order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    initializer=load(spec['initializer_source'],'_joint_eval_initializer_'+arm)
    common=load(spec['training_source'],'_joint_eval_original_common_'+arm)
    model,original,policy,original_loading=native.prepare_training_model(torch,pref,initializer,common,MODELS,revert,Config,
        classes,construction,lineage['teachers']['native0'])
    cp=torch.load(str(bound(spec['joint_checkpoints'][arm])),map_location='cpu')
    require(set(cp)=={'schema','meta','native_state_dict','readout_state_dict','native_optimizer_state_dict',
        'readout_optimizer_state_dict','private_order_state','external_rng_state'} and
        cp['schema']==stage['prefix']+'-checkpoint-v1','OWN joint native610+B8 container required')
    meta,state,head_state=cp['meta'],cp['native_state_dict'],cp['readout_state_dict']
    require(meta['source']==q['source'] and meta['spec']==q['spec'] and meta['phase']==q['phase'] and meta['arm']==arm and
        meta['teacher_checkpoint']==spec['checkpointmap']['native0'] and meta['teacher_qualification']==spec['qualificationmap']['native0'] and
        meta['initial_native_epoch']==stage['initial_native_epoch'] and meta['completed_phase_epochs']==stage['phase_epochs'] and
        meta['effective_native_epoch']==stage['native_epoch'] and meta['phase_microsteps']==stage['microsteps'] and
        meta['phase_updates']==stage['updates'] and meta['phase_raw_GT_visits']==stage['GT_visits'] and
        meta['no_pending_accumulation_gradients']is True and meta['formal_resume_eligible']is False and
        meta['auxiliary_weight']==1. and meta['kappa']==1. and meta['eps']==1e-6,'Fixed OWN trained endpoint metadata differs')
    proof=read(bound(q['result'][arm]['endpoint_proof']))
    require(proof['source']==q['source'] and proof['spec']==q['spec'] and proof['arm']==arm and
        proof['checkpoint']==spec['joint_checkpoints'][arm] and
        len(state)==len(original)==610 and state.keys()==original.keys() and
        all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and v.shape==original[n].shape and v.dtype==original[n].dtype and
            bool(torch.isfinite(v).all())for n,v in state.items()) and pref.state_proof(state)==proof['after_native610'],
        'OWN trained native610 differs from qualified endpoint')
    require(len(policy['frozen_names'])==106 and len(policy['BN_buffer_names'])==159 and
        all(torch.equal(state[n],original[n])for n in policy['frozen_names']+policy['BN_buffer_names']),
        'Original E24 typed BN106/159 must remain exact; active weights are OWN trained weights')
    loaded=model.load_state_dict(state,strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and
        all(torch.equal(v.detach().cpu(),state[n])for n,v in model.state_dict().items()),'Strict OWN native610 CPU load differs')
    readout=None
    if arm=='B':
        readout=posterior.QueryPixelPosteriorReadout(train['readout_seed'],1.,1e-6)
        expected=readout.state_dict()
        require(set(head_state)==set(expected) and len(head_state)==8 and sum(v.numel()for v in head_state.values())==9441 and
            all(v.device.type=='cpu' and v.dtype==expected[n].dtype and v.shape==expected[n].shape and bool(torch.isfinite(v).all())
                for n,v in head_state.items()) and pref.state_proof(head_state)==proof['after_readout'],'OWN finite qualified B8 differs')
        loaded=readout.load_state_dict(head_state,strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys and
            pref.tree_equal(torch,readout.state_dict(),head_state) and len(list(readout.buffers()))==0,'Strict OWN B8 CPU load differs')
        readout.requires_grad_(False).eval()
    else: require(head_state=={} and proof['after_readout']==pref.state_proof({}),'A must have no external head')
    model.requires_grad_(False).eval(); model.backbone.with_cp=False
    native.imported_library_gate(torch,model,spec)
    fusion=model.panoptic_fusion_head
    require(model.panoptic_head.num_queries==100 and model.panoptic_head.num_classes==1 and
        fusion.num_classes==fusion.num_things_classes==1 and fusion.num_stuff_classes==0 and
        fusion.test_cfg.get('max_per_image',100)==100 and fusion.test_cfg.get('instance_on',False)is True and
        fusion.test_cfg.get('panoptic_on',True)is False and fusion.test_cfg.get('semantic_on',False)is False,'Official all100 fusion required')
    for role,obj in (('native_fusion_SOURCE',type(fusion)),('interpolation_SOURCE',torch.nn.functional.interpolate)):
        require(Path(inspect.getsourcefile(obj)).resolve()==bound(spec[role]).resolve(),'Actual inference SOURCE differs')
    loading=dict(checkpoint=spec['joint_checkpoints'][arm],qualification=spec['joint_training_CPU_qualification'],arm=arm,
        endpoint_kind=spec['endpoint_kind'],effective_native_epoch=stage['native_epoch'],effective_readout_epoch=stage['head_epoch']if arm=='B'else 0,
        strict_OWN_native610_loaded=True,strict_OWN_readout8_loaded=arm=='B',trained_fullstate=pref.state_proof(state),
        trained_readout=pref.state_proof(head_state),original_teacher_loading=original_loading,BN_parameter_names=policy['frozen_names'],
        BN_buffer_names=policy['BN_buffer_names'],original_typed_BN_exact=True,active_native_equal_original_E24_required=False,
        inference_backbone_with_cp=False,training_backbone_with_cp=train['backbone_with_cp'],optimizer_or_RNG_restored=False,
        checkpoint_optimizer_and_RNG_containers_deserialized=True,
        historical_class_or_score_equality_assumed=False)
    native_anchor=pref.cpu_tree(torch,state); head_anchor=pref.cpu_tree(torch,head_state)
    del cp,original,state,head_state,proof
    return model,readout,native_anchor,head_anchor,loading


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


def budget_rows(torch,official_rows,factors,logits,posterior,geometry_kind):
    """Only score changes; support is this actual GPU geometry, never GT."""
    support=logits>0
    intersection=(support*posterior).flatten(1).sum(1)
    union=support.flatten(1).sum(1).to(posterior.dtype)+((~support)*posterior).flatten(1).sum(1)
    quality=intersection/(union+1e-6)
    require(bool(torch.isfinite(quality).all())and bool(((quality>=0)&(quality<=1)).all()),'Finite posterior budget required')
    rows=copy.deepcopy(official_rows); by_q={row['query_id']:row for row in factors}; proof=[]
    for row in rows:
        q=row['query_id']; c=posterior.new_tensor(by_q[q]['class_probability'])
        score=float(c*quality[q]); row['score']=score
        proof.append(dict(image_id=row['image_id'],query_id=q,class_probability=float(c),quality=float(quality[q]),score=score,
            quality_kind='posterior_intersection_union_ratio',geometry_kind=geometry_kind,
            predicted_intersection=float(intersection[q]),predicted_union=float(union[q]),epsilon=1e-6))
    return rows,proof


def official_factors(rows,geometry_kind):
    return [dict(image_id=r['image_id'],query_id=r['query_id'],class_probability=r['class_probability'],
        quality=r['mask_quality'],score=r['official_score'],quality_kind='official_foreground_sigmoid_mean',
        geometry_kind=geometry_kind)for r in rows]


def evaluate_arm(torch,np,masks_api,pref,old,diag,adapter,dense,exporter,spec,spec_ref,prepared,dataset,coco,output,started,arm):
    model,readout,state,head_state,loading=prepared; directory=output/arm; directory.mkdir()
    packets=directory/'FINAL_FIELDS'; packets.mkdir()
    model.to('cuda:0').eval()
    if readout is not None: readout.to('cuda:0').eval()
    before=pref.state_proof(model.state_dict()); before_head=pref.state_proof({}if readout is None else readout.state_dict())
    require(before==loading['trained_fullstate']and before_head==loading['trained_readout'],'Actual evaluation initial state differs')
    renderer=partial(diag.render_original,torch)
    names=[key for key in CONTROLS if key.startswith(arm+'_')]
    exports={key:dict(predictions=[],factors=[])for key in names}; image_rows=[]
    try:
        for index in range(187):
            require(time.monotonic()-started<spec['deadline_seconds'],'Fixed evaluation deadline exceeded')
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
                    require(len(actual)==100 and torch.equal(actual.masks,binary)and torch.equal(actual.scores,official_scores)and
                        bool((actual.labels==0).all()),'Actual whole native official fusion TorchEqual failed')
                    exports[arm+'_native_official']['predictions']+=native_rows
                    exports[arm+'_native_official']['factors']+=official_factors(native_factors,'native')
                    posterior_logits=None
                    if arm=='A':
                        budget,pf=budget_rows(torch,native_rows,native_factors,L,L.sigmoid(),'native')
                        exports['A_native_zero_budget']['predictions']+=budget; exports['A_native_zero_budget']['factors']+=pf
                    else:
                        fields=adapter.project_native_fields(readout,F,e,M,identity['actual_img_shape'],identity['original_shape'],renderer)
                        require(torch.equal(fields['native_logits'][0],L),'Actual B original native field differs')
                        posterior_logits=dense.dense_posterior_logits(torch,readout,fields)[0]
                        posterior=posterior_logits.sigmoid()
                        budget,pf=budget_rows(torch,native_rows,native_factors,L,posterior,'native')
                        exports['B_native_posterior_budget']['predictions']+=budget; exports['B_native_posterior_budget']['factors']+=pf
                        posterior_rows,posterior_factors,posterior_qids,posterior_class_packet=exporter.export_original(torch,np,masks_api,
                            model.panoptic_fusion_head,C,posterior_logits,identity['image_id'],pref)
                        require(posterior_qids==qids and posterior_class_packet==class_packet,'Own native classes/query order changed by posterior readout')
                        exports['B_posterior_official']['predictions']+=posterior_rows
                        exports['B_posterior_official']['factors']+=official_factors(posterior_factors,'L_plus_u')
                        budget,pf=budget_rows(torch,posterior_rows,posterior_factors,posterior_logits,posterior,'L_plus_u')
                        exports['B_posterior_budget']['predictions']+=budget; exports['B_posterior_budget']['factors']+=pf
                        del fields,posterior,posterior_rows,posterior_factors,posterior_qids,posterior_class_packet
                    require(versions==[v._version for v in classes+masks],'Posterior inference changed native stage tensors')
                    field_digests={name:pref.digest_tensor(value)for name,value in dict(native_masks=M,classes=C).items()}
                    packet=dict(schema=PREFIX+'-fields-v1',source=reference(__file__),spec=spec_ref,arm=arm,endpoint_kind=spec['endpoint_kind'],
                        checkpoint=spec['joint_checkpoints'][arm],dataset_index=index,image_id=identity['image_id'],identity=identity,
                        native_masks200=M.detach().cpu().clone(),final_class_logits=C.detach().cpu().clone(),
                        feature=F.detach().cpu().clone()if arm=='B'else None,coefficient=e.detach().cpu().clone()if arm=='B'else None,
                        field_digests=field_digests,prediction_query_ids=qids,GPU_original_logits_saved=False,
                        all100_fields_saved=True,posterior_geometry='L_plus_u'if arm=='B'else None)
                    if arm=='B':
                        field_digests.update(feature=pref.digest_tensor(F),coefficient=pref.digest_tensor(e))
                    packet_path=packets/('IMAGE_%04d.pth'%index)
                    with packet_path.open('xb')as stream: torch.save(packet,stream)
                    require(pref.tree_equal(torch,packet,torch.load(packet_path,map_location='cpu')),'Exact CPU field packet readback differs')
                    packet_ref=reference(packet_path)
                    del packet,F,e,M,C,classes,masks,features,L,padded,actual,ordered,binary,probabilities,native_quality,official_scores
                    if posterior_logits is not None: del posterior_logits
                    del native_rows,native_factors,budget,pf
            restored=pref.rng_snapshot(torch)
            require(all(torch.equal(a,b)for a,b in zip(external,restored)),'Private head RNG did not restore external states')
            image_rows.append(dict(arm=arm,endpoint_kind=spec['endpoint_kind'],dataset_index=index,image_id=identity['image_id'],identity=identity,
                checkpoint=spec['joint_checkpoints'][arm],qualification=spec['joint_training_CPU_qualification'],ann_ids=identity['ann_ids'],
                annotation_count=len(identity['ann_ids']),head_seed=seed,GT_removed_from_forward=True,loader_accesses_annotations=True,
                matching_or_loss_executed=False,external_RNG_before=[pref.digest_tensor(v)for v in external],
                external_RNG_after=[pref.digest_tensor(v)for v in restored],external_RNG_restored=True,prediction_query_ids=qids,
                native_class_packet=class_packet,final_field_packet=packet_ref,final_fields=field_digests,predictions=100,
                passive_pixel_decoder_calls=1,passive_mask_embed_calls=10,native_all10_tensor_versions_unchanged=True,
                native_official_fusion_TorchEqual=True,all100_queries_retained=True,no_NMS=True,no_score_cutoff=True,
                GPU_original_logits_saved=False,historical_cross_arm_class_equality_assumed=False))
            if(index+1)%50==0 or index==186: print(json.dumps(dict(arm=arm,evaluated_images=index+1,elapsed_seconds=time.monotonic()-started)),flush=True)
            del data,samples,gray,raw,identity,external,restored,qids,class_packet
        after=pref.state_proof(model.state_dict()); after_head=pref.state_proof({}if readout is None else readout.state_dict())
        require(before==after and before_head==after_head and all(torch.equal(v.detach().cpu(),state[n])for n,v in model.state_dict().items())and
            (readout is None or all(torch.equal(v.detach().cpu(),head_state[n])for n,v in readout.state_dict().items()))and
            all(not p.requires_grad and p.grad is None for m in (model,readout)if m is not None for p in m.parameters())and
            all(not mod.training for m in (model,readout)if m is not None for mod in m.modules()),'OWN native610/head8 evaluation state/eval changed')
        require(len(image_rows)==187 and len({r['image_id']for r in image_rows})==187 and
            sum(r['annotation_count']for r in image_rows)==380,'Complete exposed DEV187/380 population required')
        controls={}
        for name,row in exports.items():
            require(len(row['predictions'])==len(row['factors'])==18700,'All18700 original queries required')
            control_dir=directory/name; control_dir.mkdir()
            write(control_dir/'ALL_PREDICTIONS.json',row['predictions']); write(control_dir/'SCORE_FACTORS.json',row['factors'])
            controls[name]=dict(predictions=reference(control_dir/'ALL_PREDICTIONS.json'),score_factors=reference(control_dir/'SCORE_FACTORS.json'),
                images=187,raw_GT=380,predictions_count=18700,geometry='L_plus_u'if '_posterior_'in name and '_native_'not in name else'native',
                AP_computed_on_GPU=False,all100_queries_retained=True)
        write(directory/'IMAGE_ROWS.json',image_rows)
        write(directory/'STATE_PROOF.json',dict(schema=PREFIX+'-state-proof-v1',source=reference(__file__),spec=spec_ref,arm=arm,
            loading=loading,before_native610=before,after_native610=after,before_readout=before_head,after_readout=after_head,
            native610_and_readout8_CPU_identity_exact=True,params_gradNone=True,all_parameters_frozen=True,all_modules_eval=True,
            no_optimizer_or_RNG_restored=True,GT_removed_from_forward=True))
        return dict(arm=arm,checkpoint=spec['joint_checkpoints'][arm],qualification=spec['joint_training_CPU_qualification'],
            image_rows=reference(directory/'IMAGE_ROWS.json'),state_proof=reference(directory/'STATE_PROOF.json'),controls=controls,
            images=187,raw_GT=380,all100_fields_saved=True,field_packets=187,full_native610_exact=True,
            readout_state_tensors=8 if arm=='B'else 0,readout_parameters=9441 if arm=='B'else 0,
            effective_native_epoch=loading['effective_native_epoch'],effective_readout_epoch=loading['effective_readout_epoch'],
            no_GPU_AP=True,no_TEST=True,GT_free_inference=True)
    finally:
        model.cpu()
        if readout is not None: readout.cpu()


def run(spec_file):
    started=time.monotonic(); path=Path(spec_file)
    require(path.is_absolute(),'Absolute SPEC required'); path=path.resolve(strict=True); spec=read(path)
    require(spec['schema']==SCHEMA and spec['kind']=='mask2former'and spec['endpoint_kind']in STAGES and
        spec['arms']==list(ARMS)and spec['controls']==list(CONTROLS)and spec['evaluation_images']==187 and
        spec['evaluation_raw_GT']==380 and spec['head_seed_base']==100000 and spec['CPU_affinity']==[0,1]and
        spec['allocator_limit_MiB']==4096 and spec['whole_GPU_owner_limit_MiB']==8192 and
        type(spec['deadline_seconds'])is int and 0<spec['deadline_seconds']<=21600 and
        spec['CPU_numeric_protocol']==NUMERIC and spec['kappa']==1. and spec['eps']==1e-6 and
        spec['endpoint_selection']is False and spec['historically_exposed_DEV']is True and spec['no_TEST']is True and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Prelocked single-stage six-control evaluation SPEC differs')
    source,spec_ref,admission,lineage=source_gate(path,spec)
    output=Path(spec['output_directory'])
    require(output.is_absolute()and not output.exists(),'Fresh absolute endpoint output required'); output.mkdir(parents=True)
    write(output/'START.json',dict(schema=PREFIX+'-start-v1',source=source,spec=spec_ref,GPU_admission=admission,phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'],fixed_effective_native_epoch=lineage['stage']['native_epoch'],
        fixed_B_readout_epoch=lineage['stage']['head_epoch'],controls=list(CONTROLS),arms=list(ARMS),
        joint_training_CPU_qualification=spec['joint_training_CPU_qualification'],checkpoints=spec['joint_checkpoints'],
        GT_free_inference=True,loader_accesses_annotations=True,training_performed=False,optimizer_restored=False,
        AP_computed_on_GPU=False,no_TEST=True,endpoint_selection=False,CPU_numeric_protocol=NUMERIC))
    import psutil
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
    require(psutil.Process().cpu_affinity()==[0,1]and not torch.cuda.is_initialized(),'CPU01/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes'])==80 and CocoDataset.METAINFO['classes'][8]=='boat','boat8/null80 required')
    pref=load(spec['micro_preflight_source'],'_joint_eval_pref')
    native=load(spec['native_loader_source'],'_joint_eval_native_constructor')
    posterior=load(spec['posterior_readout_source'],'_joint_eval_readout')
    old=load(spec['native_helpers'],'_joint_eval_loader'); old.torch=torch
    diag=load(spec['renderer_source'],'_joint_eval_renderer')
    adapter=load(spec['native_adapter_source'],'_joint_eval_inference_adapter')
    dense=load(spec['posterior_dense_source'],'_joint_eval_direct_posterior')
    exporter=load(spec['official_export_source'],'_joint_eval_official_export')
    prepared={arm:prepare_arm(torch,pref,native,posterior,MODELS,revert_sync_batchnorm,Config,
        tuple(CocoDataset.METAINFO['classes']),spec,lineage,arm)for arm in ARMS}
    dataset=CocoDataset(ann_file=str(bound(spec['DEV'])),data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(bound(spec['DEV']))); require(len(dataset)==len(coco.imgs)==187 and len(coco.anns)==380 and
        [int(dataset.get_data_info(i)['img_id'])for i in range(187)]==list(coco.imgs),'Locked exposed ordered DEV187/380 required')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    result,error,cleanup={},None,None
    try:
        for arm in ARMS:
            result[arm]=evaluate_arm(torch,np,masks_api,pref,old,diag,adapter,dense,exporter,spec,spec_ref,
                prepared[arm],dataset,coco,output,started,arm)
            del prepared[arm]
            cleared=pref.clear_gpu(torch); require(cleared['allocated_after']==0,'Serial endpoint CUDA cleanup failed')
            result[arm]['cleanup']=cleared
        del prepared; cleanup=pref.clear_gpu(torch)
        for ref in spec['sources']: bound(ref)
        require(reference(path)==spec_ref and time.monotonic()-started<spec['deadline_seconds'],'Final inputs/deadline differ')
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),no_retry=True); traceback.clear_frames(exc.__traceback__)
        if 'prepared'in locals():
            for model,readout,_,_,_ in prepared.values():
                model.cpu()
                if readout is not None: readout.cpu()
            del prepared
        gc.collect(); cleanup=pref.clear_gpu(torch)
    controls={name:row for arm_result in result.values()for name,row in arm_result['controls'].items()}
    if error is None: require(set(controls)==set(CONTROLS),'Six complete controls required')
    report=dict(schema=PREFIX+'-report-v1',status='QUERY_PIXEL_JOINT_NATIVE_FIXED_ENDPOINT_DEV_EXPORT_COMPLETE_NO_AP'if error is None else
        'QUERY_PIXEL_JOINT_NATIVE_FIXED_ENDPOINT_DEV_EXPORT_STOP',source=source,spec=spec_ref,GPU_admission=admission,phase=spec['phase'],
        endpoint_kind=spec['endpoint_kind'],effective_native_epoch=lineage['stage']['native_epoch'],effective_B_readout_epoch=lineage['stage']['head_epoch'],
        joint_training_CPU_qualification=spec['joint_training_CPU_qualification'],checkpoints=spec['joint_checkpoints'],result=result,controls=controls,
        error=error,cleanup=cleanup,images_per_arm=187,raw_GT_per_arm=380,queries_per_image=100,
        CPU_numeric_protocol=NUMERIC,GT_removed_from_forward=True,loader_accesses_annotations=True,matching_or_loss_executed=False,
        one_native_forward_per_image_per_arm=True,two_passive_no_grad_hooks=True,serial_arms=True,
        B_posterior_geometry='L_plus_u',B_bounded_geometry_used=False,original_teacher_E24_only_initializer_and_BN_reference=True,
        trained_native610_used_for_classes_feature_coefficient_masks=True,historical_cross_arm_class_or_score_equality_assumed=False,
        no_TEST=True,AP_computed_on_GPU=False,official_COCO12_pending_CPU=True,training_performed=False,optimizer_or_RNG_restored=False,
        endpoint_selection=False,formal_resume_eligible=False,efficacy_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False,
        CPU_renderer_GPU_bitwise_claimed=False,certified_GPU_error_bound=False,GPU_original_logits_saved=False,
        elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    write(output/'REPORT.json',report)
    if error is None:
        write(output/'QUALIFIED.json',dict(schema=PREFIX+'-aggregate-v1',
            status='ROOT_QUERY_PIXEL_JOINT_NATIVE_FIXED_ENDPOINT_EVALUATION_PENDING_CPU_READER',
            source=source,spec=spec_ref,report=reference(output/'REPORT.json'),result=result,controls=controls,phase=spec['phase'],
            endpoint_kind=spec['endpoint_kind'],independently_CPU_qualified=False,AP_computed=False,formal_resume_eligible=False))
    require(error is None,'EVALUATION_STOP: '+str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned GPU execution required'); run(args.spec)


if __name__=='__main__':
    main()
