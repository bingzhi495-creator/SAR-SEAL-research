"""Independent CPU qualifier for the direct PointRend-style native carrier.
SOURCE_IMPLEMENTED_NUMERICAL_VALIDATION_PENDING.
Authoring is inert: no native backbone/native30/CUDA trajectory or AP replay.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import types

PREFIX='query-pixel-PointRend-style-native-integration'
QUAL_SCHEMA='ROOT-'+PREFIX+'-CPU-qualification-v1'
QUAL_STATUS='ROOT_QUERY_PIXEL_POINTREND_STYLE_NATIVE_INTEGRATION_CPU_QUALIFIED'
MODES=('native0','global','spatial_halves','spatial_reversed')
HEAD_SHAPES={name:shape for index in (1,2,3) for name,shape in
    [('fc%d.weight'%index,(256,257,1)),('fc%d.bias'%index,(256,))]}
HEAD_SHAPES.update({'predictor.weight':(1,257,1),'predictor.bias':(1,)})
NUMERIC=dict(forward_rtol=1e-4,forward_atol=1e-6,gradient_rtol=1e-3,gradient_atol=1e-7,
    native_projection_rtol=1e-4,native_projection_atol=1e-4)
SEEDS=dict(readout_seed=2026100911,order_seed=2026101021,loss_seed_base=1000000,point_seed_base=3800000)
HEAD_RECIPE=dict(type='AdamW',lr=1e-4,weight_decay=.05,betas=[.9,.999],eps=1e-8)
STAGES={
    'technical':dict(epochs=0,microsteps=4,optimizer_updates=2,raw_GT_visits_per_arm=None,
        initial_native_epoch=24,effective_native_epoch=24,effective_readout_epoch=0,
        details=[0,1,2,3],milestones=[7904,8563],max_deadline=10800,checkpoint_filename='TECHNICAL_ONLY_CPU_SNAPSHOT.pth'),
    'first24':dict(epochs=24,microsteps=17784,optimizer_updates=8892,raw_GT_visits_per_arm=39864,
        initial_native_epoch=24,effective_native_epoch=48,effective_readout_epoch=24,
        details=[0,1,17782,17783],milestones=[7904,8563],max_deadline=50400,checkpoint_filename='FINAL_PHASE_E24.pth'),
    'warm48':dict(epochs=48,microsteps=35568,optimizer_updates=17784,raw_GT_visits_per_arm=79728,
        initial_native_epoch=48,effective_native_epoch=96,effective_readout_epoch=72,
        details=[0,1,35566,35567],milestones=[15808,17126],max_deadline=100800,checkpoint_filename='FINAL_PHASE_E48_EFFECTIVE_E96.pth')}


def require(ok,message):
    if not ok: raise AssertionError(message)


def expect(value,expected,message):
    require(all(k in value and value[k]==v and (type(v)is not int or type(value[k])is int)
                for k,v in expected.items()),message)


def read(path):
    require(Path(path).is_absolute(),'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    require(set(ref)=={'path','bytes','sha256'} and Path(ref['path']).is_absolute() and
        type(ref['bytes'])is int and ref['bytes']>0 and len(ref['sha256'])==64 and
        set(ref['sha256'])<=set('0123456789abcdef') and reference(ref['path'])==ref,'Pinned input differs')
    return Path(ref['path'])


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    require(path.suffix=='.py' and len(source)==ref['bytes'] and
        hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


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
    expect(raw,dict(status='ACTUAL_OWNED_TERMINAL',returncode=0,root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0),abort=None,error=None,
        wait_errors=False,GPU_observation_errors=[],whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False,GPU_scope='whole GPU observed, not per-process WDDM allocation'),
        'STOP before hashes/imports: actual raw terminal0 required')
    peak=raw.get('whole_GPU_observed_peak_MiB')
    require(type(raw.get('original_identity'))is dict and bool(raw['original_identity']) and type(raw['returncode'])is int and
        type(raw['held_handle_exit']['actual_exit_code'])is int and
        raw.get('numeric_birth_absence',{}).get('original_birth_absent')is True and
        type(peak)in(int,float) and math.isfinite(peak) and 0<=peak<=8192,
        'Raw original identity/peak differs')


def raw_gate(ref,phase):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and path.parent.name==phase,
        'Absolute phase-bound PARENT_RECEIPT locator required before read')
    raw=read(path); terminal(raw,phase,path)
    return raw


def same_tree(torch,first,second):
    if isinstance(first,torch.Tensor):
        return isinstance(second,torch.Tensor) and first.shape==second.shape and first.dtype==second.dtype and torch.equal(first,second)
    if type(first)is not type(second): return False
    if isinstance(first,dict): return set(first)==set(second) and all(same_tree(torch,first[k],second[k]) for k in first)
    if isinstance(first,(list,tuple)): return len(first)==len(second) and all(same_tree(torch,a,b) for a,b in zip(first,second))
    return first==second


def cold_head(torch, seed):
    """Independent CPU construction and author-style reinitialization; no forward call."""
    require(type(seed) is int and 0 <= seed < 2 ** 63, 'Private head seed invalid')
    private = torch.Generator(device='cpu').manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(private.get_state())
        layers = {('fc%d' % index): torch.nn.Conv1d(257,256,1,bias=True,device='cpu',dtype=torch.float32)
                  for index in (1,2,3)}
        layers['predictor'] = torch.nn.Conv1d(257,1,1,bias=True,device='cpu',dtype=torch.float32)
        private.set_state(torch.get_rng_state())
        with torch.no_grad():
            for name, layer in layers.items():
                layer.weight.normal_(0., .001 if name == 'predictor' else math.sqrt(2./256.), generator=private)
                layer.bias.zero_()
        return {prefix+'.'+name: value.detach().clone() for prefix, layer in layers.items()
                for name, value in layer.state_dict().items()}


def point_replay(torch, helper, F256, M200, identity, before, query_ids, raw_GT, indices):
    """Independent conditional matmul/Bernoulli loss and local original-L VJP/2.
    Native F is a saved constant; native30/backbone/total F/M gradients are not replayed.
    """
    H,W = identity['original_shape']; G = query_ids.numel()
    helper.checked_tensor(torch,F256,(1,256,200,200),torch.float32,'Saved native F')
    helper.checked_tensor(torch,M200,(1,100,200,200),torch.float32,'Saved native M')
    helper.checked_tensor(torch,query_ids,(G,),torch.int64,'Matched raw-GT query IDs')
    helper.checked_tensor(torch,raw_GT,(G,H,W),torch.bool,'Complete original raw-GT raster')
    helper.checked_tensor(torch,indices,(G,12544),torch.int64,'Duplicate-preserving integer point indices')
    require(G <= 100 and torch.unique(query_ids).numel() == G and
            not bool(((query_ids < 0)|(query_ids >= 100)).any()) and
            not bool(((indices < 0)|(indices >= H*W)).any()),'Point/query identity or range differs')
    require(set(before) == set(HEAD_SHAPES),'Exact eight point-head tensors required')
    parameters = {}
    for name, shape in HEAD_SHAPES.items():
        helper.checked_tensor(torch,before[name],shape,torch.float32,name)
        parameters[name] = before[name].detach().clone().requires_grad_(True)
    L = helper.render(torch,M200.detach(),identity).detach().requires_grad_(True)
    if G:
        pixel = helper.render(torch,F256.detach(),identity)
        fine = pixel[0].flatten(1).t()[indices.reshape(-1)].reshape(G,12544,256)
        coarse = L[0,query_ids].flatten(1).gather(1,indices).unsqueeze(-1)
        target = raw_GT.flatten(1).gather(1,indices).float()
        hidden = fine
        for index in (1,2,3):
            name = 'fc%d' % index
            joined = torch.cat((hidden,coarse),dim=-1)
            activated = joined @ parameters[name+'.weight'][:,:,0].t() + parameters[name+'.bias']
            hidden = torch.where(activated > 0.,activated,torch.zeros_like(activated))
        logits = (torch.cat((hidden,coarse),dim=-1) @
                  parameters['predictor.weight'][:,:,0].t() + parameters['predictor.bias'])[:,:,0]
        # Direct output, never coarse_L + predictor output.
        loss = (torch.logaddexp(torch.zeros_like(logits),logits)-logits*target).mean(1).mean()
    else:
        logits = L.new_empty((0,12544))
        loss = L.flatten()[:0].sum()
        for value in parameters.values(): loss = loss + value.flatten()[:0].sum()
    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(logits).all()),'CPU direct point BCE finite')
    gradients = torch.autograd.grad(loss/2.,[L]+list(parameters.values()),allow_unused=False)
    if not G:
        require(float(loss.detach()) == 0. and all(bool(torch.isfinite(v).all()) and
                torch.count_nonzero(v) == 0 for v in gradients),'G0 connected loss/eight gradients must be exact zero')
    return dict(loss=float(loss.detach()),point_logits=logits.detach(),
                local_L_gradient=gradients[0][0,query_ids].detach(),
                head_gradients={n:g.detach() for n,g in zip(parameters,gradients[1:])})



def checked_tensor(torch, value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
        value.layout == torch.strided and tuple(value.shape) == tuple(shape) and value.dtype == dtype,
        label + ': CPU tensor shape/dtype/layout differs')
    require(bool(torch.isfinite(value).all()), label + ': nonfinite tensor')
    return value


def compare(torch, actual, expected, rtol, atol, label):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype, label + ': shape/dtype')
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), label + ': finite')
    difference = (actual - expected).double()
    norm = float(expected.double().norm())
    summary = dict(max_absolute_error=float(difference.abs().max()) if difference.numel() else 0.,
                   relative_L2_error=float(difference.norm()) / max(norm, 1e-30), rtol=rtol, atol=atol)
    require(torch.allclose(actual, expected, rtol=rtol, atol=atol), label + ': numerical mismatch')
    return summary


def render(torch, tensor, identity):
    current = torch.nn.functional.interpolate(tensor, size=(800, 800), mode='bilinear', align_corners=False)
    height, width = identity['actual_img_shape']
    current = current[:, :, :height, :width]
    return torch.nn.functional.interpolate(current, size=tuple(identity['original_shape']),
        mode='bilinear', align_corners=False)


def same_state(torch, first, second, label):
    require(set(first) == set(second) and all(first[n].shape == second[n].shape and first[n].dtype == second[n].dtype and
        torch.equal(first[n], second[n]) for n in first), label)


def original_teacher_state(torch, spec, loading):
    legacy = load(spec['evaluation_source'], '_joint_original_teacher_lineage')
    teacher = legacy.qualified_training(spec)['native0']; train, q = teacher['training_spec'], teacher['qualification']
    for ref in train['sources'] + [q[k] for k in ('source','reader_source','spec','report','start','GPU_admission','owner_driver')] + list(q['model_proof']['proof_refs'].values()):
        require(ref in spec['sources'], 'Original teacher/proof ref absent from current owner closure')
    old = load(spec['reused_CPU_reader'], '_joint_independent_micro_reader')
    common = load(spec['training_CPU_reader'], '_joint_independent_common_reader')
    initializer = load(spec['initializer_source'], '_joint_independent_initializer')
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    initialized = read(bound(train['initializer_qualification']))
    model, initial = common.build_initial_native(torch, old, initializer, MODELS, revert_sync_batchnorm,
        Config, train, initialized, loading['original_initialization'])
    load(spec['native_loader_source'],'_joint_native_library_binding_only').imported_library_gate(torch,model,spec)
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==bound(spec['interpolation_SOURCE']).resolve(),'Actual interpolation SOURCE differs')
    policy = old.independent_policy(torch, model, 'mask2former')
    require(policy['state_entries'] == 610 and policy['parameter_tensors'] == 451 and policy['trainable_tensors'] == 345 and
        policy['frozen_tensors'] == 106 and len(policy['BN_buffer_names']) == 159 and len(policy['BN_modules']) == 53, 'Typed native policy differs')
    cp = torch.load(str(bound(spec['checkpointmap']['native0'])), map_location='cpu'); anchor = cp['state_dict']
    expect(cp['meta'], dict(schema=common.CP_SCHEMA, mode='native0', completed_epoch=24, source=spec['training_source'],
        spec=q['spec'], phase=q['phase'], global_microsteps=17784, global_updates=8892, consumed_raw_GT=39864,
        no_pending_accumulation_gradients=True, eligible_for_automatic_resume=False), 'Qualified E24 native metadata differs')
    require(set(anchor) == set(initial) and len(anchor) == 610, 'Bare native610 required')
    for name, value in anchor.items(): checked_tensor(torch, value, initial[name].shape, initial[name].dtype, name)
    proof = read(bound(q['model_proof']['proof_refs']['state_proof'])); common.cpu_state(torch, old, anchor, proof['after'], 'Qualified teacher610')
    require(all(torch.equal(anchor[n], initial[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']), 'Teacher typedBN initializer exactness differs')
    groups, audit = old.independent_optimizer(torch, model, train['initializers']['mask2former'], policy)
    return old, common, anchor, policy, groups, audit


def GT_population(coco):
    require(len(coco.imgs)==741 and len(coco.anns)==1661,'Complete original FIT741/1661 required')
    cache={}; total=0
    for index,image_id in enumerate(coco.imgs):
        require(type(image_id)is int and image_id>0,'Canonical integer image ID required')
        image=coco.load_imgs([image_id])[0]; H,W=image['height'],image['width']
        ids=coco.get_ann_ids(img_ids=[image_id])
        require(type(H)is int and type(W)is int and H>0 and W>0 and 0<=len(ids)<=100 and
            len(ids)==len(set(ids)) and all(type(i)is int and i>0 for i in ids),'Complete raw instance identities required')
        cache[image_id]=dict(dataset_index=index,ids=ids,original_shape=[H,W],annotations=coco.load_anns(ids))
        total+=len(ids)
    require(total==1661,'Raw GT population differs')
    return cache


def original_masks(np,coco,info):
    H,W=info['original_shape']
    return np.stack([np.zeros((H,W),np.uint8) if a.get('segmentation')==[] else coco.annToMask(a)
                     for a in info['annotations']]) if info['annotations'] else np.zeros((0,H,W),np.uint8)


def digest_record(record,shape,dtype):
    require(set(record)=={'shape','dtype','sha256'} and record['shape']==list(shape) and record['dtype']==dtype and
        len(record['sha256'])==64 and set(record['sha256'])<=set('0123456789abcdef'),'Gradient digest shape/dtype/SHA differs')


def source_gate(path,spec):
    receipt=path.parent/'finite_point_cache_execution_20261005_v2'/'runs'/spec['phase']/'PARENT_RECEIPT.json'
    raw=raw_gate(dict(path=str(receipt)),spec['phase'])
    template=read(spec['native_template_spec']['path'])
    require(set(template['training_receiptmap'])==set(template['training_phasemap'])==set(MODES),'Original four raw gates required')
    for mode in MODES: raw_gate(template['training_receiptmap'][mode],template['training_phasemap'][mode])
    raw_gate(template['restoration_receipt'],template['restoration_phase'])
    raw_gate(spec['point_synthetic_CPU_receipt'],spec['point_synthetic_CPU_phase'])
    for row in spec.get('prior_receipts',[]):
        require(set(row)=={'receipt','phase'},'Explicit prior raw identity required'); raw_gate(row['receipt'],row['phase'])
    if spec['stage_kind']!='technical': raw_gate(spec['C_tiny_receipt'],spec['C_tiny_phase'])
    if spec['stage_kind']!='technical': raw_gate(spec['C_tiny_CPU_receipt'],spec['C_tiny_CPU_phase'])
    if spec['stage_kind']=='warm48': raw_gate(spec['first24_receipt'],spec['first24_phase'])
    if spec['stage_kind']=='warm48': raw_gate(spec['first24_CPU_receipt'],spec['first24_CPU_phase'])
    # Current, original/restoration, optional and own-C raws all precede any SHA/import.
    reader,source,spec_ref=reference(__file__),reference(Path(__file__).with_name('train.py')),reference(path)
    owner=read(spec['GPU_admission_file']); hashes=owner['bound_source_sha256']
    require(spec['CPU_reader']==reader and spec['entry_source']==source and reader in spec['sources'] and source in spec['sources'] and
        all(hashes.get(r['path'])==r['sha256'] for r in (reader,source,spec_ref)) and
        raw['admission_sha256']==reference(spec['GPU_admission_file'])['sha256'],'Original own-C owner reader/entry/SPEC joins differ')
    producer=load(source,'_own_C_provenance_only')
    actual_source,actual_spec,admission,context=producer.source_gate(path,spec)
    require(actual_source==source and actual_spec==spec_ref and context['template']==template and
        context['stage']==STAGES[spec['stage_kind']] and raw['admission_sha256']==admission['sha256'],'Own-C SOURCE/stage/raw admission differs')
    return producer,dict(source=source,reader_source=reader,spec=spec_ref,parent_receipt=reference(receipt),GPU_admission=admission,phase=spec['phase']),context


def CPU_owner_gate(spec_path,output,admission_path,spec,refs):
    require(admission_path.is_absolute(),'Explicit absolute own CPU admission required')
    owner=read(admission_path); admission=reference(admission_path)
    expect(owner,dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE',automatic_retry=False,allowed_cpus=[0,1],cuda_visible_devices='',
        whole_GPU_used_limit_MiB=8192,driver_sha256='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'),
        'Actual own-C CPU owner execution policy differs')
    require(type(owner['phase']) is str and owner['phase']!=spec['phase'] and type(owner['qualified_leaf_deadline_seconds']) is int and
        owner['qualified_leaf_deadline_seconds']>0 and Path(owner['entry']).resolve(strict=True)==Path(refs['reader_source']['path']) and
        owner['entry_args']==['--spec',str(spec_path),'--acceptance',str(output),'--CPU-admission',str(admission_path),'--owner-execute'],
        'Actual own-C CPU phase/entry/output arguments differ')
    require(all(owner['bound_source_sha256'].get(r['path'])==r['sha256'] for r in spec['sources']+[refs['source'],refs['reader_source'],refs['spec']]),
        'Complete SPEC SOURCE closure must be bound to actual CPU owner')
    return admission


def warm_initial_states(torch,spec,context,old,common,anchor,policy):
    if spec['stage_kind']!='warm48': return anchor,cold_head(torch,spec['readout_seed'])
    q=context['prerequisites'][-1]; row=q['result']['C']; cp=torch.load(str(bound(spec['warm_checkpoint'])),map_location='cpu')
    require(spec['warm_checkpoint']==row['checkpoint'] and cp['schema']==PREFIX+'-checkpoint-v1','Only own qualified first24 C checkpoint may initialize warm48')
    expect(cp['meta'],dict(source=q['source'],spec=q['spec'],phase=q['phase'],arm='C',stage_kind='first24',
        completed_phase_epochs=24,effective_native_epoch=48,effective_readout_epoch=24,phase_microsteps=17784,
        phase_updates=8892,phase_raw_GT_visits=39864,no_pending_accumulation_gradients=True,formal_resume_eligible=False,
        technical_weights_loaded=False,old_optimizer_restored=False,old_RNG_restored=False,**SEEDS),'Qualified own-C first24 checkpoint metadata differs')
    native,head=cp['native_state_dict'],cp['readout_state_dict']; proof=read(bound(row['endpoint_proof']))
    require(set(native)==set(anchor) and len(native)==610 and tuple(head)==tuple(HEAD_SHAPES),'Own C full warm610/head8 names differ')
    for n,v in native.items(): checked_tensor(torch,v,anchor[n].shape,anchor[n].dtype,n)
    for n,v in head.items(): checked_tensor(torch,v,HEAD_SHAPES[n],torch.float32,n)
    require(all(torch.equal(native[n],anchor[n]) for n in policy['frozen_names']+policy['BN_buffer_names']),'Original typed BN must remain exact in own C weights')
    common.cpu_state(torch,old,native,proof['after_native610'],'Own first24 warm native610')
    common.cpu_state(torch,old,head,proof['after_readout'],'Own first24 warm head8')
    require(cp['readout_optimizer_state_dict'] is not None and q['arms']['C']['initial_native610_CPU_exact'] is True and
        q['arms']['C']['endpoint_native610_CPU_finite'] is True,'Own-C actual first24 CPU state qualification required')
    # Optimizer/order/external RNG containers are inspected, never restored here.
    return native,head


def trace_check(torch,np,cv2,old,common,spec,row,cache,policy,anchor,coco):
    stage=STAGES[spec['stage_kind']]; technical=spec['stage_kind']=='technical'; epoch_count=stage['epochs'] or 1
    orders=read(bound(row['order'])); epochs=read(bound(row['epochs'])); require(len(orders)==len(epochs)==epoch_count,'Complete own-stage order/epochs required')
    generator=torch.Generator(device='cpu').manual_seed(spec['order_seed']); image_ids=list(cache)
    counts={kind:{n:0 for n in ns} for kind,ns in (('native',policy['active_names']),('readout',HEAD_SHAPES))}
    activity=copy.deepcopy(counts); boundary=None; joins=0; GT_visits=0; last_external=None; detail_rows={}; micro_count=0
    with bound(row['trace']).open(encoding='utf-8') as stream:
        for epoch in range(epoch_count):
            before=generator.get_state(); order=[0,1,0,1] if technical else torch.randperm(741,generator=generator).tolist()
            after=generator.get_state().clone()
            expect(orders[epoch],dict(phase_epoch=epoch,order_seed=spec['order_seed'],dataset_indices=order,
                image_ids=[image_ids[i] for i in order],private_CPU_RNG_before=old.tensor_digest(torch,before,'uint8'),
                private_CPU_RNG_after=old.tensor_digest(torch,after,'uint8')),'Independent private order differs')
            losses={k:[] for k in ('native','auxiliary','total')}; epoch_GT=0; initial_updates=micro_count//2
            for within,index in enumerate(order):
                g=micro_count; text=stream.readline(); require(bool(text),'Trace missing micro'); t=json.loads(text)
                stepped=(g+1)%2==0; updates=(g+1)//2; info=cache[image_ids[index]]; ids=info['ids']; identity=t['identity']
                expect(t,dict(arm='C',global_micro=g,phase_epoch=epoch,effective_native_epoch_index=stage['initial_native_epoch']+epoch,
                    within_epoch=within,auxiliary_weight=1.,accumulation_divisor=2,optimizer_update=stepped,phase_updates=updates,
                    readout_updates=updates,scheduler_u_before_step=updates-int(stepped),native_lr_factor=common.schedule_factor(spec['schedule'],updates-int(stepped)),
                    C_readout_lr=1e-4,actual_backward_calls=1,frozen_BN_parameter_grads_None=True,detailed_observation=g in stage['details'],
                    private_seed=spec['loss_seed_base']+g,point_seed=spec['point_seed_base']+g,point_count_per_GT=12544,sampling_with_replacement=True,
                    external_RNG_restored=True,dynamic_current_matching=True,gradients_accumulated_before_clip=True),'Trace recipe/counters differ')
                require(identity['image_id']==image_ids[index] and identity['dataset_index']==index and identity['ann_ids']==ids and
                    identity['original_shape']==info['original_shape'] and [r['annotation_id'] for r in identity['parser_rows'] if r['native_parser_retained']]==ids,'All raw/parser identities differ')
                H,W=identity['original_shape']; h,w=identity['actual_img_shape']; require(0<h<=800 and 0<w<=800,'Native crop dimensions differ')
                if 'padded_digest' not in info:
                    raw=original_masks(np,coco,info); padded=np.zeros((len(ids),800,800),np.uint8)
                    for j,mask in enumerate(raw): padded[j,:h,:w]=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
                    info.update(padded_digest=old.tensor_digest(torch,torch.from_numpy(padded),'uint8'),actual_shape=[h,w])
                require(info['actual_shape']==[h,w] and t['GT']['ann_ids']==ids and t['GT']['complete_all_raw_annotations'] is True and
                    t['GT']['masks']==info['padded_digest'],'Independent complete resized raw GT differs')
                md=t['metadata']; expect(md,dict(arm='C',auxiliary_weight=1.,native_atoms=30,raw_annotation_ids=ids,processed_GT_to_raw=list(range(len(ids))),
                    processed_GT_observation=dict(calls=1,correspondence_validator_passed=True),native_forward_counts=dict(head_forward_started=1,head_forward_finished=1,pixel_decoder=1,mask_embed=10),
                    fixed_teacher_matching_used=False,point_indices_sampled_here=False,backward_or_optimizer_step_performed=False,auxiliary_accumulation_scaled=False,
                    direct_point_logits=True,native_L_added_to_point_logits=False,final_e0_direct_head_input=False),'Current native30/GT/direct-head interface differs')
                require(set(md['native_atom_values'])==set(atomic_keys()) and len(md['current_stage_matches'])==10,'Thirty native atoms/ten current assignments required')
                for match in md['current_stage_matches']:
                    qs,gs=match['positive_query_indices'],match['assigned_GT_indices']
                    require(match['positives']==len(qs)==len(gs)==len(set(qs))==len(ids) and sorted(gs)==list(range(len(ids))) and
                        all(type(q)is int and 0<=q<100 for q in qs),'All-GT current assignment differs')
                final=md['current_stage_matches'][-1]; mapping=dict(zip(final['assigned_GT_indices'],final['positive_query_indices']))
                require(t['final_query_ids']==md['final_query_by_raw_GT']==[mapping[j] for j in range(len(ids))],'Actual final raw-GT/query identity differs')
                atoms=md['native_atom_values']; require(all(type(v)in(int,float) and math.isfinite(v) for v in atoms.values()),'Nonfinite native atoms')
                require(math.isclose(t['losses']['native'],math.fsum(atoms.values()),rel_tol=1e-4,abs_tol=1e-4) and
                    math.isclose(t['losses']['total'],t['losses']['native']+t['losses']['auxiliary'],rel_tol=1e-5,abs_tol=1e-5),'Observed lambda1 arithmetic differs')
                require(set(t['losses'])==set(losses),'Three losses required')
                for key,value in t['losses'].items(): require(type(value)in(int,float) and math.isfinite(value),'Loss finite required'); losses[key].append(value)
                for kind,key in (('native','native_gradient_norms'),('readout','readout_gradient_norms')):
                    norms=t[key]; require(set(norms)==set(counts[kind]) and all(v is None or type(v)in(int,float) and math.isfinite(v) and v>=0 for v in norms.values()),'Gradient membership/norm finite differs')
                    require(kind!='readout' or all(v is not None for v in norms.values()),'All C8 head gradients connected; zero permitted')
                    for n,v in norms.items(): activity[kind][n]+=int(v is not None and v>0); counts[kind][n]+=int(stepped and v is not None)
                require(set(t['BN_module_training'])==set(policy['BN_modules']) and not any(t['BN_module_training'].values()),'Typed BN53 must remain eval')
                norm=t['combined_preclip_norm']; require(norm is None if not stepped else type(norm)in(int,float) and math.isfinite(norm) and norm>=0,'Observed combined norm/clip location differs')
                end,join=t['odd_epoch_end_pending_gradients'],t['pending_boundary_gradient_join']
                require((end is not None)==(within==740 and (g+1)%2==1) and (join is not None)==(within==0 and g%2==1),'Odd741 proof location differs')
                if end is not None:
                    require(set(end)=={'native','readout'},'Boundary domains differ')
                    for kind,ns in (('native',policy['active_names']),('readout',HEAD_SHAPES)):
                        require(set(end[kind])==set(ns),'Boundary membership differs')
                        for n,d in end[kind].items():
                            if d is not None: digest_record(d,anchor[n].shape if kind=='native' else HEAD_SHAPES[n],'float32')
                    boundary=(g,end)
                if join is not None:
                    require(boundary is not None and boundary[0]==g-1,'Previous odd epoch identity differs')
                    expect(join,dict(previous_global_micro=g-1,after_previous_odd_epoch=boundary[1],before_next_forward=boundary[1],exact=True,
                        SOURCE_observed_not_CPU_gradient_replay=True),'Pending gradient continuity differs'); boundary=None; joins+=1
                require(t['external_RNG_before']==t['external_RNG_after'] and (last_external is None or last_external==t['external_RNG_before']),'External RNG continuity differs')
                for key in ('private_RNG_before','private_RNG_after','external_RNG_before','external_RNG_after'):
                    require(len(t[key])==2,'CPU/CUDA RNG packets required')
                    for d in t[key]: require(d['dtype']=='uint8' and len(d['shape'])==1 and len(d['sha256'])==64,'RNG packet differs')
                last_external=t['external_RNG_after']; epoch_GT+=len(ids); GT_visits+=len(ids); micro_count+=1
                if g in stage['details']: detail_rows[g]=t
            expect(epochs[epoch],dict(phase_epoch=epoch,images=len(order),raw_GT=epoch_GT,phase_microsteps=micro_count,
                epoch_updates=micro_count//2-initial_updates,phase_updates=micro_count//2,pending_accumulation_microsteps=micro_count%2),'Globalacc2 epoch counters differ')
            require(technical or epoch_GT==1661,'Complete epoch GT required')
            require(set(epochs[epoch]['training_loss_summary'])==set(losses),'Three epoch summaries required')
            for key,values in losses.items():
                expected=math.fsum(values); summary=epochs[epoch]['training_loss_summary'][key]
                expect(summary,dict(mean_denominator='%d image-averaged losses'%len(order)),'Loss denominator differs')
                require(math.isclose(summary['image_loss_sum'],expected,rel_tol=1e-12,abs_tol=1e-9) and
                    math.isclose(summary['image_loss_mean'],expected/len(order),rel_tol=1e-12,abs_tol=1e-12),'Trace-derived epoch loss differs')
        require(not stream.readline() and micro_count==stage['microsteps'] and joins==stage['epochs']//2 and boundary is None and
            (stage['raw_GT_visits_per_arm'] is None or GT_visits==stage['raw_GT_visits_per_arm']),'Complete counters/odd741/no dangling accumulation required')
    return dict(counts=counts,activity=activity,last_external=last_external,private_order_state=after,detail_rows=detail_rows,GT_visits=GT_visits)


def checkpoint_check(torch,old,common,spec,refs,row,anchor,original_anchor,policy,groups,audit,initial_head,trace):
    stage=STAGES[spec['stage_kind']]; updates=stage['optimizer_updates']
    cp=torch.load(str(bound(row['checkpoint'])),map_location='cpu'); proof=read(bound(row['endpoint_proof'])); start=read(bound(row['start']))
    require(set(cp)=={'schema','meta','native_state_dict','readout_state_dict','native_optimizer_state_dict',
        'readout_optimizer_state_dict','private_order_state','external_rng_state'} and cp['schema']==PREFIX+'-checkpoint-v1' and
        Path(row['checkpoint']['path']).name==stage['checkpoint_filename'],'Complete own-C checkpoint container differs')
    counts=dict(stage_kind=spec['stage_kind'],completed_phase_epochs=stage['epochs'],effective_native_epoch=stage['effective_native_epoch'],
        effective_readout_epoch=stage['effective_readout_epoch'],phase_microsteps=stage['microsteps'],phase_updates=updates,phase_raw_GT_visits=trace['GT_visits'])
    m=cp['meta']; expect(m,dict(source=refs['source'],spec=refs['spec'],phase=spec['phase'],arm='C',teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'],initial_native_epoch=stage['initial_native_epoch'],**counts,policy=policy,
        native_optimizer_param_names=[[n] for n in policy['active_names']],readout_optimizer_param_names=[[n] for n in HEAD_SHAPES],
        base_lrs=[g['lr'] for g in groups],stored_native_lr_factor=common.schedule_factor(spec['schedule'],updates-1),schedule=spec['schedule'],
        readout_optimizer=HEAD_RECIPE,**SEEDS,auxiliary_weight=1.,direct_point_logits=True,native_L_added_to_point_logits=False,
        accumulation=2,combined_clip_max_norm=.01,fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,
        old_optimizer_restored=False,old_RNG_restored=False,technical_weights_loaded=False,no_pending_accumulation_gradients=True,
        formal_resume_eligible=False,AP_computed=False),'Own-C checkpoint identity/recipe differs')
    require(json.dumps(m['native_optimizer'],sort_keys=True)==json.dumps(audit,sort_keys=True),'Independent original native AdamW audit differs')
    artifacts={k:row[k] for k in ('trace','order','epochs','start','audits')}; require(m['artifacts']==artifacts,'CP primary artifact joins differ')
    state,head=cp['native_state_dict'],cp['readout_state_dict']; require(set(state)==set(anchor) and len(state)==610 and tuple(head)==tuple(HEAD_SHAPES),'Strict endpoint native610/C8 names differ')
    for n,v in state.items(): checked_tensor(torch,v,anchor[n].shape,anchor[n].dtype,n)
    for n,v in head.items(): checked_tensor(torch,v,HEAD_SHAPES[n],torch.float32,n)
    require(all(torch.equal(state[n],original_anchor[n]) for n in policy['frozen_names']+policy['BN_buffer_names']),'Original BN106/159 endpoint exactness differs')
    expect(proof,dict(schema=PREFIX+'-endpoint-proof-v1',source=refs['source'],spec=refs['spec'],arm='C',checkpoint=row['checkpoint'],artifacts=artifacts,
        policy=policy,positive_accumulated_gradient_microcounts=trace['activity'],optimizer_step_nonNone_gradient_counts=trace['counts'],**counts,
        frozen_BN106_affine159_buffers_exact=True,no_pending_accumulation_gradients=True,odd741_gradient_joins=stage['epochs']//2,
        fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,native_loss_atoms=30,ten_current_assignments_per_micro=True,
        one_total_backward_per_micro=True,four_audit_microsteps=stage['details'],CPU_checkpoint_readback_exact=True,technical_weights_loaded=False,
        old_optimizer_restored=False,old_RNG_restored=False,formal_resume_eligible=False,native30_CPU_replay_claimed=False,
        full_optimizer_trajectory_CPU_replay_claimed=False,conditioning_isolated_claimed=False,compute_matched_claimed=False),'Own-C endpoint proof differs')
    common.cpu_state(torch,old,anchor,proof['before_native610'],'Independent own-stage initial native610')
    common.cpu_state(torch,old,state,proof['after_native610'],'Endpoint native610')
    require(proof['changed_native']=={n:not torch.equal(v,anchor[n]) for n,v in state.items()} and
        proof['teacher_loading']==start['teacher_loading'],'Changed-state/teacher-loading joins differ')
    expect(start,dict(schema=PREFIX+'-start-state-v1',source=refs['source'],spec=refs['spec'],arm='C',stage_kind=spec['stage_kind'],policy=policy,
        readout_optimizer=HEAD_RECIPE,fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,
        old_optimizer_restored=False,old_RNG_restored=False,technical_weights_loaded=False,phase_microsteps=0,phase_updates=0,
        initial_native_epoch=stage['initial_native_epoch'],phase_initial_epoch=0,auxiliary_weight=1.),'Actual fresh moment/order stage start differs')
    common.cpu_state(torch,old,anchor,start['actual_native610'],'Actual start native610')
    require(common.optimizer_snapshot(torch,old,dict(state={},param_groups=groups),[[n] for n in policy['active_names']])==start['actual_native_optimizer'] and
        json.dumps(start['native_optimizer'],sort_keys=True)==json.dumps(audit,sort_keys=True) and start['base_lrs']==[g['lr'] for g in groups],'Actual fresh native optimizer snapshot differs')
    common.validate_moments(torch,cp['native_optimizer_state_dict'],common.groups_at(groups,spec['schedule'],updates-1),policy['active_names'],state,'AdamW',audit['recipe'],trace['counts']['native'])
    require(common.optimizer_snapshot(torch,old,cp['native_optimizer_state_dict'],m['native_optimizer_param_names'])==proof['actual_native_optimizer'],'Independent native345 moment snapshot differs')
    require(common.check_rng_tensors(torch,old,cp['external_rng_state'])==trace['last_external'],'CP external RNG differs')
    private=cp['private_order_state']; expect(private,dict(seed=spec['order_seed'],generated_phase_epochs=stage['epochs']),'Fresh private order counters differ')
    require(set(private)=={'seed','generated_phase_epochs','torch_CPU_generator_state'} and torch.equal(private['torch_CPU_generator_state'],trace['private_order_state']),'CP independent private order state differs')
    parameters=[torch.nn.Parameter(initial_head[n].clone()) for n in HEAD_SHAPES]
    optimizer=torch.optim.AdamW([dict(params=[p]) for p in parameters],lr=1e-4,weight_decay=.05,betas=(.9,.999),eps=1e-8)
    expected_groups=optimizer.state_dict()['param_groups']; actual=cp['readout_optimizer_state_dict']
    require(actual['param_groups']==expected_groups and len(actual['state'])==8 and set(actual['state'])=={g['params'][0] for g in expected_groups},'Fixed-LR own-C8 AdamW groups/moment membership differs')
    for n,group in zip(HEAD_SHAPES,expected_groups):
        moments=actual['state'][group['params'][0]]; require(set(moments)=={'step','exp_avg','exp_avg_sq'},'C8 moment keys differ')
        checked_tensor(torch,moments['step'],(),torch.float32,'C step'); require(float(moments['step'])==trace['counts']['readout'][n]==updates,'Own C moment step count differs')
        for key in ('exp_avg','exp_avg_sq'): checked_tensor(torch,moments[key],HEAD_SHAPES[n],torch.float32,key)
        require(bool((moments['exp_avg_sq']>=0).all()),'C negative squared moment')
    require(common.optimizer_snapshot(torch,old,actual,m['readout_optimizer_param_names'])==proof['actual_readout_optimizer'],'Independent C8 moment snapshot differs')
    common.cpu_state(torch,old,initial_head,proof['before_readout'],'Own-stage initial C8')
    common.cpu_state(torch,old,head,proof['after_readout'],'Endpoint C8'); common.cpu_state(torch,old,initial_head,start['initial_readout'],'Actual own-stage initial C8')
    expect(row,dict(arm='C',**counts,native_parameter_tensors=451,native_state_tensors=610,native_trainable_tensors=345,readout_parameter_tensors=8,
        readout_parameters=198402,fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,
        odd741_gradient_joins=stage['epochs']//2,no_pending_accumulation_gradients=True),'Actual C result counters/population differ')
    loading=start['teacher_loading']; original=loading['original_native24_loading'] if spec['stage_kind']=='warm48' else loading
    expect(original,dict(schema='native-finetune24-fresh-optimizer-loading-proof-v1',teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'],teacher_training_source=spec['training_source'],teacher_epoch=24,
        strict_full_native610_loaded=True,typed_BN_initializer_exact=True,native_policy=policy,old_optimizer_restored=False,old_RNG_restored=False,
        old_automatic_resume_eligible=False,fresh_phase_not_formal_resume=True,registered_adapter_tensors=0),'Original native24 loading flags differ')
    common.cpu_state(torch,old,original_anchor,original['teacher_state'],'Original E24 anchor loading')
    if spec['stage_kind']=='warm48': expect(loading,dict(weights_warm_start=True,warm_checkpoint=spec['warm_checkpoint'],
        first24_CPU_qualification=spec['first24_CPU_qualification'],initial_native_epoch=48,effective_initial_readout_epoch=24,
        old_optimizer_restored=False,old_RNG_restored=False,technical_weights_loaded=False),'Own-C weights-only warm loading differs')
    return cp


def audit_check(torch,np,coco,old,spec,row,cache,cp,initial_head,trace):
    stage=STAGES[spec['stage_kind']]; require(len(row['audits'])==4,'Exactly four actual audit packets required')
    previous=None; summaries=[]; helper=types.SimpleNamespace(checked_tensor=checked_tensor,render=render)
    for g,ref in zip(stage['details'],row['audits']):
        packet=torch.load(str(bound(ref)),map_location='cpu'); t=trace['detail_rows'][g]; identity=packet['identity']; H,W=identity['original_shape']
        expect(packet,dict(schema=PREFIX+'-audit-v1',arm='C',global_micro=g,identity=t['identity'],metadata=t['metadata'],auxiliary_gradient_scale=.5,
            native_gradient_replay_claimed=False,auxiliary_loss=t['losses']['auxiliary'],native_total=t['losses']['native'],optimizer_update=t['optimizer_update'],
            native_updates=t['phase_updates'],readout_updates=t['readout_updates'],combined_preclip_norm=t['combined_preclip_norm']),'Actual detailed trace/audit joins differ')
        same_state(torch,{'raw':packet['raw_GT']},{'raw':torch.from_numpy(original_masks(np,coco,cache[identity['image_id']])).bool()},'Complete original raw GT differs')
        queries=torch.tensor(t['final_query_ids'],dtype=torch.int64); require(torch.equal(packet['query_ids'],queries),'Dynamic raw-GT/query IDs differ')
        generator=torch.Generator(device='cpu').manual_seed(spec['point_seed_base']+g)
        indices=torch.randint(H*W,(len(queries),12544),generator=generator,dtype=torch.int64)
        require(torch.equal(packet['point_indices'],indices) and old.tensor_digest(torch,indices,'int64')==t['point_indices'],'Independent duplicate-preserving point sampling differs')
        for key,shape in (('F256',(1,256,200,200)),('e0',(1,100,256)),('M200',(1,100,200,200))): checked_tensor(torch,packet[key],shape,torch.float32,key)
        projection=compare(torch,packet['M200'],torch.einsum('bqc,bchw->bqhw',packet['e0'],packet['F256']),NUMERIC['native_projection_rtol'],NUMERIC['native_projection_atol'],'Saved native projection')
        require(set(packet['native_live_total_gradient_norms'])=={'F256','e0','M200'} and all(type(v)in(int,float) and math.isfinite(v) and v>=0
            for v in packet['native_live_total_gradient_norms'].values()),'GPU-observed native live total gradient norms differ')
        replay=point_replay(torch,helper,packet['F256'],packet['M200'],identity,packet['readout_before'],queries,packet['raw_GT'],indices)
        require(math.isclose(replay['loss'],packet['auxiliary_loss'],rel_tol=NUMERIC['forward_rtol'],abs_tol=NUMERIC['forward_atol']),'CPU direct auxiliary BCE differs')
        vjp=compare(torch,packet['auxiliary_L_gradient'],replay['local_L_gradient'],NUMERIC['gradient_rtol'],NUMERIC['gradient_atol'],'Local original-L VJP/2')
        require(tuple(packet['readout_before'])==tuple(HEAD_SHAPES) and set(packet['readout_pending_before'])==set(packet['readout_accumulated_before_clip'])==set(HEAD_SHAPES),'Audit C8 populations differ')
        if g==0:
            same_state(torch,packet['readout_before'],initial_head,'Own-stage g0 direct-head initialization')
            require(packet['readout_optimizer_before']['state']=={},'Actual g0 optimizer must have fresh empty moment state')
            zero=point_replay(torch,helper,packet['F256'],packet['M200'],identity,packet['readout_before'],torch.empty((0,),dtype=torch.int64),
                torch.empty((0,H,W),dtype=torch.bool),torch.empty((0,12544),dtype=torch.int64))
            require(zero['loss']==0. and all(torch.count_nonzero(v)==0 for v in zero['head_gradients'].values()),'Independent G0/eight-head zero connection differs'); del zero
        if g%2==0: require(all(v is None for v in packet['readout_pending_before'].values()),'Fresh accumulation pair required')
        else:
            require(previous is not None and previous['global_micro']==g-1,'Adjacent audit pair required')
            same_state(torch,packet['readout_pending_before'],previous['readout_accumulated_before_clip'],'Pending C8 gradient continuity')
            same_state(torch,packet['readout_before'],previous['readout_after'],'C8 weights inside pair')
            require(same_tree(torch,packet['readout_optimizer_before'],previous['readout_optimizer_after']),'C8 optimizer unchanged inside pair')
        gradient_checks={}; clip_checks={}
        for n,value in packet['readout_accumulated_before_clip'].items():
            checked_tensor(torch,value,HEAD_SHAPES[n],torch.float32,n); pending=packet['readout_pending_before'][n]
            contribution=value if pending is None else value-pending
            gradient_checks[n]=compare(torch,contribution,replay['head_gradients'][n],NUMERIC['gradient_rtol'],NUMERIC['gradient_atol'],n+' accumulated-minus-pending')
            require(math.isclose(float(value.norm()),t['readout_gradient_norms'][n],rel_tol=1e-6,abs_tol=1e-8),'Actual C8 accumulated norm differs')
        parameters={n:torch.nn.Parameter(v.clone()) for n,v in packet['readout_before'].items()}
        optimizer=torch.optim.AdamW([dict(params=[p]) for p in parameters.values()],lr=1e-4,weight_decay=.05,betas=(.9,.999),eps=1e-8)
        require(packet['readout_optimizer_before']['param_groups']==optimizer.state_dict()['param_groups'],'Fixed C8 audit recipe differs')
        optimizer.load_state_dict(copy.deepcopy(packet['readout_optimizer_before']))
        if g%2:
            # Conditional on the observed native+head norm; no native-total gradient replay.
            scale=torch.clamp(.01/(torch.tensor(packet['combined_preclip_norm'],dtype=torch.float32)+1e-6),max=1.)
            require(set(packet['readout_clipped_gradients'])==set(HEAD_SHAPES),'Clipped C8 population differs')
            for n,p in parameters.items():
                clip_checks[n]=compare(torch,packet['readout_clipped_gradients'][n],packet['readout_accumulated_before_clip'][n]*scale,1e-6,1e-8,'Combined clip C8 '+n)
                p.grad=packet['readout_clipped_gradients'][n].clone()
            optimizer.step()
        else: require(packet['readout_clipped_gradients'] is None,'No clipping/step inside first micro')
        for n,p in parameters.items(): compare(torch,packet['readout_after'][n],p.detach(),1e-5,1e-7,'Conditional audit AdamW '+n)
        actual,expected=packet['readout_optimizer_after'],optimizer.state_dict()
        require(actual['param_groups']==expected['param_groups'] and set(actual['state'])==set(expected['state']),'Audit actual AdamW membership differs')
        for pid,state in expected['state'].items():
            require(set(actual['state'][pid])==set(state),'Audit AdamW moment keys differ')
            for key,value in state.items(): compare(torch,actual['state'][pid][key],value,1e-5,1e-8,'Audit AdamW '+key)
        if g==stage['microsteps']-1:
            same_state(torch,packet['readout_after'],cp['readout_state_dict'],'Final audit C8/CP exact')
            require(same_tree(torch,packet['readout_optimizer_after'],cp['readout_optimizer_state_dict']),'Final audit C8 optimizer/CP exact')
        previous=packet
        summaries.append(dict(global_micro=g,direct_BCE_CPU=replay['loss'],native_projection=projection,local_L_VJP=vjp,
            C8_micro_gradients=gradient_checks,C8_clip_checks=clip_checks,combined_norm_GPU_observed_not_independently_native_recomputed=True))
        del replay
    return summaries


def atomic_keys():
    return ('loss_cls','loss_mask','loss_dice') + tuple('d%d.%s'%(j,k) for j in range(9) for k in ('loss_cls','loss_mask','loss_dice'))


def qualify(spec_file,acceptance_file,CPU_admission_file):
    path=Path(spec_file); output=Path(acceptance_file)
    require(path.is_absolute() and output.is_absolute() and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'Absolute SPEC/fresh CPU acceptance/CUDA empty required')
    path=path.resolve(strict=True); spec=read(path); require(spec['stage_kind'] in STAGES,'Explicit own-C stage required')
    stage=STAGES[spec['stage_kind']]; schedule=dict(gamma=.1,milestones=stage['milestones'],warmup_start_factor=1,warmup_updates=0)
    expect(spec,dict(schema='ROOT-'+PREFIX+'-spec-v1',kind='mask2former',teacher_epoch=24,arms=['C'],
        **{k:stage[k] for k in ('epochs','microsteps','optimizer_updates','raw_GT_visits_per_arm')},accumulation=2,auxiliary_weight=1.,**SEEDS,
        CPU_affinity=[0,1],allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,CPU_numeric_protocol=NUMERIC,
        backbone_with_cp=True,no_feature_cache=True,no_DEV_selection=True,C_readout_optimizer=HEAD_RECIPE,C_readout_LR_schedule=False,
        schedule=schedule,direct_point_logits=True,native_L_added_to_point_logits=False),'Own-C prelocked protocol differs')
    require(type(spec['deadline_seconds']) is int and 0<spec['deadline_seconds']<=stage['max_deadline'],'Own stage prelocked deadline differs')
    producer,refs,context=source_gate(path,spec)
    CPU_admission_path=Path(CPU_admission_file)
    refs['CPU_admission']=CPU_owner_gate(path,output,CPU_admission_path,spec,refs)
    counters={k:stage[k] for k in ('epochs','microsteps','optimizer_updates','raw_GT_visits_per_arm')}
    report_ref=reference(Path(spec['output_directory'])/'REPORT.json'); report=read(bound(report_ref))
    start_ref=reference(Path(spec['output_directory'])/'START.json'); planned=read(bound(start_ref))
    aggregate_ref=reference(Path(spec['output_directory'])/'QUALIFIED.json'); aggregate=read(bound(aggregate_ref))
    expect(report,dict(schema=PREFIX+'-report-v1',status='POINTREND_STYLE_NATIVE_INTEGRATION_GPU_COMPLETE_PENDING_CPU',
        source=refs['source'],spec=refs['spec'],GPU_admission=refs['GPU_admission'],phase=spec['phase'],stage_kind=spec['stage_kind'],error=None,
        counters_per_arm=counters,CPU_numeric_protocol=NUMERIC,schedule=schedule,C_readout_optimizer=HEAD_RECIPE,C_readout_LR_schedule=False,
        actual_total_backward_calls=stage['microsteps'],extra_backward_or_autograd_grad=False,native_AdamW_groups=345,C_readout_AdamW_groups=8,
        separate_AdamW_optimizers=True,combined_gradient_clip=True,technical_weights_loaded=False,old_optimizer_restored=False,old_RNG_restored=False,
        no_DEV=True,no_TEST=True,no_AP=True,DEV_evaluation_separate=True,formal_resume_eligible=False,formal_initializer_qualified=False,
        native30_CPU_replay_claimed=False,full_optimizer_trajectory_CPU_replay_claimed=False,efficacy_claimed=False,direct_point_logits=True,
        native_L_added_to_point_logits=False,faithful_PointRend_reproduction_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False),'Complete own-C report differs')
    require(report['cleanup']['allocated_after']==0 and set(report['result'])=={'C'} and report['result']['C']['cleanup']['allocated_after']==0,'Own-C cleanup/result population differs')
    expect(aggregate,dict(schema=PREFIX+'-aggregate-v1',status='ROOT_POINTREND_STYLE_NATIVE_INTEGRATION_PENDING_CPU_READER',source=refs['source'],spec=refs['spec'],
        report=report_ref,result=report['result'],phase=spec['phase'],stage_kind=spec['stage_kind'],independently_CPU_qualified=False,AP_computed=False,formal_resume_eligible=False),'Aggregate primary joins differ')
    expect(planned,dict(schema=PREFIX+'-start-v1',source=refs['source'],spec=refs['spec'],GPU_admission=refs['GPU_admission'],phase=spec['phase'],stage_kind=spec['stage_kind'],
        arms=['C'],teacher_checkpoint=spec['checkpointmap']['native0'],teacher_qualification=spec['qualificationmap']['native0'],
        C_tiny_CPU_qualification=spec.get('C_tiny_CPU_qualification'),first24_CPU_qualification=spec.get('first24_CPU_qualification'),technical_weights_loaded=False,
        planned_fresh_native_optimizer_state_entries=0,planned_fresh_readout_optimizer_state_entries=0,old_optimizer_restored=False,old_RNG_restored=False,
        counters_per_arm=counters,schedule=schedule,C_readout_optimizer=HEAD_RECIPE,C_readout_LR_schedule=False,**SEEDS,auxiliary_weight=1.,accumulation=2,
        four_audit_microsteps=stage['details'],direct_point_logits=True,native_L_added_to_point_logits=False,no_DEV=True,no_TEST=True,no_AP=True),'Planned own-C START differs')
    row=report['result']['C']
    CPU_inputs=[refs[k] for k in ('spec','parent_receipt','GPU_admission')]+[report_ref,start_ref,aggregate_ref]+[
        row[k] for k in ('checkpoint','endpoint_proof','trace','order','epochs','start')]+row['audits']
    CPU_owner=read(CPU_admission_path)
    require(all(CPU_owner['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in CPU_inputs),
        'Own CPU owner must bind current raw/admission/report and complete saved input artifacts')
    # Actual CPU affinity is established before any scientific import.
    import psutil
    process=psutil.Process(); process.cpu_affinity([0,1]); require(process.cpu_affinity()==[0,1],'Actual CPU01 affinity required')
    import torch
    require(not torch.cuda.is_initialized() and torch.__version__=='2.0.1+cu118','Bound Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    import numpy as np
    import cv2
    from mmdet.datasets.api_wrappers import COCO
    actual_start=read(bound(row['start'])); loading=actual_start['teacher_loading']
    original_loading=loading['original_native24_loading'] if spec['stage_kind']=='warm48' else loading
    old,common,original_anchor,policy,groups,audit=original_teacher_state(torch,spec,original_loading)
    anchor,initial_head=warm_initial_states(torch,spec,context,old,common,original_anchor,policy)
    require(tuple(initial_head)==tuple(HEAD_SHAPES) and sum(v.numel() for v in initial_head.values())==198402,'Own C8/198402 structure differs')
    coco=COCO(str(bound(spec['FIT']))); cache=GT_population(coco)
    trace=trace_check(torch,np,cv2,old,common,spec,row,cache,policy,anchor,coco)
    cp=checkpoint_check(torch,old,common,spec,refs,row,anchor,original_anchor,policy,groups,audit,initial_head,trace)
    if spec['stage_kind']=='technical':
        require(all(v==2 for v in trace['counts']['readout'].values()) and all(v>0 for v in trace['activity']['readout'].values()),
            'Own C technical head8 actual two steps and finite gradient activity required')
    summaries=audit_check(torch,np,coco,old,spec,row,cache,cp,initial_head,trace)
    summary=dict(checkpoint=row['checkpoint'],endpoint_proof=row['endpoint_proof'],strict_native610_CPU_population=True,
        initial_native610_CPU_exact=True,endpoint_native610_CPU_finite=True,typed_BN106_affine159_buffers_CPU_exact=True,
        fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,native345_moments_match_actual_trace_counts=True,
        readout_moment_entries=8,head_parameter_tensors=8,head_parameter_scalars=198402,phase_microsteps=stage['microsteps'],
        native_updates=stage['optimizer_updates'],readout_updates=stage['optimizer_updates'],phase_raw_GT_visits=trace['GT_visits'],
        odd741_gradient_joins=stage['epochs']//2,audits=summaries,readout_gradient_activity_microcounts=trace['activity']['readout'],
        native_optimizer_nonNone_gradient_step_counts_SOURCE_observed=trace['counts']['native'])
    for ref in spec['sources']: bound(ref)
    for ref in (report_ref,start_ref,aggregate_ref): bound(ref)
    for key in ('checkpoint','endpoint_proof','trace','order','epochs','start'): bound(row[key])
    for ref in row['audits']: bound(ref)
    require(reference(path)==refs['spec'] and reference(__file__)==refs['reader_source'] and reference(report_ref['path'])==report_ref and
        reference(spec['GPU_admission_file'])==refs['GPU_admission'] and reference(CPU_admission_path)==refs['CPU_admission'] and
        not torch.cuda.is_initialized(),'Final input/SOURCE/owner binding and CUDA isolation differ')
    result=dict(schema=QUAL_SCHEMA,status=QUAL_STATUS,**refs,stage_kind=spec['stage_kind'],report=report_ref,start=start_ref,GPU_qualification=aggregate_ref,
        result=report['result'],arms={'C':summary},counters_per_arm=counters,teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'],completed_phase_epochs=stage['epochs'],effective_native_epoch=stage['effective_native_epoch'],
        effective_readout_epoch=stage['effective_readout_epoch'],phase_microsteps=stage['microsteps'],native_updates=stage['optimizer_updates'],
        readout_updates=stage['optimizer_updates'],phase_raw_GT_visits=trace['GT_visits'],C_tiny_CPU_qualification=spec.get('C_tiny_CPU_qualification'),
        first24_CPU_qualification=spec.get('first24_CPU_qualification'),CPU_numeric_protocol=NUMERIC,
        components={k:spec[k] for k in producer.MODULES},head_parameter_tensors=8,head_parameter_scalars=198402,
        CPU_execution=dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),CUDA_initialized=False),
        C8_optimizer_replay_tolerances=dict(clip_rtol=1e-6,clip_atol=1e-8,parameter_rtol=1e-5,parameter_atol=1e-7,moment_rtol=1e-5,moment_atol=1e-8),
        auxiliary_point_BCE_and_local_VJP_CPU_replayed=True,eight_head_micro_gradient_CPU_replayed=True,C8_four_audit_conditional_clip_AdamW_CPU_checked=True,
        direct_point_logits=True,native_L_added_to_point_logits=False,final_e0_direct_head_input=False,
        zero_match_independent_math_connected_exact_zero=True,actual_G0_native_integration_CPU_replayed=False,
        actual_GPU_point_logits_tensor_exported=False,actual_GPU_original_L_tensor_exported=False,local_VJP_uses_CPU_reconstructed_original_L=True,
        FP32_accumulated_gradient_subtraction_bitwise_claimed=False,native_combined_gradient_norm_GPU_observed_not_CPU_reconstructed=True,
        native30_full_CPU_replay=False,native_current_matching_optimality_CPU_replayed=False,native_backbone_CPU_forward_replayed=False,
        whole_CUDA_optimizer_trajectory_CPU_replay=False,certified_GPU_error_bound=False,formal_resume_eligible=False,
        formal_initializer_qualified=spec['stage_kind']!='technical',technical_weights_loaded=False,old_optimizer_restored=False,old_RNG_restored=False,
        no_AP=True,no_DEV=True,no_TEST=True,efficacy_claimed=False,faithful_PointRend_reproduction_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream: json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--acceptance',required=True)
    parser.add_argument('--CPU-admission',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned CPU execution required'); qualify(args.spec,args.acceptance,args.CPU_admission)


if __name__=='__main__':
    main()

