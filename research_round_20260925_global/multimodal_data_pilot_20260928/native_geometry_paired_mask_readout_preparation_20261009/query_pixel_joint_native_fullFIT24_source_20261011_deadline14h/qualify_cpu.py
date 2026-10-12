"""CPU qualification for fresh formal native A/B joint additional24.

Independent CPU checks: E24/BN/native345 recipe, strict endpoint610+B8 state,
raw741/1661 population, all17784 trace/order/epoch/counter and odd741 joins,
four local auxiliary loss/L-VJP/B8-gradient/clip/AdamW audits only.
Native30/current CUDA matching, native total gradients and full native optimizer
trajectory remain pinned SOURCE observations. No native forward, DEV or AP.
Authoring imports only stdlib; future ROOT alone can execute this consumer.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import types

PREFIX='query-pixel-joint-native-fullFIT24'
QUAL_SCHEMA='ROOT-'+PREFIX+'-CPU-qualification-v1'
QUAL_STATUS='ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_CPU_QUALIFIED'
MODES=('native0','global','spatial_halves','spatial_reversed')
DETAILS=(0,1,17782,17783)
NUMERIC=dict(forward_rtol=1e-4,forward_atol=1e-6,gradient_rtol=1e-3,gradient_atol=1e-7,
    native_projection_rtol=1e-4,native_projection_atol=1e-4)


def require(ok,message):
    if not ok: raise AssertionError(message)


def expect(value, expected, message):
    require(all(k in value and value[k]==v and (type(v)is not int or type(value[k])is int) for k,v in expected.items()),message)


def read(path):
    require(Path(path).is_absolute(),'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    require(set(ref)=={'path','bytes','sha256'} and Path(ref['path']).is_absolute() and type(ref['bytes'])is int and
        ref['bytes']>0 and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef') and reference(ref['path'])==ref,'Pinned input differs')
    return Path(ref['path'])


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    require(path.suffix=='.py' and len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def terminal(raw,phase):
    expect(raw,dict(status='ACTUAL_OWNED_TERMINAL',returncode=0,root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0),abort=None,error=None,wait_errors=False,
        GPU_observation_errors=[],whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
        GPU_scope='whole GPU observed, not per-process WDDM allocation'),'STOP before hashes/imports: raw terminal0 required')
    peak=raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('original_identity')is not None and raw.get('numeric_birth_absence',{}).get('original_birth_absent')is True and
        raw.get('phase',phase)==phase and type(peak)in(int,float) and math.isfinite(peak) and 0<=peak<=8192,'Raw identity/peak differs')


def source_gate(path,spec):
    receipt=path.parent/'finite_point_cache_execution_20261005_v2'/'runs'/spec['phase']/'PARENT_RECEIPT.json'
    raw=read(receipt); terminal(raw,spec['phase'])
    tiny=read(spec['joint_preflight_spec']['path'])
    require(set(tiny['training_receiptmap'])==set(tiny['training_phasemap'])==set(MODES),'Original four raw gates required')
    for mode in MODES: terminal(read(tiny['training_receiptmap'][mode]['path']),tiny['training_phasemap'][mode])
    for row in tiny['prior_receipts']:
        require(set(row)=={'receipt','phase'},'Explicit prior raw identity required'); terminal(read(row['receipt']['path']),row['phase'])
    terminal(read(spec['joint_preflight_receipt']['path']),spec['joint_preflight_phase'])
    # Current and all technical/original raws precede the first SHA or SOURCE execution.
    reader,source,spec_ref=reference(__file__),reference(Path(__file__).with_name('train.py')),reference(path)
    owner=read(spec['GPU_admission_file']); hashes=owner['bound_source_sha256']
    require(spec['CPU_reader']==reader and reader in spec['sources'] and spec['entry_source']==source and source in spec['sources'] and
        all(hashes.get(r['path'])==r['sha256'] for r in (reader,source,spec_ref)) and
        raw['admission_sha256']==reference(spec['GPU_admission_file'])['sha256'],'Actual original owner reader/entry/SPEC joins differ')
    producer=load(source,'_formal_joint_owned_source')
    actual_source,actual_spec,admission,tiny_entry=producer.source_gate(path,spec)
    require(actual_source==source and actual_spec==spec_ref and raw['admission_sha256']==admission['sha256'],'Current raw admission differs')
    return producer,dict(source=source,reader_source=reader,spec=spec_ref,parent_receipt=reference(receipt),GPU_admission=admission,phase=spec['phase'])


def GT_population(torch,np,cv2,coco,old):
    require(len(coco.imgs)==741 and len(coco.anns)==1661,'Complete original FIT741/1661 required')
    cache={}; total=0
    for index,image_id in enumerate(coco.imgs):
        image=coco.load_imgs([image_id])[0]; H,W=image['height'],image['width']
        ids=coco.get_ann_ids(img_ids=[image_id]); require(1<=len(ids)<=100 and len(ids)==len(set(ids)),'All raw GT identities required')
        # Original loader uses nearest resize with its separately observed h,w.
        cache[image_id]=dict(dataset_index=index,ids=ids,original_shape=[H,W],annotations=coco.load_anns(ids))
        total+=len(ids)
    require(total==1661,'Raw GT population differs'); return cache


def digest_record(record,shape,dtype):
    require(set(record)=={'shape','dtype','sha256'} and record['shape']==list(shape) and record['dtype']==dtype and
        len(record['sha256'])==64 and set(record['sha256'])<=set('0123456789abcdef'),'Gradient digest shape/dtype/SHA differs')


def trace_check(torch,np,cv2,old,common,helper,producer,spec,row,arm,cache,policy,anchor,coco):
    orders=read(bound(row['order'])); epochs=read(bound(row['epochs'])); require(len(orders)==len(epochs)==24,'Full24 order/epoch records required')
    generator=torch.Generator(device='cpu').manual_seed(spec['order_seed']); states=[]; detail_rows={}
    counts={kind:{n:0 for n in ns} for kind,ns in (('native',policy['active_names']),('readout',helper.READOUT_SHAPES if arm=='B' else {}))}
    activity=copy.deepcopy(counts); boundary=None; joins=0; GT_visits=0; last_external=None
    with bound(row['trace']).open(encoding='utf-8') as stream:
        for epoch in range(24):
            before=generator.get_state(); order=torch.randperm(741,generator=generator).tolist(); states.append(generator.get_state().clone())
            expect(orders[epoch],dict(phase_epoch=epoch,order_seed=spec['order_seed'],dataset_indices=order,
                image_ids=[list(coco_ids(cache))[i] for i in order],private_CPU_RNG_before=old.tensor_digest(torch,before,'uint8'),
                private_CPU_RNG_after=old.tensor_digest(torch,states[-1],'uint8')),'Independent private order differs')
            losses={k:[] for k in ('native','auxiliary','total')}; epoch_GT=0
            for within,index in enumerate(order):
                g=epoch*741+within; text=stream.readline(); require(bool(text),'Trace missing micro'); t=json.loads(text)
                stepped=bool((g+1)%2==0); updates=(g+1)//2; image_id=list(coco_ids(cache))[index]; info=cache[image_id]; identity=t['identity']; ids=info['ids']
                expect(t,dict(arm=arm,global_micro=g,phase_epoch=epoch,effective_native_epoch_index=24+epoch,within_epoch=within,
                    auxiliary_weight=1.,accumulation_divisor=2,optimizer_update=stepped,phase_updates=updates,
                    readout_updates=updates if arm=='B' else 0,scheduler_u_before_step=updates-int(stepped),
                    native_lr_factor=common.schedule_factor(spec['schedule'],updates-int(stepped)),B_readout_lr=1e-4 if arm=='B' else None,
                    actual_backward_calls=1,frozen_BN_parameter_grads_None=True,detailed_observation=g in DETAILS,
                    private_seed=spec['loss_seed_base']+g,point_seed=spec['point_seed_base']+g,point_count_per_GT=12544,
                    sampling_with_replacement=True,external_RNG_restored=True,dynamic_current_matching=True),'Trace recipe/counters differ')
                require(identity['image_id']==image_id and identity['dataset_index']==index and identity['ann_ids']==ids and
                    identity['original_shape']==info['original_shape'] and [v['annotation_id'] for v in identity['parser_rows'] if v['native_parser_retained']]==ids,'Complete image/raw parser identities differ')
                H,W=identity['original_shape']; h,w=identity['actual_img_shape']; require(0<h<=800 and 0<w<=800,'Native crop dimensions differ')
                if 'padded_digest' not in info:
                    padded=np.zeros((len(ids),800,800),np.uint8)
                    for j,a in enumerate(info['annotations']):
                        mask=np.zeros((H,W),np.uint8) if a.get('segmentation')==[] else coco.annToMask(a)
                        padded[j,:h,:w]=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
                    info.update(padded_digest=old.tensor_digest(torch,torch.from_numpy(padded),'uint8'),actual_shape=[h,w])
                require(info['actual_shape']==[h,w] and t['GT']['ann_ids']==ids and t['GT']['complete_all_raw_annotations']is True and
                    t['GT']['masks']==info['padded_digest'],'Independent complete resized raw GT differs')
                md=t['metadata']; expect(md,dict(arm=arm,auxiliary_weight=1.,native_atoms=30,raw_annotation_ids=ids,processed_GT_to_raw=list(range(len(ids))),
                    processed_GT_observation=dict(calls=1,correspondence_validator_passed=True),native_forward_counts=dict(head_forward_started=1,head_forward_finished=1,pixel_decoder=1,mask_embed=10),
                    fixed_teacher_matching_used=False,point_indices_sampled_here=False,backward_or_optimizer_step_performed=False,auxiliary_accumulation_scaled=False),'Observed native30/current GT interface differs')
                require(set(md['native_atom_values'])==set(helper.atomic_keys()) and len(md['current_stage_matches'])==10,'Original30/ten assignments required')
                for match in md['current_stage_matches']:
                    qs,gs=match['positive_query_indices'],match['assigned_GT_indices']
                    require(match['positives']==len(qs)==len(gs)==len(set(qs))==len(ids) and sorted(gs)==list(range(len(ids))) and all(type(q)is int and 0<=q<100 for q in qs),'Actual all-GT/current query assignment differs')
                final=md['current_stage_matches'][-1]; mapping=dict(zip(final['assigned_GT_indices'],final['positive_query_indices']))
                require(t['final_query_ids']==md['final_query_by_raw_GT']==[mapping[j] for j in range(len(ids))],'Final actual GT-query mapping differs')
                atoms=md['native_atom_values']; require(all(type(v)in(int,float) and math.isfinite(v) for v in atoms.values()),'Nonfinite native atoms')
                require(math.isclose(t['losses']['native'],math.fsum(atoms.values()),rel_tol=1e-4,abs_tol=1e-4) and
                    math.isclose(t['losses']['total'],t['losses']['native']+t['losses']['auxiliary'],rel_tol=1e-5,abs_tol=1e-5),'Observed lambda1 loss arithmetic differs')
                for key,value in t['losses'].items():
                    require(key in losses and type(value)in(int,float) and math.isfinite(value),'Finite three losses required'); losses[key].append(value)
                for kind,key in (('native','native_gradient_norms'),('readout','readout_gradient_norms')):
                    norms=t[key]; require(set(norms)==set(counts[kind]) and all(v is None or type(v)in(int,float) and math.isfinite(v) and v>=0 for v in norms.values()),'Gradient membership/finite norms differ')
                    require(kind!='readout' or all(v is not None for v in norms.values()),'All B8 gradients must be connected; zero permitted')
                    for n,v in norms.items(): activity[kind][n]+=int(v is not None and v>0); counts[kind][n]+=int(stepped and v is not None)
                require(set(t['BN_module_training'])==set(policy['BN_modules']) and not any(t['BN_module_training'].values()),'Typed BN53 modules must stay eval')
                norm=t['combined_preclip_norm']; require(norm is None if not stepped else type(norm)in(int,float) and math.isfinite(norm) and norm>=0,'Combined clip location/finite norm differs')
                end=t['odd_epoch_end_pending_gradients']; join=t['pending_boundary_gradient_join']
                require((end is not None)==(within==740 and (g+1)%2==1) and (join is not None)==(within==0 and g%2==1),'Odd741 proof location differs')
                if end is not None:
                    require(set(end)=={'native','readout'},'Boundary gradient domains differ')
                    for kind,ns in (('native',policy['active_names']),('readout',helper.READOUT_SHAPES if arm=='B' else {})):
                        require(set(end[kind])==set(ns),'Boundary gradient membership differs')
                        for n,d in end[kind].items():
                            if d is not None: digest_record(d,anchor[n].shape if kind=='native' else helper.READOUT_SHAPES[n],'float32')
                    boundary=(g,end)
                if join is not None:
                    expect(join,dict(previous_global_micro=g-1,after_previous_odd_epoch=boundary[1],before_next_forward=boundary[1],exact=True,SOURCE_observed_not_CPU_gradient_replay=True),'Pending gradient continuity differs')
                    require(boundary[0]==g-1,'Prior odd epoch identity differs'); boundary=None; joins+=1
                require(t['external_RNG_before']==t['external_RNG_after'] and (last_external is None or last_external==t['external_RNG_before']),'External RNG continuity differs')
                for key in ('private_RNG_before','private_RNG_after','external_RNG_before','external_RNG_after'):
                    require(len(t[key])==2,'CPU/CUDA RNG packets required')
                    for d in t[key]: require(d['dtype']=='uint8' and len(d['shape'])==1 and len(d['sha256'])==64,'RNG digest packet differs')
                last_external=t['external_RNG_after']; epoch_GT+=len(ids); GT_visits+=len(ids)
                if g in DETAILS: detail_rows[g]=t
            expected_updates=((epoch+1)*741)//2
            expect(epochs[epoch],dict(phase_epoch=epoch,images=741,raw_GT=1661,phase_microsteps=(epoch+1)*741,
                epoch_updates=expected_updates-(epoch*741)//2,phase_updates=expected_updates,pending_accumulation_microsteps=((epoch+1)*741)%2),'Odd741 epoch counters differ')
            require(epoch_GT==1661 and set(epochs[epoch]['training_loss_summary'])==set(losses),'Full epoch/three loss summaries required')
            for key,values in losses.items():
                expected=math.fsum(values); summary=epochs[epoch]['training_loss_summary'][key]
                expect(summary,dict(mean_denominator='741 image-averaged losses'),'Loss denominator differs')
                require(math.isclose(summary['image_loss_sum'],expected,rel_tol=1e-12,abs_tol=1e-9) and
                    math.isclose(summary['image_loss_mean'],expected/741,rel_tol=1e-12,abs_tol=1e-12),'Trace-derived epoch loss summary differs')
        require(not stream.readline() and GT_visits==39864 and joins==12 and boundary is None,'Full17784/39864/12 joins and no dangling accumulation required')
    return dict(counts=counts,activity=activity,last_external=last_external,private_order_state=states[-1],detail_rows=detail_rows)


def coco_ids(cache):
    return cache.keys()


def same_tree(torch,first,second):
    if isinstance(first,torch.Tensor):
        return isinstance(second,torch.Tensor) and first.shape==second.shape and first.dtype==second.dtype and torch.equal(first,second)
    if type(first)is not type(second): return False
    if isinstance(first,dict): return set(first)==set(second) and all(same_tree(torch,first[k],second[k]) for k in first)
    if isinstance(first,(list,tuple)): return len(first)==len(second) and all(same_tree(torch,a,b) for a,b in zip(first,second))
    return first==second


def checkpoint_check(torch,helper,old,common,producer,spec,refs,row,arm,anchor,policy,groups,audit,cold,trace):
    cp=torch.load(str(bound(row['checkpoint'])),map_location='cpu'); proof=read(bound(row['endpoint_proof'])); start=read(bound(row['start']))
    require(set(cp)=={'schema','meta','native_state_dict','readout_state_dict','native_optimizer_state_dict','readout_optimizer_state_dict','private_order_state','external_rng_state'} and cp['schema']==PREFIX+'-checkpoint-v1','Formal complete CPU checkpoint container differs')
    m=cp['meta']; expected=dict(source=refs['source'],spec=refs['spec'],phase=spec['phase'],arm=arm,teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'],initial_native_epoch=24,completed_phase_epochs=24,effective_native_epoch=48,
        phase_microsteps=17784,phase_updates=8892,phase_raw_GT_visits=39864,policy=policy,native_optimizer=audit,native_optimizer_param_names=[[n] for n in policy['active_names']],
        readout_optimizer_param_names=[[n] for n in helper.READOUT_SHAPES] if arm=='B' else [],schedule=spec['schedule'],stored_native_lr_factor=common.schedule_factor(spec['schedule'],8891),
        readout_optimizer=producer.HEAD_RECIPE if arm=='B' else None,order_seed=spec['order_seed'],loss_seed_base=spec['loss_seed_base'],point_seed_base=spec['point_seed_base'],
        readout_seed=spec['readout_seed'],auxiliary_weight=1.,kappa=1.,eps=1e-6,accumulation=2,combined_clip_max_norm=.01,fresh_native_optimizer_state_entries=0,
        fresh_readout_optimizer_state_entries=0,old_optimizer_restored=False,old_RNG_restored=False,technical_weights_loaded=False,no_pending_accumulation_gradients=True,formal_resume_eligible=False,AP_computed=False)
    # CPU expected audit contains tuples while JSON proof uses lists; compare canonical forms.
    require(json.dumps(m['native_optimizer'],sort_keys=True)==json.dumps(audit,sort_keys=True),'Original native AdamW audit differs')
    expected.pop('native_optimizer'); expect(m,expected,'Formal checkpoint recipe/identity differs')
    require(m['artifacts']=={k:row[k] for k in ('trace','order','epochs','start','audits')},'Checkpoint artifact references differ')
    state=cp['native_state_dict']; require(set(state)==set(anchor) and len(state)==610,'Strict native610 endpoint names differ')
    for n,v in state.items(): helper.checked_tensor(torch,v,anchor[n].shape,anchor[n].dtype,n)
    require(all(torch.equal(state[n],anchor[n]) for n in policy['frozen_names']+policy['BN_buffer_names']),'BN106/159 endpoint differs')
    expect(proof,dict(schema=PREFIX+'-endpoint-proof-v1',source=refs['source'],spec=refs['spec'],arm=arm,checkpoint=row['checkpoint'],policy=policy,
        positive_accumulated_gradient_microcounts=trace['activity'],optimizer_step_nonNone_gradient_counts=trace['counts'],completed_phase_epochs=24,effective_native_epoch=48,
        phase_microsteps=17784,phase_updates=8892,phase_raw_GT_visits=39864,frozen_BN106_affine159_buffers_exact=True,no_pending_accumulation_gradients=True,
        odd741_gradient_joins=12,fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,one_total_backward_per_micro=True,
        native_loss_atoms=30,ten_current_assignments_per_micro=True,four_audit_microsteps=list(DETAILS),CPU_checkpoint_readback_exact=True,
        technical_weights_loaded=False,old_optimizer_restored=False,old_RNG_restored=False,formal_resume_eligible=False,native30_CPU_replay_claimed=False,
        full_optimizer_trajectory_CPU_replay_claimed=False,conditioning_isolated_claimed=False,compute_matched_claimed=False),'Formal endpoint proof differs')
    common.cpu_state(torch,old,anchor,proof['before_native610'],'Independent original E24'); common.cpu_state(torch,old,state,proof['after_native610'],'Endpoint full610')
    require(proof['changed_native']=={n:not torch.equal(v,anchor[n]) for n,v in state.items()},'Endpoint changed mask differs')
    require(proof['artifacts']==m['artifacts'] and proof['teacher_loading']==start['teacher_loading'],'Start/endpoint lineage differs')
    expect(start,dict(schema=PREFIX+'-start-state-v1',source=refs['source'],spec=refs['spec'],arm=arm,policy=policy,
        fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,old_optimizer_restored=False,old_RNG_restored=False,
        technical_weights_loaded=False,phase_microsteps=0,phase_updates=0,initial_native_epoch=24,phase_initial_epoch=0,auxiliary_weight=1.),'Actual fresh start differs')
    common.cpu_state(torch,old,anchor,start['actual_native610'],'Actual initial native610')
    require(common.optimizer_snapshot(torch,old,dict(state={},param_groups=groups),[[n] for n in policy['active_names']])==start['actual_native_optimizer'] and
        json.dumps(start['native_optimizer'],sort_keys=True)==json.dumps(audit,sort_keys=True),'Actual fresh empty native optimizer snapshot differs')
    require(m['base_lrs']==[g['lr'] for g in groups],'Actual native base LR groups differ')
    common.validate_moments(torch,cp['native_optimizer_state_dict'],common.groups_at(groups,spec['schedule'],8891),policy['active_names'],state,'AdamW',audit['recipe'],trace['counts']['native'])
    require(common.optimizer_snapshot(torch,old,cp['native_optimizer_state_dict'],m['native_optimizer_param_names'])==proof['actual_native_optimizer'],'Independent native moment snapshot differs')
    require(common.check_rng_tensors(torch,old,cp['external_rng_state'])==trace['last_external'],'CP external RNG differs')
    private=cp['private_order_state']; expect(private,dict(seed=spec['order_seed'],generated_phase_epochs=24),'Fresh order counters differ')
    require(set(private)=={'seed','generated_phase_epochs','torch_CPU_generator_state'} and torch.equal(private['torch_CPU_generator_state'],trace['private_order_state']),'CP private741 order state differs')
    head=cp['readout_state_dict']
    if arm=='A':
        require(head=={} and cp['readout_optimizer_state_dict']is None and proof['actual_readout_optimizer']is None,'A must contain no readout')
    else:
        require(set(head)==set(helper.READOUT_SHAPES),'Strict B8 population differs')
        for n,v in head.items(): helper.checked_tensor(torch,v,helper.READOUT_SHAPES[n],torch.float32,n)
        parameters=[torch.nn.Parameter(cold[n].clone()) for n in cold]
        optimizer=torch.optim.AdamW([dict(params=[p]) for p in parameters],lr=1e-4,weight_decay=.05,betas=(.9,.999),eps=1e-8)
        expected_groups=optimizer.state_dict()['param_groups']; actual=cp['readout_optimizer_state_dict']
        require(actual['param_groups']==expected_groups and len(actual['state'])==8,'Actual B8 fixed-LR AdamW groups differ')
        for n,group in zip(cold,expected_groups):
            moments=actual['state'][group['params'][0]]; require(set(moments)=={'step','exp_avg','exp_avg_sq'} and float(moments['step'])==trace['counts']['readout'][n]==8892,'B8 moment step8892 differs')
            helper.checked_tensor(torch,moments['step'],(),torch.float32,'B step')
            for key in ('exp_avg','exp_avg_sq'): helper.checked_tensor(torch,moments[key],helper.READOUT_SHAPES[n],torch.float32,key)
            require(bool((moments['exp_avg_sq']>=0).all()),'B negative squared moment')
        require(common.optimizer_snapshot(torch,old,actual,m['readout_optimizer_param_names'])==proof['actual_readout_optimizer'],'Independent B8 moment snapshot differs')
    common.cpu_state(torch,old,cold if arm=='B' else {},proof['before_readout'],'Cold private-seed B8')
    common.cpu_state(torch,old,head,proof['after_readout'],'Final B8')
    common.cpu_state(torch,old,cold if arm=='B' else {},start['initial_readout'],'Actual cold readout')
    expect(row,dict(arm=arm,completed_phase_epochs=24,effective_native_epoch=48,phase_microsteps=17784,phase_updates=8892,phase_raw_GT_visits=39864,
        native_parameter_tensors=451,native_state_tensors=610,native_trainable_tensors=345,readout_parameter_tensors=8 if arm=='B' else 0,
        readout_parameters=9441 if arm=='B' else 0,fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,odd741_gradient_joins=12,no_pending_accumulation_gradients=True),'Formal result counters differ')
    return cp


def audit_check(torch,np,cv2,helper,old,spec,row,arm,cache,cp,cold,trace,coco):
    require(len(row['audits'])==4,'Exactly four actual audit packets required')
    summaries=[]; previous=None
    for g,ref in zip(DETAILS,row['audits']):
        packet=torch.load(str(bound(ref)),map_location='cpu'); t=trace['detail_rows'][g]; identity=packet['identity']; H,W=identity['original_shape']
        expect(packet,dict(schema=PREFIX+'-audit-v1',arm=arm,global_micro=g,identity=t['identity'],metadata=t['metadata'],auxiliary_gradient_scale=.5,
            native_gradient_replay_claimed=False,auxiliary_loss=t['losses']['auxiliary'],native_total=t['losses']['native'],optimizer_update=t['optimizer_update'],
            native_updates=t['phase_updates'],readout_updates=t['readout_updates'],combined_preclip_norm=t['combined_preclip_norm']),'Detailed trace/audit join differs')
        image=cache[identity['image_id']]; masks=np.stack([np.zeros((H,W),np.uint8) if a.get('segmentation')==[] else coco.annToMask(a) for a in image['annotations']])
        helper.same_state(torch,{'raw':packet['raw_GT']},{'raw':torch.from_numpy(masks).bool()},'Independent complete original GT differs')
        queries=torch.tensor(t['final_query_ids'],dtype=torch.int64); require(torch.equal(packet['query_ids'],queries),'Actual dynamic GT-query IDs differ')
        generator=torch.Generator(device='cpu').manual_seed(spec['point_seed_base']+g)
        indices=torch.randint(H*W,(len(queries),12544),generator=generator,dtype=torch.int64)
        require(torch.equal(packet['point_indices'],indices) and old.tensor_digest(torch,indices,'int64')==t['point_indices'],'Independent duplicate-preserving integer sampling differs')
        for key,shape in (('F256',(1,256,200,200)),('e0',(1,100,256)),('M200',(1,100,200,200))): helper.checked_tensor(torch,packet[key],shape,torch.float32,key)
        projection=helper.compare(torch,packet['M200'],torch.einsum('bqc,bchw->bqhw',packet['e0'],packet['F256']),NUMERIC['native_projection_rtol'],NUMERIC['native_projection_atol'],'Native linear projection')
        L=helper.render(torch,packet['M200'],identity); replay=helper.replay_auxiliary(torch,arm,packet['F256'],packet['e0'],L,identity,packet['readout_before'],queries,packet['raw_GT'],indices)
        require(math.isclose(replay['loss'],packet['auxiliary_loss'],rel_tol=NUMERIC['forward_rtol'],abs_tol=NUMERIC['forward_atol']),'CPU auxiliary BCE differs')
        vjp=helper.compare(torch,packet['auxiliary_L_gradient'],replay['native_logits_gradient'][0,queries],NUMERIC['gradient_rtol'],NUMERIC['gradient_atol'],'Local original-L VJP/2')
        gradient_checks={}; update_checks={}
        if arm=='A':
            require(packet['readout_before']==packet['readout_pending_before']==packet['readout_accumulated_before_clip']==packet['readout_after']=={} and
                packet['readout_optimizer_before']is packet['readout_optimizer_after']is None,'A audit must contain no head optimizer')
        else:
            require(set(packet['readout_before'])==set(packet['readout_pending_before'])==set(packet['readout_accumulated_before_clip'])==set(cold),'Audit B8 population differs')
            if g==0:
                helper.same_state(torch,packet['readout_before'],cold,'Actual g0 cold initialization')
                require(packet['readout_optimizer_before']['state']=={},'Actual g0 optimizer moment state must be fresh empty')
            if g%2==0: require(all(v is None for v in packet['readout_pending_before'].values()),'Fresh accumulation pair required')
            else:
                require(previous is not None and previous['global_micro']==g-1,'Adjacent audit pair required')
                helper.same_state(torch,packet['readout_pending_before'],previous['readout_accumulated_before_clip'],'Pending B8 gradient continuity')
                helper.same_state(torch,packet['readout_before'],previous['readout_after'],'B8 weights inside pair')
                require(same_tree(torch,packet['readout_optimizer_before'],previous['readout_optimizer_after']),'B8 optimizer moments/groups inside pair')
            for n,value in packet['readout_accumulated_before_clip'].items():
                helper.checked_tensor(torch,value,cold[n].shape,torch.float32,n)
                pending=packet['readout_pending_before'][n]; contribution=value if pending is None else value-pending
                gradient_checks[n]=helper.compare(torch,contribution,replay['readout_gradients'][n],NUMERIC['gradient_rtol'],NUMERIC['gradient_atol'],n+' auxiliary accumulated-minus-pending')
                require(math.isclose(float(value.norm()),t['readout_gradient_norms'][n],rel_tol=1e-6,abs_tol=1e-8),'Actual B8 gradient norm differs')
            parameters={n:torch.nn.Parameter(v.clone()) for n,v in packet['readout_before'].items()}
            optimizer=torch.optim.AdamW([dict(params=[p]) for p in parameters.values()],lr=1e-4,weight_decay=.05,betas=(.9,.999),eps=1e-8)
            require(packet['readout_optimizer_before']['param_groups']==optimizer.state_dict()['param_groups'],'Audit B8 exact fixed recipe differs')
            optimizer.load_state_dict(copy.deepcopy(packet['readout_optimizer_before']))
            if g%2:
                scale=torch.clamp(.01/(torch.tensor(packet['combined_preclip_norm'],dtype=torch.float32)+1e-6),max=1.)
                require(set(packet['readout_clipped_gradients'])==set(cold),'Clipped B8 membership differs')
                for n,p in parameters.items():
                    expected=packet['readout_accumulated_before_clip'][n]*scale
                    update_checks[n]=helper.compare(torch,packet['readout_clipped_gradients'][n],expected,1e-6,1e-8,'Combined clip B8 '+n)
                    p.grad=packet['readout_clipped_gradients'][n].clone()
                optimizer.step()
            else: require(packet['readout_clipped_gradients']is None,'No clip/step inside first micro')
            for n,p in parameters.items(): helper.compare(torch,packet['readout_after'][n],p.detach(),1e-5,1e-7,'Bounded audit AdamW '+n)
            actual=packet['readout_optimizer_after']; expected=optimizer.state_dict()
            require(actual['param_groups']==expected['param_groups'] and set(actual['state'])==set(expected['state']),'Audit actual AdamW membership differs')
            for pid,state in expected['state'].items():
                require(set(actual['state'][pid])==set(state),'Audit AdamW moment keys differ')
                for key,value in state.items(): helper.compare(torch,actual['state'][pid][key],value,1e-5,1e-8,'Audit AdamW '+key)
            if g==17783:
                helper.same_state(torch,packet['readout_after'],cp['readout_state_dict'],'Final audit B8/full CP exact')
                require(same_tree(torch,packet['readout_optimizer_after'],cp['readout_optimizer_state_dict']),'Final audit B8 optimizer/full CP exact')
            previous=packet
        summaries.append(dict(global_micro=g,auxiliary_BCE_CPU=replay['loss'],native_projection=projection,local_L_VJP=vjp,B8_micro_gradients=gradient_checks,
            B8_clip_checks=update_checks,combined_norm_GPU_observed_not_independently_native_recomputed=True))
        del L,replay,packet
    return summaries


def qualify(spec_file,acceptance_file):
    path=Path(spec_file); output=Path(acceptance_file)
    require(path.is_absolute() and output.is_absolute() and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Absolute SPEC/fresh CPU acceptance/CUDA empty required')
    path=path.resolve(strict=True); spec=read(path); producer,refs=source_gate(path,spec)
    expect(spec,dict(schema='ROOT-'+PREFIX+'-spec-v1',kind='mask2former',teacher_epoch=24,arms=['A','B'],epochs=24,microsteps=17784,optimizer_updates=8892,
        raw_GT_visits_per_arm=39864,accumulation=2,auxiliary_weight=1.,order_seed=2026101021,loss_seed_base=1000000,point_seed_base=3800000,readout_seed=2026100911,
        CPU_affinity=[0,1],allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,CPU_numeric_protocol=NUMERIC,kappa=1.,eps=1e-6,
        backbone_with_cp=True,no_feature_cache=True,no_DEV_selection=True,B_readout_optimizer=producer.HEAD_RECIPE,B_readout_LR_schedule=False,schedule=producer.SCHEDULE),'Formal numeric protocol differs')
    require(type(spec['deadline_seconds'])is int and 0<spec['deadline_seconds']<=50400,'Prelocked formal deadline differs')
    report_ref=reference(Path(spec['output_directory'])/'REPORT.json'); report=read(bound(report_ref)); start_ref=reference(Path(spec['output_directory'])/'START.json')
    aggregate_ref=reference(Path(spec['output_directory'])/'QUALIFIED.json'); aggregate=read(bound(aggregate_ref))
    expect(report,dict(schema=PREFIX+'-report-v1',status='QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_GPU_COMPLETE_PENDING_CPU',source=refs['source'],spec=refs['spec'],
        GPU_admission=refs['GPU_admission'],phase=spec['phase'],error=None,counters_per_arm=producer.COUNTERS,CPU_numeric_protocol=NUMERIC,
        schedule=producer.SCHEDULE,B_readout_optimizer=producer.HEAD_RECIPE,B_readout_LR_schedule=False,auxiliary_weight=1.,actual_total_backward_calls=35568,
        extra_backward_or_autograd_grad=False,serial_arms=True,native_AdamW_groups=345,B_readout_AdamW_groups=8,separate_AdamW_optimizers=True,combined_gradient_clip=True,
        technical_weights_loaded=False,old_optimizer_restored=False,old_RNG_restored=False,no_DEV=True,no_TEST=True,no_AP=True,DEV_evaluation_separate=True,
        formal_resume_eligible=False,native30_CPU_replay_claimed=False,full_optimizer_trajectory_CPU_replay_claimed=False,efficacy_claimed=False,
        compute_matched_claimed=False,conditioning_isolated_claimed=False),'Complete formal report differs')
    require(report['cleanup']['allocated_after']==0 and set(report['result'])=={'A','B'} and all(r['cleanup']['allocated_after']==0 for r in report['result'].values()),'Serial cleanup/result population differs')
    expect(aggregate,dict(schema=PREFIX+'-aggregate-v1',status='ROOT_QUERY_PIXEL_JOINT_NATIVE_FULLFIT24_PENDING_CPU_READER',source=refs['source'],spec=refs['spec'],
        report=report_ref,result=report['result'],phase=spec['phase'],independently_CPU_qualified=False,AP_computed=False,formal_resume_eligible=False),'Aggregate joins differ')
    expect(read(bound(start_ref)),dict(schema=PREFIX+'-start-v1',source=refs['source'],spec=refs['spec'],GPU_admission=refs['GPU_admission'],phase=spec['phase'],arms=['A','B'],
        joint_preflight_CPU_qualification=spec['joint_preflight_CPU_qualification'],planned_fresh_native_optimizer_state_entries=0,planned_fresh_readout_optimizer_state_entries=0,
        technical_weights_loaded=False,counters_per_arm=producer.COUNTERS,no_DEV=True,no_TEST=True,no_AP=True),'Planned START joins differ')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required')
    import torch
    import numpy as np
    import cv2
    require(not torch.cuda.is_initialized() and torch.__version__=='2.0.1+cu118','Bound Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    from mmdet.datasets.api_wrappers import COCO
    helper=load(spec['joint_preflight_CPU_reader'],'_independent_qualified_tiny_CPU_math')
    tiny_report=read(bound(spec['joint_preflight_report']))
    old,common,anchor,policy,groups,audit=helper.teacher_state(torch,spec,tiny_report['result'])
    coco=COCO(str(bound(spec['FIT']))); cache=GT_population(torch,np,cv2,coco,old)
    cold=helper.cold_readout(torch,spec['readout_seed']); summaries={}
    for arm in ('A','B'):
        row=report['result'][arm]; trace=trace_check(torch,np,cv2,old,common,helper,producer,spec,row,arm,cache,policy,anchor,coco)
        cp=checkpoint_check(torch,helper,old,common,producer,spec,refs,row,arm,anchor,policy,groups,audit,cold,trace)
        summaries[arm]=dict(checkpoint=row['checkpoint'],endpoint_proof=row['endpoint_proof'],strict_native610_CPU_population=True,initial_native610_CPU_exact=True,typed_BN106_affine159_buffers_CPU_exact=True,
            fresh_native_optimizer_state_entries=cp['meta']['fresh_native_optimizer_state_entries'],fresh_readout_optimizer_state_entries=cp['meta']['fresh_readout_optimizer_state_entries'],
            native345_moments_match_actual_trace_counts=True,readout_moment_entries=8 if arm=='B' else 0,phase_microsteps=17784,phase_updates=8892,phase_raw_GT_visits=39864,
            odd741_gradient_joins=12,audits=audit_check(torch,np,cv2,helper,old,spec,row,arm,cache,cp,cold,trace,coco)); del cp,trace
    require(report['result']['A']['order']['sha256']==report['result']['B']['order']['sha256'],'Paired independent private order stream differs')
    for ref in spec['sources']: bound(ref)
    for ref in (report_ref,start_ref,aggregate_ref): bound(ref)
    for row in report['result'].values():
        for key in ('checkpoint','endpoint_proof','trace','order','epochs','start'): bound(row[key])
        for ref in row['audits']: bound(ref)
    require(reference(path)==refs['spec'] and reference(report_ref['path'])==report_ref and not torch.cuda.is_initialized(),'Final input bindings/CUDA isolation differ')
    result=dict(schema=QUAL_SCHEMA,status=QUAL_STATUS,**refs,report=report_ref,start=start_ref,GPU_qualification=aggregate_ref,arms=summaries,
        result=report['result'],counters_per_arm=producer.COUNTERS,
        teacher_checkpoint=spec['checkpointmap']['native0'],teacher_qualification=spec['qualificationmap']['native0'],
        joint_preflight_CPU_qualification=spec['joint_preflight_CPU_qualification'],completed_phase_epochs=24,effective_native_epoch=48,
        phase_microsteps_per_arm=17784,native_updates_per_arm=8892,B_readout_updates=8892,phase_raw_GT_visits_per_arm=39864,
        CPU_numeric_protocol=NUMERIC,CPU_execution=dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),CUDA_initialized=False),
        B8_optimizer_replay_tolerances=dict(clip_rtol=1e-6,clip_atol=1e-8,parameter_rtol=1e-5,parameter_atol=1e-7,moment_rtol=1e-5,moment_atol=1e-8),
        auxiliary_four_audits_CPU_replayed=True,B8_four_audit_clip_AdamW_CPU_checked=True,actual_GPU_original_L_tensor_exported=False,
        local_VJP_uses_CPU_reconstructed_original_L=True,FP32_accumulated_gradient_subtraction_bitwise_claimed=False,
        native30_full_CPU_replay=False,native_current_matching_optimality_CPU_replayed=False,native_backbone_CPU_forward_replayed=False,
        whole_CUDA_optimizer_trajectory_CPU_replay=False,certified_GPU_error_bound=False,formal_resume_eligible=False,
        technical_weights_loaded=False,no_AP=True,no_DEV=True,no_TEST=True,efficacy_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream: json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--acceptance',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned CPU execution required'); qualify(args.spec,args.acceptance)


if __name__=='__main__':
    main()
