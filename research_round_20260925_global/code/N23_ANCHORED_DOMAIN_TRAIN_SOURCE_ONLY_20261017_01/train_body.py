"""N23 anchored finite-pool training; fixed stages, no teacher forward or DEV/AP."""
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ARMS=('A_COMMON','A_ACTIVE','B_C')


def need(ok,message):
    if not ok:raise AssertionError(message)


def validate_recipe(pre,spec):
    expected=dict(arms=list(ARMS),epochs=24,stage_windows=[[1,8],[9,16],[17,24]],
        anchor_points_per_GT=1024,anchor_loss_coefficient=1.,extension_weight=.25,
        updates_per_epoch_per_arm=1661,AdamW_lr=.0003,AdamW_weight_decay=.0001,
        AdamW_betas=[.9,.999],AdamW_eps=1e-8,
        main_AP_endpoint=24,selection='Omega',score='original',no_DEV_during_training=True,
        no_TEST=True,reference_recomputed_same_entry=True)
    need(all(pre[k]==v for k,v in expected.items()),'Fixed anchor, optimizer, budget and main evaluation')
    need(type(pre['training_seed']) is int and pre['training_seed'] in
         (202610172401,202610172402,202610172403),'Predeclared downstream order seed only')
    need([spec['first_epoch'],spec['last_epoch']] in expected['stage_windows']
         and ((spec['first_epoch']==1 and spec['resume'] is None)
              or (spec['first_epoch']>1 and isinstance(spec['resume'],dict))),
         'Only fixed eight-epoch stages and explicit full-state resume')


def load_module(control,ref,name):
    p=control.closed(ref);ms=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m);return m


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(torch,v) for v in value)
    return value


def same_tree(torch,a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(torch,a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(torch,x,y) for x,y in zip(a,b))
    return a==b


def save_tensor_object(torch,control,path,value):
    p=Path(path);need(not p.exists() and control.output in p.resolve().parents,'Fresh owned checkpoint')
    control.check()
    with p.open('xb') as stream:torch.save(value,stream)
    replay=torch.load(str(p),map_location='cpu')
    need(same_tree(torch,value,replay),'Complete checkpoint CPU readback')
    return control.ref(p)


def digest_state(torch,state):
    proof={}
    for name,value in state.items():
        t=value.detach().cpu().contiguous()
        proof[name]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
    return proof


def prepare_images(torch,np,spec,control,reader,pre):
    read=lambda ref:json.loads(control.closed(ref).read_bytes())
    old_report,new_report=read(spec['old_cache_report']),read(spec['new_cache_report'])
    accepted=read(spec['N16_ROOT_qualified'])
    need(accepted['fully_qualified'] is True and accepted['GPU_report']==spec['new_cache_report'],
         'Previously independently qualified full C cache')
    old_binding,new_binding=pre['old_cache_binding'],pre['new_cache_binding']
    for binding in (old_binding,new_binding):
        for key,value in binding.items():
            if isinstance(value,dict):control.closed(value)
    need(old_binding['teacher_checkpoint']==new_binding['teacher_checkpoint'], 'Identical frozen teacher identity')
    for report,binding in ((old_report,old_binding),(new_report,new_binding)):
        for key in ('source','spec','GPU_admission','phase','teacher_checkpoint'):
            need(report[key]==binding[key],'Exact cache report parent role '+key)
        need(report['images']==741 and report['raw_GT']==1661 and len(report['rows'])==741,
             'Complete original FIT population')
    need(old_report['packet_refs']==spec['old_packet_refs'] and new_report['packet_refs']==spec['new_packet_refs'],
         'Fixed lightweight packet lists only')
    images=[];audit=dict(images=0,GT=0,K0=0,old_points=0,new_points=0,new_Omega_points=0,
                        original_anchors_retained=True,all_canonical_GT_query_pairs_equal=True)
    for index,(old_row,new_row,old_ref,new_ref) in enumerate(zip(old_report['rows'],new_report['rows'],
                                                              spec['old_packet_refs'],spec['new_packet_refs'])):
        control.check()
        need(old_row['dataset_index']==new_row['dataset_index']==index
             and old_row['image_id']==new_row['image_id']==pre['fit_image_ids'][index]
             and old_row['raw_GT_ids']==pre['fit_ann_ids_by_image'][index], 'Fixed canonical GT order')
        old_packet=torch.load(str(control.closed(old_ref)),map_location='cpu')
        new_packet=torch.load(str(control.closed(new_ref)),map_location='cpu')
        joint=reader.build_joint_image(np,old_packet,old_row,new_packet,new_row,old_binding,new_binding)
        need(joint['raw_GT_order']==pre['fit_ann_ids_by_image'][index], 'Actual canonical GT manifest')
        batches=[]
        for batch in joint['batches']:
            converted={k:v for k,v in batch.items() if k not in ('old','new')}
            for pool in ('old','new'):
                value=batch[pool]
                converted[pool]=None if value is None else {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in value.items()}
            audit['GT']+=1;audit['K0']+=int(batch['K']==0)
            audit['old_points']+=batch['old']['n']
            if batch['new'] is not None:
                audit['new_points']+=batch['new']['n']
                audit['new_Omega_points']+=int(batch['new']['Omega_flag'].sum())
            batches.append(converted)
        images.append(batches);audit['images']+=1
        del old_packet,new_packet,joint,batches
        gc.collect();control.check()
        if (index+1)%128==0 or index==740:
            print(json.dumps(dict(stage='N23_CANONICAL_JOINT_CPU_INTAKE',images=index+1,GT=audit['GT'])),flush=True)
    need(audit['images']==741 and audit['GT']==1661 and audit['K0']==11
         and audit['old_points']==1700864 and audit['new_points']==1517568
         and audit['new_Omega_points']==24955, 'All original anchors and finite C samples retained')
    need(not torch.cuda.is_initialized(),'Joint CPU intake precedes CUDA')
    return images,audit


def qualify_fast_math(torch,head,core,loss_terms,batch):
    """Exact same batch/chunk canonical forward, loss and every gradient; no step."""
    old=batch['old'];new=batch['new']
    if new is None:
        new=dict(F=old['F'][:0],L=old['L'][:0],targets=old['targets'][:0],
                 Omega_flag=old['point_in_region'][:0],importance_weights=old['L'][:0])
    F=torch.cat((old['F'],new['F']),dim=0).to('cuda:0')
    L=torch.cat((old['L'],new['L']),dim=0).to('cuda:0')
    n=old['n'];e=old['e0'].to('cuda:0').expand(len(L),256)
    patch=L.new_zeros((len(L),49));old_target=old['targets'].to('cuda:0',dtype=torch.float32)
    new_target=new['targets'].to('cuda:0',dtype=torch.float32)
    old_flag=old['point_in_region'].to('cuda:0');omega=new['Omega_flag'].to('cuda:0')
    weights=new['importance_weights'].to('cuda:0')
    residual_a=head(F,e,L,patch,arm='PR')['residual'].detach()
    residual_b=core.fast_residual(torch,head,F,L).detach()
    need(torch.equal(residual_a,residual_b),'Fast and canonical residual bitexact on actual joint batch')
    for arm in ARMS:
        gradients=[];values=[]
        for canonical in (True,False):
            head.zero_grad(set_to_none=True)
            u=head(F,e,L,patch,arm='PR')['residual'] if canonical else core.fast_residual(torch,head,F,L)
            anchor=torch.nn.functional.binary_cross_entropy_with_logits(
                torch.where(old_flag,L[:n]+u[:n],L[:n]),old_target,reduction='none').mean()
            logits=L[n:]+u[n:] if arm=='B_C' else torch.where(omega,L[n:]+u[n:],L[n:])
            losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,new_target,reduction='none')
            terms=loss_terms.combine_loss(anchor,losses,omega,weights,arm,extension_weight=.25)
            terms['total'].backward();values.append(terms['total'].detach().clone())
            need(all(p.grad is not None for p in head.parameters()),'Actual all-eight gradient graph')
            gradients.append([p.grad.detach().clone() for p in head.parameters()])
        need(torch.equal(values[0],values[1]) and all(torch.equal(a,b) for a,b in zip(*gradients)),
             'Fast/canonical actual objective and all gradients bitexact '+arm)
        if batch['K']==0:
            need(all(not bool(g.any()) for g in gradients[0]),'K0 gives eight real zero gradients, not None')
    head.zero_grad(set_to_none=True)
    return dict(actual_batch_points=int(len(L)),residual_bitexact=True,all_three_objectives_bitexact=True,
                all_eight_gradients_each_arm_bitexact=True,optimizer_steps=0,
                scope='One actual joint batch and identical concat/chunk only; no global numeric equivalence claim')


def run(torch,np,spec,control):
    read=lambda ref:json.loads(control.closed(ref).read_bytes())
    pre=read(spec['prelock']);validate_recipe(pre,spec)
    for key in ('reader_source','adapter_source','loss_source','core_source','factory_source'):
        control.closed(spec[key])
    adapter=load_module(control,spec['adapter_source'],'dataset_adapter')
    need('dataset_adapter' not in sys.modules,'No shadow adapter import')
    sys.modules['dataset_adapter']=adapter
    reader=load_module(control,spec['reader_source'],'N23_CANONICAL_SHARED_READER')
    losses=load_module(control,spec['loss_source'],'N23_ANCHORED_LOSS_TERMS')
    core=load_module(control,spec['core_source'],'N23_ANCHORED_TRAINING_CORE')
    factory=load_module(control,spec['factory_source'],'N23_ORIGINAL_PR_FACTORY')
    outside=torch.random.get_rng_state().clone()
    images,join_audit=prepare_images(torch,np,spec,control,reader,pre)
    initial_cp=torch.load(str(control.closed(spec['initial_checkpoint'])),map_location='cpu')
    need(initial_cp['meta']['arm']=='LOG_PR' and initial_cp['meta']['seed']==202610120701
         and initial_cp['meta']['completed_head_epochs']==24 and initial_cp['meta']['updates']==39864,
         'Original strong E24 warm initialization; old optimizer is not restored')
    heads={arm:factory.make_point_head(torch,pre['training_seed']) for arm in ARMS}
    for head in heads.values():
        head.load_state_dict(initial_cp['state_dict'],strict=True)
        need(len(list(head.parameters()))==8 and sum(p.numel() for p in head.parameters())==40497,'Original capacity')
    need(all(same_tree(torch,heads[ARMS[0]].state_dict(),heads[a].state_dict()) for a in ARMS),
         'Three identical warm states')
    need(len({p.data_ptr() for h in heads.values() for p in h.parameters()})==24,'No shared parameter storage')
    initial_proof=digest_state(torch,heads[ARMS[0]].state_dict())
    order_rng=torch.Generator(device='cpu');order_rng.manual_seed(pre['training_seed'])
    history=[];orders=[];resume=None
    if spec['resume'] is not None:
        resume=torch.load(str(control.closed(spec['resume']['checkpoint'])),map_location='cpu')
        need(resume['meta']['schema']=='N23-anchored-complete-stage-state-v1'
             and resume['meta']['completed_extra_epochs']==spec['first_epoch']-1
             and resume['meta']['prelock']==spec['prelock']
             and resume['meta']['initial_state_proof']==initial_proof,'Qualified complete prior stage resume')
        for arm in ARMS:heads[arm].load_state_dict(resume['heads'][arm],strict=True)
        order_rng.set_state(resume['private_order_state']);history=resume['history'];orders=resume['epoch_orders']
    need(torch.equal(outside,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
         'CPU construction/load and outside RNG precede CUDA')
    torch.cuda.init();torch.cuda.set_per_process_memory_fraction(
        min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    for head in heads.values():head.to('cuda:0');head.train()
    optimizers={arm:torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.0001,
                                     betas=(.9,.999),eps=1e-8) for arm,head in heads.items()}
    if resume is not None:
        for arm in ARMS:optimizers[arm].load_state_dict(resume['optimizers'][arm])
    else:need(all(not opt.state for opt in optimizers.values()),'Actually new optimizers at added stage')
    output=Path(spec['output_directory']);output.mkdir()
    examples=[next(b for image in images for b in image if b['new'] is not None and bool(b['new']['Omega_flag'].any())),
              next(b for image in images for b in image if b['new'] is not None and not bool(b['new']['Omega_flag'].any())),
              next(b for image in images for b in image if b['K']==0)]
    numeric=dict(cases=[qualify_fast_math(torch,heads['A_COMMON'],core,losses,b) for b in examples],
                 covers_nonzero_Omega_and_zero_new_Omega_and_K0=True)
    for head in heads.values():head.zero_grad(set_to_none=True)
    audit_ref=control.write_json(output/'JOINT_INTAKE.json',dict(join=join_audit,fast_math=numeric,
        initial_state_proof=initial_proof,heads_equal_at_first_added_stage=spec['first_epoch']==1,
        resumed_complete_state=spec['resume'] is not None,training_started=False))
    for epoch in range(spec['first_epoch'],spec['last_epoch']+1):
        order=torch.randperm(741,generator=order_rng).tolist();orders.append(order)
        stats=core.train_one_epoch(torch,heads,optimizers,images,order,losses,control,extension_weight=.25)
        history.append(dict(epoch=epoch,statistics=stats))
        for arm in ARMS:
            opt=optimizers[arm]
            need(len(opt.state)==8 and all(int(v['step'])==1661*epoch for v in opt.state.values()),
                 'Actual per-parameter AdamW step counts '+arm)
            need(all(bool(torch.isfinite(p).all()) for p in heads[arm].parameters()),'Finite endpoint parameters')
        need(torch.equal(outside,torch.random.get_rng_state()),'Private order RNG only')
        print(json.dumps(dict(stage='N23_ANCHORED_DOMAIN_TRAIN',extra_epoch=epoch,
                              cumulative_updates_each=1661*epoch,statistics=stats)),flush=True)
        control.check()
    last=spec['last_epoch'];metadata=dict(schema='N23-anchored-complete-stage-state-v1',prelock=spec['prelock'],
        completed_extra_epochs=last,updates_per_arm=1661*last,initial_state_proof=initial_proof,
        source=spec['entry_source'],spec=control.execution_spec_ref,admission=control.execution_admission_ref,
        initial_checkpoint=spec['initial_checkpoint'],initialization_training_updates_separate=39864,
        optimizer_reset_at_extra_epoch1=True,original_optimizer_continuation=False,
        whole_added_stage_budget=24,main_AP_endpoint=24,AP=False)
    bundle=dict(meta=metadata,heads={a:cpu_tree(torch,h.state_dict()) for a,h in heads.items()},
                optimizers={a:cpu_tree(torch,o.state_dict()) for a,o in optimizers.items()},
                private_order_state=order_rng.get_state().clone(),history=history,epoch_orders=orders)
    state_ref=save_tensor_object(torch,control,output/('TRAIN_STATE_EXTRA_E%02d.pth'%last),bundle)
    cp_refs={}
    for arm in ARMS:
        cp=dict(state_dict=bundle['heads'][arm],meta=dict(metadata,arm=arm,parameter_scalars=40497,parameter_tensors=8))
        cp_refs[arm]=save_tensor_object(torch,control,output/('HEAD_%s_EXTRA_E%02d.pth'%(arm,last)),cp)
    report=dict(schema='N23-anchored-training-stage-report-v1',status='COMPLETE_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        source=spec['entry_source'],prelock=spec['prelock'],first_extra_epoch=spec['first_epoch'],last_extra_epoch=last,
        leaf_updates_each=1661*(last-spec['first_epoch']+1),cumulative_extra_updates_each=1661*last,
        arms=list(ARMS),history=history,epoch_orders=orders,join_audit_ref=audit_ref,
        checkpoint_refs=cp_refs,resume_state_ref=state_ref,initial_state_proof=initial_proof,
        actual_matched_GT=1661,original_points_per_GT=1024,new_points_fixed_empirical_pool=1517568,
        teacher_forward_calls=0,parameter_scalars_each=40497,initialization_training_updates_separate=39864,
        reset_optimizer_only_at_extra_epoch1=True,main_AP_endpoint=24,no_DEV=True,no_TEST=True,AP=False,
        CPU_checkpoint_readback_exact=True,independent_optimization_trajectory_replayed=False,
        qualified=False,original_actual0_self_certified=False,innovation_established=False,goal_complete=False)
    report_ref=control.write_json(output/'TRAIN_REPORT.json',report)
    return dict(report_ref=report_ref,checkpoint_refs=cp_refs,resume_state_ref=state_ref,
                first_extra_epoch=spec['first_epoch'],last_extra_epoch=last,leaf_updates_each=report['leaf_updates_each'],
                cumulative_extra_updates_each=1661*last,AP=False,innovation_established=False)
