"""UNBOUND formal paired dense-routing FIT training; no scientific import at import.

Root owns execution/admission and later terminal intake. This worker reuses the
qualified frozen full-F inputs, verifies them through the reader each epoch and
performs only this fixed eight-epoch segment. It never reads DEV or evaluates AP.
"""
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time


SEEDS=[202610172801,202610172802,202610172803]
MODES=['SELECTIVE','WHOLE_NODE']
STAGES=((0,8),(8,16),(16,24))


def need(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(c,value):
    return json.loads(Path(c.closed(value)).read_bytes())


def load(c,value,name):
    module_spec=importlib.util.spec_from_file_location(name,str(c.closed(value)))
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def cpu_tree(torch,value):
    """Copy all checkpoint tensors off GPU; no live tensor aliases escape."""
    if isinstance(value,torch.Tensor):
        need(not value.requires_grad and bool(torch.isfinite(value).all()),
             'Checkpoint tensors must be detached and finite')
        return value.detach().cpu().clone().contiguous()
    if isinstance(value,dict):
        return {key:cpu_tree(torch,item) for key,item in value.items()}
    if isinstance(value,list):
        return [cpu_tree(torch,item) for item in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(torch,item) for item in value)
    need(value is None or type(value) in (str,int,float,bool),'Only tensor/plain checkpoint state')
    need(type(value) is not float or math.isfinite(value),'Finite checkpoint scalar')
    return value


def same_state(torch,left,right):
    return list(left)==list(right) and all(torch.equal(left[key],right[key]) for key in left)


def state_digest(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def check_optimizer(torch,head,optimizer,expected_steps):
    parameters=list(head.parameters())
    owned=[p for group in optimizer.param_groups for p in group['params']]
    need(len(owned)==len(parameters) and len({id(p) for p in owned})==len(parameters)
         and {id(p) for p in owned}=={id(p) for p in parameters},
         'Optimizer owns exactly all parameters of its original arm')
    for group in optimizer.param_groups:
        need(group['lr']==3e-4 and group['weight_decay']==1e-4
             and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
             and group['foreach'] is False and group['fused'] is False
             and group['amsgrad'] is False and group['maximize'] is False
             and group['capturable'] is False and group['differentiable'] is False,
             'Unchanged fixed AdamW recipe, including resumed groups')
    if expected_steps==0:
        need(len(optimizer.state)==0,'Pristine optimizer must have no state or prior updates')
    else:
        need(len(optimizer.state)==len(parameters),'Every parameter has its original optimizer state')
        for parameter in parameters:
            state=optimizer.state[parameter]
            need(set(state)=={'step','exp_avg','exp_avg_sq'},'Complete ordinary AdamW state')
            step=state['step']
            need(isinstance(step,torch.Tensor) and step.numel()==1
                 and not step.requires_grad and bool(torch.isfinite(step).all())
                 and float(step.item())==expected_steps,'Every parameter step equals exact cumulative image updates')
            for name in ('exp_avg','exp_avg_sq'):
                value=state[name]
                need(isinstance(value,torch.Tensor) and value.shape==parameter.shape
                     and value.dtype==parameter.dtype and value.device==parameter.device
                     and not value.requires_grad and bool(torch.isfinite(value).all()),
                     'Original finite optimizer moments on the active parameter device')
    return dict(parameter_tensors=len(parameters),expected_step=expected_steps,
                all_parameter_steps_exact=True,recipe_exact=True)


def check_epoch(value,epoch,order=None):
    need(type(value) is dict and value['epoch']==epoch
         and value['images']==741 and value['GT_visits']==1661 and value['microbatch']==4
         and value['cumulative_updates_per_arm']==epoch*741
         and value['full_mask_writable'] is True and value['teacher_updated'] is False
         and value['optimizer_updates_per_image_per_arm']==1
         and value['loss_weighting']=='chunk_GT/image_GT; fixed microbatch BN semantics'
         and set(value['arms'])==set(MODES),'Exact complete paired epoch accounting')
    recorded=value['image_order']
    need(type(recorded) is list and len(recorded)==741
         and all(type(index) is int for index in recorded) and set(recorded)==set(range(741)),
         'Each actual epoch visits every original image once')
    if order is not None:
        need(recorded==order,'Both arms use the same precomputed original epoch order')
    for mode in MODES:
        arm=value['arms'][mode]
        need(arm['updates']==741 and arm['GT_visits']==1661
             and arm['microbatches']==value['shared_microbatches']
             and type(arm['active_parameter_tensors']) is int and arm['active_parameter_tensors']>0
             and arm['finite_gradient_checks']==741*arm['active_parameter_tensors'],
             'Every matched GT and every active gradient checked for each arm')
        for name in ('image_loss_sum','instance_loss_sum','mean_image_loss','mean_instance_loss'):
            need(type(arm[name]) in (int,float) and math.isfinite(arm[name]) and arm[name]>=0,
                 'Finite nonnegative actual training-loss summaries')
        need(arm['mean_image_loss']==arm['image_loss_sum']/741
             and arm['mean_instance_loss']==arm['instance_loss_sum']/1661,'Original loss denominators')


def run(torch,np,spec,c,pre):
    started=time.monotonic();c.check()
    seed,start,end=spec['seed'],spec['start_epoch'],spec['end_epoch']
    need(spec['operation']=='GPU_DENSE_ROUTING_PAIRED_TRAIN'
         and type(seed) is int and seed in SEEDS and (start,end) in STAGES
         and pre['schema']=='N28-dense-routing-paired-train-prelock-v1'
         and pre['images']==741 and pre['GT']==1661 and pre['epochs']==24
         and pre['stage_epochs']==8 and pre['microbatch']==4 and pre['updates_per_epoch']==741
         and pre['total_updates_per_arm']==17784 and pre['seeds']==SEEDS and pre['modes']==MODES
         and pre['lr']==3e-4 and pre['weight_decay']==1e-4
         and pre['betas']==[.9,.999] and pre['eps']==1e-8
         and not torch.cuda.is_initialized(),'Exact formal stage with CPU construction before CUDA')
    factory=load(c,spec['factory_source'],'N28_FIXED_TRAIN_FACTORY')
    reader_module=load(c,spec['reader_source'],'N28_FIXED_DENSE_FIT_READER')
    core=load(c,spec['train_core_source'],'N28_FIXED_DENSE_TRAIN_CORE')
    need(not torch.cuda.is_initialized(),'Numerical module imports remain CPU-only')
    context=read(c,spec['context'])
    reader=reader_module.prepare(torch,np,c,context)
    need(len(reader.records)==741 and sum(item['G'] for item in reader.records)==1661,
         'Complete original FIT metadata, never quality-head or sampled-point inputs')
    closure={item['path']:item for item in spec['sources']}
    need(len(closure)==len(spec['sources'])
         and all(closure.get(value['path'])==value for value in reader.input_references),
         'Every actual reader input is already declared in this stage Source closure')
    initialization=read(c,spec['initialization_qualified'])
    need(initialization['schema']=='ROOT-N28-dense-routing-initialize-qualified-v1'
         and initialization['qualified'] is True
         and initialization['worker_first_real_interface_probes_reported_complete'] is True
         and initialization['virgin_trained_updates']==0
         and initialization['factory_source']==spec['factory_source']
         and initialization['seeds']==SEEDS and initialization['modes']==MODES,
         'Only actual accepted common initial states; one-sample validation is not full-FIT qualification')
    initial_refs=initialization['result']['checkpoint_refs']
    need(type(initial_refs) is list and len(initial_refs)==3,'Three admitted pristine seed states')
    initial_ref=initial_refs[SEEDS.index(seed)]
    original_cpu_rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        heads={}
        for mode in MODES:
            torch.random.default_generator.manual_seed(seed)
            heads[mode]=factory.make_head(torch,mode)
    need(torch.equal(torch.get_rng_state(),original_cpu_rng) and not torch.cuda.is_initialized(),
         'CPU-only constructors restore external CPU RNG')
    need(same_state(torch,heads[MODES[0]].state_dict(),heads[MODES[1]].state_dict()),
         'Both modes have identical ordinary construction state before checkpoint load')
    history=[];resume_value=None;resume_proof=None
    if start==0:
        need(spec['initial_checkpoint']==initial_ref,'The requested seed uses its own admitted pristine checkpoint')
        checkpoint=torch.load(str(c.closed(initial_ref)),map_location='cpu')
        need(checkpoint['schema']=='N28-virgin-common-dense-routing-initial-v1'
             and checkpoint['seed']==seed and checkpoint['trained_updates']==0
             and checkpoint['modes']==MODES and checkpoint['factory_source']==spec['factory_source'],
             'Exact original seed/factory/common initial checkpoint with zero updates')
        for mode in MODES:
            heads[mode].load_state_dict(checkpoint['state_dict'],strict=True)
        need(same_state(torch,heads[MODES[0]].state_dict(),heads[MODES[1]].state_dict()),
             'Strict common pristine initialization is identical for both routing arms')
        del checkpoint
    else:
        resume_proof=read(c,spec['resume_qualified'])
        previous=resume_proof['result']
        need(resume_proof['schema']=='ROOT-N28-dense-routing-paired-train-qualified-v1'
             and resume_proof['qualified'] is True
             and previous['checkpoint_ref']==spec['resume_checkpoint']
             and previous['seed']==seed and previous['end_epoch']==start
             and previous['cumulative_updates_per_arm']==start*741
             and previous['new_training'] is True
             and all(previous[key] is False for key in ('AP','DEV','TEST')),
             'Exact accepted preceding segment, same seed and cumulative updates')
        resume_value=torch.load(str(c.closed(spec['resume_checkpoint'])),map_location='cpu')
        need(resume_value['schema']=='N28-dense-routing-paired-train-state-v1'
             and resume_value['seed']==seed and resume_value['completed_epoch']==start
             and resume_value['modes']==MODES
             and all(resume_value[key]==spec[key] for key in
                     ('factory_source','reader_source','train_core_source'))
             and resume_value['cumulative_updates_per_arm']==start*741
             and set(resume_value['models'])==set(resume_value['optimizers'])==set(MODES),
             'Complete original paired model/optimizer continuation checkpoint')
        history=resume_value['history']
        need(type(history) is list and len(history)==start,'Every earlier actual epoch retained')
        for epoch,value in enumerate(history,1):
            check_epoch(value,epoch)
        for mode in MODES:
            heads[mode].load_state_dict(resume_value['models'][mode],strict=True)
    need(all(next(head.parameters()).device.type=='cpu' for head in heads.values())
         and not torch.cuda.is_initialized(),'All strict state restoration starts with CPU models')
    parameter_counts={mode:sum(p.numel() for p in heads[mode].parameters()) for mode in MODES}
    need(len(set(parameter_counts.values()))==1,'Paired equal parameter capacity')
    total_memory=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/total_memory),0)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    for mode in MODES:
        heads[mode].to('cuda:0').train()
    optimizers={mode:torch.optim.AdamW(heads[mode].parameters(),lr=3e-4,weight_decay=1e-4,
        betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False,capturable=False,
        differentiable=False,foreach=False,fused=False) for mode in MODES}
    order_generator=torch.Generator(device='cpu')
    if start==0:
        order_generator.manual_seed(seed)
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        RNG_restore_kind='First stage: complete seed for private order and CPU/CUDA training RNG'
    else:
        for mode in MODES:
            optimizers[mode].load_state_dict(resume_value['optimizers'][mode])
        order_generator.set_state(resume_value['order_generator_state'])
        torch.set_rng_state(resume_value['torch_CPU_RNG_state'])
        torch.cuda.set_rng_state(resume_value['torch_CUDA_RNG_state'],0)
        need(torch.equal(order_generator.get_state(),resume_value['order_generator_state'])
             and torch.equal(torch.get_rng_state(),resume_value['torch_CPU_RNG_state'])
             and torch.equal(torch.cuda.get_rng_state(0),resume_value['torch_CUDA_RNG_state']),
             'Private order, global CPU and global CUDA RNG restored bitexact before continuation')
        RNG_restore_kind='Resume: original private order and CPU/CUDA RNG states restored bitexact'
    optimizer_start={mode:check_optimizer(torch,heads[mode],optimizers[mode],start*741) for mode in MODES}
    initial_order_state_digest=state_digest(order_generator.get_state())
    del resume_value
    out=Path(spec['output_directory']);out.mkdir()
    stage_history=[];checkpoint_refs=[]
    for epoch in range(start+1,end+1):
        c.check()
        order=torch.randperm(741,generator=order_generator,device='cpu').tolist()
        epoch_result=core.train_epoch(torch,heads,optimizers,reader,order,c,microbatch=4)
        value=dict(epoch=epoch,cumulative_updates_per_arm=epoch*741,**epoch_result)
        check_epoch(value,epoch,order)
        history.append(value);stage_history.append(value)
        optimizer_checks={mode:check_optimizer(torch,heads[mode],optimizers[mode],epoch*741) for mode in MODES}
        state=dict(schema='N28-dense-routing-paired-train-state-v1',seed=seed,completed_epoch=epoch,
            modes=MODES,factory_source=spec['factory_source'],reader_source=spec['reader_source'],
            train_core_source=spec['train_core_source'],
            models={mode:cpu_tree(torch,heads[mode].state_dict()) for mode in MODES},
            optimizers={mode:cpu_tree(torch,optimizers[mode].state_dict()) for mode in MODES},
            order_generator_state=cpu_tree(torch,order_generator.get_state()),
            torch_CPU_RNG_state=cpu_tree(torch,torch.get_rng_state()),
            torch_CUDA_RNG_state=cpu_tree(torch,torch.cuda.get_rng_state(0)),
            cumulative_updates_per_arm=epoch*741,history=list(history))
        path=out/('TRAIN_STATE_E%02d.pth'%epoch)
        with path.open('xb') as stream:
            torch.save(state,stream)
        checkpoint_refs.append(c.ref(path))
        del state
        c.check()
        print(json.dumps(dict(N28_seed=seed,completed_epoch=epoch,cumulative_updates_per_arm=epoch*741,
            mean_image_loss={mode:value['arms'][mode]['mean_image_loss'] for mode in MODES})),flush=True)
        del epoch_result,order
    need(len(stage_history)==8 and len(history)==end and len(checkpoint_refs)==8,
         'Exactly eight actual epochs and eight unique owned checkpoints in this segment')
    final_order_state_digest=state_digest(order_generator.get_state())
    for mode in MODES:
        optimizers[mode].zero_grad(set_to_none=True)
    del optimizers
    for mode in MODES:
        heads[mode].zero_grad(set_to_none=True);heads[mode].cpu()
    del heads
    gc.collect();torch.cuda.synchronize(0);c.check()
    # Only copied CPU checkpoint payloads were serialized. Recheck every owned
    # checkpoint hash and the predecessor; never overwrite a prior stage file.
    for reference in checkpoint_refs:
        need(c.ref(reference['path'])==reference,'Saved per-epoch checkpoint unchanged')
    if start==0:
        need(c.ref(initial_ref['path'])==initial_ref,'Original pristine checkpoint unchanged')
    else:
        need(c.ref(spec['resume_checkpoint']['path'])==spec['resume_checkpoint'],
             'Accepted preceding checkpoint remains untouched')
    checkpoint_ref=checkpoint_refs[-1]
    report=dict(schema='N28-dense-routing-paired-train-report-pending-v1',
        status='FIXED_STAGE_TRAINING_COMPLETE_PENDING_ROOT_ACTUAL0',
        source=spec['body_source'],factory_source=spec['factory_source'],reader_source=spec['reader_source'],
        train_core_source=spec['train_core_source'],spec=c.execution_spec_ref,
        GPU_admission=c.execution_admission_ref,phase=spec['phase'],prelock=spec['prelock'],context=spec['context'],
        seed=seed,modes=MODES,start_epoch=start,end_epoch=end,epochs_this_task=8,
        images_per_epoch=741,GT_per_epoch=1661,updates_per_epoch_per_arm=741,
        cumulative_updates_per_arm=end*741,stage_updates_per_arm=8*741,
        microbatch=4,loss_weighting='chunk_GT/image_GT; fixed microbatch BN semantics',
        full_G_single_forward_equivalence_claimed=False,parameter_counts=parameter_counts,
        optimizer=dict(type='AdamW',lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
            foreach=False,fused=False,amsgrad=False,maximize=False,capturable=False,differentiable=False,
            schedule=None,mixed_precision=False,gradient_clip=None),
        optimizer_start_checks=optimizer_start,optimizer_final_checks=optimizer_checks,
        initialization_qualified=spec['initialization_qualified'],initial_checkpoint=initial_ref,
        resume_qualified=spec.get('resume_qualified') if start else None,
        resume_checkpoint=spec.get('resume_checkpoint') if start else None,
        optimizer_state_resumed=(start>0),RNG_restore_kind=RNG_restore_kind,
        private_order_initial_state_digest=initial_order_state_digest,
        private_order_final_state_digest=final_order_state_digest,
        constructor_CPU_global_RNG_restored=True,private_order_same_both_arms=True,
        history=history,stage_history=stage_history,checkpoint_refs=checkpoint_refs,checkpoint_ref=checkpoint_ref,
        reader_input_references=reader.input_references,reader_input_refs_in_Source=True,
        full_FIT741_1661_actual_reader_checks_each_epoch=True,
        full_FIT_scope='Worker actual per-image original F/M digests, source/teacher/query identity and complete GT digest checks during this segment; reused qualified frozen carriers, no new independent teacher forward',
        point_F_substituted_for_dense=False,teacher_forward=False,teacher_or_original_weights_updated=False,
        full_mask_writable=True,AP=False,DEV=False,TEST=False,new_training=True,
        full_24_epochs_complete=(end==24),utility_evaluated=False,qualified=False,
        ROOT_actual_GPU0_verified=False,innovation_established=False,goal_complete=False,
        elapsed_seconds=time.monotonic()-started)
    report_ref=c.write_json(out/'TRAIN_REPORT.json',report)
    return dict(report_ref=report_ref,checkpoint_ref=checkpoint_ref,seed=seed,end_epoch=end,
                cumulative_updates_per_arm=end*741,qualified=False,new_training=True,
                AP=False,DEV=False,TEST=False)
