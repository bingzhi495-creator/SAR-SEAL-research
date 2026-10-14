"""UNBOUND_DRAFT: N17 paired LOCAL/CENTER ERROR numerical training body.
ROOT supplies closed factories/N16 helpers and qualified thin-packet + neighbors8 loader.
No owner gate, direct file IO, teacher/PR loading, DEV/AP or scientific execution here.
"""
import copy
import numpy as np
ARMS=('LOCAL_ERROR','CENTER_ERROR')
INIT_SEED=202610150171
REPEAT_SEEDS=(202610150171,202610150172,202610150173)
EXPECTED_FIRST_INITIALIZATION_DIGEST='5b9f58a3dece9720f84b257d03ae6928a7129e6f364e218f6bd49475d5fb7aa7'
ORDER_SEED=202610140160001
EPOCHS=24
ENDPOINTS=(8,24)
ADAMW=dict(lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)

def require(ok,message):
    if not ok:raise RuntimeError(message)

def make_paired_state(torch,scout_module,base_factory,old_body,device='cpu',seed=INIT_SEED):
    require(type(seed) is int and seed in REPEAT_SEEDS,'Only ROOT-prelocked N17 initialization seeds allowed')
    for name in ('_inputs','weighted_cross_entropy','_statistics_empty','_statistics_add','_cpu_tree'):
        require(callable(getattr(old_body,name,None)),'ROOT-bound original N16 numerical helper required: '+name)
    require(old_body.ORDER_SEED==ORDER_SEED and old_body.EPOCHS==EPOCHS and tuple(old_body.ENDPOINTS)==ENDPOINTS
        and old_body.ADAMW==ADAMW,'Original order/endpoint/AdamW recipe must remain unchanged')
    heads={arm:scout_module.make_scout(torch,seed,base_factory).to(device) for arm in ARMS}
    for head in heads.values():
        require(len(head.state_dict())==len(tuple(head.parameters()))==4 and not tuple(head.buffers())
            and sum(p.numel() for p in head.parameters())==16803 and all(p.dtype==torch.float32 for p in head.parameters())
            and tuple(head.hidden.weight.shape)==(32,521) and torch.count_nonzero(head.hidden.weight[:,513:])==0,
            'Four FP32 tensors16803 with only eight zero-added initial columns required')
    left,right=(heads[arm] for arm in ARMS)
    require(all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items()),'Both ERROR scouts must have identical full initial states')
    require(all(a.data_ptr()!=b.data_ptr() for a,b in zip(left.parameters(),right.parameters())),'Paired scouts must own independent parameter storage')
    return dict(schema='N17-paired-local-center-error-training-body-state-v1',heads=heads,
        optimizers={arm:torch.optim.AdamW(heads[arm].parameters(),**ADAMW) for arm in ARMS},_old_body=old_body,
        initialization_seed=seed,order_seed=ORDER_SEED,completed_epoch=0,updates={arm:0 for arm in ARMS},
        visited_images=0,empty_images=0,sampled_points=0,history=[],epoch_orders=[],last_order_rng=None,
        expected_first_initialization_digest=EXPECTED_FIRST_INITIALIZATION_DIGEST if seed==INIT_SEED else None,
        initialization_digest_check_owned_by_ROOT=True,scientific_source_qualified=False,actual_owned_terminal_self_certified=False)

def _inputs(torch,loaded,device,old_body,full_fit):
    require(type(loaded) is dict and set(loaded)=={'packet','neighbors8'},'ROOT loader must return packet and actual qualified neighbors8 array only')
    common=old_body._inputs(torch,loaded['packet'],device)
    array=loaded['neighbors8'];n=len(loaded['packet']['samples']['query_ids'])
    require(isinstance(array,np.ndarray) and array.dtype==np.float32 and array.shape==(n,8)
        and bool(np.isfinite(array).all()),'Actual qualified FP32[N,8] NPY array required; no resize/grid reconstruction')
    if full_fit:require(n==4096,'Qualified full FIT741 requires4096 unchanged samples per image')
    if common is None:return None
    local=torch.from_numpy(np.ascontiguousarray(array)).to(device)
    center=(common['L'].clamp(-8,8)/8).unsqueeze(1).repeat(1,8)
    common['extra8']={'LOCAL_ERROR':local,'CENTER_ERROR':center}
    # Both heads use the original ERROR target; utility labels survive only as old FIT diagnostics.
    common['labels']={arm:common['labels']['ERROR'] for arm in ARMS}
    return common

def train_one_epoch(torch,state,loader_callback,epoch,image_count=741,observer=None):
    require(type(epoch) is int and 1<=epoch<=24 and state['completed_epoch']==epoch-1
        and type(image_count) is int and image_count>0,'Contiguous fixed1..24 paired epoch required')
    old=state['_old_body'];rng=torch.Generator(device='cpu');rng.manual_seed(ORDER_SEED+epoch)
    before=rng.get_state().clone();order=torch.randperm(image_count,generator=rng).tolist();after=rng.get_state().clone()
    device=next(state['heads'][ARMS[0]].parameters()).device
    require(all(next(state['heads'][arm].parameters()).device==device for arm in ARMS),'Both ERROR scout devices must match')
    for head in state['heads'].values():head.train()
    report=dict(epoch=epoch,initialization_seed=state['initialization_seed'],order_seed=ORDER_SEED+epoch,order=order,
        visited_images=image_count,updated_images=0,empty_images=0,points=0,
        online_sum_image_weighted_CE={arm:0. for arm in ARMS},online_loss_is_not_fixed_endpoint=True)
    for index in order:
        if observer is not None:observer(dict(event='before_image',epoch=epoch,dataset_index=index))
        loaded=loader_callback(index);inputs=_inputs(torch,loaded,device,old,image_count==741)
        if inputs is None:
            report['empty_images']+=1;state['empty_images']+=1;del loaded;continue
        metadata=dict(epoch=epoch,dataset_index=index,points=inputs['n'],arms={})
        for arm in ARMS:
            head,optimizer=state['heads'][arm],state['optimizers'][arm];optimizer.zero_grad(set_to_none=True)
            logits=head(inputs['F'],inputs['e'],inputs['L'],inputs['extra8'][arm])
            loss=old.weighted_cross_entropy(torch,logits,inputs['labels'][arm],inputs['weights'])
            require(bool(torch.isfinite(loss)),'Finite paired weighted CE required');loss.backward()
            require(all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all())
                for p in head.parameters()),'All four finite FP32 gradients required')
            optimizer.step();require(all(bool(torch.isfinite(p).all()) for p in head.parameters()),'AdamW produced nonfinite parameter')
            state['updates'][arm]+=1;metadata['arms'][arm]=dict(weighted_CE=float(loss.detach().cpu()),updates=state['updates'][arm])
            report['online_sum_image_weighted_CE'][arm]+=metadata['arms'][arm]['weighted_CE'];optimizer.zero_grad(set_to_none=True)
            del logits,loss,head,optimizer
        report['updated_images']+=1;report['points']+=inputs['n'];state['sampled_points']+=inputs['n']
        if observer is not None:observer(dict(event='paired_image_update',**metadata))
        del loaded,inputs,metadata
    state['visited_images']+=image_count;state['completed_epoch']=epoch;state['epoch_orders'].append(list(order))
    state['last_order_rng']=dict(seed=ORDER_SEED+epoch,before=before,after=after)
    for arm in ARMS:report['online_mean_image_weighted_CE_'+arm]=report['online_sum_image_weighted_CE'][arm]/report['updated_images'] if report['updated_images'] else None
    state['history'].append(copy.deepcopy(report));return report

def evaluate_fixed(torch,state,loader_callback,image_count=741,observer=None):
    old=state['_old_body'];device=next(state['heads'][ARMS[0]].parameters()).device
    modes={arm:state['heads'][arm].training for arm in ARMS}
    for head in state['heads'].values():head.eval()
    result=dict(epoch=state['completed_epoch'],initialization_seed=state['initialization_seed'],nonempty_images=0,empty_images=0,sampled_points=0,
        arms={arm:dict(sum_image_weighted_CE=0.,per_stratum_flip_statistics=old._statistics_empty()) for arm in ARMS},
        scope='fixed qualified FIT samples only; no DEV/AP',primary_loss='mean per image of mean(weight*ERROR_CE)',
        statistics_scope='original importance-weighted FIT action types; no topK/deployment qualification')
    try:
        with torch.no_grad():
            for index in range(image_count):
                if observer is not None:observer(dict(event='before_fixed_endpoint_image',epoch=state['completed_epoch'],dataset_index=index))
                loaded=loader_callback(index);inputs=_inputs(torch,loaded,device,old,image_count==741)
                if inputs is None:result['empty_images']+=1;del loaded;continue
                result['nonempty_images']+=1;result['sampled_points']+=inputs['n']
                for arm in ARMS:
                    logits=state['heads'][arm](inputs['F'],inputs['e'],inputs['L'],inputs['extra8'][arm])
                    loss=old.weighted_cross_entropy(torch,logits,inputs['labels'][arm],inputs['weights'])
                    result['arms'][arm]['sum_image_weighted_CE']+=float(loss.cpu())
                    old._statistics_add(torch,result['arms'][arm]['per_stratum_flip_statistics'],logits,inputs['samples'],'ERROR')
                    del logits,loss
                del loaded,inputs
        for arm in ARMS:
            item=result['arms'][arm];item['mean_image_weighted_CE']=item['sum_image_weighted_CE']/result['nonempty_images'] if result['nonempty_images'] else None
            for group in item['per_stratum_flip_statistics'].values():
                group['help_recall_on_actual_flips']=group['positive_help']/group['actual_help'] if group['actual_help'] else None
                group['harm_selection_rate_on_actual_flips']=group['positive_harm']/group['actual_harm'] if group['actual_harm'] else None
                selected=group['positive_help']+group['positive_harm']
                group['help_precision_among_positive_actual_flips']=group['positive_help']/selected if selected else None
        return result
    finally:
        for arm in ARMS:state['heads'][arm].train(modes[arm])

def cpu_snapshot(torch,state):
    old=state['_old_body'];require(state['updates'][ARMS[0]]==state['updates'][ARMS[1]],'Paired ERROR updates must remain equal')
    optimizers={arm:old._cpu_tree(torch,state['optimizers'][arm].state_dict()) for arm in ARMS}
    for arm,value in optimizers.items():
        require(len(value['param_groups'])==1 and len(value['param_groups'][0]['params'])==4
            and len(value['state'])==(4 if state['updates'][arm] else 0),'Exactly four complete AdamW parameter states required')
        for item in value['state'].values():
            require(item['step'].numel()==1 and float(item['step'])==state['updates'][arm]
                and bool((item['exp_avg_sq']>=0).all()),'AdamW scalar step exact and second moment nonnegative required')
    return dict(schema='N17-paired-local-center-error-training-CPU-snapshot-v1',models={arm:old._cpu_tree(torch,state['heads'][arm].state_dict()) for arm in ARMS},
        optimizers=optimizers,completed_epoch=state['completed_epoch'],updates=dict(state['updates']),visited_images=state['visited_images'],
        empty_images=state['empty_images'],sampled_points=state['sampled_points'],initialization_seed=state['initialization_seed'],order_seed=ORDER_SEED,
        AdamW=dict(ADAMW),fixed_epochs=24,history=copy.deepcopy(state['history']),epoch_orders=copy.deepcopy(state['epoch_orders']),
        last_order_rng=old._cpu_tree(torch,state['last_order_rng']),allowed_fixed_checkpoints=[8,24],parameters_per_arm=16803,
        expected_first_initialization_digest=state['expected_first_initialization_digest'],initialization_digest_check_owned_by_ROOT=True,
        qualification_pending=True,scientific_source_qualified=False,actual_owned_terminal_self_certified=False)

def train_until(torch,state,loader_callback,last_epoch=24,image_count=741,observer=None,checkpoint_callback=None):
    require(last_epoch in ENDPOINTS and state['completed_epoch']<last_epoch,'Only fixed E8/E24 continuation allowed; no optimizer reset')
    endpoints={}
    for epoch in range(state['completed_epoch']+1,last_epoch+1):
        train_one_epoch(torch,state,loader_callback,epoch,image_count,observer)
        if epoch in ENDPOINTS:
            endpoint=evaluate_fixed(torch,state,loader_callback,image_count,observer);snapshot=cpu_snapshot(torch,state);endpoints[epoch]=endpoint
            if checkpoint_callback is not None:checkpoint_callback(epoch,endpoint,snapshot)
            del snapshot
    if image_count==741:
        require(state['empty_images']==0 and all(v==last_epoch*741 for v in state['updates'].values())
            and state['sampled_points']==last_epoch*741*4096,'Fixed full-FIT counts/exposure must equal prelocked recipe')
    return dict(completed_epoch=state['completed_epoch'],initialization_seed=state['initialization_seed'],updates=dict(state['updates']),
        fixed_endpoints=endpoints,qualification_pending=True,AP_evaluated=False,actual_owned_terminal_self_certified=False)
