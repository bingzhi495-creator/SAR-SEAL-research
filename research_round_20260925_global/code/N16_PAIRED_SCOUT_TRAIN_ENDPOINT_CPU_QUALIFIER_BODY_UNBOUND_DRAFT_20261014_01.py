"""UNBOUND independent endpoint CPU checks; no IO/source/RAW gate/CUDA/GT.
ROOT supplies qualified thin cache inputs and actual GPU checkpoint snapshots.
No optimizer trajectory replay or GPU point-logit/selection/AP proof is claimed.
"""
import math

ARMS = ('UTILITY', 'ERROR')
INIT_SEED = 202610140161
ORDER_SEED = 202610140160001
ADAMW = dict(lr=3e-4, weight_decay=1e-4, betas=(.9,.999), eps=1e-8)
STATE_SHAPES = {'hidden.weight':(32,513),'hidden.bias':(32,), 'output.weight':(3,32),'output.bias':(3,)}
ABS_LOGITS = 1e-4
ABS_MEAN_CE = 1e-4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _finite_number(value):
    return type(value) in (float,int) and math.isfinite(value)


def _tensor(torch, value, shape, dtype, name):
    require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.layout==torch.strided
            and value.dtype==dtype and tuple(value.shape)==tuple(shape) and not value.requires_grad
            and bool(torch.isfinite(value).all()), 'Exact finite CPU tensor required: '+name)


def _models(torch, models):
    require(type(models) is dict and set(models)==set(ARMS), 'Two fixed scout arms required')
    for arm in ARMS:
        require(isinstance(models[arm],dict) and set(models[arm])==set(STATE_SHAPES), 'Four named tensors required; dict/OrderedDict accepted')
        for name,shape in STATE_SHAPES.items():
            _tensor(torch,models[arm][name],shape,torch.float32,arm+'.'+name)


def validate_initial(torch, scout_module, initial):
    _models(torch,initial['models'])
    require(initial['initialization_seed']==INIT_SEED and initial['order_seed']==ORDER_SEED and
            initial['completed_epoch']==0 and initial['updates']=={a:0 for a in ARMS} and
            initial['visited_images']==initial['empty_images']==initial['sampled_points']==0 and
            initial['history']==initial['epoch_orders']==[] and initial['last_order_rng'] is None,
            'Actual common cold initialization evidence required')
    head=scout_module.make_scout(torch,INIT_SEED)
    require(not tuple(head.buffers()) and len(tuple(head.parameters()))==4 and sum(p.numel() for p in head.parameters())==16547,
            'CPU factory must have four16547 parameters/no buffers')
    reference_groups=torch.optim.AdamW(head.parameters(),**ADAMW).state_dict()['param_groups']
    require(set(initial['optimizers'])==set(ARMS), 'Two common initial optimizer records required')
    for arm in ARMS:
        require(all(torch.equal(v,initial['models'][arm][k]) for k,v in head.state_dict().items()),
                'Actual initial weights must equal independently seeded CPU factory')
    for arm in ARMS:
        require(initial['optimizers'][arm]['state']=={} and initial['optimizers'][arm]['param_groups']==reference_groups,
                'No fictitious initial moments; exact initial AdamW groups/hyperparameters required')
    return True


def _cache_counts(rows):
    require(type(rows) is list and bool(rows) and all(type(r['points']) is int and 0<=r['points']<=4096
            and r['dataset_index']==i for i,r in enumerate(rows)), 'Canonical per-image cache counts required')
    return len(rows),sum(r['points']>0 for r in rows),sum(r['points'] for r in rows)


def validate_state_evidence(torch, scout_module, snapshot, cache_rows, epoch):
    require(type(epoch) is int and epoch in (8,24), 'Only fixed8/24 endpoints')
    _models(torch,snapshot['models'])
    count,nonempty,points=_cache_counts(cache_rows)
    updates=epoch*nonempty
    require(type(snapshot['completed_epoch']) is int and snapshot['completed_epoch']==epoch and
            snapshot['initialization_seed']==INIT_SEED and snapshot['order_seed']==ORDER_SEED and
            snapshot['fixed_epochs']==24 and snapshot['AdamW']==ADAMW and
            set(snapshot['updates'])==set(ARMS) and all(type(snapshot['updates'][a]) is int and snapshot['updates'][a]==updates for a in ARMS),
            'Fixed protocol/epoch/exact integer paired updates required')
    require(type(snapshot['visited_images']) is int and snapshot['visited_images']==epoch*count and
            type(snapshot['empty_images']) is int and snapshot['empty_images']==epoch*(count-nonempty) and
            type(snapshot['sampled_points']) is int and snapshot['sampled_points']==epoch*points,
            'Complete epoch/image/empty/point exposure identity required')
    require(len(snapshot['history'])==len(snapshot['epoch_orders'])==epoch, 'Complete history/order prefix required')
    for number in range(1,epoch+1):
        rng=torch.Generator(device='cpu').manual_seed(ORDER_SEED+number)
        before=rng.get_state().clone();order=torch.randperm(count,generator=rng).tolist();after=rng.get_state().clone()
        history=snapshot['history'][number-1]
        require(snapshot['epoch_orders'][number-1]==order and history['order']==order and history['epoch']==number and
                history['order_seed']==ORDER_SEED+number and history['visited_images']==count and
                history['updated_images']==nonempty and history['empty_images']==count-nonempty and history['points']==points,
                'Private seed+epoch randperm and actual empty-update history identity required')
        require(set(history['online_sum_image_weighted_CE'])==set(ARMS) and
                all(_finite_number(v) and v>=0 for v in history['online_sum_image_weighted_CE'].values()), 'Finite online history required')
        if number==epoch:
            final_rng=snapshot['last_order_rng']
            require(final_rng['seed']==ORDER_SEED+number and torch.equal(final_rng['before'],before) and torch.equal(final_rng['after'],after),
                    'Actual private generator before/after state must match final epoch')
    head=scout_module.make_scout(torch,INIT_SEED)
    names=list(dict(head.named_parameters()))
    require(names==list(STATE_SHAPES), 'Actual factory named parameter order must be exact')
    reference=torch.optim.AdamW(head.parameters(),**ADAMW).state_dict()['param_groups']
    require(set(snapshot['optimizers'])==set(ARMS), 'Two independent optimizer states required')
    for arm in ARMS:
        optimizer=snapshot['optimizers'][arm]
        require(optimizer['param_groups']==reference, 'Every AdamW hyperparameter/flag and parameter-ID order must equal reference factory')
        ids=optimizer['param_groups'][0]['params']
        require(len(ids)==len(set(ids))==4 and all(type(i) is int for i in ids) and
                set(optimizer['state'])==(set(ids) if updates else set()), 'Unique full four-parameter moment coverage required')
        for ident,name in zip(ids,names):
            if not updates:
                continue
            value=optimizer['state'][ident]
            require(set(value)=={'step','exp_avg','exp_avg_sq'}, 'Plain non-amsgrad AdamW state keys required')
            step=value['step']
            _tensor(torch,step,(),torch.float32,arm+'.step')
            require(float(step)==updates, 'Finite scalar AdamW step must equal actual updates exactly, never int-truncate')
            for field in ('exp_avg','exp_avg_sq'):
                _tensor(torch,value[field],STATE_SHAPES[name],torch.float32,arm+'.'+name+'.'+field)
            require(bool((value['exp_avg_sq']>=0).all()), 'Second moment cannot be negative')
    return dict(images=count,nonempty_images=nonempty,sampled_points=points,updates=updates)


def validate_trace(trace_records, final_snapshot, cache_rows):
    """Independently connect full actual observer trace to order/history/counters."""
    count,nonempty,points=_cache_counts(cache_rows)
    cursor=0;updates={a:0 for a in ARMS}
    for epoch,order in enumerate(final_snapshot['epoch_orders'],1):
        sums={a:0. for a in ARMS}
        for index in order:
            require(cursor<len(trace_records), 'Incomplete actual training trace')
            row=trace_records[cursor];cursor+=1
            require(row==dict(event='before_image',epoch=epoch,dataset_index=index), 'Actual train trace order differs')
            if cache_rows[index]['points']==0:
                continue
            require(cursor<len(trace_records), 'Missing actual paired update trace')
            row=trace_records[cursor];cursor+=1
            require(row['event']=='paired_image_update' and row['epoch']==epoch and row['dataset_index']==index and
                    row['points']==cache_rows[index]['points'] and set(row['arms'])==set(ARMS), 'Paired trace point/arm identity required')
            for arm in ARMS:
                updates[arm]+=1
                require(type(row['arms'][arm]['updates']) is int and row['arms'][arm]['updates']==updates[arm] and
                        _finite_number(row['arms'][arm]['weighted_CE']) and row['arms'][arm]['weighted_CE']>=0, 'Actual arm update/loss trace differs')
                sums[arm]+=row['arms'][arm]['weighted_CE']
        require(sums==final_snapshot['history'][epoch-1]['online_sum_image_weighted_CE'], 'Trace online means cannot differ from checkpoint history')
        if epoch in (8,24):
            for index in range(count):
                require(cursor<len(trace_records), 'Missing full fixed endpoint readback trace')
                require(trace_records[cursor]==dict(event='before_fixed_endpoint_image',epoch=epoch,dataset_index=index), 'Fixed endpoint canonical image trace differs')
                cursor+=1
    require(cursor==len(trace_records) and updates==final_snapshot['updates'], 'Complete trace with no extra records and exact paired updates required')
    return True


def _inputs(torch,packet):
    samples=packet['samples'];n=len(samples['query_ids']);U=len(packet['sampled_F_unique'])
    _tensor(torch,packet['sampled_F_unique'],(U,256),torch.float32,'sampled_F_unique')
    _tensor(torch,packet['native_e0'],(1,100,256),torch.float32,'native_e0')
    _tensor(torch,packet['sample_to_unique'],(n,),torch.int64,'sample_to_unique')
    _tensor(torch,samples['query_ids'],(n,),torch.int64,'sample query IDs')
    require(bool(((samples['query_ids']>=0)&(samples['query_ids']<100)).all()) and
            (n==0 or bool(((packet['sample_to_unique']>=0)&(packet['sample_to_unique']<U)).all())), 'Captured input indices must be valid')
    for name in ('native_L','importance_weights'):
        _tensor(torch,samples[name],(n,),torch.float32,name)
    require(n==0 or bool((samples['importance_weights']>0).all()), 'Importance weights must be positive')
    for name in ('utility_labels','error_labels'):
        _tensor(torch,samples[name],(n,),torch.int64,name)
        require(bool(((samples[name]>=0)&(samples[name]<=2)).all()), 'Fixed labelclass range0..2 required')
    return (packet['sampled_F_unique'][packet['sample_to_unique']],packet['native_e0'][0,samples['query_ids']],samples['native_L'],samples,n)


def _mean_CE(torch,logits,labels,weights):
    require(len(labels)>0, 'Empty image has no fictitious loss')
    return float((weights*torch.nn.functional.cross_entropy(logits,labels,reduction='none')).mean())


def validate_endpoint(torch,scout_module,formula_module,initial,snapshot,GPU_endpoint,loader_callback,cache_rows,epoch,observer=None):
    validate_initial(torch,scout_module,initial)
    counts=validate_state_evidence(torch,scout_module,snapshot,cache_rows,epoch)
    require(GPU_endpoint['epoch']==epoch and GPU_endpoint['nonempty_images']==counts['nonempty_images'] and
            GPU_endpoint['empty_images']==counts['images']-counts['nonempty_images'] and GPU_endpoint['sampled_points']==counts['sampled_points'],
            'GPU fixed endpoint must cover same full qualified cache')
    heads={a:scout_module.make_scout(torch,INIT_SEED) for a in ARMS}
    for arm in ARMS:
        heads[arm].load_state_dict(snapshot['models'][arm],strict=True);heads[arm].requires_grad_(False);heads[arm].eval()
        require(all(torch.equal(v,snapshot['models'][arm][k]) for k,v in heads[arm].state_dict().items()), 'CPU parameter readback must be exact')
    sums={a:dict(factory=0.,independent=0.,max_logit_absolute_error=0.) for a in ARMS}
    total_points=valid=0
    with torch.no_grad():
        for index in range(counts['images']):
            if observer is not None:observer(dict(event='before_endpoint_CPU_image',epoch=epoch,dataset_index=index))
            packet=loader_callback(index)
            F,e,L,samples,n=_inputs(torch,packet)
            require(n==cache_rows[index]['points'], 'Full-cache input sample count must match qualified report')
            if n==0:
                del packet,F,e,L,samples
                continue
            valid+=1;total_points+=n
            for arm,key in (('UTILITY','utility_labels'),('ERROR','error_labels')):
                factory=heads[arm](F,e,L)
                independent=formula_module.independent_logits(torch,snapshot['models'][arm],F,e,L)
                _tensor(torch,factory,(n,3),torch.float32,'CPU factory logits')
                _tensor(torch,independent,(n,3),torch.float32,'CPU independent logits')
                error=float((factory.double()-independent.double()).abs().max())
                require(error<=ABS_LOGITS, 'CPU formula numeric qualification failure: absolute1e-4 logits; no relative relaxation')
                sums[arm]['max_logit_absolute_error']=max(sums[arm]['max_logit_absolute_error'],error)
                sums[arm]['factory']+=_mean_CE(torch,factory,samples[key],samples['importance_weights'])
                sums[arm]['independent']+=_mean_CE(torch,independent,samples[key],samples['importance_weights'])
                del factory,independent
            del packet,F,e,L,samples
    require(valid==counts['nonempty_images'] and total_points==counts['sampled_points'], 'Complete cached FIT endpoint CPU exposure required')
    comparisons={}
    for arm in ARMS:
        observed=GPU_endpoint['arms'][arm]['mean_image_weighted_CE']
        require((observed is None and valid==0) or (_finite_number(observed) and observed>=0 and valid>0), 'Actual GPU aggregate endpoint loss required')
        values={path:sums[arm][path]/valid if valid else None for path in ('factory','independent')}
        errors={path:abs(values[path]-observed) if valid else 0. for path in values}
        require(all(error<=ABS_MEAN_CE for error in errors.values()), 'CPU vs recorded GPU aggregate meanCE qualification failure: absolute1e-4 only')
        comparisons[arm]=dict(GPU_mean_image_weighted_CE=observed,CPU_means=values,absolute_meanCE_errors=errors,
            CPU_two_formula_max_logit_absolute_error=sums[arm]['max_logit_absolute_error'])
    return dict(epoch=epoch,actual_updates_per_arm=counts['updates'],images=counts['images'],nonempty_images=valid,
        sampled_points=total_points,comparisons=comparisons,abs_logits=ABS_LOGITS,abs_meanCE=ABS_MEAN_CE,rtol=0.,
        qualified_checks_only=True,GPU_pointwise_logits_qualified=False,selected_indices_qualified=False,AP_evaluated=False,
        optimizer_trajectory_replayed=False,scope='Exact states/moments/steps/order/counts plus same-input CPU two-formula logits and GPU aggregate weightedCE; ROOT actual0/source closure pending')
