"""UNBOUND_DRAFT: N17 endpoint CPU numerical body only, no IO/RAW/owner gate.
ROOT supplies actual CPU snapshots, qualified packet+NPY loader and endpoint aggregate.
Initial digest/source/terminal qualification belongs to ROOT. No optimizer replay,
GPU point-logit/ranking/selection/mask/AP proof or actual scientific run by author.
"""
import math
import numpy as np
ARMS=('LOCAL_ERROR','CENTER_ERROR')
INIT_SEED=202610150171
ORDER_SEED=202610140160001
ADAMW=dict(lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
STATE_SHAPES={'hidden.weight':(32,521),'hidden.bias':(32,),'output.weight':(3,32),'output.bias':(3,)}
ABS_LOGITS=1e-4
ABS_MEAN_CE=1e-4

def require(condition,message):
    if not condition:raise RuntimeError(message)
def _finite_number(value):
    return type(value) in (float,int) and math.isfinite(value)
def _tensor(torch,value,shape,dtype,name):
    require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.device.type=='cpu' and
        value.dtype==dtype and tuple(value.shape)==tuple(shape) and not value.requires_grad and bool(torch.isfinite(value).all()),
        'Exact finite CPU tensor required: '+name)
def _model(torch,state):
    require(isinstance(state,dict) and set(state)==set(STATE_SHAPES),'Four named CPU tensors required; dict/OrderedDict accepted')
    for name,shape in STATE_SHAPES.items():_tensor(torch,state[name],shape,torch.float32,name)
def _models(torch,models):
    require(type(models) is dict and set(models)==set(ARMS),'Two fixed ERROR arms required')
    for arm in ARMS:_model(torch,models[arm])
def _factory(torch,scout_module,base_factory):
    head=scout_module.make_scout(torch,INIT_SEED,base_factory)
    require(not tuple(head.buffers()) and len(tuple(head.parameters()))==4 and sum(p.numel() for p in head.parameters())==16803 and
        list(dict(head.named_parameters()))==list(STATE_SHAPES),'Exact four-tensor16803 N17 factory/parameter order required')
    return head

def _cache_counts(rows):
    require(type(rows) is list and len(rows)==741 and all(type(row['dataset_index']) is int and row['dataset_index']==index and
        type(row['points']) is int and row['points']==4096 for index,row in enumerate(rows)),
        'Only complete canonical FIT741 with4096 original slots/image qualifies, never tiny/empty/partial scope')
    return 741,741,3035136

def validate_state_evidence(torch,scout_module,base_factory,snapshot,cache_rows,epoch):
    require(type(epoch) is int and epoch in (8,24),'Only fixed E8/E24 endpoints')
    _models(torch,snapshot['models']);count,nonempty,points=_cache_counts(cache_rows);updates=epoch*741
    require(type(snapshot['completed_epoch']) is int and snapshot['completed_epoch']==epoch and snapshot['initialization_seed']==INIT_SEED and
        snapshot['order_seed']==ORDER_SEED and snapshot['fixed_epochs']==24 and snapshot['AdamW']==ADAMW and
        snapshot['parameters_per_arm']==16803 and snapshot['allowed_fixed_checkpoints']==[8,24] and
        set(snapshot['updates'])==set(ARMS) and all(type(snapshot['updates'][a]) is int and snapshot['updates'][a]==updates for a in ARMS),
        'Exact fixed endpoint/initial seed/order/optimizer/architecture/update counts required')
    require(type(snapshot['visited_images']) is int and snapshot['visited_images']==epoch*741 and
        type(snapshot['empty_images']) is int and snapshot['empty_images']==0 and
        type(snapshot['sampled_points']) is int and snapshot['sampled_points']==epoch*3035136,
        'Complete paired per-arm update/image/point exposure; do not double shared points')
    require(len(snapshot['history'])==len(snapshot['epoch_orders'])==epoch,'Complete fixed epoch history/order prefix required')
    for number in range(1,epoch+1):
        rng=torch.Generator(device='cpu').manual_seed(ORDER_SEED+number);before=rng.get_state().clone()
        order=torch.randperm(741,generator=rng).tolist();after=rng.get_state().clone();history=snapshot['history'][number-1]
        require(snapshot['epoch_orders'][number-1]==order and history['order']==order and history['epoch']==number and
            history['initialization_seed']==INIT_SEED and history['order_seed']==ORDER_SEED+number and history['visited_images']==741 and
            history['updated_images']==741 and history['empty_images']==0 and history['points']==3035136,
            'Private seed+epoch order and full original image/point history exact')
        require(set(history['online_sum_image_weighted_CE'])==set(ARMS) and
            all(_finite_number(v) and v>=0 for v in history['online_sum_image_weighted_CE'].values()),'Finite per-arm online loss sums required')
        for arm in ARMS:
            mean=history['online_mean_image_weighted_CE_'+arm]
            require(_finite_number(mean) and mean==history['online_sum_image_weighted_CE'][arm]/741,'Exact recorded per-image online mean identity')
        if number==epoch:
            actual=snapshot['last_order_rng']
            _tensor(torch,actual['before'],before.shape,torch.uint8,'private order RNG before')
            _tensor(torch,actual['after'],after.shape,torch.uint8,'private order RNG after')
            require(actual['seed']==ORDER_SEED+number and torch.equal(actual['before'],before) and torch.equal(actual['after'],after),
                'Final private order generator before/after state exact')
    head=_factory(torch,scout_module,base_factory);names=list(dict(head.named_parameters()))
    reference=torch.optim.AdamW(head.parameters(),**ADAMW).state_dict()['param_groups']
    require(set(snapshot['optimizers'])==set(ARMS),'Two full independent optimizer records required')
    for arm in ARMS:
        optimizer=snapshot['optimizers'][arm]
        require(optimizer['param_groups']==reference,'Every AdamW hyperparameter/flag and parameter-ID order exact to original factory')
        ids=optimizer['param_groups'][0]['params']
        require(len(ids)==len(set(ids))==4 and all(type(i) is int for i in ids) and set(optimizer['state'])==set(ids),
            'Four unique complete AdamW parameter/moment states required')
        for ident,name in zip(ids,names):
            value=optimizer['state'][ident]
            require(set(value)=={'step','exp_avg','exp_avg_sq'},'Plain non-amsgrad AdamW moment keys required')
            _tensor(torch,value['step'],(),torch.float32,arm+'.step')
            require(float(value['step'])==updates,'Original scalar optimizer step exact5928/17784, never int-truncate')
            for field in ('exp_avg','exp_avg_sq'):_tensor(torch,value[field],STATE_SHAPES[name],torch.float32,arm+'.'+name+'.'+field)
            require(bool((value['exp_avg_sq']>=0).all()),'Second moments must be nonnegative')
    # Trained new columns may remain zero or change; no mandatory arm divergence/equality.
    return dict(images=741,nonempty_images=741,sampled_points=3035136,updates=updates)

def validate_trace(trace_records,snapshot,cache_rows):
    """Check only the actually saved complete paired-image-update substream, not absent events."""
    _cache_counts(cache_rows);epoch=snapshot['completed_epoch']
    require(type(epoch) is int and epoch in (8,24) and type(trace_records) is list and len(trace_records)==epoch*741 and
        len(snapshot['epoch_orders'])==len(snapshot['history'])==epoch,'Complete actual paired-update trace prefix only')
    cursor=0;updates={arm:0 for arm in ARMS}
    for number,order in enumerate(snapshot['epoch_orders'],1):
        sums={arm:0. for arm in ARMS}
        for index in order:
            row=trace_records[cursor];cursor+=1
            require(type(row) is dict and set(row)=={'event','epoch','dataset_index','points','arms'} and
                row['event']=='paired_image_update' and row['epoch']==number and row['dataset_index']==index and
                row['points']==4096 and set(row['arms'])==set(ARMS),'Original paired trace full identity/slot/arm order exact')
            for arm in ARMS:
                item=row['arms'][arm];updates[arm]+=1
                require(set(item)=={'weighted_CE','updates'} and type(item['updates']) is int and item['updates']==updates[arm] and
                    _finite_number(item['weighted_CE']) and item['weighted_CE']>=0,'Trace arm update/loss exact and finite')
                sums[arm]+=item['weighted_CE']
        require(sums==snapshot['history'][number-1]['online_sum_image_weighted_CE'],'Actual trace online sums exact to saved history')
        for arm in ARMS:require(sums[arm]/741==snapshot['history'][number-1]['online_mean_image_weighted_CE_'+arm],'Trace online means exact')
    require(cursor==len(trace_records) and updates==snapshot['updates'],'No missing/extra paired updates')
    return True

def _inputs(torch,loaded):
    require(type(loaded) is dict and set(loaded)=={'packet','neighbors8'},'ROOT qualified thin-packet + actual NPY loader contract required')
    packet=loaded['packet'];samples=packet['samples'];n=len(samples['query_ids']);U=len(packet['sampled_F_unique'])
    _tensor(torch,packet['sampled_F_unique'],(U,256),torch.float32,'sampled_F_unique')
    _tensor(torch,packet['native_e0'],(1,100,256),torch.float32,'native_e0')
    _tensor(torch,packet['sample_to_unique'],(n,),torch.int64,'sample_to_unique')
    for key in ('query_ids','flat_indices'):_tensor(torch,samples[key],(n,),torch.int64,key)
    require(n>0 and bool(((samples['query_ids']>=0)&(samples['query_ids']<100)).all()) and
        bool(((packet['sample_to_unique']>=0)&(packet['sample_to_unique']<U)).all()) and bool((samples['flat_indices']>=0).all()),
        'Captured original query/unique/point indices valid, no empty qualification')
    for key in ('native_L','importance_weights'):_tensor(torch,samples[key],(n,),torch.float32,key)
    require(bool((samples['importance_weights']>0).all()),'Original importance weights positive')
    for key in ('utility_labels','error_labels'):
        _tensor(torch,samples[key],(n,),torch.int64,key);require(bool(((samples[key]>=0)&(samples[key]<=2)).all()),'Original labelclass0..2')
    array=loaded['neighbors8']
    require(isinstance(array,np.ndarray) and array.dtype==np.float32 and array.shape==(n,8) and bool(np.isfinite(array).all()),
        'Actual same-order qualified FP32[N,8] NPY features required; ROOT verifies real refs/point digest')
    extra={'LOCAL_ERROR':torch.from_numpy(np.ascontiguousarray(array)),
        'CENTER_ERROR':(samples['native_L'].clamp(-8,8)/8).unsqueeze(1).repeat(1,8)}
    return packet['sampled_F_unique'][packet['sample_to_unique']],packet['native_e0'][0,samples['query_ids']],samples['native_L'],extra,samples,n

def independent_logits(torch,state,F,e,L,extra8):
    """Explicit biased-variance LN + matmul/bias equations; no nn.LayerNorm/Linear calls."""
    _model(torch,state);n=len(L)
    for name,value,shape in (('F',F,(n,256)),('e',e,(n,256)),('rawL',L,(n,)),('extra8',extra8,(n,8))):
        _tensor(torch,value,shape,torch.float32,name)
    def normalize(value):
        centered=value-value.mean(1,keepdim=True);variance=(centered*centered).mean(1,keepdim=True)
        return centered*torch.rsqrt(variance+1e-5)
    descriptor=torch.cat((normalize(F),normalize(e),(L/8).unsqueeze(1),extra8),1)
    hidden=descriptor.matmul(state['hidden.weight'].t())+state['hidden.bias'];hidden=torch.clamp_min(hidden,0)
    logits=hidden.matmul(state['output.weight'].t())+state['output.bias'];_tensor(torch,logits,(n,3),torch.float32,'independent logits')
    return logits

def _mean_CE(torch,logits,samples):
    return float((samples['importance_weights']*torch.nn.functional.cross_entropy(logits,samples['error_labels'],reduction='none')).mean())

def compare_image(torch,heads,models,loaded):
    """Single input numerical helper; it does not independently qualify partial FIT scope."""
    _models(torch,models);require(set(heads)==set(ARMS),'Both reconstructed CPU ERROR heads required')
    F,e,L,extra,samples,n=_inputs(torch,loaded);arms={}
    with torch.no_grad():
        for arm in ARMS:
            factory=heads[arm](F,e,L,extra[arm]);independent=independent_logits(torch,models[arm],F,e,L,extra[arm])
            _tensor(torch,factory,(n,3),torch.float32,'factory logits')
            error=float((factory.double()-independent.double()).abs().max())
            require(error<=ABS_LOGITS,'CPU two-formula logits exceed original absolute1e-4/rtol0')
            arms[arm]=dict(factory=_mean_CE(torch,factory,samples),independent=_mean_CE(torch,independent,samples),max_logit_absolute_error=error)
    return dict(points=n,arms=arms)

def compare_recorded_means(sums,GPU_endpoint,valid):
    require(type(valid) is int and valid>0 and set(sums)==set(ARMS),'Complete nonempty numerical mean inputs')
    result={}
    for arm in ARMS:
        observed=GPU_endpoint['arms'][arm]['mean_image_weighted_CE']
        require(_finite_number(observed) and observed>=0,'Actual recorded GPU aggregate weightedCE finite')
        means={path:sums[arm][path]/valid for path in ('factory','independent')};errors={path:abs(value-observed) for path,value in means.items()}
        require(all(_finite_number(value) and value>=0 for value in means.values()) and
            all(error<=ABS_MEAN_CE for error in errors.values()),'CPU vs recorded GPU meanCE absolute1e-4/rtol0 compatibility failed')
        result[arm]=dict(GPU_mean_image_weighted_CE=observed,CPU_means=means,absolute_meanCE_errors=errors,
            CPU_two_formula_max_logit_absolute_error=sums[arm]['max_logit_absolute_error'])
    return result

def validate_endpoint(torch,scout_module,base_factory,snapshot,GPU_endpoint,loader_callback,cache_rows,epoch,observer=None):
    counts=validate_state_evidence(torch,scout_module,base_factory,snapshot,cache_rows,epoch)
    require(GPU_endpoint['epoch']==epoch and GPU_endpoint['initialization_seed']==INIT_SEED and
        GPU_endpoint['nonempty_images']==741 and GPU_endpoint['empty_images']==0 and GPU_endpoint['sampled_points']==3035136 and
        set(GPU_endpoint['arms'])==set(ARMS),'Actual GPU aggregate endpoint must cover same full qualified FIT741')
    heads={arm:_factory(torch,scout_module,base_factory) for arm in ARMS}
    for arm in ARMS:
        heads[arm].load_state_dict(snapshot['models'][arm],strict=True);heads[arm].requires_grad_(False);heads[arm].eval()
        require(all(torch.equal(value,snapshot['models'][arm][key]) for key,value in heads[arm].state_dict().items()),'Exact trained CPU state readback')
    sums={arm:dict(factory=0.,independent=0.,max_logit_absolute_error=0.) for arm in ARMS};total=0
    for index in range(741):
        if observer is not None:observer(dict(event='before_endpoint_CPU_image',epoch=epoch,dataset_index=index))
        loaded=loader_callback(index);row=compare_image(torch,heads,snapshot['models'],loaded)
        require(row['points']==cache_rows[index]['points']==4096,'Every actual cached image uses its same4096 slots')
        total+=row['points']
        for arm in ARMS:
            for path in ('factory','independent'):sums[arm][path]+=row['arms'][arm][path]
            sums[arm]['max_logit_absolute_error']=max(sums[arm]['max_logit_absolute_error'],row['arms'][arm]['max_logit_absolute_error'])
        del loaded,row
    require(total==3035136,'Complete same-input fixed FIT CPU exposure required')
    return dict(epoch=epoch,actual_updates_per_arm=counts['updates'],images=741,nonempty_images=741,sampled_points=total,
        comparisons=compare_recorded_means(sums,GPU_endpoint,741),abs_logits=1e-4,abs_meanCE=1e-4,rtol=0.,
        qualified_checks_only=True,initial_digest_checked_by_body=False,GPU_pointwise_logits_qualified=False,
        selected_indices_qualified=False,mask_RLE_qualified=False,AP_evaluated=False,optimizer_trajectory_replayed=False,
        trace_scope='saved paired_image_update substream only; validate_trace called separately',
        scope='Exact trained states/moments/steps/orders/counts plus full same-input CPU two formulas and recorded GPU aggregate meanCE; ROOT closure/initial digest/actual0 pending')
