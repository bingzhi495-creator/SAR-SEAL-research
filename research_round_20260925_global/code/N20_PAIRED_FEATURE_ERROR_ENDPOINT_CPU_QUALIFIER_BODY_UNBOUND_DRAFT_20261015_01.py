# UNBOUND_DRAFT: N20 independent paired endpoint CPU numeric body; no IO/owner front.
"""ROOT injects closed CPU snapshots, same actual-GPU-input sparse packets and
recorded GPU FIT aggregates. No F rerender, teacher/PR, optimizer trajectory
replay, GPU point-logit/ranking/mask/AP proof or scientific execution here.
"""
import hashlib
import json
import math

ARMS=('LOCAL_ERROR','CENTER_ERROR')
INIT_SEED=202610150201
ORDER_SEED=202610140160001
ADAMW=dict(lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
STATE_SHAPES={'spatial_weight':(9,256),'hidden.weight':(32,513),'hidden.bias':(32,),
              'output.weight':(3,32),'output.bias':(3,)}
ABS_LOGITS=1e-4
ABS_MEAN_CE=1e-4
BINDING_KEYS=('source','spec','GPU_admission','phase','N20_locked_protocol','locked_protocol',
    'parent_cache_report','neighbor_report','scout_module_source','base_scout_source',
    'paired_training_body_source','original_paired_training_body_source','initialization_seed',
    'order_seed','expected_N20_initialization_digest','frozen_teacher_checkpoint','frozen_PR_checkpoint',
    'N20_root_cache_qualified_result','N20_initialization_lock','N20_cache_CPU_Q')

def require(condition,message):
    if not condition:raise RuntimeError(message)
def _finite_number(value):
    return type(value) in (float,int) and math.isfinite(value)
def _tensor(torch,value,shape,dtype,name):
    require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.device.type=='cpu' and
        value.dtype==dtype and tuple(value.shape)==tuple(shape) and not value.requires_grad and bool(torch.isfinite(value).all()),
        'Exact finite CPU tensor required: '+name)
def _model(torch,state):
    require(isinstance(state,dict) and set(state)==set(STATE_SHAPES),'Five named CPU tensors required; dict/OrderedDict accepted')
    for name,shape in STATE_SHAPES.items():_tensor(torch,state[name],shape,torch.float32,name)
def _models(torch,models):
    require(type(models) is dict and set(models)==set(ARMS),'Two fixed ERROR arms required')
    for arm in ARMS:_model(torch,models[arm])
def _locked_digest(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit prelocked N20 SHA256 required')

def _factory(torch,scout_module,base_factory):
    base=base_factory.make_scout(torch,INIT_SEED);head=scout_module.create_head(torch)
    require(not tuple(head.buffers()) and len(tuple(head.parameters()))==5 and sum(p.numel() for p in head.parameters())==18851
        and list(dict(head.named_parameters()))==list(STATE_SHAPES),'Exact five-tensor18851 factory/parameter order required')
    require(set(base.state_dict())==set(STATE_SHAPES)-{'spatial_weight'},'Original513 four-tensor base factory required')
    with torch.no_grad():
        for name,value in base.state_dict().items():head.state_dict()[name].copy_(value)
    _model(torch,head.state_dict());require(bool((head.spatial_weight==0).all()),'Fresh spatial weights must start at zero')
    return head

def state_digest(torch,state):
    """Exact N20-tail sorted-header/length8/raw-FP32 digest, not tensor repr."""
    _model(torch,state);whole=hashlib.sha256();tensors=[]
    for name,tensor in sorted(state.items()):
        info=dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype))
        header=json.dumps(info,sort_keys=True,separators=(',',':')).encode('utf8')
        raw=tensor.detach().cpu().contiguous().numpy().tobytes(order='C')
        whole.update(len(header).to_bytes(8,'little'));whole.update(header);whole.update(raw)
        tensors.append(dict(info,raw_bytes_sha256=hashlib.sha256(raw).hexdigest()))
    return whole.hexdigest(),tensors

def validate_initialization_evidence(torch,scout_module,base_factory,initial_states,expected_digest):
    _locked_digest(expected_digest);require(type(initial_states) is dict and set(initial_states)==set(ARMS),'Both saved initial-state proofs required')
    proofs={}
    for arm in ARMS:
        head=_factory(torch,scout_module,base_factory);digest,tensors=state_digest(torch,head.state_dict())
        require(digest==expected_digest,'Reconstructed seed201 N20 initialization differs from lock')
        proofs[arm]=dict(combined_digest=digest,tensors=tensors)
    require(initial_states==proofs,'Saved initial digest and all tensor metadata must equal fresh seed201 reconstruction')
    return dict(initial_digest_exact=True,initial_tensor_metadata_exact=True,qualified=False)

def validate_report(torch,scout_module,base_factory,report,start,expected_digest):
    _locked_digest(expected_digest)
    require(report['schema']=='N20-paired-local-center-feature-error-training-GPU-report-v1' and
        report['status']=='N20_FULLFIT_PAIRED_FEATURE_ERROR_E8_E24_COMPLETE_PENDING_INDEPENDENT_CPU' and
        start['schema']=='N20-paired-feature-error-training-start-v1','Original pending report/START schemas')
    require(all(report[k]==start[k] for k in BINDING_KEYS) and report['expected_N20_initialization_digest']==expected_digest
        and report['initialization_seed']==INIT_SEED and report['order_seed']==ORDER_SEED,'Complete original20-field binding/seed/digest')
    require(report['images']==741 and report['natural_queries']==74100 and report['sampled_points_per_epoch']==3035136
        and report['epochs']==24 and report['paired_updates_per_arm']==17784 and report['point_exposure_per_arm']==72843264
        and report['parameters_per_arm']==18851 and report['tensors_per_arm']==5 and report['arms']==list(ARMS)
        and set(report['checkpoints'])==set(report['FIT_endpoints'])=={'8','24'} and len(report['history'])==len(report['epoch_orders'])==24,
        'Complete original fixed24/five-tensor/fullFIT report counts')
    require(start['parameters_per_arm']==18851 and start['tensors_per_arm']==5 and start['arms']==list(ARMS)
        and start['fixed_epochs']==24 and start['updates_each_arm']==17784 and start['point_exposure_each_arm']==72843264
        and start['AMP'] is False and report['CPU_endpoint_qualification_pending'] is True,'Original start/AMP/pending scope')
    for key in ('qualified','teacher_or_PR_loaded','full_native_L_loaded','DEV_AP_evaluated'):
        require(report[key] is False and start[key] is False,'Original pending false scope: '+key)
    for key in ('ROOT_actual_GPU0_verified','current_owned_terminal_self_certified','best_endpoint_selected',
                'old_cache_Source_or_protocol_modified','additional_FIT_annotations','new_primitive_claimed'):
        require(report[key] is False,'Original report does not self-certify: '+key)
    require(report['old_point_query_label_weight_order_preserved'] is True and start['initial_states']==report['initial_states'],
        'Original sample-order and initialization evidence')
    return validate_initialization_evidence(torch,scout_module,base_factory,report['initial_states'],expected_digest)

def validate_snapshot_report(torch,snapshot,report,epoch):
    require(type(epoch) is int and epoch in (8,24) and snapshot['completed_epoch']==epoch and
        snapshot['history']==report['history'][:epoch] and snapshot['epoch_orders']==report['epoch_orders'][:epoch],
        'Saved snapshot history/order must equal full report prefix')
    metadata=snapshot['metadata'];require(metadata['endpoint_epoch']==epoch and all(metadata[k]==report[k] for k in BINDING_KEYS),
        'Top-level snapshot full original binding/endpoint exact')
    for key in ('external_CPU_RNG','external_CUDA_RNG'):
        value=snapshot[key];require(isinstance(value,torch.Tensor) and value.ndim==1 and value.numel()>0,'Saved outside RNG vector')
        _tensor(torch,value,value.shape,torch.uint8,key)
    return True

def validate_snapshot_pair(torch,e8,e24):
    require(e8['completed_epoch']==8 and e24['completed_epoch']==24 and e8['history']==e24['history'][:8]
        and e8['epoch_orders']==e24['epoch_orders'][:8],'Original E8/E24 shared trajectory prefix')
    require(all(torch.equal(e8[k],e24[k]) for k in ('external_CPU_RNG','external_CUDA_RNG')),'Saved external RNG snapshots must agree')
    return True

def validate_optimizer_evidence(torch,scout_module,base_factory,optimizers,updates):
    require(type(updates) is int and updates in (5928,17784),'Fixed fullFIT E8/E24 update count')
    head=_factory(torch,scout_module,base_factory);names=list(dict(head.named_parameters()))
    reference=torch.optim.AdamW(head.parameters(),**ADAMW).state_dict()['param_groups']
    require(type(optimizers) is dict and set(optimizers)==set(ARMS),'Both full optimizer states')
    for arm in ARMS:
        optimizer=optimizers[arm];require(optimizer['param_groups']==reference,'All original AdamW flags/hyperparameters/ID order')
        ids=optimizer['param_groups'][0]['params']
        require(len(ids)==len(set(ids))==5 and all(type(i) is int for i in ids) and set(optimizer['state'])==set(ids),'Five unique complete AdamW states')
        for ident,name in zip(ids,names):
            value=optimizer['state'][ident];require(set(value)=={'step','exp_avg','exp_avg_sq'},'Plain non-amsgrad moments')
            _tensor(torch,value['step'],(),torch.float32,arm+'.step')
            require(float(value['step'])==updates,'Integral-valued FP32 scalar step must equal5928/17784 exactly, no int truncation')
            for field in ('exp_avg','exp_avg_sq'):_tensor(torch,value[field],STATE_SHAPES[name],torch.float32,arm+'.'+name+'.'+field)
            require(bool((value['exp_avg_sq']>=0).all()),'Nonnegative second moments')
    return True

def _cache_counts(rows):
    require(type(rows) is list and len(rows)==741 and all(type(row['dataset_index']) is int and row['dataset_index']==index and
        type(row['points']) is int and row['points']==4096 for index,row in enumerate(rows)),
        'Only complete canonical FIT741 with4096 original slots/image qualifies, never tiny/empty/partial scope')
    return 741,741,3035136

def validate_state_evidence(torch,scout_module,base_factory,snapshot,cache_rows,epoch,expected_initialization_digest=None):
    require(type(epoch) is int and epoch in (8,24),'Only fixed E8/E24 endpoints')
    _locked_digest(expected_initialization_digest)
    initial=_factory(torch,scout_module,base_factory)
    require(state_digest(torch,initial.state_dict())[0]==expected_initialization_digest and snapshot['expected_first_initialization_digest']==expected_initialization_digest,
        'New N20 initialization lock exact, no N17 digest reuse')
    require(snapshot['qualification_pending'] is True and snapshot['scientific_source_qualified'] is False and snapshot['actual_owned_terminal_self_certified'] is False,
        'Original snapshot remains pending; no terminal self-qualification')
    _models(torch,snapshot['models']);count,nonempty,points=_cache_counts(cache_rows);updates=epoch*741
    require(type(snapshot['completed_epoch']) is int and snapshot['completed_epoch']==epoch and snapshot['initialization_seed']==INIT_SEED and
        snapshot['order_seed']==ORDER_SEED and snapshot['fixed_epochs']==24 and snapshot['AdamW']==ADAMW and
        snapshot['parameters_per_arm']==18851 and snapshot['allowed_fixed_checkpoints']==[8,24] and
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
    validate_optimizer_evidence(torch,scout_module,base_factory,snapshot['optimizers'],updates)
    # Trained spatial weights may stay zero or change; neither arm equality nor divergence is forced.
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
    require(type(loaded) is dict and set(loaded)=={'packet','neighbor_packet'},'Qualified original packet plus sparse neighbor packet only')
    packet=loaded['packet'];samples=packet['samples'];n=len(samples['query_ids']);U=len(packet['sampled_F_unique'])
    _tensor(torch,packet['sampled_F_unique'],(U,256),torch.float32,'sampled_F_unique')
    _tensor(torch,packet['native_e0'],(1,100,256),torch.float32,'native_e0')
    _tensor(torch,packet['sample_to_unique'],(n,),torch.int64,'sample_to_unique')
    for key in ('query_ids','flat_indices'):_tensor(torch,samples[key],(n,),torch.int64,key)
    require(n>0 and bool(((samples['query_ids']>=0)&(samples['query_ids']<100)).all()) and
        bool(((packet['sample_to_unique']>=0)&(packet['sample_to_unique']<U)).all()) and bool((samples['flat_indices']>=0).all()),'Valid nonempty original query/slot indices')
    for key in ('native_L','importance_weights'):_tensor(torch,samples[key],(n,),torch.float32,key)
    require(bool((samples['importance_weights']>0).all()),'Original importance weights positive')
    for key in ('utility_labels','error_labels'):
        _tensor(torch,samples[key],(n,),torch.int64,key);require(bool(((samples[key]>=0)&(samples[key]<=2)).all()),'Original label classes0..2')
    neighbor=loaded['neighbor_packet'];require(type(neighbor) is dict and set(neighbor)=={'extra_F','neighbor_to_combined'},'Explicit two-key qualified sparse neighbor input')
    extra=neighbor['extra_F'];require(isinstance(extra,torch.Tensor) and extra.ndim==2,'Extra F matrix')
    E=len(extra);mapping=neighbor['neighbor_to_combined']
    _tensor(torch,extra,(E,256),torch.float32,'extra_F');_tensor(torch,mapping,(U,9),torch.int64,'neighbor_to_combined')
    require(bool(((mapping>=0)&(mapping<U+E)).all()) and torch.equal(mapping[:,4],torch.arange(U,dtype=torch.int64)),
        'Exact sparse mapping bounds and original center-column identity')
    F=packet['sampled_F_unique'][packet['sample_to_unique']]
    combined=torch.cat((packet['sampled_F_unique'],extra),0)
    neighbors={'LOCAL_ERROR':combined[mapping][packet['sample_to_unique']], 'CENTER_ERROR':F[:,None,:].repeat(1,9,1)}
    return F,packet['native_e0'][0,samples['query_ids']],samples['native_L'],neighbors,samples,n

def independent_logits(torch,state,F,e,L,neighbors):
    """Explicit spatial sum, biased-variance LN, matmul/bias and ReLU; no NN calls."""
    _model(torch,state);n=len(L)
    for name,value,shape in (('F',F,(n,256)),('e',e,(n,256)),('rawL',L,(n,)),('neighbors',neighbors,(n,9,256))):
        _tensor(torch,value,shape,torch.float32,name)
    adapted=F+(neighbors*state['spatial_weight'][None,:,:]).sum(1)
    def normalize(value):
        centered=value-value.mean(1,keepdim=True);variance=(centered*centered).mean(1,keepdim=True)
        return centered*torch.rsqrt(variance+1e-5)
    descriptor=torch.cat((normalize(adapted),normalize(e),(L/8).unsqueeze(1)),1)
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

def validate_endpoint(torch,scout_module,base_factory,snapshot,GPU_endpoint,loader_callback,cache_rows,epoch,observer=None,expected_initialization_digest=None):
    counts=validate_state_evidence(torch,scout_module,base_factory,snapshot,cache_rows,epoch,expected_initialization_digest)
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
        qualified=False,ROOT_actual_CPU0_verified=False,qualified_checks_only=True,initial_digest_checked_by_body=True,GPU_pointwise_logits_qualified=False,
        selected_indices_qualified=False,mask_RLE_qualified=False,AP_evaluated=False,optimizer_trajectory_replayed=False,
        trace_scope='saved paired_image_update substream only; validate_trace called separately',
        scope='Exact trained states/moments/steps/orders/counts plus full same-input CPU two formulas and recorded GPU aggregate meanCE; ROOT Source/terminal qualification pending; initialization reconstructed against supplied lock; same GPU-input F reused, never rerendered')
