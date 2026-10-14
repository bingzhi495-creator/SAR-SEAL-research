"""UNBOUND_DRAFT: N17 GT-independent fixed-DEV numerical body, no runner/IO/owner gate.
ROOT supplies qualified exact nativeL/e/unionF, frozen endpoint scouts and original PR.
Natural100 actual CUDA path; smaller coldCPU grids only with synthetic_only=True.
Outputs/callbacks are actual device results, never Source/terminal/numeric/AP qualification.
"""
from time import perf_counter
ARMS=('LOCAL_ERROR','CENTER_ERROR')
EPOCHS=(24,8)
MODES=('atmost','exact')
POINT_CHUNK=2048
ABSOLUTE_TOLERANCE=1e-4
STATE_SHAPES={'hidden.weight':(32,521),'hidden.bias':(32,),'output.weight':(3,32),'output.bias':(3,)}

def require(condition,message):
    if not condition:raise ValueError(message)
def _union_numpy(torch,np,value):
    if isinstance(value,torch.Tensor):
        require(value.device.type=='cpu' and value.layout==torch.strided and value.dtype==torch.int64 and value.ndim==1,'CPU dense INT64 union required')
        value=value.numpy()
    require(isinstance(value,np.ndarray) and value.dtype==np.int64 and value.ndim==1 and
        (len(value)==0 or bool(np.all(value[1:]>value[:-1]))),'Strict unique ascending INT64 union mapping required')
    return value

def gather_F(torch,np,union_flat_indices,F_unique,flat_indices):
    """Exact physical union lookup; duplicate requested slots preserve their original order."""
    union=_union_numpy(torch,np,union_flat_indices)
    require(isinstance(F_unique,torch.Tensor) and F_unique.layout==torch.strided and F_unique.dtype==torch.float32 and
        tuple(F_unique.shape)==(len(union),256) and isinstance(flat_indices,np.ndarray) and flat_indices.dtype==np.int64 and
        flat_indices.ndim==1,'FP32 unionF and original INT64 flat slots required')
    positions=np.searchsorted(union,flat_indices)
    require(len(flat_indices)==0 or (bool(np.all(positions<len(union))) and bool(np.array_equal(union[positions],flat_indices))),
        'Every actual scout/PR point must have exact captured F coverage; no zero-filled full F')
    rows=torch.from_numpy(positions.astype(np.int64,copy=True)).to(F_unique.device)
    return F_unique.index_select(0,rows)

def _cost(legacy):
    value=legacy._cost();value.update(extra8_gather_calls=0,extra8_gather_points=0,extra8_gather_seconds=0.)
    return value

def _run_points(torch,np,legacy,kernel,L,e,union,F_unique,q,indices,model,kind,cost,callback,check,
        control,epoch,mode,stage):
    n=len(indices);device=L.device
    require(indices.dtype==np.int64 and indices.ndim==1 and (n==0 or bool(np.all(indices[1:]>indices[:-1]))),
        'Canonical distinct ascending actual point supports required')
    output=dict(native_L=np.empty(n,dtype=np.float32))
    if kind=='scout':
        actual=torch.empty((n,3),dtype=torch.float32,device=device);output['logits']=np.empty((n,3),dtype=np.float32)
        output['actual_GPU_logits']=actual
    else:
        output.update(residual=np.empty(n,dtype=np.float32),combined=np.empty(n,dtype=np.float32),
            actual_GPU_residual=torch.empty(n,dtype=torch.float32,device=device),actual_GPU_combined=torch.empty(n,dtype=torch.float32,device=device))
    for first in range(0,n,POINT_CHUNK):
        check();stop=min(first+POINT_CHUNK,n);flat=indices[first:stop];index=torch.from_numpy(flat.copy()).to(device)
        legacy._sync(torch,device);tick=perf_counter()
        F=gather_F(torch,np,union,F_unique,flat);embedding=e[q].unsqueeze(0).expand(stop-first,256)
        native=L[q,index//L.shape[2],index%L.shape[2]]
        legacy._sync(torch,device);cost['gather_calls']+=1;cost['gather_points']+=stop-first;cost['gather_seconds']+=perf_counter()-tick
        if kind=='scout':
            tick=perf_counter()
            if control=='LOCAL_ERROR':
                # This slice remains the full original grid of natural query q; local index0 is its only query.
                extra=kernel.neighbor_features(torch,L[q:q+1],torch.zeros_like(index),index)
            else:extra=(native.clamp(-8,8)/8).unsqueeze(1).repeat(1,8)
            legacy._finite_tensor(torch,extra,(stop-first,8),device,'Actual eight-dimensional input')
            legacy._sync(torch,device);cost['extra8_gather_calls']+=1;cost['extra8_gather_points']+=stop-first
            cost['extra8_gather_seconds']+=perf_counter()-tick;tick=perf_counter()
            result=model(F,embedding,native,extra)
            legacy._finite_tensor(torch,result,(stop-first,3),device,'Actual four-input scout logits')
            legacy._sync(torch,device);cost['scout_calls']+=1;cost['scout_points']+=stop-first;cost['scout_seconds']+=perf_counter()-tick
            actual[first:stop].copy_(result)
        else:
            tick=perf_counter();result=model(F,embedding,native,native.new_zeros((stop-first,49)),arm='PR')
            require(isinstance(result,dict) and 'residual' in result,'Original fixed PR residual ABI required')
            residual=result['residual'];combined=native+residual
            legacy._finite_tensor(torch,residual,(stop-first,),device,'Actual original PR residual')
            legacy._finite_tensor(torch,combined,(stop-first,),device,'Actual device FP32 L+u')
            legacy._sync(torch,device);cost['PR_calls']+=1;cost['PR_points']+=stop-first;cost['PR_seconds']+=perf_counter()-tick
            output['actual_GPU_residual'][first:stop].copy_(residual);output['actual_GPU_combined'][first:stop].copy_(combined)
        tick=perf_counter();output['native_L'][first:stop]=native.detach().cpu().numpy()
        if kind=='scout':output['logits'][first:stop]=result.detach().cpu().numpy()
        else:
            output['residual'][first:stop]=residual.detach().cpu().numpy();output['combined'][first:stop]=combined.detach().cpu().numpy()
        cost['readback_seconds']+=perf_counter()-tick
        if callback is not None:
            tick=perf_counter();event=dict(query_id=q,control=control,scout_epoch=epoch,evalmode=mode,stage=stage,
                point_chunk=2048,chunk_first=first,flat_indices=torch.from_numpy(flat.copy()),sampled_F=F.detach(),e0=e[q].detach(),
                native_L=native.detach(),actual_device_type=device.type,synthetic_only=(device.type=='cpu'),
                actual_input_scope='ROOT_supplied_actual_nativeL_e0_unionF',actual_Source_REF_qualified_here=False)
            if kind=='scout':event.update(extra8=extra.detach(),actual_GPU_logits=result.detach(),scout_logits=result.detach())
            else:event.update(GPU_raw_residual=residual.detach(),GPU_combined=combined.detach())
            callback(event);cost['callback_calls']+=1;cost['callback_seconds']+=perf_counter()-tick;del event
        if kind=='scout':del extra
        else:del residual,combined
        del index,F,embedding,native,result
    return output

def evaluate_all100(torch,np,ndimage,helper,legacy_evalbody,neighbor_module,native_L,e0,union_flat_indices,F_unique,models,frozen_PR,
        *,point_callback=None,check_callback=None,synthetic_only=False):
    """Yield eight fixed policy profiles/query from exact union inputs; ROOT owns all qualification.

    Each of four actual scouts runs C once and shares its saved actual logits with
    atmost/exact. Model projections/forwards are separate. Original PR is actually
    recomputed on full C and each selected support, never blindly reuses old u.
    Callbacks see actual-device detached chunk tensors and must store/copy before
    retaining them. Consumers must release each query; body never reads native/R refs.
    """
    require(type(synthetic_only) is bool and (point_callback is None or callable(point_callback)) and
        (check_callback is None or callable(check_callback)),'Explicit synthetic mode/callable ROOT hooks required')
    check=(lambda:None) if check_callback is None else check_callback
    require(isinstance(native_L,torch.Tensor) and native_L.ndim==3 and min(native_L.shape)>0,'Original native_L[Q,H,W] required')
    device=native_L.device;Q,H,W=native_L.shape
    require((device.type=='cuda' and Q==100 and not synthetic_only) or (device.type=='cpu' and synthetic_only),
        'Actual path requires CUDA natural100; reduced CPU scope is SYNTHETIC_ONLY')
    require(not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled(),'FP32 only, AMP forbidden')
    legacy=legacy_evalbody
    require(legacy.POINT_CHUNK==2048 and legacy.ABSOLUTE_TOLERANCE==1e-4,'Unchanged original chunk/absolute tolerance recipe required')
    legacy._finite_tensor(torch,native_L,(Q,H,W),device,'Original nativeL grid')
    legacy._finite_tensor(torch,e0,(Q,256),device,'Original query embeddings')
    union=_union_numpy(torch,np,union_flat_indices)
    require(len(union)==0 or bool(((union>=0)&(union<H*W)).all()),'Captured union indices in actual original grid')
    legacy._finite_tensor(torch,F_unique,(len(union),256),device,'Actual captured unique center F')
    require(type(models) is dict and set(models)=={(epoch,arm) for epoch in EPOCHS for arm in ARMS},'All fixed E24/E8 two-arm states required')
    for key,model in list(models.items())+[('fixed_PR',frozen_PR)]:
        require(not model.training,'All supplied models frozen eval; body never changes state or mode')
        for parameter in model.parameters():
            require(parameter.device==device and parameter.dtype==torch.float32 and not parameter.requires_grad and parameter.grad is None and
                bool(torch.isfinite(parameter).all()),'All supplied parameters frozen finite FP32 on input device')
        if key!='fixed_PR':
            state=model.state_dict();require(set(state)==set(STATE_SHAPES) and not tuple(model.buffers()) and
                all(tuple(state[name].shape)==shape for name,shape in STATE_SHAPES.items()),'Four trained521/16803 N17 scout tensors required')
    # Validate physical feature coverage independently of all new scout outputs, with no GT.
    merged=np.zeros((H,W),dtype=np.bool_);domains=[]
    for q in range(Q):
        check();L_cpu=native_L[q].detach().cpu().clone().contiguous().numpy()
        domain=legacy.domains_from_native(np,ndimage,L_cpu);merged|=domain[2];domains.append((L_cpu,domain))
    require(bool(np.array_equal(union,np.flatnonzero(merged).astype(np.int64))),'Captured F union must equal complete natural-query C union exactly')
    for q in range(Q):
        check();start=perf_counter();L_cpu,(P,Omega,C,C_indices,K)=domains[q];active=C_indices if K else np.empty(0,dtype=np.int64)
        common_cost=_cost(legacy);profiles={};model_costs={};policy_costs={}
        with torch.no_grad():
            common=_run_points(torch,np,legacy,neighbor_module,native_L,e0,union,F_unique,q,active,frozen_PR,'PR',common_cost,
                point_callback,check,'COMMON_C_REFERENCE',None,'shared','full_C_PR')
            for epoch in EPOCHS:
                for arm in ARMS:
                    model_cost=_cost(legacy)
                    points=_run_points(torch,np,legacy,neighbor_module,native_L,e0,union,F_unique,q,active,models[(epoch,arm)],'scout',model_cost,
                        point_callback,check,arm,epoch,'shared-atmost-exact','scout_C_logits')
                    if K:
                        tick=perf_counter();scores=helper.error_score(points['logits'])
                        require(scores.dtype==np.float64 and scores.shape==(len(C_indices),) and bool(np.isfinite(scores).all()),'Canonical CPU FP64 ERROR score required')
                        model_cost['score_calls']+=1;model_cost['score_seconds']+=perf_counter()-tick
                    else:scores=np.empty(0,dtype=np.float64)
                    grid=np.zeros((H,W),dtype=np.float64)
                    if K:grid.ravel()[C_indices]=scores
                    model_costs['E%d_%s'%(epoch,arm)]=model_cost
                    for mode in MODES:
                        check();cost=_cost(legacy);tick=perf_counter();selected=helper.choose_candidates(L_cpu,C,grid,K,mode=mode)
                        cost['sorting_calls']+=1;cost['sorting_seconds']+=perf_counter()-tick;selected_flat=np.flatnonzero(selected).astype(np.int64)
                        require(len(selected_flat)<=K and not bool((selected&~C).any()) and (mode!='exact' or len(selected_flat)==K),'Original physical K support fixed')
                        subset=_run_points(torch,np,legacy,neighbor_module,native_L,e0,union,F_unique,q,selected_flat,frozen_PR,'PR',cost,
                            point_callback,check,arm,epoch,mode,'selected_PR')
                        compatibility=legacy._qualify_subset(np,q,arm,selected_flat,subset,common,active)
                        tick=perf_counter();mask=P.copy();mask.ravel()[selected_flat]=subset['combined']>0
                        require(bool(np.array_equal(mask[~selected],P[~selected])),'Outside selected support remains native exactly')
                        cost['mask_seconds']+=perf_counter()-tick;name='E%d_%s_%s'%(epoch,arm,mode);policy_costs[name]=cost
                        profiles[name]=dict(epoch=epoch,arm=arm,mode=mode,main_result=(epoch==24 and mode=='atmost'),mask=mask,
                            selected_flat_indices=selected_flat,actual_GPU_C_logits=points['actual_GPU_logits'],canonical_CPU_FP64_C_scores=scores,
                            native_selected_L=subset['native_L'],actual_raw_residual=subset['residual'],actual_combined=subset['combined'],
                            actual_GPU_selected_residual=subset['actual_GPU_residual'],actual_GPU_selected_combined=subset['actual_GPU_combined'],
                            PR_compatibility=compatibility,model_cost_key='E%d_%s'%(epoch,arm),policy_cost=cost)
                        del subset
                    del points,grid
        legacy._sync(torch,device)
        yield dict(query_id=q,natural_queries=Q,all100_actual_path=(not synthetic_only),synthetic_only=synthetic_only,
            native_L=L_cpu,native_mask=P,Omega=Omega,C=C,C_flat_indices=C_indices,K=K,profiles=profiles,
            common_C=common,common_C_PR_cost=common_cost,per_model_costs=model_costs,per_policy_costs=policy_costs,
            shared_diagnostic_seconds=perf_counter()-start,actual_GPU_logits_shared_atmost_exact_same_model=True,
            cross_model_forward_or_projection_shared=False,native_R_refs_loaded=False,reference_reuse_verified=False,
            numerical_qualification_conferred=False,ROOT_Source_or_terminal_qualified=False,AP_evaluated=False,
            original_scores_classes_S0_export_order_not_consumed_or_modified=True,main_epoch=24,curve_epoch=8,
            standalone_total_time_measured=False,inclusive_costs_nonadditive=True)
