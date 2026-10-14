"""UNBOUND_DRAFT: N17 selected-numeric CPU pure core, no bundle schema/IO/Source gate.
ROOT binds actual captures/carrier/model/formula/export provenance and owned terminal.
Ranking uses ONLY actual saved GPU FP32 logits through frozen canonical FP64 helper.
PR ignores e/49patch computationally; original_head wrapper must pass valid zeros49 ABI.
"""
ARMS=('LOCAL_ERROR','CENTER_ERROR')
EPOCHS=(24,8)
MODES=('atmost','exact')
POINT_CHUNK=2048
ATOL=1e-4
FORMULA_NAMES=('original_head','independent_PR')

def need(ok,message):
    if not ok:raise ValueError(message)
def _same(torch,a,b):
    return a.shape==b.shape and a.dtype==b.dtype==torch.float32 and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
def _numpy_same(np,a,b):
    return a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32))
def _error(torch,left,right):
    return float((left.double()-right.double()).abs().max()) if left.numel() else 0.

def _scout_model(torch,endpoint,head,state):
    endpoint._model(torch,state)
    need(not head.training and not tuple(head.buffers()) and len(tuple(head.parameters()))==4 and
        sum(value.numel() for value in head.parameters())==16803,'Frozen four-tensor16803 CPU N17 model required')
    for value in head.parameters():
        need(value.device.type=='cpu' and value.dtype==torch.float32 and not value.requires_grad and value.grad is None and bool(torch.isfinite(value).all()),
            'Qualified CPU scout parameters must be finite FP32, frozen and gradient-free')
    need(set(head.state_dict())==set(state) and all(_same(torch,value,state[name]) for name,value in head.state_dict().items()),
        'CPU factory must load exact supplied qualified trained state, no reinitialization')

def _prove_scout_chunks(torch,np,legacy,neighbor,endpoint,head,state,query_id,arm,grid,packet,forward_indices,common,iterator,check):
    gpu_logits=[];cursor=0
    stats=dict(chunk_calls=0,points=0,factory_vs_actual_GPU_max_absolute_error=0.,independent_vs_actual_GPU_max_absolute_error=0.,
        CPU_two_formula_max_absolute_error=0.,actual_extra8_native_grid_byte_exact=True)
    for item in iterator:
        check();need(type(item) is tuple and len(item)==6,'Scout hook must yield six explicit CPU arrays, not physical refs/member schema')
        flat,F,e,L,extra,actual=item;expected=forward_indices[cursor:cursor+2048];n=len(expected)
        need(n>0,'No extra or empty scout chunk beyond canonical actual support')
        legacy._tensor(torch,flat,torch.int64,(n,),'Scout actual flat indices')
        need(bool(np.array_equal(flat.numpy(),expected)),'All C scout chunks must be exact ascending2048 support in original order')
        for name,value,shape in (('F',F,(n,256)),('e0',e,(256,)),('rawL',L,(n,)),('extra8',extra,(n,8)),('actualGPUlogits',actual,(n,3))):
            legacy._tensor(torch,value,torch.float32,shape,name)
        need(_same(torch,e,packet['e0']) and _same(torch,L,packet['native_L'].reshape(-1)[flat]),'Captured query e0 and raw center L byte exact')
        if arm=='LOCAL_ERROR':
            expected_extra=neighbor.neighbor_features_numpy(np,grid[query_id:query_id+1].numpy(),np.zeros(n,dtype=np.int64),flat.numpy())
        else:expected_extra=neighbor.center_control_features_numpy(np,grid[query_id:query_id+1].numpy(),np.zeros(n,dtype=np.int64),flat.numpy())
        need(_numpy_same(np,extra.numpy(),expected_extra),'Actual GPU extra8 must be byte exact same-query original-neighbor/center coordinate equation')
        # The original kernel verifies observed F equality without concatenating full-C F256.
        legacy._same_common_inputs(torch,np,[dict(flat_indices=flat,sampled_F=F)],common,forward_indices)
        with torch.no_grad():
            factory=head(F.clone(),e[None].expand(n,256).clone(),L.clone(),extra.clone())
            independent=endpoint.independent_logits(torch,state,F.clone(),e[None].expand(n,256).clone(),L.clone(),extra.clone())
        legacy._tensor(torch,factory,torch.float32,(n,3),'CPU factory logits')
        legacy._tensor(torch,independent,torch.float32,(n,3),'CPU explicit logits')
        first,second=_error(torch,factory,actual),_error(torch,independent,actual)
        need(first<=ATOL and second<=ATOL,'Both same-actual-input CPU scout formulas vs actualGPU absolute1e-4/rtol0 required')
        stats['factory_vs_actual_GPU_max_absolute_error']=max(stats['factory_vs_actual_GPU_max_absolute_error'],first)
        stats['independent_vs_actual_GPU_max_absolute_error']=max(stats['independent_vs_actual_GPU_max_absolute_error'],second)
        stats['CPU_two_formula_max_absolute_error']=max(stats['CPU_two_formula_max_absolute_error'],_error(torch,factory,independent))
        stats['chunk_calls']+=1;stats['points']+=n;gpu_logits.append(actual.numpy().copy());cursor+=n
        del factory,independent,item
    need(cursor==len(forward_indices) and stats['chunk_calls']==(len(forward_indices)+2047)//2048,'Complete actual scout canonical chunk coverage required')
    return (np.concatenate(gpu_logits,axis=0) if gpu_logits else np.empty((0,3),dtype=np.float32)),stats

def proof_query(torch,np,cv2,helper,legacy_selector_proof,neighbor_module,endpoint_body,query_id,native_L_grid,e0,C_flat_indices,K,
        heads,states,common_PR_chunks,iter_scout_chunks,read_profile,formulas,rle_encoder,*,check_callback=None):
    """Prove one natural query via arrays/hooks. No unknown new storage schema is inferred.

    iter_scout_chunks(epoch,arm) yields(flat,F,e0,L,extra8,actualGPUlogits) CPU tuples.
    read_profile(epoch,arm,mode) returns(actual_selected,selectedPRchunks,actualmask,actualRLE).
    PR chunks retain six known old numerical fields; formulas(F,eExpanded,L)->(u,L+u),
    original_head uses zeros49/armPR, and independent_PR uses F/L, as ROOT-closed ABI.
    rle_encoder(maskBOOLnumpy)->original canonical RLE; export classes/S0/order are external.
    """
    legacy=legacy_selector_proof;check=(lambda:None) if check_callback is None else check_callback
    need(type(query_id) is int and 0<=query_id<100 and isinstance(native_L_grid,torch.Tensor) and native_L_grid.ndim==3 and
        native_L_grid.shape[0]==100 and min(native_L_grid.shape[1:])>0 and native_L_grid.device.type=='cpu' and
        native_L_grid.layout==torch.strided and native_L_grid.dtype==torch.float32 and not native_L_grid.requires_grad,
        'Full original natural100 CPU native-grid namespace required')
    need(legacy.POINT_CHUNK==2048 and legacy.ATOL==1e-4 and tuple(legacy.FORMULA_NAMES)==FORMULA_NAMES,'Existing locked PR kernel recipe unchanged')
    need(callable(iter_scout_chunks) and callable(read_profile) and callable(rle_encoder) and
        (check_callback is None or callable(check_callback)),'ROOT must inject array loaders/export/deadline hooks')
    need(type(formulas) is dict and set(formulas)==set(FORMULA_NAMES) and all(callable(formulas[name]) for name in FORMULA_NAMES),
        'Two distinct ROOT-closed PR formula callbacks required; names alone do not certify provenance')
    expected={(epoch,arm) for epoch in EPOCHS for arm in ARMS}
    need(type(heads) is dict and type(states) is dict and set(heads)==set(states)==expected,'All qualified E24/E8 two ERROR states required')
    for key in expected:_scout_model(torch,endpoint_body,heads[key],states[key])
    L=native_L_grid[query_id];legacy._tensor(torch,L,torch.float32,L.shape,'Actual query native L grid')
    legacy._tensor(torch,e0,torch.float32,(256,),'Actual query e0')
    P,C,indices,actualK=legacy._domains(np,cv2,L.numpy())
    need(type(K) is int and K==actualK,'Budget K exact physical absL<=2 count')
    legacy._tensor(torch,C_flat_indices,torch.int64,(len(indices),),'Recorded actual C flat indices')
    need(bool(np.array_equal(C_flat_indices.numpy(),indices)),'Independent nativeL CV2 radius8 C must match exact actual C support')
    packet=dict(native_L=L,e0=e0);active=indices if K else np.empty(0,dtype=np.int64)
    check();common=legacy._prove_PR(torch,np,common_PR_chunks,active,packet,formulas,'N17/common_C')
    profiles={};scout_stats={}
    for epoch in EPOCHS:
        for arm in ARMS:
            check();key=(epoch,arm)
            gpu_logits,stats=_prove_scout_chunks(torch,np,legacy,neighbor_module,endpoint_body,heads[key],states[key],query_id,arm,
                native_L_grid,packet,active,common_PR_chunks,iter_scout_chunks(epoch,arm),check)
            scout_stats['E%d_%s'%(epoch,arm)]=stats
            if K:
                scores=helper.error_score(gpu_logits)
                need(scores.dtype==np.float64 and scores.shape==(len(indices),) and bool(np.isfinite(scores).all()),'Actual GPU logits -> canonical stable FP64 ERROR score required')
            else:scores=np.empty(0,dtype=np.float64)
            score_grid=np.zeros(L.shape,dtype=np.float64)
            if K:score_grid.ravel()[indices]=scores
            for mode in MODES:
                check();selected=helper.choose_candidates(L.numpy(),C,score_grid,K,mode=mode);chosen=np.flatnonzero(selected).astype(np.int64)
                record=read_profile(epoch,arm,mode)
                need(type(record) is tuple and len(record)==4,'Profile hook must return four explicit arrays/RLE, no new bundle schema assumed')
                saved,blocks,actual_mask,actual_RLE=record
                legacy._tensor(torch,saved,torch.int64,(len(chosen),),'Actual selected indices')
                need(bool(np.array_equal(saved.numpy(),chosen)),'Selected indices EXACT from actualGPU logits, no CPU approximation/rank tolerance')
                legacy._tensor(torch,actual_mask,torch.bool,L.shape,'Actual complete mask')
                subset=legacy._prove_PR(torch,np,blocks,chosen,packet,formulas,'N17/E%d/%s/%s/selected'%(epoch,arm,mode))
                legacy._same_common_inputs(torch,np,blocks,common_PR_chunks,active)
                positions=np.searchsorted(indices,chosen)
                legacy._comparison(torch,torch.from_numpy(subset['residual']),torch.from_numpy(subset['combined']),
                    torch.from_numpy(common['residual'][positions]),torch.from_numpy(common['combined'][positions]),'Actual selected vs actual common C')
                mask=P.copy();mask.ravel()[chosen]=subset['combined']>0
                need(bool(np.array_equal(mask,actual_mask.numpy())) and bool(np.array_equal(actual_mask.numpy()[~selected],P[~selected])),
                    'Complete mask/native outside selected support exact, based on actualGPU combined strict>0')
                encoded=rle_encoder(mask)
                need(type(encoded) is dict and type(actual_RLE) is dict and encoded==actual_RLE,'Original canonical mask RLE must match exact actual export')
                name='E%d_%s_%s'%(epoch,arm,mode)
                profiles[name]=dict(epoch=epoch,arm=arm,mode=mode,main_result=(epoch==24 and mode=='atmost'),selected_points=len(chosen),
                    selected_indices_exact=True,full_mask_exact=True,outside_selected_native_exact=True,RLE_exact=True,
                    reconstructed_mask=torch.from_numpy(mask),selected_PR_formula_stats=subset['formula_stats'],PR_strict_sign_exact=True,
                    ranking_inputs='supplied_actual_GPU_FP32_logits_then_frozen_canonical_NumPy_FP64_ERROR_score',
                    CPU_scout_outputs_used_for_ranking=False,CPU_approximate_PR_used_for_ranking=False)
                del subset,record,blocks,mask
            del gpu_logits,score_grid,scores
    return dict(query_id=query_id,K=K,C_points=len(indices),profiles=profiles,scout_numeric=scout_stats,
        common_C_PR=dict(formula_stats=common['formula_stats'],formula_chunk_calls=common['formula_chunk_calls'],strict_sign_exact=True),
        all_selected_indices_exact=True,all_full_masks_and_RLE_exact=True,actual_extra8_native_grid_byte_exact=True,
        CPU_scout_outputs_used_for_ranking=False,atol=1e-4,rtol=0.,native_R_reference_reuse_proven_here=False,
        original_query_ids_classes_S0_export_order_require_external_binding=True,AP_evaluated=False,
        actual_capture_provenance_proven_here=False,ROOT_scientific_qualification_conferred=False,
        scope='one actual supplied query: nativegrid/extra8, CPU two scout formulas vs GPU, exact GPU-logit ranking and two PR formulas/masks/RLE; ROOT Source/state/export/all100/actual0 external')
