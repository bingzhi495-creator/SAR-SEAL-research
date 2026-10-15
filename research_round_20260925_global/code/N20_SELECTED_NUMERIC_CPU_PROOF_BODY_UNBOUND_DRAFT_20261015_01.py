"""UNBOUND_DRAFT: N20 true-F9 selected-numeric CPU pure core, no bundle schema/IO/Source gate.
ROOT binds actual captures/carrier/model/formula/export provenance and owned terminal.
Ranking uses ONLY saved canonical FP64 scores, checked exactly against actual GPU FP32 logits.
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
    need(not head.training and not tuple(head.buffers()) and len(tuple(head.parameters()))==5 and
        sum(value.numel() for value in head.parameters())==18851,'Frozen five-tensor18851 CPU N20 model required')
    for value in head.parameters():
        need(value.device.type=='cpu' and value.dtype==torch.float32 and not value.requires_grad and value.grad is None and bool(torch.isfinite(value).all()),
            'Qualified CPU scout parameters must be finite FP32, frozen and gradient-free')
    need(set(head.state_dict())==set(state) and all(_same(torch,value,state[name]) for name,value in head.state_dict().items()),
        'CPU factory must load exact supplied qualified trained state, no reinitialization')


def feature_neighbors(torch,np,context,flat,H,W,arm):
    """Independent CPU coordinate equation and sparse gather; no full-F construction."""
    need(type(context) is dict and set(context)=={'union_flat_indices','F_unique','extra_flat_indices','extra_F','neighbor_to_combined'} and
        type(H) is int and type(W) is int and min(H,W)>0 and arm in ARMS,'Explicit actual sparse F9 context')
    union,extra=context['union_flat_indices'],context['extra_flat_indices'];F=context['F_unique'];E=context['extra_F'];mapping=context['neighbor_to_combined']
    n,U,X=len(flat),len(union),len(extra)
    for value,dtype,shape in ((union,torch.int64,(U,)),(extra,torch.int64,(X,)),(flat,torch.int64,(n,)),
                             (F,torch.float32,(U,256)),(E,torch.float32,(X,256)),(mapping,torch.int64,(U,9))):
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.layout==torch.strided and value.dtype==dtype and
            tuple(value.shape)==shape and not value.requires_grad,'Exact CPU sparse F9 tensor shape/dtype')
    positions=np.searchsorted(union.numpy(),flat.numpy())
    need(n==0 or (bool((positions<U).all()) and np.array_equal(union.numpy()[positions],flat.numpy())),'Every C point covered by original actual F union')
    rows=mapping[torch.from_numpy(positions.astype(np.int64))]
    need(bool(((rows>=0)&(rows<U+X)).all()) and torch.equal(rows[:,4],torch.from_numpy(positions.astype(np.int64))),
        'Sparse F9 mapping bounds and exact center column4')
    source=rows.reshape(-1);old=source<U
    coords=torch.empty(source.shape,dtype=torch.int64);values=torch.empty((len(source),256),dtype=torch.float32)
    coords[old]=union[source[old]];values[old]=F[source[old]]
    coords[~old]=extra[source[~old]-U];values[~old]=E[source[~old]-U]
    y,x=flat.numpy()//W,flat.numpy()%W
    expected=np.stack([np.clip(y+dy,0,H-1)*W+np.clip(x+dx,0,W-1) for dy in (-1,0,1) for dx in (-1,0,1)],axis=1)
    need(np.array_equal(coords.reshape(n,9).numpy(),expected),'Actual original-grid nine-neighbor replicate coordinates exact; no row wrap')
    centers=F[torch.from_numpy(positions.astype(np.int64))]
    local=values.reshape(n,9,256)
    need(bool(torch.isfinite(centers).all()) and bool(torch.isfinite(local).all()) and _same(torch,local[:,4],centers),
        'Finite actual F values and byteexact original centers')
    return centers,(local if arm=='LOCAL_ERROR' else centers[:,None,:].repeat(1,9,1))

def _prove_scout_chunks(torch,np,legacy,endpoint,head,state,query_id,arm,grid,packet,feature_context,forward_indices,common,iterator,check):
    gpu_logits=[];cursor=0
    stats=dict(chunk_calls=0,points=0,factory_vs_actual_GPU_max_absolute_error=0.,independent_vs_actual_GPU_max_absolute_error=0.,
        CPU_two_formula_max_absolute_error=0.,actual_true_F9_or_center_inputs_byte_exact=True)
    H,W=grid.shape[1:]
    for item in iterator:
        check();need(type(item) is tuple and len(item)==6,'Scout hook yields six explicit arrays with true F9, never N17 extra8')
        flat,F,e,L,neighbors,actual=item;expected=forward_indices[cursor:cursor+2048];n=len(expected)
        need(n>0,'No extra or empty actual scout chunk')
        legacy._tensor(torch,flat,torch.int64,(n,),'Actual flat indices')
        need(np.array_equal(flat.numpy(),expected),'Complete ascending C/2048 original chunk order')
        for name,value,shape in (('F',F,(n,256)),('e0',e,(256,)),('rawL',L,(n,)),('neighbors_F',neighbors,(n,9,256)),('actualGPUlogits',actual,(n,3))):
            legacy._tensor(torch,value,torch.float32,shape,name)
        expected_F,expected_neighbors=feature_neighbors(torch,np,feature_context,flat,int(H),int(W),arm)
        need(_same(torch,F,expected_F) and _same(torch,neighbors,expected_neighbors),'Actual F9/center input byte exact from original union+new extra/map')
        need(_same(torch,e,packet['e0']) and _same(torch,L,packet['native_L'].reshape(-1)[flat]),'Same query e0 and original center L exact')
        legacy._same_common_inputs(torch,np,[dict(flat_indices=flat,sampled_F=F)],common,forward_indices)
        with torch.no_grad():
            factory=head(F.clone(),e[None].expand(n,256).clone(),L.clone(),neighbors.clone())
            independent=endpoint.independent_logits(torch,state,F.clone(),e[None].expand(n,256).clone(),L.clone(),neighbors.clone())
        legacy._tensor(torch,factory,torch.float32,(n,3),'CPU factory logits');legacy._tensor(torch,independent,torch.float32,(n,3),'CPU explicit logits')
        first,second=_error(torch,factory,actual),_error(torch,independent,actual)
        need(first<=ATOL and second<=ATOL,'Both same-input CPU scout formulas vs actualGPU absolute1e-4/rtol0')
        stats['factory_vs_actual_GPU_max_absolute_error']=max(stats['factory_vs_actual_GPU_max_absolute_error'],first)
        stats['independent_vs_actual_GPU_max_absolute_error']=max(stats['independent_vs_actual_GPU_max_absolute_error'],second)
        stats['CPU_two_formula_max_absolute_error']=max(stats['CPU_two_formula_max_absolute_error'],_error(torch,factory,independent))
        stats['chunk_calls']+=1;stats['points']+=n;gpu_logits.append(actual.numpy().copy());cursor+=n
    need(cursor==len(forward_indices) and stats['chunk_calls']==(len(forward_indices)+2047)//2048,'Complete actual C scout coverage')
    return (np.concatenate(gpu_logits,axis=0) if gpu_logits else np.empty((0,3),dtype=np.float32)),stats

def proof_query(torch,np,cv2,helper,legacy_selector_proof,endpoint_body,query_id,native_L_grid,e0,C_flat_indices,K,feature_context,
        heads,states,common_PR_chunks,iter_scout_chunks,read_scores,read_profile,formulas,rle_encoder,*,check_callback=None):
    """Prove one natural query via arrays/hooks. No unknown new storage schema is inferred.

    iter_scout_chunks(epoch,arm) yields(flat,F,e0,L,neighbors_F[N,9,256],actualGPUlogits).
    read_scores(epoch,arm) returns saved CPU FP64 canonical route scores from actual GPU logits.
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
    need(callable(iter_scout_chunks) and callable(read_scores) and callable(read_profile) and callable(rle_encoder) and
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
    check();common=legacy._prove_PR(torch,np,common_PR_chunks,active,packet,formulas,'N20/common_C')
    profiles={};scout_stats={}
    for epoch in EPOCHS:
        for arm in ARMS:
            check();key=(epoch,arm)
            gpu_logits,stats=_prove_scout_chunks(torch,np,legacy,endpoint_body,heads[key],states[key],query_id,arm,
                native_L_grid,packet,feature_context,active,common_PR_chunks,iter_scout_chunks(epoch,arm),check)
            scout_stats['E%d_%s'%(epoch,arm)]=stats
            canonical=helper.error_score(gpu_logits) if K else np.empty(0,dtype=np.float64)
            saved_scores=read_scores(epoch,arm)
            need(isinstance(saved_scores,torch.Tensor) and saved_scores.device.type=='cpu' and saved_scores.layout==torch.strided and
                saved_scores.dtype==torch.float64 and tuple(saved_scores.shape)==(len(active),) and not saved_scores.requires_grad and
                bool(torch.isfinite(saved_scores).all()),'Actual saved canonical CPU FP64 route scores')
            scores=saved_scores.numpy()
            need(canonical.dtype==scores.dtype==np.float64 and canonical.shape==scores.shape and
                np.array_equal(np.ascontiguousarray(scores).view(np.uint64),np.ascontiguousarray(canonical).view(np.uint64)),
                'Saved FP64 scores must be EXACT frozen canonical readout of actual GPU logits')
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
                subset=legacy._prove_PR(torch,np,blocks,chosen,packet,formulas,'N20/E%d/%s/%s/selected'%(epoch,arm,mode))
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
                    ranking_inputs='saved_canonical_CPU_FP64_scores_exactly_checked_against_actual_GPU_FP32_logits',
                    CPU_scout_outputs_used_for_ranking=False,CPU_approximate_PR_used_for_ranking=False)
                del subset,record,blocks,mask
            del gpu_logits,score_grid,scores
    return dict(query_id=query_id,K=K,C_points=len(indices),profiles=profiles,scout_numeric=scout_stats,
        common_C_PR=dict(formula_stats=common['formula_stats'],formula_chunk_calls=common['formula_chunk_calls'],strict_sign_exact=True),
        all_selected_indices_exact=True,all_full_masks_and_RLE_exact=True,actual_true_F9_or_center_inputs_byte_exact=True,actual_saved_FP64_scores_canonical_exact=True,
        CPU_scout_outputs_used_for_ranking=False,atol=1e-4,rtol=0.,native_R_reference_reuse_proven_here=False,
        original_query_ids_classes_S0_export_order_require_external_binding=True,AP_evaluated=False,
        actual_capture_provenance_proven_here=False,ROOT_scientific_qualification_conferred=False,
        CPU_feature_rerender_equivalence_proven=False,
        scope='one supplied actual query: original-grid true F9/center input, CPU two head formulas vs GPU, exact saved FP64 score selection and two PR formulas/mask/RLE; ROOT Source/state/export/all100/actual0 external; no CPU feature rerender equivalence')
