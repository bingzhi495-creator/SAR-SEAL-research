"""UNBOUND CPU same-GPU-input selection/PR proof body; no I/O or science run.

Ranking ONLY uses supplied actual GPU scout logits / full-C GPU rawu and
combined, through the frozen NumPy FP64 helper. CPU scout forward is absent.
ROOT must authenticate complete actual captures, Source/ad/RAW/state/formula
provenance, resource admission, original export/score order and final scope.
No GT, targets, renderer, state load, file ref or Source operation enters here.
"""
CAPTURE_SCHEMA = 'N16-all100-scout-selection-same-GPU-input-query-capture-v1'
POINT_CHUNK = 2048
ATOL = 1.e-4
FORMULA_NAMES = ('original_head', 'independent_PR')
CONTROLS = ('UTILITY','ERROR','DENSE_ERROR_FLIP','MAGNET')


class ProofFailure(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ProofFailure(message)


def _keys(value, keys, name):
    need(type(value) is dict and set(value)==set(keys),name+' exact core keys required')


def _tensor(torch, value, dtype, shape, name):
    need(isinstance(value,torch.Tensor) and value.layout==torch.strided
         and value.device.type=='cpu' and value.dtype==dtype
         and tuple(value.shape)==tuple(shape) and not value.requires_grad,name+' exact detached CPU tensor required')
    if dtype==torch.float32:
        need(bool(torch.isfinite(value).all()),name+' must be finite')


def _same_fp32(torch, left, right):
    return (left.shape==right.shape and left.dtype==right.dtype==torch.float32
            and torch.equal(left.contiguous().view(torch.int32),right.contiguous().view(torch.int32)))


def _domains(np,cv2,L):
    P,Omega=L>0,np.abs(L)<=2  # Keep ORIGINAL FP32 thresholds; no render/conversion.
    if bool(P.any()):
        dilated=cv2.dilate(np.ascontiguousarray(P,dtype=np.uint8),np.ones((17,17),dtype=np.uint8),
                          anchor=(8,8),iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        C=Omega|P|(dilated!=0)
    else:
        C=Omega.copy()
    return P,C,np.flatnonzero(C).astype(np.int64),int(np.count_nonzero(Omega))


def _blocks(torch,np,blocks,indices,packet,kind):
    need(type(blocks) is list and len(blocks)==(len(indices)+POINT_CHUNK-1)//POINT_CHUNK,
         kind+' complete canonical2048 chunk count required')
    for i,block in enumerate(blocks):
        common=('flat_indices','sampled_F','e0','native_L')
        _keys(block,common+(('scout_logits',) if kind=='scout' else ('GPU_raw_residual','GPU_combined')),kind)
        expected=indices[i*POINT_CHUNK:(i+1)*POINT_CHUNK];n=len(expected)
        _tensor(torch,block['flat_indices'],torch.int64,(n,),kind+' indices')
        need(bool(np.array_equal(block['flat_indices'].numpy(),expected)),kind+' indices must equal canonical ascending support')
        _tensor(torch,block['sampled_F'],torch.float32,(n,256),kind+' actual sampled F')
        _tensor(torch,block['e0'],torch.float32,(256,),kind+' actual e0')
        _tensor(torch,block['native_L'],torch.float32,(n,),kind+' actual native L')
        need(_same_fp32(torch,block['e0'],packet['e0']),kind+' e0 must match query input bitwise')
        native=packet['native_L'].reshape(-1)[block['flat_indices']]
        need(_same_fp32(torch,block['native_L'],native),kind+' L must match native input bitwise')
        if kind=='scout':
            _tensor(torch,block['scout_logits'],torch.float32,(n,3),'Actual GPU scout logits')
        else:
            _tensor(torch,block['GPU_raw_residual'],torch.float32,(n,),'Actual GPU original residual')
            _tensor(torch,block['GPU_combined'],torch.float32,(n,),'Actual GPU combined logits')


def _same_common_inputs(torch,np,blocks,common_blocks,C_indices):
    """Compare views/grouped rows without concatenating an entire 256D F."""
    for block in blocks:
        indices=block['flat_indices'].numpy()
        positions=np.searchsorted(C_indices,indices)
        need(bool(np.array_equal(C_indices[positions],indices)),'Point support must be inside common C')
        groups=positions//POINT_CHUNK
        for group in np.unique(groups):
            local=np.flatnonzero(groups==group).astype(np.int64)
            reference=(positions[local]%POINT_CHUNK).astype(np.int64)
            selected_rows=block['sampled_F'][torch.from_numpy(local)]
            common_rows=common_blocks[int(group)]['sampled_F'][torch.from_numpy(reference)]
            need(_same_fp32(torch,selected_rows,common_rows),'Same actual F required: no CPU render/alternate point inputs')


def _comparison(torch,left_u,left_after,right_u,right_after,scope):
    if left_u.numel():
        u_error=float((left_u.to(torch.float64)-right_u.to(torch.float64)).abs().max())
        after_error=float((left_after.to(torch.float64)-right_after.to(torch.float64)).abs().max())
    else:
        u_error=after_error=0.
    sign=torch.equal(left_after>0,right_after>0)
    need(u_error<=ATOL and after_error<=ATOL and sign,
         scope+': absolute1e-4/rtol0 AND strict>0 sign required; no fallback')
    return u_error,after_error


def _prove_PR(torch,np,blocks,indices,packet,formulas,scope):
    _blocks(torch,np,blocks,indices,packet,'PR')
    residual=np.empty(len(indices),dtype=np.float32);combined=np.empty(len(indices),dtype=np.float32)
    stats={name:dict(calls=0,points=0,residual_max_absolute_error=0.,combined_max_absolute_error=0.,strict_sign_exact=True)
           for name in FORMULA_NAMES}
    for i,block in enumerate(blocks):
        first=i*POINT_CHUNK;n=len(block['flat_indices'])
        actual_u,actual_after=block['GPU_raw_residual'],block['GPU_combined']
        residual[first:first+n]=actual_u.numpy();combined[first:first+n]=actual_after.numpy()
        for name in FORMULA_NAMES:
            F=block['sampled_F'].clone()
            e=block['e0'][None].expand(n,256).clone()
            L=block['native_L'].clone()
            with torch.no_grad():
                result=formulas[name](F,e,L)
            need(type(result) is tuple and len(result)==2,'Closed formula wrapper must return (u,L+u)')
            u,after=result
            _tensor(torch,u,torch.float32,(n,),name+' CPU u')
            _tensor(torch,after,torch.float32,(n,),name+' CPU after')
            expected=block['native_L']+u
            need(_same_fp32(torch,after,expected),name+' CPU combined must be its own FP32 nativeL+u')
            du,da=_comparison(torch,u,after,actual_u,actual_after,scope+'/'+name)
            stat=stats[name];stat['calls']+=1;stat['points']+=n
            stat['residual_max_absolute_error']=max(stat['residual_max_absolute_error'],du)
            stat['combined_max_absolute_error']=max(stat['combined_max_absolute_error'],da)
            del F,e,L,result,u,after,expected
    return dict(residual=residual,combined=combined,formula_stats=stats,
                formula_chunk_calls={name:stats[name]['calls'] for name in FORMULA_NAMES},strict_sign_exact=True)


def proof_query(torch,np,cv2,helper,packet,formulas):
    """Return mechanical proof for ONE supplied query capture; not a Q/gate.

    Only two closed ROOT wrappers original_head/independent_PR may compute PR,
    each(F_CPU[N,256],e_CPU[N,256],L_CPU[N])->(u_CPU,Lplusu_CPU), FP32.
    Actual original independent_PR(torch,state,F,L) may ignore e mathematically;
    ROOT binds/wraps its real ABI. Original40497/8state and distinct formula
    provenance are ROOT obligations, not inferred from callable names here.
    """
    _keys(packet,('schema','query_id','evalmode','native_L','e0','K','C_flat_indices','common_C_PR_chunks','controls'),'Query capture')
    need(packet['schema']==CAPTURE_SCHEMA,'New core capture schema required')
    q=packet['query_id'];mode=packet['evalmode']
    need(type(q) is int and 0<=q<100 and type(mode) is str and mode in ('atmost','exact'),'Exact query id/mode required')
    need(type(formulas) is dict and set(formulas)==set(FORMULA_NAMES)
         and all(callable(formulas[n]) for n in FORMULA_NAMES),'Two named ROOT closed formula wrappers required')
    L=packet['native_L']
    need(isinstance(L,torch.Tensor) and L.ndim==2 and min(L.shape)>0,'Native L[H,W] required')
    _tensor(torch,L,torch.float32,L.shape,'Native L')
    _tensor(torch,packet['e0'],torch.float32,(256,),'Query e0')
    L_numpy=L.numpy();P,C,C_indices,K=_domains(np,cv2,L_numpy)
    need(type(packet['K']) is int and packet['K']==K,'Recorded K must equal physical absL<=2 count')
    _tensor(torch,packet['C_flat_indices'],torch.int64,(len(C_indices),),'Recorded C indices')
    need(bool(np.array_equal(packet['C_flat_indices'].numpy(),C_indices)),'Recorded C must equal independent CV2 physical domain exactly')
    control_names=CONTROLS if mode=='atmost' else ('UTILITY','ERROR')
    _keys(packet['controls'],control_names,'Mode controls')
    forward_indices=C_indices if K else np.empty(0,dtype=np.int64)
    common=_prove_PR(torch,np,packet['common_C_PR_chunks'],forward_indices,packet,formulas,'common_C')
    results={}
    for name in control_names:
        record=packet['controls'][name]
        _keys(record,('C_scout_chunks','selected_PR_chunks','actual_selected_indices','actual_mask'),name)
        if name=='MAGNET':
            need(type(record['C_scout_chunks']) is list and not record['C_scout_chunks'],'MagNet must not invent scout calls/logits')
        else:
            _blocks(torch,np,record['C_scout_chunks'],forward_indices,packet,'scout')
            _same_common_inputs(torch,np,record['C_scout_chunks'],packet['common_C_PR_chunks'],C_indices)
        if K:
            if name=='MAGNET':
                scores=helper.magnet_score(L_numpy.ravel()[C_indices],common['residual'])
            else:
                gpu_logits=np.concatenate([b['scout_logits'].numpy() for b in record['C_scout_chunks']],axis=0)
                if name=='UTILITY':scores=helper.utility_score(gpu_logits)
                elif name=='ERROR':scores=helper.error_score(gpu_logits)
                else:
                    flip=(L_numpy.ravel()[C_indices]>0)!=(common['combined']>0)
                    scores=helper.dense_error_flip_score(flip,gpu_logits)
                del gpu_logits
            need(scores.dtype==np.float64 and scores.shape==(len(C_indices),) and bool(np.isfinite(scores).all()),'Canonical FP64 actual-GPU-input score required')
            full_score=np.zeros(L.shape,dtype=np.float64);full_score.ravel()[C_indices]=scores
            selected=helper.choose_candidates(L_numpy,C,full_score,K,mode=mode)
            del full_score,scores
        else:
            selected=np.zeros(L.shape,dtype=np.bool_)
        selected_indices=np.flatnonzero(selected).astype(np.int64)
        actual_indices=record['actual_selected_indices']
        _tensor(torch,actual_indices,torch.int64,(len(selected_indices),),name+' actual selected indices')
        need(bool(np.array_equal(actual_indices.numpy(),selected_indices)),name+' actual selected indices must be EXACT; no rank tolerance')
        _tensor(torch,record['actual_mask'],torch.bool,L.shape,name+' actual full mask')
        positions=np.searchsorted(C_indices,selected_indices)
        if name in ('UTILITY','ERROR'):
            subset=_prove_PR(torch,np,record['selected_PR_chunks'],selected_indices,packet,formulas,name+'/selected')
            _same_common_inputs(torch,np,record['selected_PR_chunks'],packet['common_C_PR_chunks'],C_indices)
            _comparison(torch,torch.from_numpy(subset['residual']),torch.from_numpy(subset['combined']),
                        torch.from_numpy(common['residual'][positions]),torch.from_numpy(common['combined'][positions]),
                        name+'/actual_selected_vs_actual_common_C')
            after=subset['combined'];formula_stats=subset['formula_stats']
        else:
            need(type(record['selected_PR_chunks']) is list and not record['selected_PR_chunks'],'Strong control must use actual saved full-C action')
            after=common['combined'][positions];formula_stats=common['formula_stats']
        mask=P.copy();mask.ravel()[selected_indices]=after>0
        actual_mask=record['actual_mask'].numpy()
        need(bool(np.array_equal(actual_mask[~selected],P[~selected])),name+' native mask must remain EXACT outside selected support')
        need(bool(np.array_equal(mask,actual_mask)),name+' complete new mask must match EXACTLY')
        results[name]=dict(selected_points=len(selected_indices),selected_indices_exact=True,full_mask_exact=True,
                           outside_selected_native_exact=True,CPU_formula_signs_make_full_mask_exact=True,
                           reconstructed_mask=torch.from_numpy(mask),formula_stats=formula_stats,
                           ranking_inputs='supplied_actual_GPU_logits_or_actual_GPU_common_C_action',
                           CPU_approximate_PR_used_for_ranking=False)
        if name in ('UTILITY','ERROR'):del subset
    return dict(query_id=q,evalmode=mode,K=K,C_points=len(C_indices),controls=results,
                common_C_PR=dict(formula_stats=common['formula_stats'],formula_chunk_calls=common['formula_chunk_calls'],strict_sign_exact=True),
                CPU_scout_forward_used_for_ranking=False,atol=ATOL,rtol=0.,
                proof_scope='supplied_capture_selection_and_same_point_PR_only',
                actual_capture_provenance_proven_here=False,ROOT_scientific_qualification_conferred=False,
                requires_ROOT_actual_training_to_deployment_input_batch_gate=True)


def proof_all100_queries(torch,np,cv2,helper,query_captures,formulas,*,evalmode='atmost',query_callback=None):
    """Consume exactly natural0..99 for ONE image; caller owns actual I/O/gates."""
    need(type(evalmode) is str and evalmode in ('atmost','exact'),'Budget mode required')
    need(query_callback is None or callable(query_callback),'Optional proof callback required')
    iterator=iter(query_captures)
    for q in range(100):
        try:packet=next(iterator)
        except StopIteration:raise ProofFailure('Incomplete all100 capture; no padding')
        need(type(packet) is dict and packet.get('query_id')==q and packet.get('evalmode')==evalmode,'Natural query0..99 and same mode required')
        report=proof_query(torch,np,cv2,helper,packet,formulas)
        if query_callback is not None:query_callback(report)
        del report,packet
    try:next(iterator)
    except StopIteration:pass
    else:raise ProofFailure('Extra query capture outside natural100')
    return dict(queries=100,evalmode=evalmode,all_selected_indices_exact=True,all_full_masks_exact=True,
                CPU_scout_forward_used_for_ranking=False,ROOT_scientific_qualification_conferred=False,
                completed_GPU_report_capture_manifest_source_state_and_RAW_gates='ROOT_external_required')
