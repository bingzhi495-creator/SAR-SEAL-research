"""UNBOUND_DRAFT: independent N16 all100 sampled FIT packet CPU validator body.

No IO, source/admission/RAW gate, checkpoint loading, native teacher forward or
cache-body import. ROOT supplies already closed actual inputs/modules at run time.
Development only read authorized ordinary excerpts; this body has NOT run on data.
"""
import hashlib

PACKET_SCHEMA = 'N16-all100-FIT-action-sampled-cache-body-v1'
PR_NAMES = ('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias',
            'hidden3.weight','hidden3.bias','final.weight','final.bias')
PR_SHAPES = ((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def tensor(torch, value, shape, dtype, label):
    require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and
        value.dtype==dtype and value.layout==torch.strided and tuple(value.shape)==tuple(shape) and
        not value.requires_grad and bool(torch.isfinite(value).all()), 'Finite CPU tensor ABI differs: '+label)

def digest(value):
    data=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(data.shape),dtype=str(data.dtype),sha256=hashlib.sha256(data.tobytes()).hexdigest())

def independent_domains(torch,np,cv2,L):
    array=L.numpy()
    indices,C_counts,K_counts={},{},{}
    for q in range(100):
        P=array[q]>0
        Omega=np.abs(array[q])<=2
        expanded=cv2.dilate(P.astype(np.uint8),np.ones((17,17),dtype=np.uint8),
            iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(np.bool_)
        C=Omega|P|expanded
        require(not np.any(Omega & ~C),'Independent physical C must contain Omega')
        indices[q]=torch.from_numpy(np.flatnonzero(C).astype(np.int64))
        C_counts[q]=len(indices[q])
        K_counts[q]=int(np.count_nonzero(Omega))
    return indices,[C_counts[q] for q in range(100)],[K_counts[q] for q in range(100)]

def independent_sample(torch,indices,matched,background,seed):
    total=len(matched)+len(background)
    present=int(bool(matched))+int(bool(background))
    plan={}
    for name,qs in (('matched',matched),('background',background)):
        if qs:
            probability=1.0/present
            plan[name]=dict(query_ids=qs,n_stratum=len(qs),n_active=total,p_s=probability,
                draws=2048 if present==2 else 4096,weight=(len(qs)/total)/probability)
    rng=torch.Generator(device='cpu')
    rng.manual_seed(seed)
    before=rng.get_state().clone()
    query,flat,weight,stratum=[],[],[],[]
    for group_id,name in enumerate(('matched','background')):
        if name not in plan:
            continue
        record=plan[name]
        ordinals=torch.randint(record['n_stratum'],(record['draws'],),generator=rng)
        for ordinal in ordinals.tolist():
            q=record['query_ids'][ordinal]
            candidates=indices[q]
            offset=int(torch.randint(len(candidates),(1,),generator=rng).item())
            query.append(q)
            flat.append(int(candidates[offset]))
            weight.append(record['weight'])
            stratum.append(group_id)
    return dict(query_ids=torch.tensor(query,dtype=torch.int64),flat_indices=torch.tensor(flat,dtype=torch.int64),
        importance_weights=torch.tensor(weight,dtype=torch.float32),stratum_ids=torch.tensor(stratum,dtype=torch.uint8),
        sampler_state_before=before,sampler_state_after=rng.get_state().clone(),plan=plan,seed=seed,
        replacement=True,duplicates_preserved=True,
        draw_order='matched then background; vector query draw then sequential point draws')

def observed_assignment(torch,packet,raw_GT):
    mapping,ann_ids=packet['raw_target_mapping'],packet['raw_GT_ids']
    G=len(raw_GT)
    require(type(mapping) is list and sorted(mapping)==list(range(G)) and ann_ids==packet['identity']['ann_ids'] and
        len(ann_ids)==G and len(set(ann_ids))==G,'Exact original raw-GT mapping/ann_ids required')
    observations=packet['all_ten_target_observations']
    matches=packet['native_stage_matches']
    require(len(observations)==len(matches)==10,'Actual ten original target observations required')
    for observation,old in zip(observations,matches):
        for name,dtype in (('labels',torch.int64),('label_weights',torch.int64),('mask_weights',torch.float32)):
            tensor(torch,observation[name],(100,),dtype,name)
        pos,neg=observation['positive_query_indices'],observation['negative_query_indices']
        assigned=observation['assigned_GT_indices']
        require(type(pos) is list and type(neg) is list and len(pos)==len(assigned)==G and
            all(type(q) is int and 0<=q<100 for q in pos+neg) and
            len(set(pos+neg))==len(pos)+len(neg) and sorted(assigned)==list(range(G)) and
            pos==observation['sampler_positive_query_indices'] and neg==observation['sampler_negative_query_indices'],
            'Observed original positive/negative/sampler/GT assignment differs')
        require(old==dict(positive_query_indices=pos,assigned_GT_indices=assigned,positives=len(pos),
            target_labels=digest(observation['labels'])),'Actual old ten-stage matching/digest not preserved')
        labels,lw,mw=(observation[name] for name in ('labels','label_weights','mask_weights'))
        require(bool((lw>=0).all()) and bool((mw>=0).all()) and
            all(int(labels[q])==0 and float(lw[q])>0 and float(mw[q])>0 for q in pos) and
            all(int(labels[q])==1 and float(mw[q])==0 for q in neg),'Fresh original foreground/background target policy differs')
        require(type(observation['mask_targets_digest']) is dict and
            set(observation['mask_targets_digest'])=={'shape','dtype','sha256'},
            'Original processed mask-target digest must remain explicit')
    final=observations[-1]
    raw_index=torch.full((100,),-2,dtype=torch.int64)
    kinds=['ignored']*100
    for q,processed in zip(final['positive_query_indices'],final['assigned_GT_indices']):
        raw_index[q]=mapping[processed]
        kinds[q]='matched'
    for q in final['negative_query_indices']:
        if float(final['label_weights'][q])>0:
            raw_index[q]=-1
            kinds[q]='background'
    valid=raw_index!=-2
    assignment=packet['assignment']
    pos,neg=final['positive_query_indices'],final['negative_query_indices']
    ignored=[q for q in range(100) if q not in set(pos+neg)]
    require(assignment['natural_query_ids']==list(range(100)) and
        torch.equal(raw_index,assignment['raw_GT_index_by_query']) and torch.equal(valid,assignment['valid_targets']) and
        assignment['supervision_kind_by_query']==kinds and
        assignment['native_positive_query_indices']==pos and assignment['native_negative_query_indices']==neg and
        assignment['native_ignored_query_indices']==ignored and
        assignment['raw_GT_annotation_id_by_query']==[ann_ids[g] if g>=0 else None for g in raw_index.tolist()] and
        all(torch.equal(assignment['final_native_'+suffix],final[name]) for suffix,name in
            (('labels','labels'),('label_weights','label_weights'),('mask_weights','mask_weights'))) and
        assignment['background_empty_set_is_new_supervision_extension'] is True and
        assignment['original_native_negative_mask_supervised'] is False and assignment['ignore_never_zero_filled'] is True,
        'Background empty-set must be explicit new supervision, not fabricated native loss evidence')
    return raw_index,valid,kinds

def absolute(torch,actual,stored,label,sign=False):
    require(actual.dtype==stored.dtype==torch.float32 and actual.shape==stored.shape and
        actual.device.type==stored.device.type=='cpu' and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(stored).all()),
        'Finite explicit formula outputs required: '+label)
    error=(actual-stored).abs()
    require(bool((error<=1e-4).all()),'Fixed absolute1e-4 PR compatibility failed: '+label)
    if sign:
        require(torch.equal(actual>0,stored>0),'Strict PR final sign failed: '+label)
    return float(error.max()) if error.numel() else 0.

def validate_packet(torch,np,cv2,packet,recipe,pr_state,pr_formula_modules,raw_GT):
    """Validate future actual packet supplied ONLY after ROOT input/owner gates.

    raw_GT is ROOT's actual COCO-annotation CPU BOOL[G,H,W], not samples.targets.
    ROOT reinjects separately closed natural_L carrier as natural_native_L.
    ROOT must separately bind annotation/processed correspondence and original
    carrier. This body checks rawGT digest and ten-stage assignment semantics;
    it does not pretend to independently prove the processed correspondence.
    pr_formula_modules contains original_head and independent_PR callable.
    """
    require(not torch.cuda.is_initialized() and torch.get_num_threads()==torch.get_num_interop_threads()==2,
        'CPU2/coldCUDA required; ROOT owns environment/deadline/resources')
    outside=torch.get_rng_state().clone()
    require(packet['schema']==PACKET_SCHEMA and recipe['point_chunk']==2048 and
        recipe['points_two_groups_each']==2048 and recipe['points_single_group']==4096 and
        recipe['num_classes']==1 and recipe['PR_atol']==1e-4 and recipe['PR_rtol']==0 and
        recipe['background_empty_set_extension'] is True and recipe['carrier_epoch']==24 and
        recipe['carrier_seed']==202610120701,'ROOT must lock exact E24/background/proposal/absolute numeric recipe')
    index=packet['dataset_index']
    require(type(index) is int and index>=0 and packet['identity']['dataset_index']==index and
        packet['natural_query_ids']==list(range(100)),'Original image/index/natural100 identity differs')
    require(packet['matching_loss_seed']==recipe['matching_loss_seed_base']+index,
        'Original teacher matching seed lineage differs')
    H,W=packet['identity']['original_shape']
    require(type(H) is int and type(W) is int and H>0 and W>0,'Actual identity original shape required')
    tensor(torch,packet['natural_native_L'],(100,H,W),torch.float32,'actual GPU all100 L')
    tensor(torch,packet['native_e0'],(1,100,256),torch.float32,'actual GPU all100 e')
    tensor(torch,packet['native_class_logits'],(1,100,2),torch.float32,'actual GPU all100 class')
    tensor(torch,raw_GT,(len(packet['raw_GT_ids']),H,W),torch.bool,'independent actual rawGT')
    require(digest(raw_GT)==packet['raw_GT_digest'],'ROOT actual rawGT digest differs')
    require(all(key in packet for key in ('actual_raw_processed_GT_correspondence','processed_target_proof','original_cache_packet')),
        'Required ROOT caller proof fields must remain explicit')
    raw_index,valid,kinds=observed_assignment(torch,packet,raw_GT)
    native_masks=packet['natural_native_L'].numpy()>0
    raw_masks=raw_GT.numpy()
    maximum_IoU=[]
    for q in range(100):
        best=0.
        for G in raw_masks:
            intersection=int(np.count_nonzero(native_masks[q]&G))
            union=int(np.count_nonzero(native_masks[q]|G))
            value=intersection/union if union else 1.
            best=max(best,value)
        maximum_IoU.append(best)
    maximum_IoU=torch.tensor(maximum_IoU,dtype=torch.float32)
    tensor(torch,packet['max_native_mask_IoU_to_any_GT'],(100,),torch.float32,'FIT-only max native IoU')
    max_IoU_error=float((maximum_IoU-packet['max_native_mask_IoU_to_any_GT']).abs().max())
    require(max_IoU_error<=1e-4 and torch.equal(maximum_IoU>=.5,packet['max_native_mask_IoU_to_any_GT']>=.5) and
        packet['max_IoU_is_FIT_diagnostic_only'] is True,'FIT-only max IoU absolute range/strict half-threshold classification differs')
    indices,Cs,Ks=independent_domains(torch,np,cv2,packet['natural_native_L'])
    require(Cs==packet['C_counts_all100'] and Ks==packet['K_counts_all100'],'All100 independent C/K counts differ')
    active=[q for q in range(100) if bool(valid[q]) and Cs[q]>0 and Ks[q]>0]
    matched=[q for q in active if kinds[q]=='matched']
    background=[q for q in active if kinds[q]=='background']
    require(active==packet['active_queries'] and matched==packet['active_matched_queries'] and
        background==packet['active_background_queries'],'Active strata/ignored/K0 policy differs')
    sample=packet['samples']
    seed=recipe['point_seed_base']+index
    expected=independent_sample(torch,indices,matched,background,seed)
    S=len(expected['query_ids'])
    for key,value in expected.items():
        if isinstance(value,torch.Tensor):
            tensor(torch,sample[key],value.shape,value.dtype,'samples.'+key)
            require(torch.equal(value,sample[key]),'Private sampler/importance exact replay differs: '+key)
        else:
            require(sample[key]==value,'Prelocked sampling metadata differs: '+key)
    q,flat=expected['query_ids'],expected['flat_indices']
    unique,inverse=torch.unique(flat,sorted=True,return_inverse=True)
    tensor(torch,packet['unique_flat_indices'],unique.shape,torch.int64,'unique sample pixels')
    tensor(torch,packet['sample_to_unique'],inverse.shape,torch.int64,'sample inverse')
    tensor(torch,packet['sampled_F_unique'],(len(unique),256),torch.float32,'actual GPU sampledF')
    require(torch.equal(unique,packet['unique_flat_indices']) and torch.equal(inverse,packet['sample_to_unique']),
        'Actual sample-to-unique F mapping differs')
    native=packet['natural_native_L'].flatten(1)[q,flat]
    tensor(torch,sample['native_L'],(S,),torch.float32,'sample nativeL')
    require(torch.equal(native,sample['native_L']),'Sample must use byte-exact actual GPU L')
    target=torch.zeros(S,dtype=torch.bool)
    for i,query in enumerate(q.tolist()):
        gt=int(raw_index[query])
        require(gt>=-1,'Ignored query cannot receive a fabricated empty target')
        if gt>=0:
            target[i]=raw_GT[gt].flatten()[flat[i]]
    tensor(torch,sample['targets'],(S,),torch.bool,'sample BOOL target')
    require(torch.equal(target,sample['targets']),'Matched or explicit background target differs from actual rawGT')
    head=pr_formula_modules['original_head']
    explicit=pr_formula_modules['independent_PR']
    require(callable(explicit) and not head.training and not tuple(head.buffers()) and
        tuple(pr_state)==PR_NAMES and tuple(name for name,_ in head.named_parameters())==PR_NAMES,
        'ROOT must supply original qualified frozen PR ABI')
    parameters=dict(head.named_parameters())
    for name,shape in zip(PR_NAMES,PR_SHAPES):
        tensor(torch,pr_state[name],shape,torch.float32,'E24 PR.'+name)
        require(torch.equal(parameters[name],pr_state[name]) and not parameters[name].requires_grad and
            parameters[name].grad is None,'Frozen original E24 PR changed')
    for name in ('actual_PR_residual','actual_combined'):
        tensor(torch,sample[name],(S,),torch.float32,name)
    maxima=dict(factory_residual=0.,factory_combined=0.,explicit_residual=0.,explicit_combined=0.,
        between_formulas_residual=0.,between_formulas_combined=0.)
    with torch.no_grad():
        for first in range(0,S,2048):
            last=min(first+2048,S)
            F=packet['sampled_F_unique'][inverse[first:last]]
            e=packet['native_e0'][0,q[first:last]]
            L=native[first:last]
            result=head(F,e,L,L.new_zeros((last-first,49)),arm='PR')
            u1=result['residual']
            l1=L+u1
            u2,l2=explicit(torch,pr_state,F,L)
            for name,a,b,sign in (('factory_residual',u1,sample['actual_PR_residual'][first:last],False),
                ('factory_combined',l1,sample['actual_combined'][first:last],True),
                ('explicit_residual',u2,sample['actual_PR_residual'][first:last],False),
                ('explicit_combined',l2,sample['actual_combined'][first:last],True),
                ('between_formulas_residual',u1,u2,False),('between_formulas_combined',l1,l2,True)):
                maxima[name]=max(maxima[name],absolute(torch,a,b,name,sign))
    P=native>0
    A=sample['actual_combined']>0
    delta=P!=A
    error=P!=target
    utility=torch.full((S,),2,dtype=torch.int64)
    utility[delta & error]=0
    utility[delta & ~error]=1
    error_class=torch.full((S,),2,dtype=torch.int64)
    error_class[P & ~target]=0
    error_class[~P & target]=1
    for name,value in (('native_P',P),('actual_after_A',A),('delta',delta),('error',error),
        ('utility_labels',utility),('error_labels',error_class)):
        tensor(torch,sample[name],value.shape,value.dtype,name)
        require(torch.equal(sample[name],value),'Independent real fixed-action class/bit differs: '+name)
    require(packet['new_background_empty_set_supervision'] is True and packet['candidate_GT_independent'] is True and
        packet['original_matcher_negative_mask_supervision'] is False and packet['paired_scout_same_samples'] is True and
        packet['fixed_refiner_has_seen_FIT'] is True and packet['PR_label_generation_chunk_calls']==(S+2047)//2048 and
        packet['actual_label_generation_points']==S and packet['helper_classes']==dict(utility=['help','harm','noop'],error=['FP','FN','correct']),
        'Exposure/class order/background provenance/cost metadata differs')
    require(torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized(),'CPU validator changed outside RNG or CUDA')
    weighted={}
    for group_id,name in enumerate(('matched','background')):
        group=expected['stratum_ids']==group_id
        weighted[name]=dict(points=int(group.sum()),
            utility=[float(expected['importance_weights'][group & (utility==k)].double().sum()) for k in range(3)],
            error=[float(expected['importance_weights'][group & (error_class==k)].double().sum()) for k in range(3)])
    return dict(dataset_index=index,image_id=packet['identity']['image_id'],identity=packet['identity'],raw_GT=len(raw_GT),
        natural_queries=100,points=S,unique_sampled_F_points=len(unique),point_seed=seed,
        matching_loss_seed=packet['matching_loss_seed'],stratum_plan=expected['plan'],
        active_queries=len(active),active_matched_queries=len(matched),active_background_queries=len(background),
        C_counts_all100=Cs,K_counts_all100=Ks,sampler_exact=True,importance_weights_exact=True,
        actual_rawGT_digest_and_assignment_semantics_checked=True,background_empty_set_explicit_extension_checked=True,
        two_same_actual_GPU_input_PR_formulae_absolute1e4_strictsign=True,PR_errors=maxima,
        utility_class_counts=torch.bincount(utility,minlength=3).tolist(),
        error_class_counts=torch.bincount(error_class,minlength=3).tolist(),labels_exact=True,
        weighted_class_counts_by_stratum=weighted,
        importance_weight_min=float(expected['importance_weights'].min()) if S else None,
        importance_weight_max=float(expected['importance_weights'].max()) if S else None,
        importance_weight_sum=float(expected['importance_weights'].double().sum()),
        native_background_queries=sum(kind=='background' for kind in kinds),
        native_background_max_IoU_ge_point5=sum(float(maximum_IoU[q])>=.5 for q in range(100) if kinds[q]=='background'),
        background_IoU_is_duplicate_or_potential_TP_diagnostic_not_label=True,
        max_native_IoU_abs_error=max_IoU_error,max_native_IoU_half_threshold_exact=True,
        required_ROOT_external_proofs=['original_cache_packet qualified ref','processed_target_proof','actual_raw_processed_GT_correspondence'],
        processed_correspondence_independently_proved_by_this_body=False,
        native_teacher_forward_independently_replayed=False,current_terminal_self_certified=False,
        source_RAW_owner_admission_qualification=False,scope='CPU numerical/semantic body only; ROOT full chain and actual terminal still required')
