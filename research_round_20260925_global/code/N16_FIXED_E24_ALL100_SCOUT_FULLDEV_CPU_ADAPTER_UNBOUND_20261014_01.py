"""UNBOUND pure full-image CPU replay adapter; no refs IO/gates/scout forward.
ROOT authenticates physical loaders/bindings/lineage and official COCO outside.
Uses actual saved GPU logits for canonical ranking and actual masks as targets.
"""

def need(ok,message):
    if not ok:raise RuntimeError(message)


def unwrap(packet,binding):
    need(type(packet) is dict and set(packet)=={'schema','binding','value'} and
        packet['schema']=='N16-actual-GPU-streamed-capture-member-v1' and packet['binding']==binding,
        'Exact actual physical wrapper schema/binding/value required')
    return packet['value']


def profile_index(profiles):
    need(type(profiles) is list and len(profiles)==500,'Exact100 strong plus400 query profiles required')
    strong,cores={},{}
    for record in profiles:
        q=record['query_id'];need(type(q) is int and 0<=q<100,'Natural100 query id required')
        if record.get('kind')=='strong_fixed_Omega':
            need(set(record)=={'query_id','kind','point_chunks'} and q not in strong,'Distinct original strong Omega record required')
            strong[q]=record
        else:
            need(set(record)=={'query_id','epoch','evalmode','core'} and record['epoch'] in (8,24)
                and record['evalmode'] in ('atmost','exact'),'Exact fixed core profile fields required')
            key=(q,record['epoch'],record['evalmode'])
            need(key not in cores,'Duplicate query/endpoint/mode profile forbidden')
            cores[key]=record
    need(set(strong)==set(range(100)) and set(cores)=={(q,e,m) for q in range(100) for e in (8,24) for m in ('atmost','exact')},
        'Complete natural100/fixed8-24/two budget modes, no padding or missing profile')
    return strong,cores


def strong_chunks(torch,np,shared,shared_ref,record,loader,binding,*,synthetic_only=False):
    q=record['query_id'];indices=np.flatnonzero(np.abs(shared['native_L'][q].numpy())<=2).astype(np.int64)
    refs=record['point_chunks'];need(type(refs) is list and len(refs)==(len(indices)+2047)//2048,'Complete original Omega2048 blocks')
    blocks=[]
    for i,ref in enumerate(refs):
        value=unwrap(loader(ref),binding);expected=indices[i*2048:(i+1)*2048];idx=value['flat_indices']
        need(value['query_id']==q and value['control']=='STRONG_FIXED_E24_OMEGA' and value['stage']=='original_Omega_PR' and
            value['evalmode']=='atmost' and value['point_chunk']==2048 and value['chunk_first']==i*2048 and
            value['shared_actual_image_inputs']==shared_ref and value['actual_observed_F_e_L_bitexact_to_shared'] is True and
            value['actual_input_scope']=='caller_actual_original_device_FP32_F_e_L' and value['synthetic_only'] is synthetic_only and
            idx.device.type=='cpu' and idx.dtype==torch.int64 and idx.shape==(len(expected),) and np.array_equal(idx.numpy(),expected),
            'Original Omega chunk identity/role/true support/batch cannot be substituted')
        positions=np.searchsorted(shared['union_flat_indices'].numpy(),expected)
        need(bool(np.all(positions<len(shared['union_flat_indices']))) and
            np.array_equal(shared['union_flat_indices'].numpy()[positions],expected),'Original Omega support outside actual input union')
        blocks.append(dict(flat_indices=idx,sampled_F=shared['F_unique'][torch.from_numpy(positions.astype(np.int64))],
            e0=shared['e0'][q],native_L=shared['native_L'][q].reshape(-1)[idx],
            GPU_raw_residual=value['GPU_raw_residual'],GPU_combined=value['GPU_combined']))
    return indices,blocks


def _export(np,masks,n10,baseline,mask):
    value=dict(baseline)
    value['segmentation']=n10.segmentation(np,masks,mask.astype(np.uint8))
    need({k:v for k,v in value.items() if k!='segmentation'}=={k:v for k,v in baseline.items() if k!='segmentation'},
        'Original native scores/class/query order untouched')
    return value


def replay_image(torch,np,cv2,masks,n10,gpu_module,proof,helper,formulas,shared_wrapper,shared_ref,
                 profiles,load_closed_physical,binding,baseline_rows,strong_rows,*,check=None,synthetic_only=False,query_start_callback=None):
    """Return twelve-policy CPU RLE lists and mechanical details for one image.
    Exactly100 natural queries; qids export order is the original actual order.
    Root keeps one actual shared F/e/L wholeimage in memory and a query-local
    closed-physical loader cache. No CPU F render or scout forward occurs.
    """
    shared=unwrap(shared_wrapper,binding)
    gpu_module.validate_shared_for_CPU(torch,np,cv2,shared)
    strong_profiles,cores=profile_index(profiles)
    qids=shared['native_export_query_ids']
    need(len(baseline_rows)==len(strong_rows)==100 and [v['query_id'] for v in baseline_rows]==qids and
        [v['query_id'] for v in strong_rows]==qids,'Original natural100 export population required')
    baseline={v['query_id']:v for v in baseline_rows};strong_expected={v['query_id']:v for v in strong_rows}
    outputs={name:{} for name in gpu_module.policy_names()};details=[]
    for q in range(100):
        if check is not None:check()
        if query_start_callback is not None:query_start_callback(q)
        P=shared['native_L'][q].numpy()>0
        native=_export(np,masks,n10,baseline[q],P)
        need(native==baseline[q],'Actual full100 native L/RLE must match original reference exactly')
        outputs['native_official'][q]=native
        packet=dict(native_L=shared['native_L'][q],e0=shared['e0'][q])
        omega,blocks=strong_chunks(torch,np,shared,shared_ref,strong_profiles[q],load_closed_physical,binding,synthetic_only=synthetic_only)
        strong_result=proof._prove_PR(torch,np,blocks,omega,packet,formulas,'strong_original_Omega')
        strong_mask=P.copy();strong_mask.reshape(-1)[omega]=strong_result['combined']>0
        strong_value=_export(np,masks,n10,baseline[q],strong_mask)
        need(strong_value==strong_expected[q],'Original strongest E24 complete RLE must reconstruct independently')
        outputs['strong_fixed_E24_Omega'][q]=strong_value
        query_details=[]
        for epoch in (24,8):
            for mode in ('atmost','exact'):
                wrapper=load_closed_physical(cores[(q,epoch,mode)]['core'])
                raw=unwrap(wrapper,binding)
                need(raw['query_id']==q and raw['epoch']==epoch and raw['evalmode']==mode,'Core cannot swap model endpoint/query/mode')
                core=gpu_module.decode_CPU_query_core(torch,np,wrapper,shared_wrapper,shared_ref,load_closed_physical,binding,synthetic_only=synthetic_only)
                result=proof.proof_query(torch,np,cv2,helper,core,formulas)
                if epoch==24 and mode=='atmost':
                    C=core['C_flat_indices'].numpy();positions=np.searchsorted(C,omega)
                    need(np.array_equal(C[positions],omega),'Original Omega is actual common C subset')
                    common_u=np.concatenate([b['GPU_raw_residual'].numpy() for b in core['common_C_PR_chunks']]) if len(C) and core['K'] else np.empty(0,np.float32)
                    common_after=np.concatenate([b['GPU_combined'].numpy() for b in core['common_C_PR_chunks']]) if len(C) and core['K'] else np.empty(0,np.float32)
                    proof._comparison(torch,torch.from_numpy(strong_result['residual']),torch.from_numpy(strong_result['combined']),
                        torch.from_numpy(common_u[positions]),torch.from_numpy(common_after[positions]),'original_Omega_vs_actual_common_C')
                for control,item in result['controls'].items():
                    if control in ('UTILITY','ERROR') or epoch==24:
                        name='E%d_%s_%s'%(epoch,control,mode)
                        need(name in outputs,'No undeclared AP policy')
                        outputs[name][q]=_export(np,masks,n10,baseline[q],item['reconstructed_mask'].numpy())
                    del item['reconstructed_mask']
                query_details.append(dict(epoch=epoch,mode=mode,proof=result))
                del wrapper,raw,core,result
        details.append(dict(query_id=q,strong_original_Omega=strong_result['formula_stats'],fixed_profiles=query_details))
        del P,native,packet,omega,blocks,strong_result,strong_mask,strong_value,query_details
    need(all(set(values)==set(range(100)) for values in outputs.values()),'All twelve AP policies require exactly100 natural outputs')
    return {name:[values[q] for q in qids] for name,values in outputs.items()},details
