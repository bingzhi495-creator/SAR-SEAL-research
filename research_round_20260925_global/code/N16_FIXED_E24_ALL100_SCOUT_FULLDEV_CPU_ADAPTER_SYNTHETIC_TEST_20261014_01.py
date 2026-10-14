"""SYNTHETIC_ONLY CPU adapter/decoder rejection tests. No actual SCI or I/O.

Fake refs below identify in-memory fixtures only, never Source-qualified files.
Torch/NumPy/CV2 operate exclusively on tiny constructed tensors. No CUDA.
"""
import copy
import importlib.util
from pathlib import Path
import types
import torch
import numpy as np
import cv2


def module(filename):
    path=Path(__file__).with_name(filename)
    spec=importlib.util.spec_from_file_location(filename,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


adapter=module('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_ADAPTER_UNBOUND_DRAFT_20261014_01.py')
gpu=module('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_02.py')
proof=module('N16_ALL100_SCOUT_SELECTOR_CPU_PROOF_BODY_UNBOUND_DRAFT_20261014_01.py')
helper=module('N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')


def synthetic_segmentation(np,masks,mask):
    # A deterministic local RLE stand-in tests preservation/ordering, NOT COCO.
    flat=mask.ravel(order='F');last=0;counts=[];run=0
    for bit in flat:
        if int(bit)==last:run+=1
        else:counts.append(run);run=1;last=int(bit)
    counts.append(run)
    return dict(size=list(mask.shape),counts=counts)


def fixture():
    rng=torch.Generator(device='cpu').manual_seed(917)
    L=torch.full((100,2,3),-1.,dtype=torch.float32)
    L[0]=torch.tensor([[-3.,-1.,1.],[3.,-3.,-3.]])
    L[1]=torch.tensor([[-3.,-3.,3.],[-3.,-3.,-3.]])  # Nonempty C, K0.
    L[2].fill_(-3.)  # Empty P/C and K0.
    C=[];K=[]
    for q in range(100):
        _,_,idx,k=proof._domains(np,cv2,L[q].numpy());C.append(torch.from_numpy(idx));K.append(k)
    union=torch.arange(6,dtype=torch.int64)
    shared=dict(schema='N16-actual-GPU-image-shared-inputs-v1',dataset_index=0,image_id=91,
        identity={'SYNTHETIC_ONLY':True},native_packet={'SYNTHETIC_ONLY':True},selector={'SYNTHETIC_ONLY':True},
        natural_query_ids=list(range(100)),native_export_query_ids=list(reversed(range(100))),
        native_L=L,e0=torch.randn((100,256),generator=rng),F_unique=torch.randn((6,256),generator=rng),
        union_flat_indices=union,C_indices=C,C_counts=[len(x) for x in C],K_counts=K,U=6,
        actual_GPU_F_captured=True,CPU_F_rerendered=False)
    binding={'SYNTHETIC_ONLY':True};physical={};counter=0
    def put(value):
        nonlocal counter
        counter+=1
        ref=dict(path='E:/SYNTHETIC_ONLY/member_%05d.pth'%counter,bytes=0,sha256=('%064x'%counter))
        physical[ref['path']]=dict(schema='N16-actual-GPU-streamed-capture-member-v1',binding=binding,value=value)
        return ref
    shared_ref=put(shared)
    def chunk(q,idx,control,stage,mode,*,epoch=None,logits=False):
        if not len(idx):return []
        value=dict(query_id=q,control=control,stage=stage,evalmode=mode,point_chunk=2048,chunk_first=0,
            flat_indices=idx.clone(),shared_actual_image_inputs=shared_ref,
            actual_observed_F_e_L_bitexact_to_shared=True,
            actual_input_scope='caller_actual_original_device_FP32_F_e_L',synthetic_only=True)
        if logits:value['scout_logits']=torch.tensor([[1.,0.,0.]],dtype=torch.float32).repeat(len(idx),1)
        else:value.update(GPU_raw_residual=torch.zeros(len(idx)),GPU_combined=L[q].ravel()[idx].clone())
        if epoch is not None:value['scout_epoch']=epoch
        return [put(value)]
    profiles=[]
    for q in range(100):
        c=C[q];active=c if K[q] else torch.empty(0,dtype=torch.int64)
        omega=torch.from_numpy(np.flatnonzero(np.abs(L[q].numpy())<=2).astype(np.int64))
        profiles.append(dict(query_id=q,kind='strong_fixed_Omega',point_chunks=chunk(q,omega,'STRONG_FIXED_E24_OMEGA','original_Omega_PR','atmost')))
        common=chunk(q,active,'COMMON_C_REFERENCE','full_C_PR','shared')
        for epoch in (24,8):
            scouts={name:chunk(q,active,name,'scout_C_logits','shared-atmost-exact',epoch=epoch,logits=True) for name in ('UTILITY','ERROR')}
            logits=np.tile(np.array([[1.,0.,0.]],dtype=np.float32),(len(c),1))
            for mode in ('atmost','exact'):
                controls={}
                for name in (('UTILITY','ERROR','DENSE_ERROR_FLIP','MAGNET') if mode=='atmost' else ('UTILITY','ERROR')):
                    if K[q]:
                        if name=='UTILITY':scores=helper.utility_score(logits)
                        elif name=='ERROR':scores=helper.error_score(logits)
                        elif name=='DENSE_ERROR_FLIP':scores=helper.dense_error_flip_score(np.zeros(len(c),dtype=np.bool_),logits)
                        else:scores=helper.magnet_score(L[q].ravel().numpy()[c.numpy()],np.zeros(len(c),dtype=np.float32))
                        score=np.zeros((2,3),dtype=np.float64);score.ravel()[c.numpy()]=scores
                        domain=np.zeros((2,3),dtype=np.bool_);domain.ravel()[c.numpy()]=True
                        selected=torch.from_numpy(np.flatnonzero(helper.choose_candidates(L[q].numpy(),domain,score,K[q],mode=mode)).astype(np.int64))
                    else:selected=torch.empty(0,dtype=torch.int64)
                    mask=np.packbits((L[q].numpy()>0).ravel().astype(np.uint8),bitorder='little')
                    controls[name]=dict(actual_selected_indices=selected,actual_mask_packbits=torch.from_numpy(mask),
                        mask_shape=[2,3],mask_bits_count=6,mask_bitorder='little',
                        C_scout_chunks=[] if name=='MAGNET' else scouts['UTILITY' if name=='UTILITY' else 'ERROR'],
                        selected_PR_chunks=chunk(q,selected,name,'selected_PR',mode,epoch=epoch) if name in ('UTILITY','ERROR') else [],
                        logits_diagnostic_reuse=True)
                core=dict(schema='N16-actual-GPU-query-core-and-physical-chunk-refs-v2',query_id=q,epoch=epoch,evalmode=mode,
                    shared_actual_image_inputs=shared_ref,K=K[q],common_C_PR_chunks=common,controls=controls)
                profiles.append(dict(query_id=q,epoch=epoch,evalmode=mode,core=put(core)))
    baseline=[dict(image_id=91,category_id=1,query_id=q,score=.2+q/1000,
        segmentation=synthetic_segmentation(np,None,(L[q].numpy()>0).astype(np.uint8))) for q in shared['native_export_query_ids']]
    return shared_ref,physical,binding,profiles,baseline


def rejects(call):
    try:call()
    except (ValueError,RuntimeError,KeyError):return
    raise AssertionError('Malformed capture was accepted')


def main():
    torch.set_num_threads(2);outside=torch.get_rng_state().clone()
    shared_ref,physical,binding,profiles,baseline=fixture();shared=physical[shared_ref['path']]
    def loader(ref):return physical[ref['path']]
    def formula(F,e,L):return torch.zeros_like(L),L.clone()
    starts=[]
    outputs,details=adapter.replay_image(torch,np,cv2,None,types.SimpleNamespace(segmentation=synthetic_segmentation),
        gpu,proof,helper,dict(original_head=formula,independent_PR=formula),shared,shared_ref,profiles,loader,binding,
        baseline,copy.deepcopy(baseline),synthetic_only=True,query_start_callback=starts.append)
    assert len(outputs)==12 and all(v==baseline for v in outputs.values())
    assert len(details)==100 and starts==list(range(100)) and torch.equal(outside,torch.get_rng_state())
    assert details[1]['fixed_profiles'][0]['proof']['K']==0 and details[1]['fixed_profiles'][0]['proof']['C_points']==6
    assert details[2]['fixed_profiles'][0]['proof']['K']==0 and details[2]['fixed_profiles'][0]['proof']['C_points']==0
    rejects(lambda:adapter.profile_index(profiles[:-1]))
    duplicate=profiles.copy();duplicate[-1]=copy.deepcopy(duplicate[-2]);rejects(lambda:adapter.profile_index(duplicate))
    cores=adapter.profile_index(profiles)[1];ref=cores[(0,24,'atmost')]['core'];original=physical[ref['path']]
    def decode():return gpu.decode_CPU_query_core(torch,np,physical[ref['path']],shared,shared_ref,loader,binding,synthetic_only=True)
    decoded=decode()
    proof.proof_query(torch,np,cv2,helper,decoded,dict(original_head=formula,independent_PR=formula))
    for mutate in (
        lambda c:c['controls']['UTILITY']['actual_mask_packbits'].__setitem__(0,255),
        lambda c:c['controls']['UTILITY'].__setitem__('mask_bitorder','big'),
        lambda c:c.__setitem__('epoch',8),
    ):
        physical[ref['path']]=copy.deepcopy(original);mutate(physical[ref['path']]['value']);rejects(decode)
    physical[ref['path']]=original
    wrong=copy.deepcopy(decoded);wrong['controls']['UTILITY']['actual_selected_indices'][0]=0
    rejects(lambda:proof.proof_query(torch,np,cv2,helper,wrong,dict(original_head=formula,independent_PR=formula)))
    def wrong_sign(F,e,L):
        u=torch.zeros_like(L);u[L>0]=-4.;return u,L+u
    rejects(lambda:proof.proof_query(torch,np,cv2,helper,decoded,dict(original_head=wrong_sign,independent_PR=formula)))
    assert not torch.cuda.is_initialized()
    print('SYNTHETIC_ONLY: full100/twelve-policy/RLE-order/K0-empty +7 rejection cases PASS; no COCO/science qualification')


if __name__=='__main__':main()
