"""SYNTHETIC_ONLY: bundle->CPU adapter100query/12policies + rejection tests.

Constructed tiny tensors, in-memory member IDs only, no actual refs/I/O/GT/models.
Production-format synthetic_only=False is a field-format fixture, never a claim
of GPU provenance. Real whole-file Source/RAW/owner qualification is NOT tested.
RLE is the earlier local stand-in; actual pycocotools/COCO is NOT tested.
"""
import copy
import importlib.util
from pathlib import Path
import types
import torch
import numpy as np
import cv2


def module(filename):
    p=Path(__file__).with_name(filename);s=importlib.util.spec_from_file_location(filename,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


old=module('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_ADAPTER_SYNTHETIC_TEST_20261014_01.py')
gpu=module('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_04.py')
adapter=module('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_ADAPTER_UNBOUND_DRAFT_20261014_02.py')
proof=module('N16_ALL100_SCOUT_SELECTOR_CPU_PROOF_BODY_UNBOUND_DRAFT_20261014_01.py')
helper=module('N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')


def fixture():
    shared_ref,physical,binding,profiles,baseline=old.fixture()
    values={k:copy.deepcopy(v['value']) for k,v in physical.items()}
    shared=values[shared_ref['path']]
    for v in values.values():
        if v.get('schema')=='N16-actual-GPU-query-core-and-physical-chunk-refs-v2':
            v['schema']='N16-actual-GPU-query-core-and-bundle-member-IDs-v3'
        if 'synthetic_only' in v:v['synthetic_only']=False
        if 'GPU_raw_residual' in v:
            idx=v['flat_indices'];u=shared['F_unique'][idx,0]*.001
            v['GPU_raw_residual']=u;v['GPU_combined']=shared['native_L'][v['query_id']].ravel()[idx]+u
    mapping={k:gpu.expected_bundle_member_id(0,v) for k,v in values.items()}
    assert len(set(mapping.values()))==len(mapping)
    def replace(v):
        if type(v) is dict:
            if set(v)=={'path','bytes','sha256'}:return mapping[v['path']]
            return {k:replace(x) for k,x in v.items()}
        if type(v) is list:return [replace(x) for x in v]
        return v
    members={mapping[k]:replace(v) for k,v in values.items()}
    registry={k:gpu.bundle_descriptor(torch,v) for k,v in members.items()}
    bundle=dict(schema=gpu.BUNDLE_SCHEMA,binding=binding,dataset_index=0,image_id=91,
        members=members,member_registry=registry,member_count=len(members),
        raw_tensor_payload_bytes=sum(v['raw_tensor_payload_bytes'] for v in registry.values()))
    return bundle,binding,mapping[shared_ref['path']],replace(profiles),baseline


def main():
    torch.set_num_threads(2);outside=torch.get_rng_state().clone()
    bundle,binding,sid,profiles,baseline=fixture();used=set()
    def f1(F,e,L):
        u=F[:,0]*.001;return u,L+u
    def f2(F,e,L):
        u=torch.mul(F[:,0],.001);return u,torch.add(L,u)
    formulas=dict(original_head=f1,independent_PR=f2)
    def replay(b):
        return adapter.replay_image_bundle(torch,np,cv2,None,types.SimpleNamespace(segmentation=old.synthetic_segmentation),
            gpu,proof,helper,formulas,b,sid,profiles,binding,baseline,copy.deepcopy(baseline),member_use=used)
    output,details=replay(bundle)
    assert len(output)==12 and all(v==baseline for v in output.values()) and len(details)==100
    assert used==set(bundle['members'])==set(bundle['member_registry'])
    assert details[1]['fixed_profiles'][0]['proof']['K']==0 and details[1]['fixed_profiles'][0]['proof']['C_points']==6
    assert details[2]['fixed_profiles'][0]['proof']['K']==0 and details[2]['fixed_profiles'][0]['proof']['C_points']==0
    core_id=adapter.profile_index(profiles)[1][(0,24,'atmost')]['core']
    def decode(b):return gpu.decode_CPU_query_core_bundle(torch,np,b,core_id,binding)
    old.rejects(lambda:gpu.lookup_CPU_bundle_member(torch,bundle,{'path':'virtual'},binding))
    bad=copy.deepcopy(bundle);bad['member_registry'][core_id]['epoch']=99
    old.rejects(lambda:gpu.lookup_CPU_bundle_member(torch,bad,core_id,binding))
    bad=copy.deepcopy(bundle);del bad['members'][core_id]
    old.rejects(lambda:gpu.validate_CPU_image_bundle(torch,np,cv2,bad,binding))
    bad=copy.deepcopy(bundle)
    bad['members'][core_id]['controls']['UTILITY']['C_scout_chunks'][0]=bad['members'][core_id]['controls']['UTILITY']['C_scout_chunks'][0].replace('e24_','e8_')
    old.rejects(lambda:decode(bad))
    bad=copy.deepcopy(bundle);bad['members'][core_id]['controls']['UTILITY']['actual_mask_packbits'][0]|=128
    old.rejects(lambda:decode(bad))
    bad=copy.deepcopy(bundle);orphan=copy.deepcopy(next(v for v in bad['members'].values() if v.get('stage')=='original_Omega_PR'))
    orphan['chunk_first']=2048;oid=gpu.expected_bundle_member_id(0,orphan)
    bad['members'][oid]=orphan;bad['member_registry'][oid]=gpu.bundle_descriptor(torch,orphan)
    bad['member_count']+=1;bad['raw_tensor_payload_bytes']+=bad['member_registry'][oid]['raw_tensor_payload_bytes']
    old.rejects(lambda:gpu.validate_CPU_image_bundle(torch,np,cv2,bad,binding))
    bad=copy.deepcopy(bundle);bad['members'][sid]['F_unique'][0,0]+=100
    old.rejects(lambda:replay(bad))
    old.rejects(lambda:gpu.lookup_CPU_bundle_member(torch,bundle,core_id,{'WRONG_BINDING':True}))
    assert torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized()
    print('SYNTHETIC_ONLY: bundle full100/12policies/allmembers/K0+8 rejection cases PASS; no actual provenance/COCO/science qualification')


if __name__=='__main__':main()
