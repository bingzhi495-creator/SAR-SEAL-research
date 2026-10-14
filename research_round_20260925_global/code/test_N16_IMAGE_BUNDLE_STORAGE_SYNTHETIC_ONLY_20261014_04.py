"""Ordinary synthetic-only actual-format bundle test. No science/model/CUDA."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import cv2

SOURCE=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_04.py')


class SpyTorch:
    def __init__(self):self.writes=self.reads=0
    def __getattr__(self,name):return getattr(torch,name)
    def save(self,*args,**kwargs):self.writes+=1;return torch.save(*args,**kwargs)
    def load(self,*args,**kwargs):self.reads+=1;return torch.load(*args,**kwargs)


def fixture(root):
    spec=importlib.util.spec_from_file_location('ordinary_bundle_fixture',str(SOURCE))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    binding={'SYNTHETIC_ONLY':True};spy=SpyTorch();hashes=[];original=m.reference
    def counted(path):hashes.append(str(path));return original(path)
    m.reference=counted
    budget=dict(maximum_total_raw_capture_payload_bytes=1024**3,maximum_process_RSS_bytes=16*1024**3,
        maximum_capture_files=187,maximum_serialized_capture_bytes=76772540416,
        serialized_shared_image_overhead_bytes=1048576,serialized_query_core_overhead_bytes=1048576,
        serialized_chunk_overhead_bytes=65536)
    sink=m.ImageBundleCaptureSink(spy,root,{'capture_budget':budget},lambda:None,
        SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=64*1024**2)),binding)
    sink.begin_image(0,17)
    L=torch.zeros(100,2,3);L[0].fill_(3.);L[1].fill_(-3.)
    allidx=torch.arange(6,dtype=torch.int64);empty=torch.empty(0,dtype=torch.int64)
    shared=dict(schema='N16-actual-GPU-image-shared-inputs-v1',dataset_index=0,image_id=17,
        identity={'SYNTHETIC_ONLY':True},native_packet={'SYNTHETIC_ONLY':True},selector={'SYNTHETIC_ONLY':True},
        natural_query_ids=list(range(100)),native_export_query_ids=list(range(100)),
        union_flat_indices=allidx,F_unique=torch.arange(6*256,dtype=torch.float32).reshape(6,256),
        e0=torch.arange(100*256,dtype=torch.float32).reshape(100,256),native_L=L,
        C_indices=[allidx,empty]+[allidx]*98,C_counts=[6,0]+[6]*98,K_counts=[0,0]+[6]*98,U=6,
        actual_GPU_F_captured=True,CPU_F_rerendered=False)
    sid=sink.write('i000/SHARED_ACTUAL_INPUTS.pth',shared)
    cores=[];omega=[]
    def chunk(q,control,stage,mode,epoch=None):
        v=dict(query_id=q,control=control,stage=stage,evalmode=mode,point_chunk=2048,chunk_first=0,
            flat_indices=allidx.clone(),actual_input_scope='caller_actual_original_device_FP32_F_e_L',
            synthetic_only=True,shared_actual_image_inputs=sid,actual_observed_F_e_L_bitexact_to_shared=True)
        if epoch is not None:v['scout_epoch']=epoch
        if stage=='scout_C_logits':v['scout_logits']=torch.tensor([1.+epoch*.001,0.,-1.]).repeat(6,1)
        else:v.update(GPU_raw_residual=torch.full((6,),.1),GPU_combined=L[q].ravel()+.1)
        return sink.write(m.expected_bundle_member_id(0,v),v)
    for q in range(100):
        active=q>=2;common=[chunk(q,'COMMON_C_REFERENCE','full_C_PR','shared')] if active else []
        omega.extend([chunk(q,'STRONG_FIXED_E24_OMEGA','original_Omega_PR','atmost')] if active else [])
        scouts={(epoch,arm):[chunk(q,arm,'scout_C_logits','shared-atmost-exact',epoch)] if active else []
                for epoch in (8,24) for arm in ('UTILITY','ERROR')}
        for epoch in (8,24):
            for mode in ('atmost','exact'):
                records={}
                for control in ('UTILITY','ERROR','DENSE_ERROR_FLIP','MAGNET') if mode=='atmost' else ('UTILITY','ERROR'):
                    selected=[chunk(q,control,'selected_PR',mode,epoch)] if active and control in ('UTILITY','ERROR') else []
                    mask=np.ones((2,3),dtype=np.bool_) if active or q==0 else np.zeros((2,3),dtype=np.bool_)
                    records[control]=dict(actual_selected_indices=allidx.clone() if active else empty.clone(),
                        actual_mask_packbits=torch.from_numpy(np.packbits(mask.ravel(),bitorder='little')),
                        mask_shape=[2,3],mask_bits_count=6,mask_bitorder='little',
                        C_scout_chunks=[] if control=='MAGNET' else scouts[(epoch,'UTILITY' if control=='UTILITY' else 'ERROR')],
                        selected_PR_chunks=selected,logits_diagnostic_reuse=control=='DENSE_ERROR_FLIP' or mode=='exact')
                core=dict(schema='N16-actual-GPU-query-core-and-bundle-member-IDs-v3',query_id=q,epoch=epoch,evalmode=mode,
                    shared_actual_image_inputs=sid,K=6 if active else 0,common_C_PR_chunks=common,controls=records)
                cores.append(sink.write(m.expected_bundle_member_id(0,core),core))
    physical,locators,stats=sink.finish_image()
    bundle=torch.load(physical['path'],map_location='cpu')
    return m,bundle,binding,cores,omega,sid,physical,locators,stats,spy,hashes


class BundleStorageRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cv2.setNumThreads(2)
        cls.temp=tempfile.TemporaryDirectory(prefix='N16_BUNDLE_SYNTHETIC_ONLY_',dir=str(SOURCE.parent))
        assert Path(cls.temp.name).resolve().parent==SOURCE.parent.resolve()
        cls.data=fixture(cls.temp.name)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_one_real_save_readback_hash_and_complete_member_coverage(self):
        m,b,binding,cores,omega,sid,ref,locators,stats,spy,hashes=self.data
        self.assertEqual((spy.writes,spy.reads,len(hashes)),(1,1,1));self.assertEqual(set(ref),{'path','bytes','sha256'})
        self.assertEqual(set(locators),set(b['members']));self.assertTrue(stats['one_actual_write_readback_hash'])
        m.validate_CPU_image_bundle(torch,np,cv2,b,binding,synthetic_only=True)
        used=set()
        for core in cores:
            packet=m.decode_CPU_query_core_bundle(torch,np,b,core,binding,member_use=used,synthetic_only=True)
            self.assertEqual(set(packet),{'schema','query_id','evalmode','native_L','e0','K','C_flat_indices','common_C_PR_chunks','controls'})
            self.assertEqual(packet['e0'].shape,(256,))
            for record in packet['controls'].values():
                self.assertEqual(record['actual_mask'].dtype,torch.bool)
                for block in record['C_scout_chunks']+record['selected_PR_chunks']:
                    self.assertTrue(m.same_float_bits(torch,block['sampled_F'],b['members'][sid]['F_unique']))
        for member in omega:m.lookup_CPU_bundle_member(torch,b,member,binding,used)
        self.assertEqual(used,set(b['member_registry']));self.assertEqual(len(cores),400)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_registry_role_or_unknown_member_rejects(self):
        m,b,binding,cores,*_=self.data
        broken=copy.deepcopy(b);broken['member_registry'][cores[0]]['epoch']=99
        with self.assertRaises(RuntimeError):m.lookup_CPU_bundle_member(torch,broken,cores[0],binding)
        with self.assertRaises(RuntimeError):m.lookup_CPU_bundle_member(torch,b,{'path':'virtual'},binding)

    def test_cross_endpoint_reference_and_mask_padding_reject(self):
        m,b,binding,cores,*_=self.data
        core=next(c for c in cores if '/q002/e24_atmost_' in c);broken=copy.deepcopy(b)
        record=broken['members'][core]['controls']['UTILITY']
        record['C_scout_chunks']=[record['C_scout_chunks'][0].replace('e24_','e8_')]
        with self.assertRaises(RuntimeError):m.decode_CPU_query_core_bundle(torch,np,broken,core,binding,synthetic_only=True)
        broken=copy.deepcopy(b);broken['members'][core]['controls']['UTILITY']['actual_mask_packbits'][-1]|=128
        with self.assertRaises(RuntimeError):m.decode_CPU_query_core_bundle(torch,np,broken,core,binding,synthetic_only=True)

    def test_missing_member_or_unbound_completed_source_reject(self):
        m,b,binding,cores,*_=self.data
        broken=copy.deepcopy(b);del broken['members'][cores[0]]
        with self.assertRaises(RuntimeError):m.validate_CPU_image_bundle(torch,np,cv2,broken,binding,synthetic_only=True)
        with self.assertRaises(RuntimeError):m.lookup_CPU_bundle_member(torch,b,cores[0],{'DIFFERENT_BINDING':True})


if __name__=='__main__':unittest.main()
