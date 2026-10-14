"""SYNTHETIC_ONLY tiny CPU reader/writer fixtures, no actual cache/GT/weights/science."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path(__file__).resolve().parent
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('_n17_body_synthetic','N17_FIT_NATIVE_L_NEIGHBOR_CACHE_BODY_UNBOUND_DRAFT_20261015_01.py')
kernel=load('_n17_kernel_synthetic','N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
def ref(path):
    p=Path(path).resolve();data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def fixture(root):
    pointer=lambda key:dict(path=str(root/(key+'.json')),bytes=0,sha256='0'*64)
    report=dict(source=pointer('source'),spec=pointer('spec'),GPU_admission=pointer('ad'),phase='SYNTHETIC',
        fixed_PR_checkpoint=pointer('PR'),teacher_checkpoint=pointer('teacher'))
    binding=dict(locked_protocol=pointer('protocol'));identity=dict(dataset_index=0,image_id=113,original_shape=[2,3],ann_ids=[39,40])
    L=torch.arange(600,dtype=torch.float32).reshape(100,2,3)/10;L[0,0,0]=-0.
    carrier=dict(schema='N16-actual-GPU-natural100-L-carrier-v1',**{k:report[k] for k in ('source','spec','GPU_admission','phase')},dataset_index=0,identity=identity,native_L=L)
    cpath=root/'carrier.pth';torch.save(carrier,cpath);cref=ref(cpath)
    q=torch.tensor([0,1,0,1],dtype=torch.int64);flat=torch.tensor([0,0,5,5],dtype=torch.int64)
    samples=dict(query_ids=q,flat_indices=flat,native_L=L.flatten(1)[q,flat],error_labels=torch.tensor([0,1,2,1]),importance_weights=torch.tensor([.5,1.,.5,1.]))
    packet=dict(**{k:report[k] for k in ('source','spec','GPU_admission','phase','fixed_PR_checkpoint','teacher_checkpoint')},
        dataset_index=0,identity=identity,protocol=binding['locked_protocol'],natural_native_L_carrier=cref,samples=samples)
    ppath=root/'packet.pth';torch.save(packet,ppath);pref=ref(ppath)
    row=dict(dataset_index=0,image_id=113,identity=copy.deepcopy(identity),packet=pref,natural_native_L_carrier=cref);events=[]
    def closed(r):
        if ref(r['path'])!=r:raise ValueError('Synthetic ref changed')
        return Path(r['path'])
    def read_torch(p):events.append(('read',p));return torch.load(str(p),map_location='cpu')
    def write_npy(index,array):
        events.append(('write',index));p=root/('out%d.npy'%index)
        with p.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        return ref(p)
    def read_npy(r):closed(r);return np.load(r['path'],allow_pickle=False)
    callbacks=dict(closed=closed,read_torch=read_torch,write_npy=write_npy,read_generated_npy=read_npy,write_manifest=lambda v:None,check=lambda:None)
    return report,binding,row,pref,cref,callbacks,events
class CacheBodyTests(unittest.TestCase):
    def test_full_row_identity_mismatch_reject_before_output(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_BAD_IDENTITY_',dir=ROOT) as directory:
            root=Path(directory).resolve();self.assertEqual(root.parent,ROOT.resolve())
            report,binding,row,pref,cref,callbacks,events=fixture(root)
            row['identity']['ann_ids']=[40,39]
            with self.assertRaises(ValueError):b.extract_image(torch,np,kernel,0,row,pref,cref,report,binding,callbacks,expected_samples=4)
            self.assertFalse(any(v[0]=='write' for v in events))
    def test_real_reader_npy_roundtrip_binding_and_original_samples_preserved(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_BODY_',dir=ROOT) as directory:
            root=Path(directory).resolve();self.assertEqual(root.parent,ROOT.resolve())
            report,binding,row,pref,cref,callbacks,events=fixture(root);old=torch.load(pref['path'],map_location='cpu')
            result=b.extract_image(torch,np,kernel,0,row,pref,cref,report,binding,callbacks,expected_samples=4)
            self.assertEqual([v[0] for v in events],['read','read','write']);self.assertEqual(result['points'],4)
            saved=callbacks['read_generated_npy'](result['neighbors8_ref']);carrier=torch.load(cref['path'],map_location='cpu')
            expected=kernel.neighbor_features_numpy(np,carrier['native_L'].numpy(),old['samples']['query_ids'].numpy(),old['samples']['flat_indices'].numpy())
            self.assertTrue(np.array_equal(saved.view(np.uint32),expected.view(np.uint32)))
            self.assertEqual(result['payload_bytes'],128);self.assertEqual(result['packet_ref'],pref)
            after=torch.load(pref['path'],map_location='cpu')
            for key in old['samples']:self.assertTrue(torch.equal(old['samples'][key],after['samples'][key]))
    def test_wrong_query_or_signed_center_binding_reject(self):
        for kind in ('query','center_bits'):
            with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_BAD_QUERY_',dir=ROOT) as directory:
                root=Path(directory).resolve();self.assertEqual(root.parent,ROOT.resolve())
                report,binding,row,pref,cref,callbacks,events=fixture(root);packet=torch.load(pref['path'],map_location='cpu')
                if kind=='query':packet['samples']['query_ids'][0]=2
                else:packet['samples']['native_L'][0]=0.
                torch.save(packet,pref['path']);pref=ref(pref['path']);row['packet']=pref
                with self.assertRaises(ValueError):b.extract_image(torch,np,kernel,0,row,pref,cref,report,binding,callbacks,expected_samples=4)
                self.assertFalse(any(v[0]=='write' for v in events))
    def test_output_order_and_native_carrier_ref_substitution_reject(self):
        for kind in ('order','carrierref'):
            with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_BAD_ORDER_',dir=ROOT) as directory:
                root=Path(directory).resolve();self.assertEqual(root.parent,ROOT.resolve())
                report,binding,row,pref,cref,callbacks,events=fixture(root)
                if kind=='order':
                    original=callbacks['read_generated_npy'];callbacks['read_generated_npy']=lambda r:original(r)[::-1].copy()
                else:
                    packet=torch.load(pref['path'],map_location='cpu');packet['natural_native_L_carrier']=dict(cref,sha256='1'*64)
                    torch.save(packet,pref['path']);pref=ref(pref['path']);row['packet']=pref
                with self.assertRaises(ValueError):b.extract_image(torch,np,kernel,0,row,pref,cref,report,binding,callbacks,expected_samples=4)
if __name__=='__main__':unittest.main(verbosity=2)
