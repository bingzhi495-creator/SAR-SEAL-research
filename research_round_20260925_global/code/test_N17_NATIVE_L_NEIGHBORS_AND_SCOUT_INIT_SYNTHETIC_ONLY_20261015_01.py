"""SYNTHETIC_ONLY CPU random/hand fixtures; no SAR input, actual weights or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Synthetic test must start with cold CUDA')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load(Path(__file__).with_name('N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py'),'_n17_synthetic')
base=load(Path(__file__).with_name('N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py'),'_n16_published_synthetic_factory')
class NativeTests(unittest.TestCase):
    def test_four_corners_saturation_and_query_isolation(self):
        L=torch.tensor([[[-16.,-8.,-4.],[0.,4.,16.]],[[8.,8.,8.],[8.,8.,8.]]],dtype=torch.float32)
        q=torch.tensor([0,0,0,0,1],dtype=torch.int64);flat=torch.tensor([0,2,3,5,0],dtype=torch.int64)
        expected=torch.tensor([[-1,-1,-1,-1,-1,0,0,.5],[-1,-.5,-.5,-1,-.5,.5,1,1],
            [-1,-1,-1,0,.5,0,0,.5],[-1,-.5,-.5,.5,1,.5,1,1],[1]*8],dtype=torch.float32)
        value=m.neighbor_features(torch,L,q,flat)
        self.assertTrue(torch.equal(value,expected));self.assertEqual(value.dtype,torch.float32)
        self.assertTrue(np.array_equal(m.neighbor_features_numpy(np,L.numpy(),q.numpy(),flat.numpy()),expected.numpy()))
    def test_non_square_native_four_edge_coordinates(self):
        L=(torch.arange(15,dtype=torch.float32)-7).reshape(1,3,5)
        q=torch.zeros(4,dtype=torch.int64);flat=torch.tensor([1,13,5,9],dtype=torch.int64)
        expected=torch.tensor([[-7,-6,-5,-7,-5,-2,-1,0],[0,1,2,5,7,5,6,7],
            [-7,-7,-6,-2,-1,3,3,4],[-4,-3,-3,1,2,6,7,7]],dtype=torch.float32)/8
        self.assertTrue(torch.equal(m.neighbor_features(torch,L,q,flat),expected))
        self.assertTrue(np.array_equal(m.neighbor_features_numpy(np,L.numpy(),q.numpy(),flat.numpy()),expected.numpy()))
    def test_one_pixel_constant_center_equivalence_and_empty(self):
        for H,W in ((1,1),(2,5)):
            L=torch.full((2,H,W),-12.,dtype=torch.float32)
            q=torch.tensor([0,1],dtype=torch.int64);flat=torch.tensor([0,H*W-1],dtype=torch.int64)
            neighbor=m.neighbor_features(torch,L,q,flat);center=m.center_control_features(torch,L,q,flat)
            self.assertTrue(torch.equal(neighbor,center));self.assertTrue(torch.equal(center,torch.full((2,8),-1.)))
            self.assertTrue(np.array_equal(m.center_control_features_numpy(np,L.numpy(),q.numpy(),flat.numpy()),center.numpy()))
        empty=torch.empty(0,dtype=torch.int64)
        self.assertEqual(m.neighbor_features(torch,L,empty,empty).shape,(0,8))
    def test_index_bool_float_nonfinite_and_out_of_range_reject(self):
        L=torch.ones((2,2,3),dtype=torch.float32);q=torch.tensor([0],dtype=torch.int64);flat=torch.tensor([0],dtype=torch.int64)
        for bad in (torch.tensor([False]),torch.tensor([0.])):
            with self.assertRaises(ValueError):m.neighbor_features(torch,L,bad,flat)
            with self.assertRaises(ValueError):m.neighbor_features(torch,L,q,bad)
            with self.assertRaises(ValueError):m.neighbor_features_numpy(np,L.numpy(),bad.numpy(),flat.numpy())
            with self.assertRaises(ValueError):m.neighbor_features_numpy(np,L.numpy(),q.numpy(),bad.numpy())
        for qbad,f in ((torch.tensor([-1]),flat),(torch.tensor([2]),flat),(q,torch.tensor([6]))):
            with self.assertRaises(ValueError):m.neighbor_features(torch,L,qbad,f)
        bad=L.clone();bad[0,0,0]=float('nan')
        with self.assertRaises(ValueError):m.neighbor_features(torch,bad,q,flat)
    def test_exact_old_init_zero_columns_rng_and_parameter_contract(self):
        before=torch.get_rng_state().clone();old=base.make_scout(torch,17);new=m.make_scout(torch,17,base.make_scout)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(sum(p.numel() for p in new.parameters()),16803);self.assertEqual(len(new.state_dict()),4)
        self.assertFalse(list(new.named_buffers()))
        expected=m.lift_initial_state(torch,old.state_dict())
        for key,value in expected.items():self.assertTrue(torch.equal(value,new.state_dict()[key]),key)
        self.assertTrue(torch.equal(new.hidden.weight[:,:513],old.hidden.weight))
        self.assertEqual(torch.count_nonzero(new.hidden.weight[:,513:]).item(),0)
        other=m.make_scout(torch,17,base.make_scout)
        self.assertNotEqual(new.hidden.weight.data_ptr(),other.hidden.weight.data_ptr())
        for key,value in new.state_dict().items():self.assertTrue(torch.equal(value,other.state_dict()[key]))
    def test_lifted_forward_preserves_synthetic_old_function_and_no_AMP(self):
        rng=torch.Generator(device='cpu').manual_seed(123)
        F=torch.randn((4,256),generator=rng);e=torch.randn((4,256),generator=rng);L=torch.randn(4,generator=rng)
        extra=torch.randn((4,8),generator=rng);old=base.make_scout(torch,19);new=m.make_scout(torch,19,base.make_scout)
        a=old(F,e,L);b=new(F,e,L,extra)
        self.assertEqual(b.dtype,torch.float32);self.assertEqual(b.shape,(4,3))
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=0), 'Only synthetic initialization tolerance, no actual criterion')
        self.assertEqual(new(F[:0],e[:0],L[:0],extra[:0]).shape,(0,3))
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            with self.assertRaises(ValueError):new(F,e,L,extra)
if __name__=='__main__':unittest.main(verbosity=2)
