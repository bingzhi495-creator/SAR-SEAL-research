"""SYNTHETIC_ONLY: one tiny CPU NPY-to-four-argument paired-training integration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path(__file__).resolve().parent
PUBLIC=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
body=load('_n17_train_synthetic',ROOT/'N17_PAIRED_LOCAL_CENTER_ERROR_TRAIN_BODY_UNBOUND_DRAFT_20261015_01.py')
kernel=load('_n17_features_synthetic',ROOT/'N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
old=load('_n16_train_synthetic',PUBLIC/'N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
base=load('_n16_scout_synthetic',PUBLIC/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
class PairedIntegration(unittest.TestCase):
    def test_actual_npy_common_error_labels_four_arguments_and_fixed_endpoints(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_TRAIN_',dir=ROOT) as directory:
            path=Path(directory).resolve();self.assertEqual(path.parent,ROOT.resolve())
            native=torch.zeros((100,2,3),dtype=torch.float32)
            native[0]=torch.tensor([[-9.,-3.,2.],[7.,10.,1.]])
            native[1]=torch.tensor([[4.,-7.,3.],[9.,-2.,-10.]])
            q=torch.tensor([0,1,0,1,0,1]);flat=torch.tensor([0,5,4,1,2,3])
            L=native.flatten(1)[q,flat].clone();extra=kernel.neighbor_features(torch,native,q,flat).numpy()
            target=path/'synthetic_neighbors.npy';np.save(target,extra,allow_pickle=False)
            rng=torch.Generator(device='cpu').manual_seed(811)
            labels=torch.tensor([0,1,2,0,1,2]);utility=torch.tensor([0,1,2,1,0,2])
            packet=dict(sampled_F_unique=torch.randn((6,256),generator=rng),native_e0=torch.randn((1,100,256),generator=rng),
                sample_to_unique=torch.arange(6),samples=dict(query_ids=q,flat_indices=flat,native_L=L,
                importance_weights=torch.tensor([.5,1.,2.,.5,1.,2.]),error_labels=labels,utility_labels=utility,
                stratum_ids=torch.tensor([0,1,0,1,0,1],dtype=torch.uint8),delta=utility!=2))
            original=copy.deepcopy(packet);reads=[]
            def loader(index):
                self.assertEqual(index,0);reads.append(target)
                return dict(packet=packet,neighbors8=np.load(target,allow_pickle=False))
            global_rng=torch.get_rng_state().clone()
            state=body.make_paired_state(torch,kernel,base.make_scout,old,device='cpu')
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
            self.assertEqual(tuple(state['heads']),('LOCAL_ERROR','CENTER_ERROR'))
            left,right=state['heads'].values()
            self.assertEqual(sum(p.numel() for p in left.parameters()),16803)
            for key,value in left.state_dict().items():self.assertTrue(torch.equal(value,right.state_dict()[key]))
            self.assertTrue(torch.equal(left.hidden.weight[:,513:],torch.zeros((32,8))))
            self.assertNotEqual(left.hidden.weight.data_ptr(),right.hidden.weight.data_ptr())
            center=(L.clamp(-8,8)/8).unsqueeze(1).repeat(1,8)
            self.assertTrue(torch.equal(left(packet['sampled_F_unique'],packet['native_e0'][0,q],L,torch.from_numpy(extra)),
                right(packet['sampled_F_unique'],packet['native_e0'][0,q],L,center)))
            observed={arm:[] for arm in state['heads']}
            hooks=[]
            for arm,head in state['heads'].items():
                def capture(module,args,arm=arm):
                    self.assertEqual(len(args),4);observed[arm].append(args[3].detach().clone())
                hooks.append(head.register_forward_pre_hook(capture))
            endpoints=[]
            result=body.train_until(torch,state,loader,last_epoch=24,image_count=1,
                checkpoint_callback=lambda epoch,endpoint,snapshot:endpoints.append((epoch,endpoint,snapshot)))
            self.assertEqual([v[0] for v in endpoints],[8,24]);self.assertEqual(len(reads),26)
            self.assertEqual(result['updates'],{'LOCAL_ERROR':24,'CENTER_ERROR':24})
            self.assertEqual(state['sampled_points'],144);self.assertEqual(len(state['history']),24)
            self.assertEqual(state['last_order_rng']['seed'],202610140160001+24)
            for tensor in observed['LOCAL_ERROR']:self.assertTrue(torch.equal(tensor,torch.from_numpy(extra)))
            for tensor in observed['CENTER_ERROR']:self.assertTrue(torch.equal(tensor,center))
            self.assertFalse(torch.equal(left.hidden.weight,right.hidden.weight))
            for epoch,endpoint,snapshot in endpoints:
                self.assertEqual(snapshot['initialization_seed'],202610150171)
                self.assertEqual(snapshot['updates'],{'LOCAL_ERROR':epoch,'CENTER_ERROR':epoch})
                self.assertEqual(len(snapshot['history']),epoch)
                for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                    self.assertEqual(len(snapshot['models'][arm]),4)
                    self.assertEqual(len(snapshot['optimizers'][arm]['state']),4)
                    for value in snapshot['optimizers'][arm]['state'].values():self.assertEqual(float(value['step']),epoch)
                    self.assertIsNotNone(endpoint['arms'][arm]['mean_image_weighted_CE'])
            for key in ('sampled_F_unique','native_e0','sample_to_unique'):self.assertTrue(torch.equal(packet[key],original[key]))
            for key in packet['samples']:self.assertTrue(torch.equal(packet['samples'][key],original['samples'][key]))
            self.assertTrue(torch.equal(native[0],torch.tensor([[-9.,-3.,2.],[7.,10.,1.]])))
            for hook in hooks:hook.remove()
if __name__=='__main__':unittest.main(verbosity=2)
