"""SYNTHETIC_ONLY: one tiny union-input/full-grid oracle integration, no SCI/data/GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy import ndimage
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path(__file__).resolve().parent;PUBLIC=ROOT
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
body=load('_n17_DEV_synthetic',ROOT/'N17_ALL100_LOCAL_CENTER_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py')
helper=load('_n16_score_synthetic',PUBLIC/'N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')
legacy=load('_n16_eval_synthetic',PUBLIC/'N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py')
kernel=load('_n17_neighbor_DEV_synthetic',ROOT/'N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
base=load('_n16_base_DEV_synthetic',PUBLIC/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
class PR(torch.nn.Module):
    def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(.25));self.calls=0
    def forward(self,F,e,L,detail,arm='PR'):
        self.calls+=1;assert arm=='PR' and detail.shape==(len(L),49)
        return dict(residual=L.new_full(L.shape,float(self.value)))
class TinyDeployment(unittest.TestCase):
    def test_union_neighbor_four_models_and_both_fixed_modes_against_full_grid(self):
        L=torch.full((4,3,4),-9.,dtype=torch.float32)
        L[0]=torch.tensor([[-1.,0.,3.,9.],[-2.,2.,-9.,-3.],[.5,-.5,4.,-4.]])
        L[1]=torch.tensor([[9.,-9.,2.,-2.],[1.,-1.,5.,-5.],[.25,-.25,3.,-3.]])
        L[2].fill_(9.)  # K0 but nonempty C; q3 is K0 empty C.
        F=torch.arange(256*12,dtype=torch.float32).reshape(256,3,4)/1000;e=torch.arange(4*256,dtype=torch.float32).reshape(4,256)/100
        union=np.array([1,5,10],dtype=np.int64);small=F.flatten(1)[:,torch.from_numpy(union)].t().contiguous()
        duplicate=np.array([5,1,5],dtype=np.int64)
        self.assertTrue(torch.equal(body.gather_F(torch,np,union,small,duplicate),F.flatten(1)[:,torch.from_numpy(duplicate)].t()))
        with self.assertRaises(ValueError):body.gather_F(torch,np,union,small,np.array([0],dtype=np.int64))
        masks=[legacy.domains_from_native(np,ndimage,L[q].numpy())[2] for q in range(4)]
        union=np.flatnonzero(np.logical_or.reduce(masks)).astype(np.int64);unique=F.flatten(1)[:,torch.from_numpy(union)].t().contiguous()
        models={}
        for epoch in (24,8):
            for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                head=kernel.make_scout(torch,202610150171,base.make_scout)
                with torch.no_grad():
                    for parameter in head.parameters():parameter.zero_()
                    if not (epoch==8 and arm=='CENTER_ERROR'):
                        head.hidden.weight[0,513:]=.125;head.output.weight[0,0]=1.
                    head.output.bias[2]=2. if epoch==8 else 0.
                head.requires_grad_(False);head.eval();models[(epoch,arm)]=head
        pr=PR().requires_grad_(False).eval();events=[]
        values=list(body.evaluate_all100(torch,np,ndimage,helper,legacy,kernel,L,e,union,unique,models,pr,
            point_callback=events.append,synthetic_only=True))
        self.assertEqual([v['query_id'] for v in values],list(range(4)))
        scout_events=[v for v in events if v['stage']=='scout_C_logits']
        self.assertEqual(len(scout_events),8) # 2 active queries *4 models, not extra atmost/exact forwards.
        for value in values:
            q=value['query_id'];native=L[q].numpy();P,Omega,C,indices,K=legacy.domains_from_native(np,ndimage,native)
            self.assertEqual(len(value['profiles']),8)
            for epoch in (24,8):
                for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                    if K:
                        flat=torch.from_numpy(indices);qid=torch.full(flat.shape,q,dtype=torch.int64)
                        extra=kernel.neighbor_features_numpy(np,L.numpy(),qid.numpy(),indices) if arm=='LOCAL_ERROR' else np.repeat((np.clip(native.ravel()[indices],-8,8)/8)[:,None],8,axis=1)
                        with torch.no_grad():logits=models[(epoch,arm)](F.flatten(1)[:,flat].t(),e[q].unsqueeze(0).expand(len(flat),256),L[q].flatten()[flat],torch.from_numpy(extra))
                        score=helper.error_score(logits.numpy());score_grid=np.zeros(native.shape,dtype=np.float64);score_grid.ravel()[indices]=score
                    for mode in ('atmost','exact'):
                        name='E%d_%s_%s'%(epoch,arm,mode);record=value['profiles'][name]
                        selected=helper.choose_candidates(native,C,score_grid,K,mode=mode) if K else np.zeros(native.shape,dtype=np.bool_)
                        expected=P.copy();expected[selected]=(native[selected]+np.float32(.25))>0
                        self.assertTrue(np.array_equal(record['selected_flat_indices'],np.flatnonzero(selected)))
                        self.assertTrue(np.array_equal(record['mask'],expected))
                        if K:self.assertTrue(torch.equal(record['actual_GPU_C_logits'],logits))
                        self.assertEqual(record['PR_compatibility']['strict_sign_exact'],True)
            if K:
                first=value['profiles']['E24_LOCAL_ERROR_atmost'];second=value['profiles']['E24_LOCAL_ERROR_exact']
                self.assertIs(first['actual_GPU_C_logits'],second['actual_GPU_C_logits'])
            else:
                for record in value['profiles'].values():self.assertEqual(len(record['selected_flat_indices']),0)
        self.assertEqual(values[0]['profiles']['E8_CENTER_ERROR_atmost']['selected_flat_indices'].size,0)
        self.assertTrue(np.array_equal(values[0]['profiles']['E8_CENTER_ERROR_exact']['selected_flat_indices'],np.array([0,1,4,5,8,9])))
        for event in scout_events:
            q=event['query_id'];flat=event['flat_indices'].numpy()
            expected=kernel.neighbor_features_numpy(np,L.numpy(),np.full(len(flat),q,dtype=np.int64),flat) if event['control']=='LOCAL_ERROR' else np.repeat((np.clip(L[q].numpy().ravel()[flat],-8,8)/8)[:,None],8,axis=1)
            self.assertTrue(np.array_equal(event['extra8'].numpy().view(np.uint32),expected.view(np.uint32)))
            self.assertTrue(torch.equal(event['sampled_F'],F.flatten(1)[:,event['flat_indices']].t()))
            self.assertTrue(torch.equal(event['e0'],e[q]))
        with self.assertRaises(ValueError):list(body.evaluate_all100(torch,np,ndimage,helper,legacy,kernel,L,e,union[:-1],unique[:-1],models,pr,synthetic_only=True))
        with self.assertRaises(ValueError):list(body.evaluate_all100(torch,np,ndimage,helper,legacy,kernel,L,e,union,unique,models,pr,synthetic_only=False))
if __name__=='__main__':unittest.main(verbosity=2)
