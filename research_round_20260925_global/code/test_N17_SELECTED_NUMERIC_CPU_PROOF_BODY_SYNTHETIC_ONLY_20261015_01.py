"""SYNTHETIC_ONLY: one tiny capture-to-CPU-core integration, no actual SCI/ref/model/data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
import cv2
from scipy import ndimage
from pycocotools import mask as masks
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path(__file__).resolve().parent;PUBLIC=ROOT
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load('_n17_selected_CPU_synthetic',ROOT/'N17_SELECTED_NUMERIC_CPU_PROOF_BODY_UNBOUND_DRAFT_20261015_01.py')
evalbody=load('_n17_eval_selected_synthetic',ROOT/'N17_ALL100_LOCAL_CENTER_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py')
kernel=load('_n17_neighbor_selected_synthetic',ROOT/'N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
endpoint=load('_n17_endpoint_selected_synthetic',ROOT/'N17_PAIRED_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261015_01.py')
helper=load('_n16_selection_selected_synthetic',PUBLIC/'N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')
legacy=load('_n16_eval_selected_synthetic',PUBLIC/'N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py')
proof=load('_n16_PRproof_selected_synthetic',PUBLIC/'N16_ALL100_SCOUT_SELECTOR_CPU_PROOF_BODY_UNBOUND_DRAFT_20261014_01.py')
base=load('_n16_base_selected_synthetic',PUBLIC/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
def rle(mask):
    value=masks.encode(np.asfortranarray(mask.astype(np.uint8)));return dict(size=value['size'],counts=value['counts'].decode('ascii'))
class PR(torch.nn.Module):
    def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(.25))
    def forward(self,F,e,L,D,arm='PR'):
        assert arm=='PR' and D.shape==(len(L),49) and not bool(D.any())
        return dict(residual=torch.full_like(L,float(self.value)))
class SelectedCoreSynthetic(unittest.TestCase):
    def test_actual_capture_array_hooks_gpu_ranking_masks_and_required_rejections(self):
        native=torch.full((100,3,4),-9.,dtype=torch.float32)
        native[0]=torch.tensor([[-1.,0.,3.,9.],[-2.,2.,-9.,-3.],[.5,-.5,4.,-4.]])
        e=torch.arange(100*256,dtype=torch.float32).reshape(100,256)/100
        F=torch.arange(256*12,dtype=torch.float32).reshape(256,3,4)/1000;union=np.arange(12,dtype=np.int64);unique=F.flatten(1).t().contiguous()
        models={}
        for epoch in (24,8):
            for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                head=kernel.make_scout(torch,202610150171,base.make_scout)
                with torch.no_grad():
                    for parameter in head.parameters():parameter.zero_()
                    if arm=='LOCAL_ERROR':head.hidden.weight[0,513:]=.125;head.output.weight[0,0]=1.
                    head.output.bias[2]=0. if epoch==24 else 2.
                models[(epoch,arm)]=head.requires_grad_(False).eval()
        pr=PR().requires_grad_(False).eval();events=[]
        value=next(evalbody.evaluate_all100(torch,np,ndimage,helper,legacy,kernel,native,e,union,unique,models,pr,
            point_callback=lambda event:events.append({key:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for key,v in event.items()}),synthetic_only=True))
        def prblock(event):return {key:event[key] for key in ('flat_indices','sampled_F','e0','native_L','GPU_raw_residual','GPU_combined')}
        common=[prblock(event) for event in events if event['stage']=='full_C_PR']
        chunks={(epoch,arm):[tuple(event[key] for key in ('flat_indices','sampled_F','e0','native_L','extra8','actual_GPU_logits')) for event in events
            if event['stage']=='scout_C_logits' and event['scout_epoch']==epoch and event['control']==arm] for epoch,arm in models}
        def iterator(epoch,arm):return iter(chunks[(epoch,arm)])
        def profile(epoch,arm,mode):
            record=value['profiles']['E%d_%s_%s'%(epoch,arm,mode)]
            subset=[prblock(event) for event in events if event['stage']=='selected_PR' and event['scout_epoch']==epoch and event['control']==arm and event['evalmode']==mode]
            return torch.from_numpy(record['selected_flat_indices']),subset,torch.from_numpy(record['mask']),rle(record['mask'])
        def original(F,e,L):
            output=pr(F,e,L,L.new_zeros((len(L),49)),arm='PR');return output['residual'],L+output['residual']
        def independent(F,e,L):
            u=torch.ones_like(L)/4;return u,L+u
        formulas=dict(original_head=original,independent_PR=independent);states={key:{name:v.clone() for name,v in head.state_dict().items()} for key,head in models.items()}
        def run(iterator=iterator,profile=profile):return core.proof_query(torch,np,cv2,helper,proof,kernel,endpoint,0,native,e[0],
            torch.from_numpy(value['C_flat_indices']),value['K'],models,states,common,iterator,profile,formulas,rle)
        result=run();self.assertEqual(len(result['profiles']),8);self.assertTrue(result['all_selected_indices_exact'])
        self.assertFalse(result['CPU_scout_outputs_used_for_ranking']);self.assertFalse(result['ROOT_scientific_qualification_conferred'])
        badchunks=copy.deepcopy(chunks);wrong=badchunks[(24,'LOCAL_ERROR')][0];wrong[4][0,0]+=.125
        with self.assertRaises(ValueError):run(iterator=lambda epoch,arm:iter(badchunks[(epoch,arm)]))
        badchunks=copy.deepcopy(chunks);badchunks[(24,'CENTER_ERROR')][0][5][0,0]+=.002
        with self.assertRaises(ValueError):run(iterator=lambda epoch,arm:iter(badchunks[(epoch,arm)]))
        def wrong_indices(epoch,arm,mode):
            indices,blocks,mask,encoded=profile(epoch,arm,mode)
            if epoch==24 and arm=='CENTER_ERROR' and mode=='exact':indices=indices.flip(0)
            return indices,blocks,mask,encoded
        with self.assertRaises(ValueError):run(profile=wrong_indices)
        def wrong_RLE(epoch,arm,mode):
            indices,blocks,mask,encoded=profile(epoch,arm,mode);return indices,blocks,mask,dict(encoded,counts='bad')
        with self.assertRaises(ValueError):run(profile=wrong_RLE)
        def wrong_sign(epoch,arm,mode):
            indices,blocks,mask,encoded=profile(epoch,arm,mode);blocks=copy.deepcopy(blocks)
            if epoch==24 and arm=='CENTER_ERROR' and mode=='exact':blocks[0]['GPU_combined'][0]=torch.tensor(1.)
            return indices,blocks,mask,encoded
        with self.assertRaises(ValueError):run(profile=wrong_sign)
if __name__=='__main__':unittest.main(verbosity=2)
