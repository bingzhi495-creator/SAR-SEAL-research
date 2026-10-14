"""SYNTHETIC_ONLY: tiny CPU formula/NPY and synthetic state-evidence ABI checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path(__file__).resolve().parent;PUBLIC=ROOT
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
qual=load('_n17_endpoint_CPU_synthetic',ROOT/'N17_PAIRED_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261015_01.py')
scout=load('_n17_scout_endpoint_synthetic',ROOT/'N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
train=load('_n17_train_endpoint_synthetic',ROOT/'N17_PAIRED_LOCAL_CENTER_ERROR_TRAIN_BODY_UNBOUND_DRAFT_20261015_01.py')
base=load('_n16_base_endpoint_synthetic',PUBLIC/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
old=load('_n16_train_endpoint_synthetic',PUBLIC/'N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
def expanded_evidence(snapshot,epoch):
    # Invented full-count metadata and scalar steps, not an executed trajectory or SAR result.
    result=copy.deepcopy(snapshot);result.update(completed_epoch=epoch,updates={a:741*epoch for a in qual.ARMS},
        visited_images=741*epoch,empty_images=0,sampled_points=741*4096*epoch,history=[],epoch_orders=[])
    trace=[];updates={a:0 for a in qual.ARMS}
    for number in range(1,epoch+1):
        rng=torch.Generator(device='cpu').manual_seed(202610140160001+number);before=rng.get_state().clone()
        order=torch.randperm(741,generator=rng).tolist();after=rng.get_state().clone();sums={a:0. for a in qual.ARMS}
        for index in order:
            arms={}
            for a,loss in zip(qual.ARMS,(.5,.75)):
                updates[a]+=1;arms[a]=dict(weighted_CE=loss,updates=updates[a]);sums[a]+=loss
            trace.append(dict(event='paired_image_update',epoch=number,dataset_index=index,points=4096,arms=arms))
        history=dict(epoch=number,initialization_seed=202610150171,order_seed=202610140160001+number,order=order,
            visited_images=741,updated_images=741,empty_images=0,points=3035136,online_sum_image_weighted_CE=sums,
            **{'online_mean_image_weighted_CE_'+a:sums[a]/741 for a in qual.ARMS})
        result['history'].append(history);result['epoch_orders'].append(order)
        result['last_order_rng']=dict(seed=202610140160001+number,before=before,after=after)
    for a in qual.ARMS:
        for value in result['optimizers'][a]['state'].values():value['step'].fill_(741*epoch)
    return result,trace
class EndpointBodySynthetic(unittest.TestCase):
    def test_two_formulas_real_NPY_and_exact_state_trace_contract(self):
        outside=torch.get_rng_state().clone();rng=torch.Generator(device='cpu').manual_seed(773)
        n=7;labels=torch.arange(n,dtype=torch.int64)%3
        packet=dict(sampled_F_unique=torch.randn((n,256),generator=rng),sample_to_unique=torch.arange(n),native_e0=torch.randn((1,100,256),generator=rng),
            samples=dict(query_ids=torch.arange(n),flat_indices=torch.arange(n)%6,native_L=torch.linspace(-10,10,n),error_labels=labels,utility_labels=labels,
            importance_weights=torch.tensor([.5,1.,2.,.5,1.,2.,1.]),stratum_ids=torch.zeros(n,dtype=torch.uint8),delta=labels!=2))
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_ENDPOINT_',dir=ROOT) as directory:
            path=Path(directory).resolve();self.assertEqual(path.parent,ROOT.resolve())
            npy=path/'neighbors.npy';np.save(npy,torch.randn((n,8),generator=rng).numpy(),allow_pickle=False)
            loader=lambda index:dict(packet=packet,neighbors8=np.load(npy,allow_pickle=False))
            state=train.make_paired_state(torch,scout,base.make_scout,old)
            train.train_one_epoch(torch,state,loader,1,image_count=1)
            actual=train.cpu_snapshot(torch,state)
            heads={a:scout.make_scout(torch,202610150171,base.make_scout) for a in qual.ARMS}
            for a in qual.ARMS:heads[a].load_state_dict(actual['models'][a]);heads[a].requires_grad_(False)
            image=qual.compare_image(torch,heads,actual['models'],loader(0))
            self.assertEqual(image['points'],n)
            for a in qual.ARMS:
                self.assertLessEqual(image['arms'][a]['max_logit_absolute_error'],1e-4)
                self.assertAlmostEqual(image['arms'][a]['factory'],image['arms'][a]['independent'],places=6)
            observed=dict(arms={a:dict(mean_image_weighted_CE=image['arms'][a]['factory']) for a in qual.ARMS})
            qual.compare_recorded_means(image['arms'],observed,1)
            bad=copy.deepcopy(observed);bad['arms'][qual.ARMS[0]]['mean_image_weighted_CE']+=.002
            with self.assertRaises(RuntimeError):qual.compare_recorded_means(image['arms'],bad,1)
            rows=[dict(dataset_index=i,points=4096) for i in range(741)]
            for epoch in (8,24):
                snapshot,trace=expanded_evidence(actual,epoch)
                counts=qual.validate_state_evidence(torch,scout,base.make_scout,snapshot,rows,epoch)
                self.assertEqual(counts['updates'],741*epoch);self.assertEqual(counts['sampled_points'],3035136)
                self.assertTrue(qual.validate_trace(trace,snapshot,rows))
                bad=copy.deepcopy(snapshot);next(iter(bad['optimizers'][qual.ARMS[0]]['state'].values()))['step'].fill_(741*epoch+.25)
                with self.assertRaises(RuntimeError):qual.validate_state_evidence(torch,scout,base.make_scout,bad,rows,epoch)
            bad=copy.deepcopy(snapshot);next(iter(bad['optimizers'][qual.ARMS[1]]['state'].values()))['exp_avg_sq'].flatten()[0]=-1
            with self.assertRaises(RuntimeError):qual.validate_state_evidence(torch,scout,base.make_scout,bad,rows,24)
            badtrace=copy.deepcopy(trace);badtrace[0]['arms'][qual.ARMS[0]]['updates']=999
            with self.assertRaises(RuntimeError):qual.validate_trace(badtrace,snapshot,rows)
            with self.assertRaises(RuntimeError):qual.validate_state_evidence(torch,scout,base.make_scout,snapshot,rows[:-1],24)
        self.assertTrue(torch.equal(outside,torch.get_rng_state()))
if __name__=='__main__':unittest.main(verbosity=2)
