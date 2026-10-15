# UNBOUND_DRAFT: NOT_EXECUTED; tiny synthetic numeric-reader behavior only.
import copy
import importlib.util
from pathlib import Path
import unittest


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class TinyEndpointReader(unittest.TestCase):
    def test_nonzero_adapter_two_formulas_and_five_fractional_step_rejection(self):
        root=Path(__file__).resolve().parent
        body=load('n20_endpoint_body',root/'N20_PAIRED_FEATURE_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261015_01.py')
        adapter=load('n20_endpoint_adapter',root/'N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py')
        base=load('n20_endpoint_base',root/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
        import torch
        self.assertFalse(torch.cuda.is_initialized());torch.set_num_threads(2);torch.set_num_interop_threads(2)
        reference=base.make_scout(torch,202610150201)
        heads={a:adapter.create_head(torch) for a in body.ARMS}
        with torch.no_grad():
            for head in heads.values():
                for key,value in reference.state_dict().items():head.state_dict()[key].copy_(value)
                head.spatial_weight[0,0]=.25
                head.requires_grad_(False);head.eval()
        models={a:{k:v.detach().clone() for k,v in head.state_dict().items()} for a,head in heads.items()}
        first=torch.linspace(-1,1,256,dtype=torch.float32);second=first.flip(0).contiguous()
        unique=torch.stack((first,second));slots=torch.tensor([1,0],dtype=torch.int64)
        e=torch.linspace(1,-1,256,dtype=torch.float32)
        sample=dict(query_ids=torch.tensor([1,0],dtype=torch.int64),flat_indices=torch.tensor([4,0],dtype=torch.int64),
            native_L=torch.tensor([-.5,.5],dtype=torch.float32),importance_weights=torch.tensor([.5,2.],dtype=torch.float32),
            utility_labels=torch.tensor([2,2],dtype=torch.int64),error_labels=torch.tensor([0,1],dtype=torch.int64))
        packet=dict(sampled_F_unique=unique,native_e0=e.reshape(1,1,256).repeat(1,100,1),sample_to_unique=slots,samples=sample)
        mapping=torch.tensor([[2,0,0,0,0,0,0,0,0],[2,1,1,1,1,1,1,1,1]],dtype=torch.int64)
        loaded=dict(packet=packet,neighbor_packet=dict(extra_F=torch.ones((1,256),dtype=torch.float32),neighbor_to_combined=mapping))
        result=body.compare_image(torch,heads,models,loaded)
        self.assertEqual(result['points'],2)
        F=torch.stack((second,first));embedded=e[None].repeat(2,1)
        effective_local=F.clone();effective_local[:,0]+=.25
        effective_center=F.clone();effective_center[:,0]+=.25*F[:,0]
        for arm,effective in (('LOCAL_ERROR',effective_local),('CENTER_ERROR',effective_center)):
            with torch.no_grad():logits=reference(effective,embedded,sample['native_L'])
            target=float((sample['importance_weights']*torch.nn.functional.cross_entropy(logits,sample['error_labels'],reduction='none')).mean())
            self.assertAlmostEqual(result['arms'][arm]['factory'],target,delta=1e-6)
            self.assertLessEqual(result['arms'][arm]['max_logit_absolute_error'],1e-4)
        # Synthetic optimizer evidence: FP32 scalar carrier with integral17784 is valid.
        optimizers={}
        for arm,head in heads.items():
            groups=torch.optim.AdamW(head.parameters(),**body.ADAMW).state_dict()['param_groups']
            values={ident:dict(step=torch.tensor(17784.,dtype=torch.float32),
                exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
                for ident,parameter in zip(groups[0]['params'],head.parameters())}
            optimizers[arm]=dict(param_groups=groups,state=values)
        body.validate_optimizer_evidence(torch,adapter,base,optimizers,17784)
        bad=copy.deepcopy(optimizers)
        first_id=bad['LOCAL_ERROR']['param_groups'][0]['params'][0]
        bad['LOCAL_ERROR']['state'][first_id]['step']=torch.tensor(17784.5,dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            body.validate_optimizer_evidence(torch,adapter,base,bad,17784)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
