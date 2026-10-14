"""CPU SYNTHETIC_ONLY endpoint audit; no science payloads/refs/IO/CUDA."""
import copy
import importlib.util
from pathlib import Path
import torch

def module(filename):
    path=Path(__file__).with_name(filename)
    spec=importlib.util.spec_from_file_location(filename,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

qualifier=module('N16_PAIRED_SCOUT_TRAIN_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261014_01.py')
train=module('N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
scout=module('N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')

def fixture(n,seed):
    rng=torch.Generator(device='cpu').manual_seed(seed)
    u=min(n,4);labels=torch.randint(3,(n,),generator=rng)
    return dict(sampled_F_unique=torch.randn((u,256),generator=rng),
        sample_to_unique=torch.arange(n,dtype=torch.int64)%max(u,1),
        native_e0=torch.randn((1,100,256),generator=rng),
        samples=dict(query_ids=torch.randint(100,(n,),generator=rng),native_L=torch.randn(n,generator=rng),
            importance_weights=torch.arange(n,dtype=torch.float32).remainder(2)+.5,
            utility_labels=labels,error_labels=(labels+1).remainder(3),
            stratum_ids=torch.arange(n).remainder(2).to(torch.uint8),delta=labels!=2))

def rejects(function):
    try:function()
    except RuntimeError:return
    raise AssertionError('Invalid synthetic evidence must be rejected')

def run():
    before=torch.get_rng_state().clone()
    packets=[fixture(7,21),fixture(0,22),fixture(5,23)]
    rows=[dict(dataset_index=i,points=len(p['samples']['query_ids'])) for i,p in enumerate(packets)]
    state=train.make_paired_state(torch,scout,device='cpu');initial=train.cpu_snapshot(torch,state)
    snapshots={};endpoints={};trace=[]
    train.train_until(torch,state,lambda i:packets[i],image_count=3,observer=trace.append,
        checkpoint_callback=lambda epoch,endpoint,snapshot:(snapshots.update({epoch:snapshot}),endpoints.update({epoch:endpoint})))
    qualifier.validate_initial(torch,scout,initial)
    qualifier.validate_trace(trace,snapshots[24],rows)
    for epoch in (8,24):
        result=qualifier.validate_endpoint(torch,scout,scout,initial,snapshots[epoch],endpoints[epoch],lambda i:packets[i],rows,epoch)
        assert result['qualified_checks_only'] and result['actual_updates_per_arm']==epoch*2
        assert result['sampled_points']==12 and result['GPU_pointwise_logits_qualified'] is False
    bad=copy.deepcopy(snapshots[24]);next(iter(bad['optimizers']['UTILITY']['state'].values()))['step'].fill_(48.25)
    rejects(lambda:qualifier.validate_state_evidence(torch,scout,bad,rows,24))
    bad=copy.deepcopy(snapshots[8]);next(iter(bad['optimizers']['ERROR']['state'].values()))['exp_avg_sq'].flatten()[0]=-1
    rejects(lambda:qualifier.validate_state_evidence(torch,scout,bad,rows,8))
    bad=copy.deepcopy(snapshots[8]);bad['epoch_orders'][0]=list(reversed(bad['epoch_orders'][0]))
    rejects(lambda:qualifier.validate_state_evidence(torch,scout,bad,rows,8))
    bad=copy.deepcopy(endpoints[24]);bad['arms']['UTILITY']['mean_image_weighted_CE']+=.002
    rejects(lambda:qualifier.validate_endpoint(torch,scout,scout,initial,snapshots[24],bad,lambda i:packets[i],rows,24))
    bad=copy.deepcopy(initial);bad['models']['ERROR']['output.bias'][0]+=.01
    rejects(lambda:qualifier.validate_initial(torch,scout,bad))
    bad=copy.deepcopy(trace);next(r for r in bad if r['event']=='paired_image_update')['arms']['UTILITY']['updates']=999
    rejects(lambda:qualifier.validate_trace(bad,snapshots[24],rows))
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
    print('SYNTHETIC_CPU_ONLY_OK: initial/E8/E24/two-logit-formulas/wCE/orders/empty/AdamW/noninteger-step/negative-moment/trace/aggregate-mismatch')

if __name__=='__main__':
    torch.set_num_threads(2);run()
