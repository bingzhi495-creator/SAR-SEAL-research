"""Actual CPU random fixtures only; no SAR images, scientific refs, caches or CUDA."""
import importlib.util
from pathlib import Path
import torch

def module(name):
    p = Path(__file__).with_name(name)
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

body = module('N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_01.py')
scout = module('N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')

def fixture(n, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    U = min(n, 4)
    labels = torch.randint(3, (n,), generator=g)
    return dict(sampled_F_unique=torch.randn((U,256), generator=g),
        sample_to_unique=torch.arange(n, dtype=torch.int64) % max(U,1),
        native_e0=torch.randn((1,100,256), generator=g),
        samples=dict(query_ids=torch.randint(100,(n,),generator=g),
            native_L=torch.randn(n,generator=g), importance_weights=torch.ones(n),
            utility_labels=labels.clone(), error_labels=labels.clone(),
            stratum_ids=torch.arange(n,dtype=torch.int64).remainder(2).to(torch.uint8),
            delta=labels!=2))

def run():
    before = torch.get_rng_state().clone()
    logits=torch.tensor([[1.,0.,-1.],[0.,0.,0.]])
    labels=torch.tensor([0,2]);w=torch.tensor([2.,.5])
    expected=(2*torch.nn.functional.cross_entropy(logits[:1],labels[:1])+
              .5*torch.nn.functional.cross_entropy(logits[1:],labels[1:]))/2
    loss=body.weighted_cross_entropy(torch,logits,labels,w)
    assert torch.equal(loss,expected)
    assert abs(float(loss)-float((w*torch.nn.functional.cross_entropy(logits,labels,reduction='none')).sum()/w.sum()))>.01
    packets=[fixture(13,11),fixture(0,12),fixture(7,13)]
    state=body.make_paired_state(torch,scout,device='cpu')
    calls=[]
    def loader(i):
        calls.append(i)
        return packets[i]
    report=body.train_one_epoch(torch,state,loader,epoch=1,image_count=3)
    gen=torch.Generator(device='cpu').manual_seed(body.ORDER_SEED+1)
    expected_order=torch.randperm(3,generator=gen).tolist()
    assert calls==report['order']==expected_order
    assert report['updated_images']==2 and report['empty_images']==1 and report['points']==20
    assert state['updates']=={'UTILITY':2,'ERROR':2}
    assert all(torch.equal(a,b) for a,b in zip(state['heads']['UTILITY'].state_dict().values(),state['heads']['ERROR'].state_dict().values()))
    endpoint=body.evaluate_fixed(torch,state,lambda i:packets[i],image_count=3)
    assert endpoint['nonempty_images']==2 and endpoint['sampled_points']==20
    assert endpoint['arms']['UTILITY']['mean_image_weighted_CE']==endpoint['arms']['ERROR']['mean_image_weighted_CE']
    saved=body.cpu_snapshot(torch,state)
    assert all(t.device.type=='cpu' for values in saved['models'].values() for t in values.values())
    assert all(float(row['step'])==2 for opt in saved['optimizers'].values() for row in opt['state'].values())
    for packet in packets:
        packet['samples']['error_labels'] = (packet['samples']['utility_labels'] + 1).remainder(3)
    body.train_one_epoch(torch,state,loader,epoch=2,image_count=3)
    assert state['updates']=={'UTILITY':4,'ERROR':4}
    assert any(not torch.equal(a,b) for a,b in zip(state['heads']['UTILITY'].state_dict().values(),state['heads']['ERROR'].state_dict().values()))
    empty_state=body.make_paired_state(torch,scout,device='cpu')
    checkpoint_epochs=[]
    result=body.train_until(torch,empty_state,lambda i:packets[1],image_count=1,
        checkpoint_callback=lambda epoch,endpoint,snapshot:checkpoint_epochs.append(epoch))
    assert checkpoint_epochs==[8,24] and result['completed_epoch']==24
    assert result['updates']=={'UTILITY':0,'ERROR':0} and empty_state['empty_images']==24
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    print('SYNTHETIC_CPU_ONLY_OK: wCE/init/paired updates/arm labels/order/empty skip/fixed8-24/endpoint/optimizer snapshot/global RNG')

if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
