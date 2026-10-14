"""CPU synthetic-only regression of ordered mappings; no real SCI/data/CUDA."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import torch

def module(filename):
    path=Path(__file__).with_name(filename)
    spec=importlib.util.spec_from_file_location(filename,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

old_body=module('N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_01.py')
new_body=module('N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
old_runner=module('N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_02.py')
new_runner=module('N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_03.py')
scout=module('N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')

def run():
    before=torch.get_rng_state().clone()
    rng=torch.Generator(device='cpu').manual_seed(431)
    source=OrderedDict(weight=torch.randn((3,4),generator=rng).requires_grad_(),
                       nested=OrderedDict(bias=torch.randn(4,generator=rng)))
    old=old_body._cpu_tree(torch,source)
    assert isinstance(old,OrderedDict) and old['weight'].requires_grad  # old RED witness
    visits=[];original_cpu=torch.Tensor.cpu
    def observed_cpu(self,*args,**kwargs):
        visits.append(tuple(self.shape));return original_cpu(self,*args,**kwargs)
    torch.Tensor.cpu=observed_cpu
    try:fixed=new_body._cpu_tree(torch,source)
    finally:torch.Tensor.cpu=original_cpu
    assert type(fixed) is dict and type(fixed['nested']) is dict
    assert sorted(visits)==[(3,4),(4,)] and fixed['weight'].device.type=='cpu'
    assert not fixed['weight'].requires_grad and fixed['weight'].data_ptr()!=source['weight'].data_ptr()
    assert fixed['nested']['bias'].data_ptr()!=source['nested']['bias'].data_ptr()
    lhs=OrderedDict(weight=source['weight'].detach().clone(),nested=fixed['nested'])
    rhs=OrderedDict(weight=source['weight'].detach().clone(),nested={'bias':fixed['nested']['bias'].clone()})
    try:old_runner.tree_equal(torch,lhs,rhs)
    except RuntimeError as error:assert 'Boolean value' in str(error)  # old RED witness
    else:raise AssertionError('Old ordered mapping comparison must expose tensor Boolean ambiguity')
    assert new_runner.tree_equal(torch,lhs,rhs)
    rhs['weight'][0,0]+=.1;assert not new_runner.tree_equal(torch,lhs,rhs)
    state=new_body.make_paired_state(torch,scout,device='cpu')
    snapshot=new_body.cpu_snapshot(torch,state)
    assert all(type(v) is dict for v in snapshot['models'].values())
    assert all(t.device.type=='cpu' and not t.requires_grad for v in snapshot['models'].values() for t in v.values())
    assert new_runner.tree_equal(torch,snapshot,snapshot)
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
    print('SYNTHETIC_CPU_ONLY_RED_GREEN_OK: ordered recursive CPU-detach-clone visits and tensor-safe exact readback; no GPU migration execution claim')

if __name__=='__main__':run()
