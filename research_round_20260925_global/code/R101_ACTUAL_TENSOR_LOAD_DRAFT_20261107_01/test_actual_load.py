"""SYNTHETIC fake tensors/modules only: these tests do not run PyTorch."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from actual_load import load_backbone_verified


def sibling_fixture():
    root=Path(__file__).resolve().parent.parent/'R101_EXPLICIT_BACKBONE_COVERAGE_DRAFT_20261107_01'
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
        return result
    coverage=module('coverage_check',root/'coverage_check.py')
    fixture=module('_synthetic_independent_R101_fixture',root/'test_coverage.py')
    return coverage,fixture.synthetic_r101_metadata


COVERAGE,FIXTURE=sibling_fixture()


class Flag:
    def __init__(self,value):self.value=value
    def all(self):return self
    def item(self):return self.value


class FakeTensor:
    """A logical shape with one constant backing cell; never allocates model tensors."""
    def __init__(self,shape,value=0,dtype='torch.float32',cell=None):
        self.shape=tuple(shape);self.dtype=dtype;self.cell=[value] if cell is None else cell
        self.device=SimpleNamespace(type='cpu');self.layout='strided';self.is_quantized=False
        self.requires_grad=False
    def numel(self):return math.prod(self.shape)
    def detach(self):return FakeTensor(self.shape,dtype=self.dtype,cell=self.cell)
    def clone(self):return FakeTensor(self.shape,self.cell[0],self.dtype)
    def copy_(self,value):self.cell[0]=value.cell[0];return self


class FakeTorch:
    Tensor=FakeTensor
    strided='strided'
    def __init__(self):self.rng=0
    def get_rng_state(self):return FakeTensor([1],self.rng,'torch.uint8')
    def equal(self,a,b):return a.shape==b.shape and a.cell[0]==b.cell[0]
    def isfinite(self,t):return Flag(math.isfinite(t.cell[0]))
    class no_grad:
        def __enter__(self):return None
        def __exit__(self,*args):return False


class FakeModel:
    def __init__(self,state,torch,mode=None):
        self.state=state;self.torch=torch;self.mode=mode;self.load_calls=0
        self.parameters={k:v for k,v in state.items() if k.endswith('weight')}
        for value in self.parameters.values():value.requires_grad=True
        self.buffers={k:v for k,v in state.items() if k not in self.parameters}
        self.buffers['pixel_mean']=FakeTensor([3,1,1],123)
    def state_dict(self):return {k:v.detach() for k,v in self.state.items()}
    def named_parameters(self,remove_duplicate=False):return list(self.parameters.items())
    def named_buffers(self,remove_duplicate=False):return list(self.buffers.items())
    def load_state_dict(self,state,strict=True,assign=False):
        assert strict is True and assign is False
        assert set(state)==set(self.state)
        self.load_calls+=1
        if self.mode=='mutate_unloaded_payload':state['head.weight'].cell[0]=99
        if self.mode=='mutate_mapped_payload':state['backbone.bottom_up.stem.conv1.weight'].cell[0]=99
        for key,value in state.items():self.state[key].copy_(value)
        if self.mode=='cast_schedule':self.state['schedule'].dtype='torch.float32'
        if self.mode=='rng':self.torch.rng+=1
        if self.mode=='replace_parameter':
            self.state['head.weight']=self.state['head.weight'].clone()
            self.state['head.weight'].requires_grad=True
            self.parameters['head.weight']=self.state['head.weight']
        if self.mode=='requires_grad':self.parameters['head.weight'].requires_grad=False
        if self.mode=='source_mutation':self.original_source['fc.bias'].cell[0]=9
        if self.mode=='nonpersistent_buffer':self.buffers['pixel_mean'].cell[0]=0
        if self.mode=='offset':self.state['backbone.bottom_up.res4.22.conv2_offset.bias'].cell[0]=9
        if self.mode=='missing_result':return SimpleNamespace(missing_keys=['head.weight'],unexpected_keys=[])
        return SimpleNamespace(missing_keys=[],unexpected_keys=[])


def synthetic_objects(mode=None):
    source_meta,target_meta=FIXTURE();torch=FakeTorch()
    source={k:FakeTensor(v['shape'],2,v['dtype']) for k,v in source_meta.items()}
    target={k:FakeTensor(v['shape'],0,v['dtype']) for k,v in target_meta.items()}
    target['head.weight']=FakeTensor([2],7)
    target['schedule']=FakeTensor([3],0.25,'torch.float64')
    model=FakeModel(target,torch,mode)
    return torch,model,source


class ActualLoadTests(unittest.TestCase):
    def test_valid_full_logical_R101_transfers_only_backbone(self):
        torch,model,source=synthetic_objects()
        result=load_backbone_verified(torch,model,source,COVERAGE)
        self.assertTrue(result['actual_tensor_load_checked'])
        self.assertEqual(result['mapped_keys'],520)
        self.assertEqual(result['mapped_state_numel'],42605504)
        self.assertEqual(result['unloaded_state_keys'],68)
        self.assertEqual(model.state['head.weight'].cell[0],7)
        self.assertEqual(model.state['schedule'].dtype,'torch.float64')
        self.assertEqual(model.state['backbone.bottom_up.res4.22.conv2_offset.weight'].cell[0],0)
        self.assertFalse(result['source_provenance_verified'])
        self.assertEqual(result['unloaded_target_backbone_offset_keys'],66)
        self.assertTrue(result['CPU_Torch_RNG_unchanged'])
        self.assertIn('not bitwise',result['equality_semantics'])

    def test_silent_unloaded_float64_cast_is_rejected_even_when_equal_values(self):
        torch,model,source=synthetic_objects('cast_schedule')
        with self.assertRaisesRegex(ValueError,'dtype'):
            load_backbone_verified(torch,model,source,COVERAGE)

    def test_nonfinite_excluded_source_or_unloaded_state_fails_before_loading(self):
        for location in ('source_fc','model_head'):
            with self.subTest(location=location):
                torch,model,source=synthetic_objects()
                (source['fc.bias'] if location=='source_fc' else model.state['head.weight']).cell[0]=float('nan')
                with self.assertRaisesRegex(ValueError,'finite'):
                    load_backbone_verified(torch,model,source,COVERAGE)
                self.assertEqual(model.load_calls,0)

    def test_RNG_consumption_or_parameter_identity_and_trainability_change_is_rejected(self):
        for mode,message in (('rng','RNG'),('replace_parameter','parameter'),('requires_grad','parameter')):
            with self.subTest(mode=mode):
                torch,model,source=synthetic_objects(mode)
                with self.assertRaisesRegex(ValueError,message):
                    load_backbone_verified(torch,model,source,COVERAGE)

    def test_source_state_remains_independent_and_unchanged(self):
        torch,model,source=synthetic_objects('source_mutation');model.original_source=source
        with self.assertRaisesRegex(ValueError,'Source tensor'):
            load_backbone_verified(torch,model,source,COVERAGE)

    def test_nonpersistent_buffer_outside_state_dict_is_also_preserved(self):
        torch,model,source=synthetic_objects('nonpersistent_buffer')
        with self.assertRaisesRegex(ValueError,'buffer'):
            load_backbone_verified(torch,model,source,COVERAGE)

    def test_independent_snapshots_detect_payload_and_offset_mutations(self):
        for mode in ('mutate_unloaded_payload','mutate_mapped_payload','offset'):
            with self.subTest(mode=mode):
                torch,model,source=synthetic_objects(mode)
                with self.assertRaisesRegex(ValueError,'Tensor values'):
                    load_backbone_verified(torch,model,source,COVERAGE)

    def test_input_must_be_plain_CPU_strided_unquantized_tensors(self):
        class SubTensor(FakeTensor):pass
        for case in ('subclass','foreign','cuda','layout','quantized'):
            with self.subTest(case=case):
                torch,model,source=synthetic_objects()
                value=source['conv1.weight']
                if case=='subclass':source['conv1.weight']=SubTensor(value.shape,2)
                if case=='foreign':source['conv1.weight']=object()
                if case=='cuda':value.device.type='cuda'
                if case=='layout':value.layout='sparse_coo'
                if case=='quantized':value.is_quantized=True
                with self.assertRaisesRegex(ValueError,'tensor|Tensor'):
                    load_backbone_verified(torch,model,source,COVERAGE)
                self.assertEqual(model.load_calls,0)

    def test_wrong_source_dtype_or_both_sides_wrong_shape_fails_before_load(self):
        for case in ('dtype','shape'):
            with self.subTest(case=case):
                torch,model,source=synthetic_objects()
                if case=='dtype':source['bn1.weight'].dtype='torch.float64'
                else:
                    source['conv1.weight'].shape=(64,3,3,7)
                    model.state['backbone.bottom_up.stem.conv1.weight'].shape=(64,3,3,7)
                with self.assertRaisesRegex(ValueError,'Architecture'):
                    load_backbone_verified(torch,model,source,COVERAGE)
                self.assertEqual(model.load_calls,0)

    def test_strict_loader_report_is_not_ignored(self):
        torch,model,source=synthetic_objects('missing_result')
        with self.assertRaisesRegex(ValueError,'Strict'):
            load_backbone_verified(torch,model,source,COVERAGE)

    def test_no_scientific_libraries_were_imported(self):
        self.assertFalse(any(key.split('.')[0] in ('torch','numpy') for key in sys.modules))


if __name__=='__main__':unittest.main()
