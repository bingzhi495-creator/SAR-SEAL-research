# UNBOUND_DRAFT / SYNTHETIC_ONLY / NOT_EXECUTED: byte-comparison regression.
# ROOT alone executes RED on ordinary tail01, then GREEN on a separate tail02.
import argparse
import importlib.util
import io
from pathlib import Path
import sys
import unittest

_parser=argparse.ArgumentParser(add_help=False)
_parser.add_argument('--tail',default=str(Path(__file__).with_name('N20_FIXEDDEV_FEATURE_ERROR_GPU_RUN_TAIL_UNBOUND_DRAFT_20261015_01.py')))
_options,_remaining=_parser.parse_known_args()
_tail_path=Path(_options.tail).resolve(strict=True)
if _tail_path.parent != Path(__file__).resolve(strict=True).parent or _tail_path.name not in (
    'N20_FIXEDDEV_FEATURE_ERROR_GPU_RUN_TAIL_UNBOUND_DRAFT_20261015_01.py',
    'N20_FIXEDDEV_FEATURE_ERROR_GPU_RUN_TAIL_UNBOUND_DRAFT_20261015_02.py'):
    raise ValueError('Only the two ordinary tmp tail versions may be loaded')
_spec=importlib.util.spec_from_file_location('_N20_ordinary_byte_tree_helpers',_tail_path)
_tail=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tail)

class ByteExactReadback(unittest.TestCase):
    def test_empty_zero_stride_scalar_and_real_serialization(self):
        import numpy as np
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        # No random fixtures, task inputs, checkpoint, file IO or CUDA allocation.
        fixtures={
            'empty_int64_numpy':torch.from_numpy(np.empty((0,),dtype=np.int64)),
            'empty_float64_numpy':torch.from_numpy(np.empty((0,),dtype=np.float64)),
            'singleton_int64_stride0':torch.tensor(7,dtype=torch.int64).expand(1),
            'singleton_float64_stride0':torch.tensor(.125,dtype=torch.float64).expand(1),
            'scalar_int64':torch.tensor(4,dtype=torch.int64),
            'scalar_float64':torch.tensor(-.25,dtype=torch.float64),
            'normal_int64':torch.arange(6,dtype=torch.int64).reshape(2,3),
            'normal_float32':torch.tensor([[.5,-2.],[3.,4.]],dtype=torch.float32),
            'noncontiguous_float64':torch.arange(6,dtype=torch.float64).reshape(2,3).t(),
        }
        self.assertEqual(fixtures['empty_int64_numpy'].numel(),0)
        self.assertEqual(fixtures['empty_float64_numpy'].numel(),0)
        self.assertEqual(fixtures['singleton_int64_stride0'].stride(),(0,))
        self.assertEqual(fixtures['singleton_float64_stride0'].stride(),(0,))
        for name,value in fixtures.items():
            with self.subTest(case=name):
                self.assertTrue(_tail.n20_same_bits(torch,value,value.clone()))
        positive=torch.tensor([0.],dtype=torch.float32)
        negative=torch.tensor([-0.],dtype=torch.float32)
        self.assertTrue(_tail.n20_same_bits(torch,positive,positive.clone()))
        self.assertTrue(_tail.n20_same_bits(torch,negative,negative.clone()))
        self.assertFalse(_tail.n20_same_bits(torch,positive,negative))
        self.assertFalse(_tail.n20_same_bits(torch,torch.tensor([1,2]),torch.tensor([[1,2]])))
        self.assertFalse(_tail.n20_same_bits(torch,torch.tensor([1.],dtype=torch.float32),torch.tensor([0x3f800000],dtype=torch.int32)))
        payload=dict(fixtures=fixtures,nested=[dict(zero=negative),('scope',torch.tensor(9,dtype=torch.int64))],truth=True)
        buffer=io.BytesIO()
        torch.save(payload,buffer)
        buffer.seek(0)
        replay=torch.load(buffer,map_location='cpu')
        self.assertTrue(_tail.n20_tree_equal(torch,payload,replay))
        replay['nested'][0]['zero']=positive
        self.assertFalse(_tail.n20_tree_equal(torch,payload,replay))
        self.assertFalse(_tail.n20_tree_equal(torch,[positive],(positive,)))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]]+_remaining)
