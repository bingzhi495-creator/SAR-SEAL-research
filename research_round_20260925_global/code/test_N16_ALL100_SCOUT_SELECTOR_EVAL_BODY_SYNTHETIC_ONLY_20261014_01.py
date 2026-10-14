"""Small CPU-only synthetic tests; no scientific file/data/model weights."""
import importlib.util
import inspect
import os
from pathlib import Path
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from scipy import ndimage
torch.set_num_threads(2)

BASE = Path('E:/SAR/seal')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load('n16_public_helper', BASE / 'github_release_20261011/research_round_20260925_global/code/N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')
body = load('n16_eval_body', BASE / 'tmp/N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py')


class Scout(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('values', torch.tensor(logits, dtype=torch.float32))
        self.calls = []
        self.eval()

    def forward(self, F, e, L):
        self.calls.append(len(L))
        return self.values.expand(len(L), 3).clone()


class PR(torch.nn.Module):
    def __init__(self, u=3., subset_shift=0., full_size=None):
        super().__init__()
        self.u, self.subset_shift, self.full_size = u, subset_shift, full_size
        self.calls = []
        self.eval()

    def forward(self, F, e, L, distances, arm):
        assert arm == 'PR' and distances.shape == (len(L), 49)
        self.calls.append(len(L))
        shift = self.subset_shift if self.full_size is not None and len(L) < self.full_size else 0.
        return {'residual': L.new_full(L.shape, self.u + shift)}


def inputs(values):
    grid = torch.tensor(values, dtype=torch.float32)
    h, w = grid.shape
    F = torch.arange(256*h*w, dtype=torch.float32).reshape(256,h,w) / max(h*w,1)
    e = torch.arange(100*256, dtype=torch.float32).reshape(100,256)
    L = grid[None].expand(100,h,w).clone()
    return F, e, L


def run(F, e, L, utility=None, error=None, pr=None, **kwargs):
    utility = utility if utility is not None else Scout((5.,0.,0.))
    error = error if error is not None else Scout((5.,5.,0.))
    pr = pr if pr is not None else PR()
    stream = body.evaluate_all100_scout_selectors(torch, np, ndimage, helper,
        F, e, L, utility, error, pr, synthetic_only=True, **kwargs)
    return stream, utility, error, pr


class Tests(unittest.TestCase):
    def test_empty_C_and_positive_K0_are_native_without_calls(self):
        for value, expected in ((-20.,False),(20.,True)):
            stream, utility, error, pr = run(*inputs([[value,value]]))
            rows = list(stream)
            self.assertEqual([r['query_id'] for r in rows], list(range(100)))
            for row in rows:
                self.assertEqual(row['K'], 0)
                self.assertEqual(set(row['controls']), {'UTILITY','ERROR','DENSE_ERROR_FLIP','MAGNET'})
                for result in row['controls'].values():
                    self.assertTrue(np.all(result['mask'] == expected))
                    self.assertEqual(len(result['selected_flat_indices']), 0)
            self.assertEqual(utility.calls + error.calls + pr.calls, [])

    def test_cheap_selection_actual_mask_and_callback(self):
        F,e,L = inputs([[-3.,-1.,4.],[-3.,-3.,-3.]])
        before = tuple(t.clone() for t in (F,e,L))
        events = []
        stream, utility, error, pr = run(F,e,L, point_callback=events.append)
        row = next(stream)
        self.assertEqual((row['K'],row['C_points']), (1,6))
        self.assertEqual(pr.calls, [6,1,1])
        self.assertEqual(utility.calls, [6])
        self.assertEqual(error.calls, [6,6])  # ERROR and dense control each own forward.
        for name,result in row['controls'].items():
            np.testing.assert_array_equal(result['selected_flat_indices'], [1])
            self.assertTrue(result['mask'][0,1])
            self.assertTrue(result['mask'][0,2])
            self.assertEqual(result['native_selected_L'].dtype, np.float32)
            self.assertEqual(result['actual_combined'].dtype, np.float32)
            self.assertEqual(result['actual_combined'][0], 2.)
        self.assertEqual(row['controls']['UTILITY']['cost']['PR_calls'], 1)
        self.assertEqual(row['common_C']['cost']['PR_calls'], 1)
        self.assertTrue(row['shared_diagnostic_runtime_only'])
        self.assertFalse(row['standalone_total_time_measured'])
        self.assertTrue(any(e['stage'] == 'selected_PR' for e in events))
        for event in events:
            self.assertNotIn('GT', event)
            self.assertEqual(event['sampled_F'].dtype, torch.float32)
            self.assertEqual(event['sampled_F'].device.type, 'cpu')
        for old,new in zip(before,(F,e,L)):
            self.assertTrue(torch.equal(old,new))

    def test_nonpositive_atmost_empty_exact_real_support(self):
        args = inputs([[-3.,-1.,4.]])
        negative_u, negative_e = Scout((0.,5.,0.)), Scout((-100.,-100.,100.))
        row = next(run(*args, utility=negative_u,error=negative_e)[0])
        self.assertEqual(len(row['controls']['UTILITY']['selected_flat_indices']), 0)
        self.assertEqual(len(row['controls']['ERROR']['selected_flat_indices']), 0)
        exact = next(run(*args,utility=negative_u,error=negative_e,evalmode='exact')[0])
        self.assertEqual(set(exact['controls']), {'UTILITY','ERROR'})
        for result in exact['controls'].values():
            np.testing.assert_array_equal(result['selected_flat_indices'], [1])
            self.assertTrue(np.all(exact['C'].ravel()[result['selected_flat_indices']]))

    def test_absolute_tolerance_failure_no_fallback(self):
        args = inputs([[-3.,-1.,4.]])
        stream = run(*args,pr=PR(subset_shift=.001,full_size=3))[0]
        with self.assertRaises(body.NumericQualificationFailure) as caught:
            next(stream)
        self.assertFalse(caught.exception.details['residual_absolute_pass'])
        self.assertEqual(caught.exception.details['rtol'], 0.)

    def test_strict_sign_failure_even_with_small_absolute_error(self):
        args = inputs([[-3.,0.,4.]])
        stream = run(*args,pr=PR(u=0.,subset_shift=1.e-6,full_size=3))[0]
        with self.assertRaises(body.NumericQualificationFailure) as caught:
            next(stream)
        self.assertTrue(caught.exception.details['residual_absolute_pass'])
        self.assertTrue(caught.exception.details['combined_absolute_pass'])
        self.assertFalse(caught.exception.details['strict_sign_exact'])

    def test_chunks_2048_and_real_noops(self):
        args = inputs([[0.]*2050])
        stream,utility,error,pr = run(*args,pr=PR(u=0.))
        row = next(stream)
        self.assertEqual(pr.calls,[2048,2,2048,2,2048,2])
        self.assertEqual(utility.calls,[2048,2])
        self.assertEqual(error.calls,[2048,2,2048,2])
        self.assertEqual(len(row['controls']['UTILITY']['selected_flat_indices']),2050)
        self.assertFalse(np.any(row['controls']['UTILITY']['mask']))
        self.assertEqual(len(row['controls']['DENSE_ERROR_FLIP']['selected_flat_indices']),0)
        self.assertEqual(len(row['controls']['MAGNET']['selected_flat_indices']),0)

    def test_no_GT_api_and_input_validation(self):
        self.assertFalse(any('gt' in p.lower() for p in inspect.signature(body.evaluate_all100_scout_selectors).parameters))
        args = inputs([[-1.]])
        with self.assertRaises(TypeError):
            body.evaluate_all100_scout_selectors(torch,np,ndimage,helper,*args,Scout((1.,0.,0.)),Scout((1.,1.,0.)),PR(),GT=np.ones((1,1)))
        invalid = args[2].clone(); invalid[0,0,0] = float('nan')
        with self.assertRaises(ValueError):
            next(run(args[0],args[1],invalid)[0])
        scout = Scout((1.,0.,0.)); scout.train()
        with self.assertRaises(ValueError):
            next(run(*args,utility=scout)[0])

    def test_saturation_finite_stable_MagNet(self):
        args = inputs([[0.,40.,-40.]])
        row = next(run(*args,pr=PR(u=1.))[0])
        result = row['controls']['MAGNET']
        np.testing.assert_array_equal(result['selected_flat_indices'], [0])
        self.assertTrue(result['mask'][0,0])
        self.assertTrue(result['mask'][0,1])
        self.assertFalse(result['mask'][0,2])


if __name__ == '__main__':
    unittest.main()
