# UNBOUND_DRAFT / NOT_EXECUTED: one manual sparse-neighbor CPU regression, no scientific data.
import importlib.util
from pathlib import Path
import unittest

BODY = Path(__file__).with_name('N20_SELECTED_NUMERIC_CPU_PROOF_BODY_UNBOUND_DRAFT_20261015_01.py')
if not BODY.is_file():
    raise FileNotFoundError(BODY)
spec = importlib.util.spec_from_file_location('_N20_future_CPU_numeric_body', BODY)
body = importlib.util.module_from_spec(spec)
spec.loader.exec_module(body)

class SparseNeighborInput(unittest.TestCase):
    def test_real_two_by_three_neighbors_and_center_control(self):
        import torch
        import numpy as np
        self.assertFalse(torch.cuda.is_initialized())
        union = torch.tensor([0, 4], dtype=torch.int64)
        context = dict(union_flat_indices=union,
            F_unique=union[:, None].repeat(1, 256).float(),
            extra_flat_indices=torch.tensor([1, 2, 3, 5], dtype=torch.int64),
            extra_F=torch.tensor([1, 2, 3, 5], dtype=torch.float32)[:, None].repeat(1, 256),
            neighbor_to_combined=torch.tensor([[0,0,2,0,0,2,4,4,1], [0,2,3,4,1,5,4,1,5]], dtype=torch.int64))
        F, local = body.feature_neighbors(torch, np, context, union, 2, 3, 'LOCAL_ERROR')
        expected = torch.tensor([[0,0,1,0,0,1,3,3,4], [0,1,2,3,4,5,3,4,5]], dtype=torch.float32)
        self.assertTrue(torch.equal(local, expected[:, :, None].repeat(1, 1, 256)))
        self.assertTrue(torch.equal(F, context['F_unique']))
        _, center = body.feature_neighbors(torch, np, context, union, 2, 3, 'CENTER_ERROR')
        self.assertTrue(torch.equal(center, F[:, None].repeat(1, 9, 1)))
        wrong = dict(context, neighbor_to_combined=context['neighbor_to_combined'].clone())
        wrong['neighbor_to_combined'][0, 2] = 3
        with self.assertRaises(ValueError):
            body.feature_neighbors(torch, np, wrong, union, 2, 3, 'LOCAL_ERROR')
        wrong['neighbor_to_combined'] = context['neighbor_to_combined'].clone()
        wrong['neighbor_to_combined'][0, 4] = 1
        with self.assertRaises(ValueError):
            body.feature_neighbors(torch, np, wrong, union, 2, 3, 'CENTER_ERROR')
        self.assertFalse(torch.cuda.is_initialized())

if __name__ == '__main__':
    unittest.main()
