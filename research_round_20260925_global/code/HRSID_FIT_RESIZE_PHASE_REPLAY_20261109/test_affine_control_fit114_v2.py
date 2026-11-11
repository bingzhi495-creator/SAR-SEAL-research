from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from affine_control_fit114_v2 import collect_arrays, forward_edges, float_center_indices


class AffineControlTests(unittest.TestCase):
    def test_scaled_crop_then_flip(self):
        trace = [
            dict(kind="resize", before=(2, 4), after=(2, 8)),
            dict(kind="crop", before=(2, 8), after=(2, 4), top=0, left=2),
            dict(kind="hflip", before=(2, 4), after=(2, 4)),
        ]
        matrix = forward_edges(trace)
        rows, cols = float_center_indices(matrix, (2, 4), match_ties=True)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(cols, [2, 2, 1, 1])

    def test_exact_half_ties_choose_larger_source_index(self):
        matrix = forward_edges([dict(kind="resize", before=(2, 4), after=(2, 2))])
        rows, cols = float_center_indices(matrix, (2, 2), match_ties=True)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(cols, [1, 3])


    def test_zero_retained_rows_keep_coordinate_witness_and_zero_mask_counts(self):
        from collections import Counter
        from fractions import Fraction
        from types import SimpleNamespace
        import torch

        raw = torch.zeros((0, 2, 4), dtype=torch.bool)
        trace = [dict(kind="resize", before=(2, 4), after=(2, 4))]
        target = dict(row=torch.empty((0,), dtype=torch.int64), size=torch.tensor([2,4]))
        unit = (Fraction(1),Fraction(1))
        zero = (Fraction(0),Fraction(0))
        ref = SimpleNamespace(source_indices=lambda *_: (np.arange(2), np.arange(4)),
                              _compose=lambda *_: (None,None,unit,zero))
        kornia = SimpleNamespace(warp_affine=lambda image,*_,**__: image)
        counts = Counter()
        collect_arrays(raw,target,trace,trace,ref=ref,one=lambda *_:dict(masks=raw),
                       row_id="row",torch=torch,kornia=kornia,tally=counts,cursor=0)
        self.assertEqual(counts["views"],1)
        self.assertEqual(counts["rows"],0)
        self.assertEqual(counts["kornia031_aligned_mask_xor"],0)


if __name__ == "__main__":
    unittest.main()
