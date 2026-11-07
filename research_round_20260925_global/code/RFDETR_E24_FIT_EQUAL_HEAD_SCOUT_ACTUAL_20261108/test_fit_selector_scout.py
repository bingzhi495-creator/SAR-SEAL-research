from unittest import TestCase
import numpy as np
from fit_selector_scout import apply_edits

class RowMajorFlipTest(TestCase):
    def test_flips_row_major_indices_in_a_fortran_decoded_mask(self):
        from types import SimpleNamespace
        mask=np.asfortranarray([[0,1,0],[1,0,0]],dtype=np.uint8)
        codec=SimpleNamespace(
            decode=lambda r: np.asfortranarray([int(x) for x in r["counts"].decode()]).reshape(r["size"]),
            encode=lambda m: dict(size=list(m.shape),counts="".join(map(str,m.flat)).encode()))
        original=dict(size=[2,3],counts="010100")
        got=apply_edits(np,codec,original,np.array([2,3],np.uint32))
        self.assertEqual(got,dict(size=[2,3],counts="011000"))
        self.assertEqual(original,dict(size=[2,3],counts="010100"))
