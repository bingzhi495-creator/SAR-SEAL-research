from types import SimpleNamespace
from unittest import TestCase
import numpy as np

from sparse_capture import pack_candidates

def encode(mask):
    return dict(size=list(mask.shape), counts=bytes(48+int(v) for v in mask.flat))

def decode(ref):
    return np.frombuffer(ref["counts"], dtype=np.uint8).reshape(ref["size"])-48

class SparseCaptureTests(TestCase):
    def test_keeps_sparse_original_grid_logits_and_detects_sign_mismatch(self):
        mask=np.array([[False,True],[False,True]])
        logits=np.where(mask,2.,-3.).astype(np.float32)
        row=dict(native_rank=2,query_id=8,score=.5,mask=mask,mask_logits=logits)
        codec=SimpleNamespace(encode=encode,decode=decode)
        geometry=lambda np, mask: np.full_like(mask,bool(mask.any()))
        result=pack_candidates([row],(2,2),np=np,codec=codec,dilate=geometry)
        self.assertEqual(result['positions'].tolist(),[0,1,2,3])
        self.assertEqual(result['logits'].tolist(),[-3.,2.,-3.,2.])
        self.assertEqual(result['offsets'].tolist(),[0,4])
        with self.assertRaises(ValueError):
            pack_candidates([dict(row,mask_logits=-logits)],(2,2),np=np,codec=codec,dilate=geometry)
