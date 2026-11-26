"""Synthetic tests of exact readback and the matched scientific gate."""
from pathlib import Path
from types import ModuleType
import os
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from pycocotools import mask
import run_dfine as w
w.np=np


class PairedReadback(unittest.TestCase):
    def test_squared_coordinates_preserve_native_row_major(self):
        points=np.array([[1,513],[1023,511]])
        self.assertEqual(w.capacities(points,512).diagonal().tolist(),[1,1])

    def test_minimum_exposure_and_sign_both_needed(self):
        self.assertEqual(w.effect_decision(.30,[.01,.2],20,8),"positive")
        self.assertEqual(w.effect_decision(-.30,[-.2,-.01],20,8),"negative")
        self.assertEqual(w.effect_decision(.30,[-.01,.2],33,10),"inconclusive")
        self.assertEqual(w.effect_decision(.60,[.01,.2],24,7),"inconclusive")

    def test_independent_rle_round_trip_and_cap(self):
        path=Path(os.environ["SAR_SEAL_PUBLISHED_SPARSE"])
        sparse=w.module(path,w.SPARSE,"_dfine_test_sparse")
        canvas=np.zeros((512,512),dtype=np.uint8)
        canvas[1,1:3]=1
        native=mask.encode(np.asfortranarray(canvas))
        native["counts"]=native["counts"].decode("ascii")
        update=sparse.xor_rle(native,np.array([2,514]))
        entry=dict(coordinates=[[2,514],[1,513]],fixed_pairs=[[0,0]],
                   bounded_pairs=[[0,0]],independent_pairs=[[0,0]],cap_squared=1)
        for arm in w.ARMS[1:]:
            entry[arm]=dict(swaps=1,rle=update)
        w.geometry_ledger_check(sparse,native,entry)
        entry["cap_squared"]=2
        with self.assertRaises(ValueError):
            w.geometry_ledger_check(sparse,native,entry)


if __name__=="__main__":
    unittest.main()

