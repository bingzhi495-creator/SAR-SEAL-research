"""Synthetic checks for P2 alignment, prototype support, and fixed deletes."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('p2', Path(__file__).with_name('p2_endpoint.py'))
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)


class Alignment(unittest.TestCase):
    def test_integer_origin_bilinear_and_clamped_edges(self):
        plane = np.arange(9,dtype=np.float32).reshape(3,3,1)
        coords = np.array([[0,0],[2,2],[2,6],[11,11]])
        np.testing.assert_allclose(p2.bilinear(plane,coords)[:,0],[0.,2.,3.,8.])

    def test_native_reference_regions_and_guard(self):
        native=np.zeros((21,21),dtype=bool)
        native[7:13,7:13]=True
        core,ring=p2.reference_cells(native)
        self.assertEqual(len(core),16)
        ring_set=set(map(tuple,ring))
        self.assertTrue((7,5) in ring_set and (7,2) in ring_set and (7,6) not in ring_set and (7,1) not in ring_set)
        self.assertFalse(any(native[y,x] for y,x in ring))

    def test_standardized_orthogonal_prototypes_and_unchanged_delete(self):
        values=np.array([[1,0],[0,1]],dtype=np.float32)
        np.testing.assert_allclose(p2.prototype_margin(values[:1],values[1:],values),[1.,-1.])
        np.testing.assert_allclose(p2.normalized_cue(np.array([20.,10.])),[1.,-1.])
        np.testing.assert_array_equal(p2.normalized_cue(np.ones(2)),[0.,0.])
        coordinates=np.array([[10,50],[11,51],[12,52]])
        base=np.array([10,11,51,52])
        np.testing.assert_array_equal(p2.frozen_deletes(np.array([0.,1.,2.]),coordinates,base,2),[12,11,51,52])


if __name__=='__main__':
    unittest.main()
