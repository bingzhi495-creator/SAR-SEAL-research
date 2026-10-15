"""One hand-derived 2x3 coordinate test; no scientific payload or I/O."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


class FeatureNeighborCoordinates(unittest.TestCase):
    def test_non_square_clamped_neighbors_reuse_existing_centers(self):
        path=Path(__file__).with_name('N20_FEATURE_NEIGHBOR_COORDINATES_UNBOUND_20261015_01.py')
        definition=importlib.util.spec_from_file_location('_N20_coordinate_SYNTHETIC',path)
        module=importlib.util.module_from_spec(definition)
        definition.loader.exec_module(module)

        centers=np.array([0,4],dtype=np.int64)
        result=module.build_coordinates(np,centers,(2,3))

        # Image flat IDs: [[0,1,2],[3,4,5]]. Combined IDs: [0,4,1,2,3,5].
        # Every row is dy=-1,0,+1, with dx=-1,0,+1 inside each dy.
        expected_extra=np.array([1,2,3,5],dtype=np.int64)
        expected_mapping=np.array([
            [0,0,2, 0,0,2, 4,4,1],
            [0,2,3, 4,1,5, 4,1,5],
        ],dtype=np.int64)

        self.assertIsInstance(result,dict)
        self.assertEqual(result['extra_flat_indices'].dtype,np.dtype(np.int64))
        self.assertEqual(result['extra_flat_indices'].shape,(4,))
        self.assertEqual(result['neighbor_to_combined'].dtype,np.dtype(np.int64))
        self.assertEqual(result['neighbor_to_combined'].shape,(2,9))
        np.testing.assert_array_equal(result['extra_flat_indices'],expected_extra)
        np.testing.assert_array_equal(result['neighbor_to_combined'],expected_mapping)
        np.testing.assert_array_equal(result['neighbor_to_combined'][:,4],np.array([0,1],dtype=np.int64))


if __name__=='__main__':
    unittest.main()
