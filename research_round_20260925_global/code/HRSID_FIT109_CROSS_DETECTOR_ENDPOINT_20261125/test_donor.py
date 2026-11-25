"""Confirm donor ambiguity and signed distance on synthetic masks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as codec

spec = importlib.util.spec_from_file_location('donor', Path(__file__).with_name('donor.py'))
donor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(donor)


class Donor(unittest.TestCase):
    def test_signed_native_opposite_distance(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[0:5, 3:8] = 1
        rle = codec.encode(np.asfortranarray(mask))
        field, crop = donor.distance(rle)
        coords = np.array([4, 2*12+5, 5*12+5, 11*12+1, 6*12+8])
        np.testing.assert_allclose(donor.gather(field,crop,coords,12), [1/3, 1., -1/3, -1., -np.sqrt(5)/3])

    def test_ambiguity_and_fixed_tie_order(self):
        self.assertEqual(donor.winner(np.array([.50, .50]), np.array([[1., .70], [.70, 1.]])), (0, False))
        self.assertEqual(donor.winner(np.array([.8, .51]), np.array([[1., .49], [.49, 1.]])), (0, True))
        self.assertEqual(donor.winner(np.array([.49, .20]), np.eye(2)), (None, False))


    def test_safe_serialization_preserves_fold_keys_and_sibling_provenance(self):
        result = donor.rename_arms({'source_sibling_pilot_sha256':'closed', 0:{'duplicate_vote':.7},
                                    'native_matched_sibling':1})
        self.assertEqual(result, {'source_sibling_pilot_sha256':'closed', 0:{'deterministic_donor_vote':.7},
                                  'native_matched_donor':1})

if __name__ == '__main__':
    unittest.main()
