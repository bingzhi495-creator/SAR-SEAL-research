"""Synthetic archive layout round-trip check."""
import unittest

import numpy as np

from frozen_refit import array_hash, package_bytes, read_archive


class ArchiveTest(unittest.TestCase):
    def test_canonical_arrays_round_trip(self):
        x=np.asarray([[1.,-2.],[3.,4.]],dtype=np.float32)
        original={"state.x":x,"scale":np.asarray([1.,2.],dtype=np.float32)}
        decoded=read_archive(package_bytes(original))
        self.assertEqual(set(original),set(decoded))
        for k,v in original.items():
            self.assertEqual(array_hash(v),array_hash(decoded[k]))


if __name__=="__main__":
    unittest.main()
