"""Synthetic target prelabel mask tests."""
import unittest

import numpy as np
from pycocotools import mask as codec

from dev_endpoint import edit, validate_edit


class MaskTests(unittest.TestCase):
    def test_native_area_change_is_rejected(self):
        m=np.zeros((4,4),dtype=bool)
        m[1,1]=True
        m[1,2]=True
        old=codec.encode(np.asfortranarray(m,dtype=np.uint8))
        action=edit(m,np.array([[4,5],[7,6]]),[(0,1)])
        validate_edit(old,action,2,1)
        with self.assertRaises(ValueError):
            validate_edit(old,action,2,2)

    def test_nonstock_action_is_rejected(self):
        m=np.zeros((4,4),dtype=bool)
        m[1,1]=True
        with self.assertRaises(ValueError):
            edit(m,np.array([[5,6]]),[(0,0)])


if __name__=="__main__":
    unittest.main()
