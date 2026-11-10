from unittest import TestCase
import numpy as np
from input_shift import associate_boxes, reflect_shift, register_mask


class ShiftTests(TestCase):
    def test_registered_interior_recovers_translation_without_wrap(self):
        a=np.arange(5*4).reshape(5,4)
        for shift in (-1,0,1):
            b=reflect_shift(a,shift,np)
            np.testing.assert_array_equal(register_mask(b,shift),a[1:-1])
        self.assertEqual(reflect_shift(a,1,np)[0,0],a[1,0])
        self.assertEqual(reflect_shift(a,-1,np)[-1,0],a[-2,0])

    def test_native_rank_box_association_is_single_use_and_ties_by_id(self):
        boxes=[(2,(1,1,5,5)),(3,(1,1,5,5))]
        truth={8:(1,1,5,5),7:(1,1,5,5),9:(8,8,9,9)}
        self.assertEqual(associate_boxes(boxes,truth,0),{7:2,8:3})

