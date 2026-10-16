# encoding: utf-8
"""UNIT_FIXTURE: pure integer/Fraction support oracle, no scientific imports/data."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('support_body.py')
SPEC=importlib.util.spec_from_file_location('n27_support_arithmetic_unit',SOURCE)
body=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(body)


def bits(value,n):
    return [(value >> i) & 1 for i in range(n)]


class SupportCeilingTests(unittest.TestCase):
    def test_exact_domain_FP_FN_and_nonempty_GT(self):
        self.assertEqual(body.support_iou_ceiling(4,2,1),Fraction(3,6))
        self.assertEqual(body.support_iou_ceiling(4,0,4),Fraction(0,4))
        self.assertEqual(body.support_iou_ceiling(4,0,0),Fraction(1,1))
        self.assertIsInstance(body.support_iou_ceiling(3,2,1),Fraction)

    def test_invalid_counts_rejected(self):
        for args in ((0,0,0),(-1,0,0),(2,-1,0),(2,0,-1),(2,0,3),(True,0,0),(2.,0,0),(2,False,0)):
            with self.subTest(args=args),self.assertRaises((TypeError,ValueError)):
                body.support_iou_ceiling(*args)

    def test_all_small_binary_masks_equal_maximum_over_allowed_edits(self):
        cases=0
        for n in range(1,5):
            for gt_int in range(1,2**n):
                gt=bits(gt_int,n);G=sum(gt)
                for native_int in range(2**n):
                    native=bits(native_int,n)
                    for support_int in range(2**n):
                        support=bits(support_int,n)
                        FPout=sum(p and not g and not s for p,g,s in zip(native,gt,support))
                        FNout=sum(not p and g and not s for p,g,s in zip(native,gt,support))
                        allowed=[]
                        for candidate_int in range(2**n):
                            candidate=bits(candidate_int,n)
                            if any(a != p and not s for a,p,s in zip(candidate,native,support)):
                                continue
                            intersection=sum(a and g for a,g in zip(candidate,gt))
                            union=sum(a or g for a,g in zip(candidate,gt))
                            allowed.append(Fraction(intersection,union))
                        expected=max(allowed)
                        self.assertEqual(body.support_iou_ceiling(G,FPout,FNout),expected,
                            (n,gt_int,native_int,support_int))
                        cases+=1
        self.assertEqual(cases,4340)


if __name__=='__main__':unittest.main()
