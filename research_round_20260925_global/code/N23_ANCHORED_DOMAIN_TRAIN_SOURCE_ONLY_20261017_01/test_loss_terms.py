# encoding: utf-8
"""UNIT_FIXTURE: NumPy arithmetic only; no Torch, real data, weights or training."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).with_name("loss_terms.py")
S=importlib.util.spec_from_file_location("n23_loss_terms_unit",P)
loss_terms=importlib.util.module_from_spec(S)
S.loader.exec_module(loss_terms)


class LossTermTests(unittest.TestCase):
    def test_three_arms_keep_anchor_coefficient_one_and_use_declared_denominator(self):
        losses=np.array([2.,6.,10.],np.float64)
        omega=np.array([True,False,True])
        weights=np.array([1.,2.,3.],np.float64)
        for arm,extension,denom in (("A_COMMON",32/6,6.),("A_ACTIVE",8.,4.),("B_C",44/6,6.)):
            result=loss_terms.combine_loss(7.,losses,omega,weights,arm)
            self.assertEqual(set(result),{"total","extension","denom"})
            self.assertAlmostEqual(float(result["extension"]),extension)
            self.assertAlmostEqual(float(result["denom"]),denom)
            self.assertAlmostEqual(float(result["total"]),7.+.25*extension)
            shifted=loss_terms.combine_loss(9.,losses,omega,weights,arm)
            self.assertAlmostEqual(float(shifted["total"]-result["total"]),2.)

    def test_empty_pool_and_zero_omega_keep_full_anchor(self):
        for arm in ("A_COMMON","A_ACTIVE","B_C"):
            empty=loss_terms.combine_loss(np.array(3.,np.float32),
                np.array([],np.float32),np.array([],bool),np.array([],np.float32),arm)
            self.assertEqual(float(empty["denom"]),0.)
            self.assertEqual(float(empty["extension"]),0.)
            self.assertEqual(float(empty["total"]),3.)
        for arm,denom,extension in (("A_COMMON",4.,0.),("A_ACTIVE",0.,0.),("B_C",4.,5.)):
            result=loss_terms.combine_loss(3.,np.array([2.,6.]),
                np.array([False,False]),np.array([1.,3.]),arm)
            self.assertEqual(float(result["denom"]),denom)
            self.assertEqual(float(result["extension"]),extension)
            self.assertEqual(float(result["total"]),3.+.25*extension)

    def test_weight_scaling_and_generic_nonnegative_extension_weight(self):
        losses=np.array([2.,6.,10.])
        omega=np.array([True,False,True])
        weights=np.array([1.,2.,3.])
        for arm in ("A_COMMON","A_ACTIVE","B_C"):
            original=loss_terms.combine_loss(7.,losses,omega,weights,arm)
            scaled=loss_terms.combine_loss(7.,losses,omega,weights*13,arm)
            self.assertAlmostEqual(float(scaled["extension"]),float(original["extension"]))
            self.assertAlmostEqual(float(scaled["denom"]),13*float(original["denom"]))
            for coefficient in (0.,.5,2.):
                result=loss_terms.combine_loss(7.,losses,omega,weights,arm,coefficient)
                self.assertAlmostEqual(float(result["total"]),7.+coefficient*float(original["extension"]))

    def test_invalid_contract_inputs_fail_closed(self):
        valid=dict(anchor_loss=3.,new_point_losses=np.array([2.,6.]),
                   omega=np.array([True,False]),weights=np.array([1.,3.]),arm="A_COMMON")
        bad=[("arm","OTHER"),("extension_weight",-1.),("extension_weight",float("nan")),
             ("extension_weight",float("inf")),("extension_weight",True),("extension_weight",".25"),
             ("anchor_loss",np.array([3.])),("anchor_loss",float("nan")),
             ("new_point_losses",np.array([[2.,6.]])),("new_point_losses",np.array([2.])),
             ("omega",np.array([True])),("weights",np.array([1.])),
             ("omega",np.array([1,0])),("new_point_losses",np.array([2,6])),
             ("weights",np.array([1,3])),("new_point_losses",np.array([2.,float("inf")])),
             ("new_point_losses",np.array([2.,-1.])),("weights",np.array([1.,float("nan")])),
             ("weights",np.array([1.,-1.]))]
        for key,value in bad:
            with self.subTest(key=key,value=value):
                args=dict(valid);args[key]=value
                with self.assertRaises(ValueError):loss_terms.combine_loss(**args)

if __name__=="__main__":unittest.main()
