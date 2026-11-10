"""Radiometry and metadata witnesses for the fixed external cohort."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_schwegmann import acquisition, gray_levels


class QualificationTest(unittest.TestCase):
    def test_fixed_power_mapping(self):
        np.testing.assert_array_equal(gray_levels(np.asarray([0.,1e-4,1.,10.,100.])),
                                      [0,0,204,255,255])
        for invalid in (-1.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                gray_levels(np.asarray([invalid]))

    def test_copol_and_crosspol_same_acquisition(self):
        product="S1A_EW_GRDM_1SDV_20150625T032843_20150625T032943_006524_008AB7_A77F.SAFE"
        names=[acquisition(dict(fullname=product,id="S1A_00A77F_"+pol+"_0001",
                                polarisation=pol,sensor="SEN1")) for pol in ("VV","VH")]
        self.assertEqual(names[0],names[1])
        self.assertEqual(names[0],("S1A","20150625","006524","008AB7"))
        adjacent=product.replace("032843","032943").replace("A77F","A780")
        self.assertEqual(acquisition(dict(fullname=adjacent,id="S1A_00A780_VV_0001",
                                          polarisation="VV",sensor="SEN1")),names[0])
        with self.assertRaises(ValueError):
            acquisition(dict(fullname=product,id="S1A_00A77E_VV_0001",polarisation="VV",sensor="SEN1"))


if __name__=="__main__":
    unittest.main()
