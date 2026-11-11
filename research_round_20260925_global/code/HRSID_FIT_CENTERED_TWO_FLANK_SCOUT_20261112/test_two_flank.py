"""Regression witnesses for pixel-centred signed flank sampling and controls."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).with_name("two_flank.py")
spec = importlib.util.spec_from_file_location("_two_flank",path)
flank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flank)


class FlankTests(unittest.TestCase):
    def mask(self):
        value = np.zeros((50,60),dtype=bool)
        value[20:25,10:34] = True
        return value

    def test_bright_parallel_sides_choose_real_offset(self):
        mask = self.mask()
        geometry = flank.from_mask(mask)
        for shift in (-2,-1,1,2):
            intensity = 20.+100.*np.roll(mask,shift,axis=0)
            scores = flank.contrasts(intensity,geometry)
            chosen = flank.selected(scores)
            self.assertEqual(flank.OFFSETS[chosen],shift)
            controls=flank.controls(scores,scores,chosen)
            self.assertTrue(all(abs(v)==abs(shift) for v in controls.values()))

    def test_split_halves_abstain(self):
        mask = self.mask()
        intensity = np.zeros_like(mask,dtype=float)
        intensity[19:24,:22] = 100.
        intensity[21:26,22:] = 100.
        self.assertEqual(flank.selected(flank.contrasts(intensity,flank.from_mask(mask))),flank.ZERO)

    def test_disjoint_cross_section_rejected(self):
        mask = self.mask()
        mask[23,:] = False
        self.assertIsNone(flank.from_mask(mask))

    def test_blank_display_has_no_action(self):
        mask = self.mask()
        values = flank.contrasts(np.ones(mask.shape),flank.from_mask(mask))
        self.assertEqual(flank.selected(values),flank.ZERO)

    def test_logit_interpolation_matches_image_at_native_shape(self):
        mask = self.mask()
        image = np.where(mask,1.,-1.)
        profile = flank.from_mask(mask)
        np.testing.assert_allclose(flank.contrasts(image,profile),flank.logit_contrasts(image,profile,image.shape),atol=1e-12)


if __name__=="__main__":
    unittest.main()

