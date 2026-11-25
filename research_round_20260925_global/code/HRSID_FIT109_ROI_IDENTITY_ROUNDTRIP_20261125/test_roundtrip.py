"""Native binary roundtrip identity controls."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('roundtrip',Path(__file__).with_name('roundtrip.py'))
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
paired=worker.load(Path('D:/SAR_SEAL_private_execution_20261021_01/HRSID_FIT109_WEAK_ROI_PAIRED_20261125_01/roi_paired.py'),worker.ROI_PAIRED,'_test_live_render')


class TestIdentity(unittest.TestCase):
    def test_score_map_preserves_binary_values_and_sign(self):
        input_mask=np.array([[0,1],[1,0]],dtype=np.uint8)
        a=worker.logits_for(input_mask)
        self.assertEqual(tuple(a.shape),(1,1,2,2))
        np.testing.assert_array_equal(a.numpy().reshape(2,2),[[-1,1],[1,-1]])

    def test_identical_crop_and_point_positions(self):
        native=np.zeros((48,48),dtype=bool)
        native[19:24,20:22]=True
        output=worker.render_identity(native,[0,0,48,48],native.astype(np.uint8),paired.render)
        np.testing.assert_array_equal(output,native)

    def test_empty_patch_and_no_outside_change(self):
        src=np.ones((55,55),dtype=bool)
        result=worker.render_identity(src,[2,2,46,46],np.zeros((48,48),dtype=np.uint8),paired.render)
        self.assertEqual(int(result.sum()),55*55-44*44)
        self.assertTrue(result[:2].all() and result[:,:2].all())


if __name__=='__main__':
    unittest.main()
