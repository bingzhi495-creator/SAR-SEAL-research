"""SYNTHETIC_ONLY regression: preserve original native INT64 label weights.

GREEN target is ordinary v02; frozen/ordinary v01 remains unchanged.
No actual SAR artifact or native teacher is loaded.
"""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('test_N16_ALL100_FIT_CACHE_CPU_QUALIFIER_SYNTHETIC_ONLY_20261014_02.py')
spec=importlib.util.spec_from_file_location('_n16_hand_packet_fixture',path)
fixture_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
torch=fixture_module.torch


class NativeLabelWeightDtypeTests(unittest.TestCase):
    def test_exact_native_new_ones_inherits_GT_int64(self):
        before=torch.get_rng_state().clone()
        gt_labels=torch.tensor([0,0],dtype=torch.int64)
        labels=gt_labels.new_full((100,),1,dtype=torch.long)
        label_weights=gt_labels.new_ones((100,))
        mask_weights=torch.zeros(100,2,3,dtype=torch.float32).new_zeros((100,))
        self.assertEqual(labels.dtype,torch.int64)
        self.assertEqual(label_weights.dtype,torch.int64)
        self.assertEqual(mask_weights.dtype,torch.float32)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_INT64_observations_and_assignment_pass(self):
        args=fixture_module.fixture()
        packet=args[0]
        for observation in packet['all_ten_target_observations']:
            observation['label_weights']=observation['label_weights'].to(torch.int64)
        packet['assignment']['final_native_label_weights']=packet['assignment']['final_native_label_weights'].to(torch.int64)
        row=fixture_module.validate(args)
        self.assertTrue(row['labels_exact'])
        self.assertEqual(packet['samples']['importance_weights'].dtype,torch.float32)

    def test_cast_FP32_native_label_weights_are_rejected(self):
        # Deliberately cast original INT64 observation to wrong FP32.
        args=fixture_module.fixture()
        for observation in args[0]['all_ten_target_observations']:
            observation['label_weights']=observation['label_weights'].to(torch.float32)
        args[0]['assignment']['final_native_label_weights']=args[0]['assignment']['final_native_label_weights'].to(torch.float32)
        self.assertEqual(args[0]['all_ten_target_observations'][0]['label_weights'].dtype,torch.float32)
        with self.assertRaisesRegex(RuntimeError,'Finite CPU tensor ABI differs: label_weights'):
            fixture_module.validate(args)


if __name__=='__main__':
    unittest.main(verbosity=2)
