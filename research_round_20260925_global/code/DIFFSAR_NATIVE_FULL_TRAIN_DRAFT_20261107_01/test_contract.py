"""Stdlib behavior checks; CUDA resume equivalence requires the actual paired run."""
import importlib.util,unittest
from pathlib import Path
S=importlib.util.spec_from_file_location('full_worker',str(Path(__file__).with_name('worker.py')))
w=importlib.util.module_from_spec(S);S.loader.exec_module(w)
class Contract(unittest.TestCase):
 def test_segment_has_fixed_target_no_replay_or_past_endpoint(self):
  self.assertEqual(w.validate_segment(0,256,'FORMAL'),256)
  self.assertEqual(w.validate_segment(36864,37050,'FORMAL'),186)
  for args in ((0,257,'FORMAL'),(256,256,'FORMAL'),(37000,37051,'FORMAL'),(0,32,'HEALTH')):
   with self.assertRaises(ValueError):w.validate_segment(*args)
 def test_weights_only_or_validation_checkpoint_cannot_resume_formal(self):
  c={'schema':'DiffSAR-native-full-state-v1','protocol':{'fixed_target':37050},'mode':'FORMAL','updates':256,
   'model':{},'gradients':{},'optimizer':{},'scheduler':{},'rng':{'python':0,'numpy':0,'torch_CPU':0,'torch_CUDA':0},
   'loader':{'consumed_indices':256,'sampler_seed':40244023,'MapDataset_rng':0,'fallback_candidates':[0,1]}}
  w.validate_checkpoint_header(c,256,'FORMAL',{'fixed_target':37050})
  for key in ('gradients','optimizer','scheduler','rng','loader'):
   bad=dict(c);del bad[key]
   with self.assertRaises(ValueError):w.validate_checkpoint_header(bad,256,'FORMAL',{'fixed_target':37050})
  with self.assertRaises(ValueError):w.validate_checkpoint_header(c,257,'FORMAL',{'fixed_target':37050})
  with self.assertRaises(ValueError):w.validate_checkpoint_header(c,256,'RESUME_VALIDATION',{'fixed_target':37050})
 def test_resume_delegates_native_sequence_without_mapping_old_samples(self):
  native=[2,0,1,1,2,0,0,2,1]
  self.assertEqual(list(w.skip_native_indices(iter(native),4)),native[4:])
  self.assertEqual(list(w.skip_native_indices(iter(native),0)),native)
  with self.assertRaises(ValueError):list(w.skip_native_indices(iter(native),-1))
if __name__=='__main__':unittest.main()
