"""Tiny metadata contracts only; no ML, numerical fixtures, files or fake model."""
import importlib.util
from pathlib import Path
import unittest


def api():
    spec=importlib.util.spec_from_file_location('U48_METADATA_ONLY',str(Path(__file__).with_name('capture_body.py')))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class Metadata(unittest.TestCase):
    def test_missing_direction_requires_actual_constructor_evidence(self):
        m=api();config=dict(bgr_to_rgb=True)
        result=m.resolve_channel_flags(config,dict(rgb_to_bgr=False))
        self.assertEqual(result['values'],dict(bgr_to_rgb=True,rgb_to_bgr=False))
        self.assertEqual(result['origins'],dict(bgr_to_rgb='original_config',rgb_to_bgr='closed_constructor_default'))
        self.assertEqual(config,{'bgr_to_rgb':True})
        with self.assertRaises(ValueError):m.resolve_channel_flags(config,{})
        with self.assertRaises(ValueError):m.resolve_channel_flags(dict(bgr_to_rgb=True,rgb_to_bgr=True),{})

    def test_existing_loading_file_must_equal_fresh_proof_and_state(self):
        m=api();loading=dict(teacher_checkpoint={'sha256':'a'*64},teacher_state={'proof_marker':'unchanged'})
        fresh=dict(original_loading_proof=loading,teacher_checkpoint=loading['teacher_checkpoint'],teacher_state_proof=loading['teacher_state'])
        self.assertIs(m.verify_loading_proof(dict(loading),fresh)['teacher_state'],loading['teacher_state'])
        with self.assertRaises(ValueError):m.verify_loading_proof(dict(loading,teacher_state={'proof_marker':'changed'}),fresh)
        with self.assertRaises(ValueError):m.verify_loading_proof(loading,dict(fresh,teacher_state_proof={'proof_marker':'changed'}))


if __name__=='__main__':unittest.main()
