"""One stdlib-only behavior test for Windows closed-reference spelling."""
import importlib.util
from pathlib import Path
import sys
import unittest

class CanonicalSourceReferenceTest(unittest.TestCase):
    def test_forward_slash_reference_preserves_file_identity_and_input(self):
        paths=[Path(__file__).with_name('run_capture.py')]
        original=dict(path='D:/SAR_SEAL_scientific_outputs_20261015/N21_NATIVE_SUPPORT/example.json',bytes=5996,sha256='a'*64)
        expected=dict(original,path=str(Path(original['path'])))
        for i,path in enumerate(paths):
            loader=importlib.util.spec_from_file_location('U32_PATH_METADATA_TEST_%d'%i,path)
            module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
            before=dict(original)
            self.assertEqual(module.canonical_reference(original),expected)
            self.assertEqual(original,before)
        self.assertFalse(any(name=='torch' or name.startswith('torch.') for name in sys.modules))

if __name__=='__main__':unittest.main()
