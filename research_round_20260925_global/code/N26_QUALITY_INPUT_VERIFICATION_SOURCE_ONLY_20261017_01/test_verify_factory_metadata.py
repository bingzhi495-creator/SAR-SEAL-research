"""Pure source-reference metadata; no actual source content, tensors or data."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
source = Path(__file__).with_name('verify_inputs_body.py')
module_spec = importlib.util.spec_from_file_location('N26_FACTORY_METADATA_ONLY', str(source))
body = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(body)


class BoundMetadataOnly:
    def __init__(self, refs):
        self.bound = {ref['path']:copy.deepcopy(ref) for ref in refs}
        self.closed_paths = []

    def closed(self, ref):
        if self.bound.get(ref['path']) != ref:
            raise AssertionError('Missing or mismatched direct bound source path')
        self.closed_paths.append(ref['path'])
        return Path(ref['path'])


class FactoryMetadataTests(unittest.TestCase):
    def refs(self):
        canonical = dict(path='E:/SYNTHETIC/canonical.py', bytes=4572, sha256='a'*64)
        original = dict(path='E:/SYNTHETIC/original.py', bytes=4572, sha256='a'*64)
        return canonical, original

    def test_distinct_bound_paths_with_same_bytes_and_sha_are_accepted(self):
        canonical, original = self.refs()
        c = BoundMetadataOnly([canonical, original])
        proof = body.check_factory_group(c, canonical, [original, original, original])
        self.assertEqual(proof['original_factory'], original)
        self.assertEqual(proof['canonical_factory'], canonical)
        self.assertTrue(proof['full_group_ref_exact'])
        self.assertTrue(proof['bound_factory_bytes_sha_exact'])
        self.assertIn(canonical['path'], c.closed_paths)
        self.assertIn(original['path'], c.closed_paths)

    def test_different_bytes_or_sha_are_rejected_even_if_each_path_is_bound(self):
        for field, value in [('bytes',4573), ('sha256','b'*64)]:
            with self.subTest(field=field):
                canonical, original = self.refs()
                original[field] = value
                c = BoundMetadataOnly([canonical, original])
                with self.assertRaisesRegex(AssertionError, 'bytes and SHA256'):
                    body.check_factory_group(c, canonical, [original, original, original])

    def test_unbound_original_or_canonical_path_is_rejected(self):
        for keep_original in (False, True):
            with self.subTest(keep_original=keep_original):
                canonical, original = self.refs()
                c = BoundMetadataOnly([original if keep_original else canonical])
                with self.assertRaisesRegex(AssertionError, 'bound source path'):
                    body.check_factory_group(c, canonical, [original, original, original])

    def test_each_capture_group_still_requires_identical_complete_refs(self):
        canonical, original = self.refs()
        alias = dict(original, path='E:/SYNTHETIC/different_group_member.py')
        c = BoundMetadataOnly([canonical, original, alias])
        with self.assertRaisesRegex(AssertionError, 'complete original factory reference'):
            body.check_factory_group(c, canonical, [original, alias, original])


if __name__ == '__main__':
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    unittest.main()
