"""Tiny invented JSON metadata only; never an actual U54 acceptance fixture."""
from copy import deepcopy
import hashlib
import unittest
import acceptance as subject


def terminal():
    return dict(root_observed_tool_exit_code=0,SSH_returncode=0,worker_returncode=0,
        reaped=True,owner_original_birth_absent=True,child_original_birth_absent=True,
        all_owned_processes_absent=True,source_and_logs_byte_verified=True)


class MetadataTests(unittest.TestCase):
    def test_original_terminal_requires_real_zero_and_all_external_closure_flags(self):
        value=terminal();before=deepcopy(value)
        self.assertEqual(subject.validate_original_terminal(value),value)
        self.assertEqual(value,before)
        for key,replacement in (('root_observed_tool_exit_code',None),('SSH_returncode',False),
            ('worker_returncode',1),('reaped',False),('owner_original_birth_absent',False),
            ('child_original_birth_absent',None),('all_owned_processes_absent',False),
            ('source_and_logs_byte_verified',False)):
            wrong=deepcopy(value);wrong[key]=replacement
            with self.subTest(key=key):
                with self.assertRaises(ValueError):subject.validate_original_terminal(wrong)

    def test_ascii_package_hash_and_worker_claim_without_root_postcheck_are_insufficient(self):
        values=[dict(path='/a/船.pt',bytes=1,sha256='a'*64)]
        expected=b'[{"bytes":1,"path":"/a/\\u8239.pt","sha256":"'+b'a'*64+b'"}]'
        self.assertEqual(subject.package_refs_sha256(values),hashlib.sha256(expected).hexdigest())
        with self.assertRaises(ValueError):
            subject.accept_validation({},terminal(),{},dict(status='COMPLETE'),{})


if __name__=='__main__':unittest.main()
