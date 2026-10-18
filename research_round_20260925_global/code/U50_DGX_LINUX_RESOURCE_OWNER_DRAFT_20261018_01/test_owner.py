"""Tiny stdlib checks only: no psutil/Torch import or process launch."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import linux_owner as owner


class OwnerContracts(unittest.TestCase):
    def test_allowlist_hashes_actual_bytes_and_rejects_unlisted_or_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'allowed.bin'
            path.write_bytes(b'ordinary fixture')
            ref = dict(path=str(path.resolve()), bytes=16,
                       sha256=hashlib.sha256(b'ordinary fixture').hexdigest())
            files = owner.FileAllowlist([ref])
            self.assertEqual(files.closed(ref), path.resolve())
            other = dict(ref, path=str((Path(directory)/'other.bin').resolve()))
            with self.assertRaises(ValueError):
                files.closed(other)
            path.write_bytes(b'changed fixture!')
            with self.assertRaises(ValueError):
                files.closed(ref)

    def test_resource_boundary_is_inclusive_and_never_substitutes_missing_measurement(self):
        limit = 16*1024**3
        self.assertEqual(owner.resource_violations(1800., limit, limit, [0, 1]), [])
        self.assertIn('deadline', owner.resource_violations(1800.1, 1, limit, [0, 1]))
        self.assertIn('RSS', owner.resource_violations(0., limit+1, limit, [0, 1]))
        self.assertIn('free_RAM', owner.resource_violations(0., 1, limit-1, [0, 1]))
        self.assertIn('CPU_affinity', owner.resource_violations(0., 1, limit, [0]))
        with self.assertRaises(ValueError):
            owner.resource_violations(0., None, limit, [0, 1])

    def test_linux_birth_parser_and_terminal_gate_do_not_confuse_reused_pid_or_live_group(self):
        fields = ['S', '1', '432', '432']+['0']*15+['987654']+['0']*8
        row = owner.parse_proc_stat('432 (name with ) spaces) '+' '.join(fields))
        self.assertEqual((row['pid'], row['start_ticks'], row['process_group']), (432, 987654, 432))
        self.assertTrue(owner.birth_absent(row, dict(row, start_ticks=987655)))
        self.assertFalse(owner.birth_absent(row, dict(row)))
        self.assertTrue(owner.terminal_success(0, True, True, [], [], 'COMPLETED_PENDING_ROOT_REVIEW'))
        self.assertFalse(owner.terminal_success(0, True, True, [row], [], 'COMPLETED_PENDING_ROOT_REVIEW'))
        self.assertFalse(owner.terminal_success(None, False, False, [], [], 'COMPLETED_PENDING_ROOT_REVIEW'))
        self.assertFalse(owner.terminal_success(0, True, True, [], ['free_RAM'], 'COMPLETED_PENDING_ROOT_REVIEW'))


if __name__ == '__main__':
    unittest.main()
