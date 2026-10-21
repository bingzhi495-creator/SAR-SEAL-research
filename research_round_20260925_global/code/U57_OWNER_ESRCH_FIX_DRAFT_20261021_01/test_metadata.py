"""Only external Path I/O is mocked. Never run owner/main or observe /proc."""
import argparse
import errno
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

OWNER_SOURCE=Path(__file__).parent/'linux_owner.py'


def load_owner():
    module=types.ModuleType('_U57_ESRCH_REGRESSION_DEFINITIONS')
    module.__file__=str(OWNER_SOURCE)
    exec(compile(OWNER_SOURCE.read_bytes(),str(OWNER_SOURCE),'exec'),module.__dict__)
    return module


class ProcessObservation(unittest.TestCase):
    def test_stat_read_esrch_means_process_disappeared(self):
        owner=load_owner()
        error=ProcessLookupError(errno.ESRCH,'SYNTHETIC_PROCESS_DISAPPEARED_DURING_STAT_READ')
        with patch.object(Path,'read_text',autospec=True,side_effect=error) as read:
            self.assertIsNone(owner.proc_identity(424242))
        read.assert_called_once()
        self.assertEqual(read.call_args[0][0],Path('/proc/424242/stat'))

    def test_permission_unknown_errors_and_incomplete_stat_still_refuse(self):
        owner=load_owner()
        for error in (PermissionError(errno.EACCES,'SYNTHETIC_DENIED'),
                      OSError(errno.EIO,'SYNTHETIC_IO_FAILURE'),
                      OSError(errno.EINVAL,'SYNTHETIC_UNKNOWN_INVALID_READ')):
            with self.subTest(error=type(error).__name__,errno=error.errno):
                with patch.object(Path,'read_text',autospec=True,side_effect=error):
                    with self.assertRaises(type(error)) as observed:owner.proc_identity(424242)
                self.assertIs(observed.exception,error)
        for text in ('424242 (SYNTHETIC) S 1','not-a-proc-stat'):
            with self.subTest(incomplete=text),patch.object(Path,'read_text',autospec=True,return_value=text):
                with self.assertRaises(ValueError):owner.proc_identity(424242)
        with patch.object(Path,'read_text',autospec=True,side_effect=FileNotFoundError(errno.ENOENT,'SYNTHETIC_GONE')):
            self.assertIsNone(owner.proc_identity(424242))
        identity=dict(pid=424242,state='S',parent_pid=42,process_group=424242,session=424242,start_ticks=9000)
        fields=['S','42','424242','424242']+['0']*15+['9000']
        with patch.object(Path,'read_text',autospec=True,return_value='424242 (SYNTHETIC (name)) '+' '.join(fields)):
            observed=owner.proc_identity(424242)
        self.assertEqual(observed,identity)
        self.assertFalse(owner.birth_absent(identity,observed))
        self.assertTrue(owner.birth_absent(identity,None))
        self.assertTrue(owner.birth_absent(identity,dict(observed,start_ticks=9001)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner-source',type=Path,default=OWNER_SOURCE)
    args,remaining=parser.parse_known_args()
    OWNER_SOURCE=args.owner_source
    unittest.main(argv=[sys.argv[0]]+remaining)
