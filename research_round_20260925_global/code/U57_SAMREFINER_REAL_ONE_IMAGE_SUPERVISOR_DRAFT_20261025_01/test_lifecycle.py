import subprocess
import sys
import unittest

import parent


class LifecycleTest(unittest.TestCase):
    def test_real_child_is_waited_and_actual_returncode_preserved(self):
        for expected in (0, 7):
            with self.subTest(expected=expected):
                child = subprocess.Popen([sys.executable, '-B', '-S', '-c',
                                          'raise SystemExit(' + str(expected) + ')'],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    result = parent.wait_child(child, lambda: None)
                    self.assertEqual(result['returncode'], expected)
                    self.assertTrue(result['reaped'])
                    self.assertEqual(child.poll(), expected)
                finally:
                    if child.poll() is None: child.kill()
                    child.wait()

    def test_deadline_returns_control_for_owned_cleanup_without_retry(self):
        child = subprocess.Popen([sys.executable, '-B', '-S', '-c',
                                  'import time; time.sleep(5)'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with self.assertRaises(TimeoutError):
                parent.wait_child(child, lambda: None, deadline_seconds=0.05)
        finally:
            if child.poll() is None: child.kill()
            child.wait()


if __name__ == '__main__':
    unittest.main()
