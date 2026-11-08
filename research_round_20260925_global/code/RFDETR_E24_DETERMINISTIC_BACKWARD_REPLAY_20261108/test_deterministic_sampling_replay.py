from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch

from deterministic_sampling_replay import StrictBackward


class StrictBackwardTest(unittest.TestCase):
    def test_native_call_gets_strict_mode_and_restores_it(self):
        old=torch.are_deterministic_algorithms_enabled()
        warned=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            calls=[]
            def stock(value):
                calls.append((torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled()))
                return value
            trace=StrictBackward(stock)
            self.assertEqual(trace("value"),"value")
            self.assertEqual(calls,[(True,False)])
            self.assertEqual(trace.calls,1)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(old,warn_only=warned)

    def test_restore_even_on_failure(self):
        old=torch.are_deterministic_algorithms_enabled()
        warned=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True,warn_only=True)
            def stock():
                raise RuntimeError("sample")
            trace=StrictBackward(stock)
            with self.assertRaises(RuntimeError):
                trace()
            self.assertEqual(trace.calls,0)
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(old,warn_only=warned)


if __name__=="__main__":
    unittest.main()
