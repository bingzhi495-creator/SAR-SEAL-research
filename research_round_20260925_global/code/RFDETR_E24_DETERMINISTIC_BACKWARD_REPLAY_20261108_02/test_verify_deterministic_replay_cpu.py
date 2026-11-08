import math
import unittest

from verify_deterministic_replay_cpu import gate


class GateTest(unittest.TestCase):
    def test_requires_matching_forward_and_both_thresholds(self):
        self.assertEqual(gate(True, 1e-5, 0.999999), "FIRST_GRADIENT_GATE_PASS")
        self.assertEqual(gate(False, 0.0, 1.0), "FORWARD_IDENTITY_GATE_FAILED")
        for relative, cosine in ((1.01e-5, 1.0), (0.0, 0.9999989), (math.nan, 1.0)):
            self.assertEqual(gate(True, relative, cosine), "FIRST_GRADIENT_GATE_FAILED")


if __name__ == "__main__":
    unittest.main()
