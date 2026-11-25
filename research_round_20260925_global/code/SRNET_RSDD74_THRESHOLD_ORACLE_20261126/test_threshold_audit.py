"""Synthetic checks for matched-pair and evaluator transition denominators."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from threshold_audit import detection_transitions, fixed_crossings


class ThresholdAuditTests(unittest.TestCase):
    def test_ignore_and_assignment_change_are_preserved(self):
        old={1:("TP",11),2:("FP",0),3:("ignore",0),4:("TP",41)}
        new={1:("TP",12),2:("TP",21),3:("FP",0),4:("TP",41)}
        result=detection_transitions(old,new)
        self.assertEqual(result["transition_counts"]["TP to TP"],2)
        self.assertEqual(result["transition_counts"]["FP to TP"],1)
        self.assertEqual(result["transition_counts"]["ignore to FP"],1)
        self.assertEqual(result["TP_to_same_reference"],1)
        self.assertEqual(result["TP_to_different_reference"],1)

    def test_exact_threshold_and_balanced_coverage(self):
        result=fixed_crossings([.499,.50,.90,.49],[.50,.49,.90,.49],.50)
        self.assertEqual(result,dict(above_to_above=1,below_to_above=1,above_to_below=1,below_to_below=1))


if __name__=="__main__":
    unittest.main()
