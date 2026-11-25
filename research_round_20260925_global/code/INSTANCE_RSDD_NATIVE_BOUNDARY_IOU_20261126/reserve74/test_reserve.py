"""Synthetic guard for archived mask metric identity before the boundary audit."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_reserve import require_metric_parity


class HistoricalMetricParity(unittest.TestCase):
    def test_native_and_no_ground_truth_metrics_preserve_identity(self):
        require_metric_parity({"AP":.6,"APl":-1.},{"AP":.6,"APl":None})

    def test_differing_or_missing_metric_rejected(self):
        with self.assertRaisesRegex(ValueError,"Archived COCO"):
            require_metric_parity({"AP":.601,"APl":-1.},{"AP":.6,"APl":None})
        with self.assertRaisesRegex(ValueError,"Archived COCO"):
            require_metric_parity({"AP":.6},{"AP":.6,"APl":None})


if __name__=="__main__":
    unittest.main()
