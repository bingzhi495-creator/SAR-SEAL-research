from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shore_diagnostic import discover, partition


class DiscoveryTest(unittest.TestCase):
    def test_exact_nonignored_iou50_discovery_and_retention_partition(self):
        records = [dict(dtIds=[1, 2, 3, 4], dtMatches=np.array([[21., 22., 0., 23.], [0., 0., 0., 0.]]),
                        dtIgnore=np.array([[False, True, False, False], [False] * 4])), None,
                   dict(dtIds=[5], dtMatches=np.array([[24.]]), dtIgnore=np.array([[False]]))]
        eligible = {21, 22, 23, 24}
        raw = discover(records, 2, eligible)
        self.assertEqual(raw, {21, 23})
        self.assertEqual(partition(eligible, raw, {23, 24}),
                         {"both": 1, "first_only": 1, "second_only": 1, "neither": 1})
        with self.assertRaisesRegex(ValueError, "foreign"):
            partition(eligible, raw, {99})


if __name__ == "__main__":
    unittest.main()
