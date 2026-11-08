import unittest
from forced_crossread import fixed_substitutions


class AttributionTest(unittest.TestCase):
    def test_partition_and_stock_fallback(self):
        native=["n0","n1","n2","n3"]
        changed={0:"c0",1:"c1",2:"c2"}
        result=fixed_substitutions(native,changed,{0,3},{1})
        self.assertEqual(result["native"],native)
        self.assertEqual(result["forced_all"],["c0","c1","c2","n3"])
        self.assertEqual(result["matched_only"],["c0","n1","n2","n3"])
        self.assertEqual(result["unmatched_only"],["n0","c1","c2","n3"])
        self.assertEqual(result["overlap_unmatched_only"],["n0","c1","n2","n3"])
        self.assertEqual(result["other_unmatched_only"],["n0","n1","c2","n3"])


if __name__=="__main__":
    unittest.main()

