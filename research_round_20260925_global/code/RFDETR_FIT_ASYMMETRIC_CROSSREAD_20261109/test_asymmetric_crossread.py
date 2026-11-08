import unittest
from asymmetric_crossread import splice


class SameBudgetCrossread(unittest.TestCase):
    def test_only_target_partition_changes(self):
        a=["a0","a1","a2"]
        b=["b0","b1","b2"]
        self.assertEqual(splice(a,b,{1}),["a0","b1","a2"])
        self.assertEqual(splice(a,b,{0,2}),["b0","a1","b2"])


if __name__=="__main__":
    unittest.main()

