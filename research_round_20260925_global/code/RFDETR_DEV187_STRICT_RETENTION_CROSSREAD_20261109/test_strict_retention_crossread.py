import unittest
from strict_retention_crossread import substitute


class StrictRetention(unittest.TestCase):
    def test_only_named_rows_change(self):
        n=[{"segmentation":"n0"},{"segmentation":"n1"},{"segmentation":"n2"}]
        e=[{"segmentation":"e0"},{"segmentation":"e1"},{"segmentation":"e2"}]
        self.assertEqual(substitute(n,e,{1}),[n[0],e[1],n[2]])
        self.assertEqual(substitute(n,e,set()),n)


if __name__=="__main__":
    unittest.main()

