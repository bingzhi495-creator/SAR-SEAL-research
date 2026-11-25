"""Focused signed support and span checks."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import unittest
import numpy as np

spec=spec_from_file_location("span_diagnostic",Path(__file__).with_name("span_diagnostic_retry.py"))
m=module_from_spec(spec)
spec.loader.exec_module(m)

class SpanTest(unittest.TestCase):
    def test_duplicate_directions_and_membership(self):
        a=np.array([1,0,-1,0,0,0,0,0],dtype=np.int8)
        b=np.array([0,-1,0,1,0,0,0,0],dtype=np.int8)
        c=np.array([0,0,0,0,1,0,-1,0],dtype=np.int8)
        rank,support,result=m.span_metrics(np.stack((a,b,a)),{"exact":a,"span_only":a+b,"out":c,
                                                          "halt":np.zeros_like(a)})
        self.assertEqual((rank,support),(2,4))
        self.assertAlmostEqual(result["exact"][0],0)
        self.assertEqual(result["exact"][2],0)
        self.assertAlmostEqual(result["span_only"][0],0)
        self.assertEqual(result["span_only"][2],2)
        self.assertEqual(result["out"][0],1)
        self.assertEqual(result["out"][3],1)
        self.assertEqual(result["halt"],[0]*5)

    def test_signed_difference_uses_final_raster(self):
        stock=np.array([[False,True,False,True,False,True]])
        positions=np.array([[0,1],[2,3],[4,5]])
        old=m.apply(stock,positions,np.array([[0,0]]),1)
        latest=m.apply(stock,positions,np.array([[1,1]]),1)
        v=m.selected_vector(m.encoded(latest),old,stock,positions,1)
        np.testing.assert_array_equal(v,np.array([-1,+1,+1,-1,0,0]))

    def test_full_quota_empty_bank_unchanged(self):
        rank,support,result=m.span_metrics(np.empty((0,4),dtype=np.int8),{"static":np.zeros(4,dtype=np.int8)})
        self.assertEqual((rank,support,result["static"]),(0,0,[0]*5))

if __name__=="__main__":
    unittest.main()
