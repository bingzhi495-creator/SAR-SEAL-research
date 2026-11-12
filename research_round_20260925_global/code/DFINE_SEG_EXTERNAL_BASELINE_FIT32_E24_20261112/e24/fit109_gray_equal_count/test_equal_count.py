import unittest

from equal_count import common_actions, transition


class EqualCount(unittest.TestCase):
    def test_common_preserves_each_own_order_and_no_forcing(self):
        lists={"raw":[[],[]],"full":[[2,0,1],[0]],"gray_zero":[[1,2],[]],"gray_shuffled":[[0,2],[1,2]]}
        out=common_actions(lists)
        self.assertEqual(out,{"raw":[[],[]],"full":[[2,0],[]],"gray_zero":[[1,2],[]],"gray_shuffled":[[0,2],[]]})
        self.assertEqual(lists["full"][0],[2,0,1])

    def test_strict_transitions(self):
        self.assertEqual(transition(True,False,True),"lost")
        self.assertEqual(transition(False,True,True),"new")
        self.assertEqual(transition(True,True,False),"clean_retained")
        self.assertEqual(transition(True,True,True),"edited_retained")


if __name__=="__main__":
    unittest.main()