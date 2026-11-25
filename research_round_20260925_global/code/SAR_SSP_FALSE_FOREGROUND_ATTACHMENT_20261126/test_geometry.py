from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from geometry import attachment

class AttachmentTest(unittest.TestCase):
    def test_disjoint_attached_and_restored_ledger(self):
        fixed=np.zeros((9,10),dtype=bool)
        fixed[2:5,2:5]=True
        fixed[8,8]=True
        variable=fixed.copy()
        variable[2,2]=False
        variable[4,5]=True
        variable[8,9]=True
        variable[0,9]=True
        author=fixed.copy()
        author[2,2]=False
        native=fixed.copy()
        native[4,5]=True
        got=attachment(native,fixed,variable,author,8)
        self.assertEqual([got[x] for x in ('main_gain','secondary_gain','unseeded_gain','all_removal','net_fp')],[1,1,1,1,2])
        self.assertEqual([got[x] for x in ('main_novel_gain','secondary_novel_gain','unseeded_novel_gain')],[0,1,1])
        self.assertEqual(got['near_kept_main'],1)

    def test_diagonal_is_not_four_connected(self):
        fixed=np.zeros((7,7),dtype=bool)
        fixed[2:4,2:4]=True
        variable=fixed.copy()
        variable[4,4]=True
        for neighbors,key in ((4,'unseeded_gain'),(8,'main_gain')):
            got=attachment(fixed,fixed,variable,fixed,neighbors)
            self.assertEqual(got[key],1)
            self.assertEqual(got['net_fp'],1)

if __name__=='__main__':
    unittest.main()
