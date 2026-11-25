from pathlib import Path
import sys
import unittest
from shapely.geometry import box
sys.path.insert(0,str(Path(__file__).resolve().parent))
from select_supplement_retry import select, midpoint_probe

class SelectionTest(unittest.TestCase):
    def test_fixed_midpoints_after_old_probes_removed(self):
        tasks=list(range(16))
        self.assertEqual(midpoint_probe(tasks,{3,7,11,15},4),[1,5,9,13])

    def test_band_order_spacing_and_footprint(self):
        rows=[
            (dict(task_id='t0',collect_id='c0',target_ipr_m=.6,center_lon_lat=(0,0)),box(-1,-1,1,1)),
            (dict(task_id='t1',collect_id='c1',target_ipr_m=1.,center_lon_lat=(2,0)),box(1.6,-1,2.4,1)),
            (dict(task_id='t2',collect_id='c2',target_ipr_m=.6,center_lon_lat=(3,0)),box(2,-.2,3.5,.2)),
            (dict(task_id='t3',collect_id='c3',target_ipr_m=.6,center_lon_lat=(7,0)),box(6,-1,8,1)),
        ]
        def dist(x,y): return abs(x[0]-y[0])*110
        chosen,_=select(rows,[(0,0)],[box(-1,-1,1,1)],dist,quota=1)
        self.assertEqual([r['collect_id'] for r in chosen],['c1','c3'])
        self.assertEqual([r['window_id'] for r in chosen],['UMB-101','UMB-102'])

if __name__=='__main__':
    unittest.main()

