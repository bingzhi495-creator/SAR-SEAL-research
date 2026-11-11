import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fixed_cell_capture import route_cells


class RouteTests(unittest.TestCase):
    def test_route_mixed_sign_only_and_break_ties_by_coordinates(self):
        a=np.array([[-2.,2.,-2.],[-2.,2.,-2.],[-2.,-2.,-2.]],np.float32)
        self.assertEqual(route_cells(a,2),[[0,0],[0,1]])
        self.assertEqual(route_cells(np.ones_like(a),8),[])


if __name__ == "__main__":
    unittest.main()
