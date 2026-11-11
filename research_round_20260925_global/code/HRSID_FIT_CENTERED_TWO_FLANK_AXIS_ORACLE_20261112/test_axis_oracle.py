from pathlib import Path
from runpy import run_path
import numpy as np
from shapely.geometry import box

probe = run_path(str(Path(__file__).with_name('axis_oracle.py')))
rect = np.array([[5., 4., 7., 8.]])
target = box(5.5, 4., 7.5, 8.)
offsets = np.array([-2., -1., -.5, 0., .5, 1., 2.])
intersect = lambda a, b: box(*a[0]).intersection(b).area
short = probe['axis_bound'](rect, target, 8., np.array([1.,0.]), offsets, intersect)
long = probe['axis_bound'](rect, target, 8., np.array([0.,1.]), offsets, intersect)
assert short['offset']==.5 and short['iou']==100.
assert long['offset']==0. and long['iou']==60.
print('Fractional transverse/long oracle and zero tie fixture passed')
