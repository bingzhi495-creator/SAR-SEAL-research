from pathlib import Path
import runpy
import numpy as np
from shapely.geometry import box

code = runpy.run_path(str(Path(__file__).with_name('hull_cocenter.py')))
rect = np.array([[1., 1., 3., 3.]])
target = box(1.5, 1., 3.5, 3.)
shifts = code['translations'](rect, target, np.array([0., 1.]), np.array([1., 0.]))
assert np.allclose(shifts['short'], [.5,0.]) and np.allclose(shifts['long'], [0.,0.])
assert np.array_equal(shifts['short'], shifts['both'])
assert not code['outside'](rect, shifts['short'], 4, 4)
assert code['outside'](rect, np.array([1.1,0.]), 4, 4)
assert code['readout'](rect, 4., target, lambda b,p: box(*b[0]).intersection(p).area)['continuous_iou_points'] == 60.
assert code['readout'](rect + np.tile(shifts['short'],2), 4., target, lambda b,p: box(*b[0]).intersection(p).area)['continuous_iou_points'] == 100.
print('Fractional co-centering, exact area, and symmetric-support bounds fixture passed')
