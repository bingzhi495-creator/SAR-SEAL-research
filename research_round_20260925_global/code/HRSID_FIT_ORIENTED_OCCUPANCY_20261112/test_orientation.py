from pathlib import Path
import runpy
import sys

sys.path.append(sys.argv[1])
from shapely import box
import numpy as np

orientation = runpy.run_path(str(Path(__file__).with_name("oriented_occupancy.py")))
target = orientation["oriented_targets"]
shape = box(2, 3, 8, 4)
cells = np.array([[1, 3], [3, 2], [3, 3], [7, 3], [3, 3]], dtype=np.int64)
base = np.array([0., 0., 1., 1., 1.])
value, reason = target(shape, (10, 10), cells, base)
assert reason == "eligible"
np.testing.assert_allclose(value["along_occupancy"], [.25, 0., 1., .75, 1.], rtol=0, atol=1e-12)
np.testing.assert_allclose(value["across_occupancy"], [0., .25, .5, .5, .5], rtol=0, atol=1e-12)
np.testing.assert_allclose(value["four_direction_occupancy"], [.125, .125, .75, .625, .75], rtol=0, atol=1e-12)
for polygon, expected in ((box(2,3,3,4), "aspect_below_2"), (box(0,3,6,4), "frame_crossing")):
    fallback, why = target(polygon, (10,10), cells, base)
    assert why == expected and all(x is base for x in fallback.values())
print("Exact directional rectangle and common fallback fixtures passed.")
