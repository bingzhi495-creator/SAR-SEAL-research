import runpy
import sys

import numpy as np
from pycocotools import mask as codec

func = runpy.run_path(sys.argv[1])["topology"]
def measure(rows):
    return func(codec.encode(np.asfortranarray(np.array(rows, dtype=np.uint8))))

assert measure([[1, 0], [0, 1]]) == (1, 0, 0, 2)
assert measure([[1, 0, 1], [0, 0, 0]]) == (2, 0, 2, 1)
assert measure([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == (1, 1, 0, 8)
assert measure([[0, 0], [0, 0]]) == (0, 0, 0, 0)
print("Synthetic topology cases passed")
