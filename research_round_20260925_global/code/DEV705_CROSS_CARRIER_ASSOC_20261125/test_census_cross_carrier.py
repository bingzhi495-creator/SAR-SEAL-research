"""Generated examples for reciprocal matching and the duplicate-only ambiguity exception."""

from pathlib import Path
from runpy import run_path

import numpy as np
from pycocotools import mask


def rle(cells):
    pixels = np.zeros((10, 10), dtype=np.uint8, order="F")
    pixels.ravel()  # Coordinates below use native row-major cell order.
    for cell in cells:
        pixels[cell // 10, cell % 10] = 1
    return mask.encode(pixels)


census = run_path(str(Path(__file__).with_name("census_cross_carrier.py")))
match = census["match_donors"]
own = rle(range(0, 36))
same = rle(range(0, 36))
similar = rle(range(0, 34))
other = rle(range(2, 38))
assert match([own], [same, similar], [0])[0]["decision"] == "accepted"
assert match([own], [same, similar], [0])[0]["ambiguity_duplicates"] is True
assert match([own], [rle(range(0, 35)), other], [0])[0]["decision"] == "ambiguous"
assert match([same, own], [same], [1])[0]["decision"] == "not_mutual"
assert match([own], [rle(range(54, 90))], [0])[0]["decision"] == "below_iou"
assert match([own], [], [0])[0]["decision"] == "no_donor"
print("synthetic association checks passed")

