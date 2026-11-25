"""Synthetic tests for stable joint/ship selection, box parity, and oracle/greedy separation."""

from pathlib import Path
from runpy import run_path
import sys

import numpy as np


module = run_path(str(Path(__file__).with_name("raw164_box_coverage_recovery.py")))
rankings = module["rankings"]
iou_matrix = module["iou_matrix"]
greedy50 = module["greedy50"]

scores = np.array([[.8,.05], [.8,.99], [.7,.98], [.6,.01]], dtype=np.float32)
joint, ship, complete = rankings(scores, limit=3)
assert joint == [(3, 0)]
assert ship == [0, 1, 2]
assert complete == [0, 1, 2, 3]
gt = np.array([[0,0,10,10], [2,0,12,10]], dtype=np.float64)
dt = np.array([[0,0,10,10], [2,2,2,6]], dtype=np.float64)
ious = iou_matrix(gt, dt)
np.testing.assert_array_equal(ious[:,0], [1, 2/3])
np.testing.assert_array_equal(ious[:,1], [0, 0])
assert (ious.max(axis=1) >= .5).sum() == 2
assert greedy50(ious).tolist() == [True, False]
native_path = Path(sys.argv[1])
original = module["module"](native_path, module["NATIVE_SOURCE_SHA"])
raw = np.array([[1.0625,-.0625,.25,.25]], dtype=np.float32)
np.testing.assert_array_equal(original.native_xyxy(raw, [512,512]), [[480,0,512,32]])
print("synthetic stable ranks, original edge clipping, native box IoU and oracle/greedy checks passed")

