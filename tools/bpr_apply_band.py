# -*- coding: utf-8 -*-
"""carrier/bpr_apply.py with a fast, output-identical boundary band for large
--band radii (BPR band-width ablation, r in {2, 8, 999 = whole image}).

bpr_apply.make_band(mask, r) = dilate(mask, 3x3, r iters) XOR erode(mask, 3x3,
r iters).  On a binary image r passes of a 3x3 rectangular kernel equal ONE
pass of a (2r+1)x(2r+1) rectangular kernel, and for r >= max(H, W) the
dilation covers the whole image while the erosion keeps only an all-foreground
image, so the band degenerates to the whole image (no guard).  cv2 runs the
iterated form in O(r) passes, which is prohibitive at r=999 on ~240k HRSID
instances; the single-pass form is O(1) in r.  Radii <= 16 keep the original
per-iteration code path byte-for-byte; larger radii use the closed forms.

--selftest compares the two forms on random blobs (no GPU, no data) and exits.
Everything else is passed to bpr_apply.main() unchanged, so the CLI, the
BPR_RESULT lines and the refined-json output are those of bpr_apply.py.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from carrier import bpr_apply  # noqa: E402

_ORIG_MAKE_BAND = bpr_apply.make_band


def big_kernel_band(mask, r):
    """Single-pass (2r+1)^2 rectangular kernel; closed form for r >= max(H, W)."""
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    if r >= max(H, W):
        if not m.any():
            return np.zeros((H, W), bool)
        if m.all():
            return np.zeros((H, W), bool)
        return np.ones((H, W), bool)
    k = 2 * r + 1
    kern = np.ones((k, k), np.uint8)
    dil = cv2.dilate(m, kern, iterations=1)
    ero = cv2.erode(m, kern, iterations=1)
    return (dil ^ ero).astype(bool)


def make_band(mask, r=4):
    if r <= 16:
        return _ORIG_MAKE_BAND(mask, r)
    return big_kernel_band(mask, r)


def selftest():
    rng = np.random.RandomState(0)
    n_ok, n_tot = 0, 0
    for trial in range(40):
        H, W = rng.randint(60, 200), rng.randint(60, 200)
        m = np.zeros((H, W), np.uint8)
        for _ in range(rng.randint(1, 4)):
            cy, cx = rng.randint(0, H), rng.randint(0, W)
            ry, rx = rng.randint(3, 30), rng.randint(3, 30)
            yy, xx = np.ogrid[:H, :W]
            m[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = 1
        if trial % 7 == 0:
            m[:] = 0                       # empty mask edge case
        for r in (2, 8, 20, 40, 999):
            ref = _ORIG_MAKE_BAND(m, r) if r < 999 else None
            got = big_kernel_band(m, r)
            if ref is None:
                # r=999 >= max(H,W): whole image iff mask non-empty
                ok = bool(got.all()) if m.any() else bool((~got).all())
            else:
                ok = bool(np.array_equal(ref, got))
            n_tot += 1
            n_ok += int(ok)
            if not ok:
                print('SELFTEST MISMATCH trial=%d r=%d H=%d W=%d' % (trial, r, H, W))
    # also check the iterated form at r=999 on one small image (feasible there)
    m = np.zeros((40, 50), np.uint8)
    m[10:20, 15:30] = 1
    ok = bool(np.array_equal(_ORIG_MAKE_BAND(m, 60), big_kernel_band(m, 60)))
    n_tot += 1
    n_ok += int(ok)
    print('BPR_BAND_SELFTEST %d/%d %s' % (n_ok, n_tot, 'PASS' if n_ok == n_tot else 'FAIL'))
    return 0 if n_ok == n_tot else 1


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    bpr_apply.make_band = make_band
    sys.exit(bpr_apply.main())
