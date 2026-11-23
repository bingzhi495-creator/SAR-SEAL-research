"""Differential checks on artificial chips only. Run with --code pointing to the published code root."""

import argparse
from hashlib import sha256
import math
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np


def load(path, expected=None):
    source = path.read_bytes()
    if expected and sha256(source).hexdigest() != expected:
        raise ValueError("Historical reference digest differs")
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def scalar_oracle(mask, ranks, endpoint):
    row, col = divmod(int(endpoint), mask.shape[1])
    shells = {}
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            s = dy*dy + dx*dx
            if not 0 < s <= 25:
                continue
            y, x = row + dy, col + dx
            valid = 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
            affinity = 1 - abs(ranks[row, col] - ranks[y, x]) if valid else 1.
            shells.setdefault(s, []).append((valid, affinity, float(mask[y,x]) if valid else 0.))
    total, full, null, cov = [], [], [], []
    for s, group in shells.items():
        weight = math.exp(-math.sqrt(s)/5)
        inside = [p for p in group if p[0]]
        a_bar = math.fsum(a for _,a,_ in inside)/len(inside) if inside else 0.
        m_bar = math.fsum(m for _,_,m in inside)/len(inside) if inside else 0.
        for valid, a, m in group:
            total.append(weight*a)
            full.append(weight*a*m)
            null.append(weight*(a_bar if valid else 1.)*m)
            if valid:
                cov.append(weight*(a-a_bar)*(m-m_bar))
    z = math.fsum(total)
    return math.fsum(full)/z, math.fsum(null)/z, math.fsum(cov)/z


class AffinityTests(unittest.TestCase):
    def test_luminance_ties(self):
        rgb = np.array([[[10,20,0], [10,20,0], [8,0,1], [254,0,8]]], dtype=np.uint8)
        np.testing.assert_array_equal(api.rank_image(rgb), [[.5, .5, .125, .875]])

    def test_original_and_direct_scalar_geometry(self):
        rng = np.random.default_rng(5217)
        exercised = 0
        for shape in ((2, 3), (9, 13), (22, 25)):
            for rep in range(5):
                mask = rng.random(shape) > .45
                mask[0,0], mask[-1,-1] = False, True
                bg, fg = np.flatnonzero(~mask), np.flatnonzero(mask)
                n = min(len(bg),len(fg),25)
                c = np.column_stack((bg[:n], fg[-n:][::-1]))
                rgb = rng.integers(0,5,(*shape,3),dtype=np.uint8) if rep else np.zeros((*shape,3),np.uint8)
                ranks = api.rank_image(rgb)
                result = api.endpoint_scores(mask, c, ranks, static)
                yy, xx = np.divmod(c, mask.shape[1])
                expected = static._initial(mask, yy, xx) / static._WEIGHTS.sum()
                np.testing.assert_array_equal(result["static"], expected)
                if rep == 0:
                    for arm in ("full", "shell"):
                        np.testing.assert_array_equal(result[arm], expected)
                for idx, coord in np.ndenumerate(c):
                    full, shell, covariance = scalar_oracle(mask, ranks, coord)
                    self.assertAlmostEqual(result["full"][idx], full, places=14)
                    self.assertAlmostEqual(result["shell"][idx], shell, places=14)
                    self.assertAlmostEqual(result["full"][idx]-result["shell"][idx], covariance, places=14)
                actions, reference = static.select(mask, c, min(n,8))
                exercised += len(actions)
                self.assertEqual(api.select_scores(result["static"], len(actions), static), reference)
        self.assertGreater(exercised, 0)

    def test_angular_shuffle_and_ties(self):
        mask = np.zeros((13,13),bool)
        mask[6,7] = True
        coords = np.array([[6*13+6,6*13+7]])
        ranks = np.full(mask.shape,.2)
        ranks[6,5], ranks[6,7] = .85,.3
        rotated = ranks.copy()
        rotated[6,5], rotated[6,7] = ranks[6,7], ranks[6,5]
        a = api.endpoint_scores(mask,coords,ranks,static)
        b = api.endpoint_scores(mask,coords,rotated,static)
        self.assertGreater(a["full"][0,0],b["full"][0,0])
        self.assertAlmostEqual(a["shell"][0,0],b["shell"][0,0],places=14)
        s = np.full((3,2),.45)
        eps = static._TOL / static._WEIGHTS.sum()
        s[1] += np.array([1.,-1.])*eps*.5
        s[2] += np.array([1.,-1.])*eps*3
        self.assertEqual(api.select_scores(s,3,static),[(2,2),(0,0),(1,1)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", type=Path, required=True)
    args, extra = p.parse_known_args()
    api = load(Path(__file__).with_name("affinity.py"))
    static = load(args.code / "FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py",
                  "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6")
    unittest.main(argv=[__file__,*extra])
