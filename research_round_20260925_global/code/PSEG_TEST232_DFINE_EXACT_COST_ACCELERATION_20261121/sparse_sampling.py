"""Bit-preserving sparse binary16 sampling for the D-FINE native-size point readout."""
from types import SimpleNamespace

import numpy as np


def axis_samples(size):
    scale = np.float32(160) / np.float32(size)
    target = np.arange(size, dtype=np.float32)
    position = np.maximum(np.float32(0), (target + np.float32(.5)) * scale - np.float32(.5))
    base = np.floor(position).astype(np.intp)
    return np.minimum(base,159), np.minimum(base+1,159), position - base.astype(np.float32)


def native_sampler(height, width):
    ybase, ynext, yfrac = axis_samples(height)
    xbase, xnext, xfrac = axis_samples(width)

    def sample(plane, coords, canvas_width):
        if canvas_width != width or not np.all((coords >= 0) & (coords < width * height)):
            raise ValueError("Expected original bounded native coordinates")
        ys, xs = np.divmod(coords, width)
        y0, y1, dy = ybase[ys], ynext[ys], yfrac[ys]
        x0, x1, dx = xbase[xs], xnext[xs], xfrac[xs]
        values = np.stack((plane[y0,x0], plane[y0,x1], plane[y1,x0], plane[y1,x1]))
        a, b, c, d = values.astype(np.float16).astype(np.float32)
        top = (np.float32(1)-dx)*a + dx*b
        bottom = (np.float32(1)-dx)*c + dx*d
        return ((np.float32(1)-dy)*top + dy*bottom).astype(np.float16)

    return sample


def replacement(original_modules, *, audit=None):
    """Replace the probability sampling callable on the fresh local shape adapter."""
    def make_native(pair, rank, height, width):
        geometry, prior = original_modules.native.make_native(pair, rank, height, width)
        optimized = native_sampler(height, width)
        if audit is None:
            sample = optimized
        else:
            def sample(plane, coords, canvas_width):
                values = optimized(plane, coords, canvas_width)
                expected = prior.sample_probability(plane, coords, canvas_width)
                if not np.array_equal(values.view(np.uint16), expected.view(np.uint16)):
                    raise ValueError("Optimized probability differs from closed binary16 sampler")
                audit["sample_calls"] += 1
                audit["sampled_values"] += values.size
                return values
        return geometry, SimpleNamespace(editable_pools=prior.editable_pools, sample_probability=sample)
    return SimpleNamespace(native=SimpleNamespace(make_native=make_native), old=original_modules.old)
