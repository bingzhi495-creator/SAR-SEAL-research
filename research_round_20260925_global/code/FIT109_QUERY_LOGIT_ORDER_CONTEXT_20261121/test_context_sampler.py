import math

import numpy as np
import pytest
import torch

from context_sampler import RING_OFFSETS_XY, sample_context, without_direction


def _rings():
    # Traverse square edges without using the production ring generator.
    for r in (1, 3):
        yield from ((x, -r) for x in range(-r, r + 1))
        for y in range(-r + 1, r):
            yield (-r, y)
            yield (r, y)
        yield from ((x, r) for x in range(-r, r + 1))


def _manual_sample(plane, flat, dx, dy):
    iy, ix = divmod(flat, 800)
    x = min(143, max(0, (ix + 0.5) * 144 / 800 - 0.5 + dx))
    y = min(143, max(0, (iy + 0.5) * 144 / 800 - 0.5 + dy))
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = min(x0 + 1, 143), min(y0 + 1, 143)
    fx, fy = x - x0, y - y0
    return ((1 - fy) * ((1 - fx) * plane[y0, x0] + fx * plane[y0, x1])
            + fy * ((1 - fx) * plane[y1, x0] + fx * plane[y1, x1]))


def test_ring_order_and_x_y_gradients():
    assert tuple(_rings()) == RING_OFFSETS_XY
    y, x = torch.meshgrid(torch.arange(144), torch.arange(144), indexing="ij")
    planes = torch.stack((x, -y)).float()
    coords = torch.tensor([[[154 * 800 + 61, 498 * 800 + 263]],
                           [[352 * 800 + 500, 740 * 800 + 468]]])
    vector = sample_context(planes, coords, [1, 1])
    assert vector.shape == (2, 1, 2, 32) and vector.dtype == torch.float32
    for row in range(2):
        for side in range(2):
            iy, ix = divmod(coords[row, 0, side].item(), 800)
            expected = [((ix + 0.5) * 144 / 800 - 0.5 + dx) if row == 0
                        else -((iy + 0.5) * 144 / 800 - 0.5 + dy) for dx, dy in _rings()]
            torch.testing.assert_close(vector[row, 0, side], torch.tensor(expected),
                                       rtol=0, atol=3e-5)


def test_native_centers_borders_and_nonuniform_query_planes():
    rng = np.random.default_rng(733)
    p = rng.normal(size=(2, 144, 144)).astype(np.float32)
    coords = np.array([[[0, 799], [799 * 800, 639999], [12 * 800 + 11, 795 * 800 + 796]],
                       [[635990, 135713], [1, 2], [855, 604000]]], dtype=np.int64)
    got = sample_context(p[:, None], coords, np.array([3, 3])).numpy()
    for r, row in enumerate(coords):
        for m, endpoints in enumerate(row):
            for side, index in enumerate(endpoints):
                expected = [_manual_sample(p[r], int(index), *offset) for offset in _rings()]
                np.testing.assert_allclose(got[r, m, side], expected, rtol=0, atol=7e-5)


def test_ragged_padding_zero_and_input_gradients():
    planes = torch.stack((torch.full((144, 144), -4.0), torch.full((144, 144), 7.0),
                          torch.full((144, 144), float("nan")))).requires_grad_()
    coords = torch.tensor([[[1200, 1953], [4, 999]],
                           [[800 * 800 - 1, 0], [-300, 820000]],
                           [[-1, 640001], [999999, -500]]])
    got = sample_context(planes, coords, [2, 1, 0])
    torch.testing.assert_close(got[0], torch.full_like(got[0], -4.0))
    torch.testing.assert_close(got[1, 0], torch.full_like(got[1, 0], 7.0))
    assert not got[1, 1].count_nonzero() and not got[2].count_nonzero()
    got.sum().backward()
    torch.testing.assert_close(planes.grad.sum((1, 2)), torch.tensor([128., 64., 0.]))
    assert torch.isfinite(planes.grad).all()


def test_directionless_keeps_each_ring_and_does_not_reorder_rows_or_sides():
    generator = torch.Generator().manual_seed(431)
    values = torch.randint(-40, 10, (2, 3, 2, 32), generator=generator).float()
    values[..., :8] += 120
    before = values.clone()
    got = without_direction(values)
    for r in range(2):
        for m in range(3):
            for side in range(2):
                v = values[r, m, side].tolist()
                assert got[r, m, side].tolist() == sorted(v[:8]) + sorted(v[8:])
    torch.testing.assert_close(values, before)


def test_empty_and_invalid_support():
    for rows, width in ((0, 0), (0, 2), (2, 0)):
        p = torch.empty(rows, 144, 144)
        coords = torch.empty(rows, width, 2, dtype=torch.long)
        got = sample_context(p, coords, [0] * rows)
        assert without_direction(got).shape == (rows, width, 2, 32)
    p = torch.zeros(1, 144, 144)
    c = torch.zeros(1, 1, 2, dtype=torch.long)
    for lengths in ([2], [-1], [0.5], []):
        with pytest.raises(ValueError):
            sample_context(p, c, lengths)
    for invalid in (-1, 640000):
        with pytest.raises(ValueError):
            sample_context(p, c + invalid, [1])
    with pytest.raises(ValueError):
        sample_context(torch.full_like(p, float("nan")), c, [1])
    with pytest.raises(ValueError):
        sample_context(p[:, :100], c, [1])
