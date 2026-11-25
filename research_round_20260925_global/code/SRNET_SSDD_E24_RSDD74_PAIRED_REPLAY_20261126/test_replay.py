import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pycocotools import mask

here = Path(__file__).parent
spec = spec_from_file_location("replay", here / "replay.py")
replay = module_from_spec(spec)
spec.loader.exec_module(replay)
code = Path(os.environ["SRNET_PUBLIC_CODE"])
base, methods = replay.edit_modules(code)


def test_equal_budget_with_different_signed_choices():
    stock = np.zeros((10, 10), dtype=bool)
    stock[2:8, 2:8] = True
    coords = np.array([[11, 22], [12, 23], [13, 24]], dtype=np.int32)
    q = np.array([[.9, .8], [.8, .7], [.3, .1]], dtype=np.float32)
    out = replay.render_pair(base, methods, stock, coords, q)
    assert out["fixed"]["swaps"] == out["free_matched_count"]["swaps"] == 3
    for result in out.values():
        assert mask.area(result["rle"]) == stock.sum()
    q[1] = [.2, .7]
    out = replay.render_pair(base, methods, stock, coords, q)
    assert out["fixed"]["swaps"] == out["free_matched_count"]["swaps"] == 2
    assert out["fixed"]["rle"] != out["free_matched_count"]["rle"]
    assert replay.render_pair(base, methods, stock, coords, np.zeros_like(q)) == {}


def test_probability_samples_under_half_open_box_without_altering_ties():
    stock = np.zeros((512, 512), dtype=bool)
    stock[20:27, 30:37] = True
    probabilities = np.full((512, 512), .40, dtype=np.float32)
    probabilities[stock] = .65
    probabilities[19, 31] = .50
    points, sampler = base.canvas(methods, (512, 512))
    box = np.asarray([29.25, 18.5, 37.5, 27.5], dtype=np.float32)
    rgb = np.repeat((stock.astype(np.uint8) * 255)[..., None], 3, axis=2)
    features = methods.pair.image_features(rgb)
    coords, values, *_ = points.paired_points(stock, box, 49, .7, probabilities, features, sampler)
    base.check_support(methods, stock, box, probabilities, coords)
    assert values.shape == (len(coords), 2, 14)
    assert np.array_equal(values[:, :, 0], probabilities.ravel()[coords])
    assert np.count_nonzero(values[:, 0, 0] == .50) == 1
    assert np.array_equal(probabilities > .5, stock)


if __name__ == "__main__":
    test_equal_budget_with_different_signed_choices()
    test_probability_samples_under_half_open_box_without_altering_ties()
    print("matched budget, zero budget and exact native support/probability ties: PASS")
