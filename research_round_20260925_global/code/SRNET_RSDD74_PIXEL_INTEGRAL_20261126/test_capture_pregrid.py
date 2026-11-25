"""Synthetic author-style unpad/resize capture fixture."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

source = Path(__file__).with_name("capture_pregrid.py")
spec = spec_from_file_location("capture_pregrid", source)
capture = module_from_spec(spec)
spec.loader.exec_module(capture)


def test_resize_observer_after_unpad():
    utils = SimpleNamespace(F=F)
    torch.set_num_threads(1)
    padded = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape((2, 6, 6)) / 72
    desired = padded[:, :4, :5].contiguous()

    def author_resize():
        p = utils.F.interpolate(padded[None, :, :4, :5], (3, 4), mode="bilinear").squeeze(0)
        return p > .5

    stock, pre, final = capture.observe_author_resize(utils, author_resize, 2, (4,5), (3,4))
    assert utils.F is F
    assert torch.equal(pre, desired)
    assert torch.equal(final, F.interpolate(desired[None], (3,4), mode="bilinear", align_corners=False)[0])
    assert torch.equal(stock, final > .5)


if __name__ == "__main__":
    test_resize_observer_after_unpad()
    print("1 synthetic author resize observation check passed")

