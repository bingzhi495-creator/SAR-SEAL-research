"""Measure native RF-DETR binary-mask donors with coordinate bit planes."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.nn.functional import interpolate
from rfdetr.datasets._torchvision import Resize


SOURCE, TARGET = 800, 576
BITS = 10


def encode_coordinates():
    weights = 1 << torch.arange(BITS)
    positions = torch.arange(SOURCE)
    encoded = (positions[None, :] & weights[:, None]) != 0
    x = encoded[:, None, :].expand(-1, SOURCE, -1)
    y = encoded[:, :, None].expand(-1, -1, SOURCE)
    return torch.cat((x, y)), weights


def decode_axis(masks, weights, axis):
    bits = masks[:BITS, 0, :] if axis == "x" else masks[BITS:, :, 0]
    return (bits.to(torch.int64) * weights[:, None]).sum(0)


@torch.inference_mode()
def audit():
    torch.set_num_threads(1)
    masks, weights = encode_coordinates()
    image = Image.new("RGB", (SOURCE, SOURCE))
    _, target = Resize((TARGET, TARGET))(image, {"masks": masks})
    stock = target["masks"]
    planes = masks[:, None].to(torch.float32)
    legacy = interpolate(planes, size=(TARGET, TARGET), mode="nearest")[:, 0].bool()
    centered = interpolate(planes, size=(TARGET, TARGET), mode="nearest-exact")[:, 0].bool()
    axis_stock = {axis: decode_axis(stock, weights, axis) for axis in ("x", "y")}
    forward = torch.arange(TARGET)
    original = torch.arange(SOURCE)
    rational_legacy = forward * SOURCE // TARGET
    rational_center = (2 * forward + 1) * SOURCE // (2 * TARGET)
    inverse_numerators = (2 * original + 1) * TARGET
    rational_inverse = inverse_numerators // (2 * SOURCE)
    raw_inverse = interpolate(forward.float().view(1, 1, 1, TARGET), size=(1, SOURCE),
                              mode="nearest-exact").flatten().to(torch.int64)

    same_stock = bool(torch.equal(stock, legacy))
    same_axes = bool(torch.equal(axis_stock["x"], axis_stock["y"]))
    center_exact = all(torch.equal(decode_axis(centered, weights, axis), rational_center) for axis in axis_stock)
    inverse_exact = bool(torch.equal(raw_inverse, rational_inverse))
    if not (same_stock and same_axes and center_exact and inverse_exact and stock.dtype == torch.bool):
        raise ValueError("Fixed stock/center mask-operator readback differed")
    native = axis_stock["x"]
    different = torch.where(native != rational_legacy)[0]
    roundtrip_native, roundtrip_center = native[raw_inverse], rational_center[raw_inverse]
    return {
        "status": "HRSID_STOCK_DIRECT_RESIZE_INDEX_ACTUAL",
        "input": "synthetic coordinate bit planes through stock CPU operators",
        "rfdetr": version("rfdetr"),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "dimensions": [SOURCE, TARGET],
        "stock_rfdetr_bool_matches_torch_legacy": same_stock,
        "stock_axes_equal": same_axes,
        "stock_minus_rational": [[int(i), int(native[i] - rational_legacy[i])] for i in different],
        "center_forward_matches_rational": center_exact,
        "common_center_inverse_matches_rational": inverse_exact,
        "common_inverse_exact_edge_ties_per_axis": int((inverse_numerators % (2 * SOURCE) == 0).sum()),
        "stock_vs_center_direct_axis_diff": int((native != rational_center).sum()),
        "stock_vs_center_roundtrip_axis_diff": int((roundtrip_native != roundtrip_center).sum()),
        "rational_vs_center_roundtrip_axis_diff": int((rational_legacy[raw_inverse] != roundtrip_center).sum()),
        "model_forward_calls": 0,
        "optimizer_updates": 0,
        "FIT": False,
        "DEV": False,
        "TEST": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (sys.version_info[:3] != (3, 10, 13) or not sys.flags.isolated or not sys.dont_write_bytecode
            or sys.flags.utf8_mode != 1 or version("rfdetr") != "1.12.0"
            or torch.__version__ != "2.10.0+cu130" or version("torchvision") != "0.25.0+cu130"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"):
        raise ValueError("Run with the fixed isolated Windows RF-DETR CPU runtime")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(audit(), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
