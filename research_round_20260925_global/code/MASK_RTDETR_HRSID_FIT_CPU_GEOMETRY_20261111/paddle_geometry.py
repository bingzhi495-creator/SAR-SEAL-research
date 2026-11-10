"""CPU replay of fixed PaddleDetection geometry classes without Paddle data/model runtime.

Selected upstream declarations retain exact computational class/function bodies.
Paddle configuration and repeated batch registry hooks are bypassed. The pinned
reader YAML supplies the sequence and arguments. Encoded inputs must be in memory.
"""

import ast
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import logging
from numbers import Integral, Number
from pathlib import Path
import random
from threading import Lock
from types import CodeType
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from pycocotools import mask
import yaml


UPSTREAM_COMMIT = "b25522a0f4bde8c80603f3ba5e3472059972e3b5"
SOURCE_SHA256 = {
    "operators.py": "b5a298403f1e683011c0953823176d2eccdb4e56708117e0439a0649f6ba450f",
    "batch_operators.py": "286015c9fca629ac9fdf3d6cba3e77680ad5fb75f2592d1e5dc670c5af97d3be",
    "mask_rtdetr_reader.yml": "aa4d8a56678441fe1bb4e31f0deb2985aba1c537fb1cdc88278a47ab8bb66dad",
}
_SAMPLE = ("Decode", "Poly2Mask", "RandomDistort", "RandomExpand", "RandomCrop", "RandomFlip")
_BATCH = ("BatchRandomResize", "NormalizeImage", "NormalizeBox", "BboxXYXY2XYWH", "Permute")
_RANDOM_LOCK = Lock()


@dataclass(frozen=True)
class StockGeometry:
    operators: dict
    batch_resize_code: CodeType
    sample_specs: tuple
    batch_specs: tuple
    batch_size: int


@dataclass(frozen=True)
class PairedView:
    before_resize: dict
    output: dict
    center_masks: np.ndarray
    source_hw: tuple[int, int]
    expanded: bool
    cropped: bool
    flipped: bool
    image_interpolation: int


def _checked(source_dir, name):
    path = source_dir / name
    with path.open("rb") as file:
        raw = file.read(512 * 1024 + 1)
    if len(raw) > 512 * 1024 or sha256(raw).hexdigest() != SOURCE_SHA256[name]:
        raise ValueError("Expected exact upstream source: " + name)
    return path, raw


def _definitions(path, raw, names):
    source = ast.parse(raw, filename=str(path))
    body = [
        entry for entry in source.body
        if isinstance(entry, (ast.ClassDef, ast.FunctionDef)) and entry.name in names
    ]
    if len(body) != len(names) or {entry.name for entry in body} != set(names):
        raise ValueError("Exact upstream operator declarations required")
    return compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")


def _identity_registration(cls):
    return cls


def _specs(specs, expected):
    if tuple(next(iter(spec)) for spec in specs if len(spec) == 1) != expected:
        raise ValueError("Exact upstream reader operation order required")
    return tuple((name, arguments) for spec in specs for name, arguments in spec.items())


def load_stock(source_dir: Path) -> StockGeometry:
    """Load only byte-pinned geometry declarations and the official reader YAML."""
    path, raw = _checked(source_dir, "operators.py")
    names = ("register_op", "BaseOperator", "Resize", "Pad", *_SAMPLE, *_BATCH[1:])
    operators = dict(
        __name__="_pinned_paddledet_operators", Sequence=Sequence, Integral=Integral,
        Number=Number, cv2=cv2, np=np, random=random, uuid=uuid, deepcopy=deepcopy,
        Image=Image, ImageDraw=ImageDraw, ImageEnhance=ImageEnhance, mask=mask,
        MUTEX=Lock(), logger=logging.getLogger(__name__), registered_ops=[],
        serializable=_identity_registration,
    )
    exec(_definitions(path, raw, names), operators)
    path, raw = _checked(source_dir, "batch_operators.py")
    batch_code = _definitions(path, raw, ("BatchRandomResize",))
    _, raw = _checked(source_dir, "mask_rtdetr_reader.yml")
    reader = yaml.safe_load(raw)["TrainReader"]
    sample = _specs(reader["sample_transforms"], _SAMPLE)
    batch = _specs(reader["batch_transforms"], _BATCH)
    if reader["batch_size"] != 4:
        raise ValueError("Expected official training batch size")
    return StockGeometry(operators, batch_code, sample, batch, reader["batch_size"])


def center_masks(masks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Integer pixel-center gather with ties choosing the upper source index."""
    if masks.ndim == 1 and not masks.size:
        return masks.copy()  # Native Poly2Mask shape for an image without rows.
    if masks.ndim != 3 or min(masks.shape[-2:]) <= 0 or min(size) <= 0:
        raise ValueError("Expected raster mask rows and a positive output shape")
    in_h, in_w = masks.shape[-2:]
    out_h, out_w = size
    yi = (2 * np.arange(out_h, dtype=np.int64) + 1) * in_h // (2 * out_h)
    xi = (2 * np.arange(out_w, dtype=np.int64) + 1) * in_w // (2 * out_w)
    return np.ascontiguousarray(masks[:, yi[:, None], xi])


def _sample_view(ops, sample):
    out = deepcopy(sample)
    out = ops[0][1](out)
    source_hw = out["image"].shape[:2]
    expanded = cropped = False
    for name, op in ops[1:]:
        old_hw = out["image"].shape[:2]
        out = op(out)
        changed_hw = old_hw != out["image"].shape[:2]
        if name == "RandomExpand":
            expanded = changed_hw
        if name == "RandomCrop":
            cropped = changed_hw
    return out, source_hw, expanded, cropped, bool(out.get("flipped", False))


def _batch_ops(stock, calls):
    # Stock BatchRandomResize chooses size and interpolation once and constructs
    # Resize. The subclass records that choice and inherits all computation.
    class WitnessResize(stock.operators["Resize"]):
        def __init__(self, target_size, keep_ratio, interp):
            calls.append((target_size, keep_ratio, int(interp)))
            super().__init__(target_size, keep_ratio, interp)

    # Keep one fresh registry/namespace per batch to avoid changing the sample
    # operator namespace or registering a second BatchRandomResize there.
    registry = dict(stock.operators, Resize=WitnessResize)
    registry["register_op"] = lambda cls: _identity_registration(cls)
    exec(stock.batch_resize_code, registry)
    return [registry[name](**arguments) for name, arguments in stock.batch_specs]


def transform_batch(stock: StockGeometry, samples: list[dict], *, seed: int) -> list[PairedView]:
    """Replay the official ordered CPU geometry on one fixed four-image batch."""
    if type(samples) is not list or len(samples) != stock.batch_size:
        raise ValueError("Expected one original four-sample batch")
    if not all(type(sample.get("image")) is bytes and sample["image"] and "im_file" not in sample
               for sample in samples):
        raise ValueError("Expected encoded in-memory images")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Expected a valid joint Python/NumPy seed")
    with _RANDOM_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            ops = [(name, stock.operators[name](**arguments)) for name, arguments in stock.sample_specs]
            prepared = [_sample_view(ops, sample) for sample in samples]
            batch = [deepcopy(item[0]) for item in prepared]
            calls = []
            for operator in _batch_ops(stock, calls):
                batch = operator(batch)
            if len(calls) != 1:
                raise ValueError("Expected one original batch resize")
            side, keep_ratio, interpolation = calls[0]
            if type(side) is not int or keep_ratio:
                raise ValueError("Expected square stock batch resize")
            size = (side, side)
            return [
                PairedView(before, after, center_masks(before["gt_segm"], size),
                           source_hw, expanded, cropped, flipped, interpolation)
                for (before, source_hw, expanded, cropped, flipped), after in zip(prepared, batch, strict=True)
            ]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
