"""Serial stock RF-DETR augmentation with paired, common-row mask supervision."""

from hashlib import sha256
import json
import pickle
import random
from threading import Lock

import numpy as np
from PIL import Image
import torch
from torch.utils.data import get_worker_info

from common_mask_supervision import ROW_ID, TARGET_FIELDS, common_targets
from trace_resize import capture_transform


PREPARED_FIELDS = TARGET_FIELDS - {ROW_ID}
MODES = ("native", "center")
_ACTIVE = Lock()


def _rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _restore(state):
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)


def _rng_bytes(state):
    python_state, numpy_state, torch_state = state
    algorithm, words, *rest = numpy_state
    return pickle.dumps((python_state, algorithm, words.tobytes(), rest, torch_state.numpy().tobytes()), protocol=4)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _tensor_bytes(tensor):
    return tensor.detach().as_subclass(torch.Tensor).contiguous().reshape(-1).view(torch.uint8)


def _tensor_digest(items):
    digest = sha256()
    for key, tensor in items:
        digest.update(_encoded([key, str(tensor.dtype), list(tensor.shape)]) + b"\n")
        digest.update(memoryview(_tensor_bytes(tensor).numpy()))
    return digest.hexdigest()


def _training_mask_digest(masks):
    # Match the frozen phase trainer's tensor digest for the selected batch mask.
    header = repr((str(masks.dtype), tuple(masks.shape))).encode()
    digest = sha256(header)
    digest.update(memoryview(_tensor_bytes(masks).numpy()))
    return digest.hexdigest()


def _prepare(image, raw):
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("Expected a CPU RGB PIL image")
    if (
        type(raw) is not dict or raw.keys() not in (PREPARED_FIELDS, TARGET_FIELDS)
        or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in raw.values())
    ):
        raise ValueError("Expected prepared CPU RF-DETR target fields")
    labels, masks = raw["labels"], raw["masks"]
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("Expected prepared integer RF-DETR labels")
    rows, hw = len(labels), (image.height, image.width)
    if (
        masks.dtype != torch.bool or tuple(masks.shape) != (rows, *hw)
        or raw["boxes"].shape != (rows, 4)
        or any(raw[key].shape != labels.shape for key in ("area", "iscrowd"))
        or raw["image_id"].dtype != torch.int64 or raw["image_id"].numel() != 1
        or any(raw[key].dtype != torch.int64 or raw[key].tolist() != list(hw) for key in ("size", "orig_size"))
    ):
        raise ValueError("Prepared RF-DETR target rows and image canvas disagree")
    ids = torch.arange(rows, dtype=torch.int64)
    if ROW_ID in raw and (raw[ROW_ID].dtype != torch.int64 or not torch.equal(raw[ROW_ID], ids)):
        raise ValueError("Prepared row IDs must be consecutive in original annotation order")
    return dict(raw, **{ROW_ID: ids})


def _check_output(image, target, raw):
    if type(target) is not dict or target.keys() != TARGET_FIELDS or any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in target.values()
    ):
        raise ValueError("Expected augmented CPU RF-DETR target fields")
    if (
        not isinstance(image, torch.Tensor) or image.device.type != "cpu" or image.dtype != torch.float32
        or image.ndim != 3 or image.shape[0] != 3 or list(image.shape[-2:]) != target["size"].tolist()
        or not torch.isfinite(image).all()
    ):
        raise ValueError("Expected a finite normalized RGB tensor on the augmented canvas")
    for key in ("image_id", "orig_size"):
        if target[key].dtype != raw[key].dtype or not torch.equal(target[key], raw[key]):
            raise ValueError("Augmentation must preserve original image identity")
    ids = target[ROW_ID]
    if (
        ids.ndim != 1 or ids.dtype != torch.int64
        or (ids < 0).any() or (ids >= len(raw["labels"])).any() or (ids[1:] <= ids[:-1]).any()
    ):
        raise ValueError("Augmentation must preserve ordered original row IDs")


def _nonempty(target):
    return int(target["masks"].flatten(1).any(dim=1).sum())


def _metadata(image, trace, raw, targets, projected, entry, after):
    native, center = targets
    shared_a, shared_b = projected
    native_live, center_live = _nonempty(native), _nonempty(center)
    common = len(shared_a[ROW_ID])
    changed = int((shared_a["masks"] != shared_b["masks"]).flatten(1).any(dim=1).sum())
    return dict(
        image_id=int(raw["image_id"].item()), raw_rows=len(raw["labels"]),
        native_present=len(native[ROW_ID]), center_present=len(center[ROW_ID]),
        native_nonempty=native_live, center_nonempty=center_live, common_nonempty=common,
        native_only_nonempty=native_live-common, center_only_nonempty=center_live-common,
        common_changed_masks=changed, crop=any(row["op"] == "crop" for row in trace),
        flipped=any(row["op"] == "hflip" and row["applied"] for row in trace),
        image_sha256=_tensor_digest((("image", image),)), trace_sha256=sha256(_encoded(trace)).hexdigest(),
        rng_before_sha256=sha256(_rng_bytes(entry)).hexdigest(), rng_after_sha256=sha256(_rng_bytes(after)).hexdigest(),
        common_row_ids_sha256=_tensor_digest(((ROW_ID, shared_a[ROW_ID]),)),
        common_nonmask_sha256=_tensor_digest((key, shared_a[key]) for key in sorted(PREPARED_FIELDS - {"masks"})),
        native_common_masks_sha256=_training_mask_digest(shared_a["masks"]),
        center_common_masks_sha256=_training_mask_digest(shared_b["masks"]),
    )


class PairedFitTransform:
    """Wrap a stateless native CPU training transform for one selected mask arm.

    Both arms run serially in the owning process with no other RNG consumers.
    The optional audit callback receives the same metadata in both arms.
    """

    def __init__(self, native_module, transform, mode, *, audit=None):
        if mode not in MODES or not callable(transform) or (audit is not None and not callable(audit)):
            raise ValueError("Expected native or center mode, a transform and an optional audit callable")
        self.native = native_module
        self.transform = transform
        self.mode = mode
        self.audit = audit

    def __call__(self, image, raw):
        if get_worker_info() is not None:
            raise RuntimeError("Paired FIT transform requires zero DataLoader workers")
        if not _ACTIVE.acquire(blocking=False):
            raise RuntimeError("Nested or concurrent paired FIT transform is unsupported")
        try:
            entry = _rng()
            try:
                return self._paired(image, raw, entry)
            except BaseException:
                _restore(entry)
                raise
        finally:
            _ACTIVE.release()

    def _paired(self, image, raw, entry):
        resize = self.native.Resize.__call__
        if getattr(resize, "_phase_trace", False) or getattr(resize, "_center_mask_resize", False):
            raise RuntimeError("Paired FIT transform cannot nest inside another mask phase patch")
        raw = _prepare(image, raw)
        images, targets, traces, after_states = [], [], [], []
        for mode in MODES:
            _restore(entry)
            independent = {key: value.clone() for key, value in raw.items()}
            with capture_transform(self.native, mode=mode) as trace:
                result_image, target = self.transform(image.copy(), independent)
            after_states.append(_rng())
            _check_output(result_image, target, raw)
            images.append(result_image)
            targets.append(target)
            traces.append(trace)
        if traces[0] != traces[1] or not any(row["op"] == "resize" for row in traces[0]):
            raise ValueError("Paired FIT augmentation must preserve the native resize trace")
        if images[0].shape != images[1].shape or not torch.equal(*map(_tensor_bytes, images)):
            raise ValueError("Paired FIT augmentation changed normalized image bytes")
        if _rng_bytes(after_states[0]) != _rng_bytes(after_states[1]):
            raise ValueError("Paired FIT augmentation changed the CPU random stream")
        projected = common_targets(*targets)
        if self.audit is not None:
            self.audit(_metadata(images[0], traces[0], raw, targets, projected, entry, after_states[0]))
        _restore(after_states[0])
        chosen = MODES.index(self.mode)
        selected = projected[chosen]
        del selected[ROW_ID]
        return images[chosen], selected
