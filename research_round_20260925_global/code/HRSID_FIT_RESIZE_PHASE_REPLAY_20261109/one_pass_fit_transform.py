"""Sample final FIT instance masks once from the original prepared raster."""

from hashlib import sha256

import torch
from torch.utils.data import get_worker_info

from common_mask_supervision import ROW_ID, TARGET_FIELDS, _live_rows
from paired_fit_transform import (
    _ACTIVE, _check_output, _encoded, _prepare, _restore, _rng, _rng_bytes, _tensor_digest, _training_mask_digest,
)
from reference_resize import source_indices
from replay_fit import normalized_trace
from trace_resize import capture_transform


def substitute_one_pass_masks(raw_masks, target, trace):
    """Rebuild only masks on a traced target's existing original row IDs.

    Passing a stepwise-center target retains its existing crop support. The
    natural one-pass transform instead supplies a mask-free box carrier.
    """
    if (not isinstance(raw_masks, torch.Tensor) or raw_masks.ndim != 3 or raw_masks.dtype != torch.bool
            or raw_masks.device.type != "cpu" or min(raw_masks.shape[-2:]) <= 0):
        raise ValueError("Expected original CPU Boolean instance masks")
    if type(target) is not dict or target.keys() not in (TARGET_FIELDS, TARGET_FIELDS - {"masks"}):
        raise ValueError("Expected stock target fields with original row identities")
    ids = target[ROW_ID]
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.int64 or ids.device.type != "cpu"
            or (ids < 0).any() or (ids >= len(raw_masks)).any() or (ids[1:] <= ids[:-1]).any()):
        raise ValueError("Expected ordered original mask row IDs")
    steps = normalized_trace(trace)
    rows, cols = source_indices(steps, raw_masks.shape[-2:])
    row_index = torch.as_tensor(rows, dtype=torch.int64)
    col_index = torch.as_tensor(cols, dtype=torch.int64)
    original = raw_masks.as_subclass(torch.Tensor)
    masks = original[ids[:, None, None], row_index[None, :, None], col_index[None, None, :]]
    result = dict(target, masks=masks)
    _live_rows(result)
    return result


class OnePassFitTransform:
    """Carry stock FIT box geometry with no intermediate mask rasterization.

    The returned rows are the stock positive-area box rows. A later native
    empty-mask guard removes rows whose final one-pass mask is empty.
    """

    def __init__(self, native_module, transform, *, audit=None):
        if not callable(transform) or (audit is not None and not callable(audit)):
            raise ValueError("Expected a stock FIT transform and optional audit callable")
        self.native = native_module
        self.transform = transform
        self.audit = audit

    def __call__(self, image, raw):
        if get_worker_info() is not None:
            raise RuntimeError("One-pass FIT transform requires zero DataLoader workers")
        if not _ACTIVE.acquire(blocking=False):
            raise RuntimeError("Nested or concurrent FIT transform is unsupported")
        try:
            entry = _rng()
            try:
                return self._apply(image, raw, entry)
            except BaseException:
                _restore(entry)
                raise
        finally:
            _ACTIVE.release()

    def _apply(self, image, raw, entry):
        resize = self.native.Resize.__call__
        if getattr(resize, "_phase_trace", False) or getattr(resize, "_center_mask_resize", False):
            raise RuntimeError("One-pass FIT transform cannot nest inside a mask phase patch")
        raw = _prepare(image, raw)
        carrier = {key: value.clone() for key, value in raw.items() if key != "masks"}
        with capture_transform(self.native, mode="native") as trace:
            image, target = self.transform(image.copy(), carrier)
        after = _rng()
        if type(target) is not dict or "masks" in target:
            raise ValueError("Stock box carrier must remain mask-free through augmentation")
        target = substitute_one_pass_masks(raw["masks"], target, trace)
        _check_output(image, target, raw)
        if self.audit is not None:
            ids, masks = target[ROW_ID], target["masks"]
            self.audit(dict(
                image_id=int(raw["image_id"].item()), raw_rows=len(raw[ROW_ID]), positive_box_rows=len(ids),
                nonempty_rows=int(masks.flatten(1).any(dim=1).sum()),
                crop=any(row["op"] == "crop" for row in trace),
                flipped=any(row["op"] == "hflip" and row["applied"] for row in trace),
                image_sha256=_tensor_digest((("image", image),)),
                trace_sha256=sha256(_encoded(trace)).hexdigest(),
                rng_before_sha256=sha256(_rng_bytes(entry)).hexdigest(),
                rng_after_sha256=sha256(_rng_bytes(after)).hexdigest(),
                retained_row_ids_sha256=_tensor_digest(((ROW_ID, ids),)),
                nonmask_sha256=_tensor_digest((key, target[key]) for key in sorted(TARGET_FIELDS - {ROW_ID, "masks"})),
                masks_sha256=_training_mask_digest(masks),
            ))
        _restore(after)
        del target[ROW_ID]
        return image, target
