"""Pair stepwise-center and one-pass masks on fixed B crop support."""

import torch
from torch.utils.data import get_worker_info

from common_mask_supervision import ROW_ID, common_targets
from one_pass_fit_transform import substitute_one_pass_masks
from paired_fit_transform import _ACTIVE, _check_output, _metadata, _prepare, _restore, _rng
from trace_resize import capture_transform


MODES = ("stepwise_center", "one_pass")


class OnePassCommonPairTransform:
    """Apply stock center augmentation once and project both masks to common rows.

    The public modes select B (stepwise center) and C|B (one pass on B's
    ordered crop rows). Both are projected onto their final nonempty
    intersection. The audit uses the existing consumed FIT ledger fields:
    the positional native_* fields describe B, and center_* fields describe
    C|B before projection or its common masks for *_common_masks_sha256.
    The audit does not represent natural C support.
    """

    def __init__(self, native_module, transform, mode, *, audit=None):
        if mode not in MODES or not callable(transform) or (audit is not None and not callable(audit)):
            raise ValueError("Expected one-pass common FIT mode, a stock transform and an optional audit callable")
        self.native = native_module
        self.transform = transform
        self.mode = mode
        self.audit = audit

    def __call__(self, image, raw):
        if get_worker_info() is not None:
            raise RuntimeError("One-pass common FIT transform requires zero DataLoader workers")
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
            raise RuntimeError("One-pass common FIT transform cannot nest inside a mask phase patch")

        raw = _prepare(image, raw)
        independent = {key: value.clone() for key, value in raw.items()}
        with capture_transform(self.native, mode="center") as trace:
            image, stepwise = self.transform(image.copy(), independent)
        after = _rng()
        _check_output(image, stepwise, raw)

        one_pass = substitute_one_pass_masks(raw["masks"], stepwise, trace)
        if not any(row["op"] == "crop" for row in trace) and not torch.equal(stepwise["masks"], one_pass["masks"]):
            raise ValueError("Direct FIT masks must match exactly for stepwise and one-pass center sampling")
        targets = stepwise, one_pass
        projected = common_targets(*targets)
        if self.audit is not None:
            self.audit(_metadata(image, trace, raw, targets, projected, entry, after))

        _restore(after)
        selected = projected[MODES.index(self.mode)]
        del selected[ROW_ID]
        return image, selected
