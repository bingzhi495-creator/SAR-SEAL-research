"""Select mask interpolation for the pinned native RF-DETR training pipeline."""

from contextlib import contextmanager
from unittest.mock import patch

from trace_resize import center_resize_masks


@contextmanager
def mask_resize_phase(native_module, mode):
    """Scope center-aligned mask resizing to a serial, CPU-transform run."""
    if mode not in ("native", "center"):
        raise ValueError("Expected mask resize mode native or center")
    resize_call = native_module.Resize.__call__
    if getattr(resize_call, "_center_mask_resize", False):
        raise RuntimeError("Nested mask resize phase is unsupported")
    if mode == "native":
        yield
        return

    def resize(self, image, target=None):
        if target is None or "masks" not in target:
            return resize_call(self, image, target)
        without_masks = target.copy()
        masks = without_masks.pop("masks")
        resized_image, resized_target = resize_call(self, image, without_masks)
        # Keep native field order and bookkeeping while replacing the mask.
        return resized_image, {**target, **resized_target, "masks": center_resize_masks(masks, self.size)}

    resize._center_mask_resize = True
    with patch.object(native_module.Resize, "__call__", resize):
        yield
