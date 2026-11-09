"""Observe the pinned RF-DETR geometry and optionally change only mask Resize."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F


def center_resize_masks(masks, size):
    """Resize Boolean instance masks with explicit nearest-exact on plain tensors."""
    if masks.ndim != 3 or masks.dtype != torch.bool or masks.device.type != "cpu":
        raise ValueError("Expected CPU Boolean instance masks shaped N,H,W")
    if masks.shape[0] == 0:
        return masks.new_empty((0, *size))
    plain = masks.as_subclass(torch.Tensor)
    resized = F.interpolate(plain[:, None].float(), size=size, mode="nearest-exact")
    return resized[:, 0].bool()


@contextmanager
def capture_transform(native_module, *, mode):
    """Trace each native geometry call during a serial, CPU-only replay.

    Resize, crop, and horizontal flip are recorded in execution order. Resize
    sizes and crop offsets describe the actual canvas operations. A flip entry
    includes its applied flag even when the native random draw chooses a no-op.
    Native crop row selection remains active and may respond to changed masks.
    """
    if mode not in ("native", "center"):
        raise ValueError("Expected mask resize mode native or center")
    resize_call = native_module.Resize.__call__
    crop_call = native_module.crop
    flip_call = native_module.RandomHorizontalFlip.__call__
    image_size = native_module._image_size
    if getattr(resize_call, "_phase_trace", False):
        raise RuntimeError("Nested transform capture is unsupported")
    trace = []

    def resize(self, image, target=None):
        before = image_size(image)
        if mode == "center" and target is not None and "masks" in target:
            def resize_masks(masks, _):
                return center_resize_masks(masks, self.size)

            # RF-DETR's Resize calls the shared helper once for masks. Scope
            # replacement to this call, so crop and flip retain stock behavior.
            with patch.object(native_module, "_apply_to_masks", resize_masks):
                result = resize_call(self, image, target)
        else:
            result = resize_call(self, image, target)
        trace.append(dict(op="resize", in_hw=before, out_hw=image_size(result[0])))
        return result

    def crop(image, target, top, left, height, width):
        before = image_size(image)
        result = crop_call(image, target, top, left, height, width)
        trace.append(dict(op="crop", in_hw=before, out_hw=image_size(result[0]),
                          top=int(top), left=int(left)))
        return result

    def flip(self, image, target=None):
        before = image_size(image)
        result = flip_call(self, image, target)
        # The pinned native no-op returns its input image object. Its actual
        # Pillow or tensor flip returns a new image, even for symmetric pixels.
        trace.append(dict(op="hflip", in_hw=before, out_hw=image_size(result[0]),
                          applied=result[0] is not image))
        return result

    resize._phase_trace = True
    with (patch.object(native_module.Resize, "__call__", resize),
          patch.object(native_module, "crop", crop),
          patch.object(native_module.RandomHorizontalFlip, "__call__", flip)):
        yield trace
