"""Opt in to equivalent direct-mask projection on the pinned HRSID FIT path."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rfdetr.datasets import _torchvision as stock_native, transforms as stock_transforms
import torch
from torch.nn.modules import module as torch_modules
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import ToDtype, ToImage

from common_mask_supervision import ROW_FIELDS, ROW_ID, _live_rows, common_targets
from one_pass_common_pair_transform import MODES, OnePassCommonPairTransform
from one_pass_fit_transform import substitute_one_pass_masks
from paired_fit_transform import (
    PREPARED_FIELDS, _check_output, _encoded, _metadata, _prepare, _restore, _rng, _rng_bytes,
    _tensor_digest, _training_mask_digest,
)
from trace_resize import capture_transform


SOURCE_HW = (800, 800)
OUTPUT_HW = (576, 576)
PINNED_SOURCES = (
    (stock_native, "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56"),
    (stock_transforms, "a4070434e1cfcb50040f1c84abddf13e37fb8f93953c76c035c32058f8b6d2a1"),
)
STOCK_CALLS = tuple((kind, "__call__", kind.__call__) for kind in (
    stock_native.Compose, stock_native.RandomSelect, stock_native.RandomChoice, stock_native.Resize,
    stock_native.RandomResize, stock_native.RandomSizedCrop, stock_native.RandomHorizontalFlip,
    stock_transforms.Normalize,
)) + (
    (stock_native, "crop", stock_native.crop),
    (stock_native, "_apply_to_masks", stock_native._apply_to_masks),
)


def _pinned_runtime(native):
    if native is not stock_native:
        return False
    try:
        return (
            version("rfdetr") == "1.12.0"
            and version("torch") == "2.10.0+cu130"
            and version("torchvision") == "0.25.0+cu130"
            and all(sha256(Path(module.__file__).read_bytes()).hexdigest() == digest
                    for module, digest in PINNED_SOURCES)
        )
    except (OSError, PackageNotFoundError):
        return False


def _sequence(transform, kind, length):
    if type(transform) is not kind:
        return None
    steps = vars(transform).get("transforms")
    return steps if type(steps) is list and len(steps) == length else None


def _stock_tensor_transform(transform, kind):
    return (
        type(transform) is kind and not transform._forward_hooks and not transform._forward_pre_hooks
        and not vars(transform).keys() & {
            "forward", "transform", "check_inputs", "make_params", "_needs_transform_list", "_transformed_types",
            "_compiled_call_impl", "_call_impl", "_wrapped_call_impl",
        }
    )


def _stock_pipeline(transform):
    steps = _sequence(transform, stock_native.Compose, 5)
    if (steps is None or torch_modules._global_forward_hooks or torch_modules._global_forward_pre_hooks
            or any(getattr(owner, name) is not original for owner, name, original in STOCK_CALLS)):
        return False
    select, flip, to_image, to_dtype, normalize = steps
    if not (
        type(select) is stock_native.RandomSelect and getattr(select, "p", None) == .5
        and type(flip) is stock_native.RandomHorizontalFlip and getattr(flip, "p", None) == .5
        and getattr(flip, "keypoint_flip_pairs", None) == []
        and _stock_tensor_transform(to_image, ToImage) and _stock_tensor_transform(to_dtype, ToDtype)
        and getattr(to_dtype, "dtype", None) is torch.float32 and getattr(to_dtype, "scale", None) is True
        and type(normalize) is stock_transforms.Normalize
        and _stock_tensor_transform(getattr(normalize, "_normalize", None), Normalize)
    ):
        return False

    direct = _sequence(getattr(select, "transform1", None), stock_native.RandomChoice, 1)
    crop = _sequence(getattr(select, "transform2", None), stock_native.Compose, 2)
    if direct is None or crop is None:
        return False
    resize, = direct
    crop_resize, crop_choice = crop
    crop_steps = _sequence(crop_choice, stock_native.RandomChoice, 1)
    return (
        type(resize) is stock_native.Resize and getattr(resize, "size", None) == OUTPUT_HW
        and type(crop_resize) is stock_native.RandomResize and getattr(crop_resize, "sizes", None) == [400, 500, 600]
        and getattr(crop_resize, "max_size", False) is None
        and crop_steps is not None and type(crop_steps[0]) is stock_native.RandomSizedCrop
        and getattr(crop_steps[0], "min_max_height", None) == (384, 600)
        and getattr(crop_steps[0], "size", None) == OUTPUT_HW
    )


def _certified_direct(trace):
    # One 800 -> 576 step maps centres to 25(2j+1)/36. Their minimum
    # distance to an integer is 1/36, above the pinned float32 CPU
    # kernels' conservative bound of 800 * 2**-21. A trailing flip
    # simply reverses the already equal column indices.
    return (
        len(trace) == 2
        and trace[0] == dict(op="resize", in_hw=SOURCE_HW, out_hw=OUTPUT_HW)
        and trace[1] == dict(op="hflip", in_hw=OUTPUT_HW, out_hw=OUTPUT_HW,
                             applied=trace[1].get("applied"))
        and type(trace[1]["applied"]) is bool
    )


def _direct_metadata(image, trace, raw, stepwise, selected, entry, after):
    # Same fields and byte encodings as the paired adapter, computed once.
    common = len(selected[ROW_ID])
    masks_sha256 = _training_mask_digest(selected["masks"])
    return dict(
        image_id=int(raw["image_id"].item()), raw_rows=len(raw["labels"]),
        native_present=len(stepwise[ROW_ID]), center_present=len(stepwise[ROW_ID]),
        native_nonempty=common, center_nonempty=common, common_nonempty=common,
        native_only_nonempty=0, center_only_nonempty=0, common_changed_masks=0,
        crop=False, flipped=trace[1]["applied"],
        image_sha256=_tensor_digest((("image", image),)), trace_sha256=sha256(_encoded(trace)).hexdigest(),
        rng_before_sha256=sha256(_rng_bytes(entry)).hexdigest(), rng_after_sha256=sha256(_rng_bytes(after)).hexdigest(),
        common_row_ids_sha256=_tensor_digest(((ROW_ID, selected[ROW_ID]),)),
        common_nonmask_sha256=_tensor_digest((key, selected[key]) for key in sorted(PREPARED_FIELDS - {"masks"})),
        native_common_masks_sha256=masks_sha256, center_common_masks_sha256=masks_sha256,
    )


class OnePassCommonDirectFastTransform(OnePassCommonPairTransform):
    """Reuse the audited stepwise result only for certified stock direct visits."""

    def __init__(self, native_module, transform, mode, *, audit=None):
        super().__init__(native_module, transform, mode, audit=audit)
        self._stock_runtime = _pinned_runtime(native_module)
        self.fast_direct_visits = 0

    def _apply(self, image, raw, entry):
        if not self._stock_runtime or getattr(image, "size", None) != SOURCE_HW or not _stock_pipeline(self.transform):
            return super()._apply(image, raw, entry)

        resize = self.native.Resize.__call__
        if getattr(resize, "_phase_trace", False) or getattr(resize, "_center_mask_resize", False):
            raise RuntimeError("One-pass common FIT transform cannot nest inside a mask phase patch")

        raw = _prepare(image, raw)
        independent = {key: value.clone() for key, value in raw.items()}
        with capture_transform(self.native, mode="center") as trace:
            image, stepwise = self.transform(image.copy(), independent)
        after = _rng()
        _check_output(image, stepwise, raw)

        fast_direct = _certified_direct(trace)
        if fast_direct:
            _, live = _live_rows(stepwise)
            selected = {**stepwise, **{key: stepwise[key][live] for key in ROW_FIELDS}}
            if self.audit is not None:
                self.audit(_direct_metadata(image, trace, raw, stepwise, selected, entry, after))
        else:
            one_pass = substitute_one_pass_masks(raw["masks"], stepwise, trace)
            if not any(row["op"] == "crop" for row in trace) and not torch.equal(stepwise["masks"], one_pass["masks"]):
                raise ValueError("Direct FIT masks must match exactly for stepwise and one-pass center sampling")
            targets = stepwise, one_pass
            projected = common_targets(*targets)
            if self.audit is not None:
                self.audit(_metadata(image, trace, raw, targets, projected, entry, after))
            selected = projected[MODES.index(self.mode)]

        _restore(after)
        del selected[ROW_ID]
        self.fast_direct_visits += int(fast_direct)
        return image, selected
