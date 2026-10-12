# -*- coding: utf-8 -*-
"""StashLowResAmplitude -- carry per-instance SAR amplitude to loss time.

SAR Loss A (intensity-weighted BCE) up-weights in-GT pixels that are DARK in the
SAR image. The mmdet mask head's `loss_by_feat` does NOT receive `batch_inputs`,
so the amplitude must ride along inside `gt_instances`. This transform, inserted
AFTER LoadImageFromFile / LoadAnnotations and BEFORE PackDetInputs, computes a
single-channel amplitude map from the (post-resize/crop/flip) image and attaches
it per instance, masked to that instance's GT mask, as `gt_masks_amp` -- a
BitmapMasks-like float field that PackDetInputs will carry into
`data_samples.gt_instances`.

Because it is computed at the END of the geometric pipeline (after the same
flip/resize/crop the masks went through), it is already aligned to the masks; no
separate warping bookkeeping is needed. The head downsamples it to the mask-
target grid via point_sample at the loss points.

Robust per-image normalisation: amplitude / p99(amplitude), clamped to [0,1].
"""
from __future__ import annotations

import numpy as np

from mmengine.structures import InstanceData

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs
from mmdet.structures.bbox import BaseBoxes


@TRANSFORMS.register_module()
class PackDetInputsSAR(PackDetInputs):
    """PackDetInputs that also packs per-instance `amp_maps` into gt_instances.

    Applies the same gt_ignore_flags valid-index filtering as the stock packer so
    the amplitude rows stay 1:1 aligned with the kept instances (masks/labels).
    """

    def transform(self, results: dict) -> dict:
        amp = results.pop('amp_maps', None)
        dense = results.pop('dense_amp', None)            # frame-level (H,W), for SCOPE
        packed = super().transform(results)
        import torch
        import numpy as np
        if dense is not None:                              # attach as a plain field on the sample
            packed['data_samples'].dense_amp = torch.from_numpy(np.asarray(dense)).float()[None]
        if amp is None:
            return packed
        amp_t = torch.from_numpy(np.asarray(amp)).float()  # (num_gts_pre, h, w)
        gi = packed['data_samples'].gt_instances
        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            if amp_t.shape[0] >= len(valid_idx):
                amp_t = amp_t[valid_idx]
        # only attach if the count matches the kept masks
        n_inst = len(gi.get('labels', [])) if 'labels' in gi else amp_t.shape[0]
        if amp_t.shape[0] == n_inst:
            gi.amp_maps = amp_t
        return packed


@TRANSFORMS.register_module()
class StashDenseAmplitude:
    """Attach a FRAME-LEVEL normalised SAR amplitude map (post-augmentation) for SCOPE.

    Computed at the END of the geometric pipeline so it is flip/resize/crop-aligned to
    the image the model sees. The head pads it to the padded input size and downsamples
    it to the mask-feature grid. Robust per-image norm: amplitude / p99, clamped [0,1].
    """

    def __init__(self, key: str = 'dense_amp'):
        self.key = key

    def __call__(self, results: dict) -> dict:
        img = results.get('img', None)
        if img is None:
            return results
        arr = np.asarray(img).astype(np.float32)
        amp = arr.mean(axis=2) if arr.ndim == 3 else arr   # grayscale amplitude
        p99 = np.percentile(amp, 99) if amp.size else 1.0
        results[self.key] = np.clip(amp / max(p99, 1e-6), 0.0, 1.0).astype(np.float32)  # (H,W)
        return results


@TRANSFORMS.register_module()
class StashLowResAmplitude:
    """Attach per-instance normalised amplitude maps for SAR Loss A."""

    def __init__(self, key: str = 'amp_maps'):
        self.key = key

    def __call__(self, results: dict) -> dict:
        img = results.get('img', None)
        masks = results.get('gt_masks', None)
        if img is None or masks is None:
            return results
        arr = np.asarray(img).astype(np.float32)
        if arr.ndim == 3:
            amp = arr.mean(axis=2)        # grayscale amplitude (SAR is ~single-ch)
        else:
            amp = arr
        p99 = np.percentile(amp, 99) if amp.size else 1.0
        amp = np.clip(amp / max(p99, 1e-6), 0.0, 1.0)   # (H,W) in [0,1]

        m = masks.to_ndarray() if hasattr(masks, 'to_ndarray') else np.asarray(masks)
        if m.ndim == 2:
            m = m[None]
        H, W = amp.shape
        per_inst = []
        for i in range(m.shape[0]):
            mi = m[i].astype(np.float32)
            if mi.shape != (H, W):
                # mask stored at a different resolution -> skip masking, use amp
                per_inst.append(amp.copy())
            else:
                per_inst.append(amp * mi)   # amplitude inside the hull only
        results[self.key] = np.stack(per_inst, 0) if per_inst else \
            np.zeros((0, H, W), np.float32)
        return results
