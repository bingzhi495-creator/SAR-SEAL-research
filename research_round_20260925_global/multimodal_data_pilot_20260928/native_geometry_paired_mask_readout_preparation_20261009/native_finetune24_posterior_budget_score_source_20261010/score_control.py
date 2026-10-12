"""Untrained, fixed-mask posterior-budget scoring control.

The caller supplies original-raster CPU FP32 logits, the actual decoded GPU
exported RLE support, and its actual class probabilities in query-ordinal order.
No mask is thresholded here. All query rows are retained, including empty masks.
CPU rendering of saved M200 is not guaranteed bitwise equal to GPU rendering.

This is a smoothed soft-IoU confidence formula, not a novel/calibrated IoU head.
Without epsilon it is SDC/(2-SDC), where SDC is Eq22 of Borges et al. (2026):
https://link.springer.com/article/10.1007/s10994-026-07096-w
With epsilon, the same identity uses SDC_eps=2I/(A+sum(p)+eps).
The ratio of expected pixel budgets is not generally expected IoU.
"""
import math


def posterior_budget_scores(torch, original_logits, exported_RLE_support,
                            class_probability, eps=1e-6):
    """Return two CPU FP32 [Q] score vectors; no training, GT, or mask mutation.

    Inputs: logits/support[Q,H,W], class_probability[Q]. The function also accepts
    query chunks; a future consumer must rejoin all actual 100 query ordinals.
    """
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite positive; the control uses 1e-6')
    if not isinstance(original_logits, torch.Tensor) or original_logits.ndim != 3 or \
            original_logits.dtype != torch.float32 or original_logits.device.type != 'cpu' or \
            any(size <= 0 for size in original_logits.shape) or not bool(torch.isfinite(original_logits).all()):
        raise ValueError('logits must be finite CPU FP32[Q,H,W]')
    if not isinstance(exported_RLE_support, torch.Tensor) or exported_RLE_support.dtype != torch.bool or \
            exported_RLE_support.device.type != 'cpu' or exported_RLE_support.shape != original_logits.shape:
        raise ValueError('actual exported support must be CPU bool[Q,H,W]')
    if not isinstance(class_probability, torch.Tensor) or class_probability.dtype != torch.float32 or \
            class_probability.device.type != 'cpu' or class_probability.shape != original_logits.shape[:1] or \
            not bool(torch.isfinite(class_probability).all()) or \
            not bool(((class_probability >= 0) & (class_probability <= 1)).all()):
        raise ValueError('actual class probabilities must be CPU FP32[Q] in [0,1]')
    with torch.no_grad():
        probability = original_logits.sigmoid()
        support = exported_RLE_support.to(dtype=torch.float32)
        intersection = (support * probability).sum(dim=(1, 2))
        union = support.sum(dim=(1, 2)) + ((1 - support) * probability).sum(dim=(1, 2))
        quality = intersection / (union + eps)
        return dict(posterior_budget=quality, class_times_posterior_budget=class_probability * quality)
