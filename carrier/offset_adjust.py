# -*- coding: utf-8 -*-
"""Offset-Adjusted Mask2Former -- a differentiable constraint on the MSDeformAttn
sampling OFFSETS inside Mask2Former's pixel decoder.

Motivation (SAR near-shore small ships): the deformable-attention encoder in the
MSDeformAttnPixelDecoder predicts, per query/head/level, `num_points` 2-D sampling
OFFSETS that are added to a reference point to gather context. On small, low-contrast
inshore ships these offsets tend to DIVERGE into bright pier/harbour clutter, so the
query aggregates background instead of the compact hull -> misses. This plug-in
constrains those predicted offsets so the sampling points stay near the compact
foreground ("query nearby").

Reference: "Query Nearby: Offset-Adjusted Mask2Former enhances small-organ
segmentation" (arXiv:2506.05897, Jun 2025). The companion repo
(github.com/earis/Offsetadjustment_Background-location_Decoder_) returned HTTP 404
at implementation time, and the paper gives the three strategies only qualitatively
(no equations / hyper-params). We therefore implement them from the STANDARD
definition and document the exact math here. The paper reports strategy 3
("Softmax x constant > 1", labelled "Sigmoid*2", constant 2) as the BEST variant.

Three selectable strategies operate on the per-point offset vector o = (dx, dy):
  * 'clamp'           : threshold-clamp -- rescale each offset so its L2 norm is
                        capped at `clamp_thr` (bounds how far a point may drift):
                            o <- o * min(1, clamp_thr / ||o||)
  * 'softmax_retract' : softmax over the `num_points` axis of NEGATIVE magnitude
                        gives per-point gates in (0,1) that sum to 1 across points;
                        multiplying retracts every offset toward the reference
                        (near points weighted up, far points suppressed):
                            g = softmax(-||o||, dim=points);  o <- o * g
  * 'softmax_scale'   : strategy above, then multiply by a learnable scale (init
                        `softmax_scale`, paper uses 2) to re-open the point spread:
                            o <- o * g * scale
  * 'none'            : identity -- BYTE-IDENTICAL to stock (GATE-0).

IMPORTANT: this file implements ONLY the offset constraint. The paper's auxiliary
FCN / high-res coarse-location head is deliberately NOT ported -- that is the
a1/refine family which this project already measured as harmful/dead.

Design: at 'none' the module is byte-identical to the stock
`MultiScaleDeformableAttention` and produces a state_dict with IDENTICAL keys, so
the strong COCO-M2F init (`load_from`) transfers unchanged. Active constraints add
at most one scalar param per attention layer (`sampling_offsets.scale`, softmax_scale
only). Pure torch; CPU-safe.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.models.layers.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from mmdet.registry import MODELS

_CONSTRAINTS = ('none', 'clamp', 'softmax_retract', 'softmax_scale')


class ConstrainedSamplingOffsets(nn.Linear):
    """Drop-in replacement for the ``sampling_offsets`` Linear.

    It IS an ``nn.Linear`` (so state_dict keys stay ``sampling_offsets.weight`` /
    ``sampling_offsets.bias`` -- COCO checkpoint compatible), but its forward reshapes
    the flat output to (..., num_heads, num_levels, num_points, 2), applies the offset
    constraint, and flattens back. At ``offset_constraint='none'`` it returns the raw
    linear output untouched (byte-identical to stock).
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int,
                 num_levels: int, num_points: int,
                 offset_constraint: str = 'none', clamp_thr: float = 4.0,
                 softmax_scale: float = 2.0, eps: float = 1e-6,
                 bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        assert offset_constraint in _CONSTRAINTS, \
            f'offset_constraint must be one of {_CONSTRAINTS}, got {offset_constraint}'
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.offset_constraint = offset_constraint
        self.clamp_thr = float(clamp_thr)
        self.eps = float(eps)
        # A learnable scale ONLY for softmax_scale -> keeps 'none'/'clamp'/'retract'
        # state_dict byte-identical to the stock Linear.
        if offset_constraint == 'softmax_scale':
            self.scale = nn.Parameter(torch.tensor(float(softmax_scale)))

    def _constrain(self, off: torch.Tensor) -> torch.Tensor:
        """off: (..., num_heads, num_levels, num_points, 2)."""
        c = self.offset_constraint
        if c == 'clamp':
            # Cap each per-point displacement at clamp_thr (in feature-grid units).
            mag = off.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            factor = (self.clamp_thr / mag).clamp(max=1.0)   # <=1, only shrinks
            return off * factor
        # softmax over the points axis (num_points is dim=-2 after keepdim norm).
        mag = off.norm(dim=-1, keepdim=True)                 # (..., num_points, 1)
        gate = torch.softmax(-mag, dim=-2)                   # sums to 1 over points
        if c == 'softmax_retract':
            return off * gate
        if c == 'softmax_scale':
            return off * gate * self.scale
        raise ValueError(f'unknown offset_constraint {c!r}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)                             # (..., H*L*P*2)
        if self.offset_constraint == 'none':
            return out                                       # byte-identical to stock
        lead = out.shape[:-1]
        off = out.view(*lead, self.num_heads, self.num_levels, self.num_points, 2)
        off = self._constrain(off)
        return off.reshape(*lead, self.num_heads * self.num_levels * self.num_points * 2)


@MODELS.register_module()
class OffsetAdjustMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    """MultiScaleDeformableAttention whose predicted sampling offsets are constrained.

    Composed by subclassing: ``__init__`` builds the stock module (so its whole
    ``forward`` is inherited verbatim), then swaps ``self.sampling_offsets`` for a
    :class:`ConstrainedSamplingOffsets` carrying the SAME initialized weights. At
    ``offset_constraint='none'`` the swap is a no-op numerically -> GATE-0 byte-exact.
    """

    def __init__(self, *args, offset_constraint: str = 'none',
                 clamp_thr: float = 4.0, softmax_scale: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert offset_constraint in _CONSTRAINTS, \
            f'offset_constraint must be one of {_CONSTRAINTS}, got {offset_constraint}'
        self.offset_constraint = offset_constraint
        self.clamp_thr = float(clamp_thr)
        self.softmax_scale = float(softmax_scale)

        old = self.sampling_offsets                          # stock nn.Linear (init'd)
        new = ConstrainedSamplingOffsets(
            old.in_features, old.out_features,
            num_heads=self.num_heads, num_levels=self.num_levels,
            num_points=self.num_points, offset_constraint=offset_constraint,
            clamp_thr=clamp_thr, softmax_scale=softmax_scale,
            bias=old.bias is not None)
        with torch.no_grad():                                # preserve the grid init
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.sampling_offsets = new


@MODELS.register_module()
class OffsetAdjustMSDeformAttnPixelDecoder(MSDeformAttnPixelDecoder):
    """MSDeformAttnPixelDecoder that injects the offset constraint into every encoder
    layer's deformable self-attention.

    The stock ``DeformableDetrTransformerEncoderLayer`` hardcodes
    ``MultiScaleDeformableAttention(**self_attn_cfg)`` (no registry ``type`` hook), so
    we swap post-construction: build the stock decoder via ``super().__init__`` (keeps
    every state_dict key), then replace each ``encoder.layers[i].self_attn`` with an
    :class:`OffsetAdjustMultiScaleDeformableAttention` rebuilt from the old module's
    attributes and loaded with its exact weights. At ``offset_constraint='none'`` no
    swap happens -> byte-identical to stock.
    """

    def __init__(self, *args, offset_constraint: str = 'none',
                 clamp_thr: float = 4.0, softmax_scale: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert offset_constraint in _CONSTRAINTS, \
            f'offset_constraint must be one of {_CONSTRAINTS}, got {offset_constraint}'
        self.offset_constraint = offset_constraint
        if offset_constraint == 'none':
            return                                           # keep stock attns

        for layer in self.encoder.layers:
            old = layer.self_attn
            # Reconstruct the exact stock attention config from the built module,
            # so we do not depend on config plumbing being stored on the layer.
            attn_kwargs = dict(
                embed_dims=old.embed_dims,
                num_heads=old.num_heads,
                num_levels=old.num_levels,
                num_points=old.num_points,
                im2col_step=old.im2col_step,
                dropout=old.dropout.p,
                batch_first=old.batch_first,
                norm_cfg=old.norm_cfg,
                value_proj_ratio=old.value_proj.out_features / old.embed_dims,
            )
            new = OffsetAdjustMultiScaleDeformableAttention(
                offset_constraint=offset_constraint, clamp_thr=clamp_thr,
                softmax_scale=softmax_scale, **attn_kwargs)
            # strict=False: the (optional) sampling_offsets.scale param is new and
            # absent from the stock/ COCO state_dict; every other key matches exactly.
            new.load_state_dict(old.state_dict(), strict=False)
            new.to(old.value_proj.weight.device)
            layer.self_attn = new
