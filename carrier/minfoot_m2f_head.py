# -*- coding: utf-8 -*-
"""Min-footprint masked attention: fix the EMPTY-attn-mask full-image fallback
(small-target pathology) with a k x k "minimal footprint floor". ZERO new params.

Pathology (SAR small ships)
---------------------------
Mask2Former's masked cross-attention restricts each query to its previous-layer
predicted mask, interpolated to each memory level (stride 8/16/32). A 9x18 px
ship is ~0.3x0.6 px at stride 32 -- after bilinear interpolate + the 0.5
sigmoid threshold the attention footprint can be ALL-EMPTY. mmdet 3.3.0 then
falls back to FULL-IMAGE attention:

  mmdet/models/dense_heads/mask2former_head.py, forward(), L437-439:
      # if a mask is all True(all background), then set it all False.
      mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
      attn_mask = attn_mask & mask_sum

  (replicated in carrier/seal_m2f_head.py, SealM2FHead.forward(), L141:
      attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False)

Convention: attn_mask True = BLOCKED (mmcv MultiheadAttention semantics), so an
all-True row means "this query may attend NOWHERE"; mmdet opens the WHOLE image
to avoid a softmax-over-(-inf) NaN. That full-image fallback dilutes exactly the
small-target queries that most need focus across the entire scene clutter.

Fix ("minimal footprint floor", parameter-free)
-----------------------------------------------
When a query's attn mask is empty at a level, do NOT fall back to the full
image: open a k x k window centred on the CENTROID of that query's CURRENT
predicted mask, computed at the mask_pred (1/4, mask_features) resolution --
which still retains the peak even when the interpolated level mask dies.
Centroid = binary centroid of (logit > 0); if even the 1/4-res mask is empty,
the argmax of the logit map (the query's best guess) is used. Coordinates are
mapped 1/4-res -> level grid with the align_corners=False pixel-centre
convention (matching the F.interpolate call that built the attn mask).

Because the fixed rows are no longer all-True, the stock full-image fallback in
forward() (cited above) never fires for them -- the fix composes with BOTH the
stock Mask2FormerHead.forward and SealM2FHead.forward without touching either.
Rows with a non-empty footprint are bitwise UNTOUCHED.

Design contract (preservation law)
----------------------------------
* ``min_footprint=0`` (default OFF): ``_forward_head`` returns the stock tensors
  unmodified -- bit-identical to the parent head, COCO weights load 1:1 (the
  knob adds NO parameters, state_dict identical). GATE-0:
  carrier/tests/cpu_test_minfoot.py (bitwise forward diff == 0).
* ``min_footprint=k`` (odd, e.g. 7): ONLY all-True (empty) attn rows change;
  each gets a k x k window (border-clamped, always >= 1 open key).
* ``minfoot_debug=True``: log per-level empty counts + footprint coordinates
  into ``self._minfoot_log`` (list; caller clears). Off by default (no sync).

Pure torch; CPU-safe. Mirrors the zero-default-knob subclassing pattern of
carrier/strip_addon.py / carrier/seal_m2f_head.py.
"""
from __future__ import annotations

import torch

from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.registry import MODELS

from carrier.seal_m2f_head import SealM2FHead


class MinFootprintMixin:
    """Post-processes the attn mask produced by ``_forward_head``.

    Must precede a Mask2FormerHead(-subclass) in the MRO. Both mmdet's
    Mask2FormerHead.forward and SealM2FHead.forward obtain every cross-attn
    mask from ``self._forward_head`` -- overriding it here is the minimal
    injection point (no forward() copy).
    """

    def __init__(self, *args, min_footprint: int = 0,
                 minfoot_debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        mf = int(min_footprint)
        assert mf == 0 or (mf >= 1 and mf % 2 == 1), (
            f'min_footprint must be 0 (off) or an ODD window size k>=1, '
            f'got {min_footprint!r}')
        self.min_footprint = mf
        self.minfoot_debug = bool(minfoot_debug)
        self._minfoot_log = []  # debug records, only appended when minfoot_debug

    # ------------------------------------------------------------------ #
    def _apply_min_footprint(self, mask_pred: torch.Tensor,
                             attn_mask: torch.Tensor,
                             attn_mask_target_size) -> torch.Tensor:
        """Open a k x k window for every EMPTY (all-blocked) attn row.

        Args:
            mask_pred: (B, Q, h, w) mask logits at mask_features (1/4) res.
            attn_mask: (B*num_heads, Q, H*W) bool, True = blocked (detached).
            attn_mask_target_size: (H, W) of the current memory level.
        Returns:
            (B*num_heads, Q, H*W) bool. Non-empty rows bitwise unchanged;
            empty rows become: blocked everywhere EXCEPT the k x k window.
        """
        k = self.min_footprint
        H, W = attn_mask_target_size
        B, Q, h, w = mask_pred.shape
        nh = attn_mask.shape[0] // B
        dev = attn_mask.device
        mp = mask_pred.detach().float()

        # --- centroid of the current predicted mask at 1/4 res ------------ #
        binm = (mp > 0).to(mp.dtype)                       # sigmoid(mp) > 0.5
        area = binm.sum(dim=(-2, -1))                      # (B, Q)
        ys = torch.arange(h, device=dev, dtype=mp.dtype).view(1, 1, h, 1)
        xs = torch.arange(w, device=dev, dtype=mp.dtype).view(1, 1, 1, w)
        cy = (binm * ys).sum(dim=(-2, -1)) / area.clamp_min(1.0)   # (B, Q)
        cx = (binm * xs).sum(dim=(-2, -1)) / area.clamp_min(1.0)
        # 1/4-res mask also empty -> fall back to the logit argmax (best guess)
        amax = mp.flatten(2).argmax(-1)                    # (B, Q)
        ay = torch.div(amax, w, rounding_mode='floor').to(mp.dtype)
        ax = (amax % w).to(mp.dtype)
        cy = torch.where(area > 0, cy, ay)
        cx = torch.where(area > 0, cx, ax)

        # --- map 1/4-res coords -> level grid (align_corners=False conv.) - #
        row = ((cy + 0.5) * (float(H) / float(h)) - 0.5).round().clamp_(0, H - 1).long()
        col = ((cx + 0.5) * (float(W) / float(w)) - 0.5).round().clamp_(0, W - 1).long()

        # --- k x k window, border-clamped (always >= 1 open position) ----- #
        r = (k - 1) // 2
        rows = torch.arange(H, device=dev).view(1, 1, H, 1)
        cols = torch.arange(W, device=dev).view(1, 1, 1, W)
        win = (((rows - row.view(B, Q, 1, 1)).abs() <= r)
               & ((cols - col.view(B, Q, 1, 1)).abs() <= r))     # (B,Q,H,W)
        win = win.view(B, 1, Q, H * W)

        # --- open the window ONLY on empty (all-blocked) rows -------------- #
        am = attn_mask.view(B, nh, Q, H * W)
        empty = am.all(-1, keepdim=True)                   # (B, nh, Q, 1)
        am = am & ~(empty & win)

        if self.minfoot_debug:
            e0 = empty.view(B, nh, Q)[:, 0]                # heads identical in stock path
            idx = e0.nonzero(as_tuple=False)
            self._minfoot_log.append(dict(
                level_hw=(int(H), int(W)), k=k,
                n_empty=int(e0.sum()),
                coords=[(int(b), int(q), int(row[b, q]), int(col[b, q]))
                        for b, q in idx[:16].tolist()],
            ))
        return am.view(B * nh, Q, H * W)

    # ------------------------------------------------------------------ #
    def _forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        cls_pred, mask_pred, attn_mask = super()._forward_head(
            decoder_out, mask_feature, attn_mask_target_size)
        if self.min_footprint > 0:
            attn_mask = self._apply_min_footprint(
                mask_pred, attn_mask, attn_mask_target_size)
        return cls_pred, mask_pred, attn_mask


@MODELS.register_module()
class MinFootprintM2FHead(MinFootprintMixin, Mask2FormerHead):
    """Stock mmdet Mask2FormerHead + min-footprint floor (r6b carrier:
    cp_m2f_cocoinit_pseg protocol). min_footprint=0 == stock, bitwise."""


@MODELS.register_module()
class MinFootprintSealM2FHead(MinFootprintMixin, SealM2FHead):
    """SealM2FHead (a0r refine carrier) + min-footprint floor, for later
    composition with the refine_scale lever. min_footprint=0 == SealM2FHead,
    bitwise."""
