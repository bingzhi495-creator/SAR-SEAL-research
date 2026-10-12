# -*- coding: utf-8 -*-
"""DKR -- per-instance Dynamic-Kernel mask Readout, with SAR-aware conditioning.

Mask2Former's mask is a RANK-1 readout (one query vector . shared feature) -> one smooth
hyperplane per instance, zero per-instance spatial DOF -> the high-IoU boundary ceiling.
DKR adds, in parallel (gated, gamma init 0 -> step-0 == carrier), a per-instance dynamic
conv stack whose weights are generated from the query.

KERNEL-SHAPE (the F1 novelty). CondInst/SOLOv2/DiffusionInst/DiffSAR all use 1x1-only dynamic
convs: rank>1 in channels but RANK-0 in space (a pixel's logit cannot depend on neighbours),
so the boundary band -- where ALL high-IoU error lives -- is unreachable by the kernel.
kernel_mode:
  '1x1'      : 3x 1x1 (the CondInst/DiffSAR readout; baseline).
  'dense3x3' : first layer dense 3x3 (spatial DOF, but 9*Ci*Co generated params -> overfits 928 imgs); control.
  'ds3x3'    : first layer DEPTHWISE-SEPARABLE dynamic 3x3 (dyn depthwise 3x3 [9*Ci] + dyn pointwise 1x1
               [Ci*Co]) -> per-instance 3x3 boundary-carving stencil. At the actual SAR widths (cs=10,
               ch=8) this is 1/3 of the dense-3x3 total-readout cost (269 vs 809 params), or ~1/4.2 on
               the dynamic 3x3 weight tensor alone (170 vs 720). THE proposal.

SAR conditioning (relative coords / scatterer-amplitude centre / hull-orientation rotation) feeds
extra substrate channels DiffSAR's readout lacks. Pure torch; nan-guarded.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_specs(mode: str, cs: int, ch: int):
    """(kind, in_ch, out_ch, k); kind: 'pw'=1x1, 'dw'=depthwise kxk, 'full'=dense kxk."""
    if mode == '1x1':
        return [('pw', cs, ch, 1), ('pw', ch, ch, 1), ('pw', ch, 1, 1)]
    if mode == 'dense3x3':
        return [('full', cs, ch, 3), ('pw', ch, ch, 1), ('pw', ch, 1, 1)]
    if mode == 'ds3x3':
        return [('dw', cs, cs, 3), ('pw', cs, ch, 1), ('pw', ch, ch, 1), ('pw', ch, 1, 1)]
    raise ValueError(mode)


def _nparams(specs):
    n = 0
    for kind, ci, co, k in specs:
        if kind == 'dw':
            n += ci * k * k + ci
        else:  # pw or full
            n += ci * co * (k * k) + co
    return n


def _apply_dynamic(x, params, specs, N):
    """x: (1, N*cs, H, W) ; params: (N, P) ; returns (1, N, H, W)."""
    cur = x
    idx = 0
    for li, (kind, ci, co, k) in enumerate(specs):
        if kind == 'dw':
            wsz = ci * k * k
            w = params[:, idx:idx + wsz].reshape(N * ci, 1, k, k); idx += wsz
            b = params[:, idx:idx + ci].reshape(N * ci); idx += ci
            cur = F.conv2d(cur, w, bias=b, stride=1, padding=k // 2, groups=N * ci)
        else:
            wsz = ci * co * k * k
            w = params[:, idx:idx + wsz].reshape(N * co, ci, k, k); idx += wsz
            b = params[:, idx:idx + co].reshape(N * co); idx += co
            cur = F.conv2d(cur, w, bias=b, stride=1, padding=k // 2, groups=N)
        if li < len(specs) - 1:
            cur = F.relu(cur)
    return cur


class DKR(nn.Module):
    def __init__(self, c: int, dkr_channels: int = 8, use_relcoord: bool = False,
                 use_scatter: bool = False, use_orient: bool = False, kernel_mode: str = '1x1'):
        super().__init__()
        self.ch = dkr_channels
        self.use_relcoord = use_relcoord
        self.use_scatter = use_scatter
        self.use_orient = use_orient
        self.kernel_mode = kernel_mode
        self.cs = dkr_channels + (2 if use_relcoord else 0)        # substrate input channels
        self.specs = _layer_specs(kernel_mode, self.cs, dkr_channels)
        self.mask_proj = nn.Conv2d(c, dkr_channels, 1)
        self.controller = nn.Linear(c, _nparams(self.specs))
        self.gamma = nn.Parameter(torch.zeros(1))                  # LayerScale init 0 -> identity at step 0

    def _relcoord(self, coarse_mask, amp=None):
        B, Q, H, W = coarse_mask.shape
        p = coarse_mask.sigmoid().clamp_min(1e-6)
        w = p * amp.clamp_min(0.0) if (self.use_scatter and amp is not None) else p
        ys = torch.linspace(-1.0, 1.0, H, device=p.device, dtype=p.dtype).view(1, 1, H, 1)
        xs = torch.linspace(-1.0, 1.0, W, device=p.device, dtype=p.dtype).view(1, 1, 1, W)
        den = w.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        cx = (w * xs).sum(dim=(-1, -2), keepdim=True) / den
        cy = (w * ys).sum(dim=(-1, -2), keepdim=True) / den
        relx = (xs - cx).expand(B, Q, H, W)
        rely = (ys - cy).expand(B, Q, H, W)
        if self.use_orient:
            mxx = (w * relx * relx).sum(dim=(-1, -2), keepdim=True) / den
            myy = (w * rely * rely).sum(dim=(-1, -2), keepdim=True) / den
            mxy = (w * relx * rely).sum(dim=(-1, -2), keepdim=True) / den
            th = 0.5 * torch.atan2(2.0 * mxy, (mxx - myy) + 1e-6)
            ct, st = torch.cos(th), torch.sin(th)
            relx, rely = relx * ct + rely * st, -relx * st + rely * ct
        return torch.nan_to_num(torch.stack([relx, rely], dim=2), nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, decoder_out, hi_features, coarse_mask=None, amp=None):
        B, Q, C = decoder_out.shape
        E = self.mask_proj(hi_features)                            # (B,ch,H,W)
        H, W = E.shape[-2:]
        N = B * Q
        feat = E.unsqueeze(1).expand(B, Q, self.ch, H, W)         # (B,Q,ch,H,W)
        if self.use_relcoord:
            rel = self._relcoord(coarse_mask, amp).to(feat.dtype)
            feat = torch.cat([feat, rel], dim=2)                  # (B,Q,cs,H,W)
        x = feat.reshape(1, N * self.cs, H, W)
        params = self.controller(decoder_out).reshape(N, -1)
        m = _apply_dynamic(x, params, self.specs, N).reshape(B, Q, H, W)
        return self.gamma * m                                      # gated; added to rank-1 by caller


@torch.no_grad()
def relcoord_field(coarse_logit):
    """B1 helper for the DiffSAR port: per-instance relcoord (centroid origin) from a coarse
    per-instance mask logit (N,1,H,W) -> (N,2,H,W) [relx,rely]. Mirrors DKR._relcoord."""
    p = coarse_logit.sigmoid().clamp_min(1e-6)
    N, _, H, W = p.shape
    ys = torch.linspace(-1., 1., H, device=p.device, dtype=p.dtype).view(1, 1, H, 1)
    xs = torch.linspace(-1., 1., W, device=p.device, dtype=p.dtype).view(1, 1, 1, W)
    den = p.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    cx = (p * xs).sum(dim=(-1, -2), keepdim=True) / den
    cy = (p * ys).sum(dim=(-1, -2), keepdim=True) / den
    relx = (xs - cx).expand(N, 1, H, W)
    rely = (ys - cy).expand(N, 1, H, W)
    return torch.nan_to_num(torch.cat([relx, rely], dim=1), nan=0., posinf=0., neginf=0.)
