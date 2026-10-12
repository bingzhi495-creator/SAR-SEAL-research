# -*- coding: utf-8 -*-
"""GeoStack -- plain module STACKING (the NO-GO fallback, per user directive).

The unified GeoField operator was a converged equal-param NO-GO (R5s full 0.641 AP75 ==
baseline R0 0.637, gamma engaged but unhelpful). User: "if NO-GO, even try plain module
stacking and see the effect." This is the diagnostic: do the SAME principles help when
stacked as INDEPENDENT blocks (full capacity each), not squeezed into one shared code?

Teacher-style (D:/SAR/UNet++/model_full.py: CoordConv + CBAM + DSConv). Here we stack three
independent blocks that sequentially enrich the refined mask feature E:
  1. POSITION block  : CoordConv (concat normalized coords, 3x3 conv, GN, SiLU).
  2. SALIENCE block  : CBAM (channel attention + spatial attention), then a 3x3 conv.
  3. SHAPE block     : per-pixel oriented free-basis depthwise conv (the proven GeoField
                       kernel, but as a standalone block, not a shared-code read-out).
E' = E + gamma * stack(E). gamma is a residual LayerScale init 0 (identity start; lr_mult=50
in the config so it engages). NOT equal-param vs GeoField (stacking adds params) -> compare
to baseline R0 AND a param-matched plain-conv control to attribute any gain to capacity vs
the stacked mechanisms.

Pure torch (no mmdet) for unit smoke + reuse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoStack(nn.Module):
    def __init__(self, c: int, m: int = 8, k: int = 5, ratio: int = 16):
        super().__init__()
        assert k % 2 == 1
        self.c, self.m, self.k, self.pad = int(c), int(m), int(k), k // 2
        # 1) position block (CoordConv)
        self.pos_conv = nn.Conv2d(c + 2, c, 3, padding=1)
        self.pos_gn = nn.GroupNorm(min(8, c), c)
        # 2) salience block (CBAM: channel + spatial)
        self.ca_fc1 = nn.Conv2d(c, max(1, c // ratio), 1, bias=False)
        self.ca_fc2 = nn.Conv2d(max(1, c // ratio), c, 1, bias=False)
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.cbam_conv = nn.Conv2d(c, c, 3, padding=1)
        self.cbam_gn = nn.GroupNorm(min(8, c), c)
        # 3) shape block (per-pixel oriented free-basis depthwise conv)
        self.wk = nn.Conv2d(c, m, 1)
        self.basis = nn.Parameter(0.01 * torch.randn(m, k * k))
        # residual gate (init 0 -> identity start; lr_mult boosts it to engage)
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _coords(b, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xs, ys], dim=1)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        b, c, h, w = e.shape
        e32 = e.float()
        coords = self._coords(b, h, w, e.device, e32.dtype)
        # 1) position
        x = F.silu(self.pos_gn(self.pos_conv(torch.cat([e32, coords], dim=1))))
        # 2) salience (CBAM): channel then spatial
        ca = torch.sigmoid(
            self.ca_fc2(F.relu(self.ca_fc1(F.adaptive_avg_pool2d(x, 1))))
            + self.ca_fc2(F.relu(self.ca_fc1(F.adaptive_max_pool2d(x, 1)))))
        x = x * ca
        sa = torch.sigmoid(self.sa_conv(
            torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)))
        x = x * sa
        x = F.silu(self.cbam_gn(self.cbam_conv(x)))
        # 3) shape: oriented free-basis per-pixel depthwise conv (fp32 softmax)
        alpha = self.wk(x)
        klogit = torch.einsum('bmhw,mn->bnhw', alpha.float(), self.basis.float())
        kp = torch.softmax(klogit, dim=1)
        xpad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        x_mix = torch.zeros_like(x)
        idx = 0
        for di in range(self.k):
            for dj in range(self.k):
                x_mix = x_mix + kp[:, idx:idx + 1, :, :] * xpad[:, :, di:di + h, dj:dj + w]
                idx += 1
        # residual enrichment of E
        e_out = e32 + self.gamma * x_mix
        return e_out.to(e.dtype)
