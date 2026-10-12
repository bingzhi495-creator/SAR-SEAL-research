# -*- coding: utf-8 -*-
"""OAEM -- Orientation-Aware Embedding (reimplemented from O2Former, arXiv:2506.11913;
no public code). SAR ships are elongated + arbitrarily oriented; OAEM enriches the
multi-scale backbone features with (a) oriented aggregation along the hull axis and
(b) a polar position encoding, so the downstream MSDeformAttn pixel-decoder + queries
can tighten masks (AP75) and disambiguate hull-vs-pier near shore (inshore).

Applied as a GATED RESIDUAL (gamma init 0) on each backbone level -> at step 0 the
output is byte-identical to the carrier, so it never breaks the COCO-init transfer or
NaNs at start; the orientation refinement is earned during training. Pure torch.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class OAEM(nn.Module):
    """Orientation-aware embedding on one feature level (B,C,H,W) -> (B,C,H,W)."""

    def __init__(self, c: int, k: int = 7):
        super().__init__()
        # oriented depthwise aggregation: horizontal (1xk), vertical (kx1), local (3x3)
        self.dw_h = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c)
        self.dw_v = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c)
        self.dw_s = nn.Conv2d(c, c, 3, padding=1, groups=c)
        # polar position encoding [cos(angle), sin(angle), radius] -> C
        self.pos = nn.Conv2d(3, c, 1)
        self.fuse = nn.Conv2d(c, c, 1)
        self.gn = nn.GroupNorm(min(8, c), c)
        self.gamma = nn.Parameter(torch.zeros(1))   # LayerScale init 0 -> identity at step 0

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f.shape
        f32 = f.float()
        ori = self.dw_h(f32) + self.dw_v(f32) + self.dw_s(f32)        # oriented aggregation
        ys = torch.linspace(-1.0, 1.0, h, device=f.device, dtype=f32.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=f.device, dtype=f32.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        ang = torch.atan2(ys, xs)
        rad = torch.sqrt(xs * xs + ys * ys + 1e-6)
        pol = torch.cat([torch.cos(ang), torch.sin(ang), rad], dim=1)  # (B,3,H,W)
        out = self.fuse(torch.relu(self.gn(ori + self.pos(pol))))
        return (f32 + self.gamma * out).to(f.dtype)                    # gated residual
