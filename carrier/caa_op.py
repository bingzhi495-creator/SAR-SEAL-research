# -*- coding: utf-8 -*-
"""CAA -- Context Anchor Attention (reimplemented from PKINet, CVPR 2024, arXiv:2403.06258).
Inshore false positives need LONG-RANGE context to tell ship-vs-pier/coast; CAA builds a
strip-shaped (1xK then Kx1 depthwise) long-range context and gates the feature with it.
Pure depthwise -> no custom CUDA op (unlike FreqFusion). Targets the #1 inshore gap via a
different mechanism than OAEM (orientation): CAA = large-context anchoring.

Gated residual (gamma init 0) -> step-0 identity (safe for COCO-init, no NaN). Pure torch.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CAA(nn.Module):
    """Context anchor attention on one level (B,C,H,W) -> (B,C,H,W)."""

    def __init__(self, c: int, k: int = 11):
        super().__init__()
        self.avg = nn.AvgPool2d(7, 1, 3)                 # local context pool
        self.conv1 = nn.Conv2d(c, c, 1)
        self.dw_h = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c)   # horizontal strip
        self.dw_v = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c)   # vertical strip
        self.conv2 = nn.Conv2d(c, c, 1)
        self.act = nn.SiLU()
        self.gamma = nn.Parameter(torch.zeros(1))        # LayerScale init 0 -> identity at step 0

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f32 = f.float()
        ctx = self.conv2(self.dw_v(self.dw_h(self.act(self.conv1(self.avg(f32))))))
        attn = torch.sigmoid(ctx)                        # long-range context attention map
        out = f32 * attn                                 # anchor the feature on context
        return (f32 + self.gamma * (out - f32)).to(f.dtype)   # gated residual, init identity
