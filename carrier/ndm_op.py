# -*- coding: utf-8 -*-
"""NDM -- frequency-domain denoising module (reimplemented in the spirit of DenoDet V2,
AAAI 2026, github.com/GrokCV/GrokSAR). SAR speckle is multiplicative high-frequency noise;
a learnable soft-threshold on the FFT amplitude (phase preserved) suppresses speckle while
keeping the high-freq stern/bow edges that AP75 + small-ship recall depend on.

Gated residual (gamma init 0) -> step-0 identity (safe for COCO-init, no NaN). FFT kept in
fp32 (AMP-unsafe). Outputs are nan-guarded. Pure torch.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NDM(nn.Module):
    """Frequency soft-threshold denoise on one level (B,C,H,W) -> (B,C,H,W)."""

    def __init__(self, c: int):
        super().__init__()
        self.thr = nn.Parameter(torch.zeros(1, c, 1, 1))   # per-channel soft-threshold (softplus -> >=0)
        self.proj = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))           # LayerScale init 0 -> identity at step 0

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f32 = f.float()
        spec = torch.fft.rfft2(f32, norm='ortho')          # (B,C,H,W//2+1) complex
        amp = spec.abs()
        pha = torch.angle(spec)
        thr = F.softplus(self.thr)                          # >=0, broadcast over (H,Wf)
        amp_d = torch.relu(amp - thr)                       # soft-threshold: drop low-amp (speckle)
        spec_d = torch.polar(amp_d, pha)                    # rebuild from denoised amp + kept phase
        out = torch.fft.irfft2(spec_d, s=f32.shape[-2:], norm='ortho')
        out = self.proj(torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))
        return (f32 + self.gamma * out).to(f.dtype)
