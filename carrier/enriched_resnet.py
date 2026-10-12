# -*- coding: utf-8 -*-
"""EnrichedResNet: ResNet-50 with teacher-style backbone enrichment (CBAM + DSConv).

User insight: the teacher puts CBAM/BatchFormer/DSConv IN THE BACKBONE (where features are
extracted, before downsampling destroys elongated-ship structure) and CoordConv in the HEAD.
All our prior experiments wrongly enriched the mask-head feature E (downstream of the info
loss) -> NO-GO. This places the enrichment UPSTREAM, at the ResNet stage outputs.

We post-process each out_indices stage of mmdet's ResNet:
  - CBAM (channel + spatial attention) at all out stages (C2-C5), residual-gated.
  - DSConv (teacher's Dynamic Snake Conv, dsconv_teacher.py) at C3/C4 only (C2 is 6GB/forward
    at 128x128 -> too heavy; C5 tiny optional), residual-gated.
All enrichment is a residual with a LayerScale gamma init 0, so at step 0 EnrichedResNet ==
stock ResNet (the detector starts identical to baseline Mask2Former; enrichment must be earned).
The enrichment params get a higher lr via a config custom_key (backbone base lr 2.5e-6 is far
too low for fresh modules). BatchFormer (training-only, cross-sample) is deferred to a follow-up.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet

from carrier.dsconv_teacher import DSConv


class _ChannelAttention(nn.Module):
    def __init__(self, c, ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(1, c // ratio), 1, bias=False)
        self.fc2 = nn.Conv2d(max(1, c // ratio), c, 1, bias=False)

    def forward(self, x):
        avg = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        mx = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        return torch.sigmoid(avg + mx)


class _SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)
        m = x.max(dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([a, m], dim=1)))


class _CBAM(nn.Module):
    """CBAM that returns an attention-MODULATED residual delta (so it can be gated)."""
    def __init__(self, c, ratio=16):
        super().__init__()
        self.ca = _ChannelAttention(c, ratio)
        self.sa = _SpatialAttention()

    def forward(self, x):
        y = x * self.ca(x)
        y = y * self.sa(y)
        return y - x  # delta, so out = x + gamma*delta


@MODELS.register_module()
class EnrichedResNet(ResNet):
    def __init__(self, *args, cbam_stages=(0, 1, 2, 3), ds_stages=(1, 2),
                 ds_kernel=9, **kwargs):
        super().__init__(*args, **kwargs)
        # out_indices stage channels for ResNet-50 (expansion=4): C2..C5 = 256,512,1024,2048
        base = self.base_channels  # 64
        exp = self.block.expansion  # 4 for Bottleneck
        all_ch = [base * (2 ** i) * exp for i in range(4)]  # [256,512,1024,2048]
        self.cbam_stages = tuple(cbam_stages)
        self.ds_stages = tuple(ds_stages)
        # CBAM per cbam stage
        self.cbam = nn.ModuleDict({str(i): _CBAM(all_ch[i]) for i in self.cbam_stages})
        self.cbam_gamma = nn.ParameterDict(
            {str(i): nn.Parameter(torch.zeros(1)) for i in self.cbam_stages})
        # DSConv (x-axis, morph=0) per ds stage -- residual delta
        self.ds = nn.ModuleDict(
            {str(i): DSConv(all_ch[i], all_ch[i], kernel_size=ds_kernel,
                            extend_scope=1, morph=0, if_offset=True) for i in self.ds_stages})
        self.ds_gamma = nn.ParameterDict(
            {str(i): nn.Parameter(torch.zeros(1)) for i in self.ds_stages})

    def forward(self, x):
        outs = list(super().forward(x))
        for i in range(len(outs)):
            if i in self.cbam_stages:
                outs[i] = outs[i] + self.cbam_gamma[str(i)] * self.cbam[str(i)](outs[i])
            if i in self.ds_stages:
                # DSConv output is GN+ReLU'd (>=0); add as gated residual (gamma init 0 = identity)
                outs[i] = outs[i] + self.ds_gamma[str(i)] * self.ds[str(i)](outs[i])
        return tuple(outs)
