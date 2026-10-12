# -*- coding: utf-8 -*-
"""Combined "kitchen-sink" model for HRSID: stack the diverse gain modules in ONE network, train once, ablate
later (user strategy 2026-07-01). Composition is nearly free because every module was built default-off with a
clean super() chain:
  a1  = resolution (config: num_transformer_feat_level=4 + pixel_decoder stride-4)
  a2  = Gaussian-Wasserstein MATCHING cost  (carrier/wass_match.py, via assigner + WassMatchM2FHead._get_targets_single)
  refine   = mask-feature upsample 1/4->1/2 (carrier/seal_m2f_head.py forward)
  boundary = boundary-Dice loss             (SealM2FHead._loss_by_feat_single, boundary_weight>0)
  maskiou  = mask-IoU rescoring             (carrier/maskiou_head.py, maskiou_weight>0 + MSIoUFusionHead infer_iou_lambda>0)
  CBAM     = feature attention on backbone outputs (NEW here; residual gamma-gate init 0 = identity)
SuperM2FHead's MRO = SuperM2FHead -> MSIoUM2FHead -> SealM2FHead -> WassMatchM2FHead -> Mask2FormerHead, so
_get_targets_single falls through to WassMatch (a2), forward=refine+maskiou, _loss_by_feat_single=boundary,
loss=maskiou.

CORRECTION (reviewer w13gekell, 2026-07-02): the claim "all flags default-off -> byte-identical to a1a2" is NOT
true at the class defaults -- SealM2FHead defaults refine_scale=2, learn_refine=True (refine ON, with random-init
mask_refine conv), unlike every sibling mechanism (a2 weight=0, lambda_repgt=0, maskiou_weight=0, CBAM gamma=0)
which ARE inert at default. The a1a2 baseline head is WassMatchM2FHead (NO refine at all). So byte-identity to
a1a2 holds ONLY when refine is EXPLICITLY disabled via refine_scale=1, learn_refine=False -- which the live
combo config (combo_m2f_hrsid.py) correctly does. Any leave-one-out ablation config instantiating SuperM2FHead
MUST set refine_scale=1, learn_refine=False explicitly or it silently reintroduces refine capacity+memory.
Ablation = flip a flag (boundary_weight/maskiou_weight/infer_iou_lambda/cbam_on to 0/False), keeping refine off.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS

from .maskiou_head import MSIoUM2FHead     # SealM2FHead(refine+boundary) + Mask-IoU
from .wass_match import WassMatchM2FHead   # a2 Gaussian-Wasserstein matching


@MODELS.register_module()
class SuperM2FHead(MSIoUM2FHead, WassMatchM2FHead):
    """a1(config) + a2(matching) + refine + boundary + maskiou in one head, purely via MRO. No new code:
    every override comes from the right parent (a2 _get_targets_single from WassMatch; refine forward +
    boundary loss from Seal; maskiou forward/loss/predict from MSIoU). All flags default-off."""
    pass


class _CBAM(nn.Module):
    """CBAM (Woo et al. ECCV2018 arXiv:1807.06521): channel then spatial attention, with a residual gamma-gate
    (gamma init 0 -> output == input at start, so pretrained features are unperturbed and GATE-0 is clean)."""

    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // r, 8)), nn.ReLU(inplace=True),
                                 nn.Linear(max(c // r, 8), c))
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))).view(B, C, 1, 1)
        xc = x * ca
        sa = torch.sigmoid(self.conv(torch.cat([xc.mean(1, keepdim=True), xc.amax(1, keepdim=True)], dim=1)))
        out = xc * sa
        return x + self.gamma * (out - x)   # gamma=0 at init -> identity


@MODELS.register_module()
class ResNetCBAM(ResNet):
    """ResNet with a CBAM block on each returned feature level. cbam_on=False -> plain ResNet (GATE-0 control
    is the SAME class with cbam_on=False, and gamma-gate makes even cbam_on=True start at identity)."""

    def __init__(self, *args, cbam_on: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam_on = cbam_on
        if cbam_on:
            exp = 4 if self.depth >= 50 else 1
            outc = [self.base_channels * (2 ** i) * exp for i in self.out_indices]
            self.cbams = nn.ModuleList([_CBAM(c) for c in outc])

    def forward(self, x):
        outs = super().forward(x)
        if self.cbam_on:
            outs = tuple(cb(o) for cb, o in zip(self.cbams, outs))
        return outs
