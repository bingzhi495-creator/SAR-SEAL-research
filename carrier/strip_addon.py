# -*- coding: utf-8 -*-
"""StripNet-lite -- an anisotropic strip-convolution add-on bolted onto the ResNet-50
backbone of Mask2Former, applied to each backbone feature map BEFORE the pixel decoder.

Motivation (SAR ships): ships are ELONGATED (high aspect ratio, oriented along the
azimuth / range axes). Square 3x3 kernels mix an isotropic neighbourhood and dilute
that shape prior. Strip / anisotropic convolutions (a 1xk plus a kx1 depthwise pair,
2025-era backbone idea) encode a long-and-thin receptive field that matches the hull
geometry. We add this as a ZERO-GATED RESIDUAL add-on so the strong COCO-pretrained
ResNet-50 (+ M2F head) is preserved 1:1 and, at init, the add-on contributes EXACTLY
zero -- GATE-0 byte-identical to the stock c0 base.

Exact design (per selected backbone output feature map C2..C5, channels 256/512/1024/
2048 for R50):
    out = x + gamma * StripBlock(x)
    StripBlock(x) = reduce(C -> C/r, 1x1)
                    -> [ DWConv 1xk  +  DWConv kx1 ]  (summed, depthwise on C/r)
                    -> BN -> ReLU
                    -> expand(C/r -> C, 1x1)
with k = ``strip_k`` (default 7, configurable), r = 4. ``gamma`` is a per-stage LEARNABLE
scalar initialised to ``strip_gate_init`` (default 0.0). Belt-and-suspenders: the FINAL
``expand`` conv weight AND bias are ALSO zero-initialised. So at init BOTH gamma == 0 AND
expand == 0 -> StripBlock(x) == 0 -> out == x BIT-FOR-BIT. During training gamma and the
expand weights grow if the anisotropic prior helps.

SGA-v2 knobs (all NEW, all defaulting to the ORIGINAL behaviour above -- the running s0
job's config semantics are untouched):
  * ``gamma_mode``  : 'scalar' (default, per-stage scalar gamma, as before) | 'channel'
                      (per-CHANNEL learnable gamma vector of shape (C,) per stage --
                      CaiT LayerScale evidence: per-channel > scalar). In BOTH modes every
                      gamma entry is initialised to ``strip_gate_init``, so gate_init=0.0
                      still gives out == x exactly (plus the zero-init expand conv).
  * ``strip_order`` : 'parallel' (default: DW 1xk + DW kx1 summed, as before) |
                      'sequential' (Strip R-CNN: DW 1xk THEN DW kx1 chained; sequential >
                      parallel by +0.21 in their ablation). Same parameters either way --
                      only the dataflow between the two DW convs changes.
  * ``strip_gate_init`` already existed; v2 configs set 0.1 (CaiT warm-start epsilon for
    shallow stacks) -- deliberately NOT byte-identical at init, but GATE-0 remains
    provable by setting it back to 0.0 in any mode.

Design contract:
  * ``strip_enable=True``  (default): builds one residual strip block per backbone output.
  * ``strip_enable=False``          : pure passthrough -- the backbone forward is stock.
  * ``StripResNet`` SUBCLASSES mmdet's ``ResNet``, so EVERY stock conv/bn state_dict key
    is UNCHANGED (``conv1.weight``, ``layer1.0.conv1.weight``, ...). The COCO checkpoint
    therefore loads 1:1; the ONLY new keys are the ``strip_blocks.*`` params, which are
    absent from the checkpoint and tolerated by mmdet's strict=False load (a warning).

Pure torch; CPU-safe. Mirrors the registration / subclassing pattern of
``carrier/offset_adjust.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS


class StripBlock(nn.Module):
    """Anisotropic strip transform: reduce -> (DW 1xk + DW kx1) -> BN -> ReLU -> expand.

    ``strip_order='parallel'`` (default): the two DW branches are SUMMED (original).
    ``strip_order='sequential'``        : DW 1xk THEN DW kx1 chained (Strip R-CNN).
    Identical parameters/state_dict either way -- only the dataflow differs.

    The ``expand`` 1x1 conv (weight AND bias) is ZERO-initialised, so a freshly built
    block outputs EXACTLY 0 for any input -- one half of the GATE-0 guarantee.
    """

    def __init__(self, channels: int, strip_k: int = 7, reduction: int = 4,
                 strip_order: str = 'parallel'):
        super().__init__()
        assert channels % reduction == 0, \
            f'channels ({channels}) must be divisible by reduction ({reduction})'
        assert strip_order in ('parallel', 'sequential'), \
            f"strip_order must be 'parallel' or 'sequential', got {strip_order!r}"
        self.strip_order = strip_order
        mid = channels // reduction
        pad = strip_k // 2
        # 1x1 channel reduction.
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        # depthwise anisotropic branches: horizontal 1xk and vertical kx1.
        self.dw_h = nn.Conv2d(mid, mid, kernel_size=(1, strip_k),
                              padding=(0, pad), groups=mid, bias=False)
        self.dw_v = nn.Conv2d(mid, mid, kernel_size=(strip_k, 1),
                              padding=(pad, 0), groups=mid, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        # 1x1 expansion back to C -- ZERO init (belt-and-suspenders for GATE-0).
        self.expand = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        if self.strip_order == 'sequential':
            y = self.dw_v(self.dw_h(y))      # DW 1xk THEN kx1 (Strip R-CNN)
        else:
            y = self.dw_h(y) + self.dw_v(y)  # summed anisotropic branches (original)
        y = self.bn(y)
        y = self.act(y)
        y = self.expand(y)                   # == 0 at init
        return y


class StripResidual(nn.Module):
    """Zero-gated residual wrapper: ``out = x + gamma * StripBlock(x)``.

    ``gamma_mode='scalar'`` (default): ``gamma`` is a per-stage LEARNABLE scalar (original).
    ``gamma_mode='channel'``         : ``gamma`` is a per-CHANNEL learnable vector (C,)
                                       (CaiT LayerScale). EVERY entry is initialised to
                                       ``gate_init``.
    With ``gate_init == 0.0`` (either mode) AND the zero-init inner ``expand`` conv,
    ``out == x`` bit-for-bit.
    """

    def __init__(self, channels: int, strip_k: int = 7, gate_init: float = 0.0,
                 reduction: int = 4, gamma_mode: str = 'scalar',
                 strip_order: str = 'parallel'):
        super().__init__()
        assert gamma_mode in ('scalar', 'channel'), \
            f"gamma_mode must be 'scalar' or 'channel', got {gamma_mode!r}"
        self.gamma_mode = gamma_mode
        self.block = StripBlock(channels, strip_k=strip_k, reduction=reduction,
                                strip_order=strip_order)
        if gamma_mode == 'channel':
            # per-channel LayerScale vector, whole vector init to gate_init.
            self.gamma = nn.Parameter(torch.full((channels,), float(gate_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_mode == 'channel':
            return x + self.gamma.view(1, -1, 1, 1) * self.block(x)
        return x + self.gamma * self.block(x)


@MODELS.register_module()
class StripResNet(ResNet):
    """ResNet-50 whose selected output feature maps get a zero-gated anisotropic strip
    add-on applied BEFORE the pixel decoder.

    Composed by subclassing: ``__init__`` builds the STOCK ``ResNet`` verbatim (so every
    conv/bn state_dict key is identical and the COCO checkpoint loads 1:1), then appends
    one :class:`StripResidual` per backbone output. ``forward`` runs the inherited stock
    forward, then post-processes the output tuple with the strip blocks. At
    ``strip_gate_init=0.0`` (and the zero-init expand conv) this is GATE-0 byte-identical;
    with ``strip_enable=False`` it is a pure passthrough (no strip params are even built).
    """

    def __init__(self, *args, strip_k: int = 7, strip_gate_init: float = 0.0,
                 strip_enable: bool = True, strip_reduction: int = 4,
                 gamma_mode: str = 'scalar', strip_order: str = 'parallel', **kwargs):
        super().__init__(*args, **kwargs)
        self.strip_enable = bool(strip_enable)
        self.strip_k = int(strip_k)
        self.strip_gate_init = float(strip_gate_init)
        self.strip_reduction = int(strip_reduction)
        self.gamma_mode = str(gamma_mode)
        self.strip_order = str(strip_order)
        if not self.strip_enable:
            return
        # channels per stage for R50-style nets: base * 2**i * block.expansion.
        stage_channels = [self.base_channels * 2 ** i * self.block.expansion
                          for i in range(self.num_stages)]
        out_channels = [stage_channels[i] for i in self.out_indices]
        self.strip_blocks = nn.ModuleList([
            StripResidual(c, strip_k=self.strip_k, gate_init=self.strip_gate_init,
                          reduction=self.strip_reduction, gamma_mode=self.gamma_mode,
                          strip_order=self.strip_order)
            for c in out_channels])

    def forward(self, x: torch.Tensor):
        outs = super().forward(x)                    # stock ResNet tuple (unchanged)
        if not self.strip_enable:
            return outs                              # pure passthrough
        return tuple(self.strip_blocks[i](o) for i, o in enumerate(outs))
