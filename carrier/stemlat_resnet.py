# -*- coding: utf-8 -*-
"""StemLatResNet -- stock mmdet ResNet that ALSO exposes the stem feature (R6A).

Why (mechanism, from the hires/bsamp/BPR triage): pure upsampling of the
stride-4 mask features is AP-neutral (hires 51.7), boundary point sampling is
+1.3, and BPR -- which reads the ORIGINAL image -- is +10.9 inshore AP75. So
the boundary information is already GONE at stride 4; any high-res mask head
must inject a signal from BEFORE the downsampling. The earliest such signal in
ResNet is the stem output: conv1(7x7,s2) + bn + relu, i.e. the stride-2, 64ch
tensor computed BEFORE maxpool (256^2 at the 512^2 PSeg protocol).

Contract (mirrors carrier/strip_addon.py's zero-risk subclass pattern):
  * SUBCLASSES mmdet ResNet: zero new parameters, every state_dict key
    unchanged -> the COCO checkpoint (and any cp_pseg checkpoint) loads 1:1.
  * ``forward`` replicates the stock dataflow LINE-FOR-LINE (mmdet 3.3.0
    resnet.py:631-646) and returns the SAME 4-level tuple, so every downstream
    consumer (pixel decoder in_channels=[256,512,1024,2048]) is untouched.
    The ONLY addition: the pre-maxpool stem tensor is stashed on
    ``self._stem_feat`` and pushed onto a module-level single-slot bus that
    the head pops (chosen over a 5-tuple return, which would break the
    pixel-decoder input contract and every config that consumes the tuple).
  * Bus discipline: backbone pushes once per forward, head pops once per
    forward (detector always runs backbone -> head sequentially on this
    single-GPU setup). The slot holds at most ONE tensor (overwritten each
    forward), so a non-popping consumer cannot accumulate memory. A popping
    head with a stock backbone fails LOUDLY (see stemlat_head.py).
"""
from __future__ import annotations

import torch

from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS

# Single-slot stem-feature bus (backbone writes, head pops). Module-level on
# purpose: the head has no reference to the backbone at build time, and a
# 5-tuple backbone return would break the 4-level pixel-decoder contract.
_STEM_BUS: dict = {}


def push_stem_feat(feat: torch.Tensor) -> None:
    _STEM_BUS['stem'] = feat


def pop_stem_feat():
    """Pop the current stem feature (None if no StemLatResNet ran)."""
    return _STEM_BUS.pop('stem', None)


@MODELS.register_module()
class StemLatResNet(ResNet):
    """ResNet that publishes its pre-maxpool stem feature for StemLatM2FHead.

    ``stemlat_push=False`` -> byte-identical stock behaviour (no bus writes).
    """

    def __init__(self, *args, stemlat_push: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.stemlat_push = bool(stemlat_push)
        self._stem_feat = None

    @property
    def stemlat_out_channels(self) -> int:
        # deep_stem: stem output has stem_channels; classic: conv1 -> 64.
        return self.stem_channels if self.deep_stem else 64

    def forward(self, x):
        """Stock mmdet 3.3.0 ResNet.forward with the stem tensor captured."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        if self.stemlat_push:
            self._stem_feat = x          # stride-2, pre-maxpool (256^2 @ 512^2)
            push_stem_feat(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
