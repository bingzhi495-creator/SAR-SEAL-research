# -*- coding: utf-8 -*-
"""StemLatM2FHead -- zero-init stem lateral into the final mask readout (R6A).

Mechanism target (hires/bsamp/BPR triage, 2026-07): the stride-4 mask features
have already LOST the boundary signal -- pure Up2 of F is AP-neutral, while
methods that read pre-downsampling signal (BPR on the raw image, +10.9 inshore
AP75) win. This head therefore injects the backbone's stride-2 pre-maxpool stem
feature (64ch, published by carrier/stemlat_resnet.py:StemLatResNet via a
single-slot bus) into the FINAL decoder layer's mask:

    lat  = ZeroConv3x3( proj_1x1(stem) )                  # 256ch, 256^2@512^2
    F_hi = Up2(F) + lat                                    # conceptual
    final logits = q . F_hi                                # final layer only

``proj`` is a normally-initialised 1x1 conv; the 3x3 conv after it is
ControlNet-style ZERO-initialised (weight AND bias) -> at init lat == 0 exactly.

EXACT-PRESERVATION REASSOCIATION (why the code does not literally build F_hi):
the readout is linear in the features, so  q.(Up2(F) + lat) == Up2(q.F) + q.lat
mathematically. The first form loses bitwise equivalence twice (einsum/interp
float associativity ~1e-6 rel; and at predict time Up2 then Up2 != Up4, which
measurably moved 20-img AP by 0.002-0.004 -- first GATE-0 attempt, honest
FAIL). The reassociated form is the SAME function with an EXACT zero bypass:

  * training (this head's forward): final = Up2(q.F) + q.lat  at 1/2 res.
    At zero init q.lat == 0 exactly -> final == Up2(coarse) bitwise. Loss is
    the stock point-sampled path (12544 normalised coords, mmdet 3.3.0
    mask2former_head.py:249-335) -- no full-res loss buffer is materialised.
  * inference: final = Up4(q.F) + Up2(q.lat)  at batch-input res, so
    MaskFormerHead.predict's size-based interpolate becomes a bitwise identity.
    At zero init this is EXACTLY the stock output (GATE-0 (b) bitwise); the
    design card allows eval-time materialisation (stock predict materialises
    the same (Q,H,W) tensor anyway).

Aux masks and every attention mask stay on the stock stride-4 path: mmdet
Mask2FormerHead._forward_head generates each attn_mask by
F.interpolate(mask_pred, attn_mask_target_size) (resolution-agnostic), and in
SealM2FHead the refined final mask is produced AFTER the decoder loop, so it
never feeds an attn_mask at all. Defaults here: refine_scale=2,
learn_refine=False (a random-init mask_refine would break COCO equivalence).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from carrier.seal_m2f_head import SealM2FHead
from carrier.stemlat_resnet import pop_stem_feat


@MODELS.register_module()
class StemLatM2FHead(SealM2FHead):
    """SealM2FHead + zero-init stem lateral, reassociated for exact GATE-0.

    Args:
        stemlat_enable: build + apply the lateral (True). False -> pure
            hires-logits head (Up2 of the coarse logits; the known AP-neutral
            axis, kept as an ablation knob).
        stemlat_in_channels: channels of the backbone stem feature (64 for the
            classic R50 stem; stem_channels if deep_stem).
    """

    def __init__(self, *args, stemlat_enable: bool = True,
                 stemlat_in_channels: int = 64, **kwargs):
        # R6A defaults: 1/2-res final mask, NO random-init refine module
        # (learn_refine=True would inject non-zero randomness at init and
        # break COCO-weight equivalence).
        kwargs.setdefault('refine_scale', 2)
        kwargs.setdefault('learn_refine', False)
        super().__init__(*args, **kwargs)
        assert not self.learn_refine, \
            'StemLatM2FHead: learn_refine must stay False (preservation law).'
        self.stemlat_enable = bool(stemlat_enable)
        self.stemlat_in_channels = int(stemlat_in_channels)
        if self.stemlat_enable:
            c = self.mask_embed[-1].out_features   # mask feature channels
            # 1x1 projection stem_ch -> c: NORMAL (default kaiming) init.
            self.stem_proj = nn.Conv2d(self.stemlat_in_channels, c, 1)
            # ControlNet-style zero conv: weight AND bias zero -> lateral == 0
            # at init regardless of stem_proj's random weights.
            self.stem_zero = nn.Conv2d(c, c, 3, padding=1)
            nn.init.zeros_(self.stem_zero.weight)
            nn.init.zeros_(self.stem_zero.bias)

    def init_weights(self) -> None:
        # Mask2FormerHead.init_weights only touches decoder_input_projs /
        # pixel_decoder / transformer_decoder, but re-zero defensively in case
        # a future init_cfg sweep touches child convs.
        super().init_weights()
        if self.stemlat_enable:
            nn.init.zeros_(self.stem_zero.weight)
            nn.init.zeros_(self.stem_zero.bias)

    def _refine_features(self, mask_features):
        """Identity: the reassociated readout upsamples LOGITS, not features
        (saves the 256-ch Up2 buffer and keeps the zero bypass exact)."""
        return mask_features

    def _lateral_logits(self, mask_embed, batch_size):
        """q . ZeroConv(proj(stem)) at stem (1/2) resolution. Exactly 0 at
        init (zero conv), so callers may add it bitwise-safely."""
        stem = pop_stem_feat()
        if stem is None:
            raise RuntimeError(
                'StemLatM2FHead: no stem feature on the bus. The backbone must '
                'be carrier.stemlat_resnet.StemLatResNet with stemlat_push='
                'True (config model.backbone.type="StemLatResNet").')
        if stem.shape[0] != batch_size:
            raise RuntimeError(
                f'StemLatM2FHead: stem batch {stem.shape[0]} != head batch '
                f'{batch_size} -- backbone/head pairing bug.')
        lat = self.stem_zero(self.stem_proj(stem))
        return torch.einsum('bqc,bchw->bqhw', mask_embed, lat)

    def _compute_refined_mask(self, query_feat, feats):
        """Final-layer readout, reassociated (see module docstring).

        feats is the UNCHANGED stride-4 mask_features (our _refine_features is
        identity). coarse == mask_pred_list[-1] bitwise (same ops, same
        tensors as the loop's _forward_head readout).
        """
        decoder_out = self.transformer_decoder.post_norm(query_feat)
        mask_embed = self.mask_embed(decoder_out)
        coarse = torch.einsum('bqc,bchw->bqhw', mask_embed, feats)
        s = self.refine_scale
        if self.training:
            # train: 1/2-res logits for the point-sampled loss.
            out = F.interpolate(coarse, scale_factor=s, mode='bilinear',
                                align_corners=False)
            if self.stemlat_enable:
                lat = self._lateral_logits(mask_embed, feats.shape[0])
                if lat.shape[-2:] != out.shape[-2:]:  # odd external shapes only
                    lat = F.interpolate(lat, size=out.shape[-2:],
                                        mode='bilinear', align_corners=False)
                out = out + lat
            return out
        # eval: batch-input-res logits (mask_features are stride 4 by the M2F
        # pixel decoder, so x4 == batch_input under pad_size_divisor=32);
        # predict's size-interpolate then becomes a bitwise identity -> at
        # zero init the output == stock output bit-for-bit.
        out = F.interpolate(coarse, scale_factor=4, mode='bilinear',
                            align_corners=False)
        if self.stemlat_enable:
            lat = self._lateral_logits(mask_embed, feats.shape[0])
            lat = F.interpolate(lat, size=out.shape[-2:], mode='bilinear',
                                align_corners=False)
            out = out + lat
        return out
