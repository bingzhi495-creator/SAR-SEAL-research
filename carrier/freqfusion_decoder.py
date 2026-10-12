# -*- coding: utf-8 -*-
"""FreqFusionPixelDecoder -- MSDeformAttnPixelDecoder with FreqFusion (TPAMI 2024,
github.com/Linwei-Chen/FreqFusion, REAL ported code in carrier/freqfusion.py) replacing
the bilinear upsample-and-add in the multi-scale fusion. Adaptive low-pass (intra-object
consistency) + high-pass (boundary high-freq) + content-aware upsampling -> sharper masks
at high IoU (the diagnosed AP75/AP90 ceiling). carafe uses mmcv's compiled op (verified
available) with a pure-torch fallback in freqfusion.py.

forward is a verbatim copy of MSDeformAttnPixelDecoder.forward with ONLY the fusion line
changed (FreqFusion instead of cur_feat + F.interpolate(lr)). FreqFusion's content encoders
init near-zero (std 1e-3) -> near-uniform masks -> close to the original bilinear fusion at
step 0, so the COCO-init transfer is only mildly perturbed.
"""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.layers.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder

from carrier.freqfusion import FreqFusion


@MODELS.register_module()
class FreqFusionPixelDecoder(MSDeformAttnPixelDecoder):
    def __init__(self, *args, ff_feat_channels: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        n_fuse = self.num_input_levels - self.num_encoder_levels
        self.freqfusions = nn.ModuleList([
            FreqFusion(hr_channels=ff_feat_channels, lr_channels=ff_feat_channels,
                       feature_resample=False)
            for _ in range(max(0, n_fuse))])

    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, Tensor]:
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            feat_hw = torch._shape_as_tensor(feat)[2:].to(feat.device)
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            feat_wh = feat_hw.unsqueeze(0).flip(dims=[0, 1])
            factor = feat_wh * self.strides[level_idx]
            reference_points = reference_points / factor
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)
            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat_hw)
            reference_points_list.append(reference_points)

        padding_masks = torch.cat(padding_mask_list, dim=1)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=1)
        num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        memory = self.encoder(
            query=encoder_inputs,
            query_pos=level_positional_encodings,
            key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_radios)
        memory = memory.permute(0, 2, 1)

        outs = torch.split(memory, num_queries_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        ff_idx = 0
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)                       # hr
            _, hr, lr = self.freqfusions[ff_idx](hr_feat=cur_feat, lr_feat=outs[-1])
            if lr.shape[-2:] != hr.shape[-2:]:                        # guard non-exact-2x
                lr = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
            y = self.output_convs[i](hr + lr)                        # FreqFusion fuse, then output conv
            outs.append(y)
            ff_idx += 1
        multi_scale_features = outs[:self.num_outs]

        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features
