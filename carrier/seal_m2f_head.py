# -*- coding: utf-8 -*-
"""SealM2FHead: Mask2FormerHead with a final-layer mask resolution refinement.

Oracle diagnostic (runs/a0_pseg_seed0/mask_quality_oracle): ~40% of the inshore
AP75 gap is mask-resolution-bound -- a 19px ship at the 1/4 mask_features stride
becomes a ~5px mask, blurry once upsampled. This head upsamples mask_features
1/4 -> 1/2 with a small learnable module and recomputes ONLY the final decoder
layer's mask at 1/2 (the 9 aux layers stay 1/4 to avoid the ~23GB OOM of raising
all layers). The final mask -- used for the main loss AND inference -- is sharper.

This is the ONE GPU lever with a clear AP mechanism (per the 2026-06-16 CPU
gates: evidence discrimination is weak everywhere; mask quality is the data-
supported AP axis). Evidence-free, geometry-only -- aligned with the honest
findings. An A0R-refine vs A0 ablation isolates the resolution effect.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.registry import MODELS


@MODELS.register_module()
class SealM2FHead(Mask2FormerHead):
    def __init__(self, *args, refine_scale: int = 2, learn_refine: bool = True,
                 infer_blend: float = 1.0, infer_size_gate=None,
                 infer_soft_amax: float = 1.0, infer_soft_e0=None, infer_soft_w: float = 4.0,
                 boundary_weight: float = 0.0, boundary_ksize: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        # Boundary loss (UNet++ teacher MaskBoundaryLoss principle, adapted to the
        # rank-1 per-query masks): a differentiable boundary-Dice between the local
        # range (max-min) of the matched soft pred mask and the GT mask -> sharper
        # high-IoU boundaries. boundary_weight=0 -> identical to the stock head.
        self.boundary_weight = float(boundary_weight)
        self.boundary_ksize = int(boundary_ksize)
        c = self.mask_embed[-1].out_features  # mask feature channels (=feat_channels)
        self.refine_scale = refine_scale
        self.learn_refine = learn_refine  # False = bilinear-only ablation (P0-1: is the
        # gain from resolution itself or from the learned sharpening? isolate it)
        # --- ADAPTIVE refinement params (inference-only; training is unchanged) ---
        # Different datasets differ in resolution / ship size / density (see
        # diagnostics/dataset_profile.py): the optimal amount of refinement is
        # dataset-dependent. These let the SAME trained checkpoint be re-evaluated
        # at different operating points WITHOUT retraining, so each dataset can
        # pick its optimum. Defaults (blend=1, gate=off) == full refinement == the
        # original A0R behaviour.
        #   infer_blend a   : final mask = a*refined + (1-a)*coarse  (a in [0,1])
        #   infer_size_gate : if set (px^2 at refined-mask resolution), instances
        #                     with refined-mask area > gate use the COARSE mask
        #                     (skip refining large instances). None = gate off.
        self.infer_blend = float(infer_blend)
        self.infer_size_gate = infer_size_gate
        # --- AMR (v2): per-instance SOFT blend (inference-only, zero new params).
        # Replaces the crude global blend + hard gate with a smooth per-instance
        # weight w_q = amax * sigmoid((e0 - e_q)/w), where e_q = sqrt(soft coarse
        # area) is the instance extent in 1/4-stride px. out_q = w_q*refined +
        # (1-w_q)*coarse. Recovers every v1 config as a special case (w->0 hard
        # gate; w->inf global blend=amax; amax=1,e0=inf full refine). Strictly more
        # expressive; SAME checkpoint (capacity held identically constant). Enabled
        # when infer_soft_e0 is not None.
        self.infer_soft_amax = float(infer_soft_amax)
        self.infer_soft_e0 = infer_soft_e0
        self.infer_soft_w = float(infer_soft_w)
        if learn_refine:
            self.mask_refine = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1),
            )

    def _refine_features(self, mask_features):
        hi = F.interpolate(mask_features, scale_factor=self.refine_scale,
                           mode="bilinear", align_corners=False)
        return hi + self.mask_refine(hi) if self.learn_refine else hi  # residual sharpen vs bilinear-only

    def _compute_refined_mask(self, query_feat, hi_features):
        """The final-layer rank-1 readout (mask_embed . E). Subclasses (DKR) override to add a
        gated per-instance dynamic-kernel branch in parallel with this rank-1 mask."""
        decoder_out = self.transformer_decoder.post_norm(query_feat)
        mask_embed = self.mask_embed(decoder_out)
        return torch.einsum("bqc,bchw->bqhw", mask_embed, hi_features)

    def _boundary_dice(self, pred_logits, gt):
        """Boundary-Dice on matched full masks. edge(x)=maxpool(x)-minpool(x)=local
        range (high at edges). pred_logits,gt: (N,h,w). Differentiable."""
        k, pad = self.boundary_ksize, self.boundary_ksize // 2
        p = pred_logits.sigmoid().unsqueeze(1)
        g = gt.unsqueeze(1).float()
        if g.shape[-2:] != p.shape[-2:]:                 # GT is full-res, pred is 1/4 -> align
            g = F.interpolate(g, size=p.shape[-2:], mode='nearest')

        def edge(x):
            return F.max_pool2d(x, k, 1, pad) + F.max_pool2d(-x, k, 1, pad)  # max-min
        bp, bg = edge(p), edge(g)
        num = 2.0 * (bp * bg).flatten(1).sum(1) + 1.0
        den = bp.flatten(1).sum(1) + bg.flatten(1).sum(1) + 1.0
        return (1.0 - num / den).mean()

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances, batch_img_metas):
        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        if self.boundary_weight > 0:
            n = cls_scores.size(0)
            (_, _, mtl, mwl, _) = self.get_targets(
                [cls_scores[i] for i in range(n)], [mask_preds[i] for i in range(n)],
                batch_gt_instances, batch_img_metas)
            mt = torch.cat(mtl, dim=0)
            mw = torch.stack(mwl, dim=0)
            mp = mask_preds[mw > 0]
            if mt.shape[0] > 0:
                loss_mask = loss_mask + self.boundary_weight * self._boundary_dice(mp, mt)
        return loss_cls, loss_mask, loss_dice

    def forward(self, x, batch_data_samples):
        # --- replicate Mask2FormerHead.forward, capturing mask_features + final query_feat ---
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        decoder_inputs, decoder_positional_encodings = [], []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            dpe = self.decoder_positional_encoding(mask).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(dpe)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list, mask_pred_list = [], []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat, key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx], query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask, query_key_padding_mask=None,
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:])
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        self._last_query_feat = query_feat  # stash final-layer query embedding (MaskIoU subclass hook; behavior-neutral)
        # --- refinement: recompute ONLY the final-layer mask at higher resolution ---
        coarse_final = mask_pred_list[-1]                            # (B, Q, h, w) 1/4 coarse final mask
        hi_features = self._refine_features(mask_features)           # (B, C, 2h, 2w)
        refined_mask = self._compute_refined_mask(query_feat, hi_features)  # (B,Q,2h,2w); hook for DKR 1/2 res

        # Training: always full refinement (loss on the sharper mask) -- UNCHANGED.
        # Inference: optionally blend/size-gate toward the coarse mask so the SAME
        # checkpoint can be re-evaluated at a dataset-tuned operating point without
        # retraining (adaptive mechanism; defaults == full refinement).
        soft = (not self.training) and (self.infer_soft_e0 is not None)
        v1blend = (not self.training) and (self.infer_blend < 1.0 or self.infer_size_gate is not None)
        if soft:
            # AMR per-instance soft blend (E0). e_q = sqrt(soft COARSE area) -> no
            # feedback from refinement; w_q = amax * sigmoid((e0 - e_q)/w).
            coarse_up = F.interpolate(coarse_final, size=refined_mask.shape[-2:],
                                      mode="bilinear", align_corners=False)
            a_q = coarse_final.sigmoid().flatten(2).sum(-1).clamp_min(1e-6)  # (B,Q) soft area, 1/4 px^2
            e_q = a_q.sqrt()                                                  # (B,Q) extent, 1/4 px
            w_q = self.infer_soft_amax * torch.sigmoid((float(self.infer_soft_e0) - e_q) / self.infer_soft_w)
            w_q = w_q.view(*w_q.shape, 1, 1)                                  # (B,Q,1,1)
            mask_pred_list[-1] = w_q * refined_mask + (1.0 - w_q) * coarse_up
        elif v1blend:
            coarse_up = F.interpolate(coarse_final, size=refined_mask.shape[-2:],
                                      mode="bilinear", align_corners=False)
            out = refined_mask
            if self.infer_blend < 1.0:
                a = self.infer_blend
                out = a * refined_mask + (1.0 - a) * coarse_up
            if self.infer_size_gate is not None:
                area = (refined_mask.sigmoid() > 0.5).flatten(2).sum(-1).float()  # (B,Q)
                use_refined = (area <= float(self.infer_size_gate)).view(*area.shape, 1, 1)
                out = torch.where(use_refined, out, coarse_up)
            mask_pred_list[-1] = out
        else:
            mask_pred_list[-1] = refined_mask  # final mask now 1/2 res (loss + inference)

        return cls_pred_list, mask_pred_list
