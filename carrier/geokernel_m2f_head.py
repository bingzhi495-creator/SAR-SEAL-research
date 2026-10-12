# -*- coding: utf-8 -*-
"""ShapePriorMask2FormerHead -- M3 "ShapePrior" geometry-conditioned mask head.

MECHANISM (design 2, "PARAMETRIC SHAPE-PRIOR FUSION" in
docs/GEOKERNEL_variants_full.md):

  Mask2Former's rank-1 readout
      mask_data = einsum('bqc,bchw->bqhw', mask_embed(decoder_out), E)
  has ZERO per-instance spatial receptive field: a query can only *select*
  pixels that the shared feature E already lit up. Where E is dark (the
  low-backscatter stern/bow) no query weight can recover the pixel.

  M3 KEEPS the stock rank-1 readout and ADDS, in logit space, a per-instance
  closed-form ORIENTED-ELLIPSE / oriented-Gaussian PRIOR mask, GATED:

      g = GeomMLP(decoder_out)            # (tx,ty,stheta,ctheta,la,lb,lg,lk)
      P = k * (1 - r2(x,y; cx,cy,theta,a,b))   # oriented-elliptical plateau
      mask_pred = mask_data + gate * P

  with gate = sigmoid(scalar head) initialised CLOSED (bias=-4 -> gate~0.018)
  so the model STARTS IDENTICAL to the rank-1 baseline and must EARN the prior.
  The prior is a geometry-addressed *write* into pixel space: it carries
  positive logit mass over the predicted hull even where E~0 -- exactly the
  diagnosed failure (dark stern/bow under-segmentation). This is structurally
  impossible for the rank-1 term, which lives in span(E).

gen_mode flag (one head class, four modes -> equal-param / equal-pipeline foils):
  'geo'    : full geometry-generated oriented-ellipse prior (the mechanism).
  'static' : ONE shared learned ellipse prior (nn.Parameter geometry, NOT from
             decoder_out); gate still per-query. The truly-static capacity foil.
  'blind'  : per-instance free prior-mask params (content-routed): GeomMLP routes
             a softmax blend over R learned global basis masks; geometry zeroed.
             The content-routing foil.
  'a0'     : gate forced OFF -> identical to stock Mask2Former (sanity).

The head subclasses the carrier SealM2FHead so the final-layer 1/4->1/2 mask
refinement is preserved; the gated prior is injected at BOTH the per-decoder-
layer readout (_forward_head) and the refined final mask (forward).

Net params: the stock mask_embed (~0.197M) is KEPT; the geometry/gate/basis
heads are tiny (a few k each, identical across geo/static/blind), so a0 ~ stock.

CPU-deterministic; eval path adds nothing stochastic.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from carrier.seal_m2f_head import SealM2FHead


@MODELS.register_module()
class ShapePriorMask2FormerHead(SealM2FHead):
    """Mask2Former head with a gated geometry-generated oriented-ellipse prior.

    Args (beyond SealM2FHead):
        gen_mode (str): one of {'geo','static','blind','a0'}.
        prior_hidden (int): GeomMLP hidden width (kept small; tiny vs mask_embed).
        prior_basis (int): number of learned global basis masks for 'blind'.
        gate_bias_init (float): init bias of the gate head (-4 -> gate~0.018).
        k0 (float): sharpness floor of the elliptical plateau.
        a_min (float): semi-axis floor (px on the H/4 grid) so the ellipse never
            degenerates.
        sar_loss_a_w (float): weight of SAR Loss A (intensity-weighted BCE add-on).
        sar_loss_b1_w (float): weight of SAR Loss B1 (boundary-band-weighted Dice).
        sar_lam_int (float): lambda_int for Loss A intensity up-weighting.
        sar_gamma_int (float): gamma for Loss A darkness focus.
        sar_band_k (int): morphological band half-width (px) for Loss B1.
    """

    def __init__(self, *args,
                 gen_mode: str = 'geo',
                 prior_hidden: int = 32,
                 prior_basis: int = 8,
                 gate_bias_init: float = -4.0,
                 k0: float = 4.0,
                 a_min: float = 1.0,
                 sar_loss_a_w: float = 0.0,
                 sar_loss_b1_w: float = 0.0,
                 sar_lam_int: float = 1.0,
                 sar_gamma_int: float = 1.0,
                 sar_band_k: int = 3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert gen_mode in ('geo', 'static', 'blind', 'a0'), gen_mode
        self.gen_mode = gen_mode
        self.k0 = float(k0)
        self.a_min = float(a_min)
        self.prior_basis = int(prior_basis)

        feat = self.mask_embed[-1].in_features  # 256 = feat_channels

        # ---- geometry / gate generator (present in ALL modes for equal params) ----
        # GeomMLP: decoder_out(256) -> hidden -> 8 raw geometry+gate+sharpness scalars
        # raw layout: (tx, ty, stheta, ctheta, la, lb, lg, lk)
        self.geom_mlp = nn.Sequential(
            nn.Linear(feat, prior_hidden), nn.ReLU(inplace=True),
            nn.Linear(prior_hidden, 8))
        # init final layer small so geometry starts near-centred / near-isotropic
        nn.init.zeros_(self.geom_mlp[-1].weight)
        nn.init.zeros_(self.geom_mlp[-1].bias)
        # gate is raw index 6 -> bias = gate_bias_init so gate sigma(-4) ~ 0.018
        with torch.no_grad():
            self.geom_mlp[-1].bias[6] = float(gate_bias_init)

        # ---- 'static': ONE shared learned ellipse (geometry as nn.Parameter) ----
        # layout (tx, ty, stheta, ctheta, la, lb, lk); gate stays per-query from GeomMLP[6]
        self.static_geom = nn.Parameter(torch.zeros(7))

        # ---- 'blind': R learned global basis masks (content-routed, geometry-blind)
        # GeomMLP's first `prior_basis` raw outputs are reused as routing logits.
        # Basis masks live on a fixed canonical grid and are resized to (H/4,W/4).
        self._blind_grid = 64
        self.blind_basis = nn.Parameter(
            0.01 * torch.randn(prior_basis, self._blind_grid, self._blind_grid))

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)

        # ---- SAR loss add-on config (default OFF) ----
        self.sar_loss_a_w = float(sar_loss_a_w)
        self.sar_loss_b1_w = float(sar_loss_b1_w)
        self.sar_lam_int = float(sar_lam_int)
        self.sar_gamma_int = float(sar_gamma_int)
        self.sar_band_k = int(sar_band_k)
        if self.sar_loss_b1_w > 0:
            # reuse the stock DiceLoss machinery, but feed pre-activated (sigmoid'd
            # then band-masked) preds, so activate=False (no internal sigmoid).
            from mmdet.models.losses import DiceLoss
            self.loss_dice_boundary = DiceLoss(
                use_sigmoid=True, activate=False, naive_dice=True,
                eps=1.0, loss_weight=self.sar_loss_b1_w, reduction='mean')

    # ------------------------------------------------------------------ #
    #  Geometry decode + closed-form oriented-elliptical rasterizer       #
    # ------------------------------------------------------------------ #
    def _decode_geom(self, raw: Tensor) -> dict:  # type: ignore[name-defined]
        """raw: (...,8) -> dict of decoded geometry on the OUTPUT grid units.

        Returns sin/cos (already L2-normalised), la/lb (pre-softplus, decoded to
        semi-axes by caller given a_min), tx/ty (pre-sigmoid, scaled by caller to
        the grid), gate (sigmoid), k (softplus + k0). Center scaling needs (H,W)
        so it is finished in _rasterize.
        """
        tx, ty = raw[..., 0], raw[..., 1]
        st, ct = raw[..., 2], raw[..., 3]
        # L2-normalise (sin,cos) so theta = atan2(st,ct) is well defined.
        # eps goes INSIDE the sqrt: clamp_min only bounds the VALUE, not the
        # gradient -- sqrt'(0)=inf gives NaN grad when st,ct~0 (the init), which
        # NaNs the weights at step 0 and crashes the Hungarian cost at step 1.
        norm = torch.sqrt(st * st + ct * ct + 1e-6)
        st = st / norm
        ct = ct / norm
        a = self.a_min + F.softplus(raw[..., 4])
        b = self.a_min + F.softplus(raw[..., 5])
        gate = torch.sigmoid(raw[..., 6])
        k = F.softplus(raw[..., 7]) + self.k0
        return dict(tx=tx, ty=ty, st=st, ct=ct, a=a, b=b, gate=gate, k=k)

    def _rasterize(self, geom: dict, H: int, W: int) -> Tensor:  # type: ignore[name-defined]
        """Closed-form oriented-elliptical occupancy LOGIT field.

        geom fields have shape (B,Q). Returns prior_logit (B,Q,H,W) = k*(1 - r2),
        positive inside the ellipse, 0 on the boundary, negative outside.
        """
        # NUMERICAL SAFETY: the unbounded k*(1-r2) term grows huge far outside the
        # ellipse and overflows fp16 (AMP) -> +/-Inf, which drives the BCE / Hungarian
        # cost to NaN ("matrix contains invalid numeric entries"). Compute in fp32 and
        # CLAMP the prior logit to a saturated-sigmoid range (correct: beyond ~+/-20
        # the gated prior is already a hard 0/1, so clamping changes nothing useful).
        odt = geom['tx'].dtype
        ys = torch.arange(H, device=geom['tx'].device, dtype=torch.float32).view(1, 1, H, 1)
        xs = torch.arange(W, device=geom['tx'].device, dtype=torch.float32).view(1, 1, 1, W)
        cx = torch.sigmoid(geom['tx'].float()).unsqueeze(-1).unsqueeze(-1) * W
        cy = torch.sigmoid(geom['ty'].float()).unsqueeze(-1).unsqueeze(-1) * H
        st = geom['st'].float().unsqueeze(-1).unsqueeze(-1)
        ct = geom['ct'].float().unsqueeze(-1).unsqueeze(-1)
        a = geom['a'].float().unsqueeze(-1).unsqueeze(-1)
        b = geom['b'].float().unsqueeze(-1).unsqueeze(-1)
        k = geom['k'].float().unsqueeze(-1).unsqueeze(-1)
        dx = xs - cx
        dy = ys - cy
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        r2 = (xr / a) ** 2 + (yr / b) ** 2
        prior = k * (1.0 - r2)
        return prior.clamp(-20.0, 20.0).to(odt)  # (B,Q,H,W)

    def _prior_logit(self, decoder_out: Tensor, H: int, W: int  # type: ignore[name-defined]
                     ) -> Tuple[Tensor, Tensor]:
        """Return (prior_logit (B,Q,H,W), gate (B,Q,1,1)) per the gen_mode.

        For 'a0' returns (zeros, zeros) so the readout is identical to stock.
        """
        B, Q = decoder_out.shape[:2]
        raw = self.geom_mlp(decoder_out)            # (B,Q,8)
        gate = torch.sigmoid(raw[..., 6]).view(B, Q, 1, 1)

        if self.gen_mode == 'a0':
            zeros = decoder_out.new_zeros((B, Q, H, W))
            return zeros, decoder_out.new_zeros((B, Q, 1, 1))

        if self.gen_mode == 'blind':
            # content-routed blend over R learned global basis masks; geometry-blind
            route = torch.softmax(raw[..., :self.prior_basis], dim=-1)   # (B,Q,R)
            basis = F.interpolate(
                self.blind_basis.unsqueeze(0), size=(H, W),
                mode='bilinear', align_corners=False)[0]                 # (R,H,W)
            prior = torch.einsum('bqr,rhw->bqhw', route, basis)
            return prior, gate

        if self.gen_mode == 'static':
            # ONE shared ellipse (geometry from nn.Parameter, NOT decoder_out);
            # gate stays per-query from GeomMLP[6].
            sg = self.static_geom
            raw_static = decoder_out.new_zeros((B, Q, 8))
            raw_static[..., 0] = sg[0]
            raw_static[..., 1] = sg[1]
            raw_static[..., 2] = sg[2]
            raw_static[..., 3] = sg[3]
            raw_static[..., 4] = sg[4]
            raw_static[..., 5] = sg[5]
            raw_static[..., 7] = sg[6]
            geom = self._decode_geom(raw_static)
            prior = self._rasterize(geom, H, W)
            return prior, gate

        # gen_mode == 'geo' : full per-query geometry-generated ellipse
        geom = self._decode_geom(raw)
        prior = self._rasterize(geom, H, W)
        return prior, gate

    # ------------------------------------------------------------------ #
    #  Override the per-decoder-layer readout: add the gated prior        #
    # ------------------------------------------------------------------ #
    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]
                      ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore[name-defined]
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        # ---- gated geometry prior, in logit space ----
        H, W = mask_feature.shape[-2:]
        prior, gate = self._prior_logit(decoder_out, H, W)
        mask_pred = mask_pred + gate * prior

        attn_mask = F.interpolate(
            mask_pred, attn_mask_target_size, mode='bilinear',
            align_corners=False)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        return cls_pred, mask_pred, attn_mask

    # ------------------------------------------------------------------ #
    #  Override forward to also inject the prior into the REFINED final    #
    #  mask (carrier SealM2FHead recomputes it at 1/2 res via mask_embed). #
    # ------------------------------------------------------------------ #
    def forward(self, x, batch_data_samples):
        # Replicate SealM2FHead.forward but, on the refined final-layer mask,
        # add the gated prior rasterised at the REFINED (1/2) resolution.
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

        # ---- refinement (carrier): recompute final-layer mask at higher res ----
        coarse_final = mask_pred_list[-1]
        hi_features = self._refine_features(mask_features)            # (B,C,2h,2w)
        decoder_out = self.transformer_decoder.post_norm(query_feat)
        mask_embed = self.mask_embed(decoder_out)
        refined_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, hi_features)
        # add the gated prior at the REFINED resolution
        Hh, Wh = hi_features.shape[-2:]
        prior_hi, gate = self._prior_logit(decoder_out, Hh, Wh)
        refined_mask = refined_mask + gate * prior_hi

        # inference-time blend/gate (carrier AMR) -- defaults == full refine
        soft = (not self.training) and (self.infer_soft_e0 is not None)
        v1blend = (not self.training) and (
            self.infer_blend < 1.0 or self.infer_size_gate is not None)
        if soft:
            coarse_up = F.interpolate(coarse_final, size=refined_mask.shape[-2:],
                                      mode='bilinear', align_corners=False)
            a_q = coarse_final.sigmoid().flatten(2).sum(-1).clamp_min(1e-6)
            e_q = a_q.sqrt()
            w_q = self.infer_soft_amax * torch.sigmoid(
                (float(self.infer_soft_e0) - e_q) / self.infer_soft_w)
            w_q = w_q.view(*w_q.shape, 1, 1)
            mask_pred_list[-1] = w_q * refined_mask + (1.0 - w_q) * coarse_up
        elif v1blend:
            coarse_up = F.interpolate(coarse_final, size=refined_mask.shape[-2:],
                                      mode='bilinear', align_corners=False)
            out = refined_mask
            if self.infer_blend < 1.0:
                a = self.infer_blend
                out = a * refined_mask + (1.0 - a) * coarse_up
            if self.infer_size_gate is not None:
                area = (refined_mask.sigmoid() > 0.5).flatten(2).sum(-1).float()
                use_refined = (area <= float(self.infer_size_gate)).view(*area.shape, 1, 1)
                out = torch.where(use_refined, out, coarse_up)
            mask_pred_list[-1] = out
        else:
            mask_pred_list[-1] = refined_mask

        return cls_pred_list, mask_pred_list

    # ------------------------------------------------------------------ #
    #  SAR loss add-on (toggleable; default off). Overrides the stock     #
    #  per-layer mask loss to thread Loss A (intensity-weighted BCE) and  #
    #  Loss B1 (boundary-band-weighted Dice), behind config flags.        #
    # ------------------------------------------------------------------ #
    def _loss_by_feat_single(self, cls_scores, mask_preds,
                             batch_gt_instances, batch_img_metas):
        if self.sar_loss_a_w <= 0 and self.sar_loss_b1_w <= 0:
            # stock path
            return super()._loss_by_feat_single(
                cls_scores, mask_preds, batch_gt_instances, batch_img_metas)

        from mmcv.ops import point_sample
        from mmdet.models.utils import get_uncertain_point_coords_with_randomness
        from mmdet.utils import reduce_mean

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(
            cls_scores_list, mask_preds_list, batch_gt_instances, batch_img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        mask_preds = mask_preds[mask_weights > 0]
        if mask_targets.shape[0] == 0:
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            losses = dict(loss_cls=loss_cls, loss_mask=loss_mask, loss_dice=loss_dice)
            if self.sar_loss_b1_w > 0:
                losses['loss_dice_boundary'] = mask_preds.sum()
            return tuple(losses.values())

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # ---- dice (stock) ----
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # ---- SAR Loss B1: boundary-band-weighted Dice (computed on same points) ----
        loss_dice_boundary = None
        if self.sar_loss_b1_w > 0:
            with torch.no_grad():
                gt = mask_targets.unsqueeze(1).float()           # (G,1,h,w)
                kk = 2 * self.sar_band_k + 1
                dil = F.max_pool2d(gt, kk, 1, self.sar_band_k)
                ero = -F.max_pool2d(-gt, kk, 1, self.sar_band_k)
                band = (dil - ero)                               # (G,1,h,w) ~ ring
                band_pts = point_sample(band, points_coords).squeeze(1)  # (G,NP)
                band_pts = (band_pts > 0.5).float()
            p = mask_point_preds.sigmoid() * band_pts            # pre-activated
            t = mask_point_targets * band_pts
            # activate=False -> DiceLoss uses p as-is (naive band-Dice)
            loss_dice_boundary = self.loss_dice_boundary(
                p, t, avg_factor=num_total_masks)

        # ---- SAR Loss A: intensity-weighted BCE (up-weight DARK in-GT pixels) ----
        flat_preds = mask_point_preds.reshape(-1)
        flat_targets = mask_point_targets.reshape(-1)
        if self.sar_loss_a_w > 0:
            with torch.no_grad():
                amp = self._gather_amp_points(
                    batch_gt_instances, mask_targets, points_coords)  # (G,NP) in [0,1]
            w_int = 1.0 + self.sar_lam_int * mask_point_targets * \
                (1.0 - amp).clamp(0, 1) ** self.sar_gamma_int
            loss_mask = self.loss_mask(
                flat_preds, flat_targets, weight=w_int.reshape(-1),
                avg_factor=num_total_masks * self.num_points)
        else:
            loss_mask = self.loss_mask(
                flat_preds, flat_targets,
                avg_factor=num_total_masks * self.num_points)

        out = [loss_cls, loss_mask, loss_dice]
        if loss_dice_boundary is not None:
            out.append(loss_dice_boundary)
        return tuple(out)

    def loss_by_feat(self, all_cls_scores, all_mask_preds,
                     batch_gt_instances, batch_img_metas):
        # Re-implement the multi-layer aggregation to carry the optional extra
        # loss key (loss_dice_boundary) per layer. Falls back to stock when the
        # SAR losses are off (handled inside _loss_by_feat_single).
        if self.sar_loss_a_w <= 0 and self.sar_loss_b1_w <= 0:
            return super().loss_by_feat(
                all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)

        num_dec_layers = len(all_cls_scores)
        batch_gt_list = [batch_gt_instances for _ in range(num_dec_layers)]
        batch_meta_list = [batch_img_metas for _ in range(num_dec_layers)]

        results = [self._loss_by_feat_single(all_cls_scores[i], all_mask_preds[i],
                                             batch_gt_list[i], batch_meta_list[i])
                   for i in range(num_dec_layers)]
        has_b1 = self.sar_loss_b1_w > 0
        loss_dict = dict()
        # last layer
        last = results[-1]
        loss_dict['loss_cls'] = last[0]
        loss_dict['loss_mask'] = last[1]
        loss_dict['loss_dice'] = last[2]
        if has_b1:
            loss_dict['loss_dice_boundary'] = last[3]
        # aux layers
        for i, res in enumerate(results[:-1]):
            loss_dict[f'd{i}.loss_cls'] = res[0]
            loss_dict[f'd{i}.loss_mask'] = res[1]
            loss_dict[f'd{i}.loss_dice'] = res[2]
            if has_b1:
                loss_dict[f'd{i}.loss_dice_boundary'] = res[3]
        return loss_dict

    # ------------------------------------------------------------------ #
    #  SAR amplitude at the mask-target grid, sampled at the loss points. #
    #  Pulls the per-instance amplitude crop from gt_instances if present #
    #  (carried as `amp_maps` by the StashLowResAmplitude transform);     #
    #  otherwise falls back to a constant 0.5 (=> Loss A reduces to a      #
    #  near-uniform interior up-weight, still finite and well-defined).    #
    # ------------------------------------------------------------------ #
    def _gather_amp_points(self, batch_gt_instances, mask_targets, points_coords):
        from mmcv.ops import point_sample
        amps = []
        for gi in batch_gt_instances:
            amp = getattr(gi, 'amp_maps', None)
            if amp is not None:
                amps.append(amp)
        if len(amps) == 0:
            # no amplitude carried -> neutral 0.5 amplitude everywhere
            return mask_targets.new_full(
                (mask_targets.shape[0], self.num_points), 0.5)
        amp_all = torch.cat(amps, dim=0).to(mask_targets)        # (G,h,w)
        if amp_all.shape[0] != mask_targets.shape[0]:
            return mask_targets.new_full(
                (mask_targets.shape[0], self.num_points), 0.5)
        amp_pts = point_sample(
            amp_all.unsqueeze(1), points_coords).squeeze(1)      # (G,NP)
        return amp_pts.clamp(0, 1)
