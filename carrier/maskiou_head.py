# -*- coding: utf-8 -*-
"""Mask-IoU quality head for Mask2Former (Mask Scoring R-CNN adapted to per-query M2F).

WHY (diagnosis-driven, source-verified 2026-06-27): mmdet's MaskFormerFusionHead ranks
instances by `det_scores = softmax(cls) * mask_confidence`, where mask_confidence is the
*foreground-average sigmoid* (maskformer_fusion_head.py:170-173) -- NOT the predicted mask
IoU. This is exactly the score/quality mismatch Mask Scoring R-CNN (Huang et al., CVPR 2019,
arXiv:1903.00241) identified: at high IoU thresholds (AP75/AP90) a blurry-but-confident mask
can outrank a tight-but-slightly-less-confident one. On PSeg-SSDD inshore (clutter/adjacency)
this directly costs AP75. This head learns to predict per-query mask IoU and ranks by it.

Orthogonal to every dead negative on this carrier: dynamic-kernel/DS-3x3 (readout DOF),
feature enrichment (OAEM/DenoDet/CAA/GeoField), boundary-loss (wash), capacity (Swin-L) --
all left mask scoring untouched. This is the one un-probed head-level bottleneck.

DEFAULT-OFF & SAFE: maskiou_weight=0 -> no IoU loss; infer_iou_lambda=0 -> scoring is
byte-identical to stock. So adding these classes never perturbs existing configs/training.
Only a0r_msiou_long_pseg.py (weight>0, lambda swept) turns the mechanism on.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet.models.seg_heads.panoptic_fusion_heads.maskformer_fusion_head import (
    MaskFormerFusionHead)
from mmdet.registry import MODELS
from mmdet.structures.mask import mask2bbox

from .seal_m2f_head import SealM2FHead


class MaskIoUHead(nn.Module):
    """Light per-query MaskIoU regressor. Input = query embedding (the per-instance
    representation in M2F) concatenated with 2 cheap mask statistics (mean fg prob,
    peak prob). Output = a raw IoU logit per query (sigmoid -> predicted IoU in [0,1]).
    No spatial conv (the DS-3x3/dynamic-kernel spatial-DOF axis is a confirmed negative
    here); the query embedding already encodes the instance."""

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_feat: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        # query_feat: (B, Q, C); mask_logits: (B, Q, h, w) -> returns (B, Q) raw logit
        p = mask_logits.sigmoid().flatten(2)            # (B, Q, hw)
        stats = torch.stack([p.mean(-1), p.amax(-1)], dim=-1)  # (B, Q, 2)
        feat = torch.cat([query_feat, stats], dim=-1)   # (B, Q, C+2)
        return self.mlp(feat).squeeze(-1)               # (B, Q)


class MaskIoUHeadV2(nn.Module):
    """Richer per-query MaskIoU regressor (2026-07-05, oracle precheck showed +7.66 AP rescoring headroom on the
    strong c0 base -> the bottleneck was the PREDICTOR, not the axis). Adds ~11 mask-shape/sharpness statistics
    (the current head used only mean+peak) so IoU is predicted better: fg-area fraction, prob mean/std/peak,
    p10/p50/p90 quantiles, foreground-vs-background separation (mean_fg - mean_bg at 0.5), boundary-band ratio
    (soft edge mass / area = sharpness proxy), and 2 log-scale area terms. No spatial conv (DS-3x3 is a known
    negative); the query embedding + these global stats encode IoU-predictability. Same (query_feat, mask_logits)
    signature as MaskIoUHead so it drops into MSIoUM2FHead unchanged."""

    N_STATS = 11

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + self.N_STATS, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, query_feat: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        B, Q, h, w = mask_logits.shape
        p = mask_logits.sigmoid().flatten(2)                       # (B,Q,hw)
        fg = (p > 0.5).float()
        area = fg.mean(-1)                                          # fg-area fraction
        mean = p.mean(-1); std = p.std(-1); peak = p.amax(-1)
        q10 = torch.quantile(p, 0.10, dim=-1); q50 = torch.quantile(p, 0.50, dim=-1)
        q90 = torch.quantile(p, 0.90, dim=-1)
        nfg = fg.sum(-1).clamp_min(1.0); nbg = (1.0 - fg).sum(-1).clamp_min(1.0)
        mean_fg = (p * fg).sum(-1) / nfg
        mean_bg = (p * (1.0 - fg)).sum(-1) / nbg
        sep = mean_fg - mean_bg                                     # fg/bg separation (sharpness)
        # boundary-band sharpness: soft edge mass = mean |grad(sigmoid)| over the map, per query
        pm = p.view(B, Q, h, w)
        gx = (pm[:, :, :, 1:] - pm[:, :, :, :-1]).abs().flatten(2).mean(-1)
        gy = (pm[:, :, 1:, :] - pm[:, :, :-1, :]).abs().flatten(2).mean(-1)
        edge = gx + gy
        logarea = torch.log(area.clamp_min(1e-4))
        stats = torch.stack([area, mean, std, peak, q10, q50, q90, sep, mean_fg, edge, logarea], dim=-1)  # (B,Q,11)
        feat = torch.cat([query_feat, stats], dim=-1)
        return self.mlp(feat).squeeze(-1)


@MODELS.register_module()
class MSIoUM2FHead(SealM2FHead):
    """SealM2FHead + a MaskIoU head. forward() reuses the base (which now stashes the
    final query_feat as self._last_query_feat); we compute & stash the per-query IoU
    logit. Training adds an IoU-regression loss on matched queries; inference threads
    the predicted IoU to MSIoUFusionHead via a data_sample attribute."""

    def __init__(self, *args, maskiou_weight: float = 0.0, maskiou_hidden: int = 256,
                 maskiou_loss_type: str = 'bce', maskiou_v2: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskiou_weight = float(maskiou_weight)
        self.maskiou_loss_type = maskiou_loss_type
        c = self.mask_embed[-1].out_features  # = feat_channels = query_feat dim
        HeadCls = MaskIoUHeadV2 if maskiou_v2 else MaskIoUHead
        self.maskiou_head = HeadCls(c, hidden=maskiou_hidden)
        self._last_pred_iou_logit = None

    def forward(self, x, batch_data_samples):
        cls_pred_list, mask_pred_list = super().forward(x, batch_data_samples)
        # base stashes self._last_query_feat (final-layer query embedding, pre post_norm)
        qf = getattr(self, '_last_query_feat', None)
        if qf is not None:
            self._last_pred_iou_logit = self.maskiou_head(qf, mask_pred_list[-1])
        return cls_pred_list, mask_pred_list

    @torch.no_grad()
    def _target_iou(self, mp_logits, mt):
        """IoU between binarized pred mask and GT mask. mp_logits,mt: (M,h,w)/(M,H,W)."""
        pb = (mp_logits.sigmoid() > 0.5).float()
        g = mt.float()
        if g.shape[-2:] != pb.shape[-2:]:
            g = F.interpolate(g.unsqueeze(1), size=pb.shape[-2:], mode='nearest').squeeze(1)
        inter = (pb * g).flatten(1).sum(1)
        union = pb.flatten(1).sum(1) + g.flatten(1).sum(1) - inter
        return inter / union.clamp_min(1e-6)

    def _maskiou_loss(self, cls_final, mask_final, batch_gt_instances, batch_img_metas):
        n = cls_final.size(0)
        (_, _, mtl, mwl, _) = self.get_targets(
            [cls_final[i] for i in range(n)], [mask_final[i] for i in range(n)],
            batch_gt_instances, batch_img_metas)
        mt = torch.cat(mtl, dim=0)                       # (M, H, W) matched GT masks
        mw = torch.stack(mwl, dim=0)                     # (B, Q) mask weights (1 = matched)
        sel = mw > 0
        if mt.shape[0] == 0 or self._last_pred_iou_logit is None:
            return self._last_pred_iou_logit.sum() * 0.0 if self._last_pred_iou_logit is not None \
                else mask_final.sum() * 0.0
        mp = mask_final[sel]                             # (M, h, w) matched pred masks
        pred_logit = self._last_pred_iou_logit[sel]      # (M,) matched IoU preds
        tgt = self._target_iou(mp, mt)                   # (M,) target IoU in [0,1]
        if self.maskiou_loss_type == 'l2':
            loss = F.mse_loss(pred_logit.sigmoid(), tgt)
        else:                                            # BCE-with-logits (default, stable)
            loss = F.binary_cross_entropy_with_logits(pred_logit, tgt)
        return self.maskiou_weight * loss

    def loss(self, x, batch_data_samples):
        # replicate MaskFormerHead.loss so we have all_* + preprocessed gt in scope
        batch_img_metas, batch_gt_instances, batch_gt_sem = [], [], []
        for ds in batch_data_samples:
            batch_img_metas.append(ds.metainfo)
            batch_gt_instances.append(ds.gt_instances)
            batch_gt_sem.append(ds.gt_sem_seg if 'gt_sem_seg' in ds else None)
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)  # sets _last_pred_iou_logit
        batch_gt_instances = self.preprocess_gt(batch_gt_instances, batch_gt_sem)
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        if self.maskiou_weight > 0:
            losses['loss_maskiou'] = self._maskiou_loss(
                all_cls_scores[-1], all_mask_preds[-1], batch_gt_instances, batch_img_metas)
        return losses

    @torch.no_grad()
    def _depth_stats(self, all_cls_scores, all_mask_preds):
        """R12: depth-consistency features from the LAST 3 decoder layers.

        Mask2Former's forward emits per-layer (cls, mask) for ALL decoder
        layers; stock inference throws layers 1..8 away. Same list index =
        same query SLOT across layers (the decoder refines queries in place),
        so cross-depth matching is index-trivial. Depth stability of a query
        is a free confidence signal.

        Layer indexing: the lists have num_layers+1 entries (entry 0 = the
        pre-decoder head on the initial queries); [-3],[-2],[-1] are the
        outputs AFTER decoder layers 7, 8, 9 (L7/L8/L9) for the 9-layer M2F.

        Resolution: every layer's mask logits live at the mask_features res
        (H/4) EXCEPT the final one -- SealM2FHead replaces mask_pred_list[-1]
        with a 2x-refined (H/2) mask. We bilinear-downsample L9's logits to
        L8's native low res before binarizing (>0), keeping memory tiny and
        the comparison at matched scale.

        IoU convention: both-empty (union==0) -> 1.0 (perfect agreement).

        Returns (B, Q, 5) float32:
          [iou_L7L8, iou_L8L9, score_L7, score_L8, score_L9]
        where score_Lk = max cls softmax EXCLUDING background at layer k.
        NOTE: 5 columns, not the 4 of the original R12 sketch -- score_L9 is
        carried explicitly because the fused instance's det score is
        cls*mask_conf (contaminated), so the TTA-side score trajectory
        (score_L9 - score_L7) cannot be recovered from .scores at fuse time.
        """
        m7, m8, m9 = all_mask_preds[-3], all_mask_preds[-2], all_mask_preds[-1]
        if m9.shape[-2:] != m8.shape[-2:]:
            m9 = F.interpolate(m9, size=m8.shape[-2:], mode='bilinear',
                               align_corners=False)

        def _pair_iou(a_logits, b_logits):
            a = (a_logits > 0).flatten(2).float()          # (B, Q, hw)
            b = (b_logits > 0).flatten(2).float()
            inter = (a * b).sum(-1)
            union = a.sum(-1) + b.sum(-1) - inter
            iou = inter / union.clamp_min(1e-6)
            return torch.where(union > 0, iou, torch.ones_like(iou))

        def _fg_score(cls_pred):
            return F.softmax(cls_pred, dim=-1)[..., :-1].amax(-1)  # (B, Q)

        return torch.stack([
            _pair_iou(m7, m8),
            _pair_iou(m8, m9),
            _fg_score(all_cls_scores[-3]),
            _fg_score(all_cls_scores[-2]),
            _fg_score(all_cls_scores[-1]),
        ], dim=-1).float()                                  # (B, Q, 5)

    def predict(self, x, batch_data_samples):
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)  # sets _last_pred_iou_logit
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if self._last_pred_iou_logit is not None:
            pred_iou = self._last_pred_iou_logit.sigmoid()           # (B, Q) in [0,1]
            for i, ds in enumerate(batch_data_samples):
                ds._maskiou = pred_iou[i]                            # thread to fusion head
        # R12 (OFF by default -> GATE-0 byte-identical, no extra attribute):
        # per-query depth stats threaded to MSIoUFusionHead via the data
        # sample, exactly like _maskiou above. Flag lives in model.test_cfg
        # (maskformer.py copies it into BOTH this head's and the fusion
        # head's test_cfg, so one --cfg-options key serves both).
        if self.test_cfg is not None and self.test_cfg.get(
                'export_depth_stats', False):
            depth = self._depth_stats(all_cls_scores, all_mask_preds)
            for i, ds in enumerate(batch_data_samples):
                ds._depth_stats = depth[i]                           # (Q, 5)
        img_shape = batch_img_metas[0]['batch_input_shape']
        # R4-2: interp_mode knob (default 'bilinear' = byte-identical). 'bicubic'
        # sharpens the 4x logit upsample (200->800) iso-contour for small ships.
        _mode = (self.test_cfg.get('interp_mode', 'bilinear')
                 if self.test_cfg is not None else 'bilinear')
        mask_pred_results = F.interpolate(
            mask_pred_results, size=(img_shape[0], img_shape[1]),
            mode=_mode, align_corners=False)
        return mask_cls_results, mask_pred_results


@MODELS.register_module()
class MSIoUFusionHead(MaskFormerFusionHead):
    """MaskFormerFusionHead whose instance scoring blends predicted mask-IoU into the
    rank: det_scores = cls * mask_conf^(1-lambda) * pred_iou^lambda. lambda=0 ->
    byte-identical to stock (the operating point is swept on a single checkpoint)."""

    def __init__(self, *args, infer_iou_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_iou_lambda = float(infer_iou_lambda)

    def instance_postprocess(self, mask_cls, mask_pred, pred_iou=None,
                             depth_stats=None):
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)
        labels_per_image = labels[top_indices]
        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() * mask_pred_binary).flatten(1).sum(1) / (
            mask_pred_binary.flatten(1).sum(1) + 1e-6)
        lam = self.infer_iou_lambda
        if lam > 0 and pred_iou is not None:
            iou_sel = pred_iou[query_indices][is_thing].clamp_min(1e-6)
            det_scores = scores_per_image * mask_scores_per_image.clamp_min(1e-6).pow(1.0 - lam) \
                * iou_sel.pow(lam)
        else:
            det_scores = scores_per_image * mask_scores_per_image   # byte-identical to stock
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)

        results = InstanceData()
        results.bboxes = bboxes
        results.labels = labels_per_image
        results.scores = det_scores
        results.masks = mask_pred_binary
        # F4 (OFF by default): ALSO export the FLOAT sigmoid probability maps so
        # the TTA merge can average sub-pixel boundary evidence instead of
        # binarized {0,1} votes (binary majority erodes boundaries: offshore
        # 0.559). Stored at the SAME resolution as results.masks (mask_pred here
        # is already cropped+rescaled to ori_shape by predict(rescale=True)), so
        # (mask_probs >= 0.5) == masks pixel-for-pixel (sigmoid is monotone;
        # logit > 0 <=> prob >= 0.5 up to the measure-zero logit==0 / fp16
        # rounding band |logit| < ~5e-4). Full-res was chosen over head-res
        # 200x200 because it guarantees exact geometric consistency with the
        # binary masks the vote clustering uses and avoids re-implementing the
        # interpolate->crop->rescale chain at fuse time; fp16 halves the cost
        # (<=100 x 800 x 800 fp16 ~ 128 MB/view on the 24 GB card at eval).
        # Flag lives in model.test_cfg (maskformer.py copies it into this head's
        # test_cfg). Default False -> attribute absent -> byte-identical output.
        if self.test_cfg is not None and self.test_cfg.get('export_mask_probs',
                                                           False):
            results.mask_probs = mask_pred.sigmoid().half()
        # R12 (OFF by default): per-KEPT-INSTANCE depth stats. ALIGNMENT:
        # kept instances are built as mask_pred[query_indices][is_thing]
        # (lines above), i.e. instance row i <-> query query_indices[is_thing][i]
        # of the FINAL layer. depth_stats is per-query (Q, 5), so indexing it
        # with the SAME [query_indices][is_thing] chain aligns it 1:1 with
        # every other per-instance field by construction (identical to the
        # battle-tested pred_iou threading just above). depth_stats is only
        # non-None when the head ran with export_depth_stats=True.
        if depth_stats is not None:
            results.depth_stats = depth_stats[query_indices][is_thing]
        return results

    def predict(self, mask_cls_results, mask_pred_results, batch_data_samples,
                rescale: bool = False, **kwargs):
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'semantic segmentation results are not supported yet.'

        results = []
        for i, (mask_cls_result, mask_pred_result, meta) in enumerate(
                zip(mask_cls_results, mask_pred_results, batch_img_metas)):
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            if rescale:
                ori_height, ori_width = meta['ori_shape'][:2]
                _mode = self.test_cfg.get('interp_mode', 'bilinear')
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None], size=(ori_height, ori_width),
                    mode=_mode, align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                result['pan_results'] = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
            if instance_on:
                pred_iou = getattr(batch_data_samples[i], '_maskiou', None)
                depth_stats = getattr(batch_data_samples[i], '_depth_stats',
                                      None)
                result['ins_results'] = self.instance_postprocess(
                    mask_cls_result, mask_pred_result, pred_iou, depth_stats)
            results.append(result)
        return results
