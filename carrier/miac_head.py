# -*- coding: utf-8 -*-
"""MIACM2FHead: Mask-IoU-Aware Classification soft-target for Mask2Former.

WHY (diagnosis-driven, 2026-07-06): on the STRONG COCO-init base (c0) the
classification score is IoU-blind -- measured rho(cls, true-mask-IoU) = -0.213.
An oracle rescore by the true mask-IoU gives +7.66 AP inshore. The sibling
MSIoU head (maskiou_head.py) attacks this at *inference* by blending a learned
IoU into the fusion rank. MIAC attacks it at the *source*: it folds the mask-IoU
into the CLASSIFICATION TARGET during training (Align-DETR IA-BCE / QFL / VarifocalNet
lineage). For each MATCHED query the hard cls target [ship=1, no-object=0] is
replaced by a SOFT target where ship-target = f(u), no-object-target = 1-f(u),
u = the query's predicted-mask-vs-GT mask-IoU. This calibrates cls-prob to
localization quality so cls * mask_conf ranks better at high IoU thresholds.

f(u) = (1 - w) + w * u^gamma, with w = iou_cls_weight (default 0.0), gamma =
iou_cls_gamma (default 1.0). w = 0  =>  f(u) = 1  =>  hard [1, 0] target  =>
loss_cls is BYTE-IDENTICAL to stock Mask2FormerHead (GATE-0 contract, same as
the maskiou/wass siblings: adding this class never perturbs an unset config).

Only loss_cls is touched; loss_mask / loss_dice are the stock returns verbatim.
The soft-cls path replicates the exact stock loss_cls arithmetic (see below), so
at w=0 the two are the same tensor to fp precision.

STOCK loss_cls arithmetic replicated (mmdet 3.3.0 Mask2FormerHead._loss_by_feat_single,
CrossEntropyLoss with use_sigmoid=False, reduction='mean', loss_weight=2.0,
class_weight=[1.0]*num_classes + [0.1], avg_non_ignore=False):
    class_weight = cls_scores.new_tensor(self.class_weight)           # (C+1,)
    loss_cls = self.loss_cls(cls_scores_flat, labels_flat, label_weights_flat,
                             avg_factor=class_weight[labels_flat].sum())
  which expands to
    elem = F.cross_entropy(cls, labels, weight=class_weight, reduction='none')  # (N,)
         = class_weight[labels] * (-log_softmax(cls)[labels])
    loss = loss_weight * (elem * label_weights).sum() / (avg_factor + eps)
For the soft version we replace the hard one-hot cross-entropy by the general
soft cross-entropy  -sum_c t[c] * log_softmax(cls)[c]  and keep EVERY other
factor (per-sample class_weight[hard_label], label_weights, avg_factor,
loss_weight, eps) identical -- so at t = hard one-hot the two are equal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.registry import MODELS

_EPS_FP32 = torch.finfo(torch.float32).eps  # exactly mmdet weight_reduce_loss eps


@MODELS.register_module()
class MIACM2FHead(Mask2FormerHead):
    """Mask2FormerHead with a Mask-IoU-Aware Classification soft target.

    iou_cls_weight=0 -> loss_cls byte-identical to stock (GATE-0).
    iou_cls_weight>0 -> for each matched query, cls soft target for the ship
    class is (1 - w) + w * u^gamma (u = mask-IoU), no-object gets the rest.
    """

    def __init__(self, *args, iou_cls_weight: float = 0.0,
                 iou_cls_gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_cls_weight = float(iou_cls_weight)
        self.iou_cls_gamma = float(iou_cls_gamma)

    @torch.no_grad()
    def _target_iou(self, mp_logits, mt):
        """IoU between binarized pred mask and GT mask. mp_logits,mt: (M,h,w)/(M,H,W).

        Identical to MaskIoUHead._target_iou (maskiou_head.py) -- binarize the
        predicted mask at sigmoid>0.5, resize GT to the pred (1/4) grid, plain
        intersection / union. detached (no grad through the target)."""
        pb = (mp_logits.sigmoid() > 0.5).float()
        g = mt.float()
        if g.shape[-2:] != pb.shape[-2:]:
            g = F.interpolate(g.unsqueeze(1), size=pb.shape[-2:],
                              mode='nearest').squeeze(1)
        inter = (pb * g).flatten(1).sum(1)
        union = pb.flatten(1).sum(1) + g.flatten(1).sum(1) - inter
        return inter / union.clamp_min(1e-6)

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances,
                             batch_img_metas):
        # Stock returns for loss_mask / loss_dice (and loss_cls when w==0).
        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        if self.iou_cls_weight <= 0:
            return loss_cls, loss_mask, loss_dice  # GATE-0: identity

        # --- recompute ONLY loss_cls with soft targets ---
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         _avg) = self.get_targets(cls_scores_list, mask_preds_list,
                                  batch_gt_instances, batch_img_metas)
        # Flatten EXACTLY like stock: (B,Q,C)->(B*Q,C); labels (B,Q)->(B*Q).
        labels = torch.stack(labels_list, dim=0)              # (B,Q)
        label_weights = torch.stack(label_weights_list, dim=0)  # (B,Q)
        mask_weights = torch.stack(mask_weights_list, dim=0)  # (B,Q)
        mask_targets = torch.cat(mask_targets_list, dim=0)    # (M,H,W) matched GT
        cls_flat = cls_scores.flatten(0, 1)                   # (N=B*Q, C+1)
        labels_flat = labels.flatten(0, 1)                    # (N,)
        label_weights_flat = label_weights.flatten(0, 1).float()  # (N,)

        num_classes = self.num_classes                        # no-object index = num_classes
        class_weight = cls_flat.new_tensor(self.class_weight)  # (C+1,)

        # avg_factor: stock uses class_weight[labels].sum() for loss_cls.
        avg_factor = class_weight[labels_flat].sum()

        # Build the soft target distribution over C+1 classes, one row per query.
        # Start from the HARD one-hot (matches stock when u-blend collapses):
        soft_target = torch.zeros_like(cls_flat)              # (N, C+1)
        soft_target[torch.arange(cls_flat.size(0), device=cls_flat.device),
                    labels_flat] = 1.0

        # Matched queries = mask_weights>0, in the SAME flatten order as cls_flat
        # (both are (B,Q)->(B*Q) row-major). mask_preds[mask_weights>0] and
        # mask_targets share that order (stock invariant reused by MSIoU/wass).
        matched = (mask_weights.flatten(0, 1) > 0)            # (N,) bool
        n_matched = int(matched.sum().item())
        if n_matched > 0 and mask_targets.shape[0] == n_matched:
            mp = mask_preds[mask_weights > 0]                 # (M,h,w) matched pred logits
            u = self._target_iou(mp, mask_targets)            # (M,) in [0,1], detached
            u = u.clamp(0.0, 1.0).pow(self.iou_cls_gamma)
            f = (1.0 - self.iou_cls_weight) + self.iou_cls_weight * u  # (M,) ship target
            f = f.to(cls_flat.dtype)
            m_idx = torch.nonzero(matched, as_tuple=False).squeeze(1)  # (M,) row ids
            m_labels = labels_flat[m_idx]                     # ship class id per matched query
            # ship-class prob = f ; no-object prob = 1 - f ; others 0.
            soft_target[m_idx] = 0.0
            soft_target[m_idx, m_labels] = f
            soft_target[m_idx, num_classes] = 1.0 - f

        # Soft cross-entropy matching the stock per-sample arithmetic:
        #   elem_i = class_weight[hard_label_i] * ( -sum_c t_i[c] * log_softmax(cls_i)[c] )
        # (stock hard case = class_weight[label]*(-log_softmax[label]); identical
        #  when t is one-hot at label.) Then weight by label_weights, sum, /avg.
        log_prob = F.log_softmax(cls_flat, dim=-1)            # (N, C+1)
        ce = -(soft_target * log_prob).sum(dim=-1)            # (N,)
        cls_w_per_sample = class_weight[labels_flat]          # (N,) same weighting as stock
        elem = ce * cls_w_per_sample * label_weights_flat     # (N,)
        loss_cls_soft = self.loss_cls.loss_weight * elem.sum() / (avg_factor + _EPS_FP32)

        return loss_cls_soft, loss_mask, loss_dice
