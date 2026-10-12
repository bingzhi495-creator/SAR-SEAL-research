# -*- coding: utf-8 -*-
"""CHARTM2FHead: Clutter-Hard Auxiliary Reinforced Targeting for Mask2Former.

WHY (diagnosis-driven, precheck PASSED 2026-07-13): on the strong COCO-init base
(c0) ~22% of near-shore ships are A-class COMPLETE MISSES -- NO decoder query
fires on them (best-query mask-IoU < 0.1 for the strict-A band; 52.5% of the
*unfired* misses carry a WEAK-but-real full-res response, 0.1<=IoU<0.5, that MORE
positive supervision can plausibly AMPLIFY). Stock Mask2Former uses a ONE-to-ONE
Hungarian match: each GT gets exactly one positive query, so positive-query
gradient on a hard GT is thin -- a single weakly-firing query is easily out-voted
by the no-object gradient and never learns to fire.

MECHANISM (one-to-many auxiliary supervision, loss-side, zero inference cost):
in ADDITION to the stock one-to-one match, each GT is ALSO matched to its top-k
best-fitting UNMATCHED queries, and an auxiliary cls+mask+dice loss (down-weighted
by aux_weight) is applied on those extra positive (query, GT) pairs. This raises
the density of positive-query supervision so MORE queries learn to fire on hard
GTs. Inference is UNTOUCHED (stock scoring/fusion; the extra positives only add a
training gradient) -> zero inference cost, no new inference params.

SAR TWIST (CHART, the novelty vs uniform Group-DETR / Co-DETR one-to-many): the
auxiliary budget is CONCENTRATED on clutter-hard near-shore GTs, not spread
uniformly. A cheap per-GT difficulty weight w_g is computed at train time from
quantities measurable *inside the loss* from the GT masks alone:
  * CROWDING -- nearest-neighbour normalized-centroid distance to other GTs in the
    same image (the precheck's strongest single miss-predictor, AUC 0.749);
    closer neighbour -> harder.
  * SMALL SIZE -- GT mask area; smaller -> harder.
(Local contrast / clutter energy -- also predictive -- are NOT used here: they need
the input pixels, which are not available in _loss_by_feat_single without threading
the image through; crowding+size are the two features that ARE cheaply available
from the masks, and crowding alone already carries the bulk of the signal.)
Each GT's aux loss is scaled by w_g (per-image mean 1, so hard_focus only
REDISTRIBUTES a fixed aux budget -- clean ablation), landing the extra supervision
on the hard positives. This is the "asymmetric supervision budget by SAR near-shore
difficulty" that differentiates CHART from uniform one-to-many.

KNOBS:
  aux_weight  (float, default 0.0) -- overall one-to-many strength. 0 => OFF.
  aux_topk    (int,   default 3)   -- extra positive queries per GT.
  hard_focus  (float, default 1.0) -- 0 = uniform aux budget, >0 = difficulty-
                                       weighted (w_g = softmax_temper(hard_focus*h_g)).

GATE-0 (byte-exactness): with aux_weight <= 0 the head returns super()'s
(loss_cls, loss_mask, loss_dice) VERBATIM on EVERY decoder layer -- no second
get_targets call, no aux arithmetic, no extra RNG consumption -- so the total
training loss over main + all aux layers is BYTE-IDENTICAL to stock
Mask2FormerHead, and inference is stock (forward/predict inherited unchanged).
The aux term is composed by ADDING to super()'s returns (same style as the
SealM2FHead boundary term / MIAC soft-cls sibling); the stock arithmetic is never
copy-pasted -- it comes straight from super()._loss_by_feat_single.

HONEST RISK (R1): uniform Group-DETR-style one-to-many is only ~+0.4 AP on a
strong Swin-L backbone -- near-zero. CHART's bet is that (a) CONCENTRATING the
budget on the clutter-hard misses (not uniform) and (b) the SAR near-shore regime
(22% complete-miss, far worse than COCO) make the gain materially larger than the
uniform baseline. This is plausible but UNPROVEN; it may still not gain. Decide on
the c0-vs-c0+CHART ablation, not on faith.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.registry import MODELS

_EPS = 1e-6


@MODELS.register_module()
class CHARTM2FHead(Mask2FormerHead):
    """Mask2FormerHead + clutter-hard one-to-many auxiliary supervision.

    aux_weight=0  -> loss byte-identical to stock Mask2FormerHead (GATE-0).
    aux_weight>0  -> adds a down-weighted cls+mask+dice loss on the top-k best
                     UNMATCHED queries per GT, concentrated on clutter-hard GTs.
    """

    def __init__(self, *args, aux_weight: float = 0.0, aux_topk: int = 3,
                 hard_focus: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_weight = float(aux_weight)
        self.aux_topk = int(aux_topk)
        self.hard_focus = float(hard_focus)

    # ------------------------------------------------------------------ SAR twist
    @torch.no_grad()
    def _chart_difficulty_weights(self, gt_masks) -> torch.Tensor:
        """Per-GT difficulty weight w_g (per-image, mean 1) from the GT masks.

        Hardness h_g = z(-nn_dist) + z(-log area):
          nn_dist = nearest-neighbour distance between normalized centroids
                    (crowding; closer -> harder; the precheck's AUC-0.749 signal),
          area    = GT mask pixel area (smaller -> harder).
        w_g = n * softmax(hard_focus * h_g) so:
          hard_focus=0 -> w_g == 1 for all (uniform budget),
          hard_focus>0 -> monotonically larger w_g on harder GTs (fixed budget
                          REDISTRIBUTED, sum(w_g)=n preserved). Detached.
        """
        m = gt_masks.float()
        n = m.shape[0]
        if n == 0:
            return m.new_zeros((0,))
        if n == 1:
            return m.new_ones((1,))
        H, W = m.shape[-2:]
        dev = m.device
        area = m.flatten(1).sum(1).clamp_min(1.0)                    # (n,)
        ys = torch.arange(H, device=dev, dtype=torch.float32)
        xs = torch.arange(W, device=dev, dtype=torch.float32)
        cy = (m.sum(2) * ys).sum(1) / area / H                      # (n,) norm centroid y
        cx = (m.sum(1) * xs).sum(1) / area / W                      # (n,) norm centroid x
        cent = torch.stack([cx, cy], dim=1)                         # (n, 2)
        d = torch.cdist(cent, cent) + torch.eye(n, device=dev) * 1e9
        nn = d.min(1).values                                        # (n,) nearest-neighbour dist

        def z(x):
            return (x - x.mean()) / (x.std() + _EPS)

        h = z(-nn) + z(-torch.log(area + 1.0))                      # higher = harder
        u = torch.exp((self.hard_focus * h).clamp(-4.0, 4.0))       # temper; hf=0 -> all 1
        w = n * u / (u.sum() + _EPS)                                # mean 1
        return w

    # ------------------------------------------------------------- one-to-many aux
    def _chart_aux_loss(self, cls_scores, mask_preds, mask_weights_list,
                        batch_gt_instances):
        """Auxiliary cls+mask+dice on top-k UNMATCHED queries per GT.

        Returns (aux_cls, aux_mask, aux_dice) each a scalar tensor connected to the
        graph (grad flows to cls_scores / mask_preds). Zero-but-connected when no
        aux pair exists. Each pair is weighted by its GT's difficulty weight w_g.
        """
        B, Q = cls_scores.shape[:2]
        h, w = mask_preds.shape[-2:]
        num_classes = self.num_classes
        class_weight = cls_scores.new_tensor(self.class_weight)     # (C+1,)
        k = max(self.aux_topk, 0)

        p_rows, t_rows, ce_list, wt_list = [], [], [], []
        for i in range(B):
            gt = batch_gt_instances[i]
            gt_masks = gt.masks
            gt_labels = gt.labels
            n_g = int(gt_masks.shape[0])
            if n_g == 0 or k == 0:
                continue
            # matched (one-to-one) queries -> excluded from the aux candidate pool
            matched = mask_weights_list[i] > 0                     # (Q,) bool
            unmatched = torch.nonzero(~matched, as_tuple=False).squeeze(1)  # (U,)
            if unmatched.numel() == 0:
                continue

            # GT masks at mask-pred resolution (nearest, like _get_targets_single)
            gt_dn = F.interpolate(gt_masks.float().unsqueeze(1), size=(h, w),
                                  mode='nearest').squeeze(1)        # (n_g, h, w)
            area_g = gt_dn.flatten(1).sum(1)                        # (n_g,)

            # per-GT difficulty weight (detached)
            w_g = self._chart_difficulty_weights(gt_masks)         # (n_g,)

            # affinity(query, gt) = soft mask mass of query inside GT / GT area
            # (the precheck's "present-but-weak response" probe). Selection only.
            with torch.no_grad():
                P = mask_preds[i, unmatched].sigmoid().flatten(1)  # (U, hw)
                G = gt_dn.flatten(1)                               # (n_g, hw)
                mass = P @ G.t()                                   # (U, n_g)
                affinity = mass / area_g.clamp_min(1.0)[None, :]   # (U, n_g)

            # greedy UNIQUE top-k, hardest GT first: each aux query serves one GT,
            # the hardest GTs get first pick of their best (weak-responding) queries.
            order = torch.argsort(w_g, descending=True).tolist()
            avail = torch.ones(unmatched.numel(), dtype=torch.bool,
                               device=cls_scores.device)
            for g in order:
                col = affinity[:, g].clone()
                col[~avail] = -1.0
                ncand = int(avail.sum().item())
                if ncand == 0:
                    break
                kk = min(k, ncand)
                sel = torch.topk(col, kk).indices                  # rows into `unmatched`
                avail[sel] = False
                q_idx = unmatched[sel]                             # (kk,) query ids
                lab = int(gt_labels[g].item())
                for qi in q_idx.tolist():
                    logp = F.log_softmax(cls_scores[i, qi], dim=-1)
                    ce_list.append(class_weight[lab] * (-logp[lab]))
                    wt_list.append(w_g[g])
                    p_rows.append(mask_preds[i, qi])               # (h, w) logits
                    t_rows.append(gt_dn[g])                        # (h, w) target

        if len(ce_list) == 0:
            z = cls_scores.sum() * 0.0
            zm = mask_preds.sum() * 0.0
            return z, zm, zm

        ce = torch.stack(ce_list)                                  # (N,)
        wt = torch.stack(wt_list).to(ce.dtype)                     # (N,)
        P = torch.stack(p_rows)                                    # (N, h, w) logits
        T = torch.stack(t_rows)                                    # (N, h, w) targets
        denom = wt.sum().clamp_min(_EPS)

        # aux cls: weighted mean soft/hard CE, scaled to main cls scale
        aux_cls = self.loss_cls.loss_weight * (wt * ce).sum() / denom

        # aux mask: per-query BCE (mean over pixels), weighted, main mask scale
        bce = F.binary_cross_entropy_with_logits(P, T, reduction='none').flatten(1).mean(1)  # (N,)
        aux_mask = self.loss_mask.loss_weight * (wt * bce).sum() / denom

        # aux dice: per-query naive Dice (eps=1, matching stock DiceLoss), weighted
        Ps = P.sigmoid().flatten(1)                                # (N, hw)
        Tf = T.flatten(1)                                          # (N, hw)
        inter = (Ps * Tf).sum(1)
        dice = 1.0 - (2.0 * inter + 1.0) / (Ps.sum(1) + Tf.sum(1) + 1.0)  # (N,)
        aux_dice = self.loss_dice.loss_weight * (wt * dice).sum() / denom

        return aux_cls, aux_mask, aux_dice

    # ---------------------------------------------------------------------- GATE-0
    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances,
                             batch_img_metas):
        # Stock (loss_cls, loss_mask, loss_dice) -- byte-exact via super().
        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        if self.aux_weight <= 0:
            return loss_cls, loss_mask, loss_dice   # GATE-0: identity, no RNG use

        # one-to-one matched sets (defines the unmatched aux pool). This 2nd
        # get_targets call runs ONLY when the mechanism is on, so it never
        # perturbs the aux_weight=0 byte-exact path.
        num_imgs = cls_scores.size(0)
        (_labels_list, _lw_list, _mt_list, mask_weights_list,
         _avg) = self.get_targets([cls_scores[i] for i in range(num_imgs)],
                                  [mask_preds[i] for i in range(num_imgs)],
                                  batch_gt_instances, batch_img_metas)
        aux_cls, aux_mask, aux_dice = self._chart_aux_loss(
            cls_scores, mask_preds, mask_weights_list, batch_gt_instances)
        return (loss_cls + self.aux_weight * aux_cls,
                loss_mask + self.aux_weight * aux_mask,
                loss_dice + self.aux_weight * aux_dice)
