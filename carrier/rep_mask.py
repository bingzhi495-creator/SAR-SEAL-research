# -*- coding: utf-8 -*-
"""RepMask: inter-instance mask-repulsion loss for Mask2Former (a3, crowd-separation axis).

WHY (diagnosis + measured pixels): HRSID inshore = median 4 ships/img, 67% multi-ship, ~27% nearest-neighbour
bbox overlap; dense bright harbour clutter (sar-pixel-physics-analysis). a2 fixes WHICH GT each query matches;
nothing in M2F stops two matched queries' masks from leaking onto each other / a neighbour ship in clutter.
RepMask penalizes probability mass a matched query leaks onto its NEAREST non-owner GT, gated to fire only
between spatially-close GTs (auto-inert offshore = 1 ship/img), and only the EXCESS over the true GT-GT
overlap (so genuinely docked side-by-side ships are not eroded -> protects recall/AP50).

Port of Repulsion Loss RepGT (Wang et al. CVPR2018 arXiv:1711.07752, box domain) to the MASK domain of a
query-based set-predictor, with excess-over-GT gating. Orthogonal to a1 (resolution) and a2 (matching metric):
a1 changes WHERE a query attends, a2 changes WHICH GT it pairs to, a3 acts AFTER the match on the supervised
mask loss with a pairwise "stay off your neighbour" term. lambda_repgt=0 -> byte-identical to the base head
(same default-off safety contract as GaussianWassersteinCost weight=0). Zero new parameters.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.registry import MODELS
from carrier.wass_match import WassMatchM2FHead, _gauss_moments

_EPS = 1e-6


def smooth_ln(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Repulsion-Loss Smooth_ln penalty (CVPR2018). x in [0,1).  -ln(1-x) for x<=sigma ;
    (x-sigma)/(1-sigma) - ln(1-sigma) for x>sigma. C0-continuous at x=sigma, monotone increasing."""
    s = float(sigma)
    lo = -torch.log((1.0 - x).clamp_min(_EPS))
    hi = (x - s) / (1.0 - s) - math.log(1.0 - s)
    return torch.where(x <= s, lo, hi)


def _centroids_norm(masks: torch.Tensor) -> torch.Tensor:
    """Normalized [x,y] centroids in [0,1] for masks (N,H,W), reusing a2's _gauss_moments geometry."""
    if masks.shape[0] == 0:
        return masks.new_zeros((0, 2))
    mu, _ = _gauss_moments(masks.float())
    return mu  # (N,2) in [0,1]


def repgt_loss(mask_preds, mask_targets_list, mask_weights_list, batch_gt_instances,
               rep_tau: float = 0.15, rep_sigma: float = 0.5):
    """Inter-instance RepGT loss (per image, so neighbours are within the same image).

    mask_preds: (B,Q,h,w) logits @ mask-feature (1/4) res.
    mask_targets_list[i]: (n_i,H,W) owner GT masks per matched query (aligned with mask_preds[i][mw>0]).
    mask_weights_list[i]: (Q,) 1 at matched queries.
    batch_gt_instances[i].masks: (G_i,H,W) ALL GT masks (tensor, preprocessed).
    Returns a scalar (mean Smooth_ln of excess-IoG over qualifying matched queries; 0 if none qualify)."""
    h, w = mask_preds.shape[-2:]
    total = mask_preds.new_zeros(())
    cnt = 0
    for i in range(len(mask_targets_list)):
        mw = mask_weights_list[i]
        sel = mw > 0
        if int(sel.sum()) == 0:
            continue
        G = batch_gt_instances[i].masks.float()          # (G,H,W) all GT
        if G.shape[0] < 2:
            continue                                     # need >=2 GT for a neighbour to exist
        mp = mask_preds[i][sel]                          # (n,h,w) matched logits
        mt = mask_targets_list[i].float()                # (n,H,W) owner GT (aligned with mp)
        mu_o = _centroids_norm(mt)                       # (n,2)
        mu_a = _centroids_norm(G)                        # (G,2)
        d = torch.cdist(mu_o, mu_a)                      # (n,G) normalized-coord centroid distance
        d = d + (d < 1e-4).to(d.dtype) * 1e9             # exclude self (owner == its own GT row, dist~0)
        nn_d, nn_idx = d.min(dim=1)                      # (n,) nearest non-owner GT per matched query
        qual = nn_d < rep_tau                            # spatial-proximity gate (auto-False offshore)
        if int(qual.sum()) == 0:
            continue
        g_rep = G[nn_idx]                                # (n,H,W) neighbour GT mask
        g_rep = F.interpolate(g_rep.unsqueeze(1), (h, w), mode='nearest').squeeze(1)  # -> pred res
        mt_d = F.interpolate(mt.unsqueeze(1), (h, w), mode='nearest').squeeze(1)
        p = mp.sigmoid()
        denom = g_rep.flatten(1).sum(1).clamp_min(_EPS)  # (n,) neighbour area
        iog = (p * g_rep).flatten(1).sum(1) / denom      # frac of neighbour the query claims
        iog_gt = (mt_d * g_rep).flatten(1).sum(1) / denom  # true GT-GT overlap (protect docked ships)
        r = (iog - iog_gt).clamp(min=0.0, max=1.0 - _EPS)  # EXCESS leakage only
        sl = smooth_ln(r[qual], rep_sigma)
        total = total + sl.sum()
        cnt += int(qual.sum())
    return total / max(cnt, 1)


class _RepMaskMixin:
    """Adds the RepGT term to _loss_by_feat_single. Host must supply get_targets + Mask2Former
    _loss_by_feat_single via super(). lambda_repgt=0 -> the extra path is skipped (exact no-op)."""

    def _set_rep(self, lambda_repgt, rep_tau, rep_sigma):
        self.lambda_repgt = float(lambda_repgt)
        self.rep_tau = float(rep_tau)
        self.rep_sigma = float(rep_sigma)

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances, batch_img_metas):
        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        if getattr(self, 'lambda_repgt', 0.0) > 0:
            n = cls_scores.size(0)
            (_, _, mtl, mwl, _) = self.get_targets(
                [cls_scores[i] for i in range(n)], [mask_preds[i] for i in range(n)],
                batch_gt_instances, batch_img_metas)
            lrep = repgt_loss(mask_preds, mtl, mwl, batch_gt_instances,
                              self.rep_tau, self.rep_sigma)
            loss_mask = loss_mask + self.lambda_repgt * lrep
        return loss_cls, loss_mask, loss_dice


@MODELS.register_module()
class RepMaskM2FHead(_RepMaskMixin, Mask2FormerHead):
    """Stock Mask2Former matching + RepGT loss (a3-alone; matches the a0 baseline head exactly when
    lambda_repgt=0). This is the GATE-0 control's base."""

    def __init__(self, *args, lambda_repgt: float = 0.0, rep_tau: float = 0.15,
                 rep_sigma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_rep(lambda_repgt, rep_tau, rep_sigma)


@MODELS.register_module()
class RepMaskWassM2FHead(_RepMaskMixin, WassMatchM2FHead):
    """a2 (Gaussian-Wasserstein MATCHING) + a3 (RepGT LOSS) stacked -- the 异轴叠加 a2+a3 arm."""

    def __init__(self, *args, lambda_repgt: float = 0.0, rep_tau: float = 0.15,
                 rep_sigma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_rep(lambda_repgt, rep_tau, rep_sigma)
