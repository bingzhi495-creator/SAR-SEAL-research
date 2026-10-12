# -*- coding: utf-8 -*-
"""Gaussian-Wasserstein matching cost for Mask2Former (cross-domain novelty, workflow w7qfmdh0u).

WHY (diagnosis-driven): inshore small-ship failure = confident mis-localization. M2F's Hungarian matching
cost = cls + mask-focal + dice -- all IoU-like, PATHOLOGICAL for ~3.5px targets (a 1px shift collapses
IoU; gradient is near step/zero), so the matcher mis-assigns small-ship queries at the source. Tiny-object
detection (NWD arXiv:2110.13389 / ISPRS-JPRS 2022; GWD ICML2021; RFLA ECCV2022; SimD IROS2024) solves this
by replacing the overlap metric with a Gaussian-Wasserstein distance: smooth, bounded, everywhere-nonzero
gradient w.r.t. small shifts, and scale-invariant (small ships not drowned by large-mask dice). Hausdorff-DETR
(WACV2023) is precedent for swapping the DETR matching metric. NOVELTY GAP vs O2Former(arXiv:2506.11913,
uses NWD in the regression LOSS w/ a box branch [needs PDF verify]): we put it in the bipartite MATCHING COST.

Orthogonal to every washed mechanism (refine/maskiou/boundary = readout/ranking; this = matching/metric axis)
and to a1 (stride-4 = resolution axis). Default-off: weight=0 in the cost -> matcher byte-identical to stock.

Implementation note: M2F matches on POINT-SAMPLED masks (no spatial extent), so a standard match_cost can't
compute a centroid. WassMatchM2FHead._get_targets_single attaches the FULL spatial masks as `.gauss_masks`
to the instances before assign; GaussianWassersteinCost reads those. Standard costs still use point-sampled
`.masks` -> the stock matching is UNCHANGED, the Gaussian term is purely additive.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmcv.ops import point_sample

from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmdet.registry import MODELS, TASK_UTILS

_EPS = 1e-6


def _gauss_moments(masks: torch.Tensor):
    """masks: (N, h, w) non-negative weights -> mu (N,2 in [0,1] as [x,y]), cov (N,2,2)."""
    N, h, w = masks.shape
    ys = torch.linspace(0.0, 1.0, h, device=masks.device, dtype=masks.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=masks.device, dtype=masks.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (hw, 2)
    mw = masks.reshape(N, -1).clamp_min(0.0)                        # (N, hw)
    Z = mw.sum(1, keepdim=True).clamp_min(_EPS)                     # (N, 1)
    mu = (mw.unsqueeze(-1) * coords.unsqueeze(0)).sum(1) / Z        # (N, 2)
    d = coords.unsqueeze(0) - mu.unsqueeze(1)                       # (N, hw, 2)
    cov = torch.einsum('nk,nki,nkj->nij', mw, d, d) / Z.unsqueeze(-1)
    eye = torch.eye(2, device=masks.device, dtype=masks.dtype)
    return mu, cov + _EPS * eye


def _sqrtm_spd_2x2(C: torch.Tensor):
    """Closed-form sqrt of a batch of SPD 2x2: sqrt(C)=(C+sqrt(det)I)/sqrt(tr+2sqrt(det)). Differentiable."""
    a, b, c, d = C[..., 0, 0], C[..., 0, 1], C[..., 1, 0], C[..., 1, 1]
    det = (a * d - b * c).clamp_min(0.0)
    s = det.sqrt()
    t = (a + d + 2.0 * s).clamp_min(_EPS).sqrt()
    eye = torch.eye(2, device=C.device, dtype=C.dtype)
    return (C + s[..., None, None] * eye) / t[..., None, None]


def gaussian_nwd(mu_p, cov_p, mu_g, cov_g, C: float):
    """Pairwise NWD = exp(-sqrt(W2^2)/C) with
    W2^2 = ||mu_p-mu_g||^2 + ||sqrt(cov_p)-sqrt(cov_g)||_F^2. Returns (Q, G).

    NOTE (reviewer w13gekell, 2026-07-02): the covariance term ||sqrt(A)-sqrt(B)||_F^2 is the EXACT squared
    Bures/2-Wasserstein distance ONLY when A,B commute (co-diagonalizable), e.g. the axis-aligned diagonal-box
    covariance of the original NWD paper (arXiv:2110.13389). _gauss_moments produces FULL (generally non-diagonal)
    covariances from rotated/elongated ship masks, so this is a (one-sided upper-bound) APPROXIMATION of the true
    W2^2, with up to ~35-63% relative error for very elongated/differently-oriented pairs. The exact Bures term is
    Tr(A)+Tr(B)-2*Tr(sqrtm(sqrtm(A) B sqrtm(A))), obtainable cheaply via the same closed form (Tr(sqrtm(M))=
    sqrt(tr M + 2 sqrt(det M))); an `exact` variant (a2b) is the planned upgrade -- all reported a2/a1a2/combo
    numbers were produced with THIS approximate form."""
    sp = _sqrtm_spd_2x2(cov_p)                       # (Q,2,2)
    sg = _sqrtm_spd_2x2(cov_g)                       # (G,2,2)
    dmu = (mu_p[:, None, :] - mu_g[None, :, :]).pow(2).sum(-1)        # (Q,G)
    dcov = (sp[:, None] - sg[None]).pow(2).sum(dim=(-1, -2))          # (Q,G)
    w2 = (dmu + dcov).clamp_min(0.0).sqrt()
    return torch.exp(-w2 / max(C, _EPS))


def _trace_sqrtm_spd_2x2(M: torch.Tensor):
    """Tr(sqrtm(M)) for SPD 2x2 via the same closed form used in _sqrtm_spd_2x2:
    Tr(sqrtm(M)) = sqrt(tr(M) + 2*sqrt(det(M))). Exact for SPD 2x2 (Cayley-Hamilton)."""
    tr = M[..., 0, 0] + M[..., 1, 1]
    det = (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]).clamp_min(0.0)
    return (tr + 2.0 * det.sqrt()).clamp_min(0.0).sqrt()


def gaussian_nwd_exact(mu_p, cov_p, mu_g, cov_g, C: float):
    """a2b: EXACT Bures/2-Wasserstein NWD (review w13gekell fix for the non-commuting case).
    W2^2 = ||mu_p-mu_g||^2 + tr(A) + tr(B) - 2*Tr(sqrtm(sqrtm(A) B sqrtm(A))). Returns (Q, G).
    Cost: two extra batched 2x2 matmuls vs the approximate form; exact for rotated/elongated ships."""
    sp = _sqrtm_spd_2x2(cov_p)                                        # (Q,2,2)
    trA = cov_p[..., 0, 0] + cov_p[..., 1, 1]                         # (Q,)
    trB = cov_g[..., 0, 0] + cov_g[..., 1, 1]                         # (G,)
    M = sp[:, None] @ cov_g[None] @ sp[:, None]                       # (Q,G,2,2), SPD
    cross = _trace_sqrtm_spd_2x2(M)                                   # (Q,G)
    dmu = (mu_p[:, None, :] - mu_g[None, :, :]).pow(2).sum(-1)        # (Q,G)
    w2 = (dmu + trA[:, None] + trB[None, :] - 2.0 * cross).clamp_min(0.0).sqrt()
    return torch.exp(-w2 / max(C, _EPS))


@TASK_UTILS.register_module()
class GaussianWassersteinCost(BaseMatchCost):
    """Additive Gaussian-Wasserstein matching cost. Reads `.gauss_masks` (full spatial) attached by
    WassMatchM2FHead. cost = weight * (1 - NWD). weight=0 -> no effect (default-off).
    exact=False (default) = the approximate commuting-case form used by ALL reported a2/a1a2 results;
    exact=True (a2b) = the exact Bures term via gaussian_nwd_exact."""

    def __init__(self, weight: float = 1.0, C: float = 0.5, exact: bool = False):
        super().__init__(weight=weight)
        self.C = float(C)
        self.exact = bool(exact)

    def __call__(self, pred_instances: InstanceData, gt_instances: InstanceData,
                 img_meta=None, **kwargs):
        if self.weight == 0 or not hasattr(pred_instances, 'gauss_masks'):
            # safe no-op: a zero (Q,G) cost
            q = pred_instances.masks.shape[0]
            g = gt_instances.masks.shape[0]
            return pred_instances.masks.new_zeros((q, g))
        pm = pred_instances.gauss_masks.sigmoid()    # (Q, h, w)
        gm = gt_instances.gauss_masks.float()        # (G, h, w)
        if gm.shape[0] == 0:
            return pm.new_zeros((pm.shape[0], 0))
        mu_p, cov_p = _gauss_moments(pm)
        mu_g, cov_g = _gauss_moments(gm)
        fn = gaussian_nwd_exact if self.exact else gaussian_nwd
        nwd = fn(mu_p, cov_p, mu_g, cov_g, self.C)             # (Q, G)
        return (1.0 - nwd) * self.weight


@MODELS.register_module()
class WassMatchM2FHead(Mask2FormerHead):
    """Mask2FormerHead that attaches full spatial masks (`.gauss_masks`) to the matched instances so
    GaussianWassersteinCost can compute centroids/covariances. Stock point-sampled matching is otherwise
    UNCHANGED. With no GaussianWassersteinCost in the assigner (base configs), behaviour == stock head."""

    def _get_targets_single(self, cls_score, mask_pred, gt_instances, img_meta):
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)

        # full spatial masks for the Gaussian cost (gt downsampled to pred res, same coord frame)
        tgt_shape = mask_pred.shape[-2:]
        if num_gts > 0:
            gt_full = F.interpolate(gt_masks.unsqueeze(1).float(), tgt_shape,
                                    mode='nearest').squeeze(1)
        else:
            gt_full = gt_masks.new_zeros((0, *tgt_shape), dtype=torch.float32)

        sampled_gt_instances = InstanceData(labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(scores=cls_score, masks=mask_points_pred)
        sampled_pred_instances.gauss_masks = mask_pred          # (Q, h, w) logits
        sampled_gt_instances.gauss_masks = gt_full             # (G, h, w) {0,1}

        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances, img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result, pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0
        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)


@MODELS.register_module()
class WassLossM2FHead(Mask2FormerHead):
    """a2-loss CONTROL ARM (the GATE-0 the paper needs, review w13gekell): the SAME Gaussian-Wasserstein
    distance as a2, but placed as an AUXILIARY LOSS on matched (post-Hungarian) pairs -- matching stays
    100% stock (no GaussianWassersteinCost in the assigner). Purpose: isolate a2's claimed novelty
    ("Wasserstein in the MATCHING COST") from the occupied O2Former-style "Wasserstein/NWD in the LOSS"
    territory. If a2 (matching placement) > this arm on the matched protocol, the placement claim is
    directly evidenced; if this arm matches a2, the paper's novelty framing must be honestly rescoped.
    lambda_wass=0 -> byte-identical to stock Mask2FormerHead (same default-off contract as siblings).
    NOTE: GT downsample here uses mode='area' (review fix; this is a NEW arm, no seed-consistency issue)."""

    def __init__(self, *args, lambda_wass: float = 0.0, wass_C: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_wass = float(lambda_wass)
        self.wass_C = float(wass_C)

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances, batch_img_metas):
        loss_cls, loss_mask, loss_dice = super()._loss_by_feat_single(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        if self.lambda_wass > 0:
            n = cls_scores.size(0)
            (_, _, mtl, mwl, _) = self.get_targets(
                [cls_scores[i] for i in range(n)], [mask_preds[i] for i in range(n)],
                batch_gt_instances, batch_img_metas)
            mw = torch.stack(mwl, dim=0)                  # (B, Q) 1 = matched
            mt = torch.cat(mtl, dim=0)                    # (M, H, W) matched GT (aligned with mask_preds[mw>0])
            if mt.shape[0] > 0:
                mp = mask_preds[mw > 0]                   # (M, h, w) matched pred logits
                h, w = mp.shape[-2:]
                gt_s = F.interpolate(mt.unsqueeze(1).float(), (h, w), mode='area').squeeze(1)
                mu_p, cov_p = _gauss_moments(mp.sigmoid())
                mu_g, cov_g = _gauss_moments(gt_s)
                sp = _sqrtm_spd_2x2(cov_p)
                sg = _sqrtm_spd_2x2(cov_g)
                # per-PAIR (diagonal) NWD -- same math as the a2 cost, evaluated only on matched pairs
                w2 = ((mu_p - mu_g).pow(2).sum(-1)
                      + (sp - sg).pow(2).sum(dim=(-1, -2))).clamp_min(0.0).sqrt()
                nwd = torch.exp(-w2 / max(self.wass_C, _EPS))
                loss_mask = loss_mask + self.lambda_wass * (1.0 - nwd).mean()
        return loss_cls, loss_mask, loss_dice
