# -*- coding: utf-8 -*-
"""MIAC_MSIoUM2FHead: COMBINE the two mask-quality mechanisms in one head.

Two independently-verified mechanisms attack the SAME diagnosis (the COCO-init
cls score is IoU-blind, rho(cls, true-mask-IoU) = -0.213; oracle rescore by the
true mask-IoU gives +7.66 AP inshore) from two different angles:

  MIAC  (carrier/miac_head.py, MIACM2FHead)
        TRAINING-time: folds mask-IoU into the CLASSIFICATION TARGET. Each matched
        query's ship-class soft target becomes (1-w)+w*u^gamma (u = pred-mask-vs-GT
        mask-IoU), calibrating cls-prob to localization quality at the SOURCE.
        Overrides ONLY _loss_by_feat_single (recomputes loss_cls). iou_cls_weight=0
        -> loss_cls byte-identical to stock (GATE-0).

  MSIoU (carrier/maskiou_head.py, MSIoUM2FHead + MSIoUFusionHead)
        A per-query mask-IoU PREDICTOR head. TRAINING adds an IoU-regression loss
        (maskiou_weight>0) on matched queries (rich 11-stat V2 regressor when
        maskiou_v2=True); INFERENCE threads the predicted IoU to MSIoUFusionHead
        which rescores det = cls * mask_score^(1-lambda) * pred_iou^lambda.
        maskiou_weight=0 -> no IoU loss; infer_iou_lambda=0 -> scoring byte-identical
        to stock (GATE-0).

MRO composition (NO copy-pasted arithmetic -- both parents keep their own
byte-exact zero-weight paths):

    MIAC_MSIoUM2FHead -> MIACM2FHead -> MSIoUM2FHead -> SealM2FHead -> Mask2FormerHead

  loss()                = MSIoUM2FHead.loss        (adds loss_maskiou, then loss_by_feat)
  forward()             = MSIoUM2FHead.forward      (refine chain + stashes pred-IoU logit)
  predict()             = MSIoUM2FHead.predict      (threads pred_iou -> fusion head)
  _loss_by_feat_single()= MIACM2FHead._loss...      (soft-cls); its super() chains through
                          SealM2FHead._loss... (boundary, inert at boundary_weight=0) ->
                          Mask2FormerHead (stock loss_cls/mask/dice).

So the three knobs are simultaneously live and independent:
  iou_cls_weight   (MIAC soft cls target, in _loss_by_feat_single)
  maskiou_weight + maskiou_v2  (MSIoU predictor training loss, in loss())
  infer_iou_lambda (MSIoU inference rescore, in MSIoUFusionHead.instance_postprocess)

GATE-0 (byte-exactness): with iou_cls_weight=0 AND maskiou_weight=0 AND
infer_iou_lambda=0 the total training loss and inference scoring are
BYTE-IDENTICAL to stock Mask2FormerHead + stock MaskFormerFusionHead --
  * MIAC returns super()'s loss_cls untouched when iou_cls_weight<=0;
  * MSIoUM2FHead.loss adds loss_maskiou ONLY when maskiou_weight>0;
  * MSIoUFusionHead rescores ONLY when infer_iou_lambda>0.
The maskiou_head module IS instantiated (an extra unused MaskIoU regressor's
params exist), but it is never READ at zero-weights, so it perturbs no loss/score.

FLAG-DEFAULT PITFALL (combo_head.py reviewer w13gekell, 2026-07-02): the MSIoU
branch extends SealM2FHead whose refine defaults are ON (refine_scale=2,
learn_refine=True) and which owns a boundary loss. Those are SIBLING mechanisms
that must NOT contaminate the MIAC+MSIoU combo -- the config MUST pass
refine_scale=1, learn_refine=False, boundary_weight=0.0 explicitly. (a2/wass and
cbam are not in this MRO at all, so they are structurally inert here.)
"""
from __future__ import annotations

from mmdet.registry import MODELS

from .miac_head import MIACM2FHead
from .maskiou_head import MSIoUM2FHead, MSIoUFusionHead  # noqa: F401 (re-export fusion head)


@MODELS.register_module()
class MIAC_MSIoUM2FHead(MIACM2FHead, MSIoUM2FHead):
    """MIAC soft-cls (training) + MSIoU predictor (training) + MSIoU rescore
    (inference) in ONE head, purely via MRO -- no new arithmetic.

    All three knobs default to their inert values (iou_cls_weight=0.0 from
    MIACM2FHead, maskiou_weight=0.0 from MSIoUM2FHead) so at defaults the head is
    byte-identical to stock Mask2FormerHead. The inference rescore lives in the
    paired MSIoUFusionHead (infer_iou_lambda, default 0.0 = stock).

    __init__ is inherited: MIACM2FHead.__init__ pops iou_cls_weight/iou_cls_gamma
    then super().__init__ (=MSIoUM2FHead) pops maskiou_weight/maskiou_v2/... then
    SealM2FHead pops refine_scale/learn_refine/boundary_weight/... All forward via
    *args/**kwargs, so a config may set every knob in one panoptic_head dict.

    NOTE: keep the Seal sibling mechanisms OFF in the config
    (refine_scale=1, learn_refine=False, boundary_weight=0.0) -- see module docstring.
    """
    pass
