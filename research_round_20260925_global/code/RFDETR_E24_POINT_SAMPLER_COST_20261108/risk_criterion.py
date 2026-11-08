"""Isolated RF-DETR 1.12 probe adapter for same-risk pixel-center BCE."""
from contextlib import contextmanager
from unittest.mock import patch

from torch.nn import functional as F

from point_allocation import prepare_strata, sample_target_points_prepared


@contextmanager
def pixel_center_bce(criterion, native_api, point_sample, arm, generator):
    """Replace only the one-image matched mask BCE. Keep stock Dice and matcher."""
    if arm not in ("B","D"):
        raise ValueError("Choose uniform B or corrected stratified D")
    stock_sampler=native_api.get_uncertain_point_coords_with_randomness
    stock_loss=criterion.loss_masks
    frame=None
    cached_masks=None
    strata=None

    def observe_sampler(logits, uncertainty, points, oversample, fraction):
        nonlocal frame
        if frame is not None:
            raise RuntimeError("Expected one stock Dice point draw per loss")
        frame=(logits,points)
        return stock_sampler(logits,uncertainty,points,oversample,fraction)

    def loss_masks(outputs,targets,matches,num_boxes):
        nonlocal frame,cached_masks,strata
        if len(targets)!=1 or len(matches)!=1 or "valid" in targets[0]:
            raise ValueError("Expected native one-image segmentation microbatch")
        frame=None
        stock=stock_loss(outputs,targets,matches,num_boxes)
        target_indices=matches[0][1]
        if not len(target_indices):
            return stock
        if frame is None:
            raise RuntimeError("Stock mask sampler did not expose matched logits")
        coarse,points=frame
        frame=None
        masks=targets[0]["masks"]
        if masks is not cached_masks:
            strata=prepare_strata(masks) if arm=="D" else None
            cached_masks=masks
        coords,weights,flat=sample_target_points_prepared(masks,target_indices,points,
            arm=arm,generator=generator,strata=strata)
        predicted=point_sample(coarse,coords,align_corners=False).squeeze(1)
        target=masks.flatten(1)[target_indices.to(masks.device)[:,None],flat].to(predicted.dtype)
        bce=F.binary_cross_entropy_with_logits(predicted,target,reduction="none")
        stock["loss_mask_ce"]=(bce*weights).mean(1).sum()/num_boxes
        return stock

    with patch.object(native_api,"get_uncertain_point_coords_with_randomness",observe_sampler), \
         patch.object(criterion,"loss_masks",loss_masks):
        yield
