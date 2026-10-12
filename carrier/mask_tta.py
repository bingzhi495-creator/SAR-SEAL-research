# -*- coding: utf-8 -*-
"""Test-time augmentation for INSTANCE SEGMENTATION with Mask2Former.

WHY THIS EXISTS: mmdet's built-in ``DetTTAModel`` explicitly refuses masks --
its ``_merge_single_sample`` asserts ``pred_instances.masks is None`` with the
message "TTA of instance segmentation does not support now." Its docstring says
only bboxes+scores under flip / multi-scale are merged. So for Mask2Former
instance segmentation we must (a) flip each predicted mask back to the original
orientation, and (b) merge instances across augmented views WITHOUT discarding
the masks.

KEY GEOMETRY FACT (source-verified against maskformer.py + MSIoUFusionHead):
the fusion head's ``predict(..., rescale=True)`` interpolates every mask back to
``ori_shape`` before returning. test.py runs inference with rescale=True. So for
every augmented view -- flip AND multi-scale -- ``pred_instances.masks`` is a
bool tensor of shape (N, ori_H, ori_W), i.e. all views share the SAME spatial
size = original image size. Multi-scale therefore needs no extra mask resizing;
we only have to undo the flip (a pure axis reversal), then pool + NMS.

MERGE STRATEGY: pool all (mask, score, label) triples across every augmented
view, derive tight bboxes from the (flip-corrected) masks via ``mask2bbox``, run
class-wise NMS on those bboxes reusing DetTTAModel's nms cfg, and KEEP the masks
at the surviving indices (do not drop them). Truncate to max_per_img by score.

infer_iou_lambda / MaskIoU scoring is orthogonal: it already ran inside each
view's fusion head, so ``pred_instances.scores`` already carries the rescored
value. TTA just merges those scores.
"""
from __future__ import annotations

from typing import List

import torch
from mmcv.ops import batched_nms
from mmengine.registry import MODELS
from mmengine.structures import InstanceData

from mmdet.models.layers import mask_matrix_nms
from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.models.test_time_augs.det_tta import DetTTAModel
from mmdet.models.utils import empty_instances
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask import mask2bbox


@MODELS.register_module()
class MaskTTAModel(DetTTAModel):
    """Merge augmented INSTANCE SEGMENTATION results (masks + scores + labels).

    Subclasses ``DetTTAModel`` only to inherit ``__init__`` / ``merge_preds``
    plumbing and the ``tta_cfg`` (nms + max_per_img). ``_merge_single_sample``
    is fully overridden to handle masks.
    """

    @staticmethod
    def _flip_mask_back(masks: torch.Tensor, flip: bool,
                        flip_direction: str) -> torch.Tensor:
        """Undo the test-time flip on a (N, H, W) bool/uint mask tensor.

        A flip is its own inverse, so we just apply the same axis reversal:
        horizontal -> reverse W (last dim); vertical -> reverse H (dim -2);
        diagonal -> reverse both.
        """
        if not flip or masks.numel() == 0:
            return masks
        if flip_direction == 'horizontal':
            return masks.flip(dims=[-1])
        elif flip_direction == 'vertical':
            return masks.flip(dims=[-2])
        elif flip_direction == 'diagonal':
            return masks.flip(dims=[-2, -1])
        else:
            raise ValueError(f'Unsupported flip_direction: {flip_direction!r}')

    def _merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge the different augmented views of ONE image into one prediction.

        Args:
            data_samples: predictions for the several augmented views of a
                single image. Each carries ``pred_instances`` with masks,
                scores, labels (and bboxes, which we recompute from the
                flip-corrected masks for consistency).
        """
        aug_masks = []
        aug_scores = []
        aug_labels = []
        # F4: optional FLOAT sigmoid maps exported by MSIoUFusionHead when
        # test_cfg.export_mask_probs=True (row-aligned 1:1 with masks). Flipped
        # back EXACTLY like the masks. Only vote_mask='prob' consumes them; all
        # other merge paths ignore this list, so collecting it is output-neutral.
        aug_probs = []
        # R12: optional per-instance (K, 5) depth-consistency stats exported by
        # MSIoUFusionHead when model.test_cfg.export_depth_stats=True
        # ([iou_L7L8, iou_L8L9, score_L7, score_L8, score_L9], row-aligned 1:1
        # with masks). Pure per-instance SCALARS -> no flip correction needed.
        # Only the vote branch consumes them (appended to vote_stats), so
        # collecting is output-neutral for every other path.
        aug_depth = []

        for data_sample in data_samples:
            inst = data_sample.pred_instances
            masks = inst.get('masks', None)
            # If a view produced no masks, this is not an instance-seg model.
            assert masks is not None, (
                'MaskTTAModel expects pred_instances.masks; got None. '
                'Use this TTA model only with instance segmentation.')
            meta = data_sample.metainfo
            flip = meta.get('flip', False)
            flip_direction = meta.get('flip_direction', None)
            masks = self._flip_mask_back(masks, flip, flip_direction)
            # DTYPE NORMALISATION (added 2026-08-02, SOLOv2 carrier 6/6).
            # torch.cat TYPE-PROMOTES across views, so one badly-typed view
            # poisons the whole pooled tensor. SOLOv2's early-exit path
            # (mmdet 3.3.0 solov2_head.py::_predict_by_feat_single ->
            # empty_results) returns FLOAT placeholders --
            #   masks  = cls_scores.new_zeros(0, *ori_shape)   # float32, not bool
            #   labels = cls_scores.new_ones(0)                # float32, not int64
            # -- so a single empty view (very common for the weakest carrier)
            # made merged_labels float32; the fused labels stayed float32 and
            # CocoMetric.results2json died with
            #   TypeError: list indices must be integers or slices, not
            #   numpy.float32   (self.cat_ids[label])
            # after a full 369-image inference pass. Casting here fixes it for
            # ANY carrier with the same habit, which is the right level: this
            # wrapper's contract is "consumes only pred_instances masks/scores/
            # labels", so it must own their dtypes.
            # GATE-0 / OUTPUT-NEUTRAL for the five carriers already run:
            # every mmdet mask head emits bool masks (`mask_pred >= thr`) and
            # int64 labels on its non-empty path, so .bool()/.long() are
            # no-ops there -- byte-identical results, no re-run needed.
            aug_masks.append(masks.bool())
            aug_scores.append(inst.scores)
            aug_labels.append(inst.labels.long())
            probs = inst.get('mask_probs', None)
            if probs is not None:
                probs = self._flip_mask_back(probs, flip, flip_direction)
            aug_probs.append(probs)
            aug_depth.append(inst.get('depth_stats', None))

        merged_masks = torch.cat(aug_masks, dim=0)      # (K, H, W) bool
        merged_scores = torch.cat(aug_scores, dim=0)    # (K,)
        merged_labels = torch.cat(aug_labels, dim=0)    # (K,)
        # (K, H, W) fp16 or None; require EVERY view to carry probs so row
        # indices stay aligned with merged_masks (a partial export would
        # misalign the members gather -> safer to fall back to seed wholesale).
        merged_probs = (torch.cat(aug_probs, dim=0)
                        if aug_probs and all(p is not None for p in aug_probs)
                        else None)
        # (K, 5) or None; require EVERY view to carry depth_stats so row
        # indices stay aligned with merged_masks (same policy as merged_probs).
        merged_depth = (torch.cat(aug_depth, dim=0)
                        if aug_depth and all(d is not None for d in aug_depth)
                        else None)
        # Same-pool audit: disabled by default. Capture candidates and the
        # original GPU cluster decisions only when an explicit trace directory
        # is supplied; no scoring or mask operation changes on the normal path.
        trace_dir = self.tta_cfg.get('trace_dir', None)
        if trace_dir and self.tta_cfg.get('merge', 'bbox_nms') != 'vote':
            raise ValueError('trace_dir supports only merge=vote')

        det_results = data_samples[0]

        # SCIC: optional per-cluster CONSENSUS features for the vote branch,
        # exported 1:1 with the fused instances behind
        # test_cfg flag ``export_vote_stats`` (default False -> GATE-0:
        # byte-identical outputs, no extra attribute). Feature order (6):
        #   [support_count, n_views_present, mean_pair_iou, min_pair_iou,
        #    seed_score, score_spread]
        # support_count      : # pooled instances in the cluster
        # n_views_present    : # distinct TTA views contributing a member
        # mean/min_pair_iou  : mean/min OFF-DIAGONAL pairwise mask-IoU among
        #                      members (singleton -> 0.0 = no corroboration)
        # seed_score         : score of the top-score member (the seed)
        # score_spread       : max - min member score
        # Only merge='vote' computes them (the final stack); 'matrix'/'bbox_nms'
        # ignore the flag.
        # R12: when BOTH flags are on (export_vote_stats here AND
        # model.test_cfg.export_depth_stats upstream, i.e. every view carries
        # per-instance depth_stats), 2 depth features are APPENDED per fused
        # instance -> vote_stats becomes (C, 8):
        #   [.. 6 consensus .., seed_depth_iou_mean, seed_score_traj]
        # seed_depth_iou_mean = (iou_L7L8 + iou_L8L9) / 2 of the SEED member
        # seed_score_traj     = score_L9 - score_L7 of the SEED member
        # The fused instance INHERITS the depth stats of its SEED view
        # instance (the top-score cluster member, merged index `idx` below --
        # the same member whose mask vote_mask='seed' keeps), because the
        # fused mask/score are seed-anchored. Backward compat: depth absent
        # on ANY view -> (C, 6) exactly as before.
        export_stats = bool(self.tta_cfg.get('export_vote_stats', False))
        with_depth = export_stats and merged_depth is not None
        n_stat_cols = 8 if with_depth else 6

        if merged_masks.shape[0] == 0:
            # nothing predicted in any view -> return an empty first sample
            results = InstanceData()
            results.masks = merged_masks
            results.scores = merged_scores
            results.labels = merged_labels
            results.bboxes = merged_masks.new_zeros((0, 4), dtype=torch.float32)
            if export_stats:
                results.vote_stats = merged_masks.new_zeros(
                    (0, n_stat_cols), dtype=torch.float32)
            det_results.pred_instances = results
            if trace_dir:
                from carrier.same_pool_trace import write_vote_trace
                write_vote_trace(trace_dir, data_samples, aug_masks,
                                 merged_scores, merged_labels, [], [], results,
                                 self.tta_cfg, merged_probs is not None)
            return det_results

        # Which cross-view merge to run. Default 'bbox_nms' reproduces the
        # historical behaviour byte-for-byte; 'matrix' switches to mask-IoU
        # Matrix-NMS (F3), which decays by TRUE mask-IoU so two adjacent
        # parallel hulls (high bbox-IoU, low mask-IoU) both survive.
        merge = self.tta_cfg.get('merge', 'bbox_nms')

        if merge == 'matrix':
            # Mask-IoU Matrix-NMS: decays scores by true mask-IoU instead of
            # deleting bbox-IoU neighbours, so a low-mask-IoU parallel-hull
            # neighbour keeps a high score. CRITICAL for TTA merge: with N views
            # each object has ~N near-identical masks (mask-IoU~1) -> a strong
            # decay (small sigma) plus a positive filter_thr must REMOVE those
            # decayed duplicates, else COCO AP collapses (filter_thr=-1 kept them
            # at ~0.55 and precision died: inshore 0.041 vs 0.371). Config-driven.
            scores2, labels2, masks2, keep = mask_matrix_nms(
                merged_masks.bool(),
                merged_labels,
                merged_scores,
                kernel=self.tta_cfg.get('matrix_kernel', 'gaussian'),
                sigma=self.tta_cfg.get('matrix_sigma', 0.5),
                max_num=self.tta_cfg.max_per_img,
                filter_thr=self.tta_cfg.get('matrix_filter_thr', 0.15))
            results = InstanceData()
            results.masks = masks2
            # Use the DECAYED scores returned by matrix-nms (do NOT re-read
            # merged_scores) so the mask-IoU decay actually takes effect.
            results.scores = scores2
            results.labels = labels2
            if masks2.shape[0] == 0:
                results.bboxes = masks2.new_zeros((0, 4), dtype=torch.float32)
            else:
                results.bboxes = mask2bbox(masks2.bool())
            det_results.pred_instances = results
            return det_results

        if merge == 'vote':
            # CONSENSUS VOTING (F2): the correct paradigm for TTA merge. Cluster
            # the N views' masks that belong to the SAME object (greedy, within a
            # class, by TRUE mask-IoU), then emit ONE fused mask per cluster.
            # Unlike SOLO-style score-decay (merge='matrix') which leaves ~N
            # near-identical duplicates at moderate scores, voting STRUCTURALLY
            # dedups; unlike bbox-NMS it keeps a low-mask-IoU parallel-hull
            # neighbour as its OWN cluster instead of deleting it.
            vote_iou = self.tta_cfg.get('vote_iou', 0.5)
            vote_frac = self.tta_cfg.get('vote_frac', 0.5)
            vote_gamma = self.tta_cfg.get('vote_gamma', 0.0)
            vote_mask = self.tta_cfg.get('vote_mask', 'majority')
            # 'adaptive' only: majority-denoise a cluster iff the seed mask is
            # SMALL (area < vote_area_thr) AND has enough views to average away
            # speckle (cluster_size >= vote_min_support); otherwise keep the seed
            # mask sharp. Defaults 1024 px (=32x32) and 2 views. Ignored unless
            # vote_mask=='adaptive', so 'seed'/'majority' stay byte-identical.
            vote_area_thr = self.tta_cfg.get('vote_area_thr', 1024)
            vote_min_support = self.tta_cfg.get('vote_min_support', 2)
            # 'consensus' only: denoise a cluster iff its view-AGREEMENT is
            # MODERATE (min_support <= size <= max_support); high-agreement
            # clusters (a clean object every view found) keep the sharp seed
            # so majority does not erode a long clean boundary.
            vote_max_support = self.tta_cfg.get('vote_max_support', 5)
            # 'prob' only (F4): threshold on the MEAN of member FLOAT sigmoid
            # maps (0.5 = ensemble MAP boundary), and the high-agreement policy
            # (size > max_support): 'mean' fuses by prob-mean (sub-pixel
            # averaging should HELP a clean boundary, unlike binary majority),
            # 'seed' mirrors 'consensus' and keeps the sharp seed. Sweep knob.
            prob_thr = self.tta_cfg.get('prob_thr', 0.5)
            prob_high_agreement = self.tta_cfg.get('prob_high_agreement',
                                                   'mean')
            n_views = len(data_samples)
            K = merged_masks.shape[0]

            # Pairwise TRUE mask-IoU via flattened bool masks (matmul intersect).
            flat = merged_masks.bool().reshape(K, -1).float()   # (K, HW)
            area = flat.sum(dim=1)                              # (K,)
            inter = flat @ flat.t()                            # (K, K)
            union = area[:, None] + area[None, :] - inter
            iou = torch.where(union > 0, inter / union,
                              torch.zeros_like(inter))

            # SCIC export: view id per pooled instance (which TTA view it came
            # from), needed for n_views_present. Built ONLY under the flag so
            # the default path stays untouched.
            if export_stats:
                view_ids = torch.cat([
                    torch.full((m.shape[0],), vi, dtype=torch.long,
                               device=merged_masks.device)
                    for vi, m in enumerate(aug_masks)]) if K > 0 else \
                    torch.zeros(0, dtype=torch.long,
                                device=merged_masks.device)
                stats_rows = []

            # Greedy agglomeration around high-score seeds, WITHIN each label.
            order = torch.argsort(merged_scores, descending=True)
            used = torch.zeros(K, dtype=torch.bool, device=merged_masks.device)
            fused_masks, fused_scores, fused_labels = [], [], []
            trace_clusters = [] if trace_dir else None
            for idx in order.tolist():
                if used[idx]:
                    continue
                same = merged_labels == merged_labels[idx]
                members = (~used) & same & (iou[idx] >= vote_iou)
                members[idx] = True  # seed always joins its own cluster
                used |= members
                cluster_size = int(members.sum())
                if trace_clusters is not None:
                    trace_clusters.append(dict(
                        seed_id=int(idx),
                        member_ids=torch.nonzero(members).flatten().tolist(),
                        cluster_size=cluster_size))
                if export_stats:
                    m_idx = torch.nonzero(members).flatten()
                    if cluster_size > 1:
                        sub = iou[m_idx][:, m_idx]
                        off = ~torch.eye(cluster_size, dtype=torch.bool,
                                         device=sub.device)
                        pair = sub[off]
                        mean_pair_iou = float(pair.mean())
                        min_pair_iou = float(pair.min())
                    else:
                        mean_pair_iou = 0.0
                        min_pair_iou = 0.0
                    mscores = merged_scores[members]
                    row = [
                        float(cluster_size),
                        float(torch.unique(view_ids[m_idx]).numel()),
                        mean_pair_iou,
                        min_pair_iou,
                        float(merged_scores[idx]),
                        float(mscores.max() - mscores.min()),
                    ]
                    if with_depth:
                        # seed inheritance: depth row of the SEED member (idx)
                        d = merged_depth[idx]
                        row.append(float((d[0] + d[1]) * 0.5))  # depth_iou_mean
                        row.append(float(d[4] - d[2]))          # score_traj
                    stats_rows.append(row)
                # fused mask: 'seed' keeps the top-score member's mask verbatim
                # (no pixel averaging -> no offshore boundary erosion, while the
                # cluster still dedups views and keeps low-IoU neighbours apart);
                # 'majority' (default) pixelwise-votes; 'adaptive' picks per
                # object (majority for small+well-supported, seed otherwise).
                # Score = support-scaled max.
                if vote_mask == 'seed':
                    fused = merged_masks[idx].bool()
                elif vote_mask == 'adaptive':
                    # AREA-ADAPTIVE fusion: near-shore ships are small + buried
                    # in speckle -> majority-mean denoises; offshore ships are
                    # large + clean -> averaging misaligned views erodes the
                    # boundary, so keep the single seed mask. Seed mask AREA is
                    # the per-object regime proxy; only denoise when there are
                    # also enough views to average (cluster_size >= min_support).
                    a = int(merged_masks[idx].bool().sum())
                    if a < vote_area_thr and cluster_size >= vote_min_support:
                        member_masks = merged_masks[members].bool().float()
                        fused = member_masks.mean(dim=0) >= vote_frac
                    else:
                        fused = merged_masks[idx].bool()
                elif vote_mask == 'consensus':
                    # AGREEMENT-band gate: denoise ONLY moderate-agreement
                    # clusters (min_support <= size <= max_support = ambiguous,
                    # high-variance, near-shore-like); keep SEED for HIGH
                    # agreement (size > max_support: a clean object every view
                    # found -> majority erodes its long boundary) and singletons.
                    # Uses the ensemble's own view-agreement as the per-object
                    # variance/regime signal (no area proxy). offshore erosion
                    # (area-adaptive 0.560) is the failure this targets.
                    if vote_min_support <= cluster_size <= vote_max_support:
                        member_masks = merged_masks[members].bool().float()
                        fused = member_masks.mean(dim=0) >= vote_frac
                    else:
                        fused = merged_masks[idx].bool()
                elif vote_mask == 'prob':
                    # SOFT-PROBABILITY fusion (F4): cluster EXACTLY as the
                    # binary modes above (greedy BINARY mask-IoU, same band gate
                    # as 'consensus'), but fuse by averaging the members' FLOAT
                    # sigmoid maps and thresholding at prob_thr. Binary majority
                    # erodes boundaries because {0,1} masks from sub-pixel
                    # misaligned views vote boundary pixels OUT (offshore
                    # 0.559); the prob-mean keeps sub-threshold evidence
                    # (0.45+0.45+0.9 -> 0.6 >= 0.5 keeps the pixel) so the
                    # fused boundary moves toward the ensemble MAP estimate
                    # instead of shrinking. Singletons / size < min_support
                    # keep the seed (nothing to average); size > max_support
                    # follows prob_high_agreement ('mean' default | 'seed').
                    # Probs missing (export flag off) -> seed, safe fallback.
                    if merged_probs is None:
                        fused = merged_masks[idx].bool()
                    elif (vote_min_support <= cluster_size <= vote_max_support
                          or (cluster_size > vote_max_support
                              and prob_high_agreement == 'mean')):
                        member_probs = merged_probs[members].float()
                        fused = member_probs.mean(dim=0) >= prob_thr
                    else:
                        fused = merged_masks[idx].bool()
                else:
                    member_masks = merged_masks[members].bool().float()
                    fused = member_masks.mean(dim=0) >= vote_frac
                fused_masks.append(fused)
                support = (cluster_size / n_views) ** vote_gamma
                fused_scores.append(merged_scores[members].max() * support)
                fused_labels.append(merged_labels[idx])

            fused_masks = torch.stack(fused_masks, dim=0)          # (C, H, W)
            fused_scores = torch.stack(fused_scores)               # (C,)
            fused_labels = torch.stack(fused_labels)               # (C,)
            keep = torch.argsort(fused_scores,
                                 descending=True)[:self.tta_cfg.max_per_img]
            fused_masks = fused_masks[keep]
            results = InstanceData()
            results.masks = fused_masks
            results.scores = fused_scores[keep]
            results.labels = fused_labels[keep]
            results.bboxes = mask2bbox(fused_masks.bool())
            if export_stats:
                vote_stats = torch.tensor(stats_rows, dtype=torch.float32,
                                          device=fused_scores.device)
                results.vote_stats = vote_stats[keep]
            det_results.pred_instances = results
            if trace_dir:
                from carrier.same_pool_trace import write_vote_trace
                write_vote_trace(trace_dir, data_samples, aug_masks,
                                 merged_scores, merged_labels, trace_clusters,
                                 keep.tolist(), results, self.tta_cfg,
                                 merged_probs is not None)
            return det_results

        # Derive tight bboxes from the (flip-corrected) masks so NMS operates in
        # a consistent, orientation-fixed frame. mask2bbox wants a bool tensor.
        merged_bboxes = mask2bbox(merged_masks.bool())  # (K, 4) float

        # Class-wise NMS on the mask-derived bboxes; KEEP the masks at survivors.
        # Capture ``dets`` (not just keep_idxs): its last column carries the
        # rescored/decayed score aligned 1:1 with keep order, so a future
        # soft_nms actually takes effect. Under hard ``nms`` dets[:, -1] equals
        # merged_scores[keep_idxs], so this stays byte-identical to before.
        dets, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                      merged_labels, self.tta_cfg.nms)
        keep_idxs = keep_idxs[:self.tta_cfg.max_per_img]
        det_scores = dets[:self.tta_cfg.max_per_img, -1]

        results = InstanceData()
        results.masks = merged_masks[keep_idxs]
        results.bboxes = merged_bboxes[keep_idxs]
        results.scores = det_scores
        results.labels = merged_labels[keep_idxs]
        det_results.pred_instances = results
        return det_results


@MODELS.register_module()
class TTACascadeRoIHead(CascadeRoIHead):
    """CascadeRoIHead whose ``predict_mask`` ignores the ``flip`` meta key.

    WHY (universality matrix, 2026-08-01): stock ``CascadeRoIHead.predict_mask``
    averages the 3 cascade stages' mask logits via ``merge_aug_masks``, a
    leftover of the removed mmdet-2.x aug-test API that consults
    ``img_meta['flip']`` and tries to UN-FLIP each stage's RoI-local 28x28
    logits. Under our msTTA pipeline the flipped views carry ``flip=True`` in
    metainfo, which (a) CRASHES: ``mask[:, :, :, ::-1]`` negative-step slicing
    on a torch tensor raises ``ValueError: step must be greater than zero``,
    and (b) would be semantically WRONG even if it ran: within ONE view all 3
    stages' logits live in the SAME (flipped) frame as the bboxes they are
    pasted onto; the whole-image flip is undone ONCE by MaskTTAModel after
    pasting. So the per-stage average must not touch flip at all.

    Parameter-free override: byte-identical weights/keys to CascadeRoIHead;
    on a non-flipped pipeline (``flip`` absent/False) stock merge_aug_masks is
    a plain mean, exactly what this override computes -> output-identical to
    stock for all non-TTA evals.
    """

    def predict_mask(self, x, batch_img_metas, results_list, rescale=False):
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, mask_rois)
            mask_preds = mask_results['mask_preds']
            # split batch mask prediction back to each image
            mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
            aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        # plain per-stage mean, NO flip handling (see class docstring)
        merged_masks = []
        for i in range(len(batch_img_metas)):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_masks.append(torch.stack(aug_mask, dim=0).mean(dim=0))
        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)
        return results_list
