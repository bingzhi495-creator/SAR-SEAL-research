# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE-0 test for F4 SOFT-PROBABILITY vote fusion (vote_mask='prob').

NO GPU. Exercises ``MaskTTAModel._merge_single_sample`` directly (``__new__`` +
hand-set tta_cfg, no backbone/checkpoint) and ``MSIoUFusionHead.instance_postprocess``
(plain nn.Module on CPU). Checks:

  (a) 3 members with SUB-PIXEL-SHIFTED soft boundaries (interior identical, two
      boundary cols where the members' sigmoid maps disagree by sub-threshold
      margins: col60 probs 0.90/0.45/0.45, col61 probs 0.90/0.10/0.10):
      prob-mean keeps col60 (mean 0.60 >= 0.5) that binary majority votes OUT
      (1/3 < 0.5) and drops col61 (mean 0.37) that the raw seed kept ->
      area(prob-fused) STRICTLY BETWEEN majority and seed; ASSERT
      area(prob) > area(majority). Identical members -> prob == majority == seed.
      PLUS flip invariance: same members presented via a flipped second view
      (masks AND probs pre-flipped, flip=True) -> byte-identical fused mask.
  (b) FALLBACK: no mask_probs attached -> vote_mask='prob' returns the SEED mask
      verbatim (not majority). PLUS singleton band gate: cluster_size 1 <
      vote_min_support -> seed even when (deliberately inconsistent) probs exist.
  (c) prob_high_agreement knob: cluster_size 3 > vote_max_support=2 ->
      'mean' fuses by prob-mean (area 420), 'seed' keeps seed verbatim (440).
  (d) GATE-0 (TTA): with mask_probs ATTACHED to the inputs, vote_mask
      'consensus' / 'majority' / 'seed' are BYTE-IDENTICAL to the pre-edit vote
      branch (replicated inline) -- probs are collected but ignored.
  (e) GATE-0 (fusion head): MSIoUFusionHead.instance_postprocess with
      export_mask_probs ABSENT == pre-edit stock math (replicated inline), no
      'mask_probs' attribute; with the flag ON all common fields are unchanged
      and (mask_probs >= 0.5) == masks pixel-for-pixel.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_voteprob_merge.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from mmdet.structures.mask import mask2bbox

from carrier.mask_tta import MaskTTAModel
from carrier.maskiou_head import MSIoUFusionHead


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_model(merge, **extra):
    """A MaskTTAModel with only tta_cfg set -- no backbone / no CUDA build."""
    m = MaskTTAModel.__new__(MaskTTAModel)
    cfg = dict(nms=dict(type='nms', iou_threshold=0.6),
               max_per_img=100,
               merge=merge)
    cfg.update(extra)
    m.tta_cfg = ConfigDict(cfg)
    return m


def make_sample(masks_bool, scores, labels, probs=None, flip=False,
                flip_direction=None):
    """One augmented view carrying masks/scores/labels (+ optional probs)."""
    inst = InstanceData()
    inst.masks = masks_bool
    inst.scores = scores
    inst.labels = labels
    if probs is not None:
        inst.mask_probs = probs
    ds = DetDataSample()
    ds.pred_instances = inst
    ds.set_metainfo(dict(flip=flip, flip_direction=flip_direction))
    return ds


def _square(r0, r1, c0, c1, H=100, W=100):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def subpixel_members():
    """3 member prob maps: identical interior (rows 40:60 x cols 40:60, p=.95),
    sub-pixel disagreeing boundary cols 60 (0.90/0.45/0.45) and 61
    (0.90/0.10/0.10), background 0.02. Binary = p >= 0.5:
    m1 = cols 40:62 (area 440), m2 = m3 = cols 40:60 (area 400).
    Expected fused areas: majority 400 < prob-mean 420 < seed 440."""
    H = W = 100
    P = np.full((3, H, W), 0.02, dtype=np.float32)
    P[:, 40:60, 40:60] = 0.95
    P[0, 40:60, 60] = 0.90
    P[1, 40:60, 60] = 0.45
    P[2, 40:60, 60] = 0.45
    P[0, 40:60, 61] = 0.90
    P[1, 40:60, 61] = 0.10
    P[2, 40:60, 61] = 0.10
    probs = torch.from_numpy(P).half()                 # fp16 like production
    masks = torch.from_numpy(P >= 0.5)                 # binary as exported
    scores = torch.tensor([0.90, 0.70, 0.50], dtype=torch.float32)
    labels = torch.zeros(3, dtype=torch.long)
    return masks, probs, scores, labels


def run_merge(model, samples):
    return model._merge_single_sample(samples).pred_instances


# --------------------------------------------------------------------------- #
# (a) sub-pixel boundary: prob-mean vs binary majority vs seed  (+ flip)
# --------------------------------------------------------------------------- #
def check_subpixel():
    print('=' * 70)
    print('(a) sub-pixel-shifted boundaries: prob-mean > majority; == when '
          'identical; flip invariance')
    print('=' * 70)
    masks, probs, scores, labels = subpixel_members()
    seed = masks[0]
    iou12 = ((masks[0] & masks[1]).sum().float()
             / (masks[0] | masks[1]).sum().float()).item()
    print(f'  member areas = {[int(m.sum()) for m in masks]}  '
          f'mask-IoU(m1,m2) = {iou12:.3f} (>= vote_iou 0.5 -> one cluster of 3)')

    band = dict(vote_min_support=2, vote_max_support=6)
    out_prob = run_merge(
        make_model('vote', vote_mask='prob', vote_iou=0.5, **band),
        [make_sample(masks.clone(), scores.clone(), labels.clone(),
                     probs.clone())])
    out_maj = run_merge(
        make_model('vote', vote_mask='majority', vote_iou=0.5),
        [make_sample(masks.clone(), scores.clone(), labels.clone())])
    a_prob = int(out_prob.masks[0].sum())
    a_maj = int(out_maj.masks[0].sum())
    a_seed = int(seed.sum())
    print(f'  area(prob-fused) = {a_prob}   area(binary-majority) = {a_maj}   '
          f'area(seed) = {a_seed}')
    print(f'  col60 kept by prob-mean (0.90+0.45+0.45)/3 = 0.60 >= 0.5 but '
          f'voted OUT by majority (1/3): {bool(out_prob.masks[0][50, 60])} vs '
          f'{bool(out_maj.masks[0][50, 60])}')
    ok_gt = (a_prob > a_maj) and len(out_prob.masks) == 1
    ok_map = (a_maj < a_prob < a_seed)   # boundary moved to MAP, not seed copy
    print(f'  ASSERT area(prob) > area(majority): {a_prob} > {a_maj} -> '
          f'{"PASS" if ok_gt else "FAIL"}')
    print(f'  prob-fused strictly between majority and seed (real fusion, not '
          f'seed copy): {"PASS" if ok_map else "FAIL"}')

    # identical members -> prob-fused == majority == seed
    probs_id = probs[0:1].repeat(3, 1, 1).clone()
    masks_id = masks[0:1].repeat(3, 1, 1).clone()
    out_id_p = run_merge(
        make_model('vote', vote_mask='prob', vote_iou=0.5, **band),
        [make_sample(masks_id.clone(), scores.clone(), labels.clone(),
                     probs_id.clone())])
    out_id_m = run_merge(
        make_model('vote', vote_mask='majority', vote_iou=0.5),
        [make_sample(masks_id.clone(), scores.clone(), labels.clone())])
    eq_id = (torch.equal(out_id_p.masks[0], masks_id[0])
             and torch.equal(out_id_p.masks[0], out_id_m.masks[0]))
    print(f'  identical members: prob-fused == member == majority-fused? '
          f'{eq_id} -> {"PASS" if eq_id else "FAIL"}')

    # flip invariance: member 2+3 delivered via a horizontally FLIPPED view
    v1 = make_sample(masks[0:1].clone(), scores[0:1].clone(),
                     labels[0:1].clone(), probs[0:1].clone())
    v2 = make_sample(masks[1:3].flip(dims=[-1]).clone(),
                     scores[1:3].clone(), labels[1:3].clone(),
                     probs[1:3].flip(dims=[-1]).clone(),
                     flip=True, flip_direction='horizontal')
    out_flip = run_merge(
        make_model('vote', vote_mask='prob', vote_iou=0.5, **band), [v1, v2])
    eq_flip = torch.equal(out_flip.masks, out_prob.masks)
    print(f'  flipped-view delivery -> byte-identical fused mask? {eq_flip} '
          f'-> {"PASS" if eq_flip else "FAIL"}')

    ok = ok_gt and ok_map and eq_id and eq_flip
    print(f'\n  (a) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (b) fallback (no probs) -> seed; singleton band gate -> seed despite probs
# --------------------------------------------------------------------------- #
def check_fallback_and_band():
    print('\n' + '=' * 70)
    print("(b) no mask_probs -> 'prob' falls back to SEED; singleton -> SEED")
    print('=' * 70)
    masks, probs, scores, labels = subpixel_members()
    seed = masks[0]
    out = run_merge(
        make_model('vote', vote_mask='prob', vote_iou=0.5,
                   vote_min_support=2, vote_max_support=6),
        [make_sample(masks.clone(), scores.clone(), labels.clone())])  # NO probs
    is_seed = torch.equal(out.masks[0], seed)
    print(f'  no probs attached: fused == seed verbatim? {is_seed} '
          f'(area {int(out.masks[0].sum())}, majority would be 400, '
          f'prob-mean 420)')

    # singleton with probs DELIBERATELY larger than the binary mask: band gate
    # (cluster_size 1 < min_support 2) must return the mask verbatim, proving
    # the seed branch was taken and the probs were NOT consulted.
    sm = torch.from_numpy(_square(80, 90, 80, 90))          # area 100
    sp = torch.from_numpy(
        np.where(_square(80, 90, 80, 92), 0.95, 0.02).astype(np.float32)).half()
    out2 = run_merge(
        make_model('vote', vote_mask='prob', vote_iou=0.5,
                   vote_min_support=2, vote_max_support=6),
        [make_sample(sm[None].clone(), torch.tensor([0.9]),
                     torch.zeros(1, dtype=torch.long), sp[None].clone())])
    is_seed2 = torch.equal(out2.masks[0], sm)
    print(f'  singleton (size 1 < min_support 2) with probs of area 120: '
          f'fused == binary mask (area {int(out2.masks[0].sum())})? {is_seed2}')
    ok = is_seed and is_seed2
    print(f'\n  (b) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (c) prob_high_agreement knob: size > max_support -> 'mean' vs 'seed'
# --------------------------------------------------------------------------- #
def check_high_agreement_knob():
    print('\n' + '=' * 70)
    print("(c) prob_high_agreement: cluster size 3 > vote_max_support=2")
    print('=' * 70)
    masks, probs, scores, labels = subpixel_members()
    seed = masks[0]
    common = dict(vote_iou=0.5, vote_min_support=2, vote_max_support=2)
    out_mean = run_merge(
        make_model('vote', vote_mask='prob', prob_high_agreement='mean',
                   **common),
        [make_sample(masks.clone(), scores.clone(), labels.clone(),
                     probs.clone())])
    out_seed = run_merge(
        make_model('vote', vote_mask='prob', prob_high_agreement='seed',
                   **common),
        [make_sample(masks.clone(), scores.clone(), labels.clone(),
                     probs.clone())])
    a_mean = int(out_mean.masks[0].sum())
    a_seed = int(out_seed.masks[0].sum())
    ok_mean = a_mean == 420 and not torch.equal(out_mean.masks[0], seed)
    ok_seed = torch.equal(out_seed.masks[0], seed)
    print(f"  'mean': fused area = {a_mean} (expect 420, prob-mean)  -> "
          f'{"PASS" if ok_mean else "FAIL"}')
    print(f"  'seed': fused == seed verbatim (area {a_seed}, expect 440)? "
          f'{ok_seed} -> {"PASS" if ok_seed else "FAIL"}')
    ok = ok_mean and ok_seed
    print(f'\n  (c) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (d) GATE-0 TTA: consensus/majority/seed byte-identical, probs attached
# --------------------------------------------------------------------------- #
def old_vote_reference(merged_masks, merged_scores, merged_labels, n_views,
                       vote_iou, vote_frac, vote_gamma, vote_mask,
                       vote_min_support, vote_max_support, max_per_img):
    """Verbatim replication of the PRE-EDIT vote branch
    (seed / majority / consensus)."""
    K = merged_masks.shape[0]
    flat = merged_masks.bool().reshape(K, -1).float()
    area = flat.sum(dim=1)
    inter = flat @ flat.t()
    union = area[:, None] + area[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    order = torch.argsort(merged_scores, descending=True)
    used = torch.zeros(K, dtype=torch.bool)
    fused_masks, fused_scores, fused_labels = [], [], []
    for idx in order.tolist():
        if used[idx]:
            continue
        same = merged_labels == merged_labels[idx]
        members = (~used) & same & (iou[idx] >= vote_iou)
        members[idx] = True
        used |= members
        cluster_size = int(members.sum())
        if vote_mask == 'seed':
            fused = merged_masks[idx].bool()
        elif vote_mask == 'consensus':
            if vote_min_support <= cluster_size <= vote_max_support:
                member_masks = merged_masks[members].bool().float()
                fused = member_masks.mean(dim=0) >= vote_frac
            else:
                fused = merged_masks[idx].bool()
        else:
            member_masks = merged_masks[members].bool().float()
            fused = member_masks.mean(dim=0) >= vote_frac
        fused_masks.append(fused)
        support = (cluster_size / n_views) ** vote_gamma
        fused_scores.append(merged_scores[members].max() * support)
        fused_labels.append(merged_labels[idx])
    fused_masks = torch.stack(fused_masks, dim=0)
    fused_scores = torch.stack(fused_scores)
    fused_labels = torch.stack(fused_labels)
    keep = torch.argsort(fused_scores, descending=True)[:max_per_img]
    fused_masks = fused_masks[keep]
    return (fused_masks, fused_scores[keep], fused_labels[keep],
            mask2bbox(fused_masks.bool()))


def gate_input():
    """Multi-cluster deterministic input: 2 families of 3 (mask-IoU high within,
    ~0 across) + a singleton of a different label -> 3 clusters, mixed sizes."""
    def band(offset, xlo, xhi):
        H = W = 100
        yy, xx = np.mgrid[0:H, 0:W]
        return (np.abs((yy - xx) - offset) <= 2) & (xx >= xlo) & (xx <= xhi)
    A0, A1, A2 = band(6, 15, 75), band(6, 15, 74), band(6, 16, 75)
    B0, B1, B2 = band(-6, 15, 75), band(-6, 15, 74), band(-6, 16, 75)
    C0 = _square(80, 95, 5, 20)  # separate label -> its own singleton cluster
    masks = np.stack([A0, B0, A1, C0, B1, A2, B2], axis=0)
    masks_t = torch.from_numpy(masks).bool()
    scores = torch.tensor([0.90, 0.85, 0.70, 0.80, 0.65, 0.50, 0.45],
                          dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    probs = torch.rand((7, 100, 100), generator=g).half()  # attached + ignored
    return masks_t, scores, labels, probs


def _gate_one(vote_mask):
    masks, scores, labels, probs = gate_input()
    ref_m, ref_s, ref_l, ref_b = old_vote_reference(
        masks.clone(), scores.clone(), labels.clone(),
        n_views=1, vote_iou=0.5, vote_frac=0.5, vote_gamma=0.0,
        vote_mask=vote_mask, vote_min_support=2, vote_max_support=5,
        max_per_img=100)
    ds = make_sample(masks.clone(), scores.clone(), labels.clone(),
                     probs.clone())   # probs ATTACHED: must be ignored
    out = run_merge(make_model('vote', vote_mask=vote_mask, vote_iou=0.5),
                    [ds])
    d_masks = (out.masks.float() - ref_m.float()).abs().sum().item()
    d_scores = (out.scores - ref_s).abs().sum().item()
    d_labels = (out.labels - ref_l).abs().sum().item()
    d_bboxes = (out.bboxes - ref_b).abs().sum().item()
    total = d_masks + d_scores + d_labels + d_bboxes
    same = (len(out.masks) == len(ref_m)
            and torch.equal(out.masks, ref_m)
            and torch.equal(out.scores, ref_s)
            and torch.equal(out.labels, ref_l)
            and torch.equal(out.bboxes, ref_b))
    print(f"  vote_mask='{vote_mask}' (probs attached): survivors "
          f"new={len(out.masks)} ref={len(ref_m)}  |diff| m={d_masks} "
          f"s={d_scores} l={d_labels} b={d_bboxes}  TOTAL={total}  "
          f"torch.equal(all)={same}")
    return (total == 0) and same


def check_gate_tta():
    print('\n' + '=' * 70)
    print("(d) GATE-0 TTA: consensus/majority/seed == pre-edit vote branch "
          "(mask_probs attached and ignored)")
    print('=' * 70)
    ok_c = _gate_one('consensus')
    ok_m = _gate_one('majority')
    ok_s = _gate_one('seed')
    ok = ok_c and ok_m and ok_s
    print(f"\n  (d) {'PASS' if ok else 'FAIL'}: consensus={ok_c}, "
          f"majority={ok_m}, seed={ok_s}")
    return ok


# --------------------------------------------------------------------------- #
# (e) GATE-0 fusion head: export flag off byte-identical, on adds fp16 probs
# --------------------------------------------------------------------------- #
def stock_instance_postprocess(head, mask_cls, mask_pred):
    """Verbatim replication of the PRE-EDIT MSIoUFusionHead.instance_postprocess
    (lam=0 path)."""
    max_per_image = head.test_cfg.get('max_per_image', 100)
    num_queries = mask_cls.shape[0]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(head.num_classes, device=mask_cls.device).\
        unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
    scores_per_image, top_indices = scores.flatten(0, 1).topk(
        max_per_image, sorted=False)
    labels_per_image = labels[top_indices]
    query_indices = top_indices // head.num_classes
    mask_pred = mask_pred[query_indices]
    is_thing = labels_per_image < head.num_things_classes
    scores_per_image = scores_per_image[is_thing]
    labels_per_image = labels_per_image[is_thing]
    mask_pred = mask_pred[is_thing]
    mask_pred_binary = (mask_pred > 0).float()
    mask_scores_per_image = (mask_pred.sigmoid() *
                             mask_pred_binary).flatten(1).sum(1) / (
                                 mask_pred_binary.flatten(1).sum(1) + 1e-6)
    det_scores = scores_per_image * mask_scores_per_image
    mask_pred_binary = mask_pred_binary.bool()
    return (mask2bbox(mask_pred_binary), labels_per_image, det_scores,
            mask_pred_binary)


def check_gate_fusion_head():
    print('\n' + '=' * 70)
    print('(e) GATE-0 fusion head: export_mask_probs off == stock; on adds '
          'consistent fp16 probs')
    print('=' * 70)
    torch.manual_seed(0)
    Q, H, W = 10, 40, 40
    mask_cls = torch.randn(Q, 2)                       # 1 thing class + bg
    logits = torch.randn(Q, H, W)
    logits = logits + 0.05 * torch.sign(logits)        # keep |logit| >= 0.05:
    # avoids the measure-zero fp16 rounding band |logit| < ~5e-4 where
    # sigmoid(logit).half() rounds to exactly 0.5 (noted caveat, not a bug).

    head_off = MSIoUFusionHead(
        num_things_classes=1, num_stuff_classes=0,
        test_cfg=ConfigDict(max_per_image=6))          # flag ABSENT
    head_on = MSIoUFusionHead(
        num_things_classes=1, num_stuff_classes=0,
        test_cfg=ConfigDict(max_per_image=6, export_mask_probs=True))

    ref_b, ref_l, ref_s, ref_m = stock_instance_postprocess(
        head_off, mask_cls.clone(), logits.clone())
    out_off = head_off.instance_postprocess(mask_cls.clone(), logits.clone())
    out_on = head_on.instance_postprocess(mask_cls.clone(), logits.clone())

    off_same = (torch.equal(out_off.bboxes, ref_b)
                and torch.equal(out_off.labels, ref_l)
                and torch.equal(out_off.scores, ref_s)
                and torch.equal(out_off.masks, ref_m))
    off_no_attr = 'mask_probs' not in out_off
    print(f'  flag OFF: fields byte-identical to pre-edit stock math? '
          f'{off_same}   mask_probs absent? {off_no_attr}')

    on_same = (torch.equal(out_on.bboxes, out_off.bboxes)
               and torch.equal(out_on.labels, out_off.labels)
               and torch.equal(out_on.scores, out_off.scores)
               and torch.equal(out_on.masks, out_off.masks))
    has_probs = 'mask_probs' in out_on
    dtype_ok = has_probs and out_on.mask_probs.dtype == torch.float16
    shape_ok = has_probs and (out_on.mask_probs.shape == out_on.masks.shape)
    consist = has_probs and torch.equal(
        out_on.mask_probs.float() >= 0.5, out_on.masks)
    print(f'  flag ON : common fields unchanged? {on_same}   probs present? '
          f'{has_probs} dtype fp16? {dtype_ok} shape==masks? {shape_ok}')
    print(f'            (mask_probs >= 0.5) == masks pixel-for-pixel? '
          f'{consist}')
    ok = off_same and off_no_attr and on_same and dtype_ok and shape_ok \
        and consist
    print(f'\n  (e) {"PASS" if ok else "FAIL"}')
    return ok


def main():
    torch.set_grad_enabled(False)
    a = check_subpixel()
    b = check_fallback_and_band()
    c = check_high_agreement_knob()
    d = check_gate_tta()
    e = check_gate_fusion_head()
    print('\n' + '=' * 70)
    print(f'OVERALL: {"ALL PASS" if (a and b and c and d and e) else "FAIL"}  '
          f'[(a) subpixel={"PASS" if a else "FAIL"}, '
          f'(b) fallback/band={"PASS" if b else "FAIL"}, '
          f'(c) high-agreement knob={"PASS" if c else "FAIL"}, '
          f'(d) GATE-0 TTA={"PASS" if d else "FAIL"}, '
          f'(e) GATE-0 fusion head={"PASS" if e else "FAIL"}]')
    print('=' * 70)
    sys.exit(0 if (a and b and c and d and e) else 1)


if __name__ == '__main__':
    main()
