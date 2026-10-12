# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE test for the AREA-ADAPTIVE vote fusion (vote_mask='adaptive').

NO GPU. Exercises ``MaskTTAModel._merge_single_sample`` directly by constructing
a MaskTTAModel WITHOUT any backbone (``__new__`` + a hand-set tta_cfg), so no
CUDA / checkpoint is ever touched. Checks:

  (a) SMALL cluster (seed area < vote_area_thr, 3 differing members) -> adaptive
      takes the MAJORITY branch: fused == pixelwise-majority(members) and fused
      != seed. Prints branch + whether fused==seed.
  (b) LARGE cluster (seed area >> vote_area_thr, 3 differing members) -> adaptive
      takes the SEED branch: fused == merged_masks[idx] verbatim (asserted),
      even though majority WOULD differ.
  (c) SMALL singleton (cluster_size 1 < vote_min_support) -> adaptive falls back
      to SEED (can't denoise 1 view). PLUS a support-gate proof: the SAME small
      3-member cluster from (a) with vote_min_support=4 (> cluster_size 3) is
      forced to SEED, isolating the support gate from the area gate.
  (d) GATE: with vote_mask='seed' and with vote_mask='majority', the edited code
      is BYTE-IDENTICAL to the pre-edit vote branch (replicated inline). Prints
      total |diff| == 0 for both.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_voteadapt_merge.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from mmdet.structures.mask import mask2bbox

from carrier.mask_tta import MaskTTAModel


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


def make_sample(masks_bool, scores, labels):
    """One augmented view (flip=False) carrying all masks/scores/labels."""
    inst = InstanceData()
    inst.masks = masks_bool
    inst.scores = scores
    inst.labels = labels
    ds = DetDataSample()
    ds.pred_instances = inst
    ds.set_metainfo(dict(flip=False, flip_direction=None))
    return ds


def _square(r0, r1, c0, c1, H=100, W=100):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def mask_iou(a, b):
    a = torch.as_tensor(a).bool(); b = torch.as_tensor(b).bool()
    inter = (a & b).sum().item(); union = (a | b).sum().item()
    return inter / union if union else 0.0


def majority_of(members_masks, frac=0.5):
    """Reference pixelwise-majority of a stack of bool masks (mean>=frac)."""
    return members_masks.bool().float().mean(dim=0) >= frac


# --------------------------------------------------------------------------- #
# (a) SMALL cluster -> adaptive picks MAJORITY
# --------------------------------------------------------------------------- #
def check_small_majority():
    print('=' * 70)
    print('(a) SMALL cluster (seed area < 1024, 3 members) -> MAJORITY branch')
    print('=' * 70)
    A0 = _square(40, 60, 40, 60)          # seed, area 400
    A1 = _square(40, 60, 40, 62)          # +2 cols -> area 440
    A2 = _square(40, 60, 40, 62)          # identical copy of A1
    masks = torch.from_numpy(np.stack([A0, A1, A2], axis=0)).bool()
    scores = torch.tensor([0.90, 0.70, 0.50], dtype=torch.float32)
    labels = torch.zeros(3, dtype=torch.long)

    seed = masks[0]                       # top-score member
    a = int(seed.sum())
    expected_maj = majority_of(masks)     # pixels in >=2 of 3 -> cols 40:62
    print(f'  seed area = {a} px  (vote_area_thr=1024)   members = 3 '
          f'(min_support=2)')
    print(f'  mask-IoU(seed,A1) = {mask_iou(A0, A1):.3f}  (vote_iou=0.5)')

    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out = make_model('vote', vote_mask='adaptive',
                     vote_area_thr=1024, vote_min_support=2
                     )._merge_single_sample([ds]).pred_instances
    fused = out.masks[0]
    is_seed = torch.equal(fused, seed)
    is_maj = torch.equal(fused, expected_maj)
    branch = 'MAJORITY' if (is_maj and not is_seed) else (
        'SEED' if is_seed else 'OTHER')
    print(f'\n  returned instances = {len(out.masks)}')
    print(f'  fused == seed?            {is_seed}')
    print(f'  fused == majority(members)? {is_maj}')
    print(f'  -> adaptive branch taken = {branch}')
    ok = is_maj and (not is_seed) and len(out.masks) == 1
    print(f'\n  (a) {"PASS" if ok else "FAIL"}: small cluster fused by MAJORITY '
          f'(differs from seed)')
    return ok


# --------------------------------------------------------------------------- #
# (b) LARGE cluster -> adaptive picks SEED verbatim
# --------------------------------------------------------------------------- #
def check_large_seed():
    print('\n' + '=' * 70)
    print('(b) LARGE cluster (seed area >> 1024, 3 members) -> SEED branch')
    print('=' * 70)
    A0 = _square(10, 90, 10, 90)          # seed, area 6400
    A1 = _square(10, 90, 10, 92)          # +2 cols -> area 6560
    A2 = _square(10, 90, 10, 92)
    masks = torch.from_numpy(np.stack([A0, A1, A2], axis=0)).bool()
    scores = torch.tensor([0.90, 0.70, 0.50], dtype=torch.float32)
    labels = torch.zeros(3, dtype=torch.long)

    seed = masks[0]
    a = int(seed.sum())
    expected_maj = majority_of(masks)
    maj_differs = not torch.equal(expected_maj, seed)
    print(f'  seed area = {a} px  (>> vote_area_thr=1024)   members = 3')
    print(f'  majority(members) would differ from seed? {maj_differs} '
          f'(so SEED is a real choice)')

    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out = make_model('vote', vote_mask='adaptive',
                     vote_area_thr=1024, vote_min_support=2
                     )._merge_single_sample([ds]).pred_instances
    fused = out.masks[0]
    is_seed = torch.equal(fused, seed)
    print(f'\n  returned instances = {len(out.masks)}')
    print(f'  fused == seed (verbatim)? {is_seed}')
    print(f'  -> adaptive branch taken = {"SEED" if is_seed else "OTHER"}')
    assert is_seed, 'FAIL: large-object adaptive did NOT return seed verbatim'
    ok = is_seed and maj_differs and len(out.masks) == 1
    print(f'\n  (b) {"PASS" if ok else "FAIL"}: large cluster kept SEED verbatim')
    return ok


# --------------------------------------------------------------------------- #
# (c) SMALL singleton -> SEED (support gate)  +  min_support isolation proof
# --------------------------------------------------------------------------- #
def check_singleton_and_support_gate():
    print('\n' + '=' * 70)
    print('(c) SMALL singleton (cluster_size 1 < min_support) -> SEED fallback')
    print('=' * 70)
    S = _square(40, 55, 40, 55)           # area 225 < 1024
    masks = torch.from_numpy(S[None]).bool()
    scores = torch.tensor([0.90], dtype=torch.float32)
    labels = torch.zeros(1, dtype=torch.long)
    seed = masks[0]
    print(f'  seed area = {int(seed.sum())} px (< 1024)   members = 1 '
          f'(< min_support=2)')
    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out = make_model('vote', vote_mask='adaptive',
                     vote_area_thr=1024, vote_min_support=2
                     )._merge_single_sample([ds]).pred_instances
    is_seed = torch.equal(out.masks[0], seed)
    print(f'  fused == seed? {is_seed}  -> branch = '
          f'{"SEED (no denoise with 1 view)" if is_seed else "OTHER"}')
    ok1 = is_seed and len(out.masks) == 1

    print('\n  support-gate isolation: the (a) SMALL 3-member cluster, but '
          'min_support=4')
    A0 = _square(40, 60, 40, 60); A1 = _square(40, 60, 40, 62)
    A2 = _square(40, 60, 40, 62)
    masks2 = torch.from_numpy(np.stack([A0, A1, A2], axis=0)).bool()
    scores2 = torch.tensor([0.90, 0.70, 0.50], dtype=torch.float32)
    labels2 = torch.zeros(3, dtype=torch.long)
    seed2 = masks2[0]; maj2 = majority_of(masks2)
    ds2 = make_sample(masks2.clone(), scores2.clone(), labels2.clone())
    out2 = make_model('vote', vote_mask='adaptive',
                      vote_area_thr=1024, vote_min_support=4
                      )._merge_single_sample([ds2]).pred_instances
    fused2 = out2.masks[0]
    is_seed2 = torch.equal(fused2, seed2)
    would_maj = not torch.equal(maj2, seed2)
    print(f'  cluster_size 3 < min_support 4  ->  fused == seed? {is_seed2}  '
          f'(majority would differ? {would_maj})')
    ok2 = is_seed2 and would_maj
    ok = ok1 and ok2
    print(f'\n  (c) {"PASS" if ok else "FAIL"}: cluster_size < min_support '
          f'forces SEED even for small objects')
    return ok


# --------------------------------------------------------------------------- #
# (d) GATE: 'seed' and 'majority' byte-identical to pre-edit vote branch
# --------------------------------------------------------------------------- #
def old_vote_reference(merged_masks, merged_scores, merged_labels, n_views,
                       vote_iou, vote_frac, vote_gamma, vote_mask, max_per_img):
    """Verbatim replication of the PRE-EDIT vote branch (seed / majority)."""
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
        if vote_mask == 'seed':
            fused_masks.append(merged_masks[idx].bool())
        else:
            member_masks = merged_masks[members].bool().float()
            fused_masks.append(member_masks.mean(dim=0) >= vote_frac)
        cluster_size = int(members.sum())
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
    return masks_t, scores, labels


def _gate_one(vote_mask):
    masks, scores, labels = gate_input()
    ref_m, ref_s, ref_l, ref_b = old_vote_reference(
        masks.clone(), scores.clone(), labels.clone(),
        n_views=1, vote_iou=0.5, vote_frac=0.5, vote_gamma=0.0,
        vote_mask=vote_mask, max_per_img=100)
    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out = make_model('vote', vote_mask=vote_mask,
                     vote_iou=0.5)._merge_single_sample([ds]).pred_instances
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
    print(f"  vote_mask='{vote_mask}': survivors new={len(out.masks)} "
          f"ref={len(ref_m)}  |diff| m={d_masks} s={d_scores} l={d_labels} "
          f"b={d_bboxes}  TOTAL={total}  torch.equal(all)={same}")
    return (total == 0) and same


def check_gate():
    print('\n' + '=' * 70)
    print("(d) GATE  (vote_mask 'seed' & 'majority' == pre-edit vote branch)")
    print('=' * 70)
    ok_seed = _gate_one('seed')
    ok_maj = _gate_one('majority')
    ok = ok_seed and ok_maj
    print(f"\n  (d) {'PASS' if ok else 'FAIL'}: seed byte-identical={ok_seed}, "
          f"majority byte-identical={ok_maj}")
    return ok


def main():
    torch.set_grad_enabled(False)
    a = check_small_majority()
    b = check_large_seed()
    c = check_singleton_and_support_gate()
    d = check_gate()
    print('\n' + '=' * 70)
    print(f'OVERALL: {"ALL PASS" if (a and b and c and d) else "FAIL"}  '
          f'[(a) small->majority={"PASS" if a else "FAIL"}, '
          f'(b) large->seed={"PASS" if b else "FAIL"}, '
          f'(c) support-gate={"PASS" if c else "FAIL"}, '
          f'(d) GATE={"PASS" if d else "FAIL"}]')
    print('=' * 70)
    sys.exit(0 if (a and b and c and d) else 1)


if __name__ == '__main__':
    main()
