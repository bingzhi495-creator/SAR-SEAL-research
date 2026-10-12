# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE-0 test for the F2 mask-CONSENSUS-VOTING merge.

NO GPU. Exercises ``MaskTTAModel._merge_single_sample`` directly by constructing
a MaskTTAModel WITHOUT building any backbone (``__new__`` + a hand-set tta_cfg),
so no CUDA / checkpoint is ever touched. Three checks:

  (a) DUP-CLUSTER: 6 masks = 3 near-identical copies of hullA (mask-IoU>0.9,
      scores 0.90/0.70/0.50) + 3 near-identical copies of a low-mask-IoU
      neighbour hullB (scores 0.85/0.65/0.45). merge='vote' must return EXACTLY
      2 instances (one per true object), with fused scores == the max of each
      cluster (0.90 and 0.85) -- NOT 6 duplicates.

  (b) NEIGHBOUR-KEEP: two parallel hulls with bbox-IoU>0.6 but mask-IoU<0.3.
      merge='vote' must KEEP BOTH as separate clusters (mask-IoU below vote_iou),
      whereas merge='bbox_nms' hard-deletes one (keeps 1).

  (c) GATE-0: on a fixed synthetic input, merge='bbox_nms' with hard nms returns
      masks/scores/labels/bboxes BYTE-IDENTICAL to the pre-edit code path
      (reference computed inline by replicating the OLD logic). Prints
      total |diff| == 0.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_vote_merge.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from mmcv.ops import batched_nms
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


def mask_iou(a, b):
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 0.0


def bbox_iou(ba, bb):
    x1 = max(ba[0], bb[0]); y1 = max(ba[1], bb[1])
    x2 = min(ba[2], bb[2]); y2 = min(ba[3], bb[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    aa = (ba[2] - ba[0]) * (ba[3] - ba[1])
    ab = (bb[2] - bb[0]) * (bb[3] - bb[1])
    union = aa + ab - inter
    return inter / union if union else 0.0


def _band(offset, xlo, xhi):
    """A thin diagonal band |(y-x)-offset|<=2 restricted to x in [xlo,xhi]."""
    H = W = 100
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.abs((yy - xx) - offset) <= 2) & (xx >= xlo) & (xx <= xhi)


# --------------------------------------------------------------------------- #
# (a) DUP-CLUSTER: 3 copies of A + 3 copies of B -> exactly 2 fused instances
# --------------------------------------------------------------------------- #
def dup_clusters():
    """3 near-identical copies of hullA (offset +6) + 3 of hullB (offset -6).

    Copies differ only by a 1-pixel span endpoint so mask-IoU within a family is
    >0.9; the two families are 12 apart in (y-x) so A-vs-B mask-IoU is ~0.
    Interleaved [A0,B0,A1,B1,A2,B2] to force the greedy to skip across labels.
    """
    A0, A1, A2 = _band(6, 15, 75), _band(6, 15, 74), _band(6, 16, 75)
    B0, B1, B2 = _band(-6, 15, 75), _band(-6, 15, 74), _band(-6, 16, 75)
    masks = np.stack([A0, B0, A1, B1, A2, B2], axis=0)
    masks_t = torch.from_numpy(masks).bool()
    scores = torch.tensor([0.90, 0.85, 0.70, 0.65, 0.50, 0.45],
                          dtype=torch.float32)
    labels = torch.zeros(6, dtype=torch.long)  # same class -> they compete
    return masks_t, scores, labels


def check_dup_cluster():
    print('=' * 70)
    print('(a) DUP-CLUSTER  (3 copies of A + 3 of B -> vote emits exactly 2)')
    print('=' * 70)
    masks, scores, labels = dup_clusters()
    A0, B0 = masks[0], masks[1]

    iou_AA = mask_iou(masks[0], masks[2])   # A0 vs A1
    iou_AA2 = mask_iou(masks[0], masks[4])  # A0 vs A2
    iou_AB = mask_iou(masks[0], masks[1])   # A0 vs B0
    print(f'  mask-IoU within hullA family: A0-A1={iou_AA:.3f}  '
          f'A0-A2={iou_AA2:.3f}  (need >0.9)')
    print(f'  mask-IoU across families:     A0-B0={iou_AB:.3f}  (need <0.3)')
    assert iou_AA > 0.9 and iou_AA2 > 0.9, 'setup wrong: A copies not near-identical'
    assert iou_AB < 0.3, 'setup wrong: A,B families overlap too much'

    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out = make_model('vote')._merge_single_sample([ds]).pred_instances
    n = len(out.masks)
    out_scores = sorted([round(s, 3) for s in out.scores.tolist()], reverse=True)
    # which true object does each fused instance match?
    sA = max([out.scores[i].item() for i in range(n)
              if mask_iou(out.masks[i], A0) > 0.9], default=0.0)
    sB = max([out.scores[i].item() for i in range(n)
              if mask_iou(out.masks[i], B0) > 0.9], default=0.0)
    print(f'\n  [vote] returned instances = {n}   fused scores = {out_scores}')
    print(f'  [vote] fused score matching hullA = {sA:.3f}  (cluster max = 0.90)')
    print(f'  [vote] fused score matching hullB = {sB:.3f}  (cluster max = 0.85)')

    ok = True
    if n != 2:
        print(f'  !! FAIL: expected 2 fused instances, got {n}'); ok = False
    if abs(sA - 0.90) > 1e-5:
        print(f'  !! FAIL: hullA fused score {sA:.3f} != cluster max 0.90'); ok = False
    if abs(sB - 0.85) > 1e-5:
        print(f'  !! FAIL: hullB fused score {sB:.3f} != cluster max 0.85'); ok = False
    print(f'\n  (a) {"PASS" if ok else "FAIL"}: vote fused 6 -> {n} '
          f'(A={sA:.3f}, B={sB:.3f}), NOT 6 duplicates')
    return ok


# --------------------------------------------------------------------------- #
# (b) NEIGHBOUR-KEEP: parallel hulls kept by vote, one dropped by bbox_nms
# --------------------------------------------------------------------------- #
def parallel_hulls():
    """Two parallel thin diagonal hulls: bbox-IoU>0.6 but mask-IoU<0.3."""
    hullA = _band(2, 15, 75)
    hullB = _band(-2, 15, 75)
    masks = np.stack([hullA, hullB], axis=0)
    masks_t = torch.from_numpy(masks).bool()
    scores = torch.tensor([0.90, 0.85], dtype=torch.float32)
    labels = torch.zeros(2, dtype=torch.long)
    return masks_t, scores, labels


def check_neighbour_keep():
    print('\n' + '=' * 70)
    print('(b) NEIGHBOUR-KEEP  (vote keeps both parallel hulls; bbox_nms drops 1)')
    print('=' * 70)
    masks, scores, labels = parallel_hulls()
    hullA, hullB = masks[0], masks[1]
    boxes = mask2bbox(masks.bool())
    ba, bb = boxes[0].tolist(), boxes[1].tolist()
    m_iou = mask_iou(hullA, hullB)
    b_iou = bbox_iou(ba, bb)
    print(f'  bbox-IoU(A,B) = {b_iou:.3f}   (nms threshold = 0.6)')
    print(f'  mask-IoU(A,B) = {m_iou:.3f}   (vote_iou = 0.5)')
    assert b_iou > 0.6, f'setup wrong: bbox-IoU {b_iou:.3f} must exceed nms thr'
    assert m_iou < 0.3, f'setup wrong: mask-IoU {m_iou:.3f} must be low'

    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out_v = make_model('vote')._merge_single_sample([ds]).pred_instances
    n_vA = sum(mask_iou(out_v.masks[i], hullA) > 0.9 for i in range(len(out_v.masks)))
    n_vB = sum(mask_iou(out_v.masks[i], hullB) > 0.9 for i in range(len(out_v.masks)))
    distinct_v = (n_vA > 0) + (n_vB > 0)

    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out_b = make_model('bbox_nms')._merge_single_sample([ds]).pred_instances

    print(f'\n  [vote]     returned = {len(out_v.masks)}  '
          f'distinct hulls = {distinct_v}  (A={n_vA}, B={n_vB})')
    print(f'  [bbox_nms] returned = {len(out_b.masks)}  '
          f'(bbox-NMS suppresses the low-mask-IoU neighbour)')

    ok = True
    if not (len(out_v.masks) == 2 and distinct_v == 2):
        print('  !! FAIL: vote did not keep BOTH parallel hulls'); ok = False
    if not (len(out_b.masks) < 2):
        print('  !! FAIL: bbox_nms did not drop a hull (expected it to)'); ok = False
    print(f'\n  (b) {"PASS" if ok else "FAIL"}: vote keeps {distinct_v} hulls; '
          f'bbox_nms keeps {len(out_b.masks)}')
    return ok


# --------------------------------------------------------------------------- #
# (c) GATE-0: bbox_nms byte-identical to pre-edit logic
# --------------------------------------------------------------------------- #
def old_reference(masks, scores, labels, nms_cfg, max_per_img):
    """Verbatim replication of the PRE-EDIT _merge_single_sample bbox path."""
    merged_bboxes = mask2bbox(masks.bool())
    _, keep_idxs = batched_nms(merged_bboxes, scores, labels, nms_cfg)
    keep_idxs = keep_idxs[:max_per_img]
    return (masks[keep_idxs], merged_bboxes[keep_idxs],
            scores[keep_idxs], labels[keep_idxs])


def fixed_synthetic_input():
    """A deterministic multi-instance input to exercise real NMS survivors."""
    torch.manual_seed(1234)
    H = W = 64
    n = 7
    masks = torch.zeros(n, H, W, dtype=torch.bool)
    rng = np.random.RandomState(1234)
    for k in range(n):
        y0 = rng.randint(0, 40); x0 = rng.randint(0, 40)
        h = rng.randint(8, 22); w = rng.randint(8, 22)
        masks[k, y0:y0 + h, x0:x0 + w] = True
    masks[5] = masks[0].clone()  # near-duplicate so NMS removes something
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.4],
                          dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    return masks, scores, labels


def check_gate0():
    print('\n' + '=' * 70)
    print('(c) GATE-0  (merge=bbox_nms hard nms == pre-edit code path, '
          'byte-identical)')
    print('=' * 70)
    masks, scores, labels = fixed_synthetic_input()
    nms_cfg = ConfigDict(dict(type='nms', iou_threshold=0.6))

    ref_m, ref_b, ref_s, ref_l = old_reference(
        masks.clone(), scores.clone(), labels.clone(), nms_cfg, 100)

    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out = make_model('bbox_nms')._merge_single_sample([ds]).pred_instances

    d_masks = (out.masks.float() - ref_m.float()).abs().sum().item()
    d_bboxes = (out.bboxes - ref_b).abs().sum().item()
    d_scores = (out.scores - ref_s).abs().sum().item()
    d_labels = (out.labels - ref_l).abs().sum().item()
    total = d_masks + d_bboxes + d_scores + d_labels

    print(f'  survivors: new={len(out.masks)}  ref={len(ref_m)}')
    print(f'  |diff| masks ={d_masks}  bboxes={d_bboxes}  '
          f'scores={d_scores}  labels={d_labels}')
    print(f'  TOTAL |diff| == {total}')
    same = (len(out.masks) == len(ref_m)
            and torch.equal(out.masks, ref_m)
            and torch.equal(out.bboxes, ref_b)
            and torch.equal(out.scores, ref_s)
            and torch.equal(out.labels, ref_l))
    ok = same and (total == 0)
    print(f'  (c) GATE-0 {"PASS" if ok else "FAIL"}: '
          f'total|diff|==0 is {total == 0}, torch.equal(all) is {same}')
    return ok


def main():
    torch.set_grad_enabled(False)
    a = check_dup_cluster()
    b = check_neighbour_keep()
    c = check_gate0()
    print('\n' + '=' * 70)
    print(f'OVERALL: {"ALL PASS" if (a and b and c) else "FAIL"}  '
          f'[(a) dup-cluster={"PASS" if a else "FAIL"}, '
          f'(b) neighbour-keep={"PASS" if b else "FAIL"}, '
          f'(c) GATE-0={"PASS" if c else "FAIL"}]')
    print('=' * 70)
    sys.exit(0 if (a and b and c) else 1)


if __name__ == '__main__':
    main()
