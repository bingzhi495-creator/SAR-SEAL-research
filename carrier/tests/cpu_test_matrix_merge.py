# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE-0 test for the F3 mask-IoU Matrix-NMS merge.

NO GPU. Exercises ``MaskTTAModel._merge_single_sample`` directly by constructing
a MaskTTAModel WITHOUT building any backbone (``__new__`` + a hand-set tta_cfg),
so no CUDA / checkpoint is ever touched. Two checks:

  (a) SYNTHETIC parallel-hull: two thin elongated hulls whose BBOXES overlap
      (bbox-IoU > the 0.6 nms threshold) but whose MASKS barely touch
      (mask-IoU ~0.1), plus one EXACT duplicate of hull-1.
        * merge='bbox_nms' (hard nms) must SUPPRESS a true hull -> keeps < 2
          distinct hulls (bbox-NMS wrongly deletes the low-mask-IoU neighbour).
        * merge='matrix' must KEEP BOTH hulls (both retain high score) and
          decay the exact duplicate to ~0 (effectively removed).

  (b) GATE-0: on a fixed synthetic input, merge='bbox_nms' with hard nms returns
      masks/scores/labels/bboxes BYTE-IDENTICAL to the pre-edit code path
      (reference computed inline by replicating the OLD logic). Prints
      total |diff| == 0.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_matrix_merge.py
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
def make_model(merge):
    """A MaskTTAModel with only tta_cfg set -- no backbone / no CUDA build."""
    m = MaskTTAModel.__new__(MaskTTAModel)
    m.tta_cfg = ConfigDict(
        dict(nms=dict(type='nms', iou_threshold=0.6),
             max_per_img=100,
             merge=merge))
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


def parallel_hulls():
    """Two parallel thin diagonal hulls + one exact duplicate of hull-1.

    Diagonal bands on a 100x100 grid:
      hull A: y-x in [0,4]      (center y=x+2, half-width 2)
      hull B: y-x in [-4,0]     (center y=x-2, half-width 2)
    They share only the y-x==0 diagonal -> low mask-IoU; both span x in [15,75]
    so their bboxes overlap heavily -> high bbox-IoU. dup is an exact copy of A.
    Returns (masks(3,H,W) bool, scores, labels).
    """
    H = W = 100
    yy, xx = np.mgrid[0:H, 0:W]
    span = (xx >= 15) & (xx <= 75)
    hullA = (np.abs((yy - xx) - 2) <= 2) & span
    hullB = (np.abs((yy - xx) + 2) <= 2) & span
    masks = np.stack([hullA, hullB, hullA.copy()], axis=0)  # A, B, dup(A)
    masks_t = torch.from_numpy(masks).bool()
    # scores: A highest, B next, dup lowest (so dup gets decayed under A).
    scores = torch.tensor([0.90, 0.85, 0.80], dtype=torch.float32)
    labels = torch.zeros(3, dtype=torch.long)  # same class -> they compete
    return masks_t, scores, labels


# --------------------------------------------------------------------------- #
# (a) parallel-hull suppression behaviour
# --------------------------------------------------------------------------- #
def check_parallel_hulls():
    print('=' * 70)
    print('(a) SYNTHETIC parallel-hull  (bbox-NMS deletes true neighbour; '
          'matrix keeps both)')
    print('=' * 70)
    masks, scores, labels = parallel_hulls()
    hullA, hullB = masks[0], masks[1]

    boxes = mask2bbox(masks.bool())
    ba, bb = boxes[0].tolist(), boxes[1].tolist()
    m_iou = mask_iou(hullA, hullB)
    b_iou = bbox_iou(ba, bb)
    print(f'  hullA bbox = {[round(v,1) for v in ba]}')
    print(f'  hullB bbox = {[round(v,1) for v in bb]}')
    print(f'  bbox-IoU(A,B) = {b_iou:.3f}   (nms threshold = 0.6)')
    print(f'  mask-IoU(A,B) = {m_iou:.3f}')
    assert b_iou > 0.6, f'setup wrong: bbox-IoU {b_iou:.3f} must exceed nms thr'
    assert m_iou < 0.3, f'setup wrong: mask-IoU {m_iou:.3f} must be low'

    # ---- bbox_nms (hard nms) ----
    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out_b = make_model('bbox_nms')._merge_single_sample([ds]).pred_instances
    n_hullA_b = sum(mask_iou(out_b.masks[i], hullA) > 0.9
                    for i in range(len(out_b.masks)))
    n_hullB_b = sum(mask_iou(out_b.masks[i], hullB) > 0.9
                    for i in range(len(out_b.masks)))
    distinct_b = (n_hullA_b > 0) + (n_hullB_b > 0)
    print(f'\n  [bbox_nms] survivors = {len(out_b.masks)}  '
          f'scores = {[round(s,3) for s in out_b.scores.tolist()]}')
    print(f'  [bbox_nms] copies matching hullA={n_hullA_b}  hullB={n_hullB_b}  '
          f'-> distinct hulls = {distinct_b}')

    # ---- matrix ----
    ds = make_sample(masks.clone(), scores.clone(), labels.clone())
    out_m = make_model('matrix')._merge_single_sample([ds]).pred_instances
    ms = out_m.scores.tolist()
    # best surviving score for each true hull
    sA = max([out_m.scores[i].item() for i in range(len(out_m.masks))
              if mask_iou(out_m.masks[i], hullA) > 0.9], default=0.0)
    sB = max([out_m.scores[i].item() for i in range(len(out_m.masks))
              if mask_iou(out_m.masks[i], hullB) > 0.9], default=0.0)
    # duplicate = the SECOND (lower) score among hullA-matching masks
    hullA_scores = sorted(
        [out_m.scores[i].item() for i in range(len(out_m.masks))
         if mask_iou(out_m.masks[i], hullA) > 0.9], reverse=True)
    dup_score = hullA_scores[1] if len(hullA_scores) > 1 else 0.0
    print(f'\n  [matrix]   returned = {len(out_m.masks)}  '
          f'scores = {[round(s,3) for s in ms]}')
    print(f'  [matrix]   best score  hullA={sA:.3f}  hullB={sB:.3f}  '
          f'duplicate(A)={dup_score:.3f}')

    # ---- assertions ----
    ok = True
    if not (distinct_b < 2):
        print('  !! FAIL: bbox_nms kept >=2 distinct hulls (expected it to '
              'suppress a true hull)'); ok = False
    if not (sA > 0.5 and sB > 0.5):
        print('  !! FAIL: matrix did not keep BOTH hulls with high score'); ok = False
    if not (dup_score < 0.2):
        print('  !! FAIL: matrix did not decay the exact duplicate to ~0'); ok = False
    print(f'\n  (a) {"PASS" if ok else "FAIL"}: bbox_nms drops a true hull '
          f'(distinct={distinct_b}<2); matrix keeps A({sA:.3f}) & B({sB:.3f}) '
          f'and decays dup({dup_score:.3f})')
    return ok


# --------------------------------------------------------------------------- #
# (b) GATE-0: bbox_nms byte-identical to pre-edit logic
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
    # add a near-duplicate pair so NMS actually removes something
    masks[5] = masks[0].clone()
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.4],
                          dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    return masks, scores, labels


def check_gate0():
    print('\n' + '=' * 70)
    print('(b) GATE-0  (merge=bbox_nms hard nms == pre-edit code path, '
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
    print(f'  (b) GATE-0 {"PASS" if ok else "FAIL"}: '
          f'total|diff|==0 is {total == 0}, torch.equal(all) is {same}')
    return ok


def main():
    torch.set_grad_enabled(False)
    a = check_parallel_hulls()
    b = check_gate0()
    print('\n' + '=' * 70)
    print(f'OVERALL: {"ALL PASS" if (a and b) else "FAIL"}  '
          f'[(a) parallel-hull={"PASS" if a else "FAIL"}, '
          f'(b) GATE-0={"PASS" if b else "FAIL"}]')
    print('=' * 70)
    sys.exit(0 if (a and b) else 1)


if __name__ == '__main__':
    main()
