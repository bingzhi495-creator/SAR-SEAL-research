# -*- coding: utf-8 -*-
"""GEPL weak supervision: evidence-affinity pairwise loss (verdict-7 core).

BoxInst's color pairwise affinity assumes neighboring pixels of similar COLOR
belong to the same instance. Under SAR speckle this is false (multiplicative
noise destroys local color/intensity coherence). We replace color affinity with
EVIDENCE affinity: neighboring pixels with similar multi-scale MNE evidence
vectors are likely the same object (ship-interior vs sea-vs-clutter are
separable in evidence space even when intensity is noisy).

  A(u,v) = exp(-||E(u) - E(v)||_1 / sigma_E)   for v in N(u)

Pairwise loss (BoxInst form): for high-affinity neighbor pairs, push their
predicted mask probabilities to agree; only inside the GT box (weak label).
Plus the standard box-projection (tightness) loss so masks fill the box.

This module is framework-agnostic torch; a numpy reference + unit test verify
the affinity and projection terms. SAM2 fusion / negative-point prompting are
separate (need SAM2 weights) and live in weak_supervision_sam.py later.
"""
from __future__ import annotations

import numpy as np


# ---------------- numpy reference (for tests / offline analysis) ----------------
def evidence_affinity_np(E: np.ndarray, dilation: int = 2, sigma_E: float = 1.0):
    """E: (C,H,W) evidence vectors. Returns affinity to the 8 neighbors at the
    given dilation: (8,H,W) in [0,1]. ||.||_1 over channels."""
    C, H, W = E.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    out = np.zeros((8, H, W), np.float32)
    for k, (dy, dx) in enumerate(offs):
        shifted = np.roll(E, shift=(dy * dilation, dx * dilation), axis=(1, 2))
        l1 = np.abs(E - shifted).sum(0)
        out[k] = np.exp(-l1 / sigma_E)
    return out


def box_projection_np(prob: np.ndarray, box) -> float:
    """Dice between the box's row/col max-projection of `prob` and the full
    box projection (=1). prob: (H,W) in [0,1]; box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = prob[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    proj_x = crop.max(0)  # over rows -> width vector
    proj_y = crop.max(1)  # over cols -> height vector
    # target projection is all ones (box fully covers the object in each axis)
    def dice(p):
        return (2 * p.sum()) / (p.sum() + p.size + 1e-6)
    return float(0.5 * (dice(proj_x) + dice(proj_y)))


# ---------------- torch losses (training) ----------------
def evidence_pairwise_loss(mask_logits, evidence, box_mask, dilation=2,
                           sigma_E=1.0, affinity_thresh=0.3):
    """torch. mask_logits,(evidence),(box_mask): (H,W)/(C,H,W)/(H,W).
    For high-evidence-affinity neighbor pairs inside box_mask, penalize
    disagreement of predicted mask log-probabilities (BoxInst pairwise form)."""
    import torch
    import torch.nn.functional as F
    p = torch.sigmoid(mask_logits)
    logp = F.logsigmoid(mask_logits)
    log1mp = F.logsigmoid(-mask_logits)
    C = evidence.shape[0]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    loss, npairs = 0.0, 0
    for dy, dx in offs:
        ev_sh = torch.roll(evidence, shifts=(dy * dilation, dx * dilation), dims=(1, 2))
        aff = torch.exp(-(evidence - ev_sh).abs().sum(0) / sigma_E)  # (H,W)
        p_sh = torch.roll(p, shifts=(dy * dilation, dx * dilation), dims=(0, 1))
        logp_sh = torch.roll(logp, shifts=(dy * dilation, dx * dilation), dims=(0, 1))
        log1mp_sh = torch.roll(log1mp, shifts=(dy * dilation, dx * dilation), dims=(0, 1))
        # log-prob that both pixels share the same label
        log_same = torch.logaddexp(logp + logp_sh, log1mp + log1mp_sh)
        w = (aff > affinity_thresh).float() * box_mask
        loss = loss + (-(log_same) * w).sum()
        npairs = npairs + w.sum()
    return loss / (npairs + 1e-6)


def box_projection_loss(mask_logits, boxes):
    """torch. Dice of row/col max-projection vs all-ones inside each box."""
    import torch
    p = torch.sigmoid(mask_logits)
    total = 0.0
    for (x0, y0, x1, y1) in boxes:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        crop = p[y0:y1, x0:x1]
        if crop.numel() == 0:
            continue
        for proj in (crop.max(0).values, crop.max(1).values):
            total = total + (1 - (2 * proj.sum()) / (proj.sum() + proj.numel() + 1e-6))
    return total / (2 * max(len(boxes), 1))


if __name__ == "__main__":
    # self-test: affinity high within a uniform-evidence ship, low across edge
    E = np.zeros((4, 32, 32), np.float32)
    E[:, 10:20, 10:20] = 5.0  # "ship" block in evidence space
    aff = evidence_affinity_np(E, dilation=1, sigma_E=1.0)
    interior = aff[:, 14, 14].mean()   # inside ship -> neighbors similar -> high
    edge = aff[:, 10, 10].mean()       # on ship boundary -> some low-affinity neighbors
    print("interior affinity %.3f (expect ~1), edge affinity %.3f (expect <1)" % (interior, edge))
    # box projection: a perfect mask filling the box -> ~1
    prob = np.zeros((32, 32), np.float32); prob[10:20, 10:20] = 1.0
    print("box projection (perfect fill) %.3f (expect ~1)" % box_projection_np(prob, (10, 10, 20, 20)))
    prob2 = np.zeros((32, 32), np.float32); prob2[12:15, 12:15] = 1.0  # under-fills
    print("box projection (under-fill) %.3f (expect <1)" % box_projection_np(prob2, (10, 10, 20, 20)))
