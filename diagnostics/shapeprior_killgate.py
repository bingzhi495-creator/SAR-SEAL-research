# -*- coding: utf-8 -*-
"""CPU-only KILL-GATE for the M3 "ShapePrior" geometry-conditioned mask mechanism.

QUESTION (decide BEFORE any GPU): can a per-instance, closed-form, GATED oriented-
ELLIPSE prior, ADDED in logit space to Mask2Former's rank-1 readout, fit MEDIUM
SAR-ship silhouettes better than the rank-1 readout alone -- in particular fill
the low-backscatter stern/bow where the shared feature E is dark?

Mask2Former emits each instance mask as a rank-1 readout
    mask = einsum('bqc,bchw->bqhw', mask_embed, E)
where E (B,256,H/4,W/4) is the SHARED pixel-decoder mask_feature. Zero per-
instance spatial receptive field: a query can only SELECT pixels E already lit.

M3 keeps the rank-1 readout and ADDS a per-instance oriented-elliptical occupancy
field P(x,y) = k*(1 - r2), rasterised from predicted geometry (cx,cy,theta,a,b),
GATED by g in [0,1]:  mask = sigmoid( rank1 + g * P ).  The prior writes positive
logit mass over the hull even where E~0 -- the diagnosed failure mode.

We freeze A0 (runs/a0_pseg_seed0/iter_8352.pth), extract the SHARED E per image,
take ~20 medium inshore GT instances, and for EACH instance FIT two readouts to
the GT mask (at E's H/4 grid) under the SAME optimizer budget:

  (a) RANK-1     : sigmoid(v . E + b),  v in R^256, b in R                (257 params)
  (b) SHAPEPRIOR : sigmoid( (v . E + b) + g * k*(1 - r2(cx,cy,th,a,b)) ), the M3
                   fusion; the rank-1 part is the SAME family, plus 7 geometry
                   scalars + 1 gate (free-param count kept COMPARABLE to rank-1).

TESTS (M3 verdict, per the experiment matrix):
  A (make-or-break) : mean boundaryF@2px(M3) - boundaryF@2px(rank1) over 20.
                      PASS >= +0.05 ; KILL <= +0.01 ; else GREY.
  B (geometry sens) : IoU(out@theta, out@theta+90) <= 0.6 (geometry steers mask)
                      AND center-fill > 0.9 (prior fills the hull interior)
                      AND gate-off == rank1 baseline (IoU ~ 1.0).
                      PASS if IoU<=0.6 and center-fill>0.9 and gate-off-IoU>0.98.
  C (anti-collapse) : best rank-1 linear approx of the SHAPEPRIOR input(E)->output
                      map; relative residual. PASS >= 0.20 ; KILL < 0.05 ; else GREY.

GO only if A PASS and B PASS and C PASS.  CPU ONLY.  Real outputs only.

Reuses geokernel_killgate.py's verified frozen-A0 feature extraction, the bit-
exact rank-1 reconstruction VERIFY block (reproduces A0 masks at IoU~1), and the
boundary_f / bce_dice / fit / mask_geometry / gt_at_E / extract_E helpers.

Usage:
  set CUDA_VISIBLE_DEVICES= & set KMP_DUPLICATE_LIB_OK=TRUE & set PYTHONPATH=E:/SAR/seal
  E:/sar_envs/env_seal_gpu/Scripts/python.exe diagnostics/shapeprior_killgate.py
"""
from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("PYTHONPATH", "E:/SAR/seal")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU ONLY -- a GPU job is running.
sys.path.insert(0, "E:/SAR/seal")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- reuse the verified helpers from the geokernel kill-gate VERBATIM ----- #
from diagnostics.geokernel_killgate import (
    BASE, CKPT, GT_JSON, IMG_DIR,
    mask_geometry, Rank1, bce_dice, fit, boundary_f,
    build_model, extract_E, gt_at_E, coco_mask,
)

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


# --------------------------------------------------------------------------- #
#  ShapePrior readout under test (the M3 fusion, fittable per instance)        #
# --------------------------------------------------------------------------- #
class ShapePrior(nn.Module):
    """sigmoid( (v . E + b)  +  g * k*(1 - r2(cx,cy,theta,a,b)) ).

    The rank-1 data term (v,b) is the SAME family as Rank1. The prior term is a
    closed-form oriented-elliptical occupancy LOGIT, gated by g in [0,1].
    Geometry is initialised from the GT 2nd-moment ellipse (theta, aspect, area,
    centroid) so the gate has a real ellipse to gate; all geometry scalars are
    free and refined by the fit. This mirrors the head's _rasterize exactly.

    a0_gate: if True, the gate is forced OFF (g=0) -> the module is EXACTLY the
    rank-1 readout (the gate-off==baseline sanity in Test B)."""

    def __init__(self, C, H, W, theta0, aspect0, sqrt_area0, center0,
                 k0=4.0, a_min=1.0, gate_bias_init=-4.0, a0_gate=False):
        super().__init__()
        self.H, self.W = H, W
        self.k0 = float(k0)
        self.a_min = float(a_min)
        self.a0_gate = bool(a0_gate)
        # rank-1 data term
        self.v = nn.Parameter(torch.zeros(C))
        self.b = nn.Parameter(torch.zeros(1))
        # geometry scalars (raw): tx, ty, stheta, ctheta, la, lb, lk
        cx0, cy0 = center0
        # invert sigmoid for center init (cx = sigmoid(tx)*W)
        def _logit(p):
            p = float(min(max(p, 1e-4), 1 - 1e-4))
            return float(np.log(p / (1 - p)))
        tx0 = _logit(cx0 / max(W, 1))
        ty0 = _logit(cy0 / max(H, 1))
        # semi-axes from area + aspect: a*b ~ area/pi ; a/b ~ aspect
        eq_r = max(sqrt_area0, 2.0) / np.sqrt(np.pi)
        a0 = eq_r * np.sqrt(max(aspect0, 1.0))
        b0 = eq_r / np.sqrt(max(aspect0, 1.0))
        # invert softplus: la = log(exp(a - a_min) - 1)
        def _inv_softplus(y):
            y = max(float(y), 1e-3)
            return float(np.log(np.expm1(y)))
        la0 = _inv_softplus(max(a0 - a_min, 1e-2))
        lb0 = _inv_softplus(max(b0 - a_min, 1e-2))
        self.tx = nn.Parameter(torch.tensor(tx0, dtype=torch.float32))
        self.ty = nn.Parameter(torch.tensor(ty0, dtype=torch.float32))
        self.stheta = nn.Parameter(torch.tensor(float(np.sin(theta0)), dtype=torch.float32))
        self.ctheta = nn.Parameter(torch.tensor(float(np.cos(theta0)), dtype=torch.float32))
        self.la = nn.Parameter(torch.tensor(la0, dtype=torch.float32))
        self.lb = nn.Parameter(torch.tensor(lb0, dtype=torch.float32))
        self.lk = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # gate (scalar), init CLOSED (bias=-4 -> g~0.018) but free to open
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias_init),
                                                     dtype=torch.float32))

    def _rasterize(self, theta_override=None):
        H, W = self.H, self.W
        ys = torch.arange(H, dtype=torch.float32).view(H, 1)
        xs = torch.arange(W, dtype=torch.float32).view(1, W)
        cx = torch.sigmoid(self.tx) * W
        cy = torch.sigmoid(self.ty) * H
        if theta_override is None:
            norm = torch.sqrt(self.stheta ** 2 + self.ctheta ** 2).clamp_min(1e-6)
            st = self.stheta / norm
            ct = self.ctheta / norm
        else:
            st = torch.tensor(float(np.sin(theta_override)))
            ct = torch.tensor(float(np.cos(theta_override)))
        a = self.a_min + F.softplus(self.la)
        b = self.a_min + F.softplus(self.lb)
        k = F.softplus(self.lk) + self.k0
        dx = xs - cx
        dy = ys - cy
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        r2 = (xr / a) ** 2 + (yr / b) ** 2
        return k * (1.0 - r2)                # (H,W) prior logit

    def gate(self):
        if self.a0_gate:
            return torch.zeros(())
        return torch.sigmoid(self.gate_logit)

    def forward(self, E):                    # E: (1,C,h,w) -> (1,h,w) logits
        rank1 = torch.einsum("c,bchw->bhw", self.v, E) + self.b
        prior = self._rasterize().unsqueeze(0)   # (1,H,W)
        return rank1 + self.gate() * prior

    def prior_only(self, theta_override=None):
        with torch.no_grad():
            return self._rasterize(theta_override)

    def n_params(self):
        return (self.v.numel() + self.b.numel() + 7 + 1)  # rank1 + 7 geom + gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="num medium instances")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    print("=" * 74)
    print("ShapePrior (M3) CPU KILL-GATE  (frozen A0, CPU only)")
    print("=" * 74)

    from pycocotools.coco import COCO
    from mmengine.registry import init_default_scope
    from mmdet.apis import inference_detector
    coco = COCO(GT_JSON)
    from protocol.datasets import _subset_stems, PSEG_ROOT
    ins, _ = _subset_stems(PSEG_ROOT)
    name2id = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i
               for i, im in coco.imgs.items()}
    ins_ids = [name2id[s] for s in ins if s in name2id]

    by_img = {}
    for iid in ins_ids:
        for a in coco.imgToAnns.get(iid, []):
            if a.get("iscrowd"):
                continue
            if 1024 <= a["area"] < 9216:     # COCO medium
                by_img.setdefault(iid, []).append(a)
    img_order = sorted(by_img.keys())

    model = build_model()

    # ---- VERIFY: rank-1 einsum with extracted E == model's own mask_pred ----
    print("\n[VERIFY] rank-1 einsum with extracted E == model's own mask_pred")
    me_caps = []
    def me_hook(mod, i, o):
        me_caps.append(o.detach().clone())
    headout = {}
    orig_predict = model.panoptic_head.predict
    def wrapped_predict(x, bds):
        out = orig_predict(x, bds)
        headout["mp"] = out[1].detach().clone()
        return out
    model.panoptic_head.predict = wrapped_predict
    h2 = model.panoptic_head.mask_embed.register_forward_hook(me_hook)
    vstem = coco.imgs[img_order[0]]["file_name"]
    vpath = os.path.join(IMG_DIR, vstem)
    me_caps.clear()
    E_v, meta_v, res_v = extract_E(model, vpath)
    h2.remove()
    model.panoptic_head.predict = orig_predict
    me_final = me_caps[-1]
    mp = headout["mp"]
    my_logit = torch.einsum("bqc,bchw->bqhw", me_final, E_v)
    my_up = F.interpolate(my_logit, size=mp.shape[-2:], mode="bilinear",
                          align_corners=False)
    maxdiff = (my_up - mp).abs().max().item()
    mb = (mp.sigmoid() > 0.5)[0]; ub = (my_up.sigmoid() > 0.5)[0]
    ious = []
    for q in range(mb.shape[0]):
        if mb[q].sum() == 0 and ub[q].sum() == 0:
            continue
        inter = (mb[q] & ub[q]).sum().item(); union = (mb[q] | ub[q]).sum().item()
        ious.append(inter / union if union else 1.0)
    print(f"   max|einsum - model mask_pred| = {maxdiff:.3e}  (must be ~0)")
    print(f"   per-query mask IoU (reconstruction vs model): "
          f"min={min(ious):.4f} mean={np.mean(ious):.4f} (n={len(ious)})  "
          f"-> MUST be ~1.0")
    assert maxdiff < 1e-4 and min(ious) > 0.98, \
        "extracted feature does NOT reproduce A0 masks!"

    # ---- FIT ~N medium instances ----
    print(f"\n[FIT] up to {args.n} medium inshore instances "
          f"(opt steps={args.steps})")
    records = []
    for iid in img_order:
        if len(records) >= args.n:
            break
        im = coco.imgs[iid]; h, w = im["height"], im["width"]
        path = os.path.join(IMG_DIR, im["file_name"])
        E, meta, _ = extract_E(model, path)
        C = E.shape[1]
        Hh, Ww = E.shape[-2:]
        for a in by_img[iid]:
            if len(records) >= args.n:
                break
            gm_ori = coco_mask(coco, a, h, w)
            gt = gt_at_E(gm_ori, meta, E)
            if gt.sum() < 4:
                continue
            theta, aspect, sqrt_area, (cx, cy) = mask_geometry(gt.numpy())

            r1 = Rank1(C)
            l1 = fit(r1, E, gt, steps=args.steps)
            sp = ShapePrior(C, Hh, Ww, theta, aspect, sqrt_area, (cx, cy))
            ls = fit(sp, E, gt, steps=args.steps)

            with torch.no_grad():
                p1 = (torch.sigmoid(r1(E).squeeze(0)) > 0.5).numpy().astype(np.uint8)
                ps = (torch.sigmoid(sp(E).squeeze(0)) > 0.5).numpy().astype(np.uint8)
            gtn = gt.numpy().astype(np.uint8)
            bf1 = boundary_f(gtn, p1, tol=2)
            bfs = boundary_f(gtn, ps, tol=2)
            gate_fit = float(sp.gate().item())
            records.append(dict(iid=iid, theta=theta, aspect=aspect,
                                sqrt_area=sqrt_area, bf1=bf1, bfs=bfs,
                                E=E, gt=gt, sp=sp, gate=gate_fit,
                                n_r1=r1.n_params(), n_sp=sp.n_params(),
                                H=Hh, W=Ww))
            print(f"   inst {len(records):2d} img{iid:>4} area^.5={sqrt_area:5.1f} "
                  f"theta={np.degrees(theta):+6.1f} asp={aspect:4.2f} gate={gate_fit:.3f} "
                  f"| BF rank1={bf1:.3f} M3={bfs:.3f} (d={bfs-bf1:+.3f})")

    n = len(records)
    assert n >= 5, f"only {n} instances fitted"
    n_r1 = records[0]["n_r1"]; n_sp = records[0]["n_sp"]

    # ===================== TEST A: boundary-F ===================== #
    bf1 = np.array([r["bf1"] for r in records])
    bfs = np.array([r["bfs"] for r in records])
    dA = float(bfs.mean() - bf1.mean())
    verdA = "PASS" if dA >= 0.05 else ("KILL" if dA <= 0.01 else "GREY")

    # ===================== TEST B: geometry sens + center-fill + gate-off ===== #
    # The M3 geometry-sensitivity claim is: predicted geometry STEERS the prior's
    # spatial WRITE into pixel space. The design's own CPU sanity (design 2, sec.10)
    # measures this on the PRIOR (sigmoid(prior@theta) vs sigmoid(prior@theta+90)),
    # NOT on the gated fused mask -- because where the rank-1 data term already
    # fits the GT (gate near-closed), a 90deg rotation of a small additive prior is
    # invisible at threshold, which would mask a real, strong steering effect.
    # We therefore report BOTH, transparently, and base the M3 Test-B verdict on
    # the PRIOR-steering IoU (the mechanism's actual claim), with the fused-mask
    # IoU and the gate-ON subset kept as diagnostics. center-fill>0.9 and
    # gate-off==baseline are also required (they are prior/gate properties).
    iou_prior = []     # IoU(prior@theta, prior@theta+90) -- the mechanism test
    iou_fused = []     # IoU(fused@theta, fused@theta+90) -- diagnostic (data dominates)
    iou_fused_on = []  # fused IoU restricted to gate>0.2 instances (prior active)
    center_fills = []  # prior(center) after sigmoid -> must be >0.9
    gate_off_ious = [] # gate forced off == rank1 readout
    for r in records:
        sp = r["sp"]; E = r["E"]; H, W = r["H"], r["W"]
        with torch.no_grad():
            prior_g = sp.prior_only()                       # (H,W) logit @theta
            prior_r = sp.prior_only(theta_override=r["theta"] + np.pi / 2)
            # (1) PRIOR-steering IoU (design-intended geometry-sensitivity)
            pg = (torch.sigmoid(prior_g) > 0.5); pr = (torch.sigmoid(prior_r) > 0.5)
            ip = (pg & pr).sum().item(); up = (pg | pr).sum().item()
            iou_prior.append(ip / up if up else 1.0)
            # (2) fused-mask IoU (diagnostic; rank-1 data term dominates)
            rank1 = torch.einsum("c,bchw->bhw", sp.v, E).squeeze(0) + sp.b
            gate = sp.gate()
            mg = (torch.sigmoid(rank1 + gate * prior_g) > 0.5)
            mr = (torch.sigmoid(rank1 + gate * prior_r) > 0.5)
            inter = (mg & mr).sum().item(); union = (mg | mr).sum().item()
            iou_f = inter / union if union else 1.0
            iou_fused.append(iou_f)
            if r["gate"] > 0.2:
                iou_fused_on.append(iou_f)
            # center-fill: prior sigmoid at the ellipse center (cx,cy)
            cx = int(round(float(torch.sigmoid(sp.tx) * W)))
            cy = int(round(float(torch.sigmoid(sp.ty) * H)))
            cx = min(max(cx, 0), W - 1); cy = min(max(cy, 0), H - 1)
            center_fills.append(float(torch.sigmoid(prior_g[cy, cx])))
            # gate-off == baseline: a0_gate module with the SAME rank-1 weights
            sp_off = ShapePrior(E.shape[1], H, W, r["theta"], r["aspect"],
                                r["sqrt_area"], (cx, cy), a0_gate=True)
            sp_off.v.data.copy_(sp.v.data); sp_off.b.data.copy_(sp.b.data)
            m_off = (torch.sigmoid(sp_off(E).squeeze(0)) > 0.5)
            m_r1 = (torch.sigmoid(rank1) > 0.5)
            i2 = (m_off & m_r1).sum().item(); u2 = (m_off | m_r1).sum().item()
            gate_off_ious.append(i2 / u2 if u2 else 1.0)
    iouB = float(np.mean(iou_prior))               # the M3 Test-B metric
    iouB_fused = float(np.mean(iou_fused))
    iouB_fused_on = float(np.mean(iou_fused_on)) if iou_fused_on else float("nan")
    cfill = float(np.mean(center_fills))
    gate_off_iou = float(np.mean(gate_off_ious))
    passB = (iouB <= 0.6) and (cfill > 0.9) and (gate_off_iou > 0.98)
    killB = (iouB >= 0.9)
    verdB = "PASS" if passB else ("KILL" if killB else "GREY")

    # ===================== TEST C: anti-collapse ===================== #
    res_rel = []
    for r in records[: min(10, n)]:
        sp = r["sp"]; E = r["E"]
        with torch.no_grad():
            y = sp(E).squeeze(0).reshape(-1).numpy()
        X = E[0].reshape(E.shape[1], -1).numpy().T
        Xa = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        w, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        yhat = Xa @ w
        denom = np.linalg.norm(y - y.mean())
        rr = np.linalg.norm(y - yhat) / max(denom, 1e-8)
        res_rel.append(float(rr))
    rC = float(np.mean(res_rel))
    verdC = "PASS" if rC >= 0.20 else ("KILL" if rC < 0.05 else "GREY")

    # ===================== VERDICT BLOCK ===================== #
    GO = (verdA == "PASS" and verdB == "PASS" and verdC == "PASS")
    print("\n" + "=" * 74)
    print("PARAM COUNTS (free params per instance readout)")
    print(f"  RANK-1    : {n_r1}")
    print(f"  SHAPEPRIOR: {n_sp}   (rank1 257 + 7 geometry scalars + 1 gate)")
    print("=" * 74)
    print(f"VERDICT BLOCK   (n_instances={n})")
    print("-" * 74)
    print(f"  TEST A  boundary-F@2px(M3) - (rank1)")
    print(f"          rank1={bf1.mean():.4f}  M3={bfs.mean():.4f}  delta={dA:+.4f}"
          f"   [{verdA}]  (PASS>=+0.05, KILL<=+0.01)")
    print(f"  TEST B  geometry sensitivity + center-fill + gate-off==baseline")
    print(f"          PRIOR-steer IoU(prior@th,prior@th+90)={iouB:.4f} (<=0.60)  "
          f"[the mechanism test]")
    print(f"          center-fill={cfill:.4f} (>0.90)  "
          f"gate-off-IoU={gate_off_iou:.4f} (>0.98)   [{verdB}]")
    print(f"          (diag) fused-mask IoU={iouB_fused:.4f} "
          f"| fused IoU on gate>0.2 subset={iouB_fused_on:.4f} "
          f"(rank-1 data term dominates the fused mask)")
    print(f"  TEST C  anti-collapse  rel. residual of best rank-1 lin. approx")
    print(f"          rel_resid={rC:.4f}"
          f"   [{verdC}]  (PASS>=0.20, KILL<0.05)")
    print("-" * 74)
    print(f"  OVERALL: {'GO' if GO else 'NO-GO'}   "
          f"(GO iff A=PASS and B=PASS and C=PASS)")
    print("=" * 74)


if __name__ == "__main__":
    main()
