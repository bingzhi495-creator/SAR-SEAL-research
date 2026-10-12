# -*- coding: utf-8 -*-
"""CPU-only KILL-GATE for the geometry-conditioned dynamic mask kernel ("GeoKernel").

QUESTION (decide BEFORE any GPU): can a small ORIENTED spatial kernel (k=5),
conditioned on a ship's geometry, fit MEDIUM SAR-ship silhouettes better than
Mask2Former's RANK-1 mask readout, at comparable effective parameters?

Mask2Former emits each instance mask as a rank-1 readout
    mask = einsum('bqc,bchw->bqhw', mask_embed, E)
where E (B,256,H/4,W/4) is the SHARED pixel-decoder mask_feature and mask_embed
is one 256-D per-query vector. Zero spatial receptive field per instance.

We freeze A0 (runs/a0_pseg_seed0/iter_8352.pth), extract the SHARED E per image,
take ~20 medium inshore GT instances, and for EACH instance FIT two readouts to
the GT mask (at E's H/4 grid) under the SAME optimizer budget:

  (a) RANK-1   : sigmoid(v . E + b),  v in R^256, b in R                 (257 params)
  (b) GEOKERNEL: sigmoid( DWConv_oriented( LowRankMix(E) ) ), where a k=5
                 depthwise kernel is built from a small ORIENTED steerable basis
                 (rotated elongated Gaussian + Gaussian-derivative) whose
                 mixing weights are free, applied to a low-rank (r) pointwise
                 mix of E's 256 channels. Free-param count kept COMPARABLE to 257.

TESTS
  A (make-or-break) : mean boundary-F@2px(geo) - boundary-F@2px(rank1) over 20.
                      PASS >= +0.05 ; KILL <= +0.01 ; else GREY.
  B (geometry sens) : IoU(out@theta, out@theta+90) ; geometry must change mask.
                      PASS <= 0.6 ; KILL >= 0.9 ; else GREY.
  C (anti-collapse) : best rank-1 linear approx of the GeoKernel input->output map;
                      relative residual. PASS >= 0.20 ; KILL < 0.05 ; else GREY.

GO only if A PASS and B PASS and C PASS.  CPU ONLY.  Real outputs only.

Usage:
  set CUDA_VISIBLE_DEVICES= & set KMP_DUPLICATE_LIB_OK=TRUE & set PYTHONPATH=E:/SAR/seal
  E:/sar_envs/env_seal_gpu/Scripts/python.exe diagnostics/geokernel_killgate.py
"""
from __future__ import annotations
import os, sys, glob, argparse
os.environ.setdefault("PYTHONPATH", "E:/SAR/seal")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU ONLY -- a GPU job is running.
sys.path.insert(0, "E:/SAR/seal")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO

BASE = "E:/SAR/seal"
CFG = os.path.join(BASE, "carrier/configs/a0_m2f_r50_pseg.py")
CKPT = os.path.join(BASE, "runs/a0_pseg_seed0/iter_8352.pth")
GT_JSON = os.path.join(BASE, "data/pseg_ssdd/annotations/test.json")
IMG_DIR = r"E:\PSeg_SSDD\voc_style\JPEGImages_test"

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


# --------------------------------------------------------------------------- #
#  GT geometry from binary mask moments                                       #
# --------------------------------------------------------------------------- #
def mask_geometry(mask: np.ndarray):
    """theta (rad, major-axis), aspect (major/minor), sqrt(area), (cx,cy) at the
    mask's own resolution. Uses 2nd central moments (robust on tiny masks)."""
    m = mask.astype(np.uint8)
    M = cv2.moments(m, binaryImage=True)
    area = M["m00"]
    if area < 1:
        return 0.0, 1.0, 0.0, (mask.shape[1] / 2, mask.shape[0] / 2)
    cx = M["m10"] / area
    cy = M["m01"] / area
    mu20 = M["mu20"] / area
    mu02 = M["mu02"] / area
    mu11 = M["mu11"] / area
    theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))   # major-axis angle
    common = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
    lam1 = 0.5 * (mu20 + mu02 + common)   # major eigenvalue
    lam2 = 0.5 * (mu20 + mu02 - common)   # minor eigenvalue
    aspect = float(np.sqrt(lam1 / max(lam2, 1e-6)))
    return float(theta), float(aspect), float(np.sqrt(area)), (float(cx), float(cy))


# --------------------------------------------------------------------------- #
#  Oriented steerable k=5 depthwise basis                                     #
# --------------------------------------------------------------------------- #
def oriented_basis(theta: float, aspect: float, k: int = 5):
    """A small ORIENTED steerable basis of k x k kernels aligned to `theta`,
    elongated by `aspect`. Returns a (NB, k, k) float tensor. The basis is:
      0: elongated oriented Gaussian (blob along major axis)
      1: 1st Gaussian-derivative across the minor axis (edge perpendicular to ship)
      2: 1st Gaussian-derivative along the major axis (bow/stern gradient)
      3: 2nd Gaussian-derivative across the minor axis (ridge/keel line)
    These are functions of geometry only (no free params); the FREE params are
    the mixing weights over this basis + a DC term (see GeoKernel)."""
    r = (k - 1) / 2.0
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    ct, st = np.cos(theta), np.sin(theta)
    # rotate coords into the ship frame: u = along major axis, v = across
    u = ct * xs + st * ys
    v = -st * xs + ct * ys
    su = max(aspect, 1.0) * 0.9 + 0.6      # longer sigma along major axis
    sv = 0.9                                # tight across the hull
    g = np.exp(-(u ** 2) / (2 * su ** 2) - (v ** 2) / (2 * sv ** 2))
    b0 = g
    b1 = -(v / sv ** 2) * g                 # d/dv  (edge across hull)
    b2 = -(u / su ** 2) * g                 # d/du  (gradient along hull)
    b3 = ((v ** 2 - sv ** 2) / sv ** 4) * g  # d2/dv2 (ridge line across hull)
    basis = np.stack([b0, b1, b2, b3], 0)
    # zero-mean the derivative filters; normalise each to unit L2
    for i in range(1, basis.shape[0]):
        basis[i] -= basis[i].mean()
    for i in range(basis.shape[0]):
        n = np.sqrt((basis[i] ** 2).sum())
        if n > 1e-8:
            basis[i] /= n
    return torch.from_numpy(basis.astype(np.float32))   # (4,k,k)


# --------------------------------------------------------------------------- #
#  Readout models                                                             #
# --------------------------------------------------------------------------- #
class Rank1(nn.Module):
    """Exactly Mask2Former's family: mask = sigmoid(v . E + b)."""
    def __init__(self, C):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(C))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, E):                       # E: (1,C,h,w)
        return torch.einsum("c,bchw->bhw", self.v, E) + self.b   # (1,h,w) logits

    def n_params(self):
        return self.v.numel() + self.b.numel()


class GeoKernel(nn.Module):
    """sigmoid( DWConv_oriented( LowRankMix(E) ) ).

    LowRankMix : 256 -> r pointwise mix (free: 256*r), produces r channels.
    DWConv     : each of the r channels convolved with a k=5 depthwise kernel
                 that is a GEOMETRY-CONDITIONED combination of the oriented basis
                 (free: per-channel basis-mix weights + bias), then summed.
    The kernel SHAPE/ORIENTATION is dictated by (theta, aspect) via the basis;
    only the mix weights are free -> genuinely oriented, not a free 5x5 filter.
    Free params kept COMPARABLE to rank-1's 257."""
    def __init__(self, C, theta, aspect, r=1, k=5):
        super().__init__()
        self.r = r
        self.k = k
        self.register_buffer("basis", oriented_basis(theta, aspect, k))  # (NB,k,k)
        NB = self.basis.shape[0]
        # low-rank pointwise mix E(256)->r
        self.mix = nn.Parameter(torch.randn(r, C) * 0.02)   # r*C
        # per-(out r-channel) basis mixing weights + per-channel bias
        self.coef = nn.Parameter(torch.zeros(r, NB))        # r*NB
        self.chan_b = nn.Parameter(torch.zeros(r))          # r
        self.out_w = nn.Parameter(torch.ones(r) / r)        # r  (combine r maps)
        self.b = nn.Parameter(torch.zeros(1))               # 1

    def _kernels(self):
        # (r, k, k) geometry-conditioned depthwise kernels
        return torch.einsum("rn,nij->rij", self.coef, self.basis)

    def forward(self, E):                       # E: (1,C,h,w)
        x = torch.einsum("rc,bchw->brhw", self.mix, E)      # (1,r,h,w)
        ker = self._kernels().unsqueeze(1)                  # (r,1,k,k)
        y = F.conv2d(x, ker, padding=self.k // 2, groups=self.r)  # (1,r,h,w)
        y = y + self.chan_b.view(1, -1, 1, 1)
        out = torch.einsum("r,brhw->bhw", self.out_w, y) + self.b
        return out                                          # (1,h,w) logits

    def n_params(self):
        return (self.mix.numel() + self.coef.numel() + self.chan_b.numel()
                + self.out_w.numel() + self.b.numel())


# --------------------------------------------------------------------------- #
#  Losses / metrics                                                           #
# --------------------------------------------------------------------------- #
def bce_dice(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def fit(model, E, gt, steps=400, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = bce_dice(model(E).squeeze(0), gt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(loss.item())


def boundary_f(gt: np.ndarray, pred: np.ndarray, tol: int = 2) -> float:
    """Boundary-F at `tol`px: contour precision/recall within a tol-px band."""
    def band(m):
        m = m.astype(np.uint8)
        if m.sum() == 0:
            return m
        k = np.ones((3, 3), np.uint8)
        return m - cv2.erode(m, k, iterations=tol)

    def contour(m):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return np.zeros((0, 2), int)
        return np.concatenate([c.reshape(-1, 2) for c in cnts], 0)

    gc, pc = contour(gt), contour(pred)
    if len(gc) == 0 and len(pc) == 0:
        return 1.0
    if len(gc) == 0 or len(pc) == 0:
        return 0.0
    gb, pb = band(gt), band(pred)
    H, W = gt.shape
    pc_i, gc_i = pc.astype(int), gc.astype(int)
    prec = gb[np.clip(pc_i[:, 1], 0, H - 1), np.clip(pc_i[:, 0], 0, W - 1)].mean()
    rec = pb[np.clip(gc_i[:, 1], 0, H - 1), np.clip(gc_i[:, 0], 0, W - 1)].mean()
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Feature extraction: shared E + GT at E resolution                          #
# --------------------------------------------------------------------------- #
def build_model():
    init_default_scope("mmdet")
    m = init_detector(CFG, CKPT, device=DEV)
    m.eval()
    return m


def extract_E(model, img_path):
    """Run A0 forward on CPU; return the shared mask_feature E (1,256,h,w),
    img_shape (post-resize HxW), pad_shape (HxW), and the model result."""
    cap = {}
    def hk(mod, i, o):
        cap["E"] = o.detach().clone()
    h = model.panoptic_head.pixel_decoder.mask_feature.register_forward_hook(hk)
    res = inference_detector(model, img_path)
    h.remove()
    meta = res.metainfo
    return cap["E"], meta, res


def gt_at_E(gt_ori: np.ndarray, meta, E):
    """Map an ori-resolution GT mask to E's grid the way the model does:
    ori --resize(scale_factor)--> img_shape, placed into a pad_shape canvas,
    then downsampled to E's (h,w). Returns float tensor (h,w) in {0,1}."""
    ih, iw = meta["img_shape"]            # post-resize H,W
    ph, pw = meta["pad_shape"][:2]        # padded H,W (E = pad/4)
    eh, ew = E.shape[-2:]
    # resize ori GT to img_shape
    g = cv2.resize(gt_ori.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((ph, pw), np.uint8)
    canvas[:ih, :iw] = g
    t = torch.from_numpy(canvas.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(eh, ew), mode="bilinear", align_corners=False)
    return (t[0, 0] > 0.5).float()


# --------------------------------------------------------------------------- #
def coco_mask(coco, ann, h, w):
    from pycocotools import mask as mu
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        r = seg
    else:
        r = mu.merge(mu.frPyObjects(seg, h, w))
    return mu.decode(r).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="num medium instances")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--rank", type=int, default=1, help="GeoKernel low-rank r")
    args = ap.parse_args()

    print("=" * 74)
    print("GeoKernel CPU KILL-GATE  (frozen A0, CPU only)")
    print("=" * 74)

    coco = COCO(GT_JSON)
    from protocol.datasets import _subset_stems, PSEG_ROOT
    ins, _ = _subset_stems(PSEG_ROOT)
    name2id = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i
               for i, im in coco.imgs.items()}
    ins_ids = [name2id[s] for s in ins if s in name2id]

    # collect medium inshore instances grouped by image (one forward per image)
    by_img = {}
    for iid in ins_ids:
        im = coco.imgs[iid]
        for a in coco.imgToAnns.get(iid, []):
            if a.get("iscrowd"):
                continue
            if 1024 <= a["area"] < 9216:     # COCO medium
                by_img.setdefault(iid, []).append(a)
    img_order = sorted(by_img.keys())        # deterministic

    model = build_model()

    # ---- verification: the rank-1 einsum with the EXTRACTED E reproduces the
    # model's OWN final-layer mask_pred EXACTLY (the tensor the model upsamples,
    # thresholds and emits). We hook the head's predict() to grab its returned
    # mask_pred (= F.interpolate(einsum(mask_embed_final, mask_features))) and
    # rebuild it from our captured (E, mask_embed_final). Bit-exact => E is the
    # exact tensor multiplied in the Mask2Former einsum. ----
    print("\n[VERIFY] rank-1 einsum with extracted E == model's own mask_pred")
    me_caps = []
    def me_hook(mod, i, o):
        me_caps.append(o.detach().clone())
    headout = {}
    orig_predict = model.panoptic_head.predict
    def wrapped_predict(x, bds):
        out = orig_predict(x, bds)
        headout["mp"] = out[1].detach().clone()   # (B,Q,Hin,Win) upsampled logits
        return out
    model.panoptic_head.predict = wrapped_predict
    h2 = model.panoptic_head.mask_embed.register_forward_hook(me_hook)
    vstem = coco.imgs[img_order[0]]["file_name"]
    vpath = os.path.join(IMG_DIR, vstem)
    me_caps.clear()
    E_v, meta_v, res_v = extract_E(model, vpath)
    h2.remove()
    model.panoptic_head.predict = orig_predict
    me_final = me_caps[-1]                                   # (1,Q,256)
    mp = headout["mp"]                                       # head's upsampled logits
    my_logit = torch.einsum("bqc,bchw->bqhw", me_final, E_v)  # native-res einsum
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

    # ---- gather ~N medium instances, one image at a time ----
    print(f"\n[FIT] up to {args.n} medium inshore instances "
          f"(opt steps={args.steps}, GeoKernel r={args.rank})")
    records = []        # per instance: dict with bf_rank1, bf_geo, models, E, gt
    for iid in img_order:
        if len(records) >= args.n:
            break
        im = coco.imgs[iid]; h, w = im["height"], im["width"]
        path = os.path.join(IMG_DIR, im["file_name"])
        E, meta, _ = extract_E(model, path)
        C = E.shape[1]
        for a in by_img[iid]:
            if len(records) >= args.n:
                break
            gm_ori = coco_mask(coco, a, h, w)
            gt = gt_at_E(gm_ori, meta, E)               # (h,w) float at E grid
            if gt.sum() < 4:                            # too tiny at H/4
                continue
            theta, aspect, sqrt_area, (cx, cy) = mask_geometry(gt.numpy())

            r1 = Rank1(C)
            l1 = fit(r1, E, gt, steps=args.steps)
            gk = GeoKernel(C, theta, aspect, r=args.rank)
            lg = fit(gk, E, gt, steps=args.steps)

            with torch.no_grad():
                p1 = (torch.sigmoid(r1(E).squeeze(0)) > 0.5).numpy().astype(np.uint8)
                pg = (torch.sigmoid(gk(E).squeeze(0)) > 0.5).numpy().astype(np.uint8)
            gtn = gt.numpy().astype(np.uint8)
            bf1 = boundary_f(gtn, p1, tol=2)
            bfg = boundary_f(gtn, pg, tol=2)
            records.append(dict(iid=iid, theta=theta, aspect=aspect,
                                sqrt_area=sqrt_area, bf1=bf1, bfg=bfg,
                                E=E, gt=gt, gk=gk, n_r1=r1.n_params(),
                                n_gk=gk.n_params()))
            print(f"   inst {len(records):2d} img{iid:>4} area^.5={sqrt_area:5.1f} "
                  f"theta={np.degrees(theta):+6.1f} asp={aspect:4.2f} "
                  f"| BF rank1={bf1:.3f} geo={bfg:.3f} (d={bfg-bf1:+.3f})")

    n = len(records)
    assert n >= 5, f"only {n} instances fitted"
    n_r1 = records[0]["n_r1"]; n_gk = records[0]["n_gk"]

    # ===================== TEST A: boundary-F ===================== #
    bf1 = np.array([r["bf1"] for r in records])
    bfg = np.array([r["bfg"] for r in records])
    dA = float(bfg.mean() - bf1.mean())
    verdA = "PASS" if dA >= 0.05 else ("KILL" if dA <= 0.01 else "GREY")

    # ===================== TEST B: geometry sensitivity ===================== #
    # For a few instances, GeoKernel output @theta vs @(theta+90deg). Rebuild the
    # SAME fitted GeoKernel but swap its oriented basis to theta+90; compare masks.
    iou_bs = []
    for r in records[: min(8, n)]:
        gk = r["gk"]; E = r["E"]
        with torch.no_grad():
            out_g = (torch.sigmoid(gk(E).squeeze(0)) > 0.5)
            gk_rot = GeoKernel(E.shape[1], r["theta"] + np.pi / 2, r["aspect"],
                               r=args.rank)
            gk_rot.load_state_dict(gk.state_dict())   # same free params...
            gk_rot.basis = oriented_basis(r["theta"] + np.pi / 2, r["aspect"],
                                          gk.k)        # ...but rotated geometry
            out_r = (torch.sigmoid(gk_rot(E).squeeze(0)) > 0.5)
        inter = (out_g & out_r).sum().item()
        union = (out_g | out_r).sum().item()
        iou_bs.append(inter / union if union else 1.0)
    iouB = float(np.mean(iou_bs))
    verdB = "PASS" if iouB <= 0.6 else ("KILL" if iouB >= 0.9 else "GREY")

    # ===================== TEST C: anti-collapse ===================== #
    # Best single rank-1 linear approx of GeoKernel's input(E)->output(logit) map.
    # Stack pixels across a few instances: y = logit_geo (pre-sigmoid), X = E^T.
    # Fit y ~= X w + c by least squares; relative residual = ||y-Xw-c|| / ||y-mean||.
    res_rel = []
    for r in records[: min(10, n)]:
        gk = r["gk"]; E = r["E"]
        with torch.no_grad():
            y = gk(E).squeeze(0).reshape(-1).numpy()           # (h*w,)
        X = E[0].reshape(E.shape[1], -1).numpy().T             # (h*w, C)
        Xa = np.concatenate([X, np.ones((X.shape[0], 1))], 1)  # + bias
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
    print(f"  RANK-1   : {n_r1}")
    print(f"  GEOKERNEL: {n_gk}   (low-rank r={args.rank}, basis=4, k=5)")
    print("=" * 74)
    print(f"VERDICT BLOCK   (n_instances={n})")
    print("-" * 74)
    print(f"  TEST A  boundary-F@2px(geo) - (rank1)")
    print(f"          rank1={bf1.mean():.4f}  geo={bfg.mean():.4f}  delta={dA:+.4f}"
          f"   [{verdA}]  (PASS>=+0.05, KILL<=+0.01)")
    print(f"  TEST B  geometry sensitivity  IoU(out@theta, out@theta+90)")
    print(f"          mean IoU={iouB:.4f}"
          f"   [{verdB}]  (PASS<=0.60, KILL>=0.90)")
    print(f"  TEST C  anti-collapse  rel. residual of best rank-1 lin. approx")
    print(f"          rel_resid={rC:.4f}"
          f"   [{verdC}]  (PASS>=0.20, KILL<0.05)")
    print("-" * 74)
    print(f"  OVERALL: {'GO' if GO else 'NO-GO'}   "
          f"(GO iff A=PASS and B=PASS and C=PASS)")
    print("=" * 74)


if __name__ == "__main__":
    main()
