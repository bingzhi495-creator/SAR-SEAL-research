# -*- coding: utf-8 -*-
"""BPR step-3: small UNet boundary-patch refiner + training loop.

Input 2 x 128 x 128 ([image_gray/255, coarse_mask]), output 1 x 128 x 128 mask
logits. Patches are stored at 64x64 (see bpr_patches.py) and bicubically
upscaled x2 here for sub-pixel supervision (GT binary -> soft edge targets).

UNet: 4 down / 4 up, base 32ch, widths (32,64,128,192,192) ~= 2.4M params.
Loss: BCE-with-logits + soft Dice. AdamW 1e-3 cosine->1e-5, bs 64, 20 epochs.
Heartbeat: touches --alive every 50 iters. Auto-resume from refiner_last.pth.

Train (GPU, from E:/SAR/seal, PYTHONPATH=E:/SAR/seal):
  python carrier/bpr_refiner.py --patches runs/bpr/patches --out runs/bpr
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNetRefiner(nn.Module):
    """4-down/4-up UNet, in 2ch (gray + coarse mask), out 1ch logits."""

    def __init__(self, in_ch=2, widths=(32, 64, 128, 192, 192)):
        super().__init__()
        w = widths
        self.enc0 = DoubleConv(in_ch, w[0])
        self.enc1 = DoubleConv(w[0], w[1])
        self.enc2 = DoubleConv(w[1], w[2])
        self.enc3 = DoubleConv(w[2], w[3])
        self.enc4 = DoubleConv(w[3], w[4])          # bottleneck (after 4th pool)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DoubleConv(w[4] + w[3], w[2])
        self.dec2 = DoubleConv(w[2] + w[2], w[1])
        self.dec1 = DoubleConv(w[1] + w[1], w[0])
        self.dec0 = DoubleConv(w[0] + w[0], w[0])
        self.head = nn.Conv2d(w[0], 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear',
                             align_corners=False)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self._up(e4, e3), e3], 1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], 1))
        d0 = self.dec0(torch.cat([self._up(d1, e0), e0], 1))
        return self.head(d0)


def bce_dice_loss(logits, target):
    """BCE-with-logits + soft Dice; target may be soft in [0,1]."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    dice = 1.0 - (num / den).mean()
    return bce + dice


def upscale_triplet(img_u8, coarse_u8, target_u8, out_hw):
    """x2 bicubic (image) / bilinear-clipped (masks) upscale to out_hw."""
    hw = (out_hw, out_hw)
    im = cv2.resize(img_u8, hw, interpolation=cv2.INTER_CUBIC)
    im = np.clip(im.astype(np.float32) / 255.0, 0.0, 1.0)
    co = cv2.resize(coarse_u8.astype(np.float32), hw,
                    interpolation=cv2.INTER_LINEAR)
    co = np.clip(co, 0.0, 1.0)
    if target_u8 is None:
        return im, co, None
    tg = cv2.resize(target_u8.astype(np.float32), hw,
                    interpolation=cv2.INTER_LINEAR)
    tg = np.clip(tg, 0.0, 1.0)
    return im, co, tg


class PatchSet(Dataset):
    def __init__(self, patch_dir, idx, out_hw=128, augment=False, seed=0):
        shards = sorted(glob.glob(os.path.join(patch_dir, 'shard_*.npz')))
        assert shards, 'no shards in %s' % patch_dir
        imgs, cos, tgs = [], [], []
        for s in shards:
            d = np.load(s)
            imgs.append(d['img'])
            cos.append(d['coarse'])
            tgs.append(d['target'])
        self.img = np.concatenate(imgs)
        self.coarse = np.concatenate(cos)
        self.target = np.concatenate(tgs)
        self.idx = idx
        self.out_hw = out_hw
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        a, b, c = self.img[j], self.coarse[j], self.target[j]
        if self.augment:
            if self.rng.rand() < 0.5:
                a, b, c = a[:, ::-1], b[:, ::-1], c[:, ::-1]
            if self.rng.rand() < 0.5:
                a, b, c = a[::-1], b[::-1], c[::-1]
            k = self.rng.randint(4)
            if k:
                a, b, c = np.rot90(a, k), np.rot90(b, k), np.rot90(c, k)
            a, b, c = np.ascontiguousarray(a), np.ascontiguousarray(b), \
                np.ascontiguousarray(c)
        im, co, tg = upscale_triplet(a, b, c, self.out_hw)
        x = torch.from_numpy(np.stack([im, co]))
        y = torch.from_numpy(tg[None])
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--alive', default='runs/_alive')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    last_p = os.path.join(args.out, 'refiner_last.pth')
    best_p = os.path.join(args.out, 'refiner.pth')

    # fixed split
    probe = PatchSet(args.patches, idx=np.arange(1), augment=False)
    n = len(probe.img)
    perm = np.random.RandomState(123).permutation(n)
    n_val = max(1, int(n * args.val_frac))
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    tr = PatchSet(args.patches, tr_idx, augment=True, seed=args.seed)
    va = PatchSet(args.patches, va_idx, augment=False)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=0,
                    drop_last=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=0)

    model = UNetRefiner().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print('BPR_REFINER params=%d device=%s train=%d val=%d' %
          (n_par, dev, len(tr), len(va)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)
    start_ep, best_val = 0, math.inf
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_ep = ck['epoch'] + 1
        best_val = ck.get('best_val', math.inf)
        print('BPR_REFINER resume epoch=%d best_val=%.4f' % (start_ep, best_val),
              flush=True)

    for ep in range(start_ep, args.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = bce_dice_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            seen += x.size(0)
            if it % 50 == 0:
                try:
                    with open(args.alive, 'w') as f:
                        f.write(str(int(time.time())))
                except OSError:
                    pass
        sched.step()
        model.eval()
        vtot, vseen = 0.0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                vtot += bce_dice_loss(model(x), y).item() * x.size(0)
                vseen += x.size(0)
        vl_loss = vtot / max(vseen, 1)
        print('BPR_TRAIN epoch=%d train_loss=%.4f val_loss=%.4f lr=%.2e secs=%.0f'
              % (ep, tot / max(seen, 1), vl_loss, sched.get_last_lr()[0],
                 time.time() - t0), flush=True)
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(dict(model=model.state_dict(), params=n_par,
                            val_loss=vl_loss, epoch=ep), best_p)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, best_val=best_val),
                   last_p)
    print('BPR_REFINER_DONE params=%d best_val=%.4f out=%s' %
          (n_par, best_val, best_p), flush=True)


if __name__ == '__main__':
    main()
