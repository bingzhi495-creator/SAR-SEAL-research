# -*- coding: utf-8 -*-
"""CPU head-level check for GeoFieldMask2FormerHead (mmdet env, e.g. base/3.2.0).

Builds the panoptic_head from each gf_r*_pseg config and verifies:
  A. the head module imports/registers; every config resolves its _base_ chain.
  B. byte-identical trainable param count across ALL six gen_mode arms (equal-param
     control at the HEAD level, not just the operator).
  C. _refine_features override is IDENTITY at init (gamma=0) for every arm -> the
     trained head starts byte-identical in OUTPUT to the carrier SealM2FHead refine
     (no ShapePrior step-0 divergence); and becomes ACTIVE once gamma!=0.

No backbone build / no weight download (only the panoptic_head is built).
"""
import sys

import torch
from mmengine.config import Config

sys.path.insert(0, 'E:/SAR/seal')
from mmdet.registry import MODELS  # noqa: E402
import carrier.seal_m2f_head        # noqa: E402,F401  (register SealM2FHead)
import carrier.geofield_m2f_head    # noqa: E402,F401  (register GeoFieldMask2FormerHead)

CFGS = {
    'R0_off': 'carrier/configs/gf_r0_pseg.py',
    'R1_capacity': 'carrier/configs/gf_r1_pseg.py',
    'R2_pos': 'carrier/configs/gf_r2_pseg.py',
    'R3_shape': 'carrier/configs/gf_r3_pseg.py',
    'R5_full': 'carrier/configs/gf_r5_pseg.py',
    'R6_full_rel': 'carrier/configs/gf_r6_pseg.py',
}


def build_head(path):
    cfg = Config.fromfile(path)
    head_cfg = cfg.model.panoptic_head
    # train_cfg/test_cfg live on the model in mmdet; the head reads them via kwargs
    head_cfg = dict(head_cfg)
    head_cfg.setdefault('train_cfg', cfg.model.get('train_cfg'))
    head_cfg.setdefault('test_cfg', cfg.model.get('test_cfg'))
    return MODELS.build(head_cfg)


def main():
    heads, counts = {}, {}
    for name, path in CFGS.items():
        h = build_head(path)
        heads[name] = h
        counts[name] = sum(p.numel() for p in h.parameters() if p.requires_grad)
        print(f'[build] {name:14s} type={type(h).__name__} gen_mode={h.gen_mode} '
              f'params={counts[name]}')

    # B. equal-param across all arms
    assert len(set(counts.values())) == 1, f'HEAD param counts differ: {counts}'
    print(f'\n[B] PASS equal-param at head level: every arm = {next(iter(counts.values()))} params')

    # C. WITHIN each head, geofield is IDENTITY at init (gamma=0): compare the head's
    #    _refine_features (geofield ON) vs the parent SealM2FHead._refine_features (geofield
    #    OFF) on the SAME weights. (Comparing across heads is wrong -- each head has its own
    #    random mask_refine init, so their base refine outputs legitimately differ.)
    from carrier.seal_m2f_head import SealM2FHead  # noqa: E402
    c = heads['R5_full'].mask_embed[-1].out_features
    x = torch.randn(2, c, 16, 20)
    for name, h in heads.items():
        h.eval()
        geo_on = h._refine_features(x)                 # GeoFieldHead override (geofield applied)
        geo_off = SealM2FHead._refine_features(h, x)   # same weights, geofield bypassed
        d = (geo_on - geo_off).abs().max().item()
        assert torch.allclose(geo_on, geo_off, atol=1e-6), f'{name}: geofield not identity @ init (max|d|={d})'
    print('[C] PASS geofield IDENTITY @ init for every arm (head == carrier refine at step 0)')

    # activate R5 -> output must change vs its own base refine
    h = heads['R5_full']
    base = SealM2FHead._refine_features(h, x)
    with torch.no_grad():
        h.geofield.gamma_shape.fill_(1.0)
        h.geofield.gamma_pos.fill_(1.0)
        h.geofield.a.fill_(0.5)
    out_active = h._refine_features(x)
    assert not torch.allclose(out_active, base, atol=1e-5), 'R5 active arm is a no-op'
    print('[C] PASS R5 becomes active once gamma!=0')

    print('\nALL GEOFIELD HEAD CHECKS PASSED')


if __name__ == '__main__':
    main()
