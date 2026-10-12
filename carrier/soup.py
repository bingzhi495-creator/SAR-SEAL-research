# -*- coding: utf-8 -*-
"""E1 -- Uniform weight soup over seeds s0/s2/s3 of c0_maskiou_v2.

NO RETRAINING. Operates purely on the three EXISTING seed checkpoints
(runs/c0_maskiou_v2/s{0,2,3}/iter_100872.pth -- identical training recipe,
different seed) and writes runs/c0_maskiou_v2/soup_s023/iter_100872.pth.

Every FLOATING-POINT parameter/buffer is uniformly averaged (1/3 each); the 53
non-float ``*.num_batches_tracked`` int64 buffers are copied verbatim from s0
(verified byte-identical across the three seeds). s1 is intentionally excluded
(no iter_100872.pth -> different basin distance).

DEVIATION FROM THE ORIGINAL SPEC (required to pass the spec's own IDENTITY GATE)
--------------------------------------------------------------------------------
The spec sketched ``avg = (t0.float()+t2.float()+t3.float())/3``, i.e. float32
accumulation. That FAILS the spec's mandatory identity gate: empirically
``(a+a+a)/3`` in float32 is NOT bit-exact (146688 / 1e6 random float32 elements
differ), so souping three identical inputs would not reproduce the input
bit-for-bit. We therefore accumulate in float64 and narrow back to the original
dtype. This is bit-exact for the identity case because f32->f64 widening is
exact, 3a and (3a)/3 are exact in f64, and narrowing an originally-f32 value
f64->f32 is exact. For genuinely different seeds it is simply a more accurate
mean (max |avg - reference_mean| well under 1e-6). Run this file directly to
execute all four CPU sanity checks and write the soup.

Usage:  python carrier/soup.py            # checks + write soup
        python carrier/soup.py --dry-run  # checks only, do not write
CPU-ONLY: forces CUDA_VISIBLE_DEVICES='' so the busy GPU is never touched.
"""
from __future__ import annotations

import os
# Force CPU-only BEFORE torch import: the GPU is busy with another eval.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
from typing import Dict, List

import torch

ROOT = 'E:/SAR/seal'
BASE = ROOT + '/runs/c0_maskiou_v2'
SEED_CKPTS = [BASE + '/s0/iter_100872.pth',
              BASE + '/s2/iter_100872.pth',
              BASE + '/s3/iter_100872.pth']
OUT = BASE + '/soup_s023/iter_100872.pth'
CFG = ROOT + '/carrier/configs/c0_maskiou_v2_m2f_hrsid.py'

StateDict = Dict[str, torch.Tensor]


def load_state_dict(path: str) -> StateDict:
    return torch.load(path, map_location='cpu')['state_dict']


def make_soup(state_dicts: List[StateDict]) -> StateDict:
    """Uniform average of float tensors (in float64); copy non-float from [0]."""
    n = len(state_dicts)
    avg: StateDict = {}
    for k, t0 in state_dicts[0].items():
        if t0.is_floating_point():
            acc = state_dicts[0][k].double()
            for sd in state_dicts[1:]:
                acc = acc + sd[k].double()
            avg[k] = (acc / n).to(t0.dtype)
        else:
            avg[k] = t0.clone()          # copy from s0 (byte-identical seeds)
    return avg


# --------------------------------------------------------------------------- #
#                               CPU sanity checks                             #
# --------------------------------------------------------------------------- #
def check1_keys(sds: List[StateDict]) -> None:
    ksets = [set(sd) for sd in sds]
    assert ksets[0] == ksets[1] == ksets[2], 'key sets differ across seeds'
    assert len(ksets[0]) == 618, f'expected 618 keys, got {len(ksets[0])}'
    nbt = [k for k in sds[0] if k.endswith('num_batches_tracked')]
    assert len(nbt) == 53, f'expected 53 num_batches_tracked, got {len(nbt)}'
    for k in nbt:
        assert torch.equal(sds[0][k], sds[1][k]) and \
               torch.equal(sds[0][k], sds[2][k]), \
               f'num_batches_tracked mismatch at {k}'
    print(f'[check1] PASS  keys=618 identical across seeds; '
          f'{len(nbt)} num_batches_tracked all byte-equal across seeds')


def check2_mean(sds: List[StateDict], avg: StateDict) -> None:
    # Independent reference mean via float64 stack on a real conv weight.
    sample = next(k for k in sds[0]
                  if sds[0][k].is_floating_point() and sds[0][k].dim() >= 2)
    ref = torch.stack([sd[sample].double() for sd in sds]).mean(0)
    diff = (avg[sample].double() - ref).abs().max().item()
    assert diff < 1e-6, f'{sample}: |avg-mean|={diff} >= 1e-6'
    print(f'[check2] PASS  sample key {sample!r}  '
          f'max|avg-elementwise_mean|={diff:.3e} < 1e-6')


def check3_identity(sd0: StateDict) -> None:
    ident = make_soup([sd0, sd0, sd0])
    bad = [k for k in sd0
           if not (ident[k].dtype == sd0[k].dtype and torch.equal(ident[k], sd0[k]))]
    assert not bad, f'IDENTITY GATE FAILED at {len(bad)} keys, e.g. {bad[:3]}'
    print(f'[check3] PASS  IDENTITY GATE: soup([s0,s0,s0]) reproduces s0 '
          f'bit-for-bit on all {len(sd0)} keys')


def check4_build(avg: StateDict) -> None:
    from mmengine.config import Config
    from mmengine.registry import MODELS, init_default_scope
    from mmengine.utils import import_modules_from_strings
    try:
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
    except Exception:
        import mmdet.models  # noqa: F401  (registers modules)
        init_default_scope('mmdet')
    cfg = Config.fromfile(CFG)
    ci = cfg.get('custom_imports', None)
    if ci:
        import_modules_from_strings(**ci)
    model = MODELS.build(cfg.model)
    res = model.load_state_dict(avg, strict=True)   # raises on any mismatch
    missing = list(getattr(res, 'missing_keys', []))
    unexpected = list(getattr(res, 'unexpected_keys', []))
    assert not missing and not unexpected, \
        f'strict load mismatch: missing={missing[:3]} unexpected={unexpected[:3]}'
    print(f'[check4] PASS  MODELS.build(cfg.model) + load_state_dict(strict=True): '
          f'0 missing / 0 unexpected keys')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='run checks only, do not write the soup file')
    args = ap.parse_args()

    print('CUDA_VISIBLE_DEVICES=%r (CPU-only, GPU untouched)'
          % os.environ.get('CUDA_VISIBLE_DEVICES'))
    for p in SEED_CKPTS:
        assert os.path.isfile(p), f'missing seed checkpoint: {p}'
    sds = [load_state_dict(p) for p in SEED_CKPTS]

    check1_keys(sds)
    avg = make_soup(sds)
    check2_mean(sds, avg)
    check3_identity(sds[0])
    check4_build(avg)

    if args.dry_run:
        print('[main] --dry-run: all checks passed; soup NOT written')
        return

    ckpt = torch.load(SEED_CKPTS[0], map_location='cpu')
    ckpt['state_dict'] = avg
    for k in ('optimizer', 'param_schedulers', 'message_hub'):
        ckpt.pop(k, None)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(ckpt, OUT)
    sz = os.path.getsize(OUT) / 1e6
    print(f'[main] WROTE {OUT}  ({sz:.1f} MB)  '
          f"top-level keys={sorted(ckpt.keys())}")


if __name__ == '__main__':
    main()
