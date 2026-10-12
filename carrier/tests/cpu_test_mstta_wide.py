# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE-0 test for the G1 wide multi-scale TTA config.

NO GPU. Builds the ``TestTimeAug`` pipeline from the config(s) and dry-runs it on
one dummy 800x800x3 array. Two checks:

  (1) WIDE config c0_v2_mstta_wide_hrsid.py emits len(scales)*2 = 8 views; each
      view's metainfo carries the right flip/flip_direction and img_shape equals
      the square scale (640, 800, 960, 1088 with keep_ratio on native 800x800).

  (2) GATE-0: take the WIDE config, swap its scale list back to {736,800,864},
      rebuild, and assert the emitted per-view metas are BYTE-IDENTICAL to the
      current 6-view baseline config c0_v2_mstta_hrsid.py. Proves the ONLY change
      the wide config introduces is the scale set; geometry / flip / merge path
      are untouched.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_mstta_wide.py
"""
from __future__ import annotations

import copy
import sys

import numpy as np

from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.registry import TRANSFORMS

register_all_modules()

WIDE_CFG = 'carrier/configs/c0_v2_mstta_wide_hrsid.py'
BASE_CFG = 'carrier/configs/c0_v2_mstta_hrsid.py'


def build_tta(tta_pipeline):
    """Build the TestTimeAug transform (index 1 of a tta_pipeline list)."""
    tta_dict = copy.deepcopy(dict(tta_pipeline[1]))
    return TRANSFORMS.build(tta_dict)


def dummy_results():
    """A minimal results dict as if LoadImageFromFile already ran on an 800x800."""
    img = np.zeros((800, 800, 3), dtype=np.float32)
    return dict(
        img=img,
        img_shape=(800, 800),
        ori_shape=(800, 800),
        img_id=0,
        img_path='dummy.jpg',
        scale_factor=(1.0, 1.0),
        instances=[],  # no GT: LoadAnnotations yields empty gt, PackDetInputs ok
    )


def _norm(v):
    """Canonicalize a meta value so two runs compare byte-identically."""
    if isinstance(v, np.ndarray):
        return ('ndarray', v.shape, [round(float(x), 6) for x in v.ravel()])
    if isinstance(v, (tuple, list)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, float):
        return round(v, 6)
    return v


META_KEYS = ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
             'flip', 'flip_direction')


def emit_metas(tta_pipeline):
    """Run TestTimeAug on the dummy and return the per-view normalized metas."""
    tta = build_tta(tta_pipeline)
    out = tta(dummy_results())
    samples = out['data_samples']
    metas = []
    for s in samples:
        mi = s.metainfo
        metas.append({k: _norm(mi.get(k, '<MISSING>')) for k in META_KEYS})
    return metas


def swap_scales(tta_pipeline, scales):
    """Return a deep copy of tta_pipeline with transforms[0] Resize scales replaced."""
    tp = copy.deepcopy([dict(x) for x in tta_pipeline])
    tp[1] = dict(tp[1])
    transforms = [list(g) for g in tp[1]['transforms']]
    transforms[0] = [
        dict(type='Resize', scale=(s, s), keep_ratio=True) for s in scales
    ]
    tp[1]['transforms'] = transforms
    return tp


def main():
    ok = True

    # ---- Check (1): WIDE config emits 8 correct views --------------------
    wide = Config.fromfile(WIDE_CFG)
    wide_metas = emit_metas(wide.tta_pipeline)
    expected_scales = [640, 800, 960, 1088]
    n_expected = len(expected_scales) * 2
    print(f'[wide] emitted {len(wide_metas)} views (expected {n_expected})')
    assert len(wide_metas) == n_expected, \
        f'view count {len(wide_metas)} != {n_expected}'

    # product(scales, flips): flips iterate fastest -> (scale0,flip),(scale0,noflip),...
    print('[wide] per-view metas:')
    for i, m in enumerate(wide_metas):
        scale = expected_scales[i // 2]
        exp_flip = (i % 2 == 0)  # first of each pair is the hflip branch
        exp_dir = 'horizontal' if exp_flip else None
        exp_shape = (scale, scale)
        got_shape = m['img_shape']
        print(f'  view{i}: img_shape={got_shape} flip={m["flip"]} '
              f'flip_direction={m["flip_direction"]} scale_factor={m["scale_factor"]}')
        if tuple(got_shape) != exp_shape:
            print(f'    !! img_shape {got_shape} != expected {exp_shape}')
            ok = False
        if m['flip'] != exp_flip:
            print(f'    !! flip {m["flip"]} != expected {exp_flip}')
            ok = False
        if m['flip_direction'] != exp_dir:
            print(f'    !! flip_direction {m["flip_direction"]} != expected {exp_dir}')
            ok = False
        if tuple(m['ori_shape']) != (800, 800):
            print(f'    !! ori_shape {m["ori_shape"]} != (800, 800)')
            ok = False
    print(f'[wide] check(1) {"PASS" if ok else "FAIL"}')

    # ---- Check (2): GATE-0 byte-identical to current 6-view baseline ------
    base = Config.fromfile(BASE_CFG)
    base_metas = emit_metas(base.tta_pipeline)
    gate0_pipeline = swap_scales(wide.tta_pipeline, [736, 800, 864])
    gate0_metas = emit_metas(gate0_pipeline)

    print(f'\n[gate0] baseline views={len(base_metas)} '
          f'gate0(wide->736/800/864) views={len(gate0_metas)}')
    same = (base_metas == gate0_metas)
    if not same:
        # pinpoint the first difference
        for i, (a, b) in enumerate(zip(base_metas, gate0_metas)):
            if a != b:
                print(f'  DIFF at view{i}:\n    baseline={a}\n    gate0   ={b}')
        if len(base_metas) != len(gate0_metas):
            print(f'  length mismatch {len(base_metas)} vs {len(gate0_metas)}')
        ok = False
    else:
        print('[gate0] baseline == wide-with-scales-swapped-back  (BYTE-IDENTICAL)')
    print(f'[gate0] check(2) {"PASS" if same else "FAIL"}')

    print(f'\nOVERALL: {"ALL PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
