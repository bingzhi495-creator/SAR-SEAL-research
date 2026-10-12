# -*- coding: utf-8 -*-
"""CPU smoke tests (a)-(d) for the M3 ShapePrior head. Real outputs only."""
from __future__ import annotations
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONPATH", "E:/SAR/seal")
sys.path.insert(0, "E:/SAR/seal")

import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS

torch.manual_seed(0); np.random.seed(0)
init_default_scope("mmdet")
import carrier.seal_m2f_head          # noqa
import carrier.geokernel_m2f_head     # noqa
import carrier.transforms_sar         # noqa

CFGS = {
    'geo':    'E:/SAR/seal/carrier/configs/sp_geo_pseg.py',
    'static': 'E:/SAR/seal/carrier/configs/sp_static_pseg.py',
    'blind':  'E:/SAR/seal/carrier/configs/sp_blind_pseg.py',
    'geo_sar':'E:/SAR/seal/carrier/configs/sp_geo_sar_pseg.py',
}


def build_head_from_cfg(path):
    cfg = Config.fromfile(path)
    model = MODELS.build(cfg.model)
    model.eval()
    return cfg, model


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    print("=" * 74)
    print("SMOKE (a): build each of the 4 gen_mode configs on CPU")
    print("=" * 74)
    built = {}
    for name, path in CFGS.items():
        cfg, model = build_head_from_cfg(path)
        head = model.panoptic_head
        built[name] = (cfg, model, head)
        gm = getattr(head, 'gen_mode', '?')
        print(f"  [{name:7s}] BUILT ok  type={type(head).__name__} gen_mode={gm}")

    # also build a 4th gen_mode 'a0' by constructing the head directly with a0
    print("\n  building gen_mode='a0' variant (gate forced off) ...")
    cfg_geo = built['geo'][0]
    mcfg = cfg_geo.model.copy()
    mcfg['panoptic_head'] = dict(cfg_geo.model['panoptic_head'])
    mcfg['panoptic_head']['gen_mode'] = 'a0'
    model_a0 = MODELS.build(mcfg)
    model_a0.eval()
    head_a0 = model_a0.panoptic_head
    print(f"  [a0     ] BUILT ok  type={type(head_a0).__name__} gen_mode={head_a0.gen_mode}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("SMOKE (b): param counts")
    print("=" * 74)
    counts = {}
    for name in ('geo', 'static', 'blind'):
        head = built[name][2]
        counts[name] = count_params(head)
    counts['a0'] = count_params(head_a0)

    # stock Mask2Former head (no ShapePrior, no refine) for the a0~stock reference
    stock_cfg = Config.fromfile('E:/SAR/seal/carrier/configs/a0_m2f_r50_pseg.py')
    stock_model = MODELS.build(stock_cfg.model)
    stock_head = stock_model.panoptic_head
    n_stock = count_params(stock_head)

    # also the carrier SealM2FHead (a0r refine) for reference
    refine_cfg = Config.fromfile('E:/SAR/seal/carrier/configs/a0r_refine_pseg.py')
    refine_model = MODELS.build(refine_cfg.model)
    n_refine = count_params(refine_model.panoptic_head)

    for name in ('geo', 'static', 'blind', 'a0'):
        print(f"  {name:7s}: {counts[name]:>10,d}")
    print(f"  {'stockM2F':7s}: {n_stock:>10,d}  (stock Mask2FormerHead)")
    print(f"  {'a0rRefn':7s}: {n_refine:>10,d}  (carrier SealM2FHead, +refine)")

    gsb = [counts['geo'], counts['static'], counts['blind']]
    spread = (max(gsb) - min(gsb)) / max(gsb) * 100.0
    print(f"\n  geo/static/blind spread = {spread:.4f}%  (must be < 1%)")
    print(f"  geo - a0 (prior heads only) = {counts['geo'] - counts['a0']:,d}")
    extra_vs_stock = counts['a0'] - n_stock
    print(f"  a0 - stockM2F = {extra_vs_stock:,d} "
          f"({extra_vs_stock / n_stock * 100:.3f}% of stock head; "
          f"refine adds {n_refine - n_stock:,d})")
    print(f"  a0 - a0rRefine (refine carrier) = {counts['a0'] - n_refine:,d} "
          f"(prior heads on top of refine)")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("SMOKE (c): gen_mode='a0' forward == stock-head readout (gate-off sanity)")
    print("=" * 74)
    # Use the a0 ShapePrior head's _forward_head vs a plain einsum readout on the
    # SAME decoder_out & mask_feature. With gate off, prior contributes 0, so the
    # mask must equal the pure rank-1 einsum (IoU ~ 1.0).
    B, Q, C, h, w = 2, 5, 256, 32, 32
    dec = torch.randn(B, Q, C)
    feat = torch.randn(B, C, h, w)
    # post_norm + cls/mask_embed are shared; emulate stock readout via the head's
    # own modules (so weights match) but WITHOUT the prior add.
    with torch.no_grad():
        d = head_a0.transformer_decoder.post_norm(dec)
        me = head_a0.mask_embed(d)
        stock_mask = torch.einsum('bqc,bchw->bqhw', me, feat)
        cls_p, sp_mask, _ = head_a0._forward_head(dec, feat, (h, w))
    maxdiff = (sp_mask - stock_mask).abs().max().item()
    sb = (stock_mask.sigmoid() > 0.5); pb = (sp_mask.sigmoid() > 0.5)
    ious = []
    for b in range(B):
        for q in range(Q):
            i = (sb[b, q] & pb[b, q]).sum().item()
            u = (sb[b, q] | pb[b, q]).sum().item()
            ious.append(i / u if u else 1.0)
    print(f"  max|a0_forward - stock_einsum| = {maxdiff:.3e}  (must be ~0)")
    print(f"  per-query mask IoU(a0, stock): min={min(ious):.4f} "
          f"mean={float(np.mean(ious)):.4f}  -> MUST be ~1.0")

    # sanity: geo mode at init (gate~0.018) should be CLOSE to stock but NOT identical
    with torch.no_grad():
        head_geo = built['geo'][2]
        d2 = head_geo.transformer_decoder.post_norm(dec)
        me2 = head_geo.mask_embed(d2)
        stock2 = torch.einsum('bqc,bchw->bqhw', me2, feat)
        _, geo_mask, _ = head_geo._forward_head(dec, feat, (h, w))
        gate_geo = torch.sigmoid(head_geo.geom_mlp(d2)[..., 6]).mean().item()
    print(f"  geo-mode mean gate at init = {gate_geo:.4f}  (must be ~0.018, closed)")
    print(f"  max|geo_forward - stock| = {(geo_mask - stock2).abs().max().item():.3e}"
          f"  (small, gate-scaled prior)")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("SMOKE (d): forward_train loss dict on 2 dummy samples (geo & geo+SAR)")
    print("=" * 74)
    _smoke_loss(built['geo'][1], 'geo (baseline loss)')
    _smoke_loss(built['geo_sar'][1], 'geo + SAR loss (A+B1)', sar=True)

    print("\nALL SMOKE (a)-(d) COMPLETED.")


def _make_dummy_batch(model, with_amp=False):
    from mmengine.structures import InstanceData, PixelData
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks
    H = W = 512
    samples = []
    inputs = []
    rng = np.random.RandomState(0)
    for s in range(2):
        img = torch.rand(3, H, W)
        inputs.append(img)
        n = 3
        masks = np.zeros((n, H, W), dtype=np.uint8)
        bboxes = []
        for i in range(n):
            y0 = rng.randint(0, H - 60); x0 = rng.randint(0, W - 60)
            hh = rng.randint(20, 60); ww = rng.randint(20, 60)
            masks[i, y0:y0 + hh, x0:x0 + ww] = 1
            bboxes.append([x0, y0, x0 + ww, y0 + hh])
        ds = DetDataSample()
        ds.set_metainfo(dict(img_shape=(H, W), ori_shape=(H, W),
                             pad_shape=(H, W), batch_input_shape=(H, W),
                             scale_factor=(1.0, 1.0)))
        gi = InstanceData()
        gi.bboxes = torch.tensor(bboxes, dtype=torch.float32)
        gi.labels = torch.zeros(n, dtype=torch.long)
        gi.masks = BitmapMasks(masks, H, W)
        if with_amp:
            amp = torch.rand(n, H, W)
            gi.amp_maps = amp
        ds.gt_instances = gi
        samples.append(ds)
    batch = dict(inputs=torch.stack(inputs), data_samples=samples)
    return batch


def _smoke_loss(model, label, sar=False):
    model.train()
    batch = _make_dummy_batch(model, with_amp=sar)
    data = model.data_preprocessor(batch, training=True)
    with torch.enable_grad():
        losses = model.loss(data['inputs'], data['data_samples'])
    keys = sorted(losses.keys())
    print(f"\n  [{label}]")
    print(f"    {len(keys)} loss keys:")
    allfinite = True
    for k in keys:
        v = losses[k]
        fin = bool(torch.isfinite(v).all())
        allfinite = allfinite and fin
        mark = "" if fin else "  <-- NON-FINITE!"
        print(f"      {k:28s} = {float(v):+.5f}  finite={fin}{mark}")
    sar_keys = [k for k in keys if 'boundary' in k]
    print(f"    SAR boundary keys present: {sar_keys if sar_keys else 'NONE'}")
    print(f"    all finite = {allfinite}")


if __name__ == "__main__":
    main()
