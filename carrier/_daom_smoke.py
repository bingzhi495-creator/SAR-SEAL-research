# -*- coding: utf-8 -*-
"""CPU-only smoke tests for DAOM-Seed (device='cpu', tiny batch). No GPU.

Runs the 5 required checks:
  (a) import config + build model on CPU
  (b) forward_train (loss) on 2 dummy samples w/ gt boxes + fake .seeds -> loss dict
  (c) eval/predict path builds NO aux group (assert aux size 0)
  (d) trainable param count DAOMSeedDINO == SeedDINO (zero new params)
  (e) self-attention mask shape + aux block isolation unit test
"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault('PYTHONPATH', 'E:/SAR/seal')
sys.path.insert(0, 'E:/SAR/seal')

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

import carrier.daom_seed_dino  # noqa: registers DAOMSeedDINO
import carrier.seed_dino       # noqa
import carrier.load_seeds      # noqa
from carrier.daom_seed_dino import DAOMSeedDINO

DEVICE = 'cpu'
torch.manual_seed(0)
np.random.seed(0)


def build_from_cfg(cfg_path):
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)
    model.to(DEVICE).train()
    return cfg, model


def make_samples(bs=2, img_hw=(512, 512), n_gt=(3, 5), with_seeds=True):
    samples = []
    H, W = img_hw
    for i in range(bs):
        ds = DetDataSample()
        ds.set_metainfo(dict(
            img_shape=(H, W), ori_shape=(H, W),
            batch_input_shape=(H, W), img_id=i,
            scale_factor=(1.0, 1.0), pad_shape=(H, W)))
        ng = n_gt[i % len(n_gt)]
        # random xyxy boxes inside the image
        cx = torch.rand(ng) * (W * 0.8) + W * 0.1
        cy = torch.rand(ng) * (H * 0.8) + H * 0.1
        bw = torch.rand(ng) * 40 + 10
        bh = torch.rand(ng) * 40 + 10
        boxes = torch.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)
        gt = InstanceData()
        gt.bboxes = boxes
        gt.labels = torch.zeros(ng, dtype=torch.long)
        ds.gt_instances = gt
        if with_seeds:
            # fake NQS seeds: xyxy abs, like LoadSeeds output (np array)
            ns = 12 + i * 5
            scx = np.random.rand(ns) * (W * 0.8) + W * 0.1
            scy = np.random.rand(ns) * (H * 0.8) + H * 0.1
            sw = np.random.rand(ns) * 30 + 8
            sh = np.random.rand(ns) * 30 + 8
            seeds = np.stack(
                [scx - sw / 2, scy - sh / 2, scx + sw / 2, scy + sh / 2],
                axis=1).astype(np.float32)
            ds.seeds = seeds
        samples.append(ds)
    return samples


def make_inputs(bs=2, img_hw=(512, 512)):
    H, W = img_hw
    return torch.randn(bs, 3, H, W, device=DEVICE)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cfg_path = 'E:/SAR/seal/carrier/configs/daom_seed_pseg.py'
    print('=' * 70)
    print('(a) IMPORT CONFIG + BUILD MODEL ON CPU')
    print('=' * 70)
    cfg, model = build_from_cfg(cfg_path)
    print('built:', type(model).__name__,
          '| seed_source=', model.seed_source,
          '| aux_num_queries=', model.aux_num_queries,
          '| k_o2m=', model.k_o2m,
          '| num_queries=', model.num_queries)

    bs = 2
    samples = make_samples(bs=bs)
    inputs = make_inputs(bs=bs)

    print()
    print('=' * 70)
    print('(b) FORWARD_TRAIN (loss) ON 2 DUMMY SAMPLES')
    print('=' * 70)
    model.train()
    losses = model.loss(inputs, samples)
    aux_keys = sorted(k for k in losses if k.startswith('aux.'))
    primary_keys = sorted(k for k in losses if not k.startswith('aux.'))
    print('total loss keys:', len(losses))
    print('PRIMARY/dn/enc keys:', primary_keys)
    print('AUX keys          :', aux_keys)
    # show a couple of scalar values to prove they are real losses
    print('sample values: loss_cls=%.4f  aux.loss_cls=%.4f  aux.loss_bbox=%.4f'
          % (float(losses['loss_cls']),
             float(losses['aux.loss_cls']),
             float(losses['aux.loss_bbox'])))
    assert aux_keys, 'expected aux.* loss keys in training mode'
    # backward sanity (CPU): make sure the aux loss is differentiable
    total = sum(v for v in losses.values())
    total.backward()
    print('backward() OK, total loss = %.4f' % float(total))

    print()
    print('=' * 70)
    print('(c) EVAL/PREDICT PATH BUILDS NO AUX GROUP')
    print('=' * 70)
    model.eval()
    with torch.no_grad():
        # run the transformer directly to inspect aux meta + query width
        feats = model.extract_feat(inputs)
        head_inputs = model.forward_transformer(feats, samples)
    aux_meta = model._aux_meta
    hs = head_inputs['hidden_states']
    print('model.training =', model.training)
    print('_aux_meta after eval forward_transformer:', aux_meta, '(empty => no aux)')
    print('hidden_states shape (eval):', tuple(hs.shape),
          '-> query width =', hs.shape[2], '== num_queries =', model.num_queries)
    assert aux_meta == {}, 'aux group must NOT be built in eval'
    assert hs.shape[2] == model.num_queries, \
        'eval query width must equal num_queries (no dn, no aux)'
    # full predict() path must run end-to-end
    with torch.no_grad():
        out = model.predict(inputs, make_samples(bs=bs, with_seeds=True))
    print('predict() OK, returned', len(out), 'DetDataSample(s);',
          'pred_instances on [0]:', len(out[0].pred_instances), 'dets')

    print()
    print('=' * 70)
    print('(d) PARAM COUNT: DAOMSeedDINO vs SeedDINO (must be EQUAL)')
    print('=' * 70)
    # build a SeedDINO from the SAME base by swapping the type back.
    seed_cfg = Config.fromfile('E:/SAR/seal/carrier/configs/seeddino_pred_pseg.py')
    init_default_scope('mmdet')
    seed_model = MODELS.build(seed_cfg.model)
    n_daom = count_trainable(model)
    n_seed = count_trainable(seed_model)
    print('DAOMSeedDINO trainable params:', n_daom)
    print('SeedDINO     trainable params:', n_seed)
    print('EQUAL?', n_daom == n_seed, '(delta =', n_daom - n_seed, ')')
    assert n_daom == n_seed, 'DAOM-Seed introduced new parameters!'

    print()
    print('=' * 70)
    print('(e) SELF-ATTENTION MASK SHAPE + AUX BLOCK ISOLATION')
    print('=' * 70)
    # toy: n_existing = num_dn + num_matching, here use small toy values.
    n_dn, n_match, n_aux = 8, 10, 6
    n_existing = n_dn + n_match
    # build a toy dn_mask matching mmdet convention:
    #   matching cannot see dn: [n_dn:, :n_dn] = True ; dn<->matching block.
    dn_mask = torch.zeros(n_existing, n_existing, dtype=torch.bool)
    dn_mask[n_dn:, :n_dn] = True   # matching cannot see dn
    # (dn groups blocking among themselves omitted for the toy; copied verbatim)
    full = DAOMSeedDINO._extend_attn_mask(dn_mask, n_existing, n_aux, 'cpu')
    L = n_existing + n_aux
    print('mask shape:', tuple(full.shape), '== (L,L) with L =', L)
    assert full.shape == (L, L)
    # top-left block must equal the original dn_mask byte-for-byte
    assert torch.equal(full[:n_existing, :n_existing], dn_mask), \
        'top-left block must preserve dn_mask exactly'
    print('top-left [%d,%d] block == original dn_mask: %s'
          % (n_existing, n_existing,
             torch.equal(full[:n_existing, :n_existing], dn_mask)))
    # aux rows: columns 0..n_existing must be all True (masked),
    #           columns n_existing..L must be all False (visible).
    aux_rows = full[n_existing:, :]
    blocked = aux_rows[:, :n_existing]
    visible = aux_rows[:, n_existing:]
    print('aux->[dn+matching] all masked (True)? ', bool(blocked.all().item()))
    print('aux->[aux] all visible (False)?       ', bool((~visible).all().item()))
    # existing rows must not see aux:
    exist_to_aux = full[:n_existing, n_existing:]
    print('[dn+matching]->aux all masked (True)? ', bool(exist_to_aux.all().item()))
    assert blocked.all() and (~visible).all() and exist_to_aux.all(), \
        'aux block is not isolated!'
    # print one aux row to make the structure visible
    print('aux row[0] (1=masked,0=visible):',
          full[n_existing].int().tolist())

    print()
    print('ALL 5 SMOKE TESTS PASSED.')


if __name__ == '__main__':
    main()
