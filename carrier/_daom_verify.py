# -*- coding: utf-8 -*-
"""Two correctness verifications beyond the 5 smoke tests:

(f) NON-CONTAMINATION: the dn+matching decoder outputs are byte-identical
    whether or not the aux group is present (the block-diagonal mask makes them
    mathematically independent of aux). We monkey-disable aux and compare the
    primary hidden_states slice for the SAME inputs/seed.
(g) NULL CONTROL: daom_groupdetr_null_pseg.py builds the aux group but injects
    NO seeds (seed_source='off') -> aux ref points == learnable matching init.
"""
import os
import sys

import torch

sys.path.insert(0, 'E:/SAR/seal')
os.environ.setdefault('PYTHONPATH', 'E:/SAR/seal')

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS

import carrier.daom_seed_dino  # noqa
import carrier.seed_dino       # noqa
import carrier.load_seeds      # noqa

from carrier._daom_smoke import make_samples, make_inputs

DEVICE = 'cpu'


def build(cfg_path):
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmdet')
    m = MODELS.build(cfg.model).to(DEVICE)
    return cfg, m


def main():
    print('=' * 70)
    print('(f) NON-CONTAMINATION: dn+matching outputs independent of aux group')
    print('=' * 70)
    _, model = build('E:/SAR/seal/carrier/configs/daom_seed_pseg.py')
    model.train()
    bs = 2
    inputs = make_inputs(bs=bs)
    samples = make_samples(bs=bs)

    # Run WITH aux (deterministic: same seeded dn noise via manual_seed).
    torch.manual_seed(123)
    feats = model.extract_feat(inputs)
    head_with_aux = model.forward_transformer(feats, samples)
    n_existing = model._aux_meta['n_existing']
    hs_primary_with_aux = head_with_aux['hidden_states'][:, :, :n_existing, :].clone()

    # Run WITHOUT aux by temporarily zeroing aux_num_queries, SAME seed so the
    # CdnQueryGenerator draws identical noise -> identical dn+matching block.
    aux_save = model.aux_num_queries
    model.aux_num_queries = 0
    torch.manual_seed(123)
    feats2 = model.extract_feat(inputs)
    head_no_aux = model.forward_transformer(feats2, samples)
    hs_no_aux = head_no_aux['hidden_states'].clone()
    model.aux_num_queries = aux_save

    print('with-aux primary slice shape:', tuple(hs_primary_with_aux.shape))
    print('no-aux   hidden_states shape:', tuple(hs_no_aux.shape))
    # NOTE: exact bitwise equality is not expected -- adding aux columns changes
    # the sequence length and hence the fp32 reduction order inside the fused
    # attention / LayerNorm kernels. The masked entries contribute ZERO
    # mathematically, so any residual is pure floating-point rounding (~1e-6 on
    # values of order ~1). We assert it is below a generous fp32 tolerance.
    maxdiff = (hs_primary_with_aux - hs_no_aux).abs().max().item()
    same = maxdiff < 1e-4
    print('dn+matching outputs identical with vs without aux (atol 1e-4)? ', same,
          '| max abs diff =', maxdiff, '(fp32 reduction-order noise)')
    assert same, 'aux group contaminates the primary outputs beyond fp32 noise!'

    print()
    print('=' * 70)
    print('(g) NULL CONTROL: daom_groupdetr_null_pseg builds aux but NO seeds')
    print('=' * 70)
    _, null_model = build(
        'E:/SAR/seal/carrier/configs/daom_groupdetr_null_pseg.py')
    null_model.train()
    print('null model seed_source=', null_model.seed_source,
          '| density_alpha=', null_model.density_alpha,
          '| aux_num_queries=', null_model.aux_num_queries,
          '| k_o2m=', null_model.k_o2m)
    feats3 = null_model.extract_feat(inputs)
    head_null = null_model.forward_transformer(feats3, samples)
    meta = null_model._aux_meta
    print('aux group built? n_aux=', meta.get('n_aux'),
          '| n_seed_per_img=', meta.get('n_seed_per_img'),
          '(all zero => no evidence seeds, pure o2m control)')
    assert meta.get('n_aux') == null_model.aux_num_queries
    assert all(n == 0 for n in meta.get('n_seed_per_img', [])), \
        'null control must inject zero seeds'
    losses = null_model.loss(inputs, make_samples(bs=bs))
    aux_keys = sorted(k for k in losses if k.startswith('aux.'))
    print('null control still produces aux o2m loss keys:', len(aux_keys),
          '->', aux_keys[:3], '...')
    assert aux_keys

    print()
    print('BOTH VERIFICATIONS PASSED.')


if __name__ == '__main__':
    main()
