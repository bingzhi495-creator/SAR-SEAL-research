# -*- coding: utf-8 -*-
"""Weight-space diagnostic for the PSeg-SSDD soup-N anomaly (N=2,5 fall below members).
Prints PAIRSOUP_DIST lines: per-seed distance to s0, pairwise delta cosine, pairwise distance."""
import sys, itertools, torch
sys.path.insert(0, 'E:/SAR/seal')
from carrier.soup import load_state_dict
seeds = [0, 2, 3, 4, 5, 6]
vecs = {}
for s in seeds:
    sd = load_state_dict(f'E:/SAR/seal/runs/cp_pseg/s{s}/iter_8352.pth')
    vecs[s] = torch.cat([v.double().flatten() for k, v in sd.items() if v.is_floating_point()])
base = vecs[0]
print(f'PAIRSOUP_DIST nparams={base.numel()} norm_s0={base.norm():.3f}')
for s in seeds[1:]:
    print(f'PAIRSOUP_DIST seed{s} dist_to_s0={(vecs[s]-base).norm():.4f}')
for a, b in itertools.combinations(seeds, 2):
    da, db = vecs[a]-base, vecs[b]-base
    cos = float((da*db).sum()/(da.norm()*db.norm()+1e-12)) if a != 0 else float('nan')
    print(f'PAIRSOUP_DIST pair=s{a}s{b} dist={(vecs[a]-vecs[b]).norm():.4f} delta_cos_vs_s0={cos:.4f}')
