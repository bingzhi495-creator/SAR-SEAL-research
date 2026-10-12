# Fusion-figure facts (measured 2026-09-03 from the dumped assets)

Source: `paper_tgrs/figs/dump_fusion_assets.py` on HRSID chip `P0101_1200_2000_3000_3800.jpg`
(image_id 1528, `test_inshore.json`), frozen soup `runs/c0_maskiou_v2/soup_s023/iter_100872.pth`,
8 views {640,800,960,1088}x{id,flip}, 800 pooled candidates, K=16 highest-scoring taken,
greedy clustering tau=0.5, deployed gate `2 <= |C| <= 6`, pixel rule `votes >= |C|/2`.

The 16 candidates form exactly four clusters, and those four exercise all four branches
of the operator. Every number below is recomputed from the PNG assets themselves
(script: see "recompute" at the bottom), not from a log.

| cluster dir | \|C\| | in band | contested px (0 < votes < \|C\|) | deployed output | IoU(output, seed) | net area change |
|---|---|---|---|---|---|---|
| `inband6`   | 6 | yes | 211 | pixel vote  | 0.948 | 805 -> 849 px (+44) |
| `outband7`  | 7 | no  | 469 | seed kept   | 1.000 | 1720 px (unchanged) |
| `pair2`     | 2 | yes | 139 | pixel vote (= union at \|C\|=2) | 0.915 | 1492 -> 1631 px (+139) |
| `single1`   | 1 | no  | 0   | seed kept   | 1.000 | 758 px (unchanged) |

**The gate's value is visible in `outband7`.** That cluster has 469 contested pixels, more
than any other, because seven near-duplicate views disagree along a long hull. Had the
majority rule been applied there, the mask would have lost 107 pixels (1720 -> 1613,
IoU 0.934 against the seed): `vote_would_be.png` versus `seed.png`. The gate emits the
seed instead, so nothing is trimmed. This is the per-instance version of the split-level
result that unconditional voting erodes offshore AP from 64.8 to 55.9.

**Member provenance** (view id and fusion-head score, from `manifest.json`):
- `inband6`: v5 0.690, v7 0.689, v1 0.683, v4 0.683, v6 0.681, v3 0.678 (seed = v5)
- `outband7`: v6 0.685, v1 0.684, v2 0.682, v3 0.681, v7 0.681, v5 0.680, v4 0.678
- `pair2`: v2 0.674, v4 0.674
- `single1`: v7 0.677

**Crop windows** are square, union bbox of the cluster padded by 12 px; sizes 266, 110,
109, 65 px. `*_x4.png` are 4x nearest-neighbour upsamples for print. `vote_count.png` is
8-bit with value k = number of members covering the pixel; `vote_count_color.png` is the
same array rendered on the blue ramp.

**Usable for a figure without any schematic drawing**: `crop_sar.png` (SAR backdrop),
`members_contours_x4.png` (all member outlines on the SAR crop), `vote_count_color_x4.png`
(the disagreement map), `seed_x4.png`, `fused_x4.png`, `vote_would_be_x4.png` (out-band
only), `union_x4.png` (pair only), `overview.png` (whole chip with the four cluster boxes),
`views_0..7.png` (each view's candidates drawn on the resized chip).

Recompute:
```
python -c "import numpy as np,os;from PIL import Image;R=r'paper_tgrs/figs/assets/fusion';L=lambda p:np.array(Image.open(os.path.join(R,p)).convert('L'))>127;m=[L(f'outband7/members_{i}.png') for i in range(7)];v=np.sum(m,0);print((v>=3.5).sum(),L('outband7/seed.png').sum())"
```
