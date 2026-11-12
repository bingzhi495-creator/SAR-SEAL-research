# D-FINE FIT109 independent boundary endpoint ranking

The frozen D-FINE E24 detector predictions and the original independently fitted BCE point head were reused on HRSID FIT109. The detector saw this cohort during training. Point-head fitting used the original five source-disjoint folds and 200 AdamW steps of 64 pairs each. Original fixed-pair BCE training, mask AP, geometry, edits and strict identity transitions replayed exactly. All methods used the same 194,640 frozen endpoint pairs and all 10,900 original predictions, with unchanged rank, score and area.

Independent ranking sorts the original add endpoints by decreasing point probability and drop endpoints by increasing point probability. The natural policy uses the original cap and accepts positive marginal gains. The matched-count policy keeps the original fixed-pair swap count for each candidate. It chose a different endpoint set on 4,405 of the 4,416 edited candidates. Eight matched-count candidates each needed one nonpositive marginal step, and the independently ranked summed predicted gain was at least the original fixed-pair sum for each same-count candidate.

| Arm | Mask AP | AP75 | AP90 | AR100 | Source-equal continuous IoU | Fixed strict .90 | Edited candidates | Swaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 49.3251 | 47.3883 | 1.5337 | 53.1250 | 80.0492 | 7 | 0 | 0 |
| Fixed BCE | 50.0428 | 48.5535 | 1.5501 | 53.7097 | 80.5044 | 6 | 4,416 | 20,809 |
| Independent, matched count | 51.6192 | 53.3188 | 1.5439 | 55.2419 | 81.1196 | 10 | 4,416 | 20,809 |
| Independent, natural count | 53.2942 | 57.6461 | 1.5439 | 57.0565 | 81.4546 | 12 | 6,043 | 36,571 |

Natural independent ranking gained +3.9691 AP over raw and +3.2514 over fixed BCE. Source-equal IoU gains over the same baselines were +1.4054 [1.1940, 1.6291] and +0.9502 [0.7655, 1.1565] with paired bootstrap 95% intervals across 412 original valid continuous matches and 93 source groups. The matched-count independent policy gained +1.5765 AP and +0.6152 IoU [0.4996, 0.7319] over fixed BCE. Both independent policies improved source-equal IoU over fixed BCE in all five folds. Natural editing touched 411 fixed-valid, four neutral and 5,628 unmatched candidates. Matched-count and fixed BCE touched 364 fixed-valid, four neutral and 4,048 unmatched candidates each.

| Arm | Original strict untouched | Original strict retained after edit | Original strict lost | Newly strict | Stable below .90 |
|---|---:|---:|---:|---:|---:|
| Raw | 7 | 0 | 0 | 0 | 405 |
| Fixed BCE | 0 | 5 | 2 | 1 | 404 |
| Independent, matched count | 0 | 7 | 0 | 3 | 402 |
| Independent, natural count | 0 | 7 | 0 | 5 | 400 |

The natural policy passed all prespecified FIT gates. The matched-count result supports a location benefit from relaxing fixed pair membership on these candidate endpoints. The natural policy also changes the count, and the observed gains cannot be assigned wholly to location. The original fixed-pair oracle gave 81.1885 source-equal IoU and 11 strict hits and was constrained to its original pairs. Natural independent ranking searches a larger feasible action set. Overall mask AP90 remained close to raw despite the continuous strict match improvement. This was one detector-training-visible source cohort. Prior loss-change and guard stops remain. No detector forward or update, DEV, official TEST or RD was used.

At fixed edit count k and fixed add/drop supports, independent endpoint selection maximizes the sum of predicted occupancy gains because the best k additions and best k removals can be chosen separately. With an optional capped count, the sorted marginal gains are nonincreasing, so the optimum uses positive gains only. This is restricted cardinality-constrained posterior ranking. RankSEG already studies posterior ranking for segmentation with mask size selection. The experiment quantifies the loss of keeping a fixed geometric pairing in this SAR cohort. The ranking operation itself has a standard optimization interpretation. [RankSEG](https://www.jmlr.org/papers/v24/22-0712.html) · [Fixed endpoint truth geometry](DFINE_E24_FIT109_PAIR_SUPPORT_ORACLE_20261112.md)
