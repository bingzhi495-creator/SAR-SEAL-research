# D-FINE FIT109 rendered intensity at equal realized swap counts

The full rendered-intensity point head produced larger gains with its natural edit count in the prior ablation. This follow-up used the same training-visible HRSID FIT109 detector predictions, original scores and 10,900 candidates, fixed point pairs, and five source-disjoint calibration folds. Each control and full head was refitted for 200 original AdamW updates per fold. The 3,000 point-optimizer updates reproduced all prior natural arm AP, realized budgets, training audits and continuous geometry exactly. There were no new detector forwards or detector updates.

For each candidate, the common count was the minimum number of positive-gain swaps chosen by the three natural heads. Each head chose that many of its own highest-ranked positive swaps from the same original pairs. No negative action was forced. Each active arm made 6,664 swaps on the same 2,059 candidates, consisting of 148 valid fixed matches, three neutral candidates and 1,908 unmatched candidates. Another 8,841 were unchanged. Original areas, ranks and scores were preserved.

| Arm | Mask AP | AP75 | AP90 | AR100 | Source-equal continuous IoU | Fixed strict .90 |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 49.3251 | 47.3883 | 1.5337 | 53.1250 | 80.0492 | 7 |
| Gray zero | 49.4521 | 47.4020 | 1.5337 | 53.1653 | 80.1207 | 5 |
| Gray shuffled | 49.4563 | 47.4020 | 1.5337 | 53.2056 | 80.1115 | 5 |
| Full rendered intensity | 49.4792 | 47.4645 | 1.5337 | 53.2661 | 80.2083 | 6 |
| Same-pair truth-only geometry oracle | | | | | 81.1885 | 11 |

Full head gains over gray-zero and gray-shuffled were +0.0271 and +0.0230 AP, with source-equal IoU gains of +0.0875 (paired bootstrap 95% interval [0.0365, 0.1465]) and +0.0968 ([0.0436, 0.1565]) across 412 valid original fixed matches and 93 source groups. The signs were positive in all five calibration folds against either control. The preregistered gray location criterion required +0.30 AP and +0.20 IoU over each control. Full versus raw gained +0.1541 AP and +0.1590 IoU, below the +0.50 AP and +0.20 IoU thresholds. AP75 was retained.

| Budget | Arm | Raw strict retained without edit | Raw strict retained after edit | Raw strict lost | Newly strict | Stable below .90 |
|---|---|---:|---:|---:|---:|---:|
| Natural | Gray zero | 3 | 2 | 2 | 0 | 405 |
| Natural | Gray shuffled | 3 | 2 | 2 | 0 | 405 |
| Natural | Full | 0 | 5 | 2 | 1 | 404 |
| Common | Gray zero | 4 | 1 | 2 | 0 | 405 |
| Common | Gray shuffled | 4 | 1 | 2 | 0 | 405 |
| Common | Full | 4 | 2 | 1 | 0 | 405 |

Both prespecified gates stopped. The equal-count experiment isolates within-budget ranking on a subset chosen jointly by the three heads. The small surviving difference supports an appearance contribution to ranking on this subset. It does not identify what portion of the original natural gains came from broader activation, larger counts, or interactions between the two. The geometry oracle uses the existing same-pair cap and provides context for those pairs. No DEV, official TEST or RD images were accessed. The input feature is rendered grayscale intensity.
