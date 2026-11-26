# DEV705 cumulative squared-distance witness actual results

The no-label coverage and original author-reference score both closed naturally with exit 0 and verified output footers. The accepted label-free closure took 44.99 s and the score worker took 59.11 s. CPU replay used the same RF-DETR center5202 candidate stream, three frozen occupancy heads, predictions, rank, writable support, exact signed budgets, and original F/L/I endpoint choices. It ran zero detector forwards and optimizer updates. The source archive bytes and original published F/L/I release remained unchanged. [Prelocked protocol](PRELOCK.md) · [Reproduction](REPRODUCE.md)

The new writer W uses I when I has no perfect matching under the original cap and its unrestricted minimum assignment cost is no larger than either F or L. Otherwise W uses L. The cost is the sum of squared distances between selected native endpoints, minimized over pair bijections separately for each policy. The actual selected F edge sum equals its unrestricted optimum on all evaluated positive-budget rows. Four synthetic tests, including 48 exhaustive assignment comparisons, passed before coverage. A separate synthetic reconstruction verified the selected W mask and union across heads before score launch.

| HRSID DEV705 frozen head | Positive candidate budgets | W differs from L | Changed source prefixes | W−L mask AP points |
|---|---:|---:|---:|---:|
| 5217 | 40,316 | 15 | 13 | 0 |
| 5218 | 40,512 | 6 | 6 | −0.004155523 |
| 5219 | 40,571 | 15 | 12 | 0 |
| Union across heads / AP mean | — | 36 | 19 | −0.001385174 |

The label-free census includes all 67,386 mapped candidates from 705 images, of which 42,325 meet the protected area range and 40,897 have endpoint support. The 36 candidate identities across 19 source prefixes passed the necessary threshold of 20 candidates across eight groups. The [coverage result](COVERAGE_SAFE.json) retains the ineligible, unsupported and zero-budget counts and every observed pattern of the cap and cost predicates by head and budget. Omitted patterns have zero count.

On the original full 1,782-reference DEV705 population, W and L differ on only **one of 1,461 original valid fixed native matches, in one of 27 source prefixes**. This first occurs in head 5218. The mean W−L continuous-IoU change is **−0.00005344 points** (95% paired source-prefix interval [−0.00016033, 0]). AP75 and AP90 were unchanged for all heads. The empty-raster sensitivity with 1,778 references gave AP changes of 0, −0.002576499 and 0 for heads 5217–5219. The full score retained all original native prediction ranks and scores, averaged head effects within hull, and equally weighted the original source prefixes with 4,096 paired PCG64(5217) resamples. The baseline N/F/L fixed-match geometry reproduced the original aggregate values. [Full scores](SCORE_SAFE.json)

The primary result is **unqualified**. The matched coverage requirement is 20 hulls in eight groups. This restrictive writer exposed only one matched hull, and the AP magnitude and paired interval also miss the decision rule. The result bounds the utility of this particular selection rule on historical DEV705. It leaves the cumulative-distance explanation of the previously observed unrestricted selection benefit unresolved. W−F retains +0.473745 AP and +0.282219 continuous-IoU points as a secondary contrast, close to the original L−F gain. Preserve the locked null and stop the W branch. The best-endpoint control, original bounded contrast, historical reserves and independent Umbra analysis retain their original scope and selection history.

[Observed coverage exit](COVERAGE_TERMINAL.json) · [Observed score exit](SCORE_TERMINAL.json) · [Repeated score-worker checks before labels](PRESCORE_SAFE.json)
