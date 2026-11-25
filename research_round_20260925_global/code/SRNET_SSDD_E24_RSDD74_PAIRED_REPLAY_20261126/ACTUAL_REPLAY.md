# Actual frozen SRNet replay on the Instance-RSDD parent reserve

The full measured reserve contains 74 images, 81 author hulls and 89 unchanged native SRNet candidates. Three images produced no native candidates. Two policies used the same frozen FIT109 occupancy head and the same signed point support. There were 45 editable candidates with positive budgets and a different mask under the two policies. Each policy made 360 additions and 360 removals, eight on each side of each positive-budget candidate. Other masks retained their native pixels. The second CPU process reproduced the complete edit ledger byte for byte and checked support, signed counts and area. Scoring began after the witnessed capture, both witnessed edit runs and both source identity screens were closed.

| Cohort | Policy | AP | AP75 | AP90 | APsmall | AR100 |
|:---|:---|---:|---:|---:|---:|---:|
| 74 images | Native | 33.265 | 12.480 | 0.064 | 0.000 | 42.222 |
| | Distance-paired | 33.283 | 12.480 | 0.064 | 0.000 | 42.222 |
| | Independent | 33.428 | 12.480 | 0.064 | 0.000 | 42.346 |
| Nested 42 | Native | 34.852 | 15.943 | 0.141 | 0.000 | 42.979 |
| | Distance-paired | 34.852 | 15.943 | 0.141 | 0.000 | 42.979 |
| | Independent | 35.070 | 15.943 | 0.141 | 0.000 | 43.191 |

Scores are percentage points. The full comparison gave independent minus distance-paired AP of +0.1451 points and continuous IoU of +0.1595 points (sensor-date 95% interval +0.0884 to +0.2362). The 67 native raster matches included 32 valid area-eligible matches, all with positive equal count and different masks, across 13 author parents and 12 sensor-date blocks. Excess and missed foreground each declined by 0.1144 points of reference area, with a sensor-date interval of −0.1627 to −0.0668 for each. Both quantities change by the same amount because edit area is fixed.

On the unchanged nested 42 images, the corresponding AP change was +0.2179 and continuous IoU change was +0.2061 (95% interval +0.1213 to +0.3062), from 22 changed eligible matches across eight parents and eight sensor-date blocks. The nested view had 47 hulls and 40 native matches.

The registered overall decision is inconclusive. The full view met the coverage criterion and the paired continuous IoU interval was positive. Its AP change fell below the registered 0.30-point practical threshold. The experiment supports a small fixed-anchor continuous improvement on this previously evaluated reserve with the locally refitted SRNet SSDD E24 carrier. The whole-frame RGB audit found zero decoded duplicates between the 741 consumed SRNet FIT images and 74 reserve images. Acquisition-footprint and geographical separation remain unverified.

[Registration](PRELOCK.md) · [Replay protocol](REPRODUCE_REPLAY.md) · [Score receipt](SCORE_SAFE.json) · [Witnessed stage closure](REPLAY_SCORE_TERMINALS_SAFE.json)

