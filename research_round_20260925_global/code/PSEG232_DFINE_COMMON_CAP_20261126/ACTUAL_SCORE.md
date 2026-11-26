# Observed PSeg-SSDD and SPSD common-cap score

The historical PSeg primary comparison was inconclusive at the predeclared practical threshold. At equal signed edit counts and with the original D-FINE occupancy head, independent endpoints (I) gained 0.021 mask AP points and 0.069 continuous-IoU points over the exact maximum-displacement pairing control (L). The decoded-image 95% interval was [0.030, 0.109]. Different masks occurred in 203 of 293 eligible matches and 125 of 143 decoded groups. The study required at least +0.30 AP points in addition to coverage and interval criteria.

| PSeg author polygon view | Images | Hulls | Eligible pairs / groups | Contrast | Mask AP points | Continuous IoU points [95%] |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Official | 232 | 546 | 293 / 143 | I − L | +0.021 | +0.069 [+0.030, +0.109] |
| Official | 232 | 546 | 293 / 143 | L − F | +0.060 | +0.037 [+0.011, +0.063] |
| Official | 232 | 546 | 293 / 143 | I − F | +0.081 | +0.106 [+0.055, +0.157] |
| Exclude exact historical FIT overlaps | 222 | 502 | 279 / 138 | I − L | +0.020 | +0.072 [+0.031, +0.116] |
| Exclude exact historical FIT overlaps | 222 | 502 | 279 / 138 | L − F | +0.080 | +0.037 [+0.009, +0.064] |
| Exclude exact historical FIT overlaps | 222 | 502 | 279 / 138 | I − F | +0.100 | +0.109 [+0.058, +0.162] |

F is the original distance-first pairing. Official N/F/L/I mask AP percentages were 17.974, 17.995, 18.055 and 18.076, respectively. Original N/F/I AP and geometry survived exact replay in the two PSeg views. AP uses the complete fixed proposal stream. The continuous intervals are 4,096 decoded-image paired draws on original area-eligible matches. Acquisition independence is unverified.

The raster all-hull sensitivity scored the same predictions on identical JPEGs, using each release's original 0.50 native assignment and assigning unmatched hulls zero. With PSeg, I − L was +0.0397 points [0.0151, 0.0650] on 546 hulls (402 matched). With SPSD it was −0.0069 [−0.0473, 0.0322] on 547 hulls (403 matched). The SPSD-minus-PSeg contrast was −0.0466 points [−0.0877, −0.0065] using shared draws. SPSD mask AP I − L was +0.062 points. This reference sensitivity combines annotation geometry and the resulting fixed match assignments. Both references use the same image groups.

Both label-free processes exited naturally with matching edit-stream bytes. The score worker independently checked the original masks, 23,200 candidate slots, the common maximum and 29,304 edits of each sign on 6,404 candidates before opening labels. No detector forwards or optimizer updates occurred. The observed score exited 0 with a verified result footer. [Protocol](PRELOCK.md) · [Aggregate score](SCORE_SAFE.json) · [Terminal score receipt](SCORE_TERMINAL.json)

