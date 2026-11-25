# Actual SRNet same-support endpoint ranking oracle

The diagnostic reuses the closed SRNet SSDD E24 reserve capture, independently accepted F/I edit ledger and original author scoring inputs. It reconstructed the common signed pools from the native probability and geometry and recovered the two actual policies from the archived masks. All stock raster matches used the original .50 matching. Parent-weighted continuous N/F/I geometry and match counts reproduced the original results. No detector or point-head forward, optimizer update, or AP computation was run.

Each eligible native match keeps its own original foreground area and F's realized count. The 32 full-cohort eligible matches each had 32 endpoints per side and used eight additions and eight removals. This count reached the fixed policy cap. The continuous oracle uses each endpoint's overlap with the author polygon, retaining eight on each side. The add side chooses the highest overlaps. The remove side chooses the lowest overlaps. Exact count and original candidate support are unchanged. The history contains 67 original raster matches. The 32 valid area-eligible matches span 13 author parents and 12 sensor-date blocks. The nested 42-image view contains 40 original matches, 22 eligible matches, eight parents and eight sensor-date blocks.

| Comparison | Full 74 chips | Fixed nested 42 chips |
|:---|---:|---:|
| I minus F | +0.1595 [0.0884, 0.2362] | +0.2061 [0.1213, 0.3062] |
| Oracle minus I | +0.5295 [0.2590, 0.8344] | +0.3590 [0.1790, 0.5915] |
| Oracle minus F | +0.6890 [0.3828, 1.0184] | +0.5651 [0.3353, 0.8803] |
| Oracle minus native | +0.7183 [0.3812, 1.0675] | +0.5442 [0.3249, 0.8363] |

Values are continuous polygon–cell IoU percentage points. Brackets mark paired 95% sensor-date block intervals from 4,096 draws. Parent-weighted intervals are in the safe result. The full-cohort fixed-budget oracle gap exceeds the registered 0.30-point materiality threshold with the blocked interval above zero, conditional on the original support and reference.

Relative to I, the oracle added 0.2090 points of true intersection and avoided dropping 0.1502 points, both normalized to author hull area. Missed and excess area each declined by 0.3592 points of hull area [−0.5346, −0.1995]. I had an addition-side shortfall on 25 of the 32 eligible matches and a removal-side shortfall on 18. The headroom reflects the ranking on both preexisting sides. This result leaves the original practical AP decision unchanged.

The secondary less-than-or-equal count envelope attains the same full geometry with 124 instead of 256 writes per side. For 26 of 32 eligible matches, at least one optimum could use fewer writes. Both outcomes use the author hull and describe attainable geometry on this retrospective cohort.

[Preregistered protocol](PRELOCK.md) · [Safe aggregate](GEOMETRY_ORACLE_SAFE.json) · [Observed closure](GEOMETRY_ORACLE_TERMINAL_SAFE.json)

