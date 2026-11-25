# Actual SRNet prior-clipping strata diagnostic

This retrospective analysis reuses the closed 74-image reserve, original fixed stock matches and the unchanged nested 42-image view. The scalar occupancy head clips the direct logit prior at the original FP32 floor and ceiling. Each addition from the fixed writable pool was checked at or below the floor. Each deletion was checked at or above the ceiling. The author native foreground rule and original endpoint ordinals were kept. Recovered edit counts were eight additions and eight deletions on all 32 valid area-eligible full-view anchors.

The full view contained **zero clipped tail endpoints** among the 1,024 addition and 1,024 deletion endpoint entries. It also contained zero exact native zeros on the addition side and zero exact native ones on the deletion side. All 256 additions and 256 removals selected by I fell in the interior. The constrained truth oracle and unrestricted truth oracle selected the same endpoints at the same signed budget. In the nested view all 704 endpoint entries on each side also fell in the interior.

| Prior-strata decomposition | Full 74 | Nested 42 |
|:---|---:|---:|
| Eligible fixed anchors | 32 | 22 |
| Parent and sensor-date groups | 13 and 12 | 8 and 8 |
| Within-bin continuous IoU gain | +0.5295 [0.2590, 0.8344] | +0.3590 [0.1790, 0.5915] |
| Cross-bin continuous IoU gain | 0.0000 | 0.0000 |

Effects are continuous polygon–cell IoU percentage points with paired 95% sensor-date intervals. The residual cross-bin deviations were below \(10^{-14}\) points. The previously reported oracle and I point estimates and paired intervals were reproduced. Full-view improvement in intersection came from 0.2090 points of correctly added hull area and 0.1502 points of avoided hull deletions, normalized by reference area. Both arose entirely on interior endpoints.

The original \(10^{-4}\) clamp was inactive on the evaluated support and cannot account for its remaining ranking gap. This conclusion concerns the retained endpoints and author fixed-match geometry. The diagnostic leaves the unsaturated additive prior and unretained image pixels outside this inference. It used author reference geometry after the lineage and replay checks, with zero model forwards and no AP calculation.

[Registered scope](PRELOCK.md) · [Original geometry ceiling](../SRNET_RSDD74_THRESHOLD_ORACLE_20261126/GEOMETRY_ORACLE_ACTUAL.md) · [Safe aggregate](STRATA_SAFE.json) · [Witnessed closure](STRATA_TERMINAL_SAFE.json)

