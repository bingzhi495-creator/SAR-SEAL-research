# UEIKAP frozen whole-RoI mask coverage

The [locked evaluation](../../UEIKAP_POINTREND_COVERAGE_PRELOCK_20261123.md) uses 342 chips, 528 released hulls and 13 acquisition families. The [pre-render correction](../../UEIKAP_POINTREND_COVERAGE_PRELOCK_ERRATUM_20261123.md) identifies the exact legacy 7 × 7 coarse sampler. Both whole-RoI arms share the original paste. The PointRend branch uses the previously fitted head, with 3,136 point-head evaluations per rendered candidate.

Use the accepted [342-chip capture and data intake](../UEIKAP_ADDITIONAL_FAMILIES_20261123/README.md) and the exact [RF-DETR center5202 and PointRend source-only fit](../INSTANCE_RSDD_POINTREND_TRANSFER_20261122/README.md). Keep original weights, predictions, raw basis arrays and released contours local. Use the qualified [B5 Windows CPython environment](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) with the standalone host interpreter if required. A [process observer](../HRSID_DEV705_CLASSICAL_ATTRACTION_20261122/observe.py) illustrates the actual-exit, footer and result-digest format. Use the phase name shown below with an observer that admits that phase. Each worker requires fresh output.

| Stage | Entry point |
| --- | --- |
| `basis` | `capture_basis.py --code DIR --native-root DIR --checkpoint FILE --protocol FILE --protocol-sha256 SHA --output RUN/basis`. Use one CUDA device and the original UEIKAP settings. |
| `prepare-edits` | `replay_point.py prepare-edits --old-root DIR --basis RUN/basis --basis-terminal FILE --fit-weights FILE --protocol FILE --protocol-sha256 SHA --code DIR --output RUN/prepare-edits --workers 8`. Use the identical FIT109 PointRend head and CPU-only visibility. |
| `verify-edits` | `replay_point.py verify-edits` with the same source arguments and count of workers, `--prepare RUN/prepare-edits --prepare-terminal FILE` and a fresh `RUN/verify-edits` output. |
| `score` | `score_coverage.py --code DIR --root DIR --render RUN --protocol FILE --protocol-sha256 SHA --output RUN/score`. Use the accepted author geometry runtime. |

Use isolated Python (`-I -B -X utf8`). Observe each terminal explicitly. The scorer verifies bound basis and rendered masks, then opens the original author hulls. Independent mask assignment reports released-hull coverage, while fixed score-ordered boxes retain one candidate per hull for continuous geometry. The labels are case-focused and the complete-scene denominator is unknown.

The [observed experiment](../../UEIKAP_POINTREND_COVERAGE_ACTUAL_20261123.md) and [verified aggregate receipt](../../results/UEIKAP_POINTREND_COVERAGE_ACTUAL_SAFE_20261123.json) preserve the negative coverage result and all execution checks.
