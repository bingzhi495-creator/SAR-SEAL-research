# PSeg-SSDD frozen comparison with spatial attraction

[Prelocked protocol](../../PSEG_TEST232_CLASSICAL_ATTRACTION_PRELOCK_20261122.md) · [Actual evaluation](../../PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122.md)

`compare.py` reuses the already accepted PSeg native candidate export, original five-arm replay, frozen HRSID FIT109 point head and native-grid spatial-attraction adaptation. Run `prepare-edits`, `verify-edits` and `score` as separate isolated CPU processes (`python -I -B -X utf8`, `CUDA_VISIBLE_DEVICES=-1`). Use the included observe.py to bind actual exits and output footers. Use observed_prepare-edits, observed_verify-edits and observed_score under the phase root as the observer destinations. Use `--help` for the exact file arguments. Pass target labels only to `score`. The original data access and physical split are linked in the prelock.

`test_compare.py` verifies acceptance of matching realized counts and rejection of unequal or missing matched arms. The actual independent check decodes every edited mask, verifies the restricted endpoint support and confirms the XOR and foreground counts. The PSeg scorer repeats the original COCO score, while polygon geometry uses Shapely 2.0.2 and GEOS 3.11.2. The released results contain only aggregate metrics.
