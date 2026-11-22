# HRSID DEV705 frozen comparison with spatial attraction

[Prelocked protocol](../../HRSID_DEV705_CLASSICAL_ATTRACTION_PRELOCK_20261122.md) · [Actual result](../../HRSID_DEV705_CLASSICAL_ATTRACTION_ACTUAL_20261122.md)

`replay.py` reuses the fixed source model sidecars and frozen head. Follow the `--help` input contract for `prepare-edits` and `verify-edits`. Record the original exit, SHA and matching output footer with `observe.py`. Both edit stages consume only image-side data. Run `score_classical.py` separately after readback with the original native-scoring adapter, polygon references and frozen geometry runtime. Its `--help` names the expected input roles. The existing [native capture release](../../DFINE_E24_DEV705_PROBABILITY_CAPTURE_20261112.md) documents how the sidecars were obtained.

The comparison uses CPython 3.10.13, isolated CPU execution and Shapely 2.0.2 with GEOS 3.11.2 for continuous geometry. Recorded native mask audit, bindings, model weights and original data are required for exact replay. Each new arm uses the same realized count obtained by dynamic spatial attraction. Small synthetic checks accompany both stages.
