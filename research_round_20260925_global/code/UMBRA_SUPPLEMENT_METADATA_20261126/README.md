# Supplemental Umbra metadata selector

[Selection and primary evaluation prelock](PRELOCK.md) · [Completed lock](LOCK.md)

Run the worker on the original CPython 3.10 environment with Shapely. It performs direct anonymous HTTPS metadata requests to the Umbra provider. Provide the exact closed [original metadata helper](../UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/select_cohort.py), original twelve-window metadata directory, the historical manifest and companion geometry snapshots, and three original pilot result files. The accepted historical fingerprints are in the helper and the supplemental worker. Use `select_supplement_retry.py --help` for CLI flags. Execute with `-I -B -X utf8` and observe natural completion with phase verify via [observe_capture.py](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/observe_capture.py).

The output contains a private ordered GEC product list and a public metadata-only aggregate. The exact lock must precede image intake. Run `python test_select_supplement.py` for deterministic midpoint and geographic exclusion fixtures.
