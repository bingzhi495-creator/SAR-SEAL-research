# Frozen HRSID point-head transfer to PSeg-trained RF-DETR

[Prelock](../../PSEG_RFDETR_FROZEN_SOURCE_HEAD_PRELOCK_20261122.md) · [Observed outcome](../../PSEG_RFDETR_FROZEN_SOURCE_HEAD_ACTUAL_20261122.md)

`capture_worker.py` keeps the accepted PSeg 5201 center E24 native predictions and gathers the matching 144-square logit planes. Use its `--help` for the target checkout, accepted checkpoint and capture references. `observe_capture.py` binds genuine original exit and output hashes. Run the independent capture reader against the original predictions and exact image roster.

`replay.py` prepares and verifies the fixed-head edit streams without labels. It reuses the archived source RF-DETR 5217 refit and released modules. `score.py` opens the accepted original-float reference only after the observed independent replay, then reproduces all saved original mask metrics and scores the new edit policies. Use the same `observe.py` from [`HRSID_DEV705_CLASSICAL_ATTRACTION_20261122`](../HRSID_DEV705_CLASSICAL_ATTRACTION_20261122) to observe the offline edit and score phases. `oracle.py` performs the optional post-hoc label-conditioned geometry check after a closed score.

The CPU path uses the qualified Python 3.10.13, NumPy 1.26.4, pycocotools 2.0.11, Shapely 2.0.2 and GEOS 3.11.2. The archived exact source head, carrier and original data references are needed for replication. The released safe summaries exclude per-image predictions, labels and model weights.
