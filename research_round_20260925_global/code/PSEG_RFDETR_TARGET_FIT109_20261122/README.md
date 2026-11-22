# PSeg target FIT109 occupancy head

[Frozen experiment](../../PSEG_RFDETR_TARGET_FIT109_PRELOCK_20261122.md) · [Observed result](../../PSEG_RFDETR_TARGET_FIT109_ACTUAL_20261122.md)

1. `capture_fit109.py` selects 109 distinct RGB groups from the accepted image-only FIT741 inventory and computes native RF-DETR Q100 masks and coarse logits. Its readback checks native mask parity independently. `observe_capture.py` records the child exits and verifies the result footer.
2. `fit_target.py` uses the verified FIT capture and original floating-point polygons to fit the compact occupancy head. `observe_fit.py` binds the completed result to the actual process exit. `test_fit_target.py` compares the full optimizer path against the small branch of the original fit on a synthetic fixture.
3. `target.py` runs `prepare-edits`, `verify-edits`, and `score` against the closed DEV187 capture and the existing six-arm edit stream. Supply a phase-specific input JSON and its independently pinned SHA-256. The two overlay phases accept a closed FIT checkpoint, preserve the earlier dynamic rule's exact per-candidate count, and use no DEV labels. The score phase accepts a verified overlay and the original float DEV label release. Use the existing [`observe.py`](../HRSID_DEV705_CLASSICAL_ATTRACTION_20261122/observe.py) for these three phases. `test_target.py` takes the absolute released `code` directory as its positional argument.

Refer to each entry point's `--help` for arguments and [the prior frozen head](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/README.md) for compatible runtime and source artifacts. This release provides the protocol, source and aggregate results. Reproduction requires the archived detector, published dataset inputs and fixed prior predictions.
