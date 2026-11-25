# Reproduce the signed posterior diagnostic

[Actual results](ACTUAL.md) · [Prelock](PRELOCK.md) · [Result summary](RESULT_SAFE.json)

Run `audit_target.py` using the same Python 3.11.15 CPU runtime as the [earlier closed diagnostic](../SAR_SSP_FROZEN_POLICY_REGRET_20261126/README.md). Use `CUDA_VISIBLE_DEVICES=-1` and `-I -B -X utf8`. Provide the exact closed original target run via `--transfer` and the accepted earlier audit directory via `--regret`. Set `--output` to a fresh file and use the existing [actual-exit observer](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/observe_capture.py) with phase verify.

The worker binds the exact accepted readback implementation, results, and successful natural exit. It adds an in-memory numerical callback after original mask and source-head replay and verifies the original audit quantities. The signed ledger takes each prediction column and author binary endpoint occupancy separately. Synthetic checks run with Python and NumPy using `python test_signed_ledger.py`. Closed prediction and annotation payloads remain private.
