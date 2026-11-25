# Reproduce the SAR-SSP attachment diagnostic

[Actual report](ACTUAL.md) · [Prelock](PRELOCK.md) · [Aggregate result](RESULT_SAFE.json)

Use the same Python 3.11.15, NumPy 1.26.4, and qualified RLE runtime as the [frozen transfer](../SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_20261126/README.md), with SciPy available. The runner checks the SHA-256 of the earlier transfer closure validator and the exact closed render and score before accepting their private inputs. Supply the released [diagnose.py](../SAR_SSP_FROZEN_POLICY_REGRET_20261126/diagnose.py) via `--closure`; `--transfer` points to the verified original run directory documented in that protocol.

Run `python -I -B -X utf8 audit_attachment_retry.py --closure <diagnose.py> --transfer <closed-run> --output <fresh-result.json>` with `CUDA_VISIBLE_DEVICES=-1`, via the [natural exit observer](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/observe_capture.py) in verify mode. Source code and all study summaries are included. Original RLEs, author pixel rasters, inference assets and subject-level scores remain private.

Run `python test_geometry.py` with NumPy and SciPy to verify the disjoint partitions and the diagonal adjacency sensitivity on synthetic masks.
