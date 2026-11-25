# Reproduce the retrospective SAR-SSP audit

Run `diagnose.py` with Python 3.11.15, the same isolated CPU runtime used for the [frozen transfer](../SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_20261126/README.md), and `CUDA_VISIBLE_DEVICES=-1`. Pass `--closed-transfer` the local original execution directory, containing the exact original worker, two verified exit receipts, and the writeback and score outputs. The diagnostic checks their hashes, recovers the original renderer and scorer, and validates its original input paths. Reproduction of the exact signed-menu mathematics alone does not require these closed files.

Use `-I -B -X utf8 diagnose.py --closed-transfer <directory> --output <new-result-file>` to rerun the audit. Run it through the existing [observer](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/observe_capture.py) in verify mode to obtain an independent exit receipt. The runner requires the same source detector capture, author archive, point head and helper files expected by the original transfer. The [portable transfer worker](../SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_20261126/reproduce_transfer.py) documents those inputs. It uses a different worker digest and cannot be substituted for the original exact archival replay.

For the mathematical check, point the `SAR_SEAL_VARIABLE_WRITER` environment variable to the released [fixed-total writer](../SPSD_PSEG_GATE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/variable_writer.py) and run `python test_frontier.py`. The test compares the full-support result to exhaustive enumeration and checks the nested regret bounds on synthetic inputs.

[Results](ACTUAL.md) · [Prelock](PRELOCK.md) · [Aggregate JSON](RESULT_SAFE.json)
