# Fixed-endpoint quota calibration

[Actual results](ACTUAL.md) · [Prelocked comparison](PRELOCK.md) · [FIT receipt](fit/RESULT_SAFE.json) · [Render receipt](render/RESULT_SAFE.json) · [Score receipt](score/RESULT_SAFE.json)

Use the original Python 3.10.13, NumPy 1.26.4, and CPU PyTorch 2.10.0+cu130 runtime with `CUDA_VISIBLE_DEVICES=-1` and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` all set to `1`. The programs expose path fields in their `KEYS` tuples. Supply one private JSON mapping those fields to local artifacts for each phase. The files, masks, checkpoints and local mappings remain private.

Run `fit_quota.py --inputs <fit-json> --out <root>/fit`, then `render_quota.py --inputs <render-json> --root <root> --output <root>/render`, and finally `score_quota.py --inputs <score-json> --root <root> --output <root>/score`. Invoke each worker with `-I -B -X utf8` and the [actual-exit observer](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/observe_capture.py). Use observer directories `observed_fit`, `observed_render`, and `observed_score` in the same root, with observer phases verify, capture, and verify. Wait for natural exit and footer verification before using any output.

The FIT program checks the [original signed-action training](../SPSD_PSEG_FIT_MATCHED_SIGNED_ACTION_SEED5297_20261125/PRELOCK.md), [five-fold deployment training](../SPSD_PSEG_FIT_MATCHED_DEPLOYMENT_ACTION_BANK_SEED5297_20261125/ACTUAL.md), [point packets](../SPSD_PSEG_TRAIN915_DFINE_POINT_PREP_20261125/README.md), and [separate author references](../../SPSD_PSEG_DUAL_REFERENCE_LABEL_EXPORT_20261125_ACTUAL.md). It loads pinned helper code in memory to decode FIT prediction records only and collect full areas from the existing FIT associations. Its original packet and five-fold prediction digests must match byte for byte.

The render and score reuse the [fixed-total replay](../SPSD_PSEG_GATE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/README.md) and [historical scorer](../SPSD_PSEG_SCORE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/ACTUAL.md). Render appends two signed quota choices to the unchanged original policy and preserves each fixed per-split action. Scoring compares every original RLE and checks the new signed support before reading the original source-development labels. The observer records the new worker identities.

For synthetic tests, set `SAR_SEAL_VARIABLE_WRITER` to [variable_writer.py](../SPSD_PSEG_GATE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/variable_writer.py) and run `python test_quota.py`.
