# Reproduce the transverse quota scoring

Use the exact historical variable-budget SAR-SSP transfer followed by the registered label-free transverse-quota render. The original transfer directory contains its observed render and score and their matching output bytes. Supply it as `--base-run`. The new stage containing `render/` and `observed_render/` is `--writer`. Bind its result, observer and source with `--render-result-sha`, `--render-terminal-sha` and `--render-worker-sha`. Provide the exact `--protocol`, `--protocol-sha`, `--prelock-commit` and a fresh `--output`.

1. Run `score_transverse.py verify-edits` through the [observed subprocess runner](../LEARNED_BOUNDED_PAIRING_20261126/rsdd/observe.py) in phase `verify-edits`. This worker reconstructs each edit mask on the original signed coordinates, verifies the quota counts and checks the realized shape moments and the original free fallback. It uses zero annotations.
2. Run `score_transverse.py score` with the same original bindings, a different output, and `--verify`, `--verify-terminal`, `--verify-result-sha` and `--verify-terminal-sha` for the independent closure. The worker rechecks both the old and new verified artifacts before loading the original author validation dataset.

Use `python -I -B -X utf8` with the qualified Python 3.11.15 CPU runtime and admitted NumPy 1.26.4 and pycocotools 2.0.11 (the historical NumPy and codec locations take precedence on the module path). Set `CUDA_VISIBLE_DEVICES=-1` and numeric-library thread counts to one. Run `test_score_transverse.py` for native RLE checks, unequal signed edits with equal total and independent rank-one PCA eligibility checks.

[Preregistered scope](PRELOCK.md)

