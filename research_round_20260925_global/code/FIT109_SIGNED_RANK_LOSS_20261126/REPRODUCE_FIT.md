# Reproduce matched FIT109 training

This worker fits two independent scalar occupancy heads per original fold and seed. The source stage must already have naturally exited and passed payload readback. Supply its output parent through `--source` and bind the source `--source-result-sha`, `--source-terminal-sha` and `--source-worker-sha`. Use the registered `--protocol` and `--protocol-sha`, exact `--prelock-commit`, and qualified release `--code` directory. Run `fit_rank.py` with a fresh `--output` through the [observed worker](../LEARNED_BOUNDED_PAIRING_20261126/rsdd/observe.py) in phase `prepare-edits`.

Use `python -I -B -X utf8`, `CUDA_VISIBLE_DEVICES=-1`, one numeric thread, NumPy 1.26.4 and CPU execution under the qualified Torch 2.10.0 build. The grouped source moments, archived point features, original seeds and architecture remain fixed. The source-prefixed sampler draws 16 candidates with four rows per draw. Each arm uses the same scheduled row ordinals. The second arm adds the registered same-side ordering term. Both update 200 times in each fold and seed. Scoring is a separate closed stage.

The restored held predictions use the original per-candidate batch shape and archived native residual head arithmetic. The zero-based point offsets and source order are in the private canonical tape. The private output contains scalar head weights and held arrays. The safe result contains counts, schedules and weight digests only.

Set `FIT109_PUBLIC_CODE` to the research release code directory and run `test_fit_rank.py` for the synthetic checks. They exercise the objective direction, tied labels, identical candidate packets, held-fold exclusion and matched initial training steps.

[Registered comparison](PRELOCK.md)

