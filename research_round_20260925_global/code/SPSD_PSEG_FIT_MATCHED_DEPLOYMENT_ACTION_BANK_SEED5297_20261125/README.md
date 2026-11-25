# FIT source and verification

The method and numerical scope are in [ACTUAL.md](ACTUAL.md). The earlier failed lock and its untouched outcome are [PRELOCK_ARCHIVED_PARITY.md](PRELOCK_ARCHIVED_PARITY.md) and [ATTEMPT01.md](ATTEMPT01.md). The completed pair used [PRELOCK.md](PRELOCK.md).

Use the existing admitted FIT-only capture, points and label bindings, old immutable FIT runner and old action/occupancy/selector code. Run fit_banks.py with a CPU environment containing CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. See --help for path arguments. Use a fresh `--out` path. Use the original observer's prepare-edits phase to confirm the worker finishes naturally and the final footer matches its safe result. The probe root must contain the two unchanged old-worker prior runs bound by the revised protocol. Their terminal and safe digests are verified inside the fitter.

After natural exit, run verify_fit.py with the FIT root, verified 14-feature occupancy core and final safe SHA. Three finite-independent synthetic tests are implemented in test_fit_banks.py. Exact raw prediction archives, labels, state tensors, and observer logs remain private. The two terminal payload digests and final parameter digests are available in the public aggregate and readback summaries.
