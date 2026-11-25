# Fixed-bank auxiliary objective experiment

[ACTUAL.md](ACTUAL.md) gives the completed fit and [PRELOCK.md](PRELOCK.md) gives the frozen settings and further evaluation thresholds. Fit only. The shared deployment action packets and disk paths must refer to previously admitted and independently closed prerequisite stages.

Use the same CPU environment as the deployment-bank fit, with CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Run fit_ranking.py with a fresh --out using the options from --help. Use the existing observer's prepare-edits phase to verify natural exit and fit footer. Then run verify_fit.py with the fit root, the verified feature core and saved aggregate SHA. Three tests in test_fit_ranking.py cover the static anchor and true ties, invalid padded pairs and candidate-balanced reduction.

Released aggregates contain hashes and counts. Raw source labels, predictions, trained tensors and process logs remain with the original execution.
