# Capacity-matched distribution feature study

[ACTUAL.md](ACTUAL.md) gives the observed fit and [PRELOCK.md](PRELOCK.md) fixes the cues and scope. The same radiometric.py implements both FIT feature preparation and post-fit candidate inference with the original PointHead.

Use the closed deployment-aligned bank admission and exact FIT feature, reference and capture bindings. Run fit_features.py with CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Use --help for paths and a fresh --out. The public observer prepare-edits phase checks natural exit and the actual result footer. After closure, run verify_fit.py with the fit root, exact training core and measured aggregate result SHA. The private state has original PointHead tensor keys and per-arm center and scaling.

The released fit and readback files are aggregate receipts. Raw candidate data, labels, learned tensors and worker logs remain in the original experiment.
