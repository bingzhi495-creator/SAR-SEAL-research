# Fixed deployment-bank component comparison

[ACTUAL.md](ACTUAL.md) records the realized fit. [PRELOCK.md](PRELOCK.md) specifies the locked initialization, objective and gate criteria.

Use the existing FIT-only admitted data and deployment-aligned bank receipts. Run fit_calibration.py with CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Use --help for paths and a fresh --out. The --admission-worker points to the verified prior fit_ranking.py used to regenerate the closed deployment bank and retains its input and digest checks. The public observer prepare-edits phase confirms natural worker closure and result footer.

After natural exit, verify_fit.py independently binds the observed terminal event, scalars, bank and checkpoint, then directly hashes loaded network tensors. Three finite tests in test_fit_calibration.py cover component formulas, the static zero and candidate balancing. The public sum files contain aggregate counts and hashes. Actual source masks, raw candidate predictions and learned tensors remain with the original experiment.
