# Frozen mask-branch cue correspondence study

[ACTUAL.md](ACTUAL.md) reports the executed FIT and [PRELOCK.md](PRELOCK.md) gives the design. The capture admission receipt binds the actual complete capture and independent readback. The fitter reads only the original FIT point and cue packets, with exact coordinates and packet hashes.

Use bind_acquisition.py only after both observer closures pass, with the verified capture and readback result digests. Run fit_spatial.py in a CPU environment with CUDA_VISIBLE_DEVICES=-1 and OMP_NUM_THREADS, MKL_NUM_THREADS, and OPENBLAS_NUM_THREADS set to 1. See --help for exact path arguments and a fresh --out. Use the public observer's prepare-edits phase for the fitter. After its natural exit run verify_fit.py with the original feature core and accepted summary digest. The helper spatial_cues.py is shared by FIT and label-free candidate-level inference. The original baseline retains the old PointHead inference path.

Included aggregates preserve closures, counts, and hashes. Original point/cue packets, labels, checkpoints and raw logs stay with the experiment.
