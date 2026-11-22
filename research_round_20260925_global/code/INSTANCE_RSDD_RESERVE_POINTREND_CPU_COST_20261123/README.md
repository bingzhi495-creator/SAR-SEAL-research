# Paired CPU refinement timing on the Instance-RSDD reserve

Use the latest release tree, the pinned [Windows CPython environment](../INSTANCE_RSDD_B5_REPRO_20261123/README.md), the [74-image frozen reproduction](../INSTANCE_RSDD_B5_REPRO_20261123/README.md), and the [closed PointRend reserve comparison](../INSTANCE_RSDD_RESERVE_POINTREND_20261123/README.md). The timing validates native and edited mask parity against those two closed experiments.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -I -B -X utf8 benchmark_cpu.py --code .. --stock-root runs/closed-74 --point-root runs/closed-74-pointrend --occupancy-weights weights/5217_all_fit.npz --point-weights weights/HEAD_PRIVATE.npz --output runs/cpu-timing
```

Both PointRend policies independently compute their four-step subdivision in each measured pass. Inputs for each image are already in memory. Candidate preparation, input loading, detector and feature capture, and parity checking are recorded separately or outside the timed single-thread refinement. The capped PointRend method receives the existing learned policy's counts. [Locked timing protocol](../../INSTANCE_RSDD_RESERVE_CPU_COST_PRELOCK_20261123.md)
