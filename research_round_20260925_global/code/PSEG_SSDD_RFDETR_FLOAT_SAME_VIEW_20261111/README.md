# Reproduce the fixed-view FIT label audit

The replay expects the already closed P125 and P269 private training-side artifacts, P395's 741-image polygon-only float overlay, and the frozen protocol in this directory. It verifies exact source and payload hashes before use. The code writes only aggregate output. The original paths, model assets, images, annotations and operational receipts are excluded.

Use the original RF-DETR 1.12.0 and decoder versions declared in the P125 capture, Python 3.12.14, CPU-only visibility and a single logical CPU. Supply all artifact paths as absolute canonical paths. The prior root is the one containing the closed P125 capture and CUDA-replay receipts. The P269 root contains its closed original terminal/result, geometry source and replay source.

Invoke the replay with --prior, --closed-p269, --overlay, --protocol, --output and --cpu-core. Set CUDA_VISIBLE_DEVICES=-1 and OMP_NUM_THREADS=1, and use the same single CPU for process affinity and --cpu-core. Its scientific output is independent of the machine-specific private launcher.

