# Two fixed synthetic FastGeodis CPU/CUDA checks

This source ran once after the active research segment ended and was accepted. Original tool, SSH and worker exited zero; the recorded process group was then empty. It is an operator environment check, not a SAR experiment, SAMRefiner reproduction, training seed, AP evaluation or proof of general CPU/GPU equality.

The two predeclared 9x9 float32 NCHW fixtures use v=1e10 and two scans: a zero image/center seed with lambda0 checks seed0 and an axis-neighbor distance1; an intensity step/left seed with lambda1 checks seed0 and a positive cross-boundary cost. Both devices must return finite, nonnegative tensors of the declared shape/dtype/device. Scalar analytic checks use atol1e-5/rtol0; CUDA explicitly synchronizes. CPU and GPU full maps are not used as each other's GT. Their observed max differences were zero only for these fixtures.

Tested stack: Linux aarch64, Python3.12, Torch2.10.0+cu130, FastGeodis1.0.5, NVIDIA GB10. Use a quiet GPU0 lease, available CPU0/1 and at least16GiB available host RAM. The source sets two CPU threads and a4GiB PyTorch allocator limit, checks process lifetime RSS and before/after available RAM, then requires allocated tensor memory0 after cleanup. It does not certify whole-card memory or continuously monitor host RAM. Root used an outer90-second GNUtimeout KILL bound and separately verified group absence; running the source alone does not reproduce that external supervision.

Example in an already prepared compatible environment:

    python -B fastgeodis_cpu_cuda_smoke.py --report smoke_report.json

The report path must be new. No package installation, real image, annotation, model, weight or CPU fallback is included. The output maps are synthetic operator distances, not dataset ground truth. Native CUDA failures must remain failures.

[Actual safe result](../../results/U57_FASTGEODIS_FIXED_CPU_CUDA_CHECK_SAFE_RESULT_20261025.json)
[Official API](https://github.com/masadcv/FastGeodis/blob/1.0.5/FastGeodis/__init__.py#L37)
[Author precision limits](https://github.com/masadcv/FastGeodis/blob/1.0.5/README.md#L21)
