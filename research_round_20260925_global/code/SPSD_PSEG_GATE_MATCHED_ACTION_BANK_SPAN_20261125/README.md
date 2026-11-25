# Frozen gate action span diagnostic

Use the known closed matched-action gate, FIT action bank source and closed TRAIN915 capture and source gate points. Use the same project spatial attraction source. The procedure imports the exact FIT implementation solely for its label-blind action_bank entry and consumes the existing final gate RLEs. No neural head is called or changed.

- Run span_diagnostic_retry.py with the CPU runtime, CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Invoke Python using -I -B -X utf8. Pass --gate, --fit, --points, --capture, --attraction, --protocol, and a new --output. Inspect --help for path argument names.
- Use the frozen PSeg attraction observe.py with --phase score to bind the actual worker process, output hash and natural exit. This observer phase name is generic and no reference scoring occurs.
- Run test_span_diagnostic.py using the same CPU runtime. It checks independence of bank direction vectors, final raster sign on removal cells, and unchanged decisions under full quota.

Only the result summary, protocol, scripts and ordinary documentation are releaseable. Original streams, checkpoints, raw labels and paths with individual identifiers are not packaged here. The retained first report has an erroneous runtime field. RESULT_SAFE_RETRY.json is authoritative. Comparison excluded elapsed_seconds and worker_sha256 and confirmed all other fields exactly equal.
