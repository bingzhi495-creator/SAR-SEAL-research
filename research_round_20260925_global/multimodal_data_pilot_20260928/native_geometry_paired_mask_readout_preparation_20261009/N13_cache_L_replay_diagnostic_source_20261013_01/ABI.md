# N13 fixed-index CPU L replay diagnostic

UNBOUND SOURCE only. ROOT alone binds and performs the finite owned numerical
diagnostic. Frozen cache/qualification v01 is unchanged. This SOURCE outputs
no Q and cannot admit cache to training.

`SCHEMA = ROOT-N13-cache-L-replay-CPU-diagnostic-spec-v1`

`REPORT_SCHEMA = N13-cache-L-replay-CPU-diagnostic-report-v1`

`REF_KEYS` (all absolute `{path,bytes,sha256}` in the exact `sources` closure):

```text
entry_source
cache_GPU_entry_source
cache_GPU_spec
cache_GPU_report
cache_GPU_RAW
cache_GPU_admission
failed_CPU_source
failed_CPU_spec
failed_CPU_RAW
failed_CPU_admission
packet
carrier
```

`RECIPE` is the exact static `dict(...)` literal in `diagnose_L_cpu.py`:

```python
dict(dataset_index=3,normal_RAW_count=38,failed_RAW_count=5,
 variants=['CPU_FP32_original_two_stage','CPU_FP64_two_stage_cast_FP32'],
 rtol=1e-4,atol=1e-4,top_errors=10,quantile=0.99,
 preserve_original_decisive_FP32_check=True,
 fail_after_report_if_original_check_fails=True,
 output_cache_qualification=False,no_GT=True,no_model=True,no_CUDA=True)
```

Other SPEC fields: `phase`, `sources`, `RAW_inputs` exactly38 normal rows,
`prior_failed_CPU_receipt`, `additional_failed_RAW_inputs` exactly4 failed rows,
`terminal_identity_pins` exactly43 path-keyed original pins,
`CPU_admission_file`, fresh absolute `output_directory`,
`deadline_seconds=1800`, `allocator_limit_MiB=4096`,
`whole_GPU_limit_MiB=8192`. `normal_RAW_count=38` may also be included as an
explicit metadata field. Each failed row has `{receipt,phase,returncode:1}`;
each identity pin has `{phase,returncode,original_identity,admission_sha256}`.

All original identity/admission/held return/reap/birth-absence and whole-resource
RAW checks precede the first SHA, scientific input or numerical import. After
the gates, original normal admission/PB/bootstrap/SOURCE/SPEC and the failed
CPU's exact source/spec/admission are joined. Original GPU is normal index37.
The failed CPU is the final failed row. Only original report row3's exact
packet/carrier paths are numerical inputs. Actual Torch/NumPy/psutil PY/PYD/DLL
files must be source-bound; own CPU01/threads2/empty CUDA and original driver
SHA/1800 seconds admission are required. No common SOURCE gate is invoked.

Only query IDs, native L/M, carrier identity and their tensor digests are
accessed numerically. No GT annotations are opened, no label field is accessed,
and no model/weight/JPEG/matching/region selection is executed. Deserializing
the core point packet necessarily also creates its unused supervision buffers;
they are not read as scientific labels. The sidecar F256 is unused.

Both variants use200→800 bilinear `align_corners=False`, metadata crop, and
`crop[:,None]`→original image dimensions. The first reproduces the original CPU
FP32 check. The second computes both interpolations in FP64 then casts to FP32
for diagnostic comparison. Saved actual GPU L is the unchanged reference.
The same original `rtol=atol=1e-4` is applied to both, and only the first controls
the terminal. No speculative CUDA-FMA emulator or synthetic input is added.

`REPORT.json` records variant original-allclose result, max/mean/p99 error,
elementwise violations and per-query counts, max error/allowance ratio, top10
ratio coordinates/actual/expected/allowed values, sign/band flips and the
pre-existing practical near-zero/near-band scopes, shape/stride/range,
identity/provenance/resource/import fields. No revised tolerance or cache Q is
produced. The report status explicitly says `NOT_CACHE_QUALIFIED`.

After saving the report and printing its ref, the SOURCE always checks the
original FP32 allclose result. Failure exits1, preserving a sixth actual failed
attempt and normal38 if ROOT's external held/reap/birth checks confirm it.
Even if the diagnostic unexpectedly exits0, it establishes no cache
qualification and cannot be substituted for the failed qualifier. No retry is
automatic and no current terminal is self-certified.
