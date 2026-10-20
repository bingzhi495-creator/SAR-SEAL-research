# U65 Root CPU reader acceptance — independent unbound D draft

This directory contains only a pure metadata receiver. It does not launch an
owner, connect to DGX, import Torch/NumPy, open packages/reports, or perform ML.
No actual U65 acceptance has been performed by developing or testing it. The
frozen U65 entry/body/Linux owner and all earlier source files are unchanged.

The public API is:

```python
accept_validation(spec, original_terminal, owner_receipt, worker_report, postcheck)
```

All five arguments are supplied JSON-compatible dictionaries. The result is a
new dictionary; inputs are not modified. Missing/invalid evidence fails closed
without returning acceptance. Root remains the sole executor and must save and
hash the returned result separately after genuinely collecting every input.

The exact accepted result schema is
`Root-U65-DGX-DEV187-CPU-reader-input-accepted-v1`; its completion status is
`ROOT_ACCEPTED_ALL187_CPU_READER_VALIDATION`. This means only that the original
CPU input-reader attempt passed these acceptance checks. It is not a trained
model, an accuracy result, a new method, or a TGRS publication claim.

## Genuine terminal observations remain external

`original_terminal` retains the original U54 exact eight-field ABI, without
relaxing its meaning:

* Integer zero, never boolean/unknown: `root_observed_tool_exit_code`,
  `SSH_returncode`, `worker_returncode`.
* Exact `True`: `reaped`, `owner_original_birth_absent`,
  `child_original_birth_absent`, `all_owned_processes_absent`,
  `source_and_logs_byte_verified`.

Root must obtain these from the real synchronous tool/SSH/owner/worker attempt
and independent original-birth/group postchecks. In particular, the SSH terminal
must reflect the actual owner command's exit. An asynchronous launch, injected
zero, receipt `controller_returning`, or owner self-description cannot establish
the owner's OS exit. Owner and child Linux PID/start-ticks identities in
`postcheck` must equal the receipt's original identities; the child must have its
own original session/process group, with the recorded owner as its parent.

The owner must report actual child exit zero, successful wait/reap, original
birth absent, final group scan resolved, no remaining owned processes, signals,
resource violations, errors, retry or termination request. Neither worker
`COMPLETE` nor a successfully transferred `.pt` file substitutes for actual
genuine U59 `prepare`/`load` calls in the original worker.

## Spec and direct byte closure

The unchanged admitted spec is
`ROOT-U65-DGX-full-DEV-CPU-reader-validation-spec-v1`, operation
`CPU_U65_DGX_DEV_FULL187_PORTABLE_READER_VALIDATION`. Its exact fourteen keys are
`schema`, `operation`, `phase`, `entry_source`, `body_source`, `owner_source`,
`source_refs`, `library_sources`, `manifest_ref`, `package_refs`, `prelock`,
`output_directory`, `owner_output_directory`, `sources`.

`source_refs` has exactly `portable_reader`, `dev_reader`, `dev_rgb_reader`,
`image_geometry`; `library_sources` has exactly `psutil`, `torch`, `numpy`.
The 187 package refs preserve original ordinal order. Direct refs comprise those
187 files, the four source roles, three library roles, and entry/body/owner/
manifest. Their canonical sorted union is derived from the supplied spec;
cross-role path aliases/conflicts are rejected. These roles yield 198 distinct
direct files, without a blindly accepted count field. They are selected-source
closure, not complete Python/OS/shared-library closure.

Root must actually byte-check this entire exact Source union, plus the spec,
worker report, owner receipt, private audit and both owner logs. Ref metadata is
exact canonical absolute POSIX `path`/integer `bytes`/lowercase SHA256. This
module only hashes supplied reference/JSON metadata, never the referenced files.

The prelock is exactly the frozen U65 `U65-DEV187-CPU-reader-validation-prelock-v1`:
187 indices `0..186`, 18700 queries, Python 3.12.14 / Torch 2.10.0+cu130 /
NumPy 1.26.4 / Linux aarch64, hidden CUDA, CPU 0/1, two threads, RSS at most
16 GiB, available RAM at least 16 GiB and deadline 1800 seconds. All original
worker restoration/source/resource flags and five body flags must be true.
Both worker and owner sampled resource evidence and all durations are checked;
the owner's retained child evidence must equal the actual worker report.
This is acceptance of the existing supervision, not a claim of an OS RSS cap.

## Postcheck and actual scalar audit

`postcheck` has exactly thirteen keys:

* `schema`: `Root-U65-explicit-byte-postcheck-v1`.
* `spec_ref`, `worker_report_ref`, `owner_receipt_ref`, `private_audit_ref`.
* `logs`: exactly `stdout.log` and `stderr.log` references.
* `verified_source_refs`: the entire canonical direct Source union.
* `verified_output_refs`: canonical six-file union of spec/report/receipt/audit/
  stdout/stderr.
* `owner_identity`, `child_identity`, `source_and_logs_byte_verified`.
* `manifest`: JSON parsed from the actual byte-verified manifest.
* `private_audit_records`: all JSON objects parsed, in order, from the actual
  byte-verified `IMAGE_CHECKS.jsonl`.

Root must read and hash genuine manifest/audit bytes before parsing. A synthetic
or independently regenerated list must never be passed as actual records. The
receiver cannot independently determine whether Root supplied truthful external
observations. Its tests intentionally use invented fixtures and grant no actual
execution evidence.

For each original ordinal, the audit's exact ten fields are `dataset_index`,
`image_id`, `queries`, `status`, `source_package`, `original_record_sha256`,
`native_predictions_metadata_sha256`, `query_ids_metadata_sha256`,
`CUDA_initialized`, `returned_tensor_objects_released`. Each row must join its
manifest index/image/package, preserve the original full-100 query permutation,
have 100 queries, `VALIDATED`, cold CUDA, and actual release marked true. The
body's manifest/complete original-row canonical hashes are checked. Native
prediction hashes remain opaque worker-produced digests; no mask reconstruction,
package deserialization or replacement field proof is invented here.

The receiver re-encodes parsed audit rows using the frozen Linux entry's exact
ASCII sorted compact JSON plus LF format and verifies its length/SHA against
the actual Root-verified audit reference. This guards row substitution as well
as partial, duplicate or reordered rows. Counts alone cannot pass.

Body attempted/validated/released/recorded counts and worker/owner validated/
audit counts must all be 187, queries 18700, actual visits exactly `0..186`,
failed loads zero and remaining returned tensor objects zero. The original U59
four-field CPU FP32/identity/native/geometry validation and weak-reference
release evidence come from the genuine frozen reader/body attempt, not from
these metadata checks alone.

## Scope and finite tests

`DEV_technical_input=True` explicitly discloses reading DEV inputs. `DEV=False`,
`AP=False`, `TEST=False` mean no performance evaluation. Model construction,
forward, optimizer, new GT rasterization/matching and training credit must all
remain absent/false/zero as specified. Legacy matched-instance/target-read
scopes are rejected. `original_annotation_records_parsed=False`,
`raw_provenance_inputs_revalidated=False`, and original metadata population
`{images:187, queries:18700, tensor_population_revalidated:False}` remain exact.
The last value describes the inherited metadata population field; the current
worker separately must genuinely load and validate all187 portable packages.

Windows provenance inside packages remains historical metadata. It is not
re-opened/re-authenticated here or by the DGX portable reader. Earlier original
export readers may have contacted GT-containing containers; this receiver does
not claim that the historical end-to-end flow never touched GT.

The five finite stdlib tests cover full-187 metadata joining, the changed query
and partial/release boundary, legacy GT-scope refusal, direct-source aliases,
and actual external-terminal requirements. They do not execute Torch,
`prepare`/`load`, SSH, actual reports/data, or owner processes. Earlier U54 full
tests are not duplicated. The source was adapted from its ordinary acceptance
module while preserving the external terminal and resource guard semantics.
