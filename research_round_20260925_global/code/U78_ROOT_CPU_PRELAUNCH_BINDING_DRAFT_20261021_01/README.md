# U78: actual U76-to-U71 prelaunch binding call, unbound source

**UNBOUND / NOT RUN ON RESEARCH INPUTS / NOT LAUNCHED.** This task wrote only
these three ordinary source files. It did not access actual Ops, qualifications,
bound training sources, reports, packages, checkpoints, GT or installed libraries;
it did not run SSH, a child process, Torch, NumPy, COCO or model evaluation. The
six finite tests use invented metadata/bytes and temporary output files only.

This fills the previous *source-level call-site gap*: the function below really
calls frozen U76 revision02 and immediately constructs/checks the same original
U71 spec from its returned four proof refs and the identical CPU request. Merely
having this source or staging it is not an actual invocation or launch gate.

## Fixed interfaces and sources

```python
prepare_binding(request, *, read_raw)   # no file I/O; returns spec/admission/record
write_binding(request, *, read_raw)     # calls prepare itself, then exclusive save
```

Request schema: `Root-U78-CPU-prelaunch-binding-request-v1`, exactly13 keys:

| Key | Meaning |
|---|---|
| schema / phase | Fixed request schema and original finite U71 phase |
| admission_source_ref | Frozen U76 revision02 code |
| acceptance_source_ref | Frozen U75 original validator code |
| entry_source / owner_source | Frozen U71 CPU execution sources |
| proof_refs | Four genuinely Root-accepted U75 artifacts in original range order |
| prediction_refs | Their four full original U66_SHARD.json artifacts |
| expected_endpoint | Entire same accepted FORMAL E50 endpoint, including CP/source identities |
| cpu_request | Unchanged original U70 seven-key request, actual source/library/input refs |
| output_directory | Future U71 ENTRY output, canonical POSIX |
| owner_output_directory | Future original U71 OWNER output, canonical POSIX |
| persistence_directory | Fresh local directory for this preparation's files |

The future WORKER output remains cpu_request.output_directory. No source table,
prepared spec, external proof replacement, arbitrary validator/runtime module or
configuration override can be supplied instead of constructing the original
spec. Input dictionaries are copied before callbacks, and raw bytes are cached
only after length/SHA verification, so later caller mutations do not rebind them.

Fixed whole-code content pins:

| Source | Bytes | SHA256 |
|---|---:|---|
| U76 revision02 |9071|`9b7ae0dbd2605598d9f83d5f31f8805aad01ff2c37006e427138af0f68d2f75b`|
| U75 acceptance |34428|`fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2`|
| U71 entry |23263|`01d13ff489ad6836d4e16bf969069bc966341671f53c5b1e44b5c1509d91acc5`|
| U71 owner |26829|`bf6e48fac5560d4f9320bb52e68bd84d1383f70154098ce62179b9f8c4a0e798`|

U71 itself enforces its frozen U70 worker pin, original five CPU limits and exact
seven-key U70 request. Its genuine close_source_union executes only the closed
U70 *stdlib metadata definitions* to validate original roles, source pins and the
direct union. U70.run_worker, U71.main/execute_owned, owner.main and all numerical
functions are never called. The pinned owner is byte-verified, never executed.

## Actual call path and finite byte boundary

1. Byte-close fixed U76/U71 source definitions and the fixed U71 owner. Root's
   read_raw(ref) returns exact original bytes for explicitly declared refs. This
   callback is a trusted transport boundary, not authority to supply qualified
   booleans. Every returned byte array is independently length/SHA verified.
2. Actually call original U76.admit_population on its exact9 original inputs:
   fixed U75 code +4 original proof bytes +4 original full U66 bytes. Its original
   U75 validator verifies the embedded raw evidence and all single-shard joins;
   U76 enforces common actual manifest, full endpoint/source identities, ranges,
   original187 IDs and shared CPU population/mask whole refs. These rules are not
   reimplemented or bypassed here.
3. Immediately construct the original U71 exact11-key envelope. Its external
   evidence list is copied directly from *this call's* U76 result, preserving
   ordinal whole refs. Its request is the same canonical CPU request, checked
   against the original U76 CPU-request digest. Sources are built from original
   U71.base_references, followed by the real validate_envelope and
   close_source_union. No caller-built source table is accepted.
4. Return the spec, the unmodified admission result and preparation record. The
   record retains exact CPU-request/binding-request metadata digests, fixed code
   refs and the finite actual read set. Only13 declared source/artifact files
   are read in the distinct-reference test fixture; this is not a hash of all
   runtime source/library/data refs. Original U71 owner still byte-closes those
   runtime inputs at its actual future admission.

No nested historical ref, CP, library payload, native/R prediction, original
annotation or DEV package is followed. The API can receive original bytes via
Root's explicit cross-host transport mapping. That does not change any actual
ref identity. All refs inside the generated executable U71 spec must pass its
original canonical-POSIX contract. U76/U75 control-source refs may identify local
transport files; they are not added to U71's scientific source union.

## Exclusive save and exact future launch artifact

write_binding prepares internally; it does not accept a previously returned
mutable spec for persistence. It requires a new canonical local directory with
an existing parent, rejects reused/existing/symlink output, and creates it
exclusively. The persistence directory and three original execution outputs
must be separate/non-nested and cannot enclose any explicit input. Foreign-host
execution output absence is **not** inferred from these metadata comparisons.

The new directory contains SPEC.json, ADMISSION.json and PREPARATION.json. Each
file is exclusively created, flushed and fsynced. A POSIX directory is also
fsynced; Windows has no claimed POSIX directory-fsync guarantee. PREPARATION
retains exact saved spec_ref/admission_ref and CPU-request digest. The returned
persisted_refs additionally contains the final preparation record's own ref.
On a write failure, original partial files are preserved and a small FAILED.json
is attempted; original exception/trace takes precedence over any audit-write
failure. An incomplete attempt is never retried or reused.

Explicit same-host CLI (future use only):

```text
python -B binding.py --request /explicit/ROOT_REQUEST.json --request-sha ORIGINAL_SHA256
```

It verifies the actual original request-file hash before any referenced input
or output action. All subsequent reads use only explicitly referenced canonical
same-host files. The CLI does no implicit host/path relocation; Windows Root may
instead call the API with an explicit original-byte reader. Saving a Windows
transport copy never replaces any POSIX Source path inside SPEC.json.

**Future Root launch must consume this successful call's exact saved SPEC bytes
and SHA, not reconstruct a dict or replace any request/proof/source ref.** If the
spec is byte-exact staged elsewhere, Root separately records and verifies that
actual deployed spec_ref; source paths remain unchanged. Only original actual
tool success/terminal evidence can establish successful preparation. A lone file
or prepared status, especially after an interrupted/FAILED attempt, is not that
observation. Root still performs actual original-terminal/CAS/account admission,
remote fresh-output/resource checks, launch and independent post/acceptance.

## Evidence limits and finite tests

PREPARED_U71_SPEC_NOT_LAUNCHED is a construction record, never a new execution
qualification. qualified/AP/CPU_launched/GPU_launched/CAS_performed remain False;
training credit is zero. Preparation's successful source-level call proves no
library ABI, runtime source closure, CPU evaluation, metric replication or method
efficacy. Pure JSON cannot authenticate that Root actually observed the embedded
terminal facts; Root must select four genuinely accepted original artifacts.

Tests reuse U76's public synthetic fixture and execute the exact frozen U76/U75
validators and original U71 metadata functions, not their old test suites. Those
invented U75 components have fake population/mask hashes. Test-only code changes
two SOURCE_PINS globals solely inside an ephemeral, already byte-closed U70
metadata definition to admit those harmless invented refs. No production pin,
source byte, U76/U75 validator, U71 union function or runtime is replaced. This
fixture exception is not real Source qualification; production has no such mode.

New finite test-first observations:

- Missing binding module: `1b78aa` exit1 → real call path `52d08d` exit0.
- Persistence feature absent (required os boundary missing): `5ab830` exit1 →
  exclusive save/fsync/no-reuse `1c4fe6` exit0.
- Explicit same CPU-request digest absent: `345aa7` exit1 → `6f4d01` exit0.
- Persistence/input nesting: `00cfcb` exit1 → `876792` exit0. Initial third
  assertion also exposed a test regex case mismatch against an existing U71
  rejection; it was corrected to the actual U76-only control-source collision,
  not claimed as a new U71 implementation failure.
- Missing explicit CLI: `bdb8e6` exit1 → request-SHA gate `9b4e76` exit0.
- Fixed source/actual admission failures before persistence and original-error
  partial evidence checks passed on their first run using the existing guards;
  no fictitious implementation RED is claimed.
- First full six new tests: `1b3a03` exit0,7.427s. A final finite run and static
  run after the explicit pre-read absolute-path check: `9b3b51` exit0, six tests
  in7.468s. Source-only AST/selected-code-pin check `5c513a` exit0 confirmed
  U76→construct U71→validate_envelope→close_source_union order and the unchanged
  four original ordinary source pins. No scientific or process execution.
