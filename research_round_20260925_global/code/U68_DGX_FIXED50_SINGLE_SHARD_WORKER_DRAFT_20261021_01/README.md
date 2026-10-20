# U68 future DGX single-shard prediction worker

**UNBOUND / UNEXECUTED.** This task imported no real Torch, NumPy or
pycocotools, read no real inputs/checkpoints/qualification chains, ran no model,
SSH or old test suite, and changed no frozen U40/U58/U59/U66/U67 or running U57
Source. The new worker connects explicit source loading and JSON I/O to the
existing U66 numerical core. It introduces no renderer, model or metric math.

```python
worker.run_worker(c, request, report={})
```

This is a future **worker body**, not a standalone CLI, owner, launcher, binder
or Root acceptance implementation. `c` must already be a genuine admitted
Linux control supplying `check()`, byte-verifying `closed(ref)` and the actual
`psutil` module. All process ownership, birth/session/group checks, hard deadline
and resource monitoring, original exit collection and Root acceptance remain
external. No caller receipt can turn synthetic tests into a real execution.

`request` has exactly ten keys:

- `worker_source`: this exact U68 source ref.
- `prediction_core_source`: the frozen U66 source ref (SHA
  `71e10c6539f34105c9726c73e9a0953973a76334da07059cf7a5fc2b619606be`).
- `source_refs`: U66's exact21 roles. Twenty known source SHA pins are retained;
  `encoder` is an explicit Root-pinned original encoder ref.
- `library_sources`: exactly `torch`, `numpy`, `psutil`, `pycocotools` (package
  initializer), `mask_extension` (the actual `_mask` binary). The mask Python
  source is already `source_refs.mask_api`. Only the package initializer may
  have zero bytes, and its exact SHA is still verified. This selected direct
  closure is not a full Python/OS/binary dependency closure.
- `endpoint_refs`: exactly `qualified`, `spec`, `pair_commit`, `order`, using
  the original U58 metadata ABI and original Root-U57 accepted endpoint schema.
- `checkpoint_refs`: original `PLAIN` and `FULL` refs, joined to those actual
  endpoint documents. Only FORMAL E50/37050 is admitted by original U58 checks;
  SMOKE, midpoints and the Windows endpoint schema are not substitutions.
- `manifest_ref`: the explicit portable manifest with original `entries`187.
  Extra top-level historical primitive metadata remains untouched.
- `shard_index`, `dataset_range`: exactly one original range, `[0,47]`,
  `[47,94]`, `[94,141]` or `[141,187]`, retaining all100 queries per image.
- `output_directory`: a fresh canonical directory with an existing parent.

Each ref is canonical POSIX `path/bytes/sha256`. Root must first bind the
manifest and `direct_references(request, entries)` to the future owner's actual
source allowlist. This pure metadata function returns the exact deduplicated
union of code/libraries/four JSON/two CP/manifest and only that shard's package
refs. All187 entries are identity/permutation metadata checked; packages are
actually opened only by the original U59 reader inside U66, for the selected
shard. The worker never recursively reads FIT sources or original Windows raw
provenance. The two historical DEV readers are closed, never executed.

All declared numerical code and selected library bytes are checked before
scientific imports. Genuine libraries must be the explicitly referenced actual
modules, with Linux/aarch64 Python3.12.14, Torch2.10.0+cu130,
NumPy1.26.4 and psutil6.1.1; CUDA starts cold, CPU affinity is0/1,
CUDA_VISIBLE_DEVICES=0 and thread environment values are2. Original source
definitions are loaded under the known `dsconv/head/model_full/...` aliases;
their dependency objects are checked and previous aliases restored even on
failure. Root still owes the genuine transitive environment admission.

U68 reads the four closed endpoint JSONs and manifest, invokes the original
U58 final-endpoint/source checks, then calls **U66.predict_shard once**. U66
still owns actual cold CPU pair construction, strict restoration of both
trained heads before CUDA initialization, original four-query numerical
readout/zero-RLE calibration/nonzero trained residual prediction and per-arm
and final GPU release. U68 does not unfreeze, reset the head, move/reorder
reader tensors or replace any mathematical function. Outer Python/NumPy/CPU
RNG, dtype/grad and backend flags are restored; each restoration is attempted
even if another fails, and the original prediction exception remains primary.
On failure U68 first saves `traceback.format_exc()` as plain text in
`report.error.traceback`, then clears frames and the exception traceback to
release object/tensor references. This preserves only the stack still visible
on entry to the U68 handler; it cannot reconstruct internal U66 frames that the
frozen U66 core has already cleared.

Successful output is exclusively written as:

- `U66_SHARD.json`: the **unchanged complete U66 JSON shape**, including both
  arms, top predictions/identity/Omega/cost aliases and all per-image proofs.
  U67 can consume this body directly after Root's separate real acceptance.
- `REPORT.json`: U68 worker I/O/cleanup report, pending Root acceptance only.

Writers use exclusive files, complete writes, flush/fsync and exact byte/SHA
refs. Existing attempts are not overwritten or retried. On failure a created
attempt retains its partial prediction evidence and a failed report where I/O
remains possible; original exceptions propagate. A pending shard file can
exist if later outer cleanup fails: it **must not** be treated as accepted
without the corresponding successful worker and genuine external terminal.
Both reports have no self qualification, no AP, TEST or training credit.

The remaining **actual execution gaps** are explicit:

1. No final E50 endpoint is supplied or accepted by this source task. Root must
   join real final documents/CPs and the actual full187 input manifest.
2. The original encoder ref must be traced by Root to the already checked
   original DEV `context.encoder_source` before binding. The portable `entries`
   schema alone carries no encoder identity. Function presence or zero identity
   does not prove original encoder provenance; no context path is guessed here.
   Closing the supplied encoder bytes authenticates only that supplied file,
   not its original DEV semantic provenance; Root must independently confirm
   the latter from original metadata before binding. No actual context, spec
   or qualification chain was read to fill this gap in the present task.
3. U60's existing Linux owner is hard-locked to SMOKE2 CPU restore and cannot
   execute U68 unchanged. A separately reviewed original supervision connection
   is still needed, including CPU01/two threads, CUDA allocator4GiB,
   RSS16GiB/freeRAM16GiB/1800s and real termination evidence. This body creates
   no replacement owner and grants no permission to bypass one. GB10's absent
   whole-device VRAM observation is not zero or passed Windows8GiB certification.
4. This task does not run the already-existing nonzero coordinate contract or
   prove DGX forward/RLE equivalence. After four genuinely accepted shards,
   genuine full-population CPU COCO evaluation through U67 still needs its
   original native/strong references, annotation object, libraries and controls.

Finite verification used only this new six-test stdlib/synthetic suite. Harmless
source definitions, invented JSON and fake runtime/control objects are explicitly
test-only; the genuine scientific import function was never called. RED/GREEN:
missing worker `f54d62` exit1 -> pending JSON `3a57ef` exit0; missing source loader
`845a97` exit1 -> `42892b` exit0; missing worker body `dbfa61` exit1 -> `2ec79c`
exit0 (after correcting a relative synthetic output path). An initial loader-test
edit misplaced old test lines (`8b41ef`); the fixture was corrected before the
clean missing-loader RED. Additional failure/state and empty-initializer checks
exposed two concrete issues (`08c20d` exit1): only the empty package initializer
is now allowed (`b8a458` exit0), and independent cleanup continues after a
previous cleanup error. Final six tests: `495f37`, exit0. The earlier two failure
scope checks first passed in `ca71c5`; no false RED is claimed for them.
Final AST check `d7a607` exited0 for both new Python files. The preceding audit
`81507c` parsed the worker but incorrectly rejected the test's allowed local
`import worker`; only that audit allowlist was corrected, not worker code.
Independent review then found missing original visible stack diagnostics. The
existing failure fixture was extended first: `fc63c3`, exit1 (`KeyError:
traceback`). Only the U68 handler was changed to save the plain-text stack before
clearing it; the same six tests passed in `fe37cd`, exit0. No old suite or
scientific import was run for this correction.

No scientific success follows from these tests. No files were published.
