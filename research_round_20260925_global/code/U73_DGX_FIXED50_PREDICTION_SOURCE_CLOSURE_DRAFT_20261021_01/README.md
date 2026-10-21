# U73: original restoration source closure, unbound draft

This is a new private thin-I/O version of U68. No actual endpoint, checkpoint,
library installation, data, qualification chain or SSH was accessed. No scientific
library, model, process or GPU was started. Synthetic tests do not establish an
actual E50 prediction, runtime ABI, AP or Root qualification. Frozen U68/U69,
U58/U59/U66 and the active training sources were not modified. This task does not
publish or bind a future run.

## The concrete missing source

Original U58 builds `endpoint.restore_source_refs` from original U57 body/entry
plus all twelve original training components: fourteen refs. Its restore path
closes all fourteen. The twelve injected restore modules cover twelve of these;
the other two are the original geometry source and original FIT portable reader.
Geometry was already in U68's twenty-one roles. The FIT reader was absent: U68's
`portable_reader` is the different U59 DEV package reader.

U73 adds exactly `source_refs.fit_portable_reader`, with SHA256
`b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557`
and exactly 19618 bytes. It is only byte-closed. It is not imported or put into
`LOAD_ORDER`, module aliases, or the model's inputs. No FIT package is opened.
The original source's ordinary basename is `portable_reader.py`; it must not be
substituted for the U59 DEV `package_io.py` source.

## Existing API, explicit new closure

The ten-key request and public APIs remain
`validate_request(request)`, `direct_references(request, entries)` and
`run_worker(c, request, *, report)`. The full manifest still describes all 187
images; one original range is selected from `[0,47]`, `[47,94]`, `[94,141]`,
`[141,187]`. No owner, CLI, admission document or qualification schema is added.

`SOURCE_ROLES` is now exactly twenty-two roles. `PREDICTION_SOURCE_ROLES` retains
the original twenty-one. `LOAD_ORDER` remains eighteen source modules, plus the
genuine mask module, giving nineteen modules. All twenty-two source refs belong
to the finite direct union and before/after byte checks. Both historical DEV
readers and the added FIT reader are closed without execution.

After the original U58 `validate_final_endpoint` and unchanged twelve-role
`validate_restore_sources` checks, new pure helper
`validate_endpoint_source_closure(endpoint, source_refs)` requires the exact
fourteen-key original restoration map. It checks that current geometry equals
the original geometry ref, and added FIT reader equals the original training
`portable_reader` ref. Equality includes path, bytes and SHA. Same bytes at a
different path are rejected. The helper alone grants no endpoint qualification.
The original twelve complete-ref guards and module `__file__` checks remain.

Only the original twenty-one refs are projected into frozen U66's
`predict_shard`. U66 still receives nineteen modules and its original exact
twenty-one-source contract. Its mathematics, natural query axes, full candidate
population, native scoring/identity, complete output schema and cleanup are
unchanged. U73's outer report uses
`U73-single-shard-worker-report-v1`; it additionally retains `source_refs` with
all twenty-two refs and `original_restore_source_closure_verified`. This flag
starts false and becomes true only after both original and new guards pass.
The original `direct_input_refs`, pending statuses, failure traceback, partial
prediction evidence, resource checks, alias/RNG restoration and exclusive JSON
writes are retained. `U66_SHARD.json` is the original complete U66 body.

Ordinary candidate byte copies are not accepted original paths. Future Root
binding must use the actual final accepted source refs and close the fourteen
required source paths; no original spec or paired identity may be rewritten to
make a new copy look original. Old U69 pins U68 and cannot supervise U73. The
separate future U74 supervisor must pin this new worker and its exact union.
Root still owns original E50 endpoint admission, source/library byte closure,
resource ownership, actual exits and final acceptance. Pending JSON is not
evidence of Root acceptance. No actual spec was created here.

## Finite test-first evidence

Only the five new synthetic tests were run; no historical suite was repeated.
Refs and source bodies are invented test objects. The full worker-path test
patches source SHA pins and scientific runtime acquisition exclusively for the
synthetic fixture. Torch/NumPy are plain namespace stubs; the U66 and original
U58 metadata calls in that fixture are source-backed harmless stubs. These tests
cannot establish genuine original E50 qualification or library compatibility.

- Exact22 finite union / missing or extra role / FIT size or pin rejection:
  old U68 baseline RED `70e761` exit1 -> GREEN `7e3561` exit0.
- Entire geometry/FIT ref equality, not same SHA at a new path:
  RED `ffeecf` exit1 -> GREEN `59d31f` exit0.
- Byte-only FIT closure with nineteen modules and no FIT execution:
  RED `86f829` exit1 -> GREEN `73eddf` exit0.
- Real `run_worker` JSON wiring to an exact21 synthetic U66, retaining outer22:
  RED `24a8c7` exit1 -> GREEN `209810` exit0.
- Three mismatched original refs stop before the synthetic core and preserve
  failed JSON/traceback; this added regression passed first (`1dbf8d` exit0).

Final five-test run: `f2c5b9` exit0. AST check `db2aad` exit0 confirmed unchanged
original `LOAD_ORDER`, twelve restore roles, ranges, core pin and library roles,
plus six unchanged byte/persistence/runtime helper functions. Original U68
remained SHA `46211627ffdf3692e6d8e322013eabbe6c5e3d943d677bb076f46c7b0c056fa7`.
No actual scientific execution follows from these checks.
