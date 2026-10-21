# U81: consume one actual saved U78 binding, then launch original U71

**UNBOUND / NOT RUN ON REAL BINDINGS / NO REAL SSH OR CPU EVALUATION.** This
source task accesses only ordinary frozen definitions and synthetic fixtures.
It does not access actual Ops, bound Source, reports/qualifications, data, GT,
checkpoint, installed libraries, real procfs or SSH. No numerical task, final
CPU/AP acceptance or research gain was produced. Root exclusively controls
ongoing numerical training; nothing here changes its trajectory or budget.

The shared source is fixed U80 revision02:26813 bytes, SHA256
`11b0e17a28a9a6e4b9f9493894c24cc69119fc0ec6db647f3d3db8a92ba6eb81`.
The final production LAUNCHER_PIN is exact, with no wildcard, inert placeholder
or fixture-source mode. The frozen U78 10913-byte source pin is also enforced.

## Production route and explicit original inputs

Only this CPU route is intended for Root production use:

```text
python -B launch.py --request-file /explicit/U81_REQUEST.json --request-sha256 ORIGINAL_SHA256
```

The CLI verifies the original request bytes/SHA, then calls
`launch_saved_binding(request, *, read_raw=None, transport=None)`. No import-time
I/O occurs. read_raw defaults to strict same-host canonical file reads; a Root
API caller may provide an explicit byte-exact transport reader. This does not
rewrite any actual ref identity. transport is forwarded unchanged to U80; its
default is the real SSH implementation, while tests use explicit stubs.

The request has exactly seven keys:

| Key | Required identity |
|---|---|
| schema | `Root-U81-U71-saved-binding-launch-request-v1` |
| binding_source_ref | Frozen U78 binding.py,10913 bytes, SHA `0b62b3b6ebb9e2fef0dc6240f876d66dae58b02cddce9f7f145f332429f82899` |
| binding_request_ref | Actual original U78 request file bytes, not a replacement dict |
| persisted_refs | Exactly spec/admission/record, the whole refs actually returned by that U78 write_binding call |
| binding_terminal | Root's actually observed original preparation tool facts below |
| launcher_source_ref | Frozen final U80 source ref, verified before definitions execute |
| launch_request | Original U80 exact ten-key CPU launch request |

binding_terminal has exactly root_observed_tool_exit_code (strict integer0),
root_observed_tool_chunk_id (nonempty), binding_source_ref, binding_request_ref
and persisted_refs. The latter three must equal this request's original inputs.
Owner-returning text, a prepared boolean, a future/planned exit or merely finding
JSON files is refused. **JSON cannot authenticate that Root truly observed an
original tool terminal. Root must supply genuine observations and select genuine
original artifacts.** No helper manufactures them from its own pending output.

U80 launch_request keeps its original schema and exact fields: schema, mode,
spec_ref, spec_copy_ref, transport_config_ref, authority_ref, expected_endpoint,
expected_final_acceptance_ref, acceptance_source_ref, persistence_directory.
mode is fixed U71_CPU. spec_copy_ref must be the original U78 saved SPEC whole
ref. spec_ref is its separately verified actual POSIX deployment ref, with the
same bytes/SHA. The private transport configuration supplies actual host/key/
account details at future runtime; none is hardcoded or published in source.

## Saved-artifact consumption, without rebuilding SPEC

U81 loads the fixed U78 *stdlib definitions* to check their original request
schema and four fixed source pins. It never calls U78.prepare_binding again or
rebuilds a spec. Every supplied raw byte array is independently length/SHA-bound
before parsing, and callbacks cannot change the copied request mid-call.

The original SPEC, ADMISSION and PREPARATION paths must match that original
request's persistence directory; recorded spec/admission whole refs must match
the actual supplied files. PREPARATION must remain admitted metadata only,
unlaunched/unqualified, with original source identity and CPU/request digests.
The same canonical CPU request must appear in SPEC and the original U78 request,
and its digest must equal both original saved records. The four proof refs and
four full-shard refs must match by ordinal across original request, ADMISSION
and saved SPEC. Original phase/output/source identities are retained.

The whole expected endpoint must equal both original U78 input and ADMISSION,
including original CP/Source identities. It must remain FORMAL37050/50/0, and
expected_final_acceptance_ref must equal its original training_qualified ref.
The fixed U75 content pin remains unchanged at the explicit launch-host ref.
No checkpoint or historical nested source is opened by this adapter.

The original U71 entry is byte-closed and its actual validate_envelope and
close_source_union functions run on the parsed saved spec. Only original U70
metadata definitions are closed for that union check; no entry.main, worker,
model or metric runs here. Owner bytes are verified but never executed locally.
The launch receives the original saved spec_copy_ref and its byte-exact actual
remote spec_ref, not a newly serialized dictionary. SPEC's POSIX Source paths
are neither translated nor replaced during transport.

## One shared actual execution boundary

After these checks U81 really invokes frozen U80.launch_cpu_saved_spec. It does
not return a planned command, add a second SSH/Popen framework or accept a
caller-injected validator. U80 owns the one actual preflight/authority-active/
SSH exec/binary-log/child.wait boundary. The CPU mode uses original U71 entry,
owner, five CPU limits and saved spec SHA. U80's own CLI is GPU-only; the CPU
API is exposed for this fixed U81 production route, not as a second direct CLI
which bypasses the genuine U78 preparation. This is a reviewed Root workflow
boundary, not an invented capability system against a malicious Root caller.

U80 must read the actual configured Root authority: all leaves reaped and no
active GPU/training/phase, strict FORMAL prefix37050 and the same actual final
training acceptance. CPU additionally needs four actual U79 committed rows,
indices0..3 in order, whose whole proof/prediction refs equal this saved spec's
proof/shard arrays. A midpoint or missing/duplicate/reordered row must refuse
before owner launch. The private config authority path must equal its actual
explicit request ref. Declared JSON alone is not Root-origin authentication.

Original remote stdlib preflight, all actual source hashes, fresh outputs and
resource gates precede the one SSH owner call. Profile-active state is committed
before that call; disconnection/unknown completion does not clear it or declare
all processes absent, and there is no automatic resubmission. The CLI returns
the actual shared SSH_returncode, including255. Negative signal values remain
in the report, with the operating system's normal SystemExit mapping; none is
rewritten to zero. owner_process_exit_code and outer tool/chunk remain unknown
until independent genuine Root observation. Shared launch completion is still
pending acceptance; it proves no CPU/AP success by itself.

## Finite synthetic verification

Fixtures use genuine frozen U78 write_binding and original U76/U75/U71 metadata
definitions on invented artifacts. As in the U78 test, only two pin globals in
an ephemeral U70 metadata definition are adjusted for fictitious population/
mask components. Production sources/pins and original validator functions are
not modified. Isolated adapter tests temporarily replace the launcher pin with
a tiny harmless boundary stub; that is not genuine U80, process or SSH proof.
The final dedicated integration must instead use fixed real U80 definitions
and its synthetic transport/authority fixtures, without running its old suite.

Observed new RED/GREEN:

- Missing adapter source: `1af697` exit1 → `8a9689` exit0.
- Missing genuine-terminal binding guard: `ca6da7` exit1 → `6f5159` exit0.
  Intermediate `4fc197` exposed a fixture sharing one mutable persisted-ref
  mapping between expected and observed inputs; independent copies fixed that
  fixture without further production logic.
- Whole endpoint/proof order/preparation scope: clean five missing rejections
  `fff50d` exit1 → `8bc6df` exit0. Earlier `a42016` used an assertion inside the
  assertRaises block; that fixture structure was corrected before the clean RED.
- Missing explicit CPU CLI: `ea662c` exit1 → `f47e81` exit0, including actual
  returncode255 propagation through the CLI boundary stub.

The genuine pinned shared-code integration first failed with development inert
pin `2b3927`, then fixed U80 revision01 reached its real CPU launcher and exposed
a cross-host local-file ABI defect: `0c6819` exit1. U78's original Windows
persisted ref uses str(Path), while U80 revision01 compared its normalized
as_posix ref as a whole. Same actual file/bytes/SHA was refused. The fixture did
not rewrite or normalize U78's original ref. Root authorized an independent U80
revision02 local-reader correction; original U78/U80rev01 stayed unchanged.

After only changing the U81 fixed U80 pin and fixture source selection to that
revision02, the same genuine shared path passed: `81df0e` exit0. It executes the
original U78 saved-file consumer and actual frozen U80 CPU launch implementation
with its explicit SyntheticTransport, not a stubbed launch function. Temporary
authority prefixes7168 and a missing committed shard are refused at their actual
authority guards before preflight/popen. A valid synthetic case actually reaches
preflight→popen→wait once, preserves binary stdout containing NUL/0xff and leaves
the scoped attempt active. AP and owner OS exit remain unknown. No real remote
preflight, resource probe, network, process, authority or evaluation is involved.

Only the new U81 suite is run; no U80/U78/U76/U75 old suite is repeated. Final
five new tests `136122` exited0 in3.414s. AST/selected-source pin check `1ae057`
exited0, confirming the single shared CPU call, original U71 metadata gates and
absence of a second process/CAS/rebuilt-spec route. Final artifact identities
accompany the handoff. No actual final
endpoint, four real accepted shards, CPU launch or final AP acceptance exists in
this source task. Prepared JSON cannot bypass the actual current Root authority.
