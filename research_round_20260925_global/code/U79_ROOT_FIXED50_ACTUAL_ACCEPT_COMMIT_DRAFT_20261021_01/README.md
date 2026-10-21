# U79: actual U75 call, exclusive proof persistence and later Root commit

Source ready, **not deployed or run on real prediction artifacts**. Current DGX
training is 7168/37050 updates per arm, below this caller's fixed-E50 gate. This
change establishes no AP, method gain, qualification, repeat or TGRS readiness.
There is no SSH, owner launch or numerical computation in this module.

`accept_and_persist` is a real call site for frozen U75, not an alternative
receipt validator. It checks U75's fixed 34428-byte/SHA identity, loads only its
stdlib definitions, and calls `accept_shard` using the original spec, original
external terminal facts and U77's later `POST.json`. U75 then closes the actual
five original output files, full prediction, sources, endpoint records and logs.
The caller exclusively writes `PROOF.json` and `PERSISTENCE.json`, flushes and
fsyncs both files and, on POSIX, their directory. Existing directories cannot be
reused. Partial files survive a write failure; the original exception propagates.
Input-containing output directories are refused.

The persistence status is `PERSISTED_AWAITING_ROOT_ACCEPT_TOOL_TERMINAL`. Neither
function nor report can observe its own future tool success or chunk. Root must
retain the original tool handle, observe its actual terminal, obtain and verify
the exact returned artifact refs, and only then supply the real acceptance-tool
facts to the separate commit operation. Original numerical, independent
collector and acceptance-tool chunks must be distinct events. A JSON field is
not authentication that Root actually observed a tool.

`commit_shard` rechecks the fixed U75 source, original proof bytes, full original
prediction bytes and actual persistence report, then calls the original U75
`validate_accepted_proof`. It joins the proof's final training acceptance ref to
the idle current authority's actual 37050-step endpoint. Current 7168-step
authority is explicitly refused. Accepted shard, attempt and proof path are
unique. It adds only the scoped prediction account; all other fields, all
training counts, failures and training acceptance remain unchanged. No CPU is
automatically started and no training credit is assigned. Four stored receipts
still require U76/U78 to check a single full population and construct the exact
CPU request.

Persistence uses a cooperative exclusive lock, expected original SHA and a
second raw-byte comparison immediately before atomic replacement. Root must be
the sole authority writer. This is not a database compare-and-swap against
uncooperative writers that ignore the lock. A failed/stale/partial attempt is
preserved for explicit inspection rather than automatically retried. Cleanup
errors cannot replace a primary write/validation exception. A leftover lock is
not automatically removed by a later invocation.

An error after atomic replacement (directory fsync, re-read or lock release)
can mean the authority **already changed** even though the tool later failed.
Root must inspect the exact authority bytes and preserved artifacts before
deciding recovery; a failed tool is not permission to blindly repeat a commit.
Duplicate protection does not authenticate an unobserved earlier tool terminal.

## Explicit invocation boundary

```text
python -B control.py --request-file /explicit/ROOT_REQUEST.json --request-sha256 ORIGINAL_SHA256
```

The same-host accept request has exactly:
`schema, acceptance_source_ref, spec_ref, postcheck_ref, original_terminal,
persistence_directory`; schema is `Root-U79-accept-shard-request-v1`.
All referenced files must remain their canonical original paths. Do not map
source paths, copy a live source closure for publication, or fabricate endpoint
records to make an incomplete training trajectory admissible.

The later Root commit request has schema `Root-U79-commit-shard-request-v1` and
exact fields `schema, authority_path, expected_authority_sha256, terminal,
accept_report_ref, proof_ref, prediction_ref, acceptance_source_ref,
transport_copies`. `terminal` contains exactly
`root_observed_tool_exit_code, root_observed_tool_chunk_id, accept_report_ref,
proof_ref`. Exit must be integer zero, not a boolean or a planned value.
Each explicit transport copy has `original_ref` and `copy_ref`, preserving the
same bytes and SHA. That binding permits a private raw artifact to be checked
on Root's host without pretending the Windows copy is the original POSIX path.
This interface contains no credentials or machine configuration. Operational
requests, private proofs and raw artifacts are outside publication scope.

## Finite new tests and their limits

Four new tests cover: the actual fixed U75 call with exclusive persistence;
fixed source-pin rejection before call/output; successful isolated authority
commit with preservation, duplicate/stale/boolean-terminal rejection and the
7168-step E50 refusal; primary write failure surviving a secondary lock-cleanup
failure. Tests use temporary directories under the independent D execution
area, never the real authority, source chain, data, weights, network or `/proc`.

The call-site fixture reuses U75's synthetic metadata builder without rerunning
its old suite. Only an ephemeral loaded module's invented encoder/source pins
are substituted to admit fake byte bodies, as in that original fixture. The
production U75 pin and production code never change. The commit protocol test
isolates U75 validation with a stub; it is not genuine proof qualification.

Observed test-first history: absent caller `4186b1` exit1; initial fixture setup
errors `7e7176` exit1; temporary files initially used the full default drive,
`d1f5a9` exit1 with disk-full (no scientific task). Explicit D temporary files
gave `6855e6` exit0 for three tests. New primary-exception test `341638` exit1
exposed cleanup masking the original error; fixed implementation `cceaf2`
exit0, four tests in 0.448s. The nonfinal-endpoint rejection used an existing
guard and passed its first added run; no invented RED is claimed for that case.

## Remaining actual work

The original U74 SSH launch, Root's genuine U77 collector/envelope invocation,
remote artifact transfer and this caller's real invocation remain unexecuted.
Original source staging, fresh output/resource checks, original held tool
terminal, independent process absence, all four genuine shards, original U78
saved spec consumed by a real CPU launch, full metrics and final Root AP
acceptance remain necessary. These source checks must not be counted as any
of those runtime or scientific results.
