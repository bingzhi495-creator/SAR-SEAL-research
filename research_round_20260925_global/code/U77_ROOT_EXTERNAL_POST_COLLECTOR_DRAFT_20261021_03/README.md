# U77 external collection and later post envelope — revision03, UNBOUND

Revision03 separates generic procfs census parsing from original owned-process
identity validation. Generic entries require PID>0 and integer parent/group/
session/start-tick fields>=0; the original owner/child validator is unchanged
(positive PID/group/session/start ticks; parent PID may be0). Root separately
reported real read-only diagnostics with PID1 start0 and kernel-process
group/session/start0; no real procfs observation was performed by this author.
The Linux [stat field definitions](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)
describe starttime in boot-relative clock ticks; the kernel's
[stat writer](https://github.com/torvalds/linux/blob/v6.14/fs/proc/array.c)
uses namespace group/session values and converts nanoseconds to ticks. Thus a
strict positive lower bound is an owned-process contract, not a generic census
field requirement. Negative/unknown values remain refusals. No boot, absence,
group-scan, output, terminal or qualification guard was relaxed.

Revision01 is preserved unchanged as a frozen draft with a known group-scan
gap: a numeric process already in the directory snapshot could disappear after
its directory check; skipping its unavailable stat could miss a child created
after the snapshot and incorrectly report resolved group absence. Revision02
changes only that scan case to UNKNOWN/refusal. The original owner/child PID
absence checks still accept a genuinely missing PID directory as absence.
Neither revision was run against real `/proc` or actual prediction evidence.

This stdlib-only CLI connects two future external Root steps to frozen U75.
It adds no model, metric, owner, training configuration, SSH client, automatic
retry, acceptance/CAS or deployment. Import performs no I/O or process probe.
No real collector, real `/proc`, scientific library, SSH, actual Source/spec,
qualification chain, data, checkpoint or Ops file was accessed in this task.
Only ordinary definitions and synthetic test fixtures were used. No frozen
source or research status document was modified.

## Explicit production calls

Both operations use the same CLI:

```
python -u collector.py --request-file /absolute/request.json --request-sha256 <sha256>
```

The original request bytes must match the supplied SHA before dispatch. The
operation comes from the request. Production CLI requires Linux. Production
`collect(request, request_ref)` has no proc-root argument; its
`production_proc_root()` returns only `Path('/proc')`. There is no CLI, request
or environment override for a fixture filesystem. Tests replace that external
OS boundary with an explicitly temporary synthetic procfs. Test observations
are never real Linux process evidence. Nothing happens automatically on import.

All refs have exact `path`, integer `bytes` and `sha256` fields. The file reader
requires a real absolute canonical path on the current host, no symlink, stable
file identity/stat and matching complete bytes/SHA. A Windows caller cannot
pretend that `/home/...` names the DGX file. Future Root must run the collection
where its actual POSIX files and original process namespace are visible.
No host/account/key or private deployment/Ops path is hardcoded.

## 1. Collect; never predict its own tool terminal

The exact seven-key request is:

| Key | Meaning |
|---|---|
| `schema` | `U77-external-collect-request-v1` |
| `operation` | `COLLECT` |
| `collector_source_ref` | This exact actual collector source identity |
| `acceptance_source_ref` | Fixed U75 source, 34428 bytes / SHA `fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2` |
| `spec_ref` | Explicit original U74 spec bytes |
| `original_terminal` | Original U75 exact eleven-field external numerical terminal |
| `output_directory` | Fresh independent directory with an existing parent |

`collect` closes the actual spec, checks the original U74 identity and four
frozen code identities through U75 `pinned_codes`, then derives exactly the
five original output locations: owner `OWNER_RECEIPT.json` and `RUN_STATE.json`,
entry `REPORT.json`, worker `REPORT.json` and complete `U66_SHARD.json`. The
prediction file is streamed for its byte identity; it is not parsed or evaluated.
The other raw JSON records and their recorded refs must agree. Actual direct
child wait must be integer0 and reaped; original owner/entry/worker completion
must remain pending Root review. Fixed U75 `validate_original_terminal` checks
the original actual Root tool/SSH/owner OS0 and exec owner/spec/SHA relationship.
Owner `returning` text supplies no owner OS exit evidence.

Both actual owner stdout/stderr logs and both explicit original SSH stream
artifacts are rehashed. Empty raw logs are legitimate and retain the empty SHA;
ordinary Source/JSON and full prediction files must be nonempty. Original SSH
stream artifacts must be byte-exact original captures, not reserialized or
reconstructed equivalent logs. If Root stages them elsewhere, it must retain
the explicit original-capture/transfer relationship privately; U77 never rewrites
an old ref into a guessed remote path.

`observe_processes` independently reads boot ID, original owner/child PID stat
identities and the numeric proc namespace. Boot must agree with the original
receipt before and after scanning. Original parent/session/PGID/startticks must
agree. A live PID, including state `Z`, or a reused original numeric PID refuses
certification. Surviving original group members refuse. Permission, malformed
or missing stat inside a still-present PID directory, incomplete enumeration,
unexpected group identity or changed boot is UNKNOWN/failure, not empty evidence.
Only an absent PID directory maps a missing stat to absence. No process is
signalled, repaired, restarted or retried.

On success, exclusive/fsync `COLLECTION.json` has schema
`U77-external-collection-v1` and status
`COLLECTED_AWAITING_ROOT_TOOL_TERMINAL`. It retains the actual request ref/body,
original terminal/spec/phase/attempt, five output refs, four raw-log refs and
independent process observations. It has no own future tool exit/chunk fields.
`root_accepted`, `qualified` and `U75_acceptance_called` remain false. Its DEV/AP
flags describe no evaluation; this task only hashes existing prediction bytes.

Actual stdout is exactly one JSON line:

```
{"status":"COLLECTED_AWAITING_ROOT_TOOL_TERMINAL","collection_ref":{"path":"...","bytes":1,"sha256":"..."}}
```

The numbers above illustrate the shape, never an actual result. Once a valid
fresh output exists, failures preserve `COLLECTION.json` with status
`UNKNOWN_OR_FAILED`, unknown observations and the visible original exception
traceback, then rethrow. Repeated calls cannot overwrite the directory/artifacts.
Invalid requests do not authorize writing an arbitrary failure directory.

## 2. Root observes the real terminal; then invoke envelope separately

Root must first observe the collector's genuine external tool exit0 and its
actual chunk. A still-running collector cannot know or write these facts.
The original numerical terminal, collector terminal and subsequent envelope
invocation are three different events. The envelope task's own later exit must
never be substituted for either earlier terminal. JSON alone cannot prove the
truth of Root observations; their genuine origin remains Root's responsibility.

The exact seven-key later request is:

| Key | Meaning |
|---|---|
| `schema` | `U77-external-envelope-request-v1` |
| `operation` | `ENVELOPE` |
| `collector_source_ref` | Same exact collector ref used in collection |
| `acceptance_source_ref` | Same fixed U75 ref |
| `collection_ref` | Actual returned `COLLECTION.json` raw ref |
| `collector_terminal` | Eight actual externally observed fields below |
| `output_directory` | Another fresh independent directory |

The exact eight `collector_terminal` keys are:

`root_observed_tool_exit_code`, `root_observed_tool_chunk_id`, `collection_ref`,
`request_ref`, `collector_source_ref`, `spec_ref`, `phase`, `attempt_id`.

Exit must have `type is int` and equal0; `False` is rejected. Chunk must be
nonempty and differ from the original numerical tool chunk. Every ref/phase/
attempt must match the closed collection and original collect request exactly.
Root's controller separately retains the actual collector command/SSH lifecycle;
U77 does not invent a command qualification protocol.

`envelope(request, request_ref)` rehashes the collection, its original request,
spec and original owner receipt. It calls the fixed U75 original-terminal and
`validate_postcheck` guards. Only then does it write exclusive/fsync `POST.json`
with the exact original U75 seventeen keys. The two top-level tool fields now
come solely from Root's actual collector terminal input; the nested
`original_terminal` remains the separate original numerical event. No new fields
or renamed post-tool fields are added to that contract.

It also saves `ENVELOPE.json` with schema `U77-external-envelope-v1`, status
`POST_WRITTEN_PENDING_ROOT_ACCEPTANCE` and actual `post_ref`. Root acceptance,
U75 invocation and the envelope task's own external terminal remain unclaimed.
Stdout is one JSON line with exactly:

```
{"status":"POST_WRITTEN_PENDING_ROOT_ACCEPTANCE","post_ref":{"path":"...","bytes":1,"sha256":"..."},"envelope_ref":{"path":"...","bytes":1,"sha256":"..."}}
```

Root's next real controlled step can consume `post_ref` in frozen U75
`accept_shard(spec_ref, original_terminal, post_ref, closed=...)`. U75 performs
the complete finite source/endpoint/body validation; U77 deliberately does not
replace it with collector success. That real U75 call, persistence/CAS and later
CPU prelaunch invocation are outside this CLI and were not executed here.

Any CLI failure writes a one-line `U77-cli-error-v1` JSON object to stderr with
type/message/original traceback and returns1. Successful commands return0 only
after their explicit artifacts are saved. Root must still truly observe that
terminal; the program never reports its own future Root tool result.

## Revision01 historical synthetic evidence

Only the five new tests run; no old suite or production collector runs. They use
temporary JSON/log files, an opaque invented prediction byte string and an
explicit temporary procfs. Original U75 stdlib definitions are loaded without
running science. The CLI test replaces the operation handler; no real `/proc`
is touched. Fixtures are not original OS facts, source qualification or accepted
predictions.

- Birth/group fixture and live PID refusal: RED `a7267d` -> GREEN `13f1c9`.
- Five actual synthetic output hashes/empty logs and pending-only artifact:
  RED `bcb4d4` -> GREEN `995115`.
- Separate external terminal and exact17 envelope: RED `620395` -> GREEN
  `669dc2`.
- CLI request SHA, dispatch and forbidden proc override: RED `e4ff53` -> GREEN
  `47bb14`.
- Seven permission/reboot/Z/reuse/incomplete-stat/group/byte-change regressions
  passed first in `d9ad6b`; no artificial RED was manufactured.

Final new five-test run `7a7496` actual exit0. AST check `77d20f` actual exit0
confirmed stdlib imports, no import-time I/O/probe, final main guard and no public
collect proc-root parameter. These checks establish synthetic behavior only.

## Revision02 finite correction evidence

One new public `collect` test creates a synthetic owned-group process A,
enumerates it, then at its stat-read boundary creates child B and removes A.
The snapshot lacks B. The unchanged revision01 logic returned a collected
artifact, so the new test failed with `ValueError not raised`: RED `c4d62a`,
actual exit1. Revision02's one-line UNKNOWN gate makes the same test pass:
GREEN `4e9043`, actual exit0. It also checks that the saved artifact is
`UNKNOWN_OR_FAILED`, with no resolved observations and `root_accepted=False`.
Both runs invoked `python -B -m unittest
test_metadata.Collector.test_enumerated_process_disappears_before_stat_is_unknown_not_group_absence`
from this revision02 directory. Static AST/exact-byte diff check `6ea985`
returned actual exit0: the scan gate is the only collector source change,
all other functions are unchanged and the frozen revision01 hash is preserved.
Only this new regression test was run for revision02; the five historical
tests and all older suites were not repeated. The fixture uses temporary
synthetic files and a patched external OS boundary, with no real process,
procfs observation, scientific import or qualification claim.

The CLI, requests, pending collection and later exact17 post envelope ABI above
are unchanged. This finite correction does not prove that a live procfs scan
is an atomic snapshot; unknown or incomplete observations remain refusals.

## Revision03 finite correction evidence

Only one new synthetic public-behavior test was run: generic PID1(start0) and
kernel-style entries(group/session/start0) parse and pass through `collect`'s
non-owned census, while original owner/child zero or boolean identity fields
and generic negative fields/PID0 still refuse. RED `d77b6c` actual exit1 failed
at the original strict parser; GREEN `fdcdb9` actual exit0/one test passed.
Command: `python -B -m unittest
test_metadata.Collector.test_generic_zero_fields_parse_and_census_but_owned_identity_remains_strict`.
All fixtures are temporary and synthetic. The old suites were not rerun;
revision01/02 remain frozen with their documented gaps. Revision02's
enumerated-process disappearance UNKNOWN gate is unchanged. CLI and the
7-key requests, 8-key collector terminal and 17-key post ABI are unchanged.
AST delta check `71f06b` actual exit0 confirms only the new generic validator
and parser dispatch changed; the strict validator and census remain identical.
