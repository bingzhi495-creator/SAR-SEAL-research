# U70 full-population CPU metrics I/O worker

**UNBOUND / NOT SCIENTIFICALLY EXECUTED.** This task used stdlib and synthetic
JSON/source/import-boundary stubs only. It imported no real NumPy, pycocotools
or Torch, loaded no actual data/weights/qualification, ran no AP, subprocess or
SSH, and modified no frozen U67/U68/U69/current training source. Synthetic
return values and their AP flags are not experimental results.

`run_worker(c, request, *, report={})` denotes a fresh caller-owned empty report
(the actual signature requires that explicit argument). This is a real future
I/O body: it performs canonical `importlib.import_module` calls, source-backed
project-module loading, input JSON reads, one frozen U67 evaluation and output
writes. It has no injectable runtime callback, CLI, owner, launch or Root gate.

The exact seven request keys are:

| Key | Required value |
| --- | --- |
| `worker_source` | This actual admitted worker's POSIX path/bytes/SHA ref |
| `core_source` | Original U67,12084 bytes / SHA `ba69cf25fea20e1014124204a2c3e1bef41d0a080f4f81397e43696644cc4fe6` |
| `module_sources` | Exactly `population_contract / eval_body / official / cocoeval / mask_api / geometry_diagnostic / population_metrics` |
| `library_sources` | Exactly `numpy / psutil / pycocotools / coco_loader / mask_extension` |
| `input_refs` | Exactly `context / legacy_context / legacy_result / native_predictions / R_predictions / DEV` |
| `shard_refs` | Four actual complete U66 JSON refs in ranges0–47/47–94/94–141/141–187 order |
| `output_directory` | Fresh canonical POSIX directory with an existing parent, containing no input |

`validate_request(request)` and `direct_references(request)` are pure metadata
APIs. The latter returns the canonical path-sorted deduplicated exact actual
ref union for a future owner. Conflicting refs at one path fail. All refs must
be positive-length except the explicitly named pycocotools package initializer,
whose zero length and SHA are still verified. No historical nested reference
is recursively added. Root must bind this exact union and independently admit
the four original source/exit chains before invoking the worker; pending JSON
does not confer that admission.

Five whole project definitions and the original mask source have fixed content
pins in `SOURCE_PINS`: N28 population/eval/geometry, U40 population_metrics,
original official49923/SHA235792..., and mask4693/SHA1281.... `official` is the
Root-traced original project's `coco_metrics` helper, not the COCO library or
the61452-byte encoder. U40's outer evaluate entry and old model wrappers are
never called. Only U67 calls original merge/official_metrics/diagnose and
relabel/delta mathematics. The genuine library versions of cocoeval and the
loader have explicit future Root-supplied file refs; no library hash or complete
binary closure is invented. Future prediction and evaluation must use the same
actual original mask source content and correctly admitted library environment.

Actual input sequence:

1. Close selected code/library bytes before scientific imports. Read only the
   explicit context/legacy/result, native/R direct18700 lists and four complete
   U66 shard JSONs. Verify the exact `context.legacy_context` bridge by bytes+SHA.
   Both contexts' native/R/legacy_ROOT_result refs must each match the actual
   supplied file content; both original187 image-ID lists must be exactly equal.
   Match legacy.DEV and legacy.COCO_source to actual annotation and official
   refs. Record every original-to-actual relocation, retaining historical Windows
   path strings unchanged; never open them or follow checkpoints/other refs.
2. Check four fixed ranges, whole common FORMAL E50/37050 endpoint and source
   metadata. Link population/mask source content to every shard. Preserve all
   original JSON fields/ordering; U67 checks the complete candidate/proof/cost
   contract. Do not substitute U69's smaller `prediction_metadata` projection.
3. Import canonical genuine `psutil`, `numpy`, `pycocotools`, `_mask`, `mask`,
   `coco`, `cocoeval`; verify actual module objects/file identities and source
   bytes. Require Linux/aarch64 Python3.12.14, NumPy1.26.4, psutil6.1.1,
   CPU affinity0/1, explicit GPU-hidden CUDA_VISIBLE_DEVICES and thread env2.
   No Torch import is allowed before, during or after this path. A fresh child
   must not already contain NumPy or pycocotools.
4. Execute the five closed stdlib project definitions and U67 under temporary
   aliases, without calling their main/run paths. Close DEV and construct actual
   `pycocotools.coco.COCO` from that actual file. Verify original ordered187 IDs,
   380 annotations and categoryID1, not merely category count. Call U67 once with
   the genuine np/COCO objects and exactly its seven modules.
5. Obtain both complete original seven-key/12-stat reference dictionaries from
   `legacy_result.control_results.native_official.COCO` and
   `.strong_fixed_E24_Omega.COCO`. No AP-only reconstruction or reinterpretation
   of old results occurs. U67 performs whole-reference exact reproduction,
   original-native fixed matching, original PLAIN Omega, conditional selected/380
   geometry and the original undefined-stat delta policy. No new math/model,
   score/threshold modification, per-shard AP average or FPS is introduced.

Exclusive compact ASCII+LF JSON writes flush/fsync and return byte/SHA refs:

- `SUMMARY.json`: original complete U67 summary, with no private pair list.
- `FIXED_NATIVE_PAIRS_PRIVATE.json`: original private pair records. Keep private.
- `REPORT.json`: worker stage/source/relocation/output/runtime evidence; includes
  actual input refs and is not automatically suitable for publication.

On success report status is `COMPLETE_PENDING_ROOT_ACCEPTANCE`, AP=True,
DEV=True, TEST=False, qualified=False, zero training credit. Before entering
U67, AP=False/NOT_STARTED describes this worker's stage. While U67 has started
but not returned, AP=None/STARTED_COMPLETION_UNKNOWN records possible partial
COCO work instead of falsely claiming no evaluation. Full return changes that
to AP=True/COMPLETE_PENDING_ROOT_ACCEPTANCE. If cleanup/persistence later fails,
already written summaries remain pending and the worker report is FAILED.

Failures preserve the visible original traceback as text before clearing frame
references; partial stage/relocation/annotation/evaluation evidence and control
observations are retained where output I/O remains possible. The original
scientific failure remains primary over cleanup errors. Every temporary source
alias and Python/NumPy RNG is restored; CPU objects are released/GC'd. Control
checks and a cooperative1800s bound supplement a future external CPU owner.
`c` must genuinely provide `closed`, `check`, `psutil` and `summary`; its selected
source/resource evidence is not an owner OS-exit observation.

The future actual metric path parses annotations and the original COCO/geometry
helpers decode GT masks as required. Current synthetic tests do neither on real
data. Original metadata/annotation-container trace is not a claim that historical
I/O never encountered GT, and it does not requalify legacy results. Root still
owes actual library/overlay ABI, four original accepted prediction exits, CPU
owner/birth/group/resource supervision, original actual terminal acceptance and
later publication review. U69's GPU/CUDA0 owner cannot be relabeled CPU.

Finite tests use the public APIs and real worker JSON writes with a synthetic
control, harmless compiled source definitions and intercepted canonical library
imports. Only fixtures replace source pins/core and file identities; production
has no alternate runtime parameter or unpinned-source mode. Synthetic mask tags,
fake core results and fake COCO objects do not test AP/RLE/numerical equivalence.
Run: `python -B test_metadata.py`. No old suite is rerun.

Actual RED→GREEN: relocation `bd9c5c`→`59b48c`; context bridge/full projection
`a293a0`→`4c3b64`; request/union `f50a28`→`a44fe0`; canonical import boundary
`1eaa11`→`2cd579`; complete I/O `775e7b`→`30db1d`; partial-AP failure scope
`703b47`→`697c78`; zero-byte initializer alias negative `f71fcf`→`262f07`.
The additional bridge-before-import/category-before-metrics negative test passed
first in `8b3c35`; no artificial RED is claimed for already-present guards.
