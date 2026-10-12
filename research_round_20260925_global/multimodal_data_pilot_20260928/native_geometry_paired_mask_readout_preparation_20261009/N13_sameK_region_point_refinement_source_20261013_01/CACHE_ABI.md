# N13 same-K region FIT cache SOURCE ABI

Status: independent new UNBOUND SOURCE; no scientific execution, binding, cache,
qualification or training is claimed by source authoring. ROOT alone authors
SPEC/admission and executes one owned finite leaf after static review.

## Files and fixed schemas

- `cache_fit.py`: frozen original E24 FIT cache producer.
- `qualify_cache_cpu.py`: independent OpenCV/GT/sampling qualifier.
- `region_common.py`: ordinary metadata, packet and explicit-region PR helpers.
- `CACHE_ABI.md`: this contract. Existing SOURCE is unchanged.

`region_common` literal names are `GPU_SCHEMA`, `CPU_SCHEMA`, `PACKET_SCHEMA`,
`CARRIER_SCHEMA`, `REPORT_SCHEMA`, `REPORT_STATUS`, `Q_SCHEMA`, `Q_STATUS`,
`ARMS`, `REF_KEYS`, `RECIPE`, `REGIONS_RECIPE`, `NUMERIC`, `RAW_KEYS`, `OWNER_SHA`.

```text
GPU_SCHEMA  ROOT-N13-sameK-region-FIT-cache-GPU-spec-v1
CPU_SCHEMA  ROOT-N13-sameK-region-FIT-cache-CPU-spec-v1
PACKET_SCHEMA N13-sameK-region-FIT-cache-packet-v1
CARRIER_SCHEMA N13-sameK-region-FIT-validation-carrier-v1
REPORT_SCHEMA N13-sameK-region-FIT-cache-report-v1
REPORT_STATUS N13_SAMEK_REGION_FIT_CACHE_GPU_COMPLETE_PENDING_CPU
Q_SCHEMA ROOT-N13-sameK-region-FIT-cache-CPU-qualification-v1
Q_STATUS ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED
ARMS LOG_PR, GEO_PR
```

## Original terminal gates and SPEC

Producer: exactly37 original successful normal RAW rows, including the actual
qualified N12 GPU02/CPU02 at indices35/36. CPU qualifier: the same37 prefix plus
new actual cacheGPU terminal at index37, exactly38 rows. Four actual failures
remain: `prior_failed_CPU_receipt` and three explicit
`additional_failed_RAW_inputs` rows `{receipt,phase,returncode:1}`. These remain
failures, never successful qualifications.

`terminal_identity_pins` is keyed by the exact absolute original receipt path,
with exactly `{phase,returncode,original_identity,admission_sha256}` per row.
All original identity/admission/held exit/reaped/birth-absent/resource gates run
before the first SHA, bound SOURCE import, scientific input read or numerical
import. SOURCE also checks original admission/PB/bootstrap and exact entry args.
The current leaf never certifies its own terminal in report or Q.

Every ref is `{path,bytes,sha256}`, absolute, unique and in `sources`. Both SPECs
have all `REF_KEYS`:

```text
entry_source, region_common_source,
context_GPU_spec, context_CPU_qualification, FIT,
original_cache_entry_source, original_cache_spec, original_cache_report,
original_cache_CPU_qualification, original_cache_CPU_source, original_cache_CPU_spec,
original_cache_GPU_admission, original_cache_GPU_receipt,
original_cache_CPU_admission, original_cache_CPU_receipt,
prior_N12_GPU_report, prior_N12_CPU_qualification, prior_N12_CPU_source,
prior_N12_CPU_spec, prior_N12_CPU_admission, prior_N12_CPU_receipt
```

Additional common fields: `phase`, `RAW_inputs`, failure/pin fields, `sources`,
`recipe=RECIPE`, `regions_recipe=REGIONS_RECIPE`, `numeric_protocol=NUMERIC`,
`image_root`, `matching_loss_seed_base` (exact old cache value),
`region_rank_seed_base`, `uniform_seed_base` (two distinct private streams),
`allocator_limit_MiB=4096`, `whole_GPU_limit_MiB=8192`, `deadline_seconds`.
Seed bases are integers in `[0,2**63-741)`.

GPU additionally has `GPU_admission_file`, fresh absolute `output_directory`.
CPU has `CPU_admission_file`, `cache_entry_source`, `cache_spec`, `cache_report`,
`cache_GPU_admission`, `cache_GPU_RAW`, fresh absolute `Q_path`.
Deadlines are positive ROOT-prelocked integers equal to the own admission;
N12's diagnostic1800 seconds is not silently reused for741 model forwards.

CPU01/threads2 and no GUI apply. Producer CUDA device0; CPU CUDA visibility
empty/uninitialized. The owner4096 allocator/8192 whole-GPU resource scope is
unchanged, and whole-leaf VRAM is not self-certified. Closure includes actual
loaded library PY/PYD/DLL paths, not just distribution RECORD or versions;
preserve the N12 SciPy actual-tree repair and bind other actual native-package
trees. Neither SOURCE imports numerical libraries at module import.

## Frozen FIT population and native regions

All741 FIT images and1661 original GT instances are retained. Original native
ten-stage matching and raw-GT annotation ordering are checked against the
qualified N07 FIT packets. The frozen loader/model/loss-observation recipe and
matching seed are unchanged; one `no_grad` frozen E24 model forward per image
creates F/e/M/C and matching supervision. No optimizer/backward or weight update
is performed. GT matching is training supervision; GT does not select regions.

Render the actual original M through200→800→actual crop→original dimensions
in natural all100 query order before selecting matched queries. Saved
`matched_native_L` is exact original GPU FP32 and determines all regions/points.
P is `L>0`; B is `abs(L)<=2`; K is B's exact pixel count. Geometry ranks all
in-domain pixels lexicographically by `(D,abs(L),flat)`, where D is the distance
to the nearest opposing P class in pixel-center Chebyshev distance. No outside
boundary and no distance truncation. Empty/full P has D=infinity and falls back
to uncertainty/flat ordering. Geometry has exact K; output ordering is flat
ascending. GPU leaf uses SciPy chessboard CDT; CPU independently uses OpenCV
DIST_C mask3. The original B equals uncertainty topK as an exact consistency
identity, not an additional method arm.

For each matched instance with K>0, share768 private rank draws between both
sorted regional supports and share256 full-domain uniform draws from a separate
generator. For K=0, share uniform1024 and do not advance the rank generator.
Sampling uses replacement; duplicates remain. Cache points are fixed across
all training epochs and all three paired head seeds. Thus this first study
changes region selection and supervision sampling together; it cannot isolate
inference-region causality.

## Core packet and separate validation carrier

`PACKET_%04d.pth` contains source/spec/phase/teacher refs, canonical identity,
original cached packet ref, query IDs `[G]` int64, query embeddings `[G,256]`
FP32, native C `[1,100,2]` FP32, `matched_native_L [G,H,W]` FP32,
`native_P_rles[G]`, `region_rles[LOG_PR/GEO_PR][G]`, `K[G]`, fallback flags,
matching/seed/GT-area/digest metadata, and `validation_carrier` ref.

`arms[LOG_PR/GEO_PR]` each contains:

```text
F_samples       float32 [G,1024,256]
native_logits   float32 [G,1024]
targets         bool    [G,1024]
point_indices   int64   [G,1024], flat original coordinates
point_in_region bool    [G,1024]
```

No physical complex SAR or new image modality is introduced. PR never uses a
JPEG49 patch; its unchanged factory receives zero patch49 only for its legacy
forward ABI. e is original native e, validated by the factory but unused by PR.

`VALIDATION_CARRIERS/CARRIER_%04d.pth` separately stores source/spec/phase/teacher
identity and query IDs, `native_F256 [1,256,200,200]`,
`matched_M200 [1,G,200,200]`. Producer emits references/digests for these carriers.
They are bound CPU qualification inputs and provenance refs for training, but
training epochs load only the core point packet, never these40MB-per-image F
carriers. Original-image256-channel F is never saved.

Report has741 `rows`, `packet_refs` and `validation_carrier_refs`, exact
`images=741`, `raw_GT=1661`, `point_count_per_arm=1700864` and source/teacher/
prior-qualification/recipe/import/resource/cleanup metadata. Each row includes
`tensor_digests={native:{query_ids,query_embeddings,native_class_logits,
matched_native_L},arms:{LOG_PR/GEO_PR:{F_samples,native_logits,targets,
point_indices,point_in_region}}}` and independent `carrier_digests` for
query IDs/F256/M200. Every actual output reference joins its canonical path.

## CPU qualification scope

CPU reconstructs P/B/geometry from saved GPU L independently, checks complete
RLEs/K/fallback, exact shared-rank and uniform RNG replay, duplicate counts,
raw GT masks/areas/digests/identity, original matching records/query map/e/C,
sampled labels/L/region flags and exact same-pixel F aliases across arms/GT.

Separate carriers permit original two-stage CPU F/L projection checks at the
new and old qualified sample points, with practical rtol/atol1e-4. CPU L/P/B/K
are never substituted for saved GPU metrics or sampling. These tolerances are
not certified GPU error bounds or bitwise interpolation claims. CPU does not
re-run teacher/matching or independently verify all610 frozen state entries.

Q uses `source=cache_entry_source`, `reader_source=CPU entry`, `spec=CPU SPEC`,
`CPU_admission`, and `GPU_spec=cache_spec`, `GPU_report=cache_report`,
`GPU_RAW=cache_GPU_RAW`, `GPU_admission=cache_GPU_admission`. It retains the38
RAW inputs, all failures/pins, exact recipe protocols, FIT/teacher/prior N12Q/
oldcacheQ refs, both741 output ref lists, and per-image independent checks.

Required True proof fields are
`full_native_L_geometry_and_sampling_CPU_replayed`, `hard_targets_CPU_checked`,
`F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked`,
`sampled_F_original_two_stage_CPU_replayed`, `exact_GPU_L_P_RLE_K_used`.
`CPU_approximate_L_substituted_for_selection`, `native_matching_replayed`,
`native610_unchanged_independent_CPU_verified`,
`current_owned_CPU_terminal_self_certified`, `actual_efficacy_established`,
`innovation_established`, `certified_GPU_error_bound` remain False.

## Shared training API

`require/read/reference/ref_shape/bound/load/write/owner_join/joined` are stdlib
metadata helpers used only after a caller's complete actual RAW prefix.
`source_gate` is the strict37/38 cache gate, not a training gate. Training owns
its dynamic original terminal prefix and never imports this SOURCE before it.

`validate_cache_input(spec,closure,owners)` returns `(gpuSpec,report,q)` and
requires the exact primary refs `cache_entry_source`, `cache_spec`,
`cache_report`, `cache_GPU_admission`, `cache_GPU_RAW`,
`cache_CPU_qualification`, `cache_CPU_source`, `cache_CPU_spec`,
`cache_CPU_admission`, `cache_CPU_RAW`, `FIT`; cacheGPU/CPU join indices37/38.

`load_packet(torch,report,index)` returns a core packet;
`packet_validate(torch,packet,row,report)` returns G;
`instance_inputs(torch,packet,g,arm,device)` returns `(F,e,L,y,regionflag)`.
No carrier is opened by these epoch input helpers.

`region_logits(torch,head,F,e,L,regionflag)` calls the unchanged PR factory
`head(F,e,L,zero_patch49,arm='PR')`, consumes its ungated `residual`, ignores its
old-band logits, and returns explicit-region `logits=where(flag,L+u,L)` plus
residual/band/editable. Both arms use the same wrapper. It changes no parameter
names, count, architecture, inputs or residual primitive.

Only ROOT's external actual current CPU exit0/reap/birth-absence gate admits this
Q to training. Cache qualification, GitHub publication and N12 diagnostic
support do not establish trained efficacy, innovation, SOTA or TGRS readiness.
