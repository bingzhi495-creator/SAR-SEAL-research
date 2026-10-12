# N11 native editable-region coverage diagnostic SOURCE

Status: **UNBOUND, UNEXECUTED SOURCE; awaiting independent static review and ROOT admission.**

This is one finite diagnostic for the unresolved coverage of native mask errors
by the original `abs(L) <= 2` editable band. It does not train, choose a new
query, load a teacher/head/optimizer, decode an image, or use TEST. The original
N07 frozen M200/class packets are reused. F256/e0 remain inside their bound
packet but are not projected or numerically consumed by this diagnostic.

Files are `run.py` (GPU exact native replay and band export), `qualify.py`
(independent CPU pixel and matching qualification), and `common.py` (shared
provenance and identity helpers). Shared provenance does not supply either
entry's pixel decomposition or oracle formulas.

## Execution and provenance ABI

Both entry points accept only `--spec ABSOLUTE_JSON --owner-execute` under the
existing original held-handle ROOT driver. GPU schema is
`ROOT-N11-native-editable-coverage-GPU-spec-v1`; CPU schema is
`ROOT-N11-native-editable-coverage-CPU-spec-v1`.

GPU requires **33 normal actual0 RAW ancestors**, ending in the actually closed
N10 fullDEV CPU qualification. CPU requires **34**, appending this new N11 GPU
attempt. Both preserve the three original actual1 failures: one
`prior_failed_CPU_receipt` and two `additional_failed_RAW_inputs`, each latter
row exactly `{receipt, phase, returncode: 1}`. They cannot qualify a normal row.

Every normal row retains `receipt`, `phase`, `admission`, `phase_binding`, and
`bootstrap`. References always have exactly `path`, `bytes`, and `sha256`.
`sources` is the unique full bound reference closure, including numerical
library Python sources and binaries actually imported at runtime. The current
SPEC is separately bound by its admission because its inclusion in its own
closure would be circular.

`terminal_identity_pins` is a dictionary keyed by each original RAW's **exact
absolute `receipt.path` string**, with one value per normal or failed locator:

```json
{
  "phase": "original phase name",
  "returncode": 0,
  "original_identity": {"all": "exact original RAW identity fields"},
  "admission_sha256": "exact original RAW admission SHA"
}
```

Failure pins have returncode 1. The entire original identity object is copied,
without truncation or reconstruction. Each entry reads these original RAWs and
requires actual held exit, ROOT reap, original-birth absence, exact identity
and admission SHA, and original resource scope **before the first payload
hash or non-stdlib import**. Only then are the closure and `common.py` loaded.
The current report/Q cannot certify its own terminal.

Required reference fields in both SPECs:

```text
entry_source, common_source,
prior_N07_report, prior_N07_CPU_qualification,
qualified_native_Q, qualified_native_Q_source, qualified_native_Q_spec,
qualified_native_Q_admission, qualified_native_Q_receipt,
pair_anchors, anchor_derivation_record,
DEV, native_predictions, reader_source, renderer_source, export_source
```

`qualified_native_Q` is the already ROOT-qualified N10 fullDEV CPU Q. Its
actual owned receipt is normal ancestor index 32; its own prior prefix has
length 32. `reader_source` is the bound old N07 CPU reader providing canonical
population, digest, official COCO, score-factor, and matching helpers.
`renderer_source` and `export_source` remain the actual original N07-bound
renderer/exporter. This source never invokes an ancestor source gate or
replays an ancestor training chain.

GPU additionally requires `GPU_admission_file` (absolute path) and a fresh
absolute `output_directory`. CPU additionally requires `CPU_admission_file`,
fresh absolute `Q_path` in an existing directory, and references:

```text
GPU_entry_source, GPU_spec, GPU_report, GPU_RAW, GPU_admission
```

CPU joins that GPU SOURCE/SPEC/admission/report to normal ancestor index 33,
not merely to its report status. Its full closure includes the GPU START,
native replay export, and coverage outputs.

Both lock `recipe` to `common.RECIPE` and `numeric_protocol` to
`common.NUMERIC`; ROOT should copy those literal structures into SPEC rather
than silently filling defaults. `deadline_seconds=1800`,
`allocator_limit_MiB=4096`, `whole_GPU_limit_MiB=8192`. GPU uses visible device
`0`; CPU uses the empty CUDA visibility string. Both require actual CPU
affinity `[0,1]` and all four thread environment variables set to `2`, with
Torch intra/inter-op threads 2. Torch 2.0.1+cu118 and NumPy 1.23.5 are pinned.
No automatic retry, GUI, or speculative next phase is started by these files.

## Compact fixed-anchor input: exact row format

ROOT extracts this input only after all33 actual gates and independent
qualification of the original N10 matching DETAILS. It copies the native
`controls_actual_COCO_matching.native_official` assignment at **indices
0, 5, 8**, retaining actual floating threshold values. Canonical labels
`0.50`, `0.75`, `0.90` are names, not approximate-threshold selectors. In
particular index 8 may have actual value `0.8999999999999999`.

`pair_anchors` schema: `ROOT-N11-qualified-native-fixed-pair-anchors-v1`.
Top-level required values:

```text
schema, qualification (reference to qualified_native_Q),
matching_details (the exact Q.matching_details reference),
DEV, native_predictions, images
```

There are exactly 187 canonical ordered image objects. Each object has:

```json
{
  "image_id": 1,
  "original_GT_ids": [101],
  "threshold_slots": [
    {
      "index": 0,
      "label": "0.50",
      "threshold": 0.5,
      "GT": [
        {
          "gt_id": 101,
          "ignored": false,
          "FN": false,
          "matched_DT": 1,
          "TP_pair_key": "1:101",
          "pair": {
            "gt_id": 101,
            "dt_id": 1,
            "query_id": 3,
            "IoU": 0.8,
            "Dice": 0.8888888888888888,
            "pixel_TP": 80,
            "pixel_FP": 10,
            "pixel_FN": 10,
            "empty_union_IoU_convention": 0
          }
        }
      ]
    }
  ]
}
```

Example values are illustrative, never actual data. Each image must contain
exactly three slots, ordered 0/5/8. `original_GT_ids` is copied from the
qualified original `actual_COCO_arrays.gtIds`; each slot's GT records retain
that exact order. The five original GT row fields are copied verbatim; `pair`
is the unchanged original `TP_pairs[TP_pair_key]`, or null for FN/ignored GT.
FN retains `matched_DT: 0`, `TP_pair_key: null`, `pair: null`; ignored GT also
retains its actual original matched_DT while receiving no pair/oracle.

`anchor_derivation_record` schema is
`ROOT-N11-qualified-native-fixed-pair-anchor-derivation-v1` and status is
`ROOT_N11_NATIVE_ANCHORS_EXACT_FROM_QUALIFIED_DETAILS`. It must include:

```text
qualification, matching_details, anchors (references),
fixed_slot_indices: [0,5,8], images: 187, raw_GT_per_slot: 380,
every_original_GT_identity_ignore_FN_and_pair_exact: true,
new_query_selection: false, no_GT_directed_inference: true
```

ROOT retains its precise extraction/row correspondence and actual terminal
qualification evidence in that record; runtime checks reference consistency.
The CPU reader then independently reruns existing official COCO matching on
the unchanged full native18700 export and compares **every** GT order, pair,
DT/query, ignore/FN field and original matched pixel total with these anchors.
It never needs to load the entire previous four-head dense matching DETAILS.

Locked full area-all/maxDet100 native cohorts:

| Original slot | Matched fixed pairs | Native FN | Raw GT retained |
|---|---:|---:|---:|
| 0 (.50) | 374 | 6 | 380 |
| 5 (.75) | 348 | 32 | 380 |
| 8 (.90) | 107 | 273 | 380 |

The current original ignored-GT count is zero. Every GT row remains present
even if native FN; no query or oracle is inferred for those rows. Different
thresholds can select different queries; no nested-pair assumption is made.

## Exact GPU prerequisite and diagnostic outputs

Natural all100 M200 is rendered by the original bound renderer in FP32 on
CUDA. Original AST-selected exporter expressions perform topk100/reorder,
hard support, full100 quality reduction and S0. Every original full dictionary
must equal the archived native export, including query order, class, score,
compressed RLE and empty queries; **there is no GPU baseline tolerance**.
Natural all100 `abs(L)<=2` band counts must equal the original row counts.

GT is used only after this inference prerequisite, for fixed-anchor pixel
diagnosis. GPU emits exact band RLEs for the union of originally matched
query IDs in the three fixed cohorts; it does not emit a GT-chosen prediction.
Every threshold retains all380 GT records and references the fixed band's
query key only for actual original matched pairs.

Fresh output files:

```text
START.json
NATIVE_EXPORT_REPLAY.json (all18700 unchanged original dictionaries)
COVERAGE.json (exact selected band RLEs and all fixed GT records)
REPORT.json (pending CPU qualification, references and conditional summaries)
```

Report schema/status are `N11-native-editable-coverage-GPU-report-v1` /
`N11_NATIVE_EDITABLE_COVERAGE_GPU_COMPLETE_PENDING_CPU`.

For original binary prediction P, GT G, frozen editable band B, let A=|G|.
The six TP/FP/FN inside/outside counts are obtained on full original rasters.
Unrestricted binary edits inside B attain the optimistic free-band oracle:

```text
P* = (P outside B) union (G inside B)
IoU* = (A - FN_out) / (A + FP_out)
IoU* >= .90 iff 10 FN_out + 9 FP_out <= A
IoU* >= .95 iff 20 FN_out + 19 FP_out <= A
```

SOURCE independently verifies the directly constructed oracle support,
positive/nonempty GT denominator, unchanged outside support, complete native
pixel totals, and native_IoU <= oracle_IoU <= 1. Exact integer comparisons
avoid float threshold rounding. This oracle is attainable for unrestricted
binary band edits; a finite learned head or bounded residual may not attain it.

Summaries report all GT/FN population, matched-pair conditional pixel error
totals, editable error fraction, and ideal-support counts. The canonical .50
summary conditions on the **374 original .50 matched pairs**, with six native
FN explicitly excluded from a selected-query bound. It is not all380-GT
ceiling, AP, simultaneous deployable predictions, efficacy, or novelty.

## Independent CPU qualification and limits

The CPU reader recomputes official COCO12 and complete assignment anchors;
reconstructs the same original M renderer in natural batch100; practically
checks every exported native support and native class/quality factor; checks
every selected exact GPU band raster; and independently computes all six
pixel counts and direct/formula oracle values.

CPU near-zero/native-band allowance is `1e-4 + 1e-4*abs(CPU_L)`. Native sign
discrepancies must lie near zero; band discrepancies must lie near abs(L)=2.
All100 band counts receive a practical count check, while selected bands
receive a full raster comparison. This is **not** GPU bitwise equality, a
certified numerical error bound, or exact GPU band certification. The original
GPU band RLE and exported support remain unchanged metric inputs. No widening
of the band or CPU mask substitution is permitted.

Q schema/status are `ROOT-N11-native-editable-coverage-CPU-qualification-v1` /
`ROOT_N11_NATIVE_EDITABLE_COVERAGE_CPU_QUALIFIED`. The Q retains all380 rows per
slot through the bound COVERAGE reference, the 829 fixed-pair occurrences,
all187 all100 practical field checks, official matching and conditional
summaries. ROOT still must qualify its original actual held/reaped terminal
before using Q numerics or making a following scientific admission.

Raw native masks, exact selected band RLEs, GT identities and execution bundles
are scientific payloads excluded from public synchronization. Public SOURCE,
this ABI, ordinary execution status and ROOT-verified aggregate findings may
be synchronized under the existing publication scope. No experimental result
or innovation is claimed by this unexecuted SOURCE.
