# Frozen HRSID refinement scoring draft

This CPU adapter uses the original official HRSID native scorer to evaluate a common native baseline N and four fixed refinement outputs for head seeds 5217, 5218 and 5219. All 13 streams use one standard native image block per official image. The adapter holds a single stream's rows in memory during COCO scoring. It reproduces all three published native views, including row fingerprints, counts, and 39 metrics, before comparing treatment arms.

The input receipt comes from an independent label-free TEST readback of the closed capture. It binds exact per-image native parity, original query IDs, unchanged candidate order and confidence, per-candidate matched budgets, actual XOR pixels and reconstructed RLEs. The scorer separately compares native rank gaps, row image/category IDs and exact binary64 confidence values for every candidate on every treatment image. The upstream sparse witness supplies query identities, including equal-confidence candidates. Standard native JSONL contains the rank and score of each mapped candidate.

Use the original CPU environment with Python 3.8.17, NumPy 1.23.5 and pycocotools 2.0.7, with `CUDA_VISIBLE_DEVICES=-1`.

```text
python -I -B -X utf8 score_refinement.py --score-source SCORE_NATIVE_PY --adapter ORIGINAL_COCO_ADAPTER_PY --baseline-score PUBLISHED_NATIVE_COCO_RESULT --readout ACCEPTED_TEST_READOUT --readout-sha256 OWNER_ACCEPTED_SHA --paths PRIVATE_PATHS_JSON --native CLOSED_NATIVE_JSONL --cohorts ORIGINAL_FROZEN_COHORTS --gt ORIGINAL_TEST_GT --output FRESH_SCORE_JSON
```

The private path map has a top-level object with keys `5217`, `5218`, `5219`. Each maps the four keys `SF`, `SU=F`, `SU`, `LW=S` to absolute JSONL paths. The accepted readback has the same structure under `streams`, with each entry containing `sha256` and `bytes`. See `validate_readout` for required upstream assertions. The accepted FIT training and native prediction digests are fixed in the source.

For each of the original official, empty-raster and exact-image-identity cohorts, output reports the native result and all twelve treatments, followed by head-seed-paired contrasts in percentage points. The contrasts compare SU with N, equal-budget independent selection with fixed pairing (SU=F minus SF), SU with the fixed pairing budget (SU minus SU=F), and SU with ordinary writing at the same total edit count (SU minus LW=S). Each contrast includes the three individual values and their arithmetic mean. A COCO metric undefined for either arm yields a null contrast. All three head seeds share one detector, and historical project selection previously used this official test split.

For generated-case checks set `OFFICIAL_SCORE_SOURCE` and `NATIVE_SCORE_ADAPTER` to the corresponding published files and run `python -I -B -X utf8 -m unittest discover -s . -v`. Three tests passed, covering tied scores, simple mask-only edits, sparse native rank gaps, altered order or scores, matched contrasts, missing arms and failed independent qualification. Actual capture, scoring, process exit, and independent score verification are separate research stages.
