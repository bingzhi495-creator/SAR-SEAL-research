# Frozen D-FINE score on the PSeg-SSDD official split

The scorer evaluates the original 232 images and 546 float-polygon annotations, then repeats scoring after removing all ten test JPEGs previously found to be byte-identical to FIT741 members. Image names, original GT IDs and native sizes determine the join. All GT instances and native predictions of retained images stay in the sensitivity view. Its annotation count is measured during scoring.

The frozen exclusions are 000001, 000011, 000229, 000389, 000501, 000631, 000639, 000641, 000739 and 001121. These names were recovered from the same split/header inventory as the closed exact-duplicate audit. Its ten train/test header collisions were previously confirmed by SHA-256 to be ten exact groups involving ten TEST images and twelve FIT741 copies. The SHA-256 of the sorted six-digit names with `.jpg` and terminating LF on each line is `22d034a3acf04dcc01eeb19469b0552e292206a4b61c42a3ff275a19be30ab67`.

The accepted image-only replay binds the original HRSID D-FINE E24 detector, the all-FIT109 source point head, the native candidate audit and the sparse edit ledger. The scorer keeps the first min(100, post-NMS count) original ranks and scores, including empty masks. It assigns the decoder's single ship class 1 to the dataset's original category 0 for evaluation. The original `test.json` byte digest is fixed; it contains the source float polygons. Each of raw, fixed, free_matched_count, free_natural and native_matched_count receives one COCO segmentation evaluation per cohort. Summaries include the 12 standard metrics and AP90. Pairwise effects are in percentage points; an undefined comparator produces a null delta. These frozen historical-target diagnostics quantify outcomes on the selected populations. Continuous geometric agreement requires a separate matched-candidate analysis.

Use Python 3.10.13, NumPy 1.26.4 and pycocotools 2.0.11 with `CUDA_VISIBLE_DEVICES=-1`.

```text
python -I -B -X utf8 score_test232.py --accepted REPLAY_INDEPENDENT_RECEIPT --accepted-sha256 OWNER_RECEIPT_SHA --manifest QUALIFIED_IMAGE_MANIFEST --predictions DEV_MASK_PREDICTIONS_PRIVATE.jsonl --candidate-audit DEV_CANDIDATE_AUDIT_PRIVATE.jsonl --edits EDITS_PRIVATE.jsonl --gt ORIGINAL_TEST_JSON --output FRESH_SCORE_JSON
```

The two capture filenames contain the upstream format's `DEV` prefix. The acceptance and the image occurrence inventory specifically bind this invocation to the official 232 TEST images. `check_acceptance` lists all receipt fields. The prescribed manifest supplies `expected_ids` and ordered `items` with `image_id`, `original_hw` and a basename, byte length and digest under `image_ref`. Each edits line carries the native image ID and a rank-keyed `changes` dictionary with per-arm `swaps` and `rle`.

Run `python -I -B -X utf8 -m unittest discover -s . -v` here for generated checks. Three tests passed in the designated runtime, covering the category-zero and filename mapping, a high-confidence false positive on an excluded image, a mask-only AP90 loss, rank truncation, unchanged per-rank budgets, receipt failures, and exclusion identity.
