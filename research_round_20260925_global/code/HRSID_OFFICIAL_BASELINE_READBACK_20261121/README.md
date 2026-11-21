# Independent readback of the frozen HRSID TEST baseline export

The verifier runs after the owner observes natural exit of the original export process, session 88756. It checks the owner receipt first, then verifies the original admission, worker source, result, ordered image-only manifest and prediction stream. The verifier takes no annotation file, original image archive or detector weights.

For each official image it checks the native 164-query, top-100 slot accounting, class-zero rank gaps, descending scores, original image identity, category 1 and 800×800 mask dimensions. For each distinct compressed mask it validates run coverage, decodes pixels, counts foreground area and checks canonical re-encoding. Empty masks and empty prediction lists contribute to separate counts. The pair of official images with identical RGB checksums must have identical ranks, scores and rasters. The final receipt retains the complete prediction SHA-256, the ordered per-block digest and a digest over ranks, scores and packed native masks. These checks establish file integrity and internal consistency of the closed export. The separate scoring stage consumes the accepted receipt.

## Invocation after terminal evidence

Use the isolated export interpreter (Python 3.10.13), with CUDA visibility set to `-1`. Supply the paths to the actual original source and admission named by the owner. The terminal receipt has status `HRSID_OFFICIAL_TEST_NATIVE_EXPORT_ORIGINAL_TERMINAL`, `tool_session_handle=88756`, `original_process_terminal=true`, `original_exit=0`, the original stdout footer's `result_sha256`, the launched `export_source_sha256`, and its `admission_sha256`.

```text
python -I -B -X utf8 verify_export.py --terminal ORIGINAL_TERMINAL.json --terminal-sha256 OWNER_RECEIPT_SHA256 --admission ROOT_EXPORT_ADMISSION.json --export-source export_native.py --result RESULT_SAFE.json --manifest TEST_IMAGE_ONLY_MANIFEST_PRIVATE.json --predictions STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl --output INDEPENDENT_EXPORT_READBACK_SAFE.json
```

Run the CPU synthetic fixtures in this directory with `python -I -B -X utf8 -m unittest discover -s . -p test_verify_export.py -v`. The fixtures include sparse rank gaps, tied scores, empty predictions, duplicate image contents, asymmetric foreground, changed duplicate masks, malformed runs, changed official order, wrong categories and a missing terminal condition.
