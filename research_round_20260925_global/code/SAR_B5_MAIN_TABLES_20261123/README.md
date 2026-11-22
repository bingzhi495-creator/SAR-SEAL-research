# Reproduce the original 740-image and 1,962-image frozen tables

Use the isolated Windows environment in the [previous clean-run setup](../INSTANCE_RSDD_B5_REPRO_20261123/README.md). These commands start from official image data and the original locally supplied frozen artifacts. The published comparison records are read by the final verifier. They are separate from the inputs to capture, editing and scoring. [Predeclared comparison](../../SAR_B5_MAIN_TABLES_REPRODUCTION_PRELOCK_20261123.md)

## Inputs

- The center5202 RF-DETR E24 checkpoint has SHA-256 `011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11`. Supply the three original `5217_all_fit.npz`, `5218_all_fit.npz` and `5219_all_fit.npz` bundles in one local directory. Each CLI checks every model input against its immutable digest.
- Supply the original HRSID PNG ZIP, including `annotations/test2017.json`. The file digest is `f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159`. The input step builds the published main and two sensitivity populations. [HRSID archive and input study](../../HRSID_OFFICIAL_PNG_TEST_INPUT_ACTUAL_20261121.md)
- Supply the original Instance-RSDD author `annotation_coco/test.json`, original RSDD-SAR image ZIP and the original paper PDF with Appendix Table 1. For a local extracted JPEG directory, supply the public ZIP central directory with `--image-dir` and `--zip-directory`. The converter verifies each original JPEG's CRC and decoded pixels. [Author data locations](../INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/README.md)

The assets are used locally. The private image metadata, raw annotations, predictions and trained weights remain outside the release. Run the commands below from this directory using the qualified Python. Supply fresh output locations.

## HRSID official 1,962 images

The original native operator is CPU FP32, with one Torch thread in each process. Set the number of independent workers to fit local memory. The score preserves the primary official annotations, the nonempty sensitivity and the exact identity deduplication sensitivity.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python portable_hrsid1962.py prepare --archive data/HRSID_PNG.zip --output runs/hrsid-intake
python portable_hrsid1962.py capture --archive data/HRSID_PNG.zip --intake runs/hrsid-intake --checkpoint weights/checkpoint_23.ckpt --heads weights/point-heads --workers 8 --output runs/hrsid-capture
python portable_hrsid1962.py score --archive data/HRSID_PNG.zip --intake runs/hrsid-intake --capture runs/hrsid-capture --output runs/hrsid-score
python verify_main_tables.py hrsid --published-results ../../results --readout runs/hrsid-capture/RESULT_SAFE.json --score runs/hrsid-score/RESULT_SAFE.json --output runs/hrsid-verified.json
```

## Instance-RSDD author test, 740 images

A single original GPU detector capture supplies the frozen predictions for both head seeds. Both CPU replays recompute the original seven editing policies and check the matched budgets before scoring against the original author file.

```powershell
python prepare_rsdd740.py --author-test data/annotation_coco/test.json --paper-pdf data/rsdd-paper.pdf --archive data/RSDD-SAR.zip --output runs/rsdd-intake
$env:CUDA_VISIBLE_DEVICES = '0'
python portable_rsdd740.py capture --intake runs/rsdd-intake --checkpoint weights/checkpoint_23.ckpt --output runs/rsdd-capture
$env:CUDA_VISIBLE_DEVICES = '-1'
foreach ($seed in @(5217,5218)) {
    python portable_rsdd740.py replay --intake runs/rsdd-intake --capture runs/rsdd-capture --seed $seed --point-head "weights/point-heads/${seed}_all_fit.npz" --workers 3 --output "runs/replay$seed"
    python portable_rsdd740.py score --intake runs/rsdd-intake --capture runs/rsdd-capture --replay "runs/replay$seed" --author-test data/annotation_coco/test.json --seed $seed --output "runs/score$seed"
}
python verify_main_tables.py rsdd --published-results ../../results --intake runs/rsdd-intake/RESULT_SAFE.json --capture runs/rsdd-capture/RESULT_SAFE.json --edits-5217 runs/replay5217/RESULT_SAFE.json --edits-5218 runs/replay5218/RESULT_SAFE.json --score-5217 runs/score5217/RESULT_SAFE.json --score-5218 runs/score5218/RESULT_SAFE.json --output runs/rsdd-verified.json
```

The checker compares the original and regenerated full-precision AP, AR and available continuous geometry with a tolerance of 0.000001 percentage points. Discrete counts and canonical stream digests must match exactly. HRSID DEV705, PointRend and the 74-image parent reserve are separate evaluation scopes.
