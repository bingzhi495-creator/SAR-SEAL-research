# Reproduce the original FIT109 source table

Use the qualified isolated Python 3.10.13 runtime with NumPy 1.26.4, torch 2.10.0+cu130 on CPU, pycocotools 2.0.11 and the pinned Shapely/GEOS runtime supplied as `$shapelySite`. Supply the [original capture](../RFDETR_MATCHED_POINT_FORMAL_20261120/run_fit109.py) and its independently accepted receipt, the original closed 109-image feature basis with its witnessed exit and readback, the original closed point-head fit and weights, the original source roster, and the native HRSID FIT annotations. Keep the three resource roots separately read-only. The capture, original fit and basis must carry the exact artifact versions specified by the worker. Supply the release code directory as `$code`, this source folder as `$scripts`, and a fresh existing work directory as `$work`.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$observer = Join-Path $code 'LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py'
& $py -I -B -X utf8 "$scripts/test_prepare_source.py"
if ($LASTEXITCODE -ne 0) { throw 'Canonical source fixture failed.' }
& $py -I -B -X utf8 $observer --log "$work/observed_source" --phase prepare-edits `
  --worker "$scripts/prepare_source.py" --result "$work/actual/RESULT_SAFE.json" -- `
  --code $code --capture $capture --acceptance $acceptance --basis $basis --heads $heads `
  --source-cohort $sourceCohort --fit-gt $fitGT --shapely-site $shapelySite --output "$work/actual"
if ($LASTEXITCODE -ne 0) { throw 'Observed canonical FIT109 preparation failed.' }
```

The private `POINTS_PRIVATE.npz` stores candidate-level arrays in the original roster and joint candidate rank order. `candidate_index` is a zero-based index into flattened stock rows. `offset` slices paired `coordinates`, `features` and `truth` arrays. Coordinates are native row-major linear sample positions, first addition and then removal. Features are float32 with shape paired rows × 2 × 14. Truth is float64 fractional occupancy with shape paired rows × 2. `role` uses unmatched 0, neutral 1 and valid 2. Targets and continuous-area fields are NaN for unsupervised candidates. Fold and group values remain aligned with the original stock roster. The original F budgets `archived_f_k` are candidate-by-seed arrays in `archived_seeds` order. The full native stock stream `STOCK_PRIVATE.jsonl` is keyed by image and retains every candidate with its original rank, query ID, confidence, segmentation and area. Private rows, labels, model weights, masks, and feature arrays stay with the local archive. 

