# Frozen PointRend on the 74-image Instance-RSDD reserve

Use the [closed 74-image reproduction](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) to regenerate the author source roster, intake, native candidates and prior occupancy edits in the pinned software environment. The comparison needs the original RF-DETR center5202 E24 and PointRend FIT109 head. Reconstruct the latter using the [source fit entry point](../INSTANCE_RSDD_POINTREND_TRANSFER_20261122/README.md) or supply the original pinned NPZ. [Frozen protocol](../../INSTANCE_RSDD_RESERVE_POINTREND_PRELOCK_20261123.md) · [Actual result](../../INSTANCE_RSDD_RESERVE_POINTREND_ACTUAL_20261123.md)

Run the following in this directory with the qualified Python. Change the input variables to the existing local assets and use fresh output directories.

```powershell
$stock = 'runs/closed-74'
$checkpoint = 'weights/checkpoint_23.ckpt'
$pointRend = 'weights/HEAD_PRIVATE.npz'
$authorTrain = 'data/annotation_coco/train.json'
$env:CUDA_VISIBLE_DEVICES = '0'
$worker = 'capture_basis_reserve.py'
$log = @(python -I -B -X utf8 $worker --code .. --intake "$stock/conversion" --roster "$stock/reserve.csv" --capture "$stock/capture" --checkpoint $checkpoint --output runs/basis)
$actual = $LASTEXITCODE
if ($actual -ne 0) { throw "Basis capture failed with exit $actual" }
$result = (Get-FileHash 'runs/basis/RESULT_SAFE.json' -Algorithm SHA256).Hash.ToLowerInvariant()
$footer = $log[-1] | ConvertFrom-Json
if ($footer.result_sha256 -ne $result) { throw 'Basis result receipt mismatch' }
@{actual_exit=$actual; output_footer_verified=$true; result_sha256=$result; worker_sha256=(Get-FileHash $worker -Algorithm SHA256).Hash.ToLowerInvariant()} |
    ConvertTo-Json -Compress | Set-Content -Encoding ascii 'runs/basis-terminal.json'

$env:CUDA_VISIBLE_DEVICES = '-1'
python -I -B -X utf8 replay_reserve_pointrend.py --code .. --intake "$stock/conversion" --roster "$stock/reserve.csv" --capture "$stock/capture" --occupancy "$stock/replay" --basis runs/basis --basis-terminal runs/basis-terminal.json --fit-weights $pointRend --protocol ../../INSTANCE_RSDD_RESERVE_POINTREND_PRELOCK_20261123.md --output runs/replay --workers 4
if ($LASTEXITCODE -ne 0) { throw 'Replay failed' }
python -I -B -X utf8 score_reserve_pointrend.py --code .. --intake "$stock/conversion" --roster "$stock/reserve.csv" --capture "$stock/capture" --replay runs/replay --author-train $authorTrain --output runs/score
if ($LASTEXITCODE -ne 0) { throw 'Scoring failed' }
```

The feature capture checks the closed reserve prediction streams. The replay starts two separate pools of CPU workers and requires identical full edit bytes. The scoring program checks both realizations' common budgets before decoding the original author hulls for the selected 74 images. Predictions, features, original annotations and trained weights stay local.
