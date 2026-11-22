# Fractional and half-cell target comparisons

Use the [locked experiment](../../RFDETR_FRACTIONAL_TIE_TRANSFER_PRELOCK_20261123.md), the accepted [HRSID detector feature capture](../RFDETR_CENTER5202_FULL_CAPTURE_20261120/PROTOCOL.md), original [RF-DETR FIT109 training](../RFDETR_MATCHED_POINT_FORMAL_20261120/run_fit109.py) inputs and receipts, and the [reproduced 74-image Instance-RSDD reserve](../INSTANCE_RSDD_B5_REPRO_20261123/README.md). The three archived fractional heads and accepted native inputs remain local. Run from this directory with the published code and the qualified CPython 3.10 environment. The source trainer uses the original Shapely site and checks the full original NPZ bundles before fitting new labels.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$plan = '../../RFDETR_FRACTIONAL_TIE_TRANSFER_PRELOCK_20261123.md'
$lock = (Get-FileHash -LiteralPath $plan -Algorithm SHA256).Hash.ToLowerInvariant()
$stock = 'runs/closed-74'
$authorTrain = 'data/annotation_coco/train.json'

function Invoke-Observed($worker, $phase, $result, $terminal, $arguments) {
    $out = @(python -I -B -X utf8 $worker @arguments)
    $actual = $LASTEXITCODE
    if ($actual -ne 0) { throw "$phase failed with exit $actual" }
    $sha = (Get-FileHash -LiteralPath $result -Algorithm SHA256).Hash.ToLowerInvariant()
    if (($out[-1] | ConvertFrom-Json).result_sha256 -ne $sha) { throw "$phase footer differs" }
    @{phase=$phase; actual_exit=$actual; output_footer_verified=$true; result_sha256=$sha;
      worker_sha256=(Get-FileHash -LiteralPath $worker -Algorithm SHA256).Hash.ToLowerInvariant()} |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $terminal -Encoding ascii
}

$fit = @('--code','..','--capture','runs/closed-fit-capture/FIT109_CONTROL/capture',
    '--acceptance','runs/closed-fit-capture/INDEPENDENT_RESULT_SAFE.json',
    '--source-cohort','data/FIT109_cohort.json',
    '--fit-images','data/HRSID_source/train','--fit-gt','data/HRSID_source/train/_annotations.coco.json',
    '--shapely-site','dependencies/original-shapely','--historical-fit','runs/closed-all-fit',
    '--protocol',$plan,'--protocol-sha256',$lock,'--output','runs/train')
Invoke-Observed 'train_fractional.py' 'train' 'runs/train/RESULT_SAFE.json' 'runs/train-terminal.json' $fit
$replay = @('--code','..','--stock-root',$stock,'--source','runs/train','--source-terminal','runs/train-terminal.json',
    '--protocol',$plan,'--output','runs/replay','--workers','5')
Invoke-Observed 'replay_fractional.py' 'replay' 'runs/replay/RESULT_SAFE.json' 'runs/replay-terminal.json' $replay
python -I -B -X utf8 score_fractional.py --code .. --roster "$stock/reserve.csv" --intake "$stock/conversion" --capture "$stock/capture" --stock "$stock/replay" --replay runs/replay --replay-terminal runs/replay-terminal.json --protocol $plan --author-train $authorTrain --output runs/score
```

The scorer binds every arm to the archived per-candidate count and source seed. It reads the official reserve polygons after verifying full label-free replay. Output publication includes aggregate metrics and digests. Head bundles, predictions, source features and full annotations stay local.
