# Frozen vessel-length study on SAR Ship Gulf

The [locked protocol](../../GULF_AIS_FROZEN_PRELOCK_20261123.md) tests signed same-count HRSID boundary edits against AIS declared lengths on the [fixed public Gulf SAR release](https://zenodo.org/records/21000650). It uses an independent, metadata-selected SAR cohort and the original RF-DETR carrier, native masks and source point head. Length is the full-pixel projected span of a component at nominal 10 m spacing, on the original component fixed major axis. The registry values, local radar footprint and actual vessel association have separate error sources.

Use the [verified B5 Windows CPython 3.10.13 environment](../INSTANCE_RSDD_B5_REPRO_20261123/README.md). The checkpoint and 5217 occupancy bundle are identical to the [closed NASTaR run](../NASTAR_AIS_FROZEN_20261123/README.md). Keep private datasets, vessel identifiers, weights, all prediction masks and per-chip geometry on the operator's authorized devices. The public source and aggregate output contain no per-vessel data.

Set `$py` to the qualified interpreter, `$ckpt` to the pinned center5202 E24 checkpoint and `$head` to the pinned full-fit HRSID seed 5217 point head. Run from this directory with a fresh run root. `observe.py` binds actual process exit, worker identity and footer to its recorded result.

```powershell
$run = 'runs/gulf-ais'
$protocol = '../../GULF_AIS_FROZEN_PRELOCK_20261123.md'
$sha = (Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash.ToLowerInvariant()
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null
$lock = @('--protocol',$protocol,'--protocol-sha256',$sha)
function Invoke-Observed($worker,$phase,$options) {
    & $py -I -B -X utf8 observe.py --phase $phase --worker $worker --result "$run/$phase/RESULT_SAFE.json" --observer "$run/observed_$phase" -- $py -I -B -X utf8 $worker @options --output "$run/$phase"
    if ($LASTEXITCODE -ne 0) { throw "$phase stopped before successful observed closure" }
}

Invoke-Observed 'census.py' 'census' $lock
Invoke-Observed 'intake.py' 'intake' ($lock + @('--census',"$run/census",'--census-terminal',"$run/observed_census/TERMINAL.json",'--workers','10'))

$inputs = $lock + @('--code','..','--intake',"$run/intake",'--intake-terminal',"$run/observed_intake/TERMINAL.json")
$env:CUDA_VISIBLE_DEVICES = '0'
Invoke-Observed 'frozen.py' 'capture' (@('capture') + $inputs + @('--checkpoint',$ckpt))

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$cap = @('--capture',"$run/capture",'--capture-terminal',"$run/observed_capture/TERMINAL.json")
Invoke-Observed 'frozen.py' 'replay' (@('replay') + $inputs + $cap + @('--point-head',$head,'--workers','5'))
Invoke-Observed 'score.py' 'score' ($inputs + $cap + @('--replay',"$run/replay",'--replay-terminal',"$run/observed_replay/TERMINAL.json"))
```

Image intake uses only the archive's VV data array. The capture and replay validate their original full-size masks and native ranks before the scorer reads registry fields. The measurement uses native pixel coordinates and a fixed axis defined by the raw component. Use the reported component and common coverage alongside the paired mean absolute error change. The original mask-benchmark AP values remain in the [evidence matrix](../../SAR_SHIP_MASK_TGRS_EVIDENCE_MATRIX_20261112.md).

[Observed experiment](../../GULF_AIS_FROZEN_ACTUAL_20261123.md) · [Aggregate score and execution receipts](../../results/GULF_AIS_FROZEN_ACTUAL_SAFE_20261123.json)
