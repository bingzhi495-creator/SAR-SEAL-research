# NASTaR frozen detector and AIS length evaluation

Use the published [Windows CPython environment](../INSTANCE_RSDD_B5_REPRO_20261123/README.md), the original HRSID RF-DETR center5202 E24 checkpoint and point head seed 5217. The scoring stage adds pyproj 3.7.1 and PROJ 9.5.1. Supply the unpacked matching CPython 3.10 Windows amd64 [author wheel](https://pypi.org/project/pyproj/3.7.1/#files) directory to `--geo-site`. Its wheel SHA-256 is `3537668992a709a2e7f068069192138618c00d0ba113572fdd5ee5ffde8222f3`. The input step retrieves label-free images from the [original Bristol catalog](https://data.bris.ac.uk/data/dataset/2tfa6x37oerz2lyiw6hp47058). [Locked protocol](../../NASTAR_AIS_PHYSICAL_PRELOCK_20261123.md)

The following PowerShell commands run from this directory with the pinned Python and empty output directories. Replace the three local asset variables.

```powershell
$checkpoint = 'weights/checkpoint_23.ckpt'
$head = 'weights/5217_all_fit.npz'
$geoSite = 'dependencies/pyproj-3.7.1'
$protocol = '../../NASTAR_AIS_PHYSICAL_PRELOCK_20261123.md'
python -I -B -X utf8 prepare_nastar.py lock --output runs/intake
if ($LASTEXITCODE -ne 0) { throw 'Image roster lock failed' }
$lock = (Get-FileHash 'runs/intake/cohort_lock.json' -Algorithm SHA256).Hash.ToLowerInvariant()
python -I -B -X utf8 prepare_nastar.py prepare --output runs/intake --lock-sha256 $lock
if ($LASTEXITCODE -ne 0) { throw 'Image intake failed' }
$manifest = (Get-FileHash 'runs/intake/intake.json' -Algorithm SHA256).Hash.ToLowerInvariant()

function Invoke-Observed($worker, $phase, $result, $terminal, $arguments) {
    $log = @(python -I -B -X utf8 $worker @arguments)
    $actual = $LASTEXITCODE
    if ($actual -ne 0) { throw "$phase failed with exit $actual" }
    $sha = (Get-FileHash -LiteralPath $result -Algorithm SHA256).Hash.ToLowerInvariant()
    if (($log[-1] | ConvertFrom-Json).result_sha256 -ne $sha) { throw "$phase footer differs" }
    @{phase=$phase; actual_exit=$actual; output_footer_verified=$true; result_sha256=$sha;
      worker_sha256=(Get-FileHash -LiteralPath $worker -Algorithm SHA256).Hash.ToLowerInvariant()} |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $terminal -Encoding ascii
}
$base = @('--code','..','--intake','runs/intake','--intake-sha256',$manifest,'--protocol',$protocol)
$env:CUDA_VISIBLE_DEVICES = '0'
Invoke-Observed 'frozen_nastar.py' 'capture' 'runs/capture/RESULT_SAFE.json' 'runs/capture-terminal.json' (@('capture') + $base + @('--checkpoint',$checkpoint,'--output','runs/capture'))
$env:CUDA_VISIBLE_DEVICES = '-1'
Invoke-Observed 'frozen_nastar.py' 'replay' 'runs/replay/RESULT_SAFE.json' 'runs/replay-terminal.json' (@('replay') + $base + @('--capture','runs/capture','--capture-terminal','runs/capture-terminal.json','--point-head',$head,'--output','runs/replay','--workers','5'))
python -I -B -X utf8 score_nastar_ais.py --code .. --intake runs/intake --capture runs/capture --replay runs/replay --replay-terminal runs/replay-terminal.json --protocol $protocol --geo-site $geoSite --output runs/score
```

Only the scorer reads the selected author AIS rows, after validating both independent writeback decisions and the observed replay exit. It reports physical extent errors against declared AIS dimensions. The private raster data and AIS rows stay local.
