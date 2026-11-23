# SAR Ship Gulf closed-anchor geolocation diagnostic

Reconstruct the exact previously scored 25 Gulf anchors using [the closed Gulf export](../GULF_AIS_FROZEN_20261123/README.md). Use the [fixed diagnostic plan](../../GULF_AIS_GEOSCALE_DIAGNOSTIC_LOCK_20261123.md) and qualified B5 CPython. The same pyproj 3.7.1 and PROJ 9.5.1 setup is documented in the [archive NovaSAR scorer](../NASTAR_AIS_FROZEN_20261123/README.md). The author's original archive and VV annotation XMLs remain in the local closed intake.

The `anchors.py` helper pins the original successful score and terminal and reproves its scientific aggregates. The `grid.py` helper interpolates the XML geolocation and zero-Doppler time at native sample indices, and reconstructs the author-style inverse polynomial. The diagnostic projects full-cell corner footprints along the fixed native axis and, on the same valid-heading subset, the AIS heading vector. It reports both unsigned timing possibilities. These are diagnostic comparisons on the old cohort.

From this directory set `$py` to the qualified Windows interpreter and `$geoSite` to the published pyproj wheel extracted locally. The scope uses a new run output and the original [observer](../GULF_AIS_FROZEN_20261123/observe.py).

```powershell
$run = 'runs/gulf-geolocation'
$closed = 'runs/closed-gulf-ais'
$lock = '../../GULF_AIS_GEOSCALE_DIAGNOSTIC_LOCK_20261123.md'
$old = '../../GULF_AIS_FROZEN_PRELOCK_20261123.md'
$sha = (Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash.ToLowerInvariant()
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null
& $py -I -B -X utf8 ../GULF_AIS_FROZEN_20261123/observe.py --phase score --worker diagnose.py --result "$run/score/RESULT_SAFE.json" --observer "$run/observed_score" -- $py -I -B -X utf8 diagnose.py --code .. --closed-root $closed --gulf-protocol $old --diagnostic-lock $lock --diagnostic-lock-sha256 $sha --geo-site $geoSite --output "$run/score"
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic run did not close successfully' }
```

[Observed results](../../GULF_AIS_GEOSCALE_ACTUAL_20261123.md) · [Aggregate evidence and terminal](../../results/GULF_AIS_GEOSCALE_ACTUAL_SAFE_20261123.json)
