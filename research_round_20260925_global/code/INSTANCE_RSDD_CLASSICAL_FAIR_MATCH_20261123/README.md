# Instance-RSDD spatial attraction at equal realized counts

The [prelocked protocol](../../INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_PRELOCK_20261123.md) compares five edited masks against raw on all 74 already exposed parent-reserve chips. Each candidate uses the same signed endpoint support and the exact positive count realized by dynamic classical spatial attraction. The static control uses the same classical kernel without updates. The three fixed occupancy heads use identical signed counts.

Use the [verified B5 stock archive](../INSTANCE_RSDD_B5_REPRO_20261123/README.md), the [closed 74-chip PointRend roster](../INSTANCE_RSDD_RESERVE_POINTREND_20261123/README.md) to bind the same native candidates and previous occupancy output, and the three frozen FIT109 full-fit bundles. Keep all three checkpoints and the author images and annotations local. The scope is the 74-image archive. The earlier 740-image result supplies historical context.

Run in the B5 isolated CPython 3.10.13 CPU environment. Use the unchanged [stage observer](../HRSID_DEV705_CLASSICAL_ATTRACTION_20261122/observe.py) to bind each worker's actual exit and footer. From this code directory, with `$py` set to the qualified standalone interpreter, use a new run directory.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$root = 'runs/attraction-74'
$protocol = '../../INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_PRELOCK_20261123.md'
$sha = (Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash.ToLowerInvariant()
$observe = '../HRSID_DEV705_CLASSICAL_ATTRACTION_20261122/observe.py'
$source = @('--cohort','74','--code','..','--stock-root','runs/closed-b5-74',
    '--point-root','runs/closed-point-74','--protocol',$protocol,'--protocol-sha256',$sha)
$replay = $source + @('--weights-dir','weights/FIT109_frozen')
New-Item -ItemType Directory -Path $root -ErrorAction Stop | Out-Null
& $py -I -B -X utf8 $observe --worker replay.py --phase prepare-edits --result "$root/prepare-edits/RESULT_SAFE.json" --observer "$root/observed_prepare-edits" -- $py -I -B -X utf8 replay.py prepare-edits @replay --workers 6 --output "$root/prepare-edits"
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
& $py -I -B -X utf8 $observe --worker replay.py --phase verify-edits --result "$root/verify-edits/RESULT_SAFE.json" --observer "$root/observed_verify-edits" -- $py -I -B -X utf8 replay.py verify-edits @replay --workers 4 --prepare "$root/prepare-edits" --prepare-terminal "$root/observed_prepare-edits/TERMINAL.json" --output "$root/verify-edits"
if ($LASTEXITCODE -ne 0) { throw 'Independent replay failed' }
& $py -I -B -X utf8 $observe --worker score.py --phase score --result "$root/score/RESULT_SAFE.json" --observer "$root/observed_score" -- $py -I -B -X utf8 score.py @source --prepare "$root/prepare-edits" --prepare-terminal "$root/observed_prepare-edits/TERMINAL.json" --verify "$root/verify-edits" --verify-terminal "$root/observed_verify-edits/TERMINAL.json" --author-labels 'data/annotation_coco/train.json' --output "$root/score"
if ($LASTEXITCODE -ne 0) { throw 'Scoring failed' }
```

The edit worker keeps the archived candidate order and rechecks the original natural seed-5217 output. Both worker invocations verify every signed pixel set, and the scorer rechecks the edit counts before admitting the author annotations. The independent replay changes only execution partition. Actual occupancy-head invocations are in `counts.point_head_forwards`. The legacy `model_forwards` and `detector_forwards` fields count new carrier detector calls. [Results and receipts](../../INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_ACTUAL_20261123.md)
