# Phoenix and occupancy signed-side cross on the 74-chip reserve

The [locked comparison](../../RSDD74_PHOENIX_SIGNED_SIDES_PRELOCK_20261123.md) preserves exact frozen stock masks, scores, candidate order, endpoint locations and the dynamic classical count from the [observed Phoenix comparison](../../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md). First letters select additions and second letters select removals. P is Phoenix, and L is one frozen occupancy seed. PP and each LL are archived. LP and PL combine signed mask differences in native pixel space.

Use the existing [B5 Windows CPU environment](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) and the closed [74-chip PointRend and native archives](../INSTANCE_RSDD_RESERVE_POINTREND_20261123/README.md). Keep original annotation files, weights, candidates and raw edits in authorized research storage. Use `-I -B -X utf8` with the qualified Python. The unchanged [process observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py) records actual zero exits and matching result footers.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$root = 'runs/phoenix-signed-sides-74'
$phx = 'runs/closed-phoenix-74'
$lock = '../../RSDD74_PHOENIX_SIGNED_SIDES_PRELOCK_20261123.md'
$sha = (Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash.ToLowerInvariant()
$obs = '../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py'
$common = @('--code','..','--stock-root','runs/closed-b5-74','--point-root','runs/closed-point-74',
    '--phx-edits',"$phx/edits",'--phx-edits-terminal',"$phx/observed_edits/TERMINAL.json",
    '--protocol',$lock,'--protocol-sha256',$sha)
New-Item -ItemType Directory -Path $root -ErrorAction Stop | Out-Null
& $py -I -B -X utf8 $obs --phase edits --worker replay.py --result "$root/edits/RESULT_SAFE.json" --observer "$root/observed_edits" -- $py -I -B -X utf8 replay.py @common --output "$root/edits"
if ($LASTEXITCODE -ne 0) { throw 'Verified edit replay failed' }
& $py -I -B -X utf8 $obs --phase score --worker score.py --result "$root/score/RESULT_SAFE.json" --observer "$root/observed_score" -- $py -I -B -X utf8 score.py @common --phx-score "$phx/score" --phx-score-terminal "$phx/observed_score/TERMINAL.json" --replay "$root/edits" --replay-terminal "$root/observed_edits/TERMINAL.json" --author-labels 'data/annotation_coco/train.json' --output "$root/score"
if ($LASTEXITCODE -ne 0) { throw 'Closed factorial score failed' }
```

The scorer validates every historical AP and the signed fixed-area interaction. The two reported 97.5% intervals are paired over sensor/date groups, with equal parent weights, and adjust the two side hypotheses together. The secondary sensor views do not replace the 74-chip result. AP interactions reflect the observed complete candidate-set response to a simultaneous mask change.

[Observed experimental result](../../RSDD74_PHOENIX_SIGNED_SIDES_ACTUAL_20261123.md) · [Aggregate results and observed receipts](../../results/RSDD74_PHOENIX_SIGNED_SIDES_ACTUAL_SAFE_20261123.json)
