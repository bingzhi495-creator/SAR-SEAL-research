# Signed-side ablation of fixed pairing

This [prespecified experiment](../../INDEPENDENT_SIDE_FACTORIAL_PRELOCK_20261123.md) crosses the archived balanced edits from the fixed-pair and independent policies. The first letter identifies the policy for additions and the second for removals. FF and II are the original arms, IF releases additions, and FI releases removals. Every candidate has the same scores, masks before editing and actual counts of additions and removals in all four arms.

Use the [74-image verified capture and edit archive](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) and the [740-image verified capture and both occupancy replay archives](../SAR_B5_MAIN_TABLES_20261123/README.md). The 740-image stock root is the run directory containing `capture`, `replay5217`, `replay5218` and their matching original observations. Keep original author labels local. Run from this source directory in the same B5 CPython 3.10 environment with Shapely 2.0.2 and GEOS 3.11.2.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$protocol = '../../INDEPENDENT_SIDE_FACTORIAL_PRELOCK_20261123.md'
$sha = (Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash.ToLowerInvariant()
$cohort = 74
$stock = 'runs/closed-reserve74'
$label = 'data/annotation_coco/train.json'
# For the author test set use cohort 740, its verified stock root and author test.json.
$common = @('--code','..','--cohort',"$cohort",'--stock-root',$stock,'--protocol',$protocol,'--protocol-sha256',$sha)
New-Item -ItemType Directory -Path runs/side-factorial -ErrorAction Stop | Out-Null
$replay = 'runs/side-factorial/replay'
$terminal = 'runs/side-factorial/replay-terminal.json'
$replayArgs = $common + @('--output',$replay)
# Define Invoke-Observed as shown in the linked observer example.
Invoke-Observed 'replay_sides.py' 'replay' "$replay/RESULT_SAFE.json" $terminal $replayArgs
python -I -B -X utf8 score_sides.py @common --replay $replay --replay-terminal $terminal --author-labels $label --output runs/side-factorial/score
if ($LASTEXITCODE -ne 0) { throw 'Side ablation scoring failed' }
```

The [observer example](../RFDETR_FRACTIONAL_TIE_20261123/README.md) checks the actual replay exit and output footer before admitting masks to scoring. The scorer verifies the frozen inputs and signed mask sets before opening the author labels. It reproduces the original arm metrics and reports all 13 COCO metrics, original matched geometry, group intervals, and exposure for both primary contrasts.

[Verified results and run receipts](../../INDEPENDENT_SIDE_FACTORIAL_ACTUAL_20261123.md).
