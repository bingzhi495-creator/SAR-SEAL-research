# Location substitution at PointRend's signed edit counts

The [locked comparison](../../POINTREND_SIGNED_QUOTA_SWAP_PRELOCK_20261123.md) keeps the exact original detections, scores, PointRend prediction and per-side counts within each candidate's capped signed support. For each of the three frozen occupancy heads, it changes the locations using that head's ordering while leaving the counts fixed. Two paired comparisons use native or PointRend pixels outside the support.

Use the [74-image frozen stock and PointRend output](../INSTANCE_RSDD_RESERVE_POINTREND_20261123/README.md), the [740-image clean stock](../SAR_B5_MAIN_TABLES_20261123/README.md), and the [actual 740-image PointRend archive](../../INSTANCE_RSDD_POINTREND_TRANSFER_ACTUAL_20261122.md). The 740-image arm also needs the original [support splice](../../INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_ACTUAL_20261123.md) and its closed replay. All source and target inputs remain local to the operator.

In the qualified B5 Windows CPU environment, create a new output directory for each replay and score. From this directory, use the explicit original run roots. The `Invoke-Observed` PowerShell helper in the [fractional supervision reproducibility entry](../RFDETR_FRACTIONAL_TIE_20261123/README.md) records a worker's actual exit and accepted result hash. Label paths enter the scoring stage.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$protocol = '../../POINTREND_SIGNED_QUOTA_SWAP_PRELOCK_20261123.md'
$sha = (Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash.ToLowerInvariant()
$heads = 'weights/FIT109_frozen'
$opt74 = @('--code','..','--cohort','74','--stock-root','runs/closed-stock-74','--point-root','runs/closed-point-74')
$opt740 = @('--code','..','--cohort','740','--stock-root','runs/closed-stock-740',
    '--point-root','runs/closed-point-740','--support-root','runs/closed-support-740')
$replay74 = $opt74 + @('--weights-dir',$heads,'--protocol',$protocol,'--protocol-sha256',$sha,'--output','runs/quota-74')
$replay740 = $opt740 + @('--weights-dir',$heads,'--protocol',$protocol,'--protocol-sha256',$sha,'--output','runs/quota-740')
Invoke-Observed 'replay_quota_swap.py' 'replay' 'runs/quota-74/RESULT_SAFE.json' 'runs/terminal-74.json' $replay74
Invoke-Observed 'replay_quota_swap.py' 'replay' 'runs/quota-740/RESULT_SAFE.json' 'runs/terminal-740.json' $replay740
python -I -B -X utf8 score_quota_swap.py @opt74 --replay runs/quota-74 --replay-terminal runs/terminal-74.json --protocol $protocol --author-labels data/annotation_coco/train.json --output runs/score-74
if ($LASTEXITCODE -ne 0) { throw 'Reserve score failed' }
python -I -B -X utf8 score_quota_swap.py @opt740 --replay runs/quota-740 --replay-terminal runs/terminal-740.json --protocol $protocol --author-labels data/annotation_coco/test.json --output runs/score-740
if ($LASTEXITCODE -ne 0) { throw 'Author test score failed' }
```

The scores recheck the archived controls and the cross-background intersection invariant on original matched polygons. Both previous comparisons and current changes use frozen candidates. [Verified results and receipts](../../POINTREND_SIGNED_QUOTA_SWAP_ACTUAL_20261123.md). The whole-RoI hybrid requires the original PointRend path.
