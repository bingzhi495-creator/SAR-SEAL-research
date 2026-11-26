# Reproduce label-free DEV705 witness coverage

Use the isolated B5 CPU environment from [the original reproduction](../LEARNED_BOUNDED_PAIRING_20261126/hrsid/REPRODUCE.md). The runner reuses the previously closed RF-DETR center5202 capture and three-head B5 sparse replay with the verified L render and prelabel receipt. It uses original scores, ranks, support, positive budgets and frozen endpoint choices. It reads no author polygon or label-derived matching roster.

Set `$code` to the directory containing this folder, `$source` to the closed B5 `dev705` run root, and `$bounded` to the closed learned bounded pairing `hrsid` run root. Set `$py` to the qualified CPython 3.10.13 executable. Before starting, verify the original capture/replay and bounded replay/score receipts show natural exit 0 and that no relevant PIDs or overlapping task are active. Create an empty `$work` parent. The worker creates its own fresh `coverage` subdirectory.

~~~powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$scripts = Join-Path $code 'DEV705_MIN_CUMULATIVE_WITNESS_20261126'
$protocol = Join-Path $scripts 'PRELOCK.md'
$solver = Join-Path $scripts 'coverage.py'
$observer = Join-Path $code 'LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py'
$sha = (Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash.ToLowerInvariant()
& $py -I -B -X utf8 $observer --log "$work/observed_coverage" --phase coverage `
    --worker $solver --result "$work/coverage/RESULT_SAFE.json" -- `
    --code $code --source $source --bounded $bounded --protocol $protocol `
    --protocol-sha256 $sha --output "$work/coverage"
if ($LASTEXITCODE -ne 0) {throw 'Coverage was not verified after natural exit'}
~~~

The program checks exact integer cumulative minima on F, L and I's selected endpoint sets. It tests the existence of a capped matching by minimizing the number of cap-violating edges over all assignments. It records all observed per-head budget and condition combinations. Omitted combinations have zero count. The released record is aggregate. The observer's full stdout/stderr remain private.

For synthetic tests use `python -I -B -X utf8 test_coverage.py` in the source folder. This covers a positive witness, the false positive from substituting L's capped sum, feasibility under a capped assignment even when an unrestricted optimum exceeds the cap, and 48 brute-force comparisons up to eight pairs.


## Author-reference scoring after the necessary gate

The original protocol requires a qualified observation showing at least 20 changed candidate identities across at least eight source groups. Verify natural closure and check the relevant PIDs again. Use the unchanged closed original B5 and bounded-pair roots. The scorer reconstructs exactly the same observed W selection census before loading the original DEV705 author reference. It adapts the original fixed geometry and COCO evaluation, using verified archived F/L/I comparator values from the original observed score. It runs W on all 67,386 ranked candidates and retains the original 1,782 annotations and 1,778 empty-raster sensitivity. It checks frozen N/F/L continuous geometry against the observed original results. The original fixed match denominator is 1,461 hulls across 27 source groups. W's changed-hull subset does not control the denominator.

~~~powershell
$cover = "$work/coverage/RESULT_SAFE.json"
$coverTerm = "$work/observed_coverage/TERMINAL.json"
$coverSha = (Get-FileHash -LiteralPath $cover -Algorithm SHA256).Hash.ToLowerInvariant()
$coverTermSha = (Get-FileHash -LiteralPath $coverTerm -Algorithm SHA256).Hash.ToLowerInvariant()
& $py -I -B -X utf8 $observer --log "$work/observed_score" --phase score `
    --worker (Join-Path $scripts 'score_witness.py') --result "$work/score/RESULT_SAFE.json" -- `
    --code $code --source $source --bounded $bounded --protocol $protocol --protocol-sha256 $sha `
    --coverage $cover --coverage-sha256 $coverSha --coverage-terminal $coverTerm `
    --coverage-terminal-sha256 $coverTermSha --output "$work/score"
if ($LASTEXITCODE -ne 0) {throw 'Score was not verified after natural exit'}
~~~
