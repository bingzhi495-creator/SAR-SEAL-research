# Reproduce the frozen prior ablation

Use the existing qualified, CPU-only [SRNet replay runtime](../SRNET_SSDD_E24_RSDD74_PAIRED_REPLAY_20261126/REPRODUCE_REPLAY.md). Supply your accepted interpreter as `$py`, this source folder as `$scripts`, and a fresh existing directory as `$work`. The `$previous` directory is the original private SRNet replay root. It must contain the original bound capture, replay, score and terminal records. Controlled assets and their records are subject to the original archive access and distribution terms. No detector run is needed.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$observer = Join-Path $scripts '../LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py'
$protocol = Join-Path $scripts 'PRELOCK.md'
$pin = 'd14c88a154fa2c3e436f0763e70c995372063cd62603fb1c9d94b3c20f8367e2'
$commit = '17948d219a19c7cee62639456f328524dddec398'
function Sha([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

& $py -I -B -X utf8 "$scripts/test_prior_replay.py"
if ($LASTEXITCODE -ne 0) { throw 'Frozen readout arithmetic fixture failed.' }

$renderer = "$scripts/prior_replay.py"
foreach ($phase in 'prepare-edits','verify-edits') {
    & $py -I -B -X utf8 $observer --log "$work/observed_$phase" --phase $phase `
        --worker $renderer --result "$work/$phase/RESULT_SAFE.json" -- `
        --previous $previous --protocol $protocol --protocol-sha256 $pin `
        --prelock-commit $commit --output "$work/$phase"
    if ($LASTEXITCODE -ne 0) { throw "Frozen readout failed in $phase." }
}

$render = "$work/prepare-edits"
$verify = "$work/verify-edits"
$rt = "$work/observed_prepare-edits/TERMINAL.json"
$vt = "$work/observed_verify-edits/TERMINAL.json"
& $py -I -B -X utf8 $observer --log "$work/observed_score" --phase score `
    --worker "$scripts/prior_score.py" --result "$work/score/RESULT_SAFE.json" -- `
    --previous $previous --protocol $protocol --protocol-sha256 $pin --prelock-commit $commit `
    --render $render --render-terminal $rt --render-terminal-sha256 (Sha $rt) `
    --render-result-sha256 (Sha "$render/RESULT_SAFE.json") --render-worker-sha256 (Sha $renderer) `
    --verify $verify --verify-terminal $vt --verify-terminal-sha256 (Sha $vt) `
    --verify-result-sha256 (Sha "$verify/RESULT_SAFE.json") --output "$work/score"
if ($LASTEXITCODE -ne 0) { throw 'Frozen score failed.' }
```

The render and verification stream files stay with the private input archive. The score publishes aggregate views, contrast estimates and run receipts.

