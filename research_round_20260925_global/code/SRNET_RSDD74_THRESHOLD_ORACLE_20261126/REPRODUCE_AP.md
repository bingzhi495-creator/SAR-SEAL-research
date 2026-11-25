# Reproduce the threshold diagnostic

Use the [original SRNet protocol and CPU runtime](../SRNET_SSDD_E24_RSDD74_PAIRED_REPLAY_20261126/REPRODUCE_REPLAY.md). The original `$previous` score root must store its exact private authority and bound capture, replay, score and terminal records. The worker verifies all records before reading the same historical author reference. Supply the accepted `$py` interpreter, `$scripts` source folder, a fresh existing `$work` folder, and the [released observer](../LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py) as `$observer`.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$protocol = Join-Path $scripts 'PRELOCK.md'
$pin = '23e98ad86fae80a9fbfed17f22f9654f66cec33342cb4226fb60d931f411a4c9'

& $py -I -B -X utf8 "$scripts/test_threshold_audit.py"
if ($LASTEXITCODE -ne 0) { throw 'Threshold diagnostic fixtures failed.' }
& $py -I -B -X utf8 $observer --log "$work/observed_ap" --phase score `
  --worker "$scripts/threshold_audit.py" --result "$work/actual/RESULT_SAFE.json" -- `
  --previous $previous --protocol $protocol --protocol-sha256 $pin --output "$work/actual"
if ($LASTEXITCODE -ne 0) { throw 'Observed threshold diagnostic failed.' }
```

The observed diagnostic result stays in the private work directory. Its auxiliary per-arm status field stores counts by native image identifier. The public projection reconstructs global status totals from the nine transition counts and checks consistency across comparators. Its schema includes only aggregate scalars and fixed semantic keys. Set `$observedDiagnostic` to the directory holding the original witnessed diagnostic. Its exact result and terminal are bound to the public projection. A fresh run supplies new wall-clock and process metadata, so the pinned original receipts remain the inputs to this projection.

```powershell
& $py -I -B -X utf8 "$scripts/project_ap_release.py" `
  --result "$observedDiagnostic/actual/RESULT_SAFE.json" --terminal "$observedDiagnostic/observed_ap/TERMINAL.json" `
  --output "$work/AP_AGGREGATE_SAFE.json"
if ($LASTEXITCODE -ne 0) { throw 'Aggregate release projection failed.' }
```
