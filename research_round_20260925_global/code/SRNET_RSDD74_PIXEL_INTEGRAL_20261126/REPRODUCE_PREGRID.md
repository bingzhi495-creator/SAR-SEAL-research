# Observe the native SRNet probability grid before its final resize

The worker loads the same qualified Windows CPU environment and immutable inputs as the [earlier capture](../SRNET_SSDD_E24_RSDD74_PAIRED_REPLAY_20261126/REPRODUCE_CAPTURE.md). Use the previous capture root as `$original` and the accepted legacy Python 3.10.13 executable as `$py`. The earlier root must contain its exact capture worker, observer closure and private native probability archive. Supply the original author source tree, project SSDD FIT741 E24 checkpoint root, JPEG-only intake, closed projection pilot, published prelock file and fresh existing working folder. Set `$protocolSha` and `$prelockCommit` to the independently verified publication pins.

The observer relays the original author interpolation call unchanged. It records the float32 input after the projected grid is unpadded to its actual `img_shape` and the float32 output before the stock 0.50 threshold. Earlier archive probabilities and exact stock output identities are checked against this call. The independent readback opens the newly written pregrids and reconstructs the same 512-by-512 probabilities with the qualified legacy runtime.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$observer = Join-Path $sourceDir '../LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py'
$worker = Join-Path $sourceDir 'capture_pregrid.py'
function Sha([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
& $py -I -B -X utf8 "$sourceDir/test_capture_pregrid.py"
if ($LASTEXITCODE -ne 0) { throw 'Projection observation fixture failed.' }
& $py -I -B -X utf8 $observer --log "$work/observed_capture" --phase capture `
    --worker $worker --result "$work/actual/RESULT_SAFE.json" -- capture `
    --source $author --fit $sourceFit --intake $intake --pilot $pilot --original $original `
    --protocol $acceptedProtocol --protocol-sha256 $protocolSha --prelock-commit $prelockCommit `
    --workers 4 --output "$work/actual"
if ($LASTEXITCODE -ne 0) { throw 'Observed author pregrid capture failed.' }

$term = "$work/observed_capture/TERMINAL.json"
$captured = "$work/actual/RESULT_SAFE.json"
& $py -I -B -X utf8 $observer --log "$work/observed_readback" --phase verify `
    --worker $worker --result "$work/readback/RESULT_SAFE.json" -- verify `
    --capture "$work/actual" --capture-terminal $term --capture-terminal-sha256 (Sha $term) `
    --capture-result-sha256 (Sha $captured) --original $original --protocol $acceptedProtocol `
    --protocol-sha256 $protocolSha --prelock-commit $prelockCommit --output "$work/readback"
if ($LASTEXITCODE -ne 0) { throw 'Observed pregrid readback failed.' }
```

The private index preserves every earlier original row and candidate record. It adds `pregrid_file` with byte count and SHA-256, `resized_hw`, `padded_hw`, `native_hw` and `final_resize_called`. Each pregrid archive has one field, `pregrid`, with shape candidate count by unpadded height by unpadded width. The origin is the top-left sample of the unpadded grid. A row without candidates has an empty first dimension and invokes no final resize. The earlier `probability_file` references the previous capture root. All per-image rows and arrays remain private.

