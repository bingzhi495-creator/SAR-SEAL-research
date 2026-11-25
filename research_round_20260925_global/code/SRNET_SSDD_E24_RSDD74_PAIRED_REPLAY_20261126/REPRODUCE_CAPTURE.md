# Frozen SRNet native capture on the 74-image reserve

The worker uses the original SRNet author code and the project's closed PSeg-SSDD FIT741 E24 model. It verifies the earlier two-pass native projection qualification and retains the source-trained model's le90, 608-pixel, 0.30 confidence, 0.10 rotated NMS and strict native probability (p>0.50) settings. Each independent CPU process uses one numerical thread. The original image stream contributes all 74 rows in its frozen order.

Each candidate retains its original score, native rotated box, class, rank, native COCO RLE and axis-aligned bounds enclosing the four rotated corners, clipped to the 512-pixel canvas. The saved probability array contains every native pixel of every stock candidate, including empty masks. Each array is read back and checked bitwise against its memory representation. The independent native projection must reproduce the exact original binary mask. All per-image arrays, candidate rows and images stay in the private execution directory.

## Execution inputs

Use the approved Windows legacy CPU runtime from the [existing SRNet run](../../SRNET_SSDD_FIT741_BASELINE_NATIVE_PATH_20261111.md). Supply the exact author-code directory, closed project checkpoint root, closed Instance-RSDD JPEG-only intake, earlier closed probability-projection pilot, accepted protocol and a fresh work folder. The pilot's `native_parity.py`, aggregate result and observer receipt retain their original subpaths. The [pilot instructions](../SRNET_SSDD_E24_RSDD74_NATIVE_PARITY_20261126/REPRODUCE.md) document them.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

& $py -I -B -X utf8 "$sourceDir/test_native_capture.py"
if ($LASTEXITCODE -ne 0) { throw 'Native capture fixture failed.' }

& $py -I -B -X utf8 $observer --log "$work/observed_capture" --phase capture `
  --worker "$sourceDir/native_capture.py" --result "$work/actual/RESULT_SAFE.json" -- `
  --source $author --fit $sourceFit --intake $intake --pilot $pilot `
  --protocol $acceptedProtocol --protocol-sha256 $protocolSha --workers 4 --output "$work/actual"
if ($LASTEXITCODE -ne 0) { throw 'Observed native capture failed.' }
```

Use the original linked [observer](../LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py). The private stream is `INDEX_PRIVATE.jsonl`. Each row references one `PROBABILITIES_PRIVATE` archive by exact byte count and SHA-256, with array key `probability` and shape `[N,512,512]` in preserved candidate order. A no-candidate row has `N=0`. The qualified use is reserved for a separately authorized experiment. This capture performs no optimization, mask editing or scoring.
