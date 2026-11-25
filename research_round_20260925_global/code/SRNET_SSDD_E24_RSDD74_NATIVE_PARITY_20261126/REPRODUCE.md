# Reproduce the 8-chip SRNet projection check

Obtain the exact [author SRNet code](https://github.com/XIDIAN212Lab/SRNet/tree/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6), the accepted Windows CPU environment, the closed project FIT741 training folder, and the closed Instance-RSDD JPEG-only intake. The checkpoint must be the project SSDD E24 refit with SHA-256 pinned in the script. The reader supplies the checkpoint, intake and original receipts through the arguments below. The source environment uses CPython 3.10.13 and the package versions recorded in the result. [Earlier SRNet evaluation](../../SRNET_SSDD_FIT741_BASELINE_NATIVE_PATH_20261111.md) · [Reserve intake](../INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/README.md)

In PowerShell, assign `$py` to the qualified interpreter, `$scriptDir` to this code directory, `$observer` to the released [worker observer](../LEARNED_BOUNDED_PAIRING_20261126/hrsid/observe_fresh.py), `$author` to the pinned source checkout, `$fit` to the closed PSeg-SSDD FIT741 E24 training root, and `$intake` to the closed reserve JPEG intake root. Set `$work` to a new existing work directory.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$protocol = Join-Path $scriptDir 'PRELOCK.md'
$lock = '1149e32546372f8ecfefc221ee2c175901a665bdf48ef305c299894ebc3978e4'

& $py -I -B -X utf8 "$scriptDir/test_native_projection.py"
if ($LASTEXITCODE -ne 0) { throw 'Native projection fixture failed.' }

& $py -I -B -X utf8 $observer --log "$work/observed_pilot" --phase verify `
  --worker "$scriptDir/native_parity.py" --result "$work/actual/RESULT_SAFE.json" -- `
  --source $author --fit $fit --intake $intake --protocol $protocol --protocol-sha256 $lock `
  --output "$work/actual"
if ($LASTEXITCODE -ne 0) { throw 'Observed native parity verification failed.' }
```

The worker generates an image-only COCO index with an empty annotation list for the original dataset loader. The index, model weights, ROI arrays, JPEGs and process logs remain local. The result and terminal receipt expose only the counts and integrity summaries. See the [frozen pilot design](PRELOCK.md).
