# HRSID DEV705 frozen learned bounded pairing

This runner replays the three archived frozen RF-DETR 5202 point heads (seeds 5217, 5218 and 5219) on the historical DEV705 author-TRAIN-derived split. It compares archived distance-first fixed pairs (F), a posterior-optimal exact k matching (L) under the maximum squared displacement realized by F, and archived independent endpoints (I) at F's exact signed count. The primary comparison is I−L. The explanatory comparison is L−F. See the [common preregistered protocol](../PRELOCK.md).

## Prerequisites

Use Windows CPython 3.10.13 with the qualified B5 runtime, including NumPy 1.26.4, Torch 2.10.0+cu130, pycocotools 2.0.11, Shapely 2.0.2 and Pillow 12.3.0. The replay and score run on CPU. Read the published [B5 dev runners](../../SAR_B5_MAIN_TABLES_20261123) for upstream dependency and capture details. Signed matching uses the released [exact solver](../../PHOENIX_BOUNDED_MATCH_20261123/matching.py). The scorer uses the released [compressed RLE routine](../../RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py).

The observer requires byte-exact authorized B5 intake, capture, replay and score inputs and observed terminal receipts. The parent of the private dev folders is the reader-supplied `$source`. It contains `intake`, `capture`, `replay`, `score`, and their matching `observed_*` folders. The replay additionally uses the matching official HRSID PNG archive and the original three frozen FIT109 point head states. Archive images and author annotations can be obtained through the HRSID data instructions and the authorized B5 preparation. No alternative generated stems, substitute labels, or weights satisfy the observed lineage checks. This release contains scripts and aggregate receipts. The bound inputs themselves require the reader's authorized copy.

## Run in a fresh work directory

In PowerShell set `$py` to the qualified Python executable, `$code` to the release's `code` directory, `$scripts` to this source directory, and `$protocol` to the exact common PRELOCK file. Set `$source`, `$imageArchive`, and `$heads` to the qualified inputs described above. Set `$work` to a new empty directory for outputs. From that shell:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$protocolSha = 'f6b235aa2d5abb76e6948f04e6ed78ddc2edcb72fced3d11691a8e1e62e8b1dc'
$solver = Join-Path $code 'PHOENIX_BOUNDED_MATCH_20261123/matching.py'
$observer = Join-Path $scripts 'observe_fresh.py'

& $py -I -B -X utf8 $observer --log "$work/observed_replay" --phase replay `
    --worker "$scripts/render_dev.py" --result "$work/render/RESULT_SAFE.json" -- `
    --code $code --source $source --archive $imageArchive --heads $heads `
    --matching $solver --protocol $protocol --protocol-sha256 $protocolSha `
    --workers 6 --output "$work/render"
if ($LASTEXITCODE -ne 0) { throw 'Replay did not naturally qualify.' }

$renderEnd = "$work/observed_replay/TERMINAL.json"
$renderEndSha = (Get-FileHash -LiteralPath $renderEnd -Algorithm SHA256).Hash.ToLowerInvariant()
$renderResultSha = (Get-FileHash -LiteralPath "$work/render/RESULT_SAFE.json" -Algorithm SHA256).Hash.ToLowerInvariant()

& $py -I -B -X utf8 $observer --log "$work/observed_score" --phase score `
    --worker "$scripts/score_dev.py" --result "$work/score/RESULT_SAFE.json" -- `
    --code $code --source $source --render "$work/render" --render-terminal $renderEnd `
    --render-terminal-sha256 $renderEndSha --render-result-sha256 $renderResultSha `
    --protocol $protocol --protocol-sha256 $protocolSha --workers 4 --output "$work/score"
if ($LASTEXITCODE -ne 0) { throw 'Author scoring did not naturally qualify.' }
```

The score worker verifies the independent candidate, support, quota, cap and closure census before it opens the historical DEV author reference. It recomputes both archived comparators and the native anchor. The exact original COCO and fixed-match geometry metrics must match the closed B5 receipt. Only then does it write the new score summary. `STDOUT.log` and `STDERR.log` are operator-local. Leave frozen source and raw replay outputs outside the release.

To run the independent synthetic checks, invoke `test_render_dev.py` and `test_score_dev.py` with the same Python flags and dependencies.
