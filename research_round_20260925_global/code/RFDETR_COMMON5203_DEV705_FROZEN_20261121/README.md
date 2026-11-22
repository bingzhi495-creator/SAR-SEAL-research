# Frozen endpoint selection on a second RF-DETR detector

This package evaluates an independently trained common-retention RF-DETR Seg Nano center E24 checkpoint (detector seed 5203) on the 705-image HRSID development set. Three point-head fits from the first detector (seeds 5217, 5218 and 5219), their scalers, candidate support and edit budgets stay frozen. The second detector supplies its own native predictions, scores and masks.

The primary comparison is independent selection versus fixed pairing at the same count and with the same head (SU=F minus SF). Report the paired effects within each detector and their difference, including all three head seeds. Secondary comparisons include SU against SF, N and LW=S, and SU against SU=F. The original protocol sets no advancement threshold for this replication. Two detector fits provide two observed effects; the shared point-head seeds do not add independent detector fits. [Original training and comparison protocol](../RFDETR_COMMON5203_E24_REPLICATION_20261121/PRELOCK.json) · [Frozen capture protocol](PRELOCK.json)

## Evaluation

The native and refinement passes each execute a CUDA FP32 forward with TF32 disabled. They verify agreement of the logits, boxes and all 164 mask planes on each image. A CPU reader reconstructs each treatment stream, checking native query identity, rank, score, edits and area. For posterior interpolation disagreements, the capture preserves the evidence specified in the frozen protocol.

The scorer uses all 1,782 original annotations for its primary COCO evaluation. Sensitivity scoring removes the same four empty raster annotations and retains all 705 images. Each native or treatment stream receives the twelve standard COCO metrics and AP90. Continuous IoU uses original valid native matches at raster IoU 0.50, including unchanged candidates with native mask area from 32 through 1,023 pixels². The score uses the observed edited area for LW=S. It averages the three seed effects within each match and resamples filename-prefix groups 4,096 times with PCG64 seed 5217. The interval describes variation across the represented sources of this fixed detector. Strict continuous IoU >0.90 losses and gains are reported separately. The earlier detector contributes only its published aggregate to the comparison. [Historical DEV705 score implementation](../RFDETR_MATCHED_POINT_FORMAL_20261120/score_dev.py) · [First detector results](../../results/RFDETR_CENTER5202_DEV705_MATCHED_SCORE_PUBLIC_SAFE_20261120.json)

## Inputs and environment

Run the examples from this directory using PowerShell and a Python 3.10.13 environment. Capture checks RF-DETR 1.12.0, PyTorch 2.10.0+cu130, NumPy 1.26.4, Pillow 12.3.0 and pycocotools 2.0.11. Continuous geometry requires Shapely 2.0.2 and GEOS 3.11.2. An actual capture requires separately available frozen inputs and the exact historical treatment implementation identified below.

| Environment variable | Local input or output |
|---|---|
| `SAR_SEAL_RELEASE_ROOT` | Optional path to the research release containing sibling `code`, `configs` and `results` directories. The default uses this package's location under `code`. |
| `SAR_SEAL_TRAINING_RECEIPT` and `SAR_SEAL_TRAINING_SHA` | Closed 5203 training acceptance file and its independently observed SHA-256. |
| `SAR_SEAL_CHECKPOINT` | Accepted 5203 E24 detector checkpoint. |
| `SAR_SEAL_DEV705_INPUTS` | Closed DEV705 input directory supplying the original manifest and the published evaluator's audit and annotation inputs. |
| `SAR_SEAL_PNG_ARCHIVE` | Original HRSID PNG archive. |
| `SAR_SEAL_TREATMENT_SOURCE` | Historical frozen treatment implementation with SHA-256 `a50da3d740d9620a23c98b978b7cfaa383040f1e7542e2eb5da32ca4b0106da6`. Use the [exact archived treatment source](../HRSID_OFFICIAL_FROZEN_REFINEMENT_CAPTURE_READBACK_20261121/capture.py). |
| `SAR_SEAL_FIT_RESULT`, `SAR_SEAL_FIT_ACCEPTANCE`, `SAR_SEAL_POINT_HEADS` | Closed original FIT109 point-head result, its acceptance and the directory of frozen heads/scalers. |
| `SAR_SEAL_SHAPELY_SITE` | Optional directory for the pinned geometry installation when the interpreter cannot otherwise import it. |
| `SAR_SEAL_CAPTURE_OUT`, `SAR_SEAL_READBACK_OUT`, `SAR_SEAL_SCORE_OUT` | New capture directory, new independent readback directory, and new score JSON file. |

## Commands

The synthetic suite uses generated masks, logits, labels and receipts. It requires the published modules, the Python dependencies above and the geometry runtime. Its capture tests exercise the relocated checkpoint loader, raw-output digests, same-device correction criteria and all twelve reconstructed streams.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python -I -B -X utf8 -m unittest discover -s . -v
```

For real replay, make the inputs available and start the capture process. The capture launches its two original CUDA stages and records their exits.

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
python -I -B -X utf8 ./capture_dev.py run --training-acceptance-sha256 $env:SAR_SEAL_TRAINING_SHA --output $env:SAR_SEAL_CAPTURE_OUT
```

After successful capture, set `SAR_SEAL_ADMISSION_SHA` to the independently recorded admission digest and `SAR_SEAL_TERMINAL_SHA` to the observed capture terminal digest. Run the independent readback on CPU. Record its actual exit and obtain `SAR_SEAL_READBACK_SHA` from the successful output footer.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -I -B -X utf8 ./score_5203.py readback --capture $env:SAR_SEAL_CAPTURE_OUT --training $env:SAR_SEAL_TRAINING_RECEIPT --training-sha256 $env:SAR_SEAL_TRAINING_SHA --admission-sha256 $env:SAR_SEAL_ADMISSION_SHA --terminal-sha256 $env:SAR_SEAL_TERMINAL_SHA --output $env:SAR_SEAL_READBACK_OUT
```

With the readback footer verified and its digest set, run the score phase on CPU and record its terminal status.

```powershell
python -I -B -X utf8 ./score_5203.py score --native (Join-Path $env:SAR_SEAL_CAPTURE_OUT 'native/DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl') --accepted $env:SAR_SEAL_READBACK_OUT --accepted-sha256 $env:SAR_SEAL_READBACK_SHA --training $env:SAR_SEAL_TRAINING_RECEIPT --training-sha256 $env:SAR_SEAL_TRAINING_SHA --output $env:SAR_SEAL_SCORE_OUT
```

Planning estimates are 1–2 hours for capture, 5–25 minutes for independent readback and 15–60 minutes for CPU scoring on the reference host. These are estimates for the second detector.

## Portability and verification

The public copy resolves input locations through environment variables and uses the invoking interpreter for both capture subprocesses. The pinned historical source loader resolves its modules from the supplied release tree. Actual local paths and per-image records are written only to operator-selected output locations. The frozen capture protocol is byte-identical to the original (SHA-256 `664fa755a55e4ea732fe3c0ae04c67c16e4ad00079a70dc942d8aee1d2c22127`). Published implementation, detector, head, dataset identity and runtime checks are retained. New runs record the hashes of the portable adapters they actually execute.

Nine generated CPU tests passed for the portable capture and score adapters. The stream reconstruction test uses a published synthetic point witness and checks all twelve outputs byte for byte. These tests supply evidence for the adapters; scientific results require a completed independently accepted detector capture and score.

The first live refinement attempt stopped on a same-device mask parity mismatch. PyTorch dispatches CUDA bilinear interpolation to a different implementation under deterministic mode. The corrected fallback invokes the native RF-DETR interpolation path and restores both prior deterministic settings. It still checks every pixel against the frozen native mask before taking the sigmoid. On the failing real image, the corrected path reproduced the author's interpolation tensor exactly and rejected a deliberately altered native mask. The frozen protocol remains byte-identical.
