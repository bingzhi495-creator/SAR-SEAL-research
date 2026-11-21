# RF-DETR common-retention center E24 replication

This owner trains one additional center detector on Windows. It reconstructs the original 5202 trainer from its nine pinned sources and changes its downstream training seed to 5203. The original repeat hook sets that seed after the COCO model has been initialized and expanded from 100 to 164 queries. It records the effective Lightning, Torch CPU, and CUDA seeds before the first training step. Training uses the same 2,937 FIT images and their original index, the same common-row target selection, and the original 576-pixel, BF16, batch-one recipe with four-step gradient accumulation. The E100 scheduler runs to the fixed E24 endpoint.

## Execution on the historical Windows host

Set `SAR_SEAL_RUNTIME_ROOT` to the private execution directory that contains the historical RF-DETR Windows runtime. Set `SAR_SEAL_OWNER_PYTHON` to the original Anaconda coordinator interpreter and `SAR_SEAL_PROTOCOL_COMMIT` to the 40-character Git commit containing this `replicate.py` and `PRELOCK.json`. Run from this directory. The output must be a fresh direct child of the private execution directory with a name starting with `RFDETR_COMMON5203_E24_`. The historical runtime may be checked against the original host paths in the pinned workers.

On a release mirror without Git metadata, the run owner can verify the commit, its root tree and the two intended blobs through the GitHub API. Supply the saved JSON proof and its independently recorded SHA-256 to `prepare`. The proof has `status` equal to `GITHUB_API_COMMIT_TREE_AND_BLOBS_VERIFIED`, `source` equal to `github_api`, `repository` equal to `bingzhi495-creator/SAR-SEAL-research`, the full `commit` and root `tree` SHA-1 values, and a two-entry `files` list. Each file record contains its full repository-relative `path`, Git `git_blob_sha`, and ordinary `sha256` of the exact blob payload. The intended directory is `research_round_20260925_global/code/RFDETR_COMMON5203_E24_REPLICATION_20261121`. The owner vouches for the remote commit-to-tree lookup. The local check verifies the file paths, both file hashes, and the owner-supplied proof digest. Preparation saves the proof bytes for subsequent stages.

```powershell
$privateRoot = $env:SAR_SEAL_RUNTIME_ROOT
$sciencePython = Join-Path $privateRoot 'RFDETR_WINDOWS_112_RUNTIME_PRIVATE_20261108_01/PYTHON_HOST_31013/python.exe'
$ownerPython = $env:SAR_SEAL_OWNER_PYTHON
$runPath = Join-Path $privateRoot 'RFDETR_COMMON5203_E24_ACTUAL_PRIVATE_20261121_01'
$commit = $env:SAR_SEAL_PROTOCOL_COMMIT

$proofArguments = @()
if ($env:SAR_SEAL_PUBLICATION_PROOF) {
    $proofArguments = @('--publication-proof', $env:SAR_SEAL_PUBLICATION_PROOF,
                        '--publication-proof-sha256', $env:SAR_SEAL_PUBLICATION_PROOF_SHA256)
}
& $ownerPython -I -B -X utf8 .\replicate.py prepare --work $runPath --commit $commit @proofArguments
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
& $ownerPython -I -B -X utf8 .\replicate.py qualify --work $runPath
if ($LASTEXITCODE -ne 0) { throw 'Qualification failed' }
& $ownerPython -I -B -X utf8 .\replicate.py train --work $runPath
if ($LASTEXITCODE -ne 0) { throw 'E24 training failed' }
$env:CUDA_VISIBLE_DEVICES = '-1'
& $sciencePython -I -B -X utf8 .\replicate.py accept --work $runPath
if ($LASTEXITCODE -ne 0) { throw 'Independent CPU readback failed' }
```

`prepare` reads ordinary source and commits the derived worker bytes to the fresh private output. `qualify` runs two freshly initialized 32-visit, eight-update windows, one per common-row mask view. The original independent validator must reproduce the complete paired visit sequence, nonmask supervision, actual seed and changed-mask digests. `train` independently reconstructs the short-pair receipt before starting the single E24 center arm. The owner uses the historical containment and original process waits, in a private job with its own control files. It preserves the original worker environment, including GPU 0 and two CPU library threads. Schedule it when the training GPU is available. This owner does not reserve the GPU for other launchers. The earlier center arm took 13,721 seconds; concurrent CPU work may affect runtime.

`accept` reads the new original process closures, 70,560 common and native-ledger visit entries and all E1, E8 and E24 archives. It checks that the model's training seed is 5203, the shared initial tensor digest, all AdamW/EMA clocks, the exact FIT image set and the accepted archive SHA. The complete single-arm receipt is `CENTER_E24_ACCEPTED_PRIVATE.json`. Fixed archives live under `e24_center/NATIVE_ACTUAL_PRIVATE`. Use the E24 EMA from `checkpoint_23.ckpt`. Independently retain the terminal exit for each top-level command. A failed training attempt keeps its logs and all partial outputs in its original directory.

## Downstream measurements

The separate DEV705 experiment needs this detector's own native EMA export. Compare every native rank, unmapped slot, score and RLE against the reference generated by the same detector. Retain original detection boxes, query IDs and eligible raw mask planes. Keep the existing common5202-fitted point heads and scalers to test fixed-head transfer across detector seeds. Apply all five frozen policies and the original DEV705 scorer to those predictions. If the research question changes to refitting each point head per detector, that requires a new 109-image FIT capture and separate fitting. Completion of this training command establishes only the new training endpoint.

## Generated checks

Set `SAR_SEAL_ORIGINAL_CODE` to the ordinary release's `research_round_20260925_global/code` directory and `SAR_SEAL_RUNTIME_ROOT` to the private runtime root. Use the coordinator interpreter with `-I -B -X utf8 test_replication.py -v`. The suite generates the short-pair evidence, 70,560 E24 visits and all three archive fixtures. It exercises both layers of Windows containment with generated processes, including the original isolated science interpreter. It accesses historical helper source without accessing scientific images, labels or weights.
