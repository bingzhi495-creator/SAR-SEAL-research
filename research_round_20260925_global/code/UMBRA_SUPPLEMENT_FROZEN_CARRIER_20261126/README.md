# Frozen supplemental Umbra carrier

This runner is an admission adapter for the original U12 detector and edit operators. Keep the new worker, protocol and admission lock in a separate execution root. Use the published research code directory containing the original U12, Instance-RSDD and UEIKAP operators. Supply the exact source project checkpoint and point bundle separately.

After the coordinator admits the closed metadata, native intake/readback and two blind packets, pin the following JSON in the same root as `supplemental.py` and `PRELOCK.md`. Each placeholder is a full lower-case SHA-256 identifier.

```json
{
  "schema": "umbra-supplement-frozen-carrier-v1",
  "worker_sha256": "WORKER",
  "protocol_sha256": "PRELOCK",
  "manifest_sha256": "METADATA_MANIFEST",
  "metadata_result_sha256": "METADATA_RESULT",
  "metadata_terminal_sha256": "METADATA_TERMINAL",
  "intake_result": "NATIVE_RESULT",
  "intake_terminal": "NATIVE_TERMINAL",
  "readback_result": "NATIVE_READBACK_RESULT",
  "readback_terminal": "NATIVE_READBACK_TERMINAL",
  "packet_result": "BLIND_PACKET_RESULT",
  "packet_terminal": "BLIND_PACKET_TERMINAL",
  "anonymous_roster": "ANONYMOUS_ROSTER"
}
```

Run in sequence with the accepted Instance-RSDD B5 runtime and original UEIKAP observer. The native intake root has `intake`, `readback`, `observed_intake` and `observed_readback` subdirectories. The packet builder supplies the standalone anonymous `ROSTER.json` with the exact roster bytes shared by both reviewers.

```text
PYTHON -I -B -X utf8 CODE/UEIKAP_RESERVED_HULL_FROZEN_20261122/observe.py
  --phase PHASE --worker RUN/supplemental.py
  --result RUN/PHASE/RESULT_SAFE.json --observer RUN/observed_PHASE
  -- PYTHON -I -B -X utf8 RUN/supplemental.py PHASE --code CODE
  --intake-root NATIVE_ROOT --packet-result PACKET_RESULT
  --packet-terminal PACKET_TERMINAL --roster ROSTER
  --output RUN/PHASE --admission-sha256 ADMISSION_DIGEST PHASE_ARGS
```

| Phase | CUDA visibility | Phase arguments | Required previous closure |
| --- | --- | --- | --- |
| `capture` | `0` | `--checkpoint CENTER5202_E24` | Native intake/readback and blind packets |
| `verify-capture` | `-1` | None | Capture |
| `prepare-edits` | `-1` | `--point-head FIT109_SEED5217` | Capture and independent capture readback |
| `verify-edits` | `-1` | `--point-head FIT109_SEED5217` | Prepared edits and capture readback |

The four outputs preserve source query ranks and anonymous image suffixes, including any omitted slots. Verify each stage's observed terminal before relying on it. The scientific scoring gate is separate.

The implementation was checked with three synthetic tests covering preservation of anonymous ID gaps, acceptance and rejection of observed partial intake, and grayscale pixel replication with exact image-digest validation. No supplemental capture is implied by passing those tests.

After all four observers have actually reached successful termination, bind the four terminal/result pairs and the three private streams into the aggregate receipt.

```text
PYTHON -I -B -X utf8 RUN/close_receipts.py --admission-sha256 ADMISSION_DIGEST
```

The coordinator-only paths are `capture/PREDICTIONS_PRIVATE.jsonl`, `capture/CANDIDATE_AUDIT_PRIVATE.jsonl`, and `prepare-edits/EDITS_PRIVATE.jsonl`. The final `RESULT_SAFE.json` binds their SHA-256 values to the exact roster and four observed terminals. The released safe aggregate contains no images, weights, candidate records, masks or labels.
