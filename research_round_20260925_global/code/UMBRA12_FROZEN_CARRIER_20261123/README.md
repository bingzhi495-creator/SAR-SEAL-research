# Frozen Umbra capture and edit replay

The [prospective protocol](../../UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md) fixes the roster, frozen HRSID detector, point head and primary comparison. The [original intake](../../UMBRA_PROSPECTIVE_GEC_ANNOTATION_ACTUAL_20261122.md) supplies the exact anonymous grayscale inputs. Predictions, sparse edits, point-head weights, per-image diagnostics and both human reference files stay in the coordinator workspace.

Use the accepted B5 Python environment. With [observe.py](../UEIKAP_RESERVED_HULL_FROZEN_20261122/observe.py), run each phase with its exact worker and fresh output in the same run root. Pass the worker phase as the first argument. The observer directory is `observed_<phase>` and the observed result is `<phase>/RESULT_SAFE.json`. Keep the captured and verified phase directories side by side.

~~~shell
python -I -B -X utf8 capture_umbra.py capture \
  --code RELEASE/code --protocol PROTOCOL --protocol-sha256 PROTOCOL_SHA \
  --intake CLOSED_INTAKE --intake-terminal INTAKE_TERMINAL \
  --intake-readback INDEPENDENT_INTAKE_RESULT --reviewer-receipt PACKET_RESULT \
  --roster ANONYMOUS_ROSTER --checkpoint CENTER5202_E24 --output RUN/capture
~~~

Set the CUDA choice to `0` for capture. For `verify-capture` use the same arguments with `--output RUN/verify-capture`, omit the checkpoint argument, and set `CUDA_VISIBLE_DEVICES=-1`. The independent worker reconstructs original native rank, RLE and probability-plane invariants on CPU.

~~~shell
python -I -B -X utf8 edit_umbra.py prepare-edits \
  --code RELEASE/code --protocol PROTOCOL --intake CLOSED_INTAKE \
  --capture-worker EXACT_CAPTURE_WORKER --point-head FIT109_HEAD_5217 \
  --output RUN/prepare-edits
~~~

Keep `CUDA_VISIBLE_DEVICES=-1` for both edit phases. Repeat with `verify-edits` and output `RUN/verify-edits`. The replay checks its own written ledger against independently recovered support, pair decisions, side ranks, exact flip counts and RLE masks. Open the accepted masks for scientific scoring when both original reviewer returns and completeness status are pinned. The four workers do not read the human returns.
