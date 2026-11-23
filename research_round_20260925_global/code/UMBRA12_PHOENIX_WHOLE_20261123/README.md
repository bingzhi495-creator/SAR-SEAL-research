# Umbra author whole-mask Phoenix

Use the [secondary prelock](../../UMBRA12_PHOENIX_WHOLE_PRELOCK_20261123.md), the [independently verified twelve-window carrier and edit ledger](../../UMBRA12_FROZEN_CARRIER_ACTUAL_20261123.md), and the same exact original native intake used for both anonymous reviewer packets. Obtain the accepted author revision and XL1 instance weights using the [original whole-mask author runner](../RSDD74_PHOENIX_WHOLEMASK_20261123/README.md). All input and output masks, test windows, reviewer files and per-candidate edits stay in private project storage.

Run the request locally with the accepted B5 Python environment. Apply the [actual-exit observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py) to each phase and use sibling `observed_request`, `observed_inference`, and `observed_edits` directories. The observer result is each phase's `RESULT_SAFE.json`.

~~~shell
python -I -B -X utf8 build_request.py \
  --intake ORIGINAL_NATIVE_INTAKE --roster EXACT_ANONYMOUS_ROSTER \
  --capture-root CLOSED_UMBRA_CARRIER_RUN \
  --carrier-closure RELEASE/results/UMBRA12_FROZEN_CARRIER_ACTUAL_SAFE_20261123.json \
  --protocol PROTOCOL --protocol-sha256 PROTOCOL_SHA --output RUN/request
~~~

Transfer the closed request and its original observer receipt to the qualified DGX CPU environment. Use the released, unmodified author-source wrappers and set `CUDA_VISIBLE_DEVICES=-1`.

~~~shell
python -I -B -X utf8 infer_whole.py --source-root PHOENIX_AUTHOR \
  --whole-worker RELEASE/code/RSDD74_PHOENIX_WHOLEMASK_20261123/infer_whole.py \
  --baseline-worker RELEASE/code/INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/infer.py \
  --request RUN/request/REQUEST_PRIVATE.json --request-result RUN/request/RESULT_SAFE.json \
  --request-terminal RUN/observed_request/TERMINAL.json --request-sha256 REQUEST_SHA \
  --protocol PROTOCOL --protocol-sha256 PROTOCOL_SHA --output RUN/inference --workers 4 --threads 2
~~~

Return the closed inference result, private native-RLE ledger and observer receipt. Keep the original request and inference snapshots unchanged. Verify slots, decode, binary writes, empty outputs and additions and deletions independently in the B5 environment.

~~~shell
python -I -B -X utf8 verify_whole.py --code RELEASE/code --run RUN \
  --request-worker EXACT_REQUEST_WORKER --inference-worker EXACT_INFERENCE_WORKER \
  --protocol PROTOCOL --output RUN/edits
~~~

Run `python -I -B -X utf8 test_verify_whole.py --code RELEASE/code` to check original compact position gaps, preservation of author-returned empty masks, and recomputed changes without opening any private image or label.

After the [dual-reference review admission](../UMBRA12_DUAL_REVIEW_SCORE_20261123/README.md) has closed, observe the secondary scorer in phase `score`.

~~~shell
python -I -B -X utf8 score_phoenix.py --code RELEASE/code \
  --protocol PROTOCOL --review-run REVIEW_RUN --blind-run PRIMARY_BLIND_RUN \
  --blind-receipt RELEASE/results/UMBRA12_FROZEN_CARRIER_ACTUAL_SAFE_20261123.json \
  --whole-run RUN --output WHOLE_SCORE_OUT
~~~

Evaluation uses the original candidate match for overlap and reports independent rematching separately. The previously locked primary decision still depends on the two signed-budget arms and both reviewers.
