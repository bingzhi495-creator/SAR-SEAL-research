# PSeg-SSDD Phoenix support and flip-count transfer

The [predeclared experiment](../../PSEG232_PHOENIX_SUPPORT_BUDGET_PRELOCK_20261123.md) repeats the fixed Phoenix writer comparison on the complete historical PSeg-SSDD allocation. It reuses the [closed D-FINE capture and native replay](../../PSEG_SSDD_TEST232_DFINE_FROZEN_ACTUAL_20261121.md) and the [original dynamic-attraction budget and verification](../../PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122.md). The image-only owner directory must contain the exact admitted images and manifests from the original run. Obtain the annotation geometry separately through the sources documented in the historical protocol.

Use the B5 host environment for request generation and scoring. Run the author XL1 inference in the [qualified DGX CPU environment](../RSDD74_PHOENIX_WHOLEMASK_20261123/README.md). The actor uses the [previous exact-count engine](../RSDD74_PHOENIX_SUPPORT_BUDGET_20261123/infer_factorial.py) and the [original pinned Phoenix actor](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/infer.py) as immutable source dependencies. The original grids can be rectangular or of different sizes across images. The author actor performs its own internal resize and returns the scored grid to native size.

Wrap each worker in [the actual-exit observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py) using the corresponding phase (`request`, `inference`, or `score`), a fresh observer directory and each worker's `RESULT_SAFE.json` as the result path. `LOCK` is the protocol file and `LOCK_SHA` its published SHA-256.

~~~shell
python -I -B -X utf8 build_request.py \
  --code RELEASE/code --owner IMAGE_ONLY_OWNER --capture DFINE_CAPTURE \
  --original-replay CLOSED_REPLAY --original-readback CLOSED_READBACK_JSON \
  --classical-root CLOSED_CLASSICAL --protocol LOCK --protocol-sha256 LOCK_SHA \
  --output RUN/request

python -I -B -X utf8 infer_factorial.py \
  --source-root PHOENIX_AUTHOR --baseline-worker EXISTING_ENDPOINT_ACTOR \
  --factorial-worker EXISTING_RSDD_FACTORIAL_ACTOR \
  --request RUN/request/REQUEST_PRIVATE.json --request-result RUN/request/RESULT_SAFE.json \
  --request-terminal RUN/observed_request/TERMINAL.json --request-sha256 REQUEST_SHA \
  --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/inference --workers 4 --threads 2

python -I -B -X utf8 score_factorial.py \
  --code RELEASE/code --owner IMAGE_ONLY_OWNER --capture DFINE_CAPTURE \
  --classical-root CLOSED_CLASSICAL --request RUN/request \
  --request-terminal RUN/observed_request/TERMINAL.json --inference RUN/inference \
  --inference-terminal RUN/observed_inference/TERMINAL.json --author-labels OFFICIAL_TEST_FLOAT_JSON \
  --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/score
~~~

Inference uses `CUDA_VISIBLE_DEVICES=-1`, the official five iterations, and native binary masks. Keep the request PNGs adjacent to the request. The scorer validates both closed ledgers and their observation receipts before loading target labels. Preserve the request, inference masks, endpoint scores and labels as private research data. Public aggregates distinguish the two fixed populations and paired decoded-image group intervals.

Run the adjacent scoring test with `python -B test_score_factorial.py --code RELEASE/code` to exercise the original evaluator on generated images.
