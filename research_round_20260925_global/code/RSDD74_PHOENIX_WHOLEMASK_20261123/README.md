# Whole-mask Phoenix on Instance-RSDD 74

The [predeclared comparison](../../RSDD74_PHOENIX_WHOLEMASK_PRELOCK_20261123.md) uses the closed [Instance-RSDD 74 reproduction](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) and the archived [matched-count Phoenix comparison](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/README.md). It keeps all 7,346 stock detections and preserves their scores and order. Two policies use the author binary writer with separate candidate batches. The LVIS policy writes all 6,079 nonempty masks. The COCO policy writes 172 masks over 1% of the full image area. The remaining masks pass through. SAR AP is computed with the original Instance-RSDD COCO scoring rules for both policies.

## Inputs

Obtain the exact published RF-DETR 74-chip replay and previous signed-count Phoenix run through the two linked procedures. Obtain the author labels through [the Instance-RSDD authors](https://github.com/XIDIAN212Lab/SRNet#the-instance-rsdd-dataset). Match the SHA-256 checks in the workers. For inference, use the [pinned Phoenix source](https://github.com/naver-ai/Phoenix/tree/13ee3996ef3d0292fab144316cc8e03b69c0ecfa), the official XL1 checkpoint and acquisition receipt admitted by that prior comparison. The exact CPU environment has FastGeodis 1.0.5, Torch 2.10.0+cu130, Pillow 12.3.0, NumPy 1.26.4, and pycocotools 2.0.11. Mask batches of two are an execution choice. The author baseline worker in the comparison directory verifies the source and model before loading.

All private PNGs, masks, full outputs and author labels stay in project-owned storage. Only code, the experiment lock and aggregate results are release material.

## Run in fresh output directories

Use [the prior observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py) to wrap each command. Set the observer phase to `request`, `inference` or `score`, use the corresponding worker as `--worker`, and its output `RESULT_SAFE.json` as `--result`. Supply the unmodified protocol and its SHA-256 in every phase. The original request and inference run directories contain observed terminal receipts verified by the workers.

~~~shell
python -I -B -X utf8 build_request.py \
  --baseline OLD_PHOENIX_RUN --stock-root B5_RUN \
  --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/request
~~~

Transfer the observed request, its 74 PNGs and byte-identical runtime sources to the trusted compute host. Validate the archive and source checksums after transfer. Run this stage without target labels and with `CUDA_VISIBLE_DEVICES=-1`.

~~~shell
python -I -B -X utf8 infer_whole.py \
  --source-root PHOENIX_AUTHOR --baseline-worker PINNED_OLD_INFER \
  --request RUN/request/REQUEST_PRIVATE.json --request-sha256 REQUEST_SHA \
  --request-result RUN/request/RESULT_SAFE.json --request-terminal RUN/observed_request/TERMINAL.json \
  --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/inference --workers 4 --threads 2
~~~

Return the terminated inference ledger and its result and terminal receipts. Score against the original train JSON for the previously locked reserve. The scorer validates exact candidates, replay arms and every gate before reading annotations. Its original native matches and rematched COCO coverage answer different questions.

~~~shell
python -I -B -X utf8 score_whole.py \
  --code RELEASE/code --stock-root B5_RUN --point-root POINTREND_RUN \
  --phx-edits OLD_PHOENIX_RUN/edits --phx-edits-terminal OLD_PHOENIX_RUN/observed_edits/TERMINAL.json \
  --phx-score OLD_PHOENIX_RUN/score --phx-score-terminal OLD_PHOENIX_RUN/observed_score/TERMINAL.json \
  --request RUN/request --request-terminal RUN/observed_request/TERMINAL.json \
  --inference RUN/inference --inference-terminal RUN/observed_inference/TERMINAL.json \
  --author-labels AUTHOR_TRAIN_JSON --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/score
~~~

Run the adjacent `test_*.py` files with `python -B -m unittest discover -s .` in the code directory. The scoring test reuses the original evaluation software and can use the published relative code layout or accept a separate cited release path as defined in the test. Run `python -I -B verify_binary_parity.py --whole-root RUN --matched-root OLD_PHOENIX_RUN --output FRESH_SAFE_JSON` to reproduce the label-free agreement of both area policies and of old endpoint decisions with the whole masks.
