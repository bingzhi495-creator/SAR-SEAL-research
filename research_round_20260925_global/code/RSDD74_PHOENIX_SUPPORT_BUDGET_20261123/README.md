# Phoenix support and flip-count comparison

The [predeclared 2×2 comparison](../../RSDD74_PHOENIX_SUPPORT_BUDGET_PRELOCK_20261123.md) crosses two writable supports with a fixed-count and a threshold rule. It uses the same frozen full Phoenix score field and active candidates from the [original 74-chip endpoint comparison](../../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md). The recomputed fields are used in memory. The released summaries supply no image pixels, masks or scores at specific pixels.

Obtain the exact closed endpoint request and inference, [area ablation](../../RSDD74_PHOENIX_AREA_ABLATION_ACTUAL_20261123.md) and [whole-mask transfer](../../RSDD74_PHOENIX_WHOLEMASK_ACTUAL_20261123.md). Use their unchanged private artifacts. The endpoint request and its admitted PNG inputs must share a directory. Use the [author dependency environment](../RSDD74_PHOENIX_WHOLEMASK_20261123/README.md) and unchanged original Phoenix inference worker for source, checkpoint and environment verification. Run the same four CPU workers with two threads and author decoder chunks of two on the qualified DGX environment, with `CUDA_VISIBLE_DEVICES=-1`.

Wrap each command in [the existing observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py), set the matching `inference` or `score` phase, and point it at a fresh result and observer directory. `LOCK` names the published protocol and `LOCK_SHA` is its SHA-256.

~~~shell
python -I -B -X utf8 infer_factorial.py \
  --source-root PHOENIX_AUTHOR --baseline-worker ENDPOINT_WORKER \
  --request OLD_MATCHED/REQUEST_DIR/REQUEST_PRIVATE.json \
  --matched-endpoints OLD_MATCHED/inference/ENDPOINT_PROBABILITIES_PRIVATE.jsonl \
  --area-edits OLD_AREA/writer/EDITS_PRIVATE.jsonl \
  --whole-masks OLD_WHOLE/inference/WHOLE_MASKS_PRIVATE.jsonl \
  --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/inference --workers 4 --threads 2

python -I -B -X utf8 score_factorial.py \
  --code RELEASE/code --stock-root B5 --point-root POINTREND74 \
  --matched-root OLD_MATCHED --area-root OLD_AREA --whole-root OLD_WHOLE \
  --inference RUN/inference --inference-terminal RUN/observed_inference/TERMINAL.json \
  --author-labels AUTHOR_TRAIN_JSON --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/score
~~~

The score worker checks archived output and independent recomputation on observed masks before it reads [the author polygon annotations](https://github.com/XIDIAN212Lab/SRNet#the-instance-rsdd-dataset). It distinguishes all references rematched by COCO at IoU 0.50 from the original candidate pairs that retain that match. Fixed geometry continues to include original eligible pairs with no divergence between the two compared arms.

Run `python -B test_score_factorial.py --code RELEASE/code` for the count and coverage checks against generated inputs, including the historical evaluation implementation.
