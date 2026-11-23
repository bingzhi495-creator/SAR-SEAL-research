# Phoenix signed quota ablation

This comparison uses the exact Phoenix endpoint probabilities and all native candidates from the closed [same-budget experiment](../../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md). The [protocol](../../RSDD74_PHOENIX_AREA_ABLATION_PRELOCK_20261123.md) fixes every candidate's total flips. The three writers compare equal adds and drops, a two-cell net area tolerance and unrestricted signed allocation. All use the same within-side ordering. No model is called.

Prerequisites are the [archived 74-chip stock replay](../INSTANCE_RSDD_B5_REPRO_20261123), the closed [PointRend cohort](../INSTANCE_RSDD_RESERVE_POINTREND_20261123) needed for historical grouping, and the closed [Phoenix request, inference, edits and score](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123). Use the same B5 CPU renderer and scorer. Obtain the labels through the [Instance-RSDD authors](https://github.com/XIDIAN212Lab/SRNet#the-instance-rsdd-dataset). Keep the actual input and edited-mask streams in authorized research storage.

Run each worker under [the existing observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py) in fresh output paths. Use phase `edits` for the writer and `score` for scoring. Supply the corresponding `RESULT_SAFE.json` path and exact worker to the observer. Mark `LOCK_SHA` as the published protocol's SHA-256.

~~~shell
python -I -B -X utf8 write_ablation.py \
  --matched-root OLD --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/writer

python -I -B -X utf8 score_ablation.py \
  --code RELEASE/code --stock-root B5 --point-root POINTREND74 \
  --phx-edits OLD/edits --phx-edits-terminal OLD/observed_edits/TERMINAL.json \
  --phx-score OLD/score --phx-score-terminal OLD/observed_score/TERMINAL.json \
  --request OLD/request --request-terminal OLD/observed_request/TERMINAL.json \
  --inference OLD/inference --inference-terminal OLD/observed_inference/TERMINAL.json \
  --endpoints OLD/inference/ENDPOINT_PROBABILITIES_PRIVATE.jsonl \
  --writer RUN/writer --writer-terminal RUN/observed_edits/TERMINAL.json \
  --protocol LOCK --protocol-sha256 LOCK_SHA --author-labels AUTHOR_TRAIN_JSON --output RUN/score
~~~

The scorer validates every signed allocation and complete balanced-mask parity before loading the original labels. Eligibility to decide a contrast counts actual differing masks among fixed original matches. The geometric estimate also retains identical fixed masks. AP evaluates the entire native candidate stream.

Run `python -B -m unittest discover -s .` from this directory. Run `python -B test_score_ablation.py --code RELEASE/code` directly to exercise the existing COCO scoring and paired geometry from a separate code copy.
