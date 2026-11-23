# Balanced Phoenix endpoint replay

[Prelocked comparison](../../PSEG232_PHOENIX_BALANCED_RANK_PRELOCK_20261123.md) and [observed result](../../PSEG232_PHOENIX_BALANCED_RANK_ACTUAL_20261123.md). Use the original B5 host runtime, the exact closed archives and the [parent reproduction](../PSEG232_PHOENIX_SUPPORT_BUDGET_20261123/README.md). The initial request and recovered inference may live in separate directories. The parent score and terminal files must match the accepted digests in the worker. Keep frozen prediction masks, endpoint probabilities, labels and run archives private.

Run `score_balanced.py` under the parent actual-exit observer, phase `score`, with a fresh observer path and score directory.

~~~shell
python -I -B -X utf8 score_balanced.py \
  --factorial-source RELEASE/code/PSEG232_PHOENIX_SUPPORT_BUDGET_20261123/score_factorial.py \
  --code RELEASE/code --owner IMAGE_ONLY_OWNER --capture DFINE_CAPTURE \
  --classical-root CLOSED_CLASSICAL --request INITIAL_RUN/request \
  --request-terminal INITIAL_RUN/observed_request/TERMINAL.json \
  --inference RECOVERED_RUN/inference --inference-terminal RECOVERED_RUN/observed_inference/TERMINAL.json \
  --author-labels OFFICIAL_TEST_FLOAT_JSON --parent-score RECOVERED_RUN/score \
  --parent-terminal RECOVERED_RUN/observed_score/TERMINAL.json \
  --protocol RELEASE/PSEG232_PHOENIX_BALANCED_RANK_PRELOCK_20261123.md --output NEW_RUN/score
~~~

The exact-count source arm keeps the native image geometry, detector slot, score and endpoint ordinal. It reuses the prior scorer for SHA-256 and original mask/geometry checks. The observer must confirm process exit, result footer and digest.
