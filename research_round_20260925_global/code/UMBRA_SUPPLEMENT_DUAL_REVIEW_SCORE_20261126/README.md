# Supplemental Umbra independent A/B review evaluation

Use the [cohort lock](../UMBRA_SUPPLEMENT_METADATA_20261126/LOCK.md), [prediction-blind packet build](../../UMBRA_SUPPLEMENT_REVIEW_PACKETS_ACTUAL_20261126.md), and [independently verified capture and masks](../../UMBRA_SUPPLEMENT_FROZEN_CARRIER_ACTUAL_20261126.md). Keep the native imagery, anonymous roster, original reviews and all mask streams with the coordinator. Obtain the original A and B returns separately. Actual independence comes from the collection procedure.

Use the verified B5 Python runtime and [actual-exit observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py). Wrap the first command with phase `request` and sibling observer directory `REVIEW_RUN/observed_request`. Wrap the second command with phase `score` and a fresh score observer directory.

~~~shell
python -I -B -X utf8 admit_reviews.py \
  --protocol CODE/UMBRA_SUPPLEMENT_METADATA_20261126/LOCK.md \
  --validator CODE/UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/validate_reviews.py \
  --packet-run CLOSED_PACKET_RUN --packet-terminal-sha256 PACKET_TERMINAL_SHA256 \
  --packet-worker-sha256 PACKET_WORKER_SHA256 \
  --a A_ORIGINAL_JSON --a-sha256 SHA256_A --b B_ORIGINAL_JSON --b-sha256 SHA256_B \
  --output REVIEW_RUN/request

python -I -B -X utf8 score_frozen.py --code CODE \
  --protocol CODE/UMBRA_SUPPLEMENT_METADATA_20261126/LOCK.md \
  --review-run REVIEW_RUN --blind-run CLOSED_FROZEN_MASK_RUN \
  --blind-receipt-sha256 e58875be4a9af7707c5fc6ee64dc1106eca70040be95be1a8df9a25cc938e04d \
  --output FRESH_SCORE_DIR
~~~

The two label returns and observed admission are checked before scoring can access frozen masks. Each original collect is a separate group. Both references use the original native match at raster IoU 0.50, continuous polygon-cell IoU and equal acquisition weighting. Unmatched, zero-quota and unchanged hulls contribute zero paired gain. Each separate review must have at least 30 native-matched hulls receiving a positive fixed-pair budget in at least 12 acquisitions and a positive 95% paired lower interval bound to confirm the supplemental result. Original U12 records remain separate. The output reports both directions and coverage even when the result is inconclusive.

The packet lineage was verified against the actual closed packet receipt. Three procedural tests used only synthetic labels and mask records. Actual annotation admission and scoring await reviewer returns.
