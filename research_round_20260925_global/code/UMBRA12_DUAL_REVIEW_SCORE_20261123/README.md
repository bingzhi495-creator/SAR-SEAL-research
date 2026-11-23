# Dual-reference Umbra scoring

Follow the [prelocked cohort and scoring protocol](../../UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md) and use the [observed closed capture and edit ledger](../../UMBRA12_FROZEN_CARRIER_ACTUAL_20261123.md). Use the accepted B5 Python runtime. Keep the exact anonymous roster, both original reviewer returns, the private admission ledger and all predictions in the coordinator workspace. Obtain the two original files separately from the assigned reviewers. The admission binds files to the assigned roles, byte identities, passed geometry checks and their joint completion status. Actual reviewer independence comes from the collection procedure.

Wrap both commands with the existing [actual-exit observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py). For admission use phase `request`, a fresh sibling `REVIEW_RUN/observed_request`, and result `REVIEW_RUN/request/RESULT_SAFE.json`. For scoring use phase `score`, a fresh score observer directory, and result `SCORE_OUT/RESULT_SAFE.json`.

~~~shell
python -I -B -X utf8 admit_reviews.py \
  --validator RELEASE/code/UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/validate_reviews.py \
  --protocol RELEASE/UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md \
  --roster ANONYMOUS_ROSTER --a A_ORIGINAL_JSON --a-sha256 SHA256_A \
  --b B_ORIGINAL_JSON --b-sha256 SHA256_B --output REVIEW_RUN/request
~~~

~~~shell
python -I -B -X utf8 score.py --code RELEASE/code \
  --protocol RELEASE/UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md \
  --review-run REVIEW_RUN --blind-run CLOSED_CAPTURE_AND_EDIT_RUN \
  --blind-receipt RELEASE/results/UMBRA12_FROZEN_CARRIER_ACTUAL_SAFE_20261123.json \
  --output SCORE_OUT
~~~

The scorer validates the original A and B freeze receipt and recomputes joint completeness and agreement before reading any blind mask streams. Original detector ranks determine one-to-one raster-IoU matches at 0.50. Continuous hull overlap uses the original polygon and exact union of native pixel cells. Unmatched hulls contribute zero effect. Equal-weight groups are the hull-bearing acquisition windows. The score reports each reviewer independently, the intersection of the prelocked coverage and interval decisions, and separately rematched coverage. An empty confirmed-hull reference yields null paired statistics.

Run `python -I -B -X utf8 test_score.py --code RELEASE/code` for the isolated integration tests. The tests use generated footprints and role files and temporary fixture receipts. Their outputs carry no measured Umbra effects.
