# HRSID FIT36 fixed-order shift scout

Use the exact previously closed [registered-shift capture and source score](../../HRSID_FIT_CENTERED_TWO_FLANK_SCOUT_ACTUAL_20261112.md#固定查询的配准位移试点实测). The worker reads the image-only request, five-column sparse responses and the archived FIT source annotation, and verifies their saved receipts. See constants in the worker for archive roles. No evaluation labels enter the run.

Run the worker under the [actual-exit observer](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py) with phase `score`, worker `score_robust.py` and a fresh output and observer directory. Use the exact prelocked file and its SHA-256. Use the qualified CPU Python environment with NumPy 1.26.4 and Shapely 2.0.2 / GEOS 3.11.2.

~~~shell
python -I -B -X utf8 score_robust.py --archive-root PRIVATE_ARCHIVE \
  --code RELEASE/code --protocol LOCK --protocol-sha256 LOCK_SHA --output RUN/score
~~~

Run `python -B test_robust.py` for synthetic order, tie and source-group resampling checks. [Published result](../../HRSID36_SHIFT_ROBUST_ORDER_ACTUAL_20261123.md)
