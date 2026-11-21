# FIT109 ranking ceiling and shared-capacity supervision

`diagnose_ranking.py` uses the same completed PointRend FIT109 support and independently verified OOF predictions as the previous diagnostic. It scores native-probability sorting only at the original per-candidate edit count. The two true occupancy sorters and the at-most-budget variant are restricted to fixed-match geometry.

`train_rank_fit109.py` uses the published 14-dimensional residual head, the exact original endpoint data and the five source folds. `PLAN.json` fixes the objective weight, candidate sampling, train-budget draw, 200 updates per fold and the advancement checks. The evaluator preserves the original candidates and uses the existing paired-budget plan after fitting. Both real executions produced the same full metric record on the isolated CPU runtime.

Use `--help` on the scripts for the required archived FIT experiment paths and digest arguments. All raw data, feature arrays, checkpoint artifacts, closed prediction witnesses and machine-specific admission metadata stay in the private experiment. `test_rank_objectives.py` verifies the small selection problems, masking, ties, and gradients. [Evidence and protocol](../../FIT109_POINT_RANKING_AND_DECISION_SUPERVISION_20261121.md) · [Original PointRend stage](../RFDETR_POINTREND_FIT109_20261121/README.md)
