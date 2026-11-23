# Bounded Phoenix matching

Use the [prelock](../../PHOENIX_BOUNDED_MATCH_PRELOCK_20261123.md) and the exact [paired Phoenix experiment inputs](../PHOENIX_PAIR_OPERATOR_20261123/README.md). Use the frozen CPU environment and the same original candidate streams, archive order, point probabilities and labels. The previous SAFE scores are reference checks for the historical policies.

1. Run `python -B test_matching.py` in this folder. The synthetic test exhaustively compares all valid endpoint sets and matchings on randomized small instances, including exact binary32 ties and subnormal values.
2. Use the original separate observer for phase `score` with new paths for the observer and scorer. Run the matching scorer for the relevant dataset with the corresponding previous pairing scorer's arguments, replacing its protocol by the new prelock, and add `--matching-sha256 99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c`. The protocol digest is `176ff6bd2bfb99f7474a4ffb89e2c634d9e955ee7af7d3b8a921d2485e7019f9`.
3. For PSeg use `--pair-operator-source` with the previous PSeg pairing scorer and `--prior-score` with the [previous PSeg SAFE JSON](../../results/PSEG232_PHOENIX_PAIR_OPERATOR_ACTUAL_SAFE_20261123.json). Instance-RSDD uses the [previous RSDD SAFE JSON](../../results/RSDD74_PHOENIX_PAIR_OPERATOR_ACTUAL_SAFE_20261123.json) in the same release tree by default.
4. Verify the observer's actual exit, worker hash, result hash and final footer. The three masks and original quotas are checked before labels open. Only aggregate SAFE output belongs in the release.

To reproduce the supplementary graph, pass the two released SAFE score summaries and an output prefix to plot_contrasts.py. It reads only aggregate scores and writes editable SVG, PDF and a preview.
