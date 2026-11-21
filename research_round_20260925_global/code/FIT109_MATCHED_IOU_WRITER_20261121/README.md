# FIT109 threshold analysis and matched-IoU writer

`analyze_pointrend_gap.py` computes observed AP at each COCO matching threshold and the algebraic intersection and predicted-area components of the fixed-match gain. `diagnose_bj.py` is an independent replacement-order and recall-grid analysis of the original B and J outputs.

`run_ratio_writer.py` uses already accepted five-fold PointRend weights and a separate frozen local edit protocol. It first reconstructs the prior balanced and joint edits from the same freshly rendered point posteriors. Then it seals the T and R edit indices before opening the FIT polygon annotations for evaluation. Each adaptation makes the same $2k$ realized changes for every eligible candidate.

Run with the admitted Python 3.10.13 CPU environment and `-I -B -X utf8`, by using `--help` to pass the closed FIT109 experiments and `IOU_WRITER_PRELOCK.json`. The original scientific closure records and private model/data files are required. The corrected runner uses a separate name for the candidate confidence and completed a full reproduction with the same per-candidate prediction bytes as the original prelabel witness. `verify_ratio_saved.py` independently verified the original recovered T/R scores with the original private terminal and score receipts. Its own immutable private outcome inputs remain with that execution.

`python -B -m unittest discover -s . -p test_ratio_writer.py` checks the optimizer against exhaustive small edit sets. [Empirical findings](../../FIT109_POINTREND_AP_GEOMETRY_AND_IOU_WRITER_20261121.md) · [Original PointRend FIT stage and prerequisites](../RFDETR_POINTREND_FIT109_20261121/README.md)
