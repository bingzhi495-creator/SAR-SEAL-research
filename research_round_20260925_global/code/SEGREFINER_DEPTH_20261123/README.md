# Same-count SegRefiner-LR deletion location assessment

Use the [frozen location protocol](../../SEGREFINER_DEPTH_PRELOCK_20261123.md), qualified [stock replay](../../SEGREFINER_STOCK_ACTUAL_20261123.md) and [signed diagnosis](../../SEGREFINER_SIGN_ACTUAL_20261123.md). The analysis uses all original candidates and the exact labeled cohorts from the parent evaluations.

1. Run `test_depth_priority.py` with Python, NumPy and SciPy. It checks the two location rules against explicit Euclidean nearest-cell rankings on fixed and randomized masks.
2. Use the B5 CPU geometry runtime. Launch the worker under the actual-exit observer, phase `score`, with fresh score and observer directories. Follow `--help` for all parent result and annotation paths.
3. Pin the archived stock source, inference and terminal. The Instance-RSDD wrapper takes the original frozen stock run and the digest of `depth_priority.py` via `--depth-sha256`. PSeg takes the previously verified stock and signed runs and receives the same hash via `--depth-priority-sha256`. Use module digest `3b6e196f6163b055b7a68405a371de1a5f5275ea5d1b5dd5fbbe70ded7047ca4` and protocol digest `2c7a106bd8c70de20902a76b11d98ae0136b8708441cd150cf1829600d2163af`.
4. The safe result retains aggregate counts, COCO and continuous geometry, coverage and stored result hashes. Each candidate keeps its stock added cells and number of deleted cells. Periphery and proximity also keep the same accepted stock deletion positions and relocate exactly the same number. Both original rows are rescored and checked against the parent artifact.
