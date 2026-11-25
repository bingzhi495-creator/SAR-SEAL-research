# Independent crossover verification and scoring

Reproduce in the original CPU environment for the Instance-RSDD reserve. Use Python 3.10.13, NumPy 1.26.4, pycocotools 2.0.11 and Shapely 2.0.2 with GEOS 3.11.2. Disable CUDA and use one thread for OpenMP, MKL and OpenBLAS. Use the observed-exit helper in the released bounded-pairing case.

To reproduce the label-free crossover, run render_pr.py in the same original CPU runtime and point it at the released code, the closed B5 and reserve replays, the original PointRend basis and replay, the existing learned bounded experiment, and the frozen PointRend head. Supply the confirmed prelock hash and commit. Confirm the successful render receipt and its parity with the original PointRend ledger before continuing. Its --help lists the exact path arguments. The old PointRend replay uses its own fixed 31,391-swap reference. All crossover arms use the original distance-first 30,841 swaps.

Pass the published code root, the original reserve evaluation root, the earlier bounded-pairing rsdd root, the exact published prelock file, the confirmed commit and hash, the now-closed new render root, its exact worker and receipt hashes, and the original reserve author-label path. The independent verify phase uses the path for identity qualification only. It reconstructs masks from existing signed coordinates without reading labels.

Start verify with an absent private verify output directory and place the observer receipt in the sibling observed_verify directory. Both the verified aggregate and its natural terminal are mandatory inputs to score. Invoke score in a new absent output directory with the same original arguments and the exact prerequisite verify hashes. For both, launch the child with Python flags -I -B -X utf8. The verifier uses the observer's verify-edits mode and the scorer uses score. See score_pr.py --help for available arguments.

Replay can be tested on generated arrays with test_score_pr.py. Run the real scorer only after the qualified label-free render has closed and its receipts agree. Keep all imagery, masks, raw reference records, labels, weights, and logs in private storage.

