# Frozen UEIKAP hull boundary evaluation

[Prelocked protocol](../../UEIKAP_RESERVED_HULL_PRELOCK_20261122.md) · [Closed evaluation](../../UEIKAP_RESERVED_HULL_ACTUAL_20261122.md)

Run the original author capture with the accepted HRSID E24 EMA, verified author checkout, native codec, original 512-square PNGs and the ordered image-only roster. The image stages consume the already used [PSeg candidate code](../PSEG_TEST232_DFINE_FROZEN_CAPTURE_20261121), [frozen point head and replay](../PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121) and [classical control](../FIT109_CLASSICAL_ATTRACTION_20261121). The CLI exposes all pinned paths and checksums via `python -I -B -X utf8 ueikap_transfer.py --help`.

Run `capture`, `verify-capture`, `prepare-edits`, and `verify-edits` in order. Use the original GPU runtime for capture and a deterministic single-thread CPU process for edit replay. The process observer records the actual exit and binds its result footer. After independent readback, run `score.py --help` for the arguments to the separate label-based evaluation. The scorer uses NumPy 1.23.5, pycocotools 2.0.7, Shapely 2.0.2 and GEOS 3.11.2 as qualified for this run. `test_score.py` exercises synthetic cases for matching, analytic geometry, acquisition weighting and coverage.

The author ZIP is available through [Zenodo](https://zenodo.org/records/18391027). Exact replay requires the original reservation and intake manifests, the source-time capture and point-head qualification inputs, and matching author assets. The release includes the research code and public aggregate, with image and label originals obtained from the author archive.

After the frozen evaluation closes, diagnose.py counts box and mask opportunities for the top 100 and the complete native list. check_bounds.py independently recomputes coverage using dense intersections and produces the ideal equal-area swap bounds. Run each with --help for the pinned inputs and output file. [Diagnostic interpretation](../../UEIKAP_RESERVED_HULL_ACTUAL_20261122.md#candidate-coverage-and-a-geometric-upper-bound)
