# Replay the fixed-budget geometry diagnostic

The observed worker consumes the original closed paired replay directory via a private JSON authority. It expects `paired_replay`, `protocol`, `protocol_sha` and `prelock_commit`. The worker binds the original score worker, its verified exit, the score receipt and the original scoring authority by recorded SHA-256. It reuses the original source identity, capture and replay safeguards. Access to the original snapshots is required for an exact historical closure replay.

Run `python -I -B -X utf8 oracle.py --authority AUTHORITY_PRIVATE.json --authority-sha SHA256 --output FRESH/actual` through the existing [observed subprocess runner](../LEARNED_BOUNDED_PAIRING_20261126/rsdd/observe.py) as phase `score`. Bind the closed run's accepted CPU runtime, use one thread and set `CUDA_VISIBLE_DEVICES=-1`. The archived CPU scoring stack uses NumPy 1.26.4, pycocotools 2.0.11, Shapely 2.0.2 and GEOS 3.11.2. The exact historical authority, readback receipts, model candidate payload and author labels stay in private storage.

For the synthetic check, set `SRNET_PUBLIC_CODE` to the research release code directory and run `python -I -B -X utf8 test_oracle.py`. It compares fractional occupancy selection with brute-force subsets and reproduces the stock evaluator's matching of equal-score candidates.

[Scope and decisions](PRELOCK.md) · [Actual result](GEOMETRY_ORACLE_ACTUAL.md)

