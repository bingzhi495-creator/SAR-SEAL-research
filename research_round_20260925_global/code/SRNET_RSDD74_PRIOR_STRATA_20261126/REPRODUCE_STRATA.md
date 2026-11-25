# Replay the SRNet prior strata diagnostic

Use the original historical [same-support oracle](../SRNET_RSDD74_THRESHOLD_ORACLE_20261126/GEOMETRY_ORACLE_ACTUAL.md) and its exact private closure. Supply a private JSON authority with `protocol`, `protocol_sha`, `prelock_commit` and `geometry_oracle`. The `geometry_oracle` path points to the original oracle directory, which contains its worker, private authority, witnessed terminal and matching safe result. The worker verifies these bindings, reconstructs the signed point set and archived I choices, then opens only the admitted author TRAIN reference rows.

Run `python -I -B -X utf8 strata.py --authority AUTHORITY_PRIVATE.json --authority-sha SHA256 --output FRESH/actual` under the [observed subprocess runner](../LEARNED_BOUNDED_PAIRING_20261126/rsdd/observe.py) in phase `score`. Use the archived qualified CPU interpreter and original numerical packages, `CUDA_VISIBLE_DEVICES=-1`, and one thread. This is an exact historical replay, so relocating or regenerating its private authorities changes the digests.

Set `SRNET_PUBLIC_CODE`, `SRNET_SHAPELY_SITE` and `SRNET_ORACLE_SOURCE` to use `test_strata.py`. The tests use synthetic endpoints and polygons. They check the inclusive FP32 thresholds, real zero and one values, exact per-bin quota against exhaustive enumeration, and signed intersection decomposition against direct raster geometry.

The output contains only aggregate metrics and original lineage hashes. [Registered scope](PRELOCK.md)

