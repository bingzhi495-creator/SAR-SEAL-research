# Label-free D-FINE point packets

The prepared source packets keep FIT and source_gate in separate directories. Each directory has POINT_INDEX_PRIVATE.jsonl and POINTS_PRIVATE/ containing one NPZ per roster image. The status and whole index digests are registered in RESULT_SAFE.json. Use an actual naturally terminal observer receipt with verified footer before downstream file reads. Each NPZ is verified by its input index byte length and SHA-256 before use.

| Array | Shape and type | Semantics |
| --- | --- | --- |
| native_ranks | N int16 | one-based rank within that image's retained original native masks, supported candidates only |
| point_offsets | N+1 int32 | exclusive interval in each of the signed coordinate, feature and layer arrays |
| signed_coordinates | P by 2 int32 | linear C-order native indices, addition then removal, original assignment row order |
| features | P by 2 by 14 float32 | original rectangular D-FINE occupancy feature order, same signed columns |
| layers | P by 2 int8 | original inner/outside ring depths |
| static_offsets | N+1 int32 | exclusive interval in static_side_indices, difference gives realized candidate dynamic k |
| static_side_indices | K by 2 int16 | original static selection at dynamic k, each index local to that candidate's sliced point rows |

Original native masks come from the closed capture stream using native_archive_ordinal, image_id and native_ranks. The metadata index also binds source_group, original_hw, RGB hash, per-file hash and counts. Candidates with supported positions and zero exchanges are retained. Out-of-area and zero-support candidate bounds remain in the original stream.

Run prepare_points_retry.py with the original isolated CPU runtime, CUDA disabled, and --train-root, --split-dir, --code, --protocol, and --output. Observe using the existing point-edit observer with --phase prepare-edits. Use prepared paths only after it actually closes. No target or author-label paths are required. The worker installs no packages. Point files and indexes contain private per-image records and may not be published.
