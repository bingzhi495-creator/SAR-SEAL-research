# Run the exact fixed-count mask writer

This CPU entry starts at the learned occupancy scores. It calls the published NumPy selectors used in the fixed-count experiments. It produces COCO result lists for the original distance-first pairing and the independently ranked endpoints at the same realized addition and removal counts. Scores and all other candidate fields carry through unchanged.

Tested with Python 3.10, NumPy 1.26.4, SciPy 1.15.3, and pycocotools 2.0.11. Run from this directory using your local input and a fresh output directory whose parent exists.

```sh
python -m pip install numpy==1.26.4 scipy==1.15.3 pycocotools==2.0.11
python test_replay_fixed_count.py
python replay_fixed_count.py --input candidates.jsonl --output replay
```

Use one JSON object per line. Supply every native candidate that belongs in the COCO evaluation, including candidates receiving no edits.

- `candidate` is the original record with integer `image_id` and `category_id`, numeric `score`, and `segmentation` containing compressed COCO RLE with `size` in height, width order. Extra candidate fields are retained.
- `pairs` contains at most 32 distinct endpoint pairs in the original pairing row order. Each pair is a zero-based addition index and removal index in the native row-major raster. They must refer to original background and original foreground, respectively.
- `occupancy` contains the corresponding learned add and removal-site foreground probabilities in that same order. The writer casts these to FP32 before forming pair gains, preserving the original tie rules.
- Optional `cap` can lower the default `min(8, foreground_pixels // 10, number_of_pairs)`. For an unsupported candidate pass empty lists. A candidate with zero realized exchanges keeps its original RLE.

The output contains `fixed.json`, `independent.json`, and `summary.json` with aggregate counts. The synthetic regression exercises an FP32 gain tie where using Python double subtraction would select a different pair. A second regression verifies that a malformed later candidate cannot publish earlier, partial results.

## Closed reserve replay

The verified 74-image timing cache supplies 7,346 native candidate masks. All 4,092 positive-budget writes were recomputed from the frozen FP32 scores, including 30,841 exchanges for each policy. The other 3,254 slots were passed through. Both COCO RLE streams equal the historical masks at every slot. The check uses placeholder scores only to verify record alignment and does no evaluation or network training. [Safe replay proof](RESERVE_REPLAY_SAFE.json) · [Original cache instructions](../LEARNED_BOUNDED_PAIRING_20261126/rsdd_timing/README.md)

A holder of the locally regenerated cache can rerun the exact parity test. The `collect` directory must have the observed `observed_collect/TERMINAL.json` sibling, as produced by the cache instructions.

```sh
python verify_reserve_replay.py --collect runs/timing/collect --output runs/writer-check
```

To reproduce the full metrics, use the [original dataset and checkpoint pipeline](../SAR_B5_MAIN_TABLES_20261123/README.md). The original detector checkpoint and the three original head bundles are separately supplied artifacts. This writer can use reader-supplied endpoint scores without those historical artifacts.

