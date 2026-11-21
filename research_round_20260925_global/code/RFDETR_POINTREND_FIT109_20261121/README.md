# FIT109 PointRend-style comparison

This is the source-held-out evaluation of a PointRend-style point head on the frozen RF-DETR center seed 5202 E24 carrier. The FIT cohort, imported endpoint support, original stock ranks and confidences, per-candidate edit counts and update budget are fixed in `POINTREND_FIT109_PRELOCK.json`. The detector was trained on the FIT population. The PointRend adaptation uses the RF-DETR spatial basis and the original candidate boxes and query logits.

## Reproduction

1. Run `capture_pointrend.py --help` and use the pinned prelock and historical independently accepted FIT109 capture. The wrapper records one full FP32 spatial tensor and aligned raw logits per image. It uses the original RF-DETR 1.12.0 CPU environment with Python 3.10.13, PyTorch 2.10.0+cu130 and NumPy 1.26.4. Run it with `-I -B -X utf8`, `CUDA_VISIBLE_DEVICES=-1` and local model assets.
2. Observe the original process exit and record its footer, source digest and index digest in the owner observation. Run `verify_point_capture.py --help` to independently check the closed original candidate and native RLE lineage before admission.
3. Run `run_point_fit109.py --help` using the pinned module manifest, independently verified new capture and prior accepted point-support and budget records. The exact baseline and fixed annotations are read from the closed FIT layout. Point heads use five source folds and 200 updates per fold. Geometry uses the pinned Shapely 2.0.2 and GEOS 3.11.2 runtime. Observe the original natural exit before inspecting the output.
4. Run `verify_point_fit109.py --help` on the owner-closed execution to independently reconstruct the six native mask streams and recompute COCO, fixed-match overlap and the prespecified decision. The output checks also cover the held fold schedule and parameter changes from initialization.

The original data, full maps, candidate logits, weights, per-object predictions and operational admission records remain private. The preceding public study describes the original endpoint support. [Frozen point support and budget provenance](../../RFDETR_QUERY_CHANNEL_FIT109_ACTUAL_20261121.md) · [PointRend subdivision and head](../RFDETR_POINTREND_SUBDIVISION_20261121/README.md) · [Actual results and limitations](../../RFDETR_POINTREND_FIT109_ACTUAL_20261121.md)

Synthetic tests can be run in the qualified runtime with `python -B -m unittest discover -s . -p "test_*.py" -v`. The TGRS draft reports the measured matched policies and the unconstrained RoI paste comparison.
