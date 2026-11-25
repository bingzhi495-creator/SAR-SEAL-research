# Author pre-resize probability capture

The closed label-free capture observed the SRNet author's final bilinear resize directly. The original call received the rotated probability paste after unpadding to the content shape. For all 89 candidates, every float32 native probability bit reproduced the previously qualified native archive. Candidate ranks, confidence, native oriented boxes, clipped support boxes and stock COCO masks all matched the original records. The four-worker capture ran for 122.84 seconds under the pinned legacy CPU environment.

The 74 input chips had content shape 608 × 608 and native shape 512 × 512. The author called final interpolation on the 71 rows with candidates. The three empty rows remained empty and made no resize call. The saved unpadded grids contain 32,900,096 float32 values in original candidate order. A separate observed readback reconstructed every archived native probability bit from those grids without forwarding the detector. The two closed phases accessed no reference annotations and performed no optimization or scoring.

[Protocol](PRELOCK.md) · [Capture summary](PREGRID_CAPTURE_SAFE.json) · [Independent readback](PREGRID_READBACK_SAFE.json) · [Reproduction](REPRODUCE_PREGRID.md)

